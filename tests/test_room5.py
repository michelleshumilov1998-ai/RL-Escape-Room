"""Tests for Room 5 — the Adaptive Storage Facility.

Covers the checks the Room 5 brief lists: layout generation and validation, the
seed split, static and dynamic collisions including crossings, Wait, conveyors,
the mission stages, radar correctness and bounds, partial observability, the
feature representation, the semi-gradient update, reproducibility, replay,
saving/loading, and the graphs.
"""

import json
import math
import os

import pytest

from plots import room5_plots
from renderers import base_renderer, fallback_renderer, iso_canvas, room5_scene
from rooms.room5 import experiments, layout as L, q_agent, simulate, storage
from rooms.room5 import environment as E
from rooms.room5.environment import Room5Env
from rooms.room5.features import FEATURE_SETS, FeatureExtractor


@pytest.fixture(scope="module")
def layout():
    return L.generate(L.seed_for("training", 0), "training")


@pytest.fixture
def env(layout):
    environment = Room5Env(layout=layout)
    environment.reset(layout)
    return environment


@pytest.fixture(scope="module")
def trained():
    return q_agent.train(episodes=900, seed=0, training_layouts=8,
                         validation_layouts=3, validation_every=300)


# ---------------- 1-5: layouts ----------------

def test_every_layout_has_one_start_terminal_and_exit():
    """1, 2, 3."""
    for index in range(20):
        candidate = L.generate(L.seed_for("training", index), "training")
        flat = "".join("".join(row) for row in candidate.grid)
        assert flat.count(L.START) == 1
        assert flat.count(L.TERMINAL) == 1
        assert flat.count(L.EXIT) == 1
        assert candidate.tile_at(*candidate.start) == L.START
        assert candidate.tile_at(*candidate.terminal) == L.TERMINAL
        assert candidate.tile_at(*candidate.exit_cell) == L.EXIT


def test_both_legs_of_every_mission_are_walkable():
    """4, 5: start -> terminal and terminal -> exit."""
    for split in L.SPLIT_NAMES:
        for index in range(10):
            candidate = L.generate(L.seed_for(split, index), split)
            assert L.is_valid(candidate)
            assert candidate.shortest_mission_length() is not None


def test_every_difficulty_generates_valid_layouts():
    for difficulty in L.DIFFICULTIES:
        for index in range(8):
            candidate = L.generate(L.seed_for("training", index), "training",
                                   difficulty)
            assert L.is_valid(candidate)
            assert candidate.difficulty == difficulty


def test_only_valid_characters_appear():
    candidate = L.generate(1000, "training")
    for row in candidate.grid:
        for character in row:
            assert character in L.VALID_TILES


def test_layout_generation_is_reproducible():
    """29."""
    first = L.generate(4242, "training")
    second = L.generate(4242, "training")
    assert first.grid == second.grid
    assert first.terminal == second.terminal
    assert [robot.route for robot in first.robot_templates] == \
        [robot.route for robot in second.robot_templates]


def test_training_and_test_seeds_never_overlap():
    """30: the whole generalisation claim rests on this."""
    assert not L.splits_overlap(60, 20, 30)
    training = set(L.seeds("training", 60))
    test = set(L.seeds("test", 30))
    validation = set(L.seeds("validation", 20))
    assert not training & test
    assert not training & validation
    assert not validation & test


def test_obstacle_density_rises_with_difficulty():
    easy = L.generate(1000, "training", "Easy").obstacle_density()
    hard = L.generate(1000, "training", "Hard").obstacle_density()
    assert hard > easy


# ---------------- 6-13: movement, robots, collisions ----------------

def test_static_obstacles_block_movement(env, layout):
    """6."""
    blocked = None
    for row in range(layout.size):
        for col in range(layout.size):
            if layout.blocks(row, col) and layout.in_bounds(row - 1, col) \
                    and not layout.blocks(row - 1, col):
                blocked = (row, col)
                break
        if blocked:
            break
    assert blocked is not None
    env.position = (blocked[0] - 1, blocked[1])
    before = env.position
    _, reward, _, info = env.step(E.MOVE_DOWN)
    assert env.position == before
    assert info["collision"]
    assert reward <= env.collision_penalty


def test_robots_stay_on_their_routes_and_inside_the_map(env, layout):
    """7, 8."""
    for _ in range(60):
        _, _, done, info = env.step(E.WAIT)
        for robot, cell in zip(env.robots, info["robot_cells"]):
            assert tuple(cell) in set(robot.route)
            assert layout.in_bounds(*cell)
        if done:
            break


def test_loop_and_bounce_patrols_behave_differently():
    """9, 10."""
    route = [(1, 1), (1, 2), (1, 3)]
    loop = L.Robot(route, "loop", index=0)
    assert [loop.advance() for _ in range(4)] == [(1, 2), (1, 3), (1, 1), (1, 2)]
    bounce = L.Robot(route, "bounce", index=0)
    assert [bounce.advance() for _ in range(4)] == [(1, 2), (1, 3), (1, 2), (1, 1)]


def test_a_same_cell_robot_collision_ends_the_episode(env):
    """11."""
    assert env.robots, "this layout has no robots to test with"
    robot = env.robots[0]
    # Stand where the robot is about to step.
    ahead = robot.route[(robot.index + 1) % len(robot.route)]
    env.position = ahead
    _, reward, done, info = env.step(E.WAIT)
    assert done
    assert info["caught"]
    assert reward <= env.robot_penalty


def test_a_crossing_collision_is_detected():
    """12: R-5 and a robot swapping places must count as a collision."""
    grid = [[L.FLOOR for _ in range(6)] for _ in range(6)]
    grid[4][1] = L.START
    grid[2][2] = L.TERMINAL
    grid[1][4] = L.EXIT
    robot = L.Robot([(3, 1), (3, 2)], "loop", index=1)     # about to step to (3,1)
    handmade = L.Layout(1, "training", grid, (4, 1), (2, 2), (1, 4), [robot],
                        "Easy", 4, 60)
    env = Room5Env(layout=handmade)
    env.reset(handmade)
    env.position = (3, 1)          # where the robot is heading
    env.robots[0].index = 1        # robot at (3,2), moving to (3,1)
    # R-5 steps to (3,2) while the robot steps to (3,1): a straight swap.
    _, reward, done, info = env.step(E.MOVE_RIGHT)
    assert done and info["caught"], "a crossing swap slipped through"
    assert reward <= env.robot_penalty


def test_waiting_holds_position(env):
    """13."""
    before = env.position
    _, reward, done, info = env.step(E.WAIT)
    assert env.position == before
    assert not info["collision"]


# ---------------- 14-18: conveyors and the mission stages ----------------

def test_a_conveyor_pushes_the_robot_one_extra_cell():
    """14, 15."""
    grid = [[L.FLOOR for _ in range(6)] for _ in range(6)]
    grid[4][1] = L.START
    grid[2][2] = L.TERMINAL
    grid[1][4] = L.EXIT
    grid[4][2] = L.CONVEYOR_RIGHT
    handmade = L.Layout(1, "training", grid, (4, 1), (2, 2), (1, 4), [],
                        "Easy", 4, 60)
    env = Room5Env(layout=handmade)
    env.reset(handmade)
    _, reward, _, info = env.step(E.MOVE_RIGHT)
    assert info["conveyor"]
    assert env.position == (4, 3), "the conveyor should have carried it on"
    assert reward <= env.step_cost + env.conveyor_penalty


def test_a_conveyor_cannot_push_out_of_bounds():
    """15: the boundary case."""
    grid = [[L.FLOOR for _ in range(6)] for _ in range(6)]
    grid[4][1] = L.START
    grid[2][2] = L.TERMINAL
    grid[1][4] = L.EXIT
    grid[4][5] = L.CONVEYOR_RIGHT          # pushes towards the wall
    handmade = L.Layout(1, "training", grid, (4, 4), (2, 2), (1, 4), [],
                        "Easy", 4, 60)
    env = Room5Env(layout=handmade)
    env.reset(handmade)
    env.position = (4, 4)
    env.step(E.MOVE_RIGHT)
    assert handmade.in_bounds(*env.position)
    assert env.position == (4, 5), "it must stay put rather than leave the map"


def test_the_terminal_activates_once_and_unlocks_the_exit(env, layout):
    """16, 17, 18."""
    env.position = layout.terminal
    assert env.stage == 0
    # Step onto it from a neighbour so the activation path runs.
    for action, (delta_row, delta_col) in E.ACTION_DELTAS.items():
        neighbour = (layout.terminal[0] - delta_row, layout.terminal[1] - delta_col)
        if action != E.WAIT and not layout.blocks(*neighbour):
            env.position = neighbour
            _, reward, _, info = env.step(action)
            break
    assert env.stage == 1
    assert info["terminal_activated"]
    assert reward >= env.terminal_reward - abs(env.step_cost) - 5

    # A second visit pays nothing extra.
    env.position = layout.terminal
    _, second_reward, _, second_info = env.step(E.WAIT)
    assert not second_info["terminal_activated"]


def test_the_exit_is_locked_before_the_terminal(env, layout):
    """17."""
    neighbour = None
    for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        candidate = (layout.exit_cell[0] + delta_row,
                     layout.exit_cell[1] + delta_col)
        if layout.in_bounds(*candidate) and not layout.blocks(*candidate):
            neighbour = candidate
            break
    assert neighbour is not None

    env.position = neighbour
    env.stage = 0
    action = next(action for action, (dr, dc) in E.ACTION_DELTAS.items()
                  if action != E.WAIT
                  and (neighbour[0] + dr, neighbour[1] + dc) == layout.exit_cell)
    _, reward, done, info = env.step(action)
    assert not done, "the exit must not finish the mission at stage 0"
    assert info["early_exit"]
    assert env.position == neighbour


def test_the_exit_completes_the_mission_after_activation(env, layout):
    """18."""
    neighbour = None
    for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        candidate = (layout.exit_cell[0] + delta_row,
                     layout.exit_cell[1] + delta_col)
        if layout.in_bounds(*candidate) and not layout.blocks(*candidate):
            neighbour = candidate
            break
    env.position = neighbour
    env.stage = 1
    action = next(action for action, (dr, dc) in E.ACTION_DELTAS.items()
                  if action != E.WAIT
                  and (neighbour[0] + dr, neighbour[1] + dc) == layout.exit_cell)
    _, reward, done, info = env.step(action)
    assert done and info["reached_exit"]
    assert reward >= env.exit_reward - 10


def test_the_charger_pays_only_once(env, layout):
    chargers = [(row, col) for row in range(layout.size)
                for col in range(layout.size)
                if layout.tile_at(row, col) == L.CHARGER]
    if not chargers:
        pytest.skip("this layout has no charging station")
    env.position = chargers[0]
    env.charger_used = False
    _, first, _, first_info = env.step(E.WAIT)
    assert first_info["charged"]
    _, second, _, second_info = env.step(E.WAIT)
    assert not second_info["charged"]
    assert second < first


# ---------------- 19-21: radar and partial observability ----------------

def test_radar_distances_are_correct_and_normalised(env, layout):
    """19, 20."""
    observation = env.observation()
    assert len(observation["static_radar"]) == 8
    assert len(observation["dynamic_radar"]) == 4
    for value in observation["static_radar"] + observation["dynamic_radar"]:
        assert 0.0 < value <= 1.0
    assert 0.0 <= observation["nearest_robot"] <= 1.0


def test_a_radar_ray_reports_an_adjacent_obstacle_as_closest():
    """19: hand-checked against a layout built for the purpose."""
    grid = [[L.FLOOR for _ in range(7)] for _ in range(7)]
    grid[5][1] = L.START
    grid[2][3] = L.TERMINAL
    grid[1][5] = L.EXIT
    grid[2][2] = L.SHELF                # directly north of (3,2)
    handmade = L.Layout(1, "training", grid, (5, 1), (2, 3), (1, 5), [],
                        "Easy", 4, 60)
    env = Room5Env(layout=handmade)
    env.reset(handmade)
    env.position = (3, 2)
    rays = env.static_radar()
    # North is index 0; one cell away out of a range of 4.
    assert rays[0] == pytest.approx(1 / 4)


def test_the_observation_never_contains_the_map(env):
    """21: partial observability has to be real."""
    observation = env.observation()
    assert set(observation.keys()) == {"static_radar", "dynamic_radar", "target",
                                       "stage", "nearest_robot",
                                       "previous_action"}
    flat = []
    for value in observation.values():
        flat.extend(value if isinstance(value, list) else [value])
    assert len(flat) == 8 + 4 + 3 + 1 + 1 + 5
    # Nothing that could identify the layout or an absolute position.
    assert not hasattr(observation, "grid")
    assert "seed" not in observation


def test_the_objective_features_point_at_the_right_target(env, layout):
    """22."""
    observation = env.observation()
    row, col = env.position
    expected_row = (layout.terminal[0] - row) / layout.size
    assert observation["target"][0] == pytest.approx(expected_row)
    env.stage = 1
    after = env.observation()
    expected_exit = (layout.exit_cell[0] - row) / layout.size
    assert after["target"][0] == pytest.approx(expected_exit)


def test_visible_cells_are_a_display_concern_only(env):
    """The fog of war exists, but the agent's observation does not include it."""
    assert env.visible_cells()
    assert "visible_cells" not in env.observation()


# ---------------- 23-28: features and the update ----------------

def test_the_feature_vector_shape_is_stable(env):
    """23."""
    extractor = FeatureExtractor()
    observation = env.observation()
    counts = set()
    for _ in range(12):
        env.step(E.WAIT)
        counts.add(len(extractor.active_features(env.observation(), 0)))
    assert len(counts) == 1, "the number of active features drifted: %s" % counts


def test_action_features_are_separated(env):
    """24."""
    extractor = FeatureExtractor()
    observation = env.observation()
    blocks = [set(extractor.active_features(observation, action))
              for action in E.ACTIONS]
    for first in range(len(blocks)):
        for second in range(first + 1, len(blocks)):
            assert not blocks[first] & blocks[second]


def test_every_feature_index_is_in_range(env):
    extractor = FeatureExtractor()
    for _ in range(30):
        env.step(E.WAIT)
        for action in E.ACTIONS:
            for index in extractor.active_features(env.observation(), action):
                assert 0 <= index < extractor.total_features


def test_every_feature_set_builds_and_shrinks_sensibly(env):
    observation = env.observation()
    sizes = []
    for name, groups in FEATURE_SETS.items():
        extractor = FeatureExtractor(groups=groups)
        active = extractor.active_features(observation, 0)
        assert active
        sizes.append(extractor.total_features)
    assert sizes == sorted(sizes), "the sets should grow from A to D"


def test_q_value_matches_a_manual_sum(env):
    """25."""
    agent = q_agent.LinearAgent()
    observation = env.observation()
    indices = agent.extractor.active_features(observation, E.MOVE_UP)
    for position, index in enumerate(indices):
        agent.weights[index] = float(position + 1)
    assert agent.q_value(observation, E.MOVE_UP) == pytest.approx(
        sum(range(1, len(indices) + 1)))


def test_the_semi_gradient_update_matches_a_numerical_example(env):
    """26."""
    agent = q_agent.LinearAgent()
    observation = env.observation()
    indices = agent.extractor.active_features(observation, E.MOVE_RIGHT)
    assert agent.q_value(observation, E.MOVE_RIGHT) == 0.0

    alpha = 0.5
    target = 8.0
    delta = agent.update(observation, E.MOVE_RIGHT, target, alpha)
    assert delta == pytest.approx(8.0)
    expected_each = alpha * 8.0 / len(indices)
    for index in indices:
        assert agent.weights[index] == pytest.approx(expected_each)
    assert agent.q_value(observation, E.MOVE_RIGHT) == pytest.approx(alpha * 8.0)


def test_only_the_active_features_change(env):
    agent = q_agent.LinearAgent()
    observation = env.observation()
    indices = set(agent.extractor.active_features(observation, E.WAIT))
    agent.update(observation, E.WAIT, 3.0, 0.5)
    changed = {index for index, weight in enumerate(agent.weights)
               if weight != 0.0}
    assert changed == indices


def test_a_terminal_update_omits_the_bootstrap(env):
    """27."""
    agent = q_agent.LinearAgent()
    observation = env.observation()
    delta = agent.update(observation, E.WAIT, 5.0, 1.0)
    assert delta == pytest.approx(5.0)


def test_epsilon_greedy_behaves(env):
    """28."""
    import random
    agent = q_agent.LinearAgent()
    observation = env.observation()
    for index in agent.extractor.active_features(observation, E.MOVE_DOWN):
        agent.weights[index] = 5.0
    rng = random.Random(0)
    assert {agent.epsilon_greedy(observation, 0.0, rng) for _ in range(15)} == \
        {E.MOVE_DOWN}
    assert len({agent.epsilon_greedy(observation, 1.0, rng)
                for _ in range(200)}) > 1


# ---------------- training, replay, storage ----------------

def test_training_is_reproducible():
    def run():
        return q_agent.train(episodes=40, seed=3, training_layouts=3,
                             validation_every=0)["history"]["reward"]
    assert run() == run()


def test_no_weight_becomes_nan_or_infinite():
    """33."""
    result = q_agent.train(episodes=150, seed=0, training_layouts=4,
                           validation_every=0)
    assert result["agent"].is_finite()
    assert math.isfinite(result["weight_norm"])


def test_a_short_smoke_run_completes():
    """37."""
    result = q_agent.train(episodes=12, seed=0, training_layouts=2,
                           validation_every=0)
    assert len(result["history"]["reward"]) == 12


def test_training_records_the_validation_curve_on_held_out_layouts():
    result = q_agent.train(episodes=200, seed=0, training_layouts=3,
                           validation_layouts=2, validation_every=100)
    assert result["history"]["validation_episode"]
    assert len(result["history"]["validation_success"]) == \
        len(result["history"]["validation_episode"])
    # The validation seeds must not be training seeds.
    assert not set(result["training_seeds"]) & set(result["validation_seeds"])


def test_shaping_cannot_be_farmed_without_finishing(env):
    """A round trip must lose money, or the agent could circle for ever."""
    total = 0.0
    for action in ([E.MOVE_UP] * 3 + [E.MOVE_DOWN] * 3):
        _, reward, done, _ = env.step(action)
        total += reward
        if done:
            break
    assert total < 0


def test_replay_reproduces_the_stored_episode(trained, layout):
    """31."""
    first = simulate.run_episode(trained["agent"], layout)
    second = simulate.run_episode(trained["agent"], layout)
    assert first["frames"] == second["frames"]


def test_replay_frames_carry_everything_the_brief_lists(trained, layout):
    run = simulate.run_episode(trained["agent"], layout)
    for frame in run["frames"]:
        for key in ("step", "agent_position", "action", "reward",
                    "cumulative_reward", "stage", "current_target",
                    "radar_observation", "visible_cells",
                    "moving_robot_positions", "moving_robot_directions",
                    "conveyor_event", "collision", "terminal_activated",
                    "exit_unlocked", "done", "success", "event"):
            assert key in frame, "frame %d is missing %r" % (frame["step"], key)


def test_the_stage_never_goes_backwards(trained, layout):
    run = simulate.run_episode(trained["agent"], layout)
    stages = [frame["stage"] for frame in run["frames"]]
    assert stages == sorted(stages)


def test_saving_and_loading_restore_identical_predictions(tmp_path, monkeypatch,
                                                         env):
    """32."""
    monkeypatch.setattr(storage, "SAVE_DIRECTORY", str(tmp_path))
    result = q_agent.train(episodes=60, seed=0, training_layouts=3,
                           validation_every=0)
    path = storage.save_model("unit_test", result)
    assert os.path.isfile(path)

    fresh = q_agent.LinearAgent(FeatureExtractor(
        groups=FEATURE_SETS[result["feature_set"]]))
    storage.load_into("unit_test", fresh)
    assert fresh.weights == result["agent"].weights

    observation = env.observation()
    for action in E.ACTIONS:
        assert fresh.q_value(observation, action) == pytest.approx(
            result["agent"].q_value(observation, action))


def test_loading_refuses_an_incompatible_feature_representation(tmp_path,
                                                               monkeypatch):
    monkeypatch.setattr(storage, "SAVE_DIRECTORY", str(tmp_path))
    result = q_agent.train(episodes=20, seed=0, training_layouts=2,
                           validation_every=0)
    path = storage.save_model("checks", result)

    mismatched = q_agent.LinearAgent(
        FeatureExtractor(groups=FEATURE_SETS["A. Target only"]))
    with pytest.raises(storage.ModelFileError):
        storage.load_into("checks", mismatched)
    assert all(weight == 0.0 for weight in mismatched.weights), \
        "a refused load must leave the agent untouched"

    with pytest.raises(storage.ModelFileError, match="No saved model"):
        storage.load_into("absent", q_agent.LinearAgent())

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for field, value, message in (("format_version", 99, "file format"),
                                  ("room", 1, "not a Room 5 model"),
                                  ("environment_version", "x", "environment")):
        broken = dict(payload)
        broken[field] = value
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(broken, handle)
        with pytest.raises(storage.ModelFileError, match=message):
            storage.load_into("checks", q_agent.LinearAgent())


# ---------------- scene, renderer, graphs ----------------

def test_the_scene_is_valid_and_hides_the_map_by_default(trained, layout):
    run = simulate.run_episode(trained["agent"], layout)
    scene = room5_scene.build_scene(layout, frames=run["frames"])
    assert base_renderer.validate_scene(scene) == []
    assert scene["meta"]["roomNumber"] == 5
    assert scene["meta"]["fogOfWar"] is True
    restored = json.loads(json.dumps(scene))
    assert restored["episodeId"] == scene["episodeId"]


def test_the_engine_handles_every_room_five_tile_type(layout):
    scene = room5_scene.build_scene(layout)
    kinds = {kind for row in scene["grid"]["tiles"] for kind in row}
    engine = iso_canvas.build_html(scene)
    assert "__SCENE__" not in engine
    for kind in sorted(kinds):
        assert '"%s"' % kind in engine, "the engine has no branch for %r" % kind
    for routine in ("function drawShelf(", "function drawCrate(",
                    "function drawTerminal(", "function drawConveyor(",
                    "function drawRadar("):
        assert routine in engine, "the engine is missing %s" % routine


def test_the_fallback_renderer_draws_the_warehouse(trained, layout):
    run = simulate.run_episode(trained["agent"], layout)
    scene = room5_scene.build_scene(layout, frames=run["frames"])
    kinds = {kind for row in scene["grid"]["tiles"] for kind in row}
    assert kinds - set(fallback_renderer.KNOWN_KINDS) == set(), \
        "the fallback has no colour for: %s" % (kinds
                                               - set(fallback_renderer.KNOWN_KINDS))
    figure = fallback_renderer.render_frame(scene, scene["frames"][-1])
    assert len(figure.axes[0].patches) >= 100
    figure.clf()


def test_every_required_graph_is_generated(trained, layout):
    """34."""
    results = [trained]
    run = simulate.run_episode(trained["agent"], layout)
    figures = {
        "episode reward": room5_plots.episode_reward(results),
        "moving average": room5_plots.moving_average_reward(results),
        "episode length": room5_plots.episode_length(results),
        "training success": room5_plots.training_success(results),
        "validation success": room5_plots.validation_success(results),
        "terminal rate": room5_plots.terminal_rate(results),
        "robot collisions": room5_plots.robot_collisions(results),
        "static collisions": room5_plots.static_collisions(results),
        "timeout rate": room5_plots.timeout_rate(results),
        "epsilon": room5_plots.epsilon_decay(results),
        "td error": room5_plots.td_error(results),
        "weight norm": room5_plots.weight_norm(results),
        "training time": room5_plots.training_time(results),
        "feature activity": room5_plots.feature_activity(results),
        "radar": room5_plots.radar_over_time(run),
        "nearest robot": room5_plots.nearest_robot_distance(run),
        "actions": room5_plots.action_distribution([run]),
        "layout map": room5_plots.layout_map(layout, run),
    }
    assert len(figures) == 18
    for name, figure in figures.items():
        assert figure is not None and figure.axes, name
        figure.clf()


def test_the_graphs_survive_being_given_nothing():
    for builder in (room5_plots.episode_reward, room5_plots.moving_average_reward,
                    room5_plots.episode_length, room5_plots.training_success,
                    room5_plots.validation_success, room5_plots.terminal_rate,
                    room5_plots.robot_collisions, room5_plots.static_collisions,
                    room5_plots.timeout_rate, room5_plots.epsilon_decay,
                    room5_plots.td_error, room5_plots.weight_norm,
                    room5_plots.training_time, room5_plots.feature_activity):
        figure = builder([])
        assert figure is not None
        figure.clf()
    assert room5_plots.radar_over_time(None) is not None
    assert room5_plots.generalisation_chart([]) is not None
    assert room5_plots.experiment_chart([]) is not None


# ---------------- experiments ----------------

def test_the_layout_count_experiment_evaluates_on_the_same_unseen_layouts():
    rows = experiments.layout_count_experiment(episodes=60, repeats=1)
    assert [row["value"] for row in rows] == list(experiments.LAYOUT_COUNTS)
    for row in rows:
        for key in ("mean_training_success", "mean_validation_success",
                    "mean_test_success"):
            assert key in row


def test_the_stress_test_labels_the_distribution_shift():
    rows = experiments.generalisation_stress_test(episodes=40, repeats=1)
    assert len(rows) == len(L.DIFFICULTIES)
    shifted = [row for row in rows if row["distribution_shift"]]
    assert len(shifted) == len(L.DIFFICULTIES) - 1


def test_the_baseline_policies_run_and_are_not_good():
    pool = L.generate_pool("test", 4)
    rows = experiments.baseline_policies(pool)
    assert len(rows) == 2
    for row in rows:
        assert row["mean_test_success"] <= 0.5, \
            "a trivial baseline should not be solving the room"


def test_best_performing_row_picks_the_highest_metric():
    rows = [{"mean_test_success": 0.1, "value": "a"},
            {"mean_test_success": 0.9, "value": "b"}]
    assert experiments.best_performing_row(rows)["value"] == "b"
    assert experiments.best_performing_row([]) is None
