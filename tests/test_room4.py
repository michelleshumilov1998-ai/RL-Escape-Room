"""Tests for Room 4 — the Drone Wind Tunnel.

Covers the thirty checks the Room 4 brief lists: numerical bounds, the dynamics,
wind, seeded turbulence, collisions, boundaries, the two-part landing rule, the
tile coder, the semi-gradient update, reproducibility, saving and loading, replay,
the renderer, the graphs, and the generalisation split.
"""

import json
import math
import os

import pytest

from plots import room4_plots
from renderers import base_renderer, iso_canvas, room4_scene
from rooms.room4 import chamber, experiments, sarsa_agent, simulate, storage
from rooms.room4 import environment as E
from rooms.room4.environment import Room4Env
from rooms.room4.tile_coder import TileCoder

REQUIRED_FRAME_KEYS = ("state", "x", "y", "vx", "vy", "speed", "action",
                       "thrust_x", "thrust_y", "wind_x", "wind_y",
                       "turbulence_x", "turbulence_y", "reward",
                       "cumulative_reward", "distance_to_goal", "collision",
                       "hard_landing", "safe_landing", "done", "event")


@pytest.fixture
def env():
    return Room4Env()


@pytest.fixture(scope="module")
def trained():
    return sarsa_agent.train(episodes=800, seed=1)


# ----------------------------------------------------------------------
# 1-4: numbers stay sane, and the dynamics are what they claim
# ----------------------------------------------------------------------

def test_state_values_stay_within_bounds(env):
    """1, 3: positions stay in the hall and velocity is clipped."""
    import random
    rng = random.Random(0)
    env.reset(seed=0)
    for _ in range(400):
        state, _, done, _ = env.step(rng.choice(env.actions()))
        x, y, velocity_x, velocity_y = state
        assert 0.0 <= x <= chamber.WIDTH
        assert 0.0 <= y <= chamber.HEIGHT
        assert abs(velocity_x) <= env.v_max + 1e-9
        assert abs(velocity_y) <= env.v_max + 1e-9
        assert all(math.isfinite(value) for value in state)
        if done:
            env.reset(seed=1)


def test_every_action_produces_a_valid_next_state(env):
    """2."""
    for action in env.actions():
        env.reset(seed=0)
        state, reward, done, info = env.step(action)
        assert len(state) == 4
        assert all(math.isfinite(value) for value in state)
        assert math.isfinite(reward)
        assert isinstance(done, bool)
        assert "speed" in info


def test_velocity_clipping_holds_at_full_thrust(env):
    """3."""
    env.reset(seed=0)
    for _ in range(60):
        state, _, _, _ = env.step(E.THRUST_RIGHT)
    assert abs(state[2]) <= env.v_max + 1e-9


def test_drag_is_applied(env):
    """4: with no thrust and no wind, speed decays by the drag factor."""
    quiet = Room4Env(wind_multiplier=0.0)
    quiet.reset(seed=0)
    quiet.state = (5.0, 1.0, 2.0, 0.0)
    state, _, _, _ = quiet.step(E.NO_THRUST)
    assert state[2] == pytest.approx(2.0 * quiet.drag, abs=1e-9)


def test_thrust_changes_velocity_not_position_directly(env):
    quiet = Room4Env(wind_multiplier=0.0)
    quiet.reset(seed=0)
    quiet.state = (5.0, 5.0, 0.0, 0.0)
    state, _, _, _ = quiet.step(E.THRUST_UP)
    assert state[3] > 0, "thrust up must produce upward velocity"
    assert state[1] > 5.0, "and the position follows from the velocity"


# ----------------------------------------------------------------------
# 5-6: wind and turbulence
# ----------------------------------------------------------------------

def test_wind_is_the_sum_of_the_fans():
    """5."""
    x, y = 5.0, 4.5
    expected_x = expected_y = 0.0
    for fan in chamber.FANS:
        force_x, force_y = fan.force_at(x, y)
        expected_x += force_x
        expected_y += force_y
    actual_x, actual_y = chamber.wind_at(x, y)
    assert actual_x == pytest.approx(expected_x)
    assert actual_y == pytest.approx(expected_y)


def test_wind_depends_only_on_position():
    """This is what keeps the environment Markovian without a clock."""
    first = chamber.wind_at(3.0, 7.0)
    second = chamber.wind_at(3.0, 7.0)
    assert first == second


def test_wind_falls_off_to_nothing_outside_a_fan():
    fan = chamber.FANS[0]
    far_x = fan.x + fan.radius + 1.0
    assert fan.force_at(far_x, fan.y) == (0.0, 0.0)


def test_the_wind_multiplier_scales_the_field():
    base = chamber.wind_at(5.0, 4.5, 1.0)
    doubled = chamber.wind_at(5.0, 4.5, 2.0)
    assert doubled[0] == pytest.approx(base[0] * 2)
    assert doubled[1] == pytest.approx(base[1] * 2)


def test_seeded_turbulence_is_reproducible():
    """6."""
    def fly():
        turbulent = Room4Env(turbulence=0.15)
        turbulent.reset(seed=99)
        return [turbulent.step(E.THRUST_RIGHT)[0] for _ in range(12)]
    assert fly() == fly()


def test_turbulence_is_off_by_default(env):
    env.reset(seed=0)
    _, _, _, info = env.step(E.NO_THRUST)
    assert info["turbulence_x"] == 0.0
    assert info["turbulence_y"] == 0.0


# ----------------------------------------------------------------------
# 7-9: obstacles and boundaries
# ----------------------------------------------------------------------

def test_obstacles_detect_collisions():
    """7."""
    circle = next(o for o in chamber.OBSTACLES if o.kind == "circle")
    assert circle.contains(circle.x, circle.y)
    assert not circle.contains(circle.x + circle.radius * 3, circle.y)
    rect = next(o for o in chamber.OBSTACLES if o.kind == "rect")
    middle_x = (rect.x0 + rect.x1) / 2
    middle_y = (rect.y0 + rect.y1) / 2
    assert rect.contains(middle_x, middle_y)
    assert not rect.contains(rect.x1 + 2.0, middle_y)


def test_the_drone_cannot_pass_through_an_obstacle():
    """8: substeps stop a fast drone tunnelling through a thin barrier."""
    rect = next(o for o in chamber.OBSTACLES if o.kind == "rect")
    quiet = Room4Env(wind_multiplier=0.0, v_max=6.0)
    quiet.reset(seed=0)
    # Aim straight at the barrier from the left, as fast as possible.
    quiet.state = (rect.x0 - 0.5, (rect.y0 + rect.y1) / 2, 6.0, 0.0)
    saw_collision = False
    for _ in range(6):
        state, _, _, info = quiet.step(E.THRUST_RIGHT)
        if info["collision"]:
            saw_collision = True
        assert chamber.blocking_obstacle(state[0], state[1]) is None, \
            "the drone ended up inside an obstacle"
    assert saw_collision


def test_a_collision_costs_and_slows_the_drone():
    circle = next(o for o in chamber.OBSTACLES if o.kind == "circle")
    quiet = Room4Env(wind_multiplier=0.0)
    quiet.reset(seed=0)
    quiet.state = (circle.x - circle.radius - 0.3, circle.y, 2.5, 0.0)
    _, reward, _, info = quiet.step(E.THRUST_RIGHT)
    assert info["collision"]
    assert reward <= quiet.obstacle_penalty


def test_boundary_handling_keeps_the_drone_inside(env):
    """9."""
    env.reset(seed=0)
    env.state = (0.3, 5.0, -3.0, 0.0)
    state, reward, _, info = env.step(E.THRUST_LEFT)
    assert info["boundary"]
    assert state[0] >= 0.0
    assert reward <= env.boundary_penalty
    assert chamber.inside_bounds(state[0], state[1])


# ----------------------------------------------------------------------
# 10-13: the landing rule
# ----------------------------------------------------------------------

def test_a_safe_landing_needs_position_and_speed(env):
    """10."""
    env.reset(seed=0)
    goal_x, goal_y = chamber.GOAL_POSITION
    env.state = (goal_x, goal_y, 0.0, 0.0)
    _, reward, done, info = env.step(E.NO_THRUST)
    assert info["safe_landing"]
    assert done
    assert reward > 100


def test_arriving_too_fast_is_not_a_success(env):
    """11: a hard landing, which does not count and does not end the episode."""
    env.reset(seed=0)
    goal_x, goal_y = chamber.GOAL_POSITION
    env.state = (goal_x - 0.2, goal_y, 1.4, 0.0)
    _, reward, done, info = env.step(E.NO_THRUST)
    assert info["hard_landing"]
    assert not info["safe_landing"]
    assert not done, "a hard landing lets the drone come round again"
    assert reward < 0


def test_a_severe_crash_terminates(env):
    """12.

    Arranged so the speed *after* drag and wind is unambiguously over the crash
    threshold: no wind, and a starting velocity chosen from the drag factor.
    """
    quiet = Room4Env(wind_multiplier=0.0, v_max=10.0)
    quiet.reset(seed=0)
    goal_x, goal_y = chamber.GOAL_POSITION
    entry_speed = (quiet.crash_speed + 1.0) / quiet.drag
    quiet.state = (goal_x, goal_y, entry_speed, 0.0)

    _, reward, done, info = quiet.step(E.NO_THRUST)
    assert info["speed"] >= quiet.crash_speed
    assert info["crashed"]
    assert done
    assert reward <= quiet.crash_penalty


def test_a_fast_drone_cannot_fly_straight_through_the_pad():
    """The pad is checked inside the substep walk, not only at the end.

    At full speed the drone covers more than the pad's diameter in a single step,
    so testing only the final position would miss the arrival entirely.
    """
    quiet = Room4Env(wind_multiplier=0.0, v_max=12.0)
    quiet.reset(seed=0)
    goal_x, goal_y = chamber.GOAL_POSITION
    speed = 12.0
    assert speed * quiet.dt > chamber.GOAL_RADIUS * 2, \
        "this test is pointless unless one step overshoots the pad"
    quiet.state = (goal_x - chamber.GOAL_RADIUS * 0.5, goal_y, speed, 0.0)

    _, _, done, info = quiet.step(E.NO_THRUST)
    assert info["crashed"] or info["hard_landing"] or info["safe_landing"], \
        "the drone passed through the pad without the arrival being noticed"


def test_reward_shaping_uses_the_distance_difference(env):
    """13: the shaping term is progress_scale x (previous - current) distance."""
    quiet = Room4Env(wind_multiplier=0.0, thrust_cost=0.0)
    quiet.reset(seed=0)
    quiet.state = (5.0, 5.0, 0.0, 0.0)
    before = chamber.distance_to_goal(5.0, 5.0)
    state, reward, _, info = quiet.step(E.THRUST_RIGHT)
    after = chamber.distance_to_goal(state[0], state[1])
    expected = quiet.step_cost + quiet.progress_scale * (before - after)
    assert reward == pytest.approx(expected, abs=1e-9)


def test_shaping_cannot_be_farmed_without_landing(env):
    """Circling must not pay: a loop back to the start nets the step costs."""
    quiet = Room4Env(wind_multiplier=0.0)
    quiet.reset(seed=0)
    total = 0.0
    for action in ([E.THRUST_RIGHT] * 6 + [E.THRUST_LEFT] * 12
                   + [E.THRUST_RIGHT] * 6):
        _, reward, done, _ = quiet.step(action)
        total += reward
        if done:
            break
    assert total < 0, "a round trip must not be profitable"


# ----------------------------------------------------------------------
# 14-17: the tile coder
# ----------------------------------------------------------------------

def test_the_tile_coder_returns_one_index_per_tiling(env):
    """14."""
    coder = TileCoder(env.state_bounds(), 8, 8, 5)
    indices = coder.active_tiles((5.0, 5.0, 0.0, 0.0), 0)
    assert len(indices) == 8
    assert len(set(indices)) == 8, "the tilings must not collide with each other"


def test_active_indices_are_inside_the_weight_vector(env):
    """15: checked across the corners and the middle of the state space."""
    coder = TileCoder(env.state_bounds(), 8, 8, 5)
    extremes = [0.0, chamber.WIDTH]
    velocities = [-env.v_max, 0.0, env.v_max]
    for x in extremes:
        for y in extremes:
            for velocity_x in velocities:
                for velocity_y in velocities:
                    for action in range(5):
                        for index in coder.active_tiles(
                                (x, y, velocity_x, velocity_y), action):
                            assert 0 <= index < coder.total_features


def test_out_of_range_values_are_clamped_not_wrapped(env):
    coder = TileCoder(env.state_bounds(), 8, 8, 5)
    inside = coder.active_tiles((chamber.WIDTH, chamber.HEIGHT,
                                 env.v_max, env.v_max), 0)
    beyond = coder.active_tiles((chamber.WIDTH + 50, chamber.HEIGHT + 50,
                                 env.v_max * 9, env.v_max * 9), 0)
    assert inside == beyond


def test_action_features_never_overlap(env):
    """16."""
    coder = TileCoder(env.state_bounds(), 8, 8, 5)
    state = (3.0, 7.0, 1.0, -1.0)
    blocks = [set(coder.active_tiles(state, action)) for action in range(5)]
    for first in range(5):
        for second in range(first + 1, 5):
            assert not blocks[first] & blocks[second]


def test_nearby_states_share_features_and_distant_ones_do_not(env):
    """The whole reason for tile coding."""
    coder = TileCoder(env.state_bounds(), 8, 8, 5)
    here = set(coder.active_tiles((5.0, 5.0, 0.0, 0.0), 0))
    nearby = set(coder.active_tiles((5.06, 5.0, 0.0, 0.0), 0))
    far = set(coder.active_tiles((0.5, 9.5, 0.0, 0.0), 0))
    assert len(here & nearby) >= 6
    assert not here & far


def test_the_tile_coder_rejects_a_nonsense_configuration(env):
    with pytest.raises(ValueError):
        TileCoder(env.state_bounds(), 0, 8, 5)
    with pytest.raises(ValueError):
        TileCoder(env.state_bounds(), 8, 1, 5)


def test_offsets_are_reproducible(env):
    first = TileCoder(env.state_bounds(), 8, 8, 5)
    second = TileCoder(env.state_bounds(), 8, 8, 5)
    assert first.offsets == second.offsets


# ----------------------------------------------------------------------
# 17-20: the agent and the update
# ----------------------------------------------------------------------

def test_q_value_matches_a_manual_sum(env):
    """17."""
    agent = sarsa_agent.SemiGradientAgent(env, 8, 8)
    state = (4.0, 6.0, 0.5, -0.5)
    indices = agent.coder.active_tiles(state, E.THRUST_UP)
    for position, index in enumerate(indices):
        agent.weights[index] = float(position + 1)
    expected = sum(range(1, len(indices) + 1))
    assert agent.q_value(state, E.THRUST_UP) == pytest.approx(expected)


def test_the_semi_gradient_update_matches_a_numerical_example(env):
    """18."""
    agent = sarsa_agent.SemiGradientAgent(env, 4, 8, alpha=0.4)
    state = (2.0, 2.0, 0.0, 0.0)
    action = E.THRUST_RIGHT
    indices = agent.coder.active_tiles(state, action)

    assert agent.effective_alpha == pytest.approx(0.4 / 4)
    assert agent.q_value(state, action) == 0.0

    target = 10.0
    delta = agent.update(state, action, target)
    assert delta == pytest.approx(10.0)
    # Each active weight moved by effective_alpha * delta.
    for index in indices:
        assert agent.weights[index] == pytest.approx(0.1 * 10.0)
    # So the new estimate is 4 * 1.0 = 4.0.
    assert agent.q_value(state, action) == pytest.approx(4.0)


def test_only_the_active_weights_change(env):
    agent = sarsa_agent.SemiGradientAgent(env, 8, 8)
    state = (1.0, 1.0, 0.0, 0.0)
    indices = set(agent.coder.active_tiles(state, E.NO_THRUST))
    agent.update(state, E.NO_THRUST, 5.0)
    changed = {index for index, weight in enumerate(agent.weights)
               if weight != 0.0}
    assert changed == indices


def test_a_terminal_update_omits_the_bootstrap_value(env):
    """19: at a terminal step the target is the reward alone."""
    agent = sarsa_agent.SemiGradientAgent(env, 4, 8, alpha=1.0, gamma=0.9)
    state = (8.0, 8.0, 0.0, 0.0)
    reward = 7.0
    delta = agent.update(state, E.NO_THRUST, reward)
    assert delta == pytest.approx(reward)


def test_epsilon_greedy_explores_and_exploits(env):
    """20."""
    import random
    agent = sarsa_agent.SemiGradientAgent(env, 8, 8)
    state = (5.0, 5.0, 0.0, 0.0)
    for index in agent.coder.active_tiles(state, E.THRUST_UP):
        agent.weights[index] = 10.0

    rng = random.Random(0)
    greedy = [agent.epsilon_greedy(state, 0.0, rng) for _ in range(20)]
    assert set(greedy) == {E.THRUST_UP}

    random_choices = [agent.epsilon_greedy(state, 1.0, rng) for _ in range(200)]
    assert len(set(random_choices)) > 1


def test_the_weight_norm_is_maintained_correctly(env):
    agent = sarsa_agent.SemiGradientAgent(env, 8, 8)
    agent.update((3.0, 3.0, 0.0, 0.0), E.THRUST_UP, 4.0)
    agent.update((6.0, 2.0, 1.0, 0.0), E.THRUST_LEFT, -2.0)
    exact = math.sqrt(sum(weight * weight for weight in agent.weights))
    assert agent.weight_norm() == pytest.approx(exact, abs=1e-9)


# ----------------------------------------------------------------------
# 21, 29, 30: training
# ----------------------------------------------------------------------

def test_training_is_reproducible_under_the_same_seed():
    """21."""
    def run():
        return sarsa_agent.train(episodes=40, seed=5)["history"]["reward"]
    assert run() == run()


def test_no_weight_becomes_infinite_or_nan():
    """29."""
    result = sarsa_agent.train(episodes=200, seed=0)
    assert result["agent"].is_finite()
    assert math.isfinite(result["weight_norm"])


def test_a_short_smoke_run_completes():
    """30."""
    result = sarsa_agent.train(episodes=15, seed=0)
    assert len(result["history"]["reward"]) == 15
    assert result["runtime_seconds"] > 0


def test_training_records_everything_the_graphs_need():
    result = sarsa_agent.train(episodes=25, seed=0)
    for key in ("reward", "length", "success", "epsilon", "final_distance",
                "landing_speed", "collisions", "weight_norm", "td_error",
                "active_features", "hard_landing", "crash", "timeout", "elapsed"):
        assert key in result["history"], "history is missing %r" % key
        assert len(result["history"][key]) == 25


def test_the_drone_learns_to_land_safely(trained):
    """The room has to actually work."""
    batch = simulate.run_many(trained["agent"], episodes=8)
    assert batch["success_rate"] >= 0.5, \
        "the trained drone did not land safely often enough"
    assert batch["mean_landing_speed"] <= 1.0


def test_a_worked_example_update_is_recorded(trained):
    example = trained["example_update"]
    assert example is not None
    assert len(example["active_tiles"]) == trained["tile_coder"]["num_tilings"]
    assert example["td_error"] == pytest.approx(
        example["target"] - example["q_before"])


# ----------------------------------------------------------------------
# 22-23: saving, loading and replay
# ----------------------------------------------------------------------

def test_saving_and_loading_restore_identical_weights(tmp_path, monkeypatch):
    """22."""
    monkeypatch.setattr(storage, "SAVE_DIRECTORY", str(tmp_path))
    result = sarsa_agent.train(episodes=60, seed=0)
    path = storage.save_model("unit_test", result)
    assert os.path.isfile(path)

    fresh = sarsa_agent.SemiGradientAgent(Room4Env(), 8, 8)
    storage.load_into("unit_test", fresh)
    assert fresh.weights == result["agent"].weights

    state = (3.5, 6.5, 0.4, -0.2)
    for action in E.ACTIONS:
        assert fresh.q_value(state, action) == pytest.approx(
            result["agent"].q_value(state, action))


def test_loading_validates_the_tile_coder_and_the_room(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "SAVE_DIRECTORY", str(tmp_path))
    result = sarsa_agent.train(episodes=20, seed=0)
    path = storage.save_model("checks", result)

    with pytest.raises(storage.ModelFileError, match="No saved model"):
        storage.load_into("absent", sarsa_agent.SemiGradientAgent(Room4Env(), 8, 8))

    # A different tile configuration must be refused, not silently loaded.
    mismatched = sarsa_agent.SemiGradientAgent(Room4Env(), 4, 6)
    with pytest.raises(storage.ModelFileError):
        storage.load_into("checks", mismatched)
    assert all(weight == 0.0 for weight in mismatched.weights), \
        "a refused load must leave the agent untouched"

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for field, value, message in (
            ("format_version", 99, "file format"),
            ("room", 1, "not a Room 4 model"),
            ("environment_version", "other", "environment"),
            ("num_actions", 3, "actions")):
        broken = dict(payload)
        broken[field] = value
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(broken, handle)
        with pytest.raises(storage.ModelFileError, match=message):
            storage.load_into("checks",
                              sarsa_agent.SemiGradientAgent(Room4Env(), 8, 8))


def test_replay_reproduces_the_stored_trajectory(trained):
    """23."""
    first = simulate.run_episode(trained["agent"], seed=3)
    second = simulate.run_episode(trained["agent"], seed=3)
    assert first["frames"] == second["frames"]


def test_a_turbulent_replay_is_exact(trained):
    """Turbulence is random, so the frames must hold what actually happened."""
    parameters = {"turbulence": 0.12}
    first = simulate.run_episode(trained["agent"], seed=8,
                                 environment_parameters=parameters)
    second = simulate.run_episode(trained["agent"], seed=8,
                                  environment_parameters=parameters)
    assert first["frames"] == second["frames"]
    assert any(frame["turbulence_x"] != 0.0 for frame in first["frames"][1:])


def test_every_frame_carries_the_required_fields(trained):
    run = simulate.run_episode(trained["agent"], seed=0)
    for frame in run["frames"]:
        for key in REQUIRED_FRAME_KEYS:
            assert key in frame, "frame %d is missing %r" % (frame["step"], key)


def test_frames_join_up_continuously(trained):
    run = simulate.run_episode(trained["agent"], seed=0)
    for previous, current in zip(run["frames"], run["frames"][1:]):
        assert (current["from_x"], current["from_y"]) == \
            (previous["x"], previous["y"])


# ----------------------------------------------------------------------
# 24-26: the renderer, the graphs and the split
# ----------------------------------------------------------------------

def test_the_renderer_draws_the_drone_fans_obstacles_and_pad(trained):
    """24."""
    run = simulate.run_episode(trained["agent"], seed=0)
    scene = room4_scene.build_scene(frames=run["frames"])
    assert base_renderer.validate_scene(scene) == []
    assert scene["meta"]["sceneKind"] == "continuous"
    assert len(scene["continuous"]["fans"]) == len(chamber.FANS)
    assert len(scene["continuous"]["obstacles"]) == len(chamber.OBSTACLES)
    assert scene["continuous"]["windField"], "no wind samples for the renderer"

    engine = iso_canvas.build_html(scene)
    assert "__SCENE__" not in engine
    for routine in ("function drawDrone(", "function drawFan(",
                    "function drawObstacle(", "function drawLandingPad(",
                    "function drawWindField("):
        assert routine in engine, "the engine is missing %s" % routine

    restored = json.loads(json.dumps(scene))
    assert restored["episodeId"] == scene["episodeId"]


def test_the_scene_maps_metres_onto_cells_consistently():
    row, col = room4_scene.world_to_cell(*chamber.GOAL_POSITION)
    assert col == pytest.approx(chamber.GOAL_POSITION[0])
    assert row == pytest.approx(chamber.HEIGHT - chamber.GOAL_POSITION[1])


def test_every_required_graph_is_generated(trained):
    """25."""
    results = [trained]
    batch = simulate.run_many(trained["agent"], episodes=3)
    run = batch["runs"][0]
    figures = {
        "episode reward": room4_plots.episode_reward(results),
        "moving average": room4_plots.moving_average_reward(results),
        "episode length": room4_plots.episode_length(results),
        "success rate": room4_plots.success_rate(results),
        "final distance": room4_plots.final_distance(results),
        "landing speed": room4_plots.landing_speed(results),
        "collisions": room4_plots.collision_count(results),
        "epsilon": room4_plots.epsilon_decay(results),
        "weight norm": room4_plots.weight_norm(results),
        "td error": room4_plots.td_error(results),
        "active tiles": room4_plots.active_tiles(results),
        "training time": room4_plots.training_time(results),
        "trajectories": room4_plots.trajectories(batch["runs"]),
        "wind field": room4_plots.wind_field(),
        "velocity": room4_plots.velocity_over_time(run),
        "forces": room4_plots.wind_and_thrust(run),
        "landing": room4_plots.landing_analysis(run),
        "value slice": room4_plots.value_slice(trained["agent"], samples=10),
    }
    assert len(figures) == 18
    for name, figure in figures.items():
        assert figure is not None and figure.axes, name
        figure.clf()


def test_the_graphs_survive_being_given_nothing():
    for builder in (room4_plots.episode_reward, room4_plots.moving_average_reward,
                    room4_plots.episode_length, room4_plots.success_rate,
                    room4_plots.final_distance, room4_plots.landing_speed,
                    room4_plots.collision_count, room4_plots.epsilon_decay,
                    room4_plots.weight_norm, room4_plots.td_error,
                    room4_plots.active_tiles, room4_plots.training_time):
        figure = builder([])
        assert figure is not None
        figure.clf()
    assert room4_plots.value_slice(None) is not None
    assert room4_plots.trajectories([]) is not None
    assert room4_plots.experiment_chart([]) is not None
    assert room4_plots.generalisation_chart([]) is not None


def test_the_evaluation_starts_are_never_trained_on():
    """26: the generalisation split has to be real."""
    training = set(chamber.TRAINING_STARTS)
    validation = set(chamber.VALIDATION_STARTS)
    test = set(chamber.TEST_STARTS)
    assert training and validation and test
    assert not training & validation
    assert not training & test
    assert not validation & test


def test_every_release_point_is_clear_of_obstacles():
    assert chamber.start_is_clear()


def test_training_only_ever_starts_from_the_training_points():
    """The split is enforced in the training loop, not just declared."""
    result = sarsa_agent.train(episodes=30, seed=0,
                               training_starts=chamber.TRAINING_STARTS)
    assert result["training_starts"] == [list(start)
                                         for start in chamber.TRAINING_STARTS]
    for start in result["training_starts"]:
        assert tuple(start) not in set(chamber.TEST_STARTS)


def test_the_generalisation_test_labels_seen_and_unseen_groups():
    rows = experiments.generalisation_test(episodes=60, repeats=1)
    assert len(rows) == 3
    seen = [row for row in rows if row["seen_during_training"]]
    unseen = [row for row in rows if not row["seen_during_training"]]
    assert len(seen) == 1
    assert len(unseen) == 2
    for row in rows:
        assert "mean_eval_success" in row
        assert row["starts"]


def test_the_tile_resolution_experiment_reports_memory_cost():
    rows = experiments.tile_resolution_experiment(episodes=25, repeats=1)
    assert len(rows) == len(experiments.TILE_CONFIGURATIONS)
    for row in rows:
        assert row["weights"] > 0
    # More tilings and finer tiles must mean more weights.
    assert rows[0]["weights"] < rows[-1]["weights"]
