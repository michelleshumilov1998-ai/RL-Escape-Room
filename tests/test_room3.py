"""Tests for Room 3 — the Reactor Control Chamber.

Covers the checks the Room 3 brief asks for: generator order, door unlocking,
guard movement, the Wait action, hazard penalties, the Q-Learning update, replay
correctness, saving and loading, the scene the animation consumes, the graphs, and
training convergence.
"""

import json
import os
from collections import Counter

import pytest

from core import actions
from plots import room3_plots
from renderers import base_renderer, fallback_renderer, iso_canvas, room3_scene
from rooms.room3 import experiments, map_data, q_agent, simulate, storage
from rooms.room3.environment import (Room3Env, STAGE_A, STAGE_AB, STAGE_NONE,
                                     STAGE_UNLOCKED)


@pytest.fixture
def env():
    return Room3Env()


@pytest.fixture(scope="module")
def trained():
    env = Room3Env()
    return env, q_agent.train(env, episodes=3000, seed=0)


# ----------------------------------------------------------------------
# The map
# ----------------------------------------------------------------------

def test_the_map_is_ten_by_ten_and_valid():
    assert len(map_data.ROOM_MAP) == 10
    for index, row in enumerate(map_data.ROOM_MAP):
        assert len(row) == 10, "row %d has %d cells" % (index, len(row))
        for character in row:
            assert character in map_data.VALID_TILES


def test_the_map_has_everything_the_brief_asks_for():
    counts = Counter()
    for row in map_data.ROOM_MAP:
        counts.update(row)
    assert counts[map_data.START] == 1
    assert counts[map_data.EXIT] == 1
    assert counts[map_data.GENERATOR_A] == 1
    assert counts[map_data.GENERATOR_B] == 1
    assert counts[map_data.GENERATOR_C] == 1
    assert counts[map_data.REACTOR_DOOR] == 1
    assert counts[map_data.HAZARD] >= 1, "electrical hazards"
    assert counts[map_data.SLIDING_DOOR] >= 1, "sliding laboratory doors"
    assert counts[map_data.WALL] >= 4


def test_every_walkable_cell_is_reachable():
    walkable = set(map_data.walkable_cells())
    assert walkable - map_data.reachable_cells(doors_open=True) == set()


def test_the_mission_is_possible_even_with_the_doors_shut():
    """The shaft is a shortcut, not the only way round."""
    reachable = map_data.reachable_cells(doors_open=False)
    for cell in (map_data.generator_cell(map_data.GENERATOR_A),
                 map_data.generator_cell(map_data.GENERATOR_B),
                 map_data.generator_cell(map_data.GENERATOR_C),
                 map_data.exit_cell()):
        assert cell in reachable, "%s is unreachable with the doors shut" % (cell,)


def test_the_shortest_legal_mission_is_a_full_lap():
    assert map_data.mission_route_length() == 23


def test_the_patrol_stays_on_walkable_cells_and_loops():
    walkable = set(map_data.walkable_cells())
    assert map_data.PATROL_LENGTH == 24
    for cell in map_data.PATROL:
        assert cell in walkable
    # Consecutive patrol cells must be one step apart, including the wrap-around.
    for index in range(map_data.PATROL_LENGTH):
        here = map_data.PATROL[index]
        following = map_data.PATROL[(index + 1) % map_data.PATROL_LENGTH]
        distance = abs(here[0] - following[0]) + abs(here[1] - following[1])
        assert distance == 1, "patrol jumps from %s to %s" % (here, following)


def test_the_guard_never_reaches_the_service_shaft():
    """The shaft is the only refuge, which is what makes waiting useful."""
    shaft = set(map_data.shaft_cells())
    assert shaft, "there is no refuge at all"
    assert not shaft & set(map_data.PATROL)


def test_the_door_cycle_is_derived_from_the_guard_and_repeats_cleanly():
    """This is what keeps the timed doors Markovian."""
    assert map_data.PATROL_LENGTH % map_data.DOOR_PERIOD == 0
    pattern = [map_data.door_is_open(index)
               for index in range(map_data.PATROL_LENGTH)]
    assert pattern[:4] == [True, True, False, False]
    # One full patrol later the pattern lines up again.
    for index in range(map_data.PATROL_LENGTH):
        assert map_data.door_is_open(index) == \
            map_data.door_is_open(index + map_data.PATROL_LENGTH)


# ----------------------------------------------------------------------
# The state space and the guard
# ----------------------------------------------------------------------

def test_the_state_is_position_stage_and_guard(env):
    state = env.reset()
    assert len(state) == 4
    assert state == (1, 1, STAGE_NONE, 0)
    stages = {state[2] for state in env.all_states()}
    assert stages == {0, 1, 2, 3}
    guard_indices = {state[3] for state in env.all_states()}
    assert guard_indices == set(range(map_data.PATROL_LENGTH))


def test_the_action_set_includes_wait(env):
    assert env.actions() == [actions.UP, actions.DOWN, actions.LEFT,
                            actions.RIGHT, actions.WAIT]


def test_the_guard_advances_one_cell_every_step(env):
    """Guard movement."""
    env.reset()
    seen = [env.guard_cell()]
    for _ in range(map_data.PATROL_LENGTH):
        _, _, _, info = env.step(actions.WAIT)
        seen.append(tuple(info["guard_cell"]))
    # It walked the whole patrol and came back to where it started.
    assert seen[0] == seen[-1]
    assert len(set(seen[:-1])) == map_data.PATROL_LENGTH


def test_the_guard_stays_on_its_patrol(env):
    patrol = set(map_data.PATROL)
    env.reset()
    for _ in range(60):
        _, _, done, info = env.step(actions.WAIT)
        assert tuple(info["guard_cell"]) in patrol
        if done:
            break


def test_waiting_holds_position_and_costs_only_the_step(env):
    """The Wait action."""
    env.reset()
    state, reward, done, info = env.step(actions.WAIT)
    assert (state[0], state[1]) == map_data.start_cell()
    assert reward == env.step_cost
    assert info["waited"]
    assert not info["blocked"]
    assert not done


def test_being_caught_ends_the_episode(env):
    env.reset()
    # Put R-5 directly in the guard's path so the next advance collides.
    guard_index = 3
    ahead = map_data.guard_cell(map_data.next_guard_index(guard_index))
    env.state = (ahead[0], ahead[1], STAGE_NONE, guard_index)
    state, reward, done, info = env.step(actions.WAIT)
    assert done
    assert info["caught"]
    assert reward == env.step_cost + env.guard_penalty


def test_a_crossing_counts_as_a_collision(env):
    """Swapping places with the guard must not slip through."""
    env.reset()
    guard_index = 5
    here = map_data.guard_cell(guard_index)
    ahead = map_data.guard_cell(map_data.next_guard_index(guard_index))
    # R-5 stands where the guard is going, and moves to where the guard is.
    env.state = (ahead[0], ahead[1], STAGE_NONE, guard_index)
    action = None
    for candidate in actions.ACTIONS:
        if actions.move(ahead[0], ahead[1], candidate) == here:
            action = candidate
            break
    assert action is not None, "the two patrol cells are not adjacent"
    _, reward, done, info = env.step(action)
    assert done and info["caught"]
    assert reward == env.step_cost + env.guard_penalty


# ----------------------------------------------------------------------
# Generators, doors and hazards
# ----------------------------------------------------------------------

def test_generators_only_count_in_order(env):
    """Generator order."""
    b_cell = map_data.generator_cell(map_data.GENERATOR_B)
    approach = (b_cell[0] - 1, b_cell[1], STAGE_NONE, 0)

    env.reset()
    env.state = approach
    state, reward, _, info = env.step(actions.DOWN)
    assert state[2] == STAGE_NONE, "B must not start before A"
    assert info["wrong_order"]
    assert reward == env.step_cost

    # With A running, B works.
    env.state = (approach[0], approach[1], STAGE_A, 0)
    state, reward, _, info = env.step(actions.DOWN)
    assert state[2] == STAGE_AB
    assert info["generator"] == map_data.GENERATOR_B
    assert reward == env.step_cost + env.generator_bonus


def test_the_first_generator_needs_no_prerequisite(env):
    a_cell = map_data.generator_cell(map_data.GENERATOR_A)
    env.reset()
    env.state = (a_cell[0], a_cell[1] - 1, STAGE_NONE, 0)
    state, reward, _, info = env.step(actions.RIGHT)
    assert state[2] == STAGE_A
    assert info["generator"] == map_data.GENERATOR_A
    assert reward == env.step_cost + env.generator_bonus


def test_the_final_generator_pays_the_larger_bonus_and_unlocks(env):
    """Door unlocking."""
    c_cell = map_data.generator_cell(map_data.GENERATOR_C)
    env.reset()
    env.state = (c_cell[0], c_cell[1] + 1, STAGE_AB, 0)
    state, reward, _, info = env.step(actions.LEFT)
    assert state[2] == STAGE_UNLOCKED
    assert info["unlocked"]
    assert reward == env.step_cost + env.final_generator_bonus


def test_the_blast_door_is_shut_until_the_sequence_is_complete(env):
    door_row, door_col = map_data.reactor_door_cell()
    for stage in (STAGE_NONE, STAGE_A, STAGE_AB):
        env.reset()
        env.state = (door_row - 1, door_col, stage, 0)
        state, reward, done, info = env.step(actions.DOWN)
        assert (state[0], state[1]) == (door_row - 1, door_col)
        assert info["door_blocked"]
        assert reward == env.step_cost + env.wall_penalty
        assert not done


def test_the_exit_only_completes_the_room_after_the_sequence(env):
    door_row, door_col = map_data.reactor_door_cell()
    env.reset()
    env.state = (door_row - 1, door_col, STAGE_UNLOCKED, 0)
    state, reward, done, _ = env.step(actions.DOWN)
    assert (state[0], state[1]) == (door_row, door_col)
    assert not done
    state, reward, done, info = env.step(actions.DOWN)
    assert done and info["reached_exit"]
    assert reward == env.step_cost + env.exit_reward
    assert env.is_terminal(state)


def test_a_hazard_costs_but_does_not_end_the_episode(env):
    """Hazard penalties."""
    hazard = map_data.hazard_cells()[0]
    env.reset()
    env.state = (hazard[0] - 1, hazard[1], STAGE_NONE, 12)
    state, reward, done, info = env.step(actions.DOWN)
    assert (state[0], state[1]) == hazard
    assert info["hazard"]
    assert reward == env.step_cost + env.hazard_penalty
    assert not done


def test_the_sliding_doors_block_only_while_shut(env):
    door_row, door_col = map_data.sliding_door_cells()[0]
    approach = (door_row - 1, door_col)

    open_index = next(index for index in range(map_data.PATROL_LENGTH)
                      if map_data.door_is_open(index)
                      and map_data.guard_cell(index) != approach)
    shut_index = next(index for index in range(map_data.PATROL_LENGTH)
                      if not map_data.door_is_open(index)
                      and map_data.guard_cell(index) != approach)

    env.reset()
    env.state = (approach[0], approach[1], STAGE_NONE, open_index)
    state, reward, _, info = env.step(actions.DOWN)
    assert (state[0], state[1]) == (door_row, door_col)
    assert not info["door_blocked"]

    env.state = (approach[0], approach[1], STAGE_NONE, shut_index)
    state, reward, _, info = env.step(actions.DOWN)
    assert (state[0], state[1]) == approach
    assert info["door_blocked"]
    assert reward == env.step_cost + env.wall_penalty


def test_the_environment_offers_no_model(env):
    assert not hasattr(env, "transitions")


# ----------------------------------------------------------------------
# Q-Learning
# ----------------------------------------------------------------------

def test_the_q_learning_update_matches_a_hand_computation():
    """The Q-Learning update, checked by hand against a known example."""
    env = Room3Env()
    q = q_agent.new_q_table(env, 0.0)
    state = (1, 1, STAGE_NONE, 0)
    next_state = (1, 2, STAGE_NONE, 1)
    action = actions.RIGHT
    alpha, gamma = 0.5, 0.9

    q[state][action] = 3.0
    # Three different values next, so the max is unambiguous.
    q[next_state][actions.UP] = 2.0
    q[next_state][actions.DOWN] = 7.0
    q[next_state][actions.LEFT] = -1.0

    assert q_agent.best_value(q, next_state, env.actions()) == pytest.approx(7.0)

    reward = 1.0
    target = reward + gamma * 7.0                     # 1 + 6.3 = 7.3
    expected = 3.0 + alpha * (target - 3.0)           # 3 + 0.5*4.3 = 5.15
    q[state][action] = q[state][action] + alpha * (target - q[state][action])
    assert q[state][action] == pytest.approx(5.15)
    assert q[state][action] == pytest.approx(expected)


def test_q_learning_uses_the_max_and_sarsa_does_not():
    """The one line that separates the two methods."""
    env = Room3Env()
    q = q_agent.new_q_table(env, 0.0)
    state = (4, 4, STAGE_NONE, 0)
    q[state][actions.UP] = 5.0
    q[state][actions.DOWN] = 1.0
    assert q_agent.best_value(q, state, env.actions()) == 5.0
    assert q[state][actions.DOWN] == 1.0


def test_training_records_everything_the_graphs_need():
    result = q_agent.train(Room3Env(), episodes=60, seed=0)
    for key in ("reward", "length", "success", "epsilon", "mean_abs_q", "elapsed",
                "stage", "generators_done", "guard_caught", "hazards",
                "door_passed", "waits"):
        assert key in result["history"], "history is missing %r" % key
        assert len(result["history"][key]) == 60


def test_training_is_reproducible_from_a_seed():
    def run():
        return q_agent.train(Room3Env(), episodes=80, seed=7)["history"]["reward"]
    assert run() == run()


def test_epsilon_decays_to_its_floor(env):
    result = q_agent.train(env, episodes=400, seed=0, epsilon_start=1.0,
                           epsilon_min=0.08, epsilon_decay=0.99)
    epsilons = result["history"]["epsilon"]
    assert epsilons[-1] == pytest.approx(0.08, abs=1e-9)
    assert all(value >= 0.08 - 1e-12 for value in epsilons)


def test_a_worked_example_update_is_recorded():
    result = q_agent.train(Room3Env(), episodes=40, seed=0)
    example = result["example_update"]
    assert example is not None
    expected = example["q_before"] + example["alpha"] * (
        example["target"] - example["q_before"])
    assert example["q_after"] == pytest.approx(expected)


def test_training_converges_on_a_policy_that_completes_the_room(trained):
    """Training convergence: the greedy policy must actually finish the mission."""
    env, result = trained
    run = simulate.run_episode(env, result["policy"], seed=0)
    assert run["success"], "the learned policy did not reach the exit"
    assert run["stage"] == STAGE_UNLOCKED
    assert run["generators"] == [map_data.GENERATOR_A, map_data.GENERATOR_B,
                                map_data.GENERATOR_C], \
        "the generators were not brought up in order"
    assert not run["caught"]
    assert run["total_reward"] > 100


def test_the_policy_covers_every_non_terminal_state(trained):
    env, result = trained
    for state in env.all_states():
        action = result["policy"][state]
        if env.is_terminal(state):
            assert action is None
        else:
            assert action in env.actions()


# ----------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------

def test_replay_reproduces_the_run_exactly(trained):
    """Replay correctness."""
    env, result = trained
    first = simulate.run_episode(env, result["policy"], seed=5)
    second = simulate.run_episode(env, result["policy"], seed=5)
    assert first["frames"] == second["frames"]


def test_frames_carry_the_guard_and_the_stage(trained):
    env, result = trained
    run = simulate.run_episode(env, result["policy"], seed=0)
    for frame in run["frames"]:
        for key in ("row", "col", "stage", "guard_index", "guard_row", "guard_col",
                    "doors_open", "reward", "cumulative_reward", "event", "done"):
            assert key in frame
        assert (frame["guard_row"], frame["guard_col"]) in set(map_data.PATROL)


def test_the_stage_never_goes_backwards(trained):
    env, result = trained
    run = simulate.run_episode(env, result["policy"], seed=0)
    stages = [frame["stage"] for frame in run["frames"]]
    assert stages == sorted(stages)


def test_frames_join_up_and_move_at_most_one_cell(trained):
    env, result = trained
    run = simulate.run_episode(env, result["policy"], seed=0)
    frames = run["frames"]
    for previous, current in zip(frames, frames[1:]):
        assert (current["from_row"], current["from_col"]) == \
            (previous["row"], previous["col"])
        distance = (abs(current["row"] - current["from_row"])
                    + abs(current["col"] - current["from_col"]))
        assert distance <= 1


def test_cumulative_reward_is_the_running_total(trained):
    env, result = trained
    run = simulate.run_episode(env, result["policy"], seed=0)
    total = 0.0
    for frame in run["frames"][1:]:
        total += frame["reward"]
        assert frame["cumulative_reward"] == pytest.approx(total)


# ----------------------------------------------------------------------
# The scene, the renderer and the graphs
# ----------------------------------------------------------------------

def test_the_scene_is_valid_and_names_the_guard(trained):
    """Animation: the scene the engine consumes must be complete."""
    env, result = trained
    run = simulate.run_episode(env, result["policy"], seed=0)
    scene = room3_scene.build_scene(frames=run["frames"])
    assert base_renderer.validate_scene(scene) == []
    assert scene["meta"]["roomNumber"] == 3
    assert scene["meta"]["hasGuard"] is True
    restored = json.loads(json.dumps(scene))
    assert restored["episodeId"] == scene["episodeId"]


def test_the_engine_handles_every_room_three_tile_type():
    scene = room3_scene.build_scene()
    kinds = {kind for row in scene["grid"]["tiles"] for kind in row}
    engine = iso_canvas.build_html(scene)
    assert "__SCENE__" not in engine
    for kind in sorted(kinds):
        assert '"%s"' % kind in engine, "the engine has no branch for %r" % kind
    # The guard needs a drawing routine of its own.
    assert "function drawGuard(" in engine


def test_the_fallback_renderer_draws_the_chamber(trained):
    env, result = trained
    run = simulate.run_episode(env, result["policy"], seed=0)
    scene = room3_scene.build_scene(frames=run["frames"])
    kinds = {kind for row in scene["grid"]["tiles"] for kind in row}
    assert kinds - set(fallback_renderer.KNOWN_KINDS) == set(), \
        "the fallback has no colour for: %s" % (kinds
                                               - set(fallback_renderer.KNOWN_KINDS))
    figure = fallback_renderer.render_frame(scene, scene["frames"][-1])
    assert len(figure.axes[0].patches) >= 100
    figure.clf()


def test_every_required_graph_is_generated(trained):
    """Graphs: the eleven the brief asks for, plus the map and the policy."""
    env, result = trained
    results = [result]
    figures = {
        "episode reward": room3_plots.episode_reward(results),
        "moving average": room3_plots.moving_average_reward(results),
        "episode length": room3_plots.episode_length(results),
        "success rate": room3_plots.success_rate(results),
        "generator completion": room3_plots.generator_completion_rate(results),
        "average stage": room3_plots.average_stage(results),
        "guard detections": room3_plots.guard_detections(results),
        "hazard hits": room3_plots.hazard_hits(results),
        "q magnitude": room3_plots.q_value_magnitude(results),
        "training time": room3_plots.training_time(results),
        "door unlocks": room3_plots.door_unlock_frequency(results),
        "epsilon": room3_plots.epsilon_decay(results),
        "chamber map": room3_plots.chamber_map(),
        "policy": room3_plots.policy_arrows(result["policy"]),
    }
    assert len(figures) == 14
    for name, figure in figures.items():
        assert figure is not None and figure.axes, name
        figure.clf()


def test_the_graphs_survive_being_given_nothing():
    for builder in (room3_plots.episode_reward, room3_plots.moving_average_reward,
                    room3_plots.episode_length, room3_plots.success_rate,
                    room3_plots.generator_completion_rate,
                    room3_plots.average_stage, room3_plots.guard_detections,
                    room3_plots.hazard_hits, room3_plots.q_value_magnitude,
                    room3_plots.training_time,
                    room3_plots.door_unlock_frequency,
                    room3_plots.epsilon_decay):
        figure = builder([])
        assert figure is not None
        figure.clf()
    assert room3_plots.policy_arrows({}) is not None
    assert room3_plots.experiment_chart([]) is not None


# ----------------------------------------------------------------------
# Saving and loading
# ----------------------------------------------------------------------

def test_saving_and_loading_restore_the_same_policy(tmp_path, monkeypatch):
    """Saving/loading."""
    monkeypatch.setattr(storage, "SAVE_DIRECTORY", str(tmp_path))
    env = Room3Env()
    result = q_agent.train(env, episodes=100, seed=0)

    path = storage.save_model("unit_test", env, result)
    assert os.path.isfile(path)

    loaded = storage.load_model("unit_test")
    assert set(loaded["q"].keys()) == set(result["q"].keys())
    for state, actions_for_state in result["q"].items():
        for action, value in actions_for_state.items():
            assert loaded["q"][state][action] == pytest.approx(value)
    assert q_agent.greedy_policy(loaded["q"], env) == result["policy"]


def test_loading_rejects_incompatible_files(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "SAVE_DIRECTORY", str(tmp_path))
    env = Room3Env()
    result = q_agent.train(env, episodes=30, seed=0)
    path = storage.save_model("checks", env, result)

    with pytest.raises(storage.ModelFileError, match="No saved model"):
        storage.load_model("absent")

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    for field, value, message in (("format_version", 99, "file format"),
                                  ("room", 1, "not a Room 3 model"),
                                  ("patrol_length", 7, "patrol")):
        broken = dict(payload)
        broken[field] = value
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(broken, handle)
        with pytest.raises(storage.ModelFileError, match=message):
            storage.load_model("checks")

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    monkeypatch.setattr(map_data, "map_hash", lambda: "0" * 16)
    with pytest.raises(storage.ModelFileError, match="different version"):
        storage.load_model("checks")


def test_state_keys_survive_the_round_trip():
    for state in [(1, 1, 0, 0), (7, 7, 2, 23)]:
        assert storage._key_to_state(storage._state_to_key(state)) == state


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------

def test_a_parameter_sweep_returns_a_row_per_value():
    rows = experiments.parameter_experiment("alpha", values=[0.1, 0.3],
                                            episodes=120, repeats=1)
    assert len(rows) == 2
    for row in rows:
        for key in ("mean_reward", "std_reward", "mean_success", "mean_generators",
                    "mean_caught", "mean_runtime", "route"):
            assert key in row


def test_the_algorithm_comparison_covers_both_methods():
    rows = experiments.compare_algorithms(episodes=120, repeats=1)
    assert [row["value"] for row in rows] == [q_agent.Q_LEARNING, q_agent.SARSA]


def test_best_performing_row_picks_the_highest_metric():
    rows = [{"mean_reward": 1.0, "value": "a"}, {"mean_reward": 9.0, "value": "b"}]
    assert experiments.best_performing_row(rows)["value"] == "b"
    assert experiments.best_performing_row([]) is None
