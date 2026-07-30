"""Tests for Room 2's replay, rendering, saved models and graphs.

Covers the remaining checks from the Room 2 brief: replay correctness,
saving/loading, that the graphs are generated, and that the animation scene the
renderer consumes is complete and valid.
"""

import json
import os

import pytest

from core import actions
from plots import room2_plots
from renderers import base_renderer, fallback_renderer, iso_canvas, room2_scene
from rooms.room2 import map_data, sarsa_agent, simulate, storage
from rooms.room2.environment import Room2Env

REQUIRED_FRAME_KEYS = ("state", "row", "col", "has_keycard", "from_row", "from_col",
                       "action", "reward", "cumulative_reward", "event",
                       "pit_fall", "bridge_collapsed", "keycard_collected",
                       "reached_exit", "collapsed", "done", "status")


@pytest.fixture(scope="module")
def trained():
    env = Room2Env()
    return env, sarsa_agent.train(env, episodes=800, seed=0)


@pytest.fixture(scope="module")
def bridge_run():
    """A Q-Learning run in Markov mode, which actually crosses the bridge."""
    env = Room2Env(include_bridge_state=True)
    result = sarsa_agent.train(env, algorithm=sarsa_agent.Q_LEARNING,
                               episodes=2500, seed=0)
    return env, simulate.run_episode(env, result["policy"], seed=1)


# ----------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------

def test_every_frame_carries_the_required_fields(trained):
    env, result = trained
    run = simulate.run_episode(env, result["policy"], seed=3)
    for frame in run["frames"]:
        for key in REQUIRED_FRAME_KEYS:
            assert key in frame, "frame %d is missing %r" % (frame["step"], key)


def test_replay_reproduces_the_run_exactly(trained):
    """Replay correctness: the same policy and seed give an identical recording."""
    env, result = trained
    first = simulate.run_episode(env, result["policy"], seed=11)
    second = simulate.run_episode(env, result["policy"], seed=11)
    assert len(first["frames"]) == len(second["frames"])
    for left, right in zip(first["frames"], second["frames"]):
        assert left == right
    for key in ("success", "steps", "total_reward", "pit_falls",
                "bridges_collapsed", "keycard_collected"):
        assert first[key] == second[key]


def test_frames_join_up_into_a_continuous_walk(trained):
    env, result = trained
    run = simulate.run_episode(env, result["policy"], seed=5)
    frames = run["frames"]
    for previous, current in zip(frames, frames[1:]):
        assert (current["from_row"], current["from_col"]) == \
            (previous["row"], previous["col"])


def test_each_step_moves_at_most_one_cell(trained):
    """Room 2 has no teleporter, so nothing may ever jump."""
    env, result = trained
    run = simulate.run_episode(env, result["policy"], seed=6)
    for frame in run["frames"][1:]:
        distance = (abs(frame["row"] - frame["from_row"])
                    + abs(frame["col"] - frame["from_col"]))
        assert distance <= 1, "unexplained jump at step %d" % frame["step"]


def test_cumulative_reward_is_the_running_total(trained):
    env, result = trained
    run = simulate.run_episode(env, result["policy"], seed=7)
    total = 0.0
    for frame in run["frames"][1:]:
        total += frame["reward"]
        assert frame["cumulative_reward"] == pytest.approx(total)


def test_the_collapsed_list_only_ever_grows(bridge_run):
    """The replay must be able to redraw the sector at any point."""
    env, run = bridge_run
    seen = 0
    for frame in run["frames"]:
        assert len(frame["collapsed"]) >= seen
        seen = len(frame["collapsed"])
    assert seen > 0, "this run never crossed a bridge"


def test_a_collapse_is_recorded_on_the_frame_where_it_happens(bridge_run):
    env, run = bridge_run
    collapses = [frame for frame in run["frames"] if frame["bridge_collapsed"]]
    assert collapses, "no collapse recorded"
    for frame in collapses:
        assert frame["collapsed_cell"] is not None
        assert tuple(frame["collapsed_cell"]) in set(map_data.collapsing_cells())
        assert frame["collapsed_cell"] in frame["collapsed"]


def test_the_keycard_flag_never_goes_backwards(trained):
    env, result = trained
    for seed in range(10):
        run = simulate.run_episode(env, result["policy"], seed=seed)
        held = False
        for frame in run["frames"]:
            if frame["has_keycard"]:
                held = True
            assert frame["has_keycard"] or not held


def test_a_falling_run_is_recorded_as_a_fall(trained):
    """Raising exploration produces falls, and they must be recorded properly."""
    env, result = trained
    for seed in range(200):
        run = simulate.run_episode(env, result["policy"], epsilon=0.4, seed=seed)
        if run["pit_falls"]:
            last = run["frames"][-1]
            assert last["pit_fall"]
            assert last["done"]
            assert not run["success"]
            assert run["status"] == simulate.STATUS_FELL
            return
    pytest.skip("no fall occurred in 200 exploratory episodes")


def test_run_many_averages_over_episodes(trained):
    env, result = trained
    batch = simulate.run_many(env, result["policy"], episodes=10)
    assert batch["episodes"] == 10
    for key in ("success_rate", "mean_return", "std_return", "mean_steps",
                "pit_fall_rate", "keycard_rate", "span_use_rate"):
        assert key in batch


# ----------------------------------------------------------------------
# The scene the animation consumes
# ----------------------------------------------------------------------

def test_the_scene_is_valid_and_complete(trained):
    env, result = trained
    run = simulate.run_episode(env, result["policy"], seed=3)
    scene = room2_scene.build_scene(frames=run["frames"])
    assert base_renderer.validate_scene(scene) == []
    assert scene["grid"]["rows"] == 10 and scene["grid"]["cols"] == 10
    assert len(scene["frames"]) == len(run["frames"])
    assert scene["meta"]["roomNumber"] == 2


def test_the_scene_is_json_serialisable(trained):
    env, result = trained
    run = simulate.run_episode(env, result["policy"], seed=3)
    scene = room2_scene.build_scene(frames=run["frames"])
    restored = json.loads(json.dumps(scene))
    assert restored["episodeId"] == scene["episodeId"]


def test_the_engine_handles_every_room_two_tile_type():
    """Animation runs: the engine must have a branch for every tile on the map."""
    scene = room2_scene.build_scene()
    kinds = {kind for row in scene["grid"]["tiles"] for kind in row}
    expected = {map_data.tile_kind(map_data.tile_at(row, col))
                for row in range(10) for col in range(10)}
    assert kinds == expected

    engine = iso_canvas.build_html(scene)
    assert "__SCENE__" not in engine
    for kind in sorted(kinds):
        assert '"%s"' % kind in engine, "the engine has no branch for %r" % kind


def test_the_fallback_renderer_draws_the_sector(trained):
    env, result = trained
    run = simulate.run_episode(env, result["policy"], seed=3)
    scene = room2_scene.build_scene(frames=run["frames"])

    kinds = {kind for row in scene["grid"]["tiles"] for kind in row}
    assert kinds - set(fallback_renderer.KNOWN_KINDS) == set()

    for frame in (None, scene["frames"][0], scene["frames"][-1]):
        figure = fallback_renderer.render_frame(scene, frame)
        assert len(figure.axes[0].patches) >= 100
        figure.clf()


def test_the_episode_id_ignores_view_options_but_the_signature_does_not():
    first = room2_scene.build_scene(options={"camera": "iso"})
    second = room2_scene.build_scene(options={"camera": "top"})
    assert first["episodeId"] == second["episodeId"]
    assert first["signature"] != second["signature"]


def test_room_two_and_room_one_scenes_do_not_share_an_episode_id():
    """The renderer stores playback position under this key, so it must differ."""
    from renderers import iso_scene
    assert room2_scene.build_scene()["episodeId"] != \
        iso_scene.build_scene()["episodeId"]


# ----------------------------------------------------------------------
# The graphs
# ----------------------------------------------------------------------

def test_every_required_graph_is_generated(trained):
    """Graphs generated: all ten the brief asks for, plus the policy view."""
    env, result = trained
    results = [result]
    figures = {
        "episode reward": room2_plots.episode_reward(results),
        "moving average": room2_plots.moving_average_reward(results),
        "episode length": room2_plots.episode_length(results),
        "success rate": room2_plots.success_rate(results),
        "epsilon decay": room2_plots.epsilon_decay(results),
        "q convergence": room2_plots.q_value_convergence(results),
        "pit falls": room2_plots.pit_falls(results),
        "bridge usage": room2_plots.bridge_usage(results),
        "keycard rate": room2_plots.keycard_rate(results),
        "training time": room2_plots.training_time(results),
        "sector map": room2_plots.sector_map(),
        "policy": room2_plots.policy_arrows(result["policy"], has_keycard=True),
    }
    assert len(figures) == 12
    for name, figure in figures.items():
        assert figure is not None, name
        assert figure.axes, name
        figure.clf()


def test_the_graphs_survive_being_given_nothing():
    """Before training there is no history, and nothing may crash."""
    for builder in (room2_plots.episode_reward, room2_plots.moving_average_reward,
                    room2_plots.episode_length, room2_plots.success_rate,
                    room2_plots.epsilon_decay, room2_plots.q_value_convergence,
                    room2_plots.pit_falls, room2_plots.bridge_usage,
                    room2_plots.keycard_rate, room2_plots.training_time):
        figure = builder([])
        assert figure is not None
        figure.clf()
    assert room2_plots.policy_arrows({}) is not None
    assert room2_plots.experiment_chart([]) is not None


def test_the_moving_average_is_a_trailing_average():
    values = [1.0, 2.0, 3.0, 4.0]
    assert sarsa_agent.moving_average(values, 1) == values
    smoothed = sarsa_agent.moving_average(values, 2)
    assert smoothed[0] == pytest.approx(1.0)
    assert smoothed[1] == pytest.approx(1.5)
    assert smoothed[3] == pytest.approx(3.5)


# ----------------------------------------------------------------------
# Saving and loading
# ----------------------------------------------------------------------

@pytest.mark.parametrize("include_bridge_state", [False, True])
def test_saving_and_loading_restore_the_same_q_table(tmp_path, monkeypatch,
                                                     include_bridge_state):
    """Saving/loading, for both state representations."""
    monkeypatch.setattr(storage, "SAVE_DIRECTORY", str(tmp_path))
    env = Room2Env(include_bridge_state=include_bridge_state)
    result = sarsa_agent.train(env, episodes=120, seed=0)

    path = storage.save_model("unit_test", env, result)
    assert os.path.isfile(path)

    loaded = storage.load_model("unit_test")
    assert set(loaded["q"].keys()) == set(result["q"].keys())
    for state, actions_for_state in result["q"].items():
        for action, value in actions_for_state.items():
            assert loaded["q"][state][action] == pytest.approx(value)
    assert loaded["algorithm"] == result["algorithm"]
    assert loaded["include_bridge_state"] == include_bridge_state


def test_a_loaded_model_produces_the_same_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "SAVE_DIRECTORY", str(tmp_path))
    env = Room2Env()
    result = sarsa_agent.train(env, episodes=150, seed=0)
    storage.save_model("policy_test", env, result)

    loaded = storage.load_model("policy_test")
    rebuilt = sarsa_agent.greedy_policy(loaded["q"], env)
    assert rebuilt == result["policy"]


def test_the_keycard_flag_survives_the_round_trip():
    """A bool written as 1 has to come back as True, or lookups miss silently."""
    for state in [(8, 1, False), (7, 1, True), (5, 4, True, 3)]:
        key = storage._state_to_key(state)
        assert storage._key_to_state(key) == state
        assert isinstance(storage._key_to_state(key)[2], bool)


def test_loading_rejects_a_model_from_a_different_map(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "SAVE_DIRECTORY", str(tmp_path))
    env = Room2Env()
    result = sarsa_agent.train(env, episodes=40, seed=0)
    storage.save_model("stale", env, result)

    monkeypatch.setattr(map_data, "map_hash", lambda: "0000000000000000")
    with pytest.raises(storage.ModelFileError, match="different version"):
        storage.load_model("stale")


def test_loading_a_missing_model_raises_a_readable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "SAVE_DIRECTORY", str(tmp_path))
    with pytest.raises(storage.ModelFileError, match="No saved model"):
        storage.load_model("does_not_exist")


def test_loading_rejects_a_wrong_format_version(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "SAVE_DIRECTORY", str(tmp_path))
    env = Room2Env()
    result = sarsa_agent.train(env, episodes=40, seed=0)
    path = storage.save_model("versioned", env, result)

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["format_version"] = 99
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    with pytest.raises(storage.ModelFileError, match="file format"):
        storage.load_model("versioned")


def test_a_room_one_save_is_not_accepted_as_a_room_two_model(tmp_path, monkeypatch):
    """The two rooms write into the same folder, so the room must be checked."""
    monkeypatch.setattr(storage, "SAVE_DIRECTORY", str(tmp_path))
    path = storage.save_path("wrong_room")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"format_version": storage.FORMAT_VERSION, "room": 1,
                   "map_hash": map_data.map_hash(), "q": {}, "algorithm": "x"},
                  handle)
    with pytest.raises(storage.ModelFileError, match="not a Room 2 model"):
        storage.load_model("wrong_room")
