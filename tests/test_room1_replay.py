"""Tests for replay, rendering and saved files — required checks 25, 26 and 27.

Numbered comments refer to the checklist in the project brief.
"""

import os

import pytest

from core import actions, tiles
from renderers import base_renderer, fallback_renderer, iso_canvas, iso_scene
from rooms.room1 import dp_solver, map_data, simulate, storage
from rooms.room1.environment import Room1Env

REQUIRED_FRAME_KEYS = ("state", "row", "col", "has_battery", "previous_direction",
                       "action", "actual_direction", "reward", "cumulative_reward",
                       "event", "laser_hit", "slipped", "used_teleport",
                       "battery_collected", "done")


@pytest.fixture(scope="module")
def env():
    return Room1Env()


@pytest.fixture(scope="module")
def policy(env):
    return dp_solver.value_iteration(env, gamma=0.95, theta=1e-6,
                                     max_sweeps=2000)["policy"]


@pytest.fixture(scope="module")
def run(env, policy):
    return simulate.run_episode(env, policy, seed=7)


# ----------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------

def test_every_frame_carries_the_required_fields(run):
    for frame in run["frames"]:
        for key in REQUIRED_FRAME_KEYS:
            assert key in frame, "frame %d is missing %r" % (frame["step"], key)


def test_replay_reproduces_the_original_path_exactly(env, policy):
    """25. Replay preserves and reproduces the original path exactly.

    Running the same seed again must produce an identical recording, and playing
    the stored frames back must never resample anything.
    """
    first = simulate.run_episode(env, policy, seed=99)
    second = simulate.run_episode(env, policy, seed=99)

    assert len(first["frames"]) == len(second["frames"])
    for left, right in zip(first["frames"], second["frames"]):
        assert left == right

    for key in ("success", "steps", "total_reward", "laser_hits",
                "wall_collisions", "teleports_used", "battery_collected"):
        assert first[key] == second[key]


def test_a_different_seed_can_give_a_different_recording(env, policy):
    """If every seed gave the same episode, the room would not be stochastic."""
    paths = set()
    for seed in range(40):
        run = simulate.run_episode(env, policy, seed=seed)
        paths.add(tuple((frame["row"], frame["col"]) for frame in run["frames"]))
    assert len(paths) > 1


def test_frames_join_up_into_a_continuous_walk(run):
    """Each frame must start where the previous one ended."""
    frames = run["frames"]
    for previous, current in zip(frames, frames[1:]):
        assert (current["from_row"], current["from_col"]) == \
            (previous["row"], previous["col"])


def test_cumulative_reward_is_the_running_total(run):
    total = 0.0
    for frame in run["frames"][1:]:
        total += frame["reward"]
        assert frame["cumulative_reward"] == pytest.approx(total)


def test_each_step_moves_at_most_one_cell_unless_teleported_or_reset(run):
    """A jump of more than one cell has to be explained by the recording."""
    for frame in run["frames"][1:]:
        distance = (abs(frame["row"] - frame["from_row"])
                    + abs(frame["col"] - frame["from_col"]))
        if distance > 1:
            assert frame["used_teleport"] or frame["laser_hit"], \
                "unexplained jump at step %d" % frame["step"]


def test_a_laser_hit_in_a_recording_lands_on_the_start_platform(env, policy):
    for seed in range(200):
        run = simulate.run_episode(env, policy, seed=seed)
        hits = [frame for frame in run["frames"] if frame["laser_hit"]]
        if hits:
            for frame in hits:
                assert (frame["row"], frame["col"]) == map_data.start_cell()
                assert frame["reward"] == env.step_cost + env.laser_penalty
            return
    pytest.skip("no laser hit occurred in 200 episodes with this policy")


def test_the_battery_flag_never_goes_backwards(env, policy):
    for seed in range(30):
        run = simulate.run_episode(env, policy, seed=seed)
        held = False
        for frame in run["frames"]:
            if frame["has_battery"]:
                held = True
            assert frame["has_battery"] or not held, \
                "the battery was lost again at step %d" % frame["step"]


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def test_the_scene_is_valid_and_complete(run):
    scene = iso_scene.build_scene(frames=run["frames"])
    assert base_renderer.validate_scene(scene) == []
    assert scene["grid"]["rows"] == 10
    assert scene["grid"]["cols"] == 10
    assert len(scene["frames"]) == len(run["frames"])


def test_the_scene_is_json_serialisable(run):
    """The scene crosses into JavaScript, so it has to survive JSON."""
    import json
    scene = iso_scene.build_scene(frames=run["frames"])
    restored = json.loads(json.dumps(scene))
    assert restored["episodeId"] == scene["episodeId"]


def test_the_renderer_can_draw_every_tile_type():
    """26. The renderer can draw every tile type.

    Checked two ways: the scene names a kind for every tile on the map, and the
    JavaScript engine has a branch for each of those kinds.
    """
    scene = iso_scene.build_scene()
    kinds_on_the_map = {kind for row in scene["grid"]["tiles"] for kind in row}
    expected = {tiles.tile_kind(tile) for tile in
                {map_data.tile_at(row, col)
                 for row in range(10) for col in range(10)}}
    assert kinds_on_the_map == expected

    engine = iso_canvas.build_html(scene)
    for kind in sorted(kinds_on_the_map):
        assert '"%s"' % kind in engine, "the engine has no branch for %r" % kind


def test_the_fallback_renderer_draws_every_tile_type_in_python(run):
    """The Matplotlib fallback must handle the whole tile alphabet too."""
    scene = iso_scene.build_scene(frames=run["frames"])

    kinds = {kind for row in scene["grid"]["tiles"] for kind in row}
    assert kinds - set(fallback_renderer.KNOWN_KINDS) == set(), \
        "the fallback has no colour for: %s" % (kinds
                                               - set(fallback_renderer.KNOWN_KINDS))

    for frame in (None, scene["frames"][0], scene["frames"][-1]):
        figure = fallback_renderer.render_frame(scene, frame)
        assert figure is not None
        # One patch per tile face at the very least.
        assert len(figure.axes[0].patches) >= 100
        figure.clf()


def test_the_built_html_contains_the_scene_and_no_placeholder(run):
    scene = iso_scene.build_scene(frames=run["frames"])
    html = iso_canvas.build_html(scene)
    assert "__SCENE__" not in html
    assert scene["episodeId"] in html
    # A literal closing script tag inside the data would break the page.
    payload = html.split("const SCENE = ", 1)[1].split(";\n", 1)[0]
    assert "</script>" not in payload


def test_the_episode_id_ignores_view_options_but_the_signature_does_not(run):
    first = iso_scene.build_scene(frames=run["frames"], options={"camera": "iso"})
    second = iso_scene.build_scene(frames=run["frames"], options={"camera": "top"})
    assert first["episodeId"] == second["episodeId"]
    assert first["signature"] != second["signature"]


# ----------------------------------------------------------------------
# Saving and loading
# ----------------------------------------------------------------------

def test_saving_and_loading_restore_the_same_policy(env, tmp_path, monkeypatch):
    """27. Saving and loading restore the same policy."""
    monkeypatch.setattr(storage, "SAVE_DIRECTORY", str(tmp_path))
    solution = dp_solver.value_iteration(env, gamma=0.95, theta=1e-6,
                                        max_sweeps=2000)

    path = storage.save_solution("unit_test", env, solution)
    assert os.path.isfile(path)

    loaded = storage.load_solution("unit_test")
    assert loaded["policy"] == solution["policy"]
    assert loaded["algorithm"] == solution["algorithm"]
    assert loaded["iterations"] == solution["iterations"]
    for state, value in solution["values"].items():
        assert loaded["values"][state] == pytest.approx(value)


def test_loading_rejects_a_file_from_a_different_map(env, tmp_path, monkeypatch):
    """A stale policy must never be shown on top of an edited map."""
    monkeypatch.setattr(storage, "SAVE_DIRECTORY", str(tmp_path))
    solution = dp_solver.value_iteration(env, gamma=0.95, theta=1e-3,
                                        max_sweeps=50)
    storage.save_solution("stale", env, solution)

    monkeypatch.setattr(map_data, "map_hash", lambda: "0000000000000000")
    with pytest.raises(storage.SolutionFileError, match="different version"):
        storage.load_solution("stale")


def test_loading_a_missing_file_raises_a_readable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "SAVE_DIRECTORY", str(tmp_path))
    with pytest.raises(storage.SolutionFileError, match="No saved result"):
        storage.load_solution("does_not_exist")


def test_loading_rejects_a_wrong_format_version(env, tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(storage, "SAVE_DIRECTORY", str(tmp_path))
    solution = dp_solver.value_iteration(env, gamma=0.95, theta=1e-3,
                                        max_sweeps=50)
    path = storage.save_solution("versioned", env, solution)

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["format_version"] = 99
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    with pytest.raises(storage.SolutionFileError, match="file format"):
        storage.load_solution("versioned")


def test_state_keys_survive_the_round_trip():
    """The state is a tuple, and JSON has no tuples."""
    for state in [(0, 0, False, actions.NONE), (9, 9, True, actions.RIGHT)]:
        key = storage._state_to_key(state)
        assert storage._key_to_state(key) == state
