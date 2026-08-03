"""The numbers the training dashboard shows, and what they mean.

These are the claims a viva can be asked about, so they are asserted rather
than trusted:

    room 5's time limit          300 decisions x 10 ticks x 0.02 s = 60 s
    episode reward               the SUM of every step reward, not the last
    convergence                  mean(|TD error|), not a total and not a loss
    Reset                        clears the history the graphs are drawn from
    evaluation and replay        never appear in the training history
    graph data                   always finite
"""

import math

import pytest

from game import config, rooms
from game.session import Session


# ----------------------------------------------------------------------
# Room 5's decision clock
# ----------------------------------------------------------------------

def test_room_five_decision_lasts_two_tenths_of_a_second():
    room = rooms.room(5)
    assert room["action_repeat"] == 10
    assert room["dt"] == pytest.approx(0.02)
    assert room["action_repeat"] * room["dt"] == pytest.approx(0.2)


def test_room_five_episode_is_sixty_simulated_seconds():
    room = rooms.room(5)
    decisions = room["max_steps"]
    assert decisions == 300
    seconds = decisions * room["action_repeat"] * room["dt"]
    assert seconds == pytest.approx(60.0)


def test_the_max_steps_control_states_the_right_arithmetic():
    """The explanation the player reads must match the code."""
    explanation = config.PARAMETERS["max_steps"]["explanation"]
    assert "0.2 seconds" in explanation
    assert "60" in explanation
    assert "10 physics ticks" in explanation
    # The figures that were wrong must be gone.
    assert "0.1 s" not in explanation
    assert "30 seconds" not in explanation


def test_room_four_records_physics_ticks_not_decisions():
    """Room 4 has no action repeat, so a step is one tick of 0.02 s."""
    room = rooms.room(4)
    assert room.get("action_repeat") in (None, 1)
    assert room["dt"] == pytest.approx(0.02)
    assert room["max_steps"] == 1800
    assert room["max_steps"] * room["dt"] == pytest.approx(36.0)


# ----------------------------------------------------------------------
# Room 2's collapse probability
# ----------------------------------------------------------------------

def test_the_collapse_explanation_describes_one_draw_per_crossing():
    explanation = config.PARAMETERS["collapse_chance"]["explanation"]
    assert "once" in explanation
    assert "crossing" in explanation
    # The claims that were wrong.
    assert "Two sections" not in explanation
    assert "two sections" not in explanation


def test_the_collapse_probability_is_the_crossing_probability():
    """Measured, not assumed: the slider value IS the failure rate.

    Four planks. If the draw were made per plank the rate would be
    1-(1-p)^4, which at p=0.25 is 0.68 rather than 0.25.
    """
    from game.grid import GridWorld, RIGHT, UP

    for chance in (0.10, 0.25):
        env = GridWorld(rooms.room(2), slip=0.0, collapse=chance)
        fell = 0
        attempts = 600
        for seed in range(attempts):
            env.reset(seed=seed)
            env.step(UP)
            for _ in range(7):
                _state, _reward, done, info = env.step(RIGHT)
                if done:
                    fell += 1 if info["hazard"] else 0
                    break
        rate = fell / attempts
        compounded = 1 - (1 - chance) ** 4
        assert abs(rate - chance) < 0.06, (
            "crossing failure %.3f is not the slider value %.2f"
            % (rate, chance))
        assert abs(rate - compounded) > 0.08


# ----------------------------------------------------------------------
# What the graphs are drawn from
# ----------------------------------------------------------------------

def trained(room, episodes=60):
    session = Session(room, parameters={"episodes": episodes})
    session.play()
    while session.state == "TRAINING":
        session.advance(budget_ms=60.0)
    return session


def test_episode_reward_is_the_sum_of_every_step_reward():
    """Not the final reward, and not a mean."""
    session = trained(2, episodes=40)
    batch = session.batch()
    rows = {row["episode"]: row for row in batch["history"]}
    checked = 0
    for episode in batch["episodes"]:
        row = rows.get(episode["index"])
        if row is None:
            continue
        # `steps[0]` is the opening frame and carries no reward.
        total = sum(step["reward"] for step in episode["steps"])
        assert row["reward"] == pytest.approx(total, abs=1e-6), (
            "episode %d: history says %r, the frames sum to %r"
            % (episode["index"], row["reward"], total))
        checked += 1
    assert checked, "no episode could be cross-checked"


def test_convergence_is_the_mean_absolute_td_error():
    """A mean, so it cannot grow simply because an episode was long."""
    session = trained(2, episodes=60)
    rows = session.batch()["history"]
    assert rows
    for row in rows:
        assert row["convergence"] >= 0.0, "a mean of absolute values is >= 0"
        assert math.isfinite(row["convergence"])

    # A total would scale with episode length; a mean must not. The longest
    # episode is not systematically the one with the largest value.
    longest = max(rows, key=lambda row: row["steps"])
    biggest = max(rows, key=lambda row: row["convergence"])
    assert not (longest is biggest and longest["steps"] > 3 * min(
        row["steps"] for row in rows)), (
        "convergence looks like a total rather than a mean")


def test_every_graph_number_is_finite():
    session = trained(2, episodes=60)
    numeric = ("reward", "steps", "epsilon", "convergence", "success",
               "collision", "timeout", "boundary", "weightNorm")
    for row in session.batch()["history"]:
        for key in numeric:
            value = row.get(key)
            if value is None:
                continue
            assert isinstance(value, (int, float))
            assert math.isfinite(value), "%s is %r" % (key, value)


def test_reset_clears_the_training_history():
    session = trained(2, episodes=40)
    assert session.batch()["history"], "nothing was recorded"
    session.reset()
    assert session.batch()["history"] == [], (
        "the graphs would still be drawn from the previous run")
    assert session.batch()["episodes"] == []


def test_evaluation_does_not_enter_the_training_history():
    """A frozen-weight measurement is not an episode that was learned from."""
    session = trained(5, episodes=40)
    before = len(session.batch()["history"])
    session.evaluate("validation")
    after = len(session.batch()["history"])
    assert after == before, (
        "evaluation added %d rows to the training history" % (after - before))


def test_the_greedy_replay_does_not_enter_the_training_history():
    session = trained(2, episodes=40)
    before = len(session.batch()["history"])
    session.start_replay()
    assert len(session.batch()["history"]) == before


# ----------------------------------------------------------------------
# The dashboard's own arithmetic
# ----------------------------------------------------------------------

def test_best_return_comes_from_the_whole_history_not_the_sample():
    """The replay sample is a few dozen rows; the history is every episode."""
    session = trained(2, episodes=200)
    batch = session.batch()
    history = batch["history"]
    sample = batch["episodes"]
    assert len(history) > len(sample), (
        "this test needs more episodes than the recorder keeps")
    best_overall = max(row["reward"] for row in history)
    best_sampled = max(episode["totalReward"] for episode in sample)
    # The sample can never beat the full history, and usually is worse.
    assert best_overall >= best_sampled - 1e-9


def test_the_declared_charts_only_name_recorded_fields():
    """A chart naming a field nothing writes would draw an empty panel."""
    for number in rooms.ROOM_NUMBERS:
        room = rooms.room(number)
        charts = room.get("charts") or []
        if not charts:
            continue
        session = trained(number, episodes=30) if number != 1 else None
        if number == 1:
            # The planner's curve is not per-episode; it has its own source.
            assert all(chart.get("source") == "sweeps" for chart in charts)
            continue
        rows = session.batch()["history"]
        assert rows
        for chart in charts:
            if chart.get("source"):
                continue          # checkpoints, not per-episode history
            for key in (chart.get("keys") or [chart["key"]]):
                assert key in rows[0], (
                    "room %d graph %r reads %r, which no episode row has"
                    % (number, chart["label"], key))
