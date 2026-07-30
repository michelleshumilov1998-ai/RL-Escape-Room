"""Smoke tests for the Streamlit interface and the experiment module.

These run the real app script with Streamlit's own test harness, so a broken
widget call or a bad table shows up here instead of in the browser.  They are
slower than the rest of the suite because pressing "Solve Room" really does
solve the room.
"""

import pandas as pd
import pytest
from streamlit.dataframe_util import convert_pandas_df_to_arrow_bytes
from streamlit.testing.v1 import AppTest

from core import story
from rooms.room1 import experiments

EPISODES = 5  # keep the smoke tests quick; the maths is covered elsewhere


def _app():
    return AppTest.from_file("app.py", default_timeout=900).run()


def _press(app, label):
    for button in app.button:
        if button.label == label:
            return button.click().run()
    raise AssertionError("no button %r; found %s"
                         % (label, [button.label for button in app.button]))


def _room_one():
    app = _app()
    app.session_state["screen"] = "room"
    app.session_state["current_room"] = 1
    app.session_state["intro_seen"] = [1]
    app.session_state["r1_episodes"] = EPISODES
    return app.run()


# ----------------------------------------------------------------------
# The screens
# ----------------------------------------------------------------------

def test_the_start_screen_renders_without_error():
    app = _app()
    assert not app.exception
    assert any("PROJECT R-5" in block.value for block in app.markdown)


def test_the_room_briefing_renders_and_is_skippable():
    app = _app()
    app.session_state["screen"] = "room"
    app.session_state["current_room"] = 1
    app = app.run()
    assert not app.exception
    app = _press(app, "▸ Enter chamber")
    assert not app.exception
    assert 1 in app.session_state["intro_seen"]


def test_the_room_page_renders_before_anything_is_solved():
    app = _room_one()
    assert not app.exception
    labels = [button.label for button in app.button]
    for expected in ("▸ Solve Room", "▶ Run Agent", "↻ Reset Animation",
                     "⇄ Compare Algorithms", "↓ Save Results", "↑ Load Results"):
        assert expected in labels


def _room_two():
    app = _app()
    app.session_state["screen"] = "room"
    app.session_state["current_room"] = 2
    app.session_state["intro_seen"] = [1, 2]
    app.session_state["solved_rooms"] = [1]
    app.session_state["r2_episodes"] = 250
    app.session_state["r2_eval"] = 5
    return app.run()


def test_the_room_two_page_renders_before_training():
    app = _room_two()
    assert not app.exception
    labels = [button.label for button in app.button]
    for expected in ("▸ Train", "▶ Run Agent", "▶ Play Replay", "‖ Pause",
                     "↻ Reset Animation", "↓ Save Model", "↑ Load Model"):
        assert expected in labels


@pytest.mark.slow
def test_training_and_running_room_two_from_the_interface():
    app = _room_two()

    app = _press(app, "▸ Train")
    assert not app.exception
    result = app.session_state["rooms"][2]["primary"]
    assert result["algorithm"] == "SARSA"
    assert result["history"]["reward"]

    app = _press(app, "▶ Run Agent")
    assert not app.exception
    run = app.session_state["rooms"][2]["run"]
    assert run["keycard_collected"], "the exit cannot open without the keycard"
    if run["success"]:
        assert 2 in app.session_state["solved_rooms"]
        assert "Security Keycard" in app.session_state["inventory"]


@pytest.mark.slow
def test_room_two_save_and_load_round_trip_through_the_interface():
    app = _room_two()
    app = _press(app, "▸ Train")
    app = _press(app, "↓ Save Model")
    assert not app.exception
    app = _press(app, "↑ Load Model")
    assert not app.exception
    assert app.session_state["rooms"][2]["primary"]["policy"]


@pytest.mark.slow
def test_room_two_analysis_view_and_compare_both_render():
    app = _room_two()
    app = _press(app, "▸ Train")

    app.session_state["r2_view"] = "Analysis"
    app = app.run()
    assert not app.exception

    app.session_state["r2_view"] = "Gameplay"
    app.session_state["r2_algorithm"] = "Compare both"
    app = app.run()
    app = _press(app, "▸ Train")
    assert not app.exception
    algorithms = [result["algorithm"]
                  for result in app.session_state["rooms"][2]["results"]]
    assert algorithms == ["SARSA", "Q-Learning"]


@pytest.mark.slow
def test_changing_a_room_two_slider_does_not_retrain():
    app = _room_two()
    app = _press(app, "▸ Train")
    first = app.session_state["rooms"][2]["primary"]

    app.session_state["r2_alpha"] = 0.5
    app = app.run()
    assert not app.exception
    assert app.session_state["rooms"][2]["primary"] is first
    assert any("Train" in block.value for block in app.info)


def _room_three():
    app = _app()
    app.session_state["screen"] = "room"
    app.session_state["current_room"] = 3
    app.session_state["intro_seen"] = [1, 2, 3]
    app.session_state["solved_rooms"] = [1, 2]
    app.session_state["r3_episodes"] = 200
    app.session_state["r3_eval"] = 3
    return app.run()


def test_the_room_three_page_renders_before_training():
    app = _room_three()
    assert not app.exception
    labels = [button.label for button in app.button]
    for expected in ("▸ Train", "◆ Evaluate", "▶ Play Replay", "‖ Pause",
                     "↻ Reset Animation", "↓ Save Model", "↑ Load Model"):
        assert expected in labels


@pytest.mark.slow
def test_training_and_evaluating_room_three_from_the_interface():
    app = _room_three()

    app = _press(app, "▸ Train")
    assert not app.exception
    result = app.session_state["rooms"][3]["primary"]
    assert result["algorithm"] == "Q-Learning"
    assert result["history"]["reward"]

    app = _press(app, "◆ Evaluate")
    assert not app.exception
    run = app.session_state["rooms"][3]["run"]
    assert run["frames"]
    if run["success"]:
        assert 3 in app.session_state["solved_rooms"]


@pytest.mark.slow
def test_room_three_analysis_view_renders():
    app = _room_three()
    app = _press(app, "▸ Train")
    app.session_state["r3_view"] = "Analysis"
    app = app.run()
    assert not app.exception


def _room_four():
    app = _app()
    app.session_state["screen"] = "room"
    app.session_state["current_room"] = 4
    app.session_state["intro_seen"] = [1, 2, 3, 4]
    app.session_state["solved_rooms"] = [1, 2, 3]
    app.session_state["r4_episodes"] = 60
    app.session_state["r4_eval"] = 3
    return app.run()


def test_the_room_four_page_renders_before_training():
    app = _room_four()
    assert not app.exception
    labels = [button.label for button in app.button]
    for expected in ("▸ Train", "▶ Run Drone", "▶ Play Replay", "‖ Pause",
                     "↻ Reset Animation", "↓ Save Model", "↑ Load Model"):
        assert expected in labels


@pytest.mark.slow
def test_training_and_flying_room_four_from_the_interface():
    app = _room_four()

    app = _press(app, "▸ Train")
    assert not app.exception
    result = app.session_state["rooms"][4]["primary"]
    assert result["history"]["reward"]
    assert result["agent"].is_finite()

    app = _press(app, "▶ Run Drone")
    assert not app.exception
    run = app.session_state["rooms"][4]["run"]
    assert run["frames"]
    if run["success"]:
        assert 4 in app.session_state["solved_rooms"]


def _room_five():
    app = _app()
    app.session_state["screen"] = "room"
    app.session_state["current_room"] = 5
    app.session_state["intro_seen"] = [1, 2, 3, 4, 5]
    app.session_state["solved_rooms"] = [1, 2, 3, 4]
    app.session_state["r5_episodes"] = 60
    app.session_state["r5_train_layouts"] = 3
    app.session_state["r5_val_layouts"] = 2
    app.session_state["r5_test_layouts"] = 2
    app.session_state["r5_val_every"] = 0
    return app.run()


def test_the_room_five_page_renders_before_training():
    app = _room_five()
    assert not app.exception
    labels = [button.label for button in app.button]
    for expected in ("▸ Train", "◆ Evaluate", "▶ Run Agent", "‖ Pause",
                     "⟳ Generate Layout", "↓ Save Model", "↑ Load Model"):
        assert expected in labels


@pytest.mark.slow
def test_training_and_running_room_five_from_the_interface():
    app = _room_five()

    app = _press(app, "▸ Train")
    assert not app.exception
    result = app.session_state["rooms"][5]["primary"]
    assert result["history"]["reward"]
    assert result["agent"].is_finite()
    # The layouts trained on and the ones tested on must be different warehouses.
    assert not set(result["training_seeds"]) & set(result["validation_seeds"])

    app = _press(app, "▶ Run Agent")
    assert not app.exception
    run = app.session_state["rooms"][5]["run"]
    assert run["frames"]
    if run["success"]:
        assert 5 in app.session_state["solved_rooms"]


@pytest.mark.slow
def test_room_five_evaluates_on_unseen_layouts_from_the_interface():
    app = _room_five()
    app = _press(app, "▸ Train")
    app = _press(app, "◆ Evaluate")
    assert not app.exception
    splits = app.session_state["rooms"][5]["splits"]
    assert [outcome["split"] for outcome in splits] == ["training", "validation",
                                                        "test"]
    assert sum(outcome["seen_during_training"] for outcome in splits) == 1


def test_the_placeholder_page_is_no_longer_reachable():
    """Every chamber is built, so nothing routes to rooms/locked_page.py."""
    assert all(story.room(number)["implemented"]
               for number in story.ROOM_NUMBERS)


def test_jumping_to_a_room_whose_predecessor_is_unsolved_still_renders():
    """The rail marks such a room LOCKED, but a direct jump must not crash."""
    app = _app()
    app.session_state["screen"] = "room"
    app.session_state["current_room"] = 3
    app.session_state["intro_seen"] = [3]
    app.session_state["solved_rooms"] = []
    app = app.run()
    assert not app.exception


def test_the_room_cleared_screen_renders_without_error():
    app = _app()
    app.session_state["screen"] = "transition"
    app.session_state["transition_room"] = 1
    app = app.run()
    assert not app.exception


def test_every_room_cleared_screen_renders():
    for number in (1, 2, 3, 4, 5):
        app = _app()
        app.session_state["screen"] = "transition"
        app.session_state["transition_room"] = number
        app.session_state["solved_rooms"] = list(range(1, number + 1))
        app = app.run()
        assert not app.exception, "room %d transition raised" % number


def test_the_completion_screen_renders_once_every_room_is_solved():
    app = _app()
    app.session_state["screen"] = "complete"
    app.session_state["solved_rooms"] = [1, 2, 3, 4, 5]
    app.session_state["inventory"] = ["Navigation Chart", "Security Keycard",
                                      "Reactor Key", "Flight Controller",
                                      "Master Override"]
    app = app.run()
    assert not app.exception
    assert any("COMPLETE" in block.value for block in app.markdown)
    for table in app.dataframe:
        convert_pandas_df_to_arrow_bytes(table.value)


# ----------------------------------------------------------------------
# The buttons that do real work
# ----------------------------------------------------------------------

@pytest.mark.slow
def test_solving_and_running_from_the_interface():
    app = _room_one()

    app = _press(app, "▸ Solve Room")
    assert not app.exception
    solution = app.session_state["rooms"][1]["solution"]
    assert solution["converged"]

    app = _press(app, "▶ Run Agent")
    assert not app.exception
    run = app.session_state["rooms"][1]["run"]
    assert run["success"]
    # Escaping Room 1 collects its component and unlocks the next chamber.
    assert 1 in app.session_state["solved_rooms"]
    assert "Navigation Chart" in app.session_state["inventory"]


@pytest.mark.slow
def test_pausing_does_not_discard_the_recording():
    app = _room_one()
    app = _press(app, "▸ Solve Room")
    app = _press(app, "▶ Run Agent")
    frames_before = len(app.session_state["rooms"][1]["run"]["frames"])

    app = _press(app, "‖ Pause")
    assert not app.exception
    assert app.session_state["rooms"][1]["playing"] is False
    assert len(app.session_state["rooms"][1]["run"]["frames"]) == frames_before


@pytest.mark.slow
def test_changing_a_slider_does_not_re_solve_the_room():
    """Nothing may be recomputed except when a button is pressed."""
    app = _room_one()
    app = _press(app, "▸ Solve Room")
    first = app.session_state["rooms"][1]["solution"]

    app.session_state["r1_gamma"] = 0.80
    app = app.run()
    assert not app.exception
    # Same object, so no new solve happened; the page says it is stale instead.
    assert app.session_state["rooms"][1]["solution"] is first
    assert any("Solve Room" in block.value for block in app.info)


@pytest.mark.slow
def test_every_table_in_the_interface_can_be_displayed():
    """Streamlit converts tables through Arrow, which rejects mixed columns."""
    app = _room_one()
    app = _press(app, "▸ Solve Room")
    app = _press(app, "⇄ Compare Algorithms")
    app = _press(app, "≣ Run slipping sweep")
    assert not app.exception
    assert len(app.dataframe) >= 2
    for table in app.dataframe:
        convert_pandas_df_to_arrow_bytes(table.value)


@pytest.mark.slow
def test_the_analysis_view_renders():
    app = _room_one()
    app = _press(app, "▸ Solve Room")
    app.session_state["r1_view_mode"] = "Analysis"
    app = app.run()
    assert not app.exception


# ----------------------------------------------------------------------
# The experiments
# ----------------------------------------------------------------------

def test_the_slipping_experiment_reports_a_row_per_risk_level():
    rows = experiments.slipping_experiment(episodes=EPISODES)
    assert len(rows) == len(experiments.RISK_LEVELS)
    for row in rows:
        for key in ("success_rate", "mean_return", "std_return", "mean_steps",
                    "mean_laser_hits", "route", "start_state_value"):
            assert key in row


def test_the_slipping_experiment_shows_the_route_changing():
    """The result Room 1 exists to demonstrate."""
    rows = experiments.slipping_experiment(episodes=EPISODES)
    routes = {row["value"]: row["route"] for row in rows}
    assert "Frozen shaft" in routes["Low risk"]
    assert "Frozen shaft" not in routes["High risk"]


def test_the_gamma_experiment_covers_every_requested_value():
    rows = experiments.gamma_experiment(episodes=EPISODES)
    assert [row["value"] for row in rows] == experiments.GAMMA_VALUES


def test_every_sweepable_parameter_actually_runs():
    for parameter, values in experiments.SWEEPABLE_PARAMETERS.items():
        rows = experiments.parameter_experiment(parameter, values[:2],
                                                episodes=2)
        assert len(rows) == 2
        assert all(row["parameter"] == parameter for row in rows)


def test_experiment_rows_can_be_put_into_a_table():
    rows = experiments.gamma_experiment(episodes=2)
    table = pd.DataFrame([{"Value": row["value"],
                           "Mean return": "%.1f" % row["mean_return"],
                           "Route": row["route"]} for row in rows])
    convert_pandas_df_to_arrow_bytes(table)


def test_best_performing_row_picks_the_highest_metric():
    rows = [{"mean_return": 1.0, "value": "a"}, {"mean_return": 9.0, "value": "b"},
            {"mean_return": 3.0, "value": "c"}]
    assert experiments.best_performing_row(rows)["value"] == "b"
    assert experiments.best_performing_row([]) is None
