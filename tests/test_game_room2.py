"""Room 2's bridge collapse probability: the control, and what it drives.

The chamber is a risk-versus-return decision and the risk is a slider, so the
slider is the room. These tests hold the whole path from the control the player
sees down to the transition the environment samples:

    config.PARAMETERS['collapse_chance']   what the interface is told to draw
        -> Session.parameters                what the run is set to
            -> GridWorld.collapse            what the model actually uses
                -> transitions()             what a step really does

Nothing here tests SARSA. The algorithm, the grid, the bridge geometry and the
reward table are all out of scope and are asserted to be untouched.
"""

import pytest

from game import algorithms, config, rooms
from game.grid import COLLAPSING, TILE_KINDS, GridWorld
from game.session import Session

ROOM = 2
FIELD = "collapse_chance"


def world(chance):
    """A room 2 world at one collapse probability."""
    return GridWorld(rooms.room(ROOM), slip=0.0, collapse=chance)


def planks(env):
    """Every collapsing section, as (row, col)."""
    return [(row, col)
            for row in range(env.rows)
            for col in range(env.cols)
            if env.grid[row][col] == COLLAPSING]


# ----------------------------------------------------------------------
# 1-4. The control the player is given
# ----------------------------------------------------------------------

def test_room_two_exposes_a_bridge_collapse_probability():
    assert FIELD in rooms.room(ROOM)["parameters"]
    assert FIELD in config.PARAMETERS
    assert "collapse" in config.PARAMETERS[FIELD]["label"].lower()


def test_there_is_exactly_one_collapse_parameter():
    """A second one would be a control that changes nothing."""
    named = [name for name in config.PARAMETERS
             if "collapse" in name or "bridge" in name]
    assert named == [FIELD]


def test_the_default_is_valid_and_on_a_step():
    spec = config.PARAMETERS[FIELD]
    assert spec["minimum"] <= spec["default"] <= spec["maximum"]
    # A default off the step grid cannot be returned to once it is moved.
    steps = (spec["default"] - spec["minimum"]) / spec["step"]
    assert abs(steps - round(steps)) < 1e-9


def test_the_range_is_a_probability():
    spec = config.PARAMETERS[FIELD]
    assert spec["minimum"] == 0.0
    assert spec["maximum"] == 1.0
    assert spec["step"] == pytest.approx(0.05)


def test_changing_it_requires_a_reset():
    """It changes the world, so a policy learned under the old one is void."""
    assert config.PARAMETERS[FIELD]["scope"] == "reset"


# ----------------------------------------------------------------------
# 5, 10. The control reaches the backend, and is reported back
# ----------------------------------------------------------------------

@pytest.mark.parametrize("chance", [0.0, 0.15, 0.5, 0.85, 1.0])
def test_the_control_reaches_the_environment(chance):
    session = Session(ROOM, parameters={FIELD: chance})
    assert session.parameters[FIELD] == pytest.approx(chance)
    assert session.env.collapse == pytest.approx(chance)


def test_the_value_appears_in_the_parameter_schema_the_screen_builds():
    session = Session(ROOM, parameters={FIELD: 0.45})
    schema = {entry["key"]: entry
              for entry in session.describe()["definition"]["parameterSchema"]}
    assert FIELD in schema
    assert schema[FIELD]["min"] == 0.0
    assert schema[FIELD]["max"] == 1.0
    assert schema[FIELD]["step"] == pytest.approx(0.05)
    # The schema carries the value this run actually started with.
    assert schema[FIELD]["default"] == pytest.approx(0.45)


def test_the_effective_value_appears_in_the_live_status():
    session = Session(ROOM, parameters={FIELD: 0.35})
    rows = dict(session.readout())
    assert "Bridge collapse probability" in rows
    assert rows["Bridge collapse probability"] == "0.35"


def test_the_status_reports_the_world_not_the_pending_slider():
    """A reset-scope value that has been moved but not applied is not in force.

    Reporting the stored value during that window would label a run with a
    probability it was never trained at.
    """
    session = Session(ROOM, parameters={FIELD: 0.20})
    session.play()
    session.advance(budget_ms=20.0)
    session.set_parameters({FIELD: 0.90})          # stored, not yet applied
    assert session.parameters[FIELD] == pytest.approx(0.90)
    assert session.env.collapse == pytest.approx(0.20)
    assert dict(session.readout())["Bridge collapse probability"] == "0.20"
    assert session.stale


def test_changing_it_marks_the_run_stale_with_an_environment_message():
    session = Session(ROOM, parameters={FIELD: 0.10})
    session.play()
    session.advance(budget_ms=20.0)
    session.set_parameters({FIELD: 0.60})
    assert session.stale
    assert session.stale_reason


def test_a_reset_applies_it_and_clears_the_staleness():
    session = Session(ROOM, parameters={FIELD: 0.10})
    session.play()
    session.advance(budget_ms=20.0)
    session.set_parameters({FIELD: 0.60})
    session.reset()
    assert not session.stale
    assert session.env.collapse == pytest.approx(0.60)


def test_a_value_outside_the_range_is_clamped():
    session = Session(ROOM)
    session.set_parameters({FIELD: 4.0})
    assert session.parameters[FIELD] == 1.0
    session.set_parameters({FIELD: -2.0})
    assert session.parameters[FIELD] == 0.0


# ----------------------------------------------------------------------
# 6-8. What the probability actually does to a transition
# ----------------------------------------------------------------------

def crossings(env):
    """Every (state, action) whose step lands on a sound bridge section.

    Terminal states are skipped: `transitions` short-circuits them to a single
    self-loop before the collapse rule is ever consulted, so they say nothing
    about the probability.
    """
    for state in env.all_states():
        if env.is_terminal(state):
            continue
        for action in env.actions():
            if env.lands_on_sound_plank(state, action):
                yield state, action


def test_zero_never_collapses_a_section():
    """At 0.00 a crossing is certain: one outcome, probability 1."""
    env = world(0.0)
    assert planks(env), "room 2 has no collapsing sections"
    checked = 0
    for state, action in crossings(env):
        checked += 1
        outcomes = env.transitions(state, action)
        assert len(outcomes) == 1
        probability, _next, _reward, _done = outcomes[0]
        assert probability == pytest.approx(1.0)
    assert checked, "no step in the room lands on a sound section"


def test_one_always_collapses_a_section():
    """At 1.00 every step onto a sound section is the failure, with p = 1."""
    env = world(1.0)
    checked = 0
    for state, action in crossings(env):
        checked += 1
        outcomes = env.transitions(state, action)
        # The model keeps the surviving branch in the list at probability
        # zero rather than pruning it, so this is about the *weights* rather
        # than the length: all of the mass is on the failure.
        failed = [entry for entry in outcomes
                  if (entry[1], entry[2], entry[3])
                  == env.resolve(state, action, True)]
        assert len(failed) == 1
        assert failed[0][0] == pytest.approx(1.0)
        assert failed[0][3] is True, "a failed crossing must end the run"
        others = [entry[0] for entry in outcomes if entry not in failed]
        assert all(weight == pytest.approx(0.0) for weight in others)
    assert checked, "no step in the room lands on a sound section"


@pytest.mark.parametrize("chance", [0.05, 0.3, 0.65, 0.95])
def test_intermediate_values_are_read_as_probabilities(chance):
    """The slider's number is the probability of the failure branch."""
    env = world(chance)
    checked = 0
    for state, action in crossings(env):
        checked += 1
        outcomes = env.transitions(state, action)
        weights = sorted(round(entry[0], 10) for entry in outcomes)
        assert weights == sorted([round(chance, 10), round(1 - chance, 10)])
        assert sum(entry[0] for entry in outcomes) == pytest.approx(1.0)

        # The heavier branch is the one the probability names.
        failed = [entry for entry in outcomes
                  if (entry[1], entry[2], entry[3])
                  == env.resolve(state, action, True)]
        assert len(failed) == 1
        assert failed[0][0] == pytest.approx(chance)
    assert checked


def test_the_probability_is_seeded_and_reproducible():
    """Two runs at the same setting and seed sample identically."""
    def walk(seed):
        env = world(0.4)
        state = env.reset(seed=seed)
        seen = []
        for _ in range(40):
            state, reward, done, info = env.step(3)      # RIGHT, onto the span
            seen.append((state, round(reward, 6), done, info["hazard"]))
            if done:
                break
        return seen

    assert walk(11) == walk(11)
    # And a different seed is a different sample, or it is not stochastic.
    assert any(walk(seed) != walk(11) for seed in range(12, 30))


def test_the_probability_governs_how_often_a_crossing_actually_fails():
    """End to end through `step`, not only through the model."""
    from game.grid import RIGHT
    results = {}
    for chance in (0.0, 1.0):
        env = world(chance)
        fell = 0
        for seed in range(40):
            env.reset(seed=seed)
            for _ in range(6):
                _state, _reward, done, info = env.step(RIGHT)
                if done:
                    fell += 1 if info["hazard"] else 0
                    break
        results[chance] = fell
    assert results[0.0] == 0
    assert results[1.0] == 40


# ----------------------------------------------------------------------
# 11. Replay carries the value it was recorded under
# ----------------------------------------------------------------------

def trained(chance, episodes=40):
    session = Session(ROOM, parameters={FIELD: chance, "episodes": episodes})
    session.play()
    while session.state == "TRAINING":
        session.advance(budget_ms=25.0)
    return session


def test_the_recording_says_what_it_was_recorded_under():
    session = trained(0.45)
    batch = session.batch()
    assert batch["collapseChance"] == pytest.approx(0.45)
    assert batch["episodes"], "nothing was recorded"
    for episode in batch["episodes"]:
        assert episode["collapseChance"] == pytest.approx(0.45)
    for entry in batch["history"]:
        assert entry["collapseChance"] == pytest.approx(0.45)


def test_the_final_route_says_what_it_was_flown_against():
    session = trained(0.25)
    session.start_replay()
    assert session.replay["collapseChance"] == pytest.approx(0.25)


def test_two_runs_at_different_risks_are_distinguishable_afterwards():
    """The point of storing it: a reward means nothing without the risk."""
    low = trained(0.05).batch()
    high = trained(0.80).batch()
    assert low["collapseChance"] != high["collapseChance"]


# ----------------------------------------------------------------------
# 12. Nothing else moved
# ----------------------------------------------------------------------

def test_the_algorithm_grid_and_rewards_are_untouched():
    room = rooms.room(ROOM)
    assert room["algorithm_default"] == "sarsa"
    assert len(room["grid"]) == 10
    assert all(len(line) == 10 for line in room["grid"])
    # The documented reward table, which is what the README states.
    assert room["rewards"] == {"step": -1.0, "wall": -2.0,
                               "hazard": -100.0, "goal": 100.0}
    # Four collapsing sections, as the README's map has always shown.
    assert sum(line.count(COLLAPSING) for line in room["grid"]) == 4


def test_no_other_room_gained_a_collapse_control():
    for number in rooms.ROOM_NUMBERS:
        if number == ROOM:
            continue
        assert FIELD not in rooms.room(number)["parameters"], (
            "room %d now exposes a collapse control" % number)
        assert Session(number).environment_collapse_chance() is None


def test_no_other_room_reports_a_collapse_row_in_its_readout():
    for number in rooms.ROOM_NUMBERS:
        if number == ROOM:
            continue
        rows = dict(Session(number).readout())
        assert not any("collapse" in label.lower() for label in rows)


# ----------------------------------------------------------------------
# 13. The maintenance void is scenery, and must stay scenery
# ----------------------------------------------------------------------

def test_the_void_is_decor_and_not_a_tile():
    """`decor` dresses the screen. It must not reach the environment."""
    room = rooms.room(ROOM)
    assert room["decor"], "room 2 lost its maintenance void"
    # Not a tile: the grid is untouched by it.
    for line in room["grid"]:
        assert "V" not in line and "v" not in line
    # GridWorld has never heard of it.
    env = world(0.1)
    assert not hasattr(env, "decor")


def test_the_void_changes_no_transition_anywhere():
    """The proof that the scenery is scenery: the MDP is bit-identical.

    Built once with the decor present and once with it stripped out, and
    every (state, action) compared. If a decorative rectangle ever starts
    changing a step, this is what says so.
    """
    plain = dict(rooms.room(ROOM))
    plain.pop("decor", None)
    with_decor = GridWorld(rooms.room(ROOM), slip=0.2, collapse=0.3)
    without = GridWorld(plain, slip=0.2, collapse=0.3)

    assert list(with_decor.all_states()) == list(without.all_states())
    for state in with_decor.all_states():
        assert with_decor.is_terminal(state) == without.is_terminal(state)
        for action in with_decor.actions():
            assert (with_decor.transitions(state, action)
                    == without.transitions(state, action))


def test_the_void_is_drawn_but_only_in_room_two():
    session = Session(ROOM)
    described = session.describe()["definition"]
    kinds = {entity["type"] for entity in described["entities"]}
    assert "void" in kinds, "the maintenance void is not being drawn"
    # 5 rows by 4 columns of it.
    assert sum(1 for e in described["entities"] if e["type"] == "void") == 20
    for number in rooms.ROOM_NUMBERS:
        if number == ROOM:
            continue
        other = Session(number).describe()["definition"]
        assert not any(e["type"] == "void" for e in other["entities"]), (
            "room %d gained scenery it never asked for" % number)


def test_the_collapsing_tile_kind_is_unchanged():
    assert TILE_KINDS[COLLAPSING] == "collapsing"
    assert algorithms.describe_all(["sarsa"])[0]["label"] == "SARSA"
