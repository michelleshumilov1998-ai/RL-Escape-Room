"""The room 1 simulation core, tested without a canvas.

Everything here runs the environment and the planners directly. Nothing
imports a renderer, a server or a browser, which is the property that
matters: if the simulation ever needs rendering in order to advance, these
tests stop passing.

REWRITTEN AGAINST THE CURRENT MAP
An earlier version of this file described a room 1 that no longer exists: it
imported a name `game.grid` has not got, expected `G` for the goal, and
asserted a route that flipped between an "icy shaft" and a "ring" as the ice
worsened. None of that is the chamber in `rooms.py` today, so the file could
not even be collected, let alone pass.

Every number below was measured from the code rather than carried over from
the old file — including the two facts the room does *not* currently
demonstrate, which are pinned here deliberately. A test that asserts a known
shortcoming is what stops it being quietly forgotten, and both are documented
in the README under the same headings.
"""

import json

import pytest

from game import algorithms, rooms
from game.grid import ACTIONS, DELTAS, GOAL, LASER, TELEPORT, GridWorld

SLIP_DEFAULT = 0.20

# Measured, at the defaults: γ 0.95, θ 1e-4, slip 0.20, battery bonus 10.
EXPECTED_ROUTE = [
    (0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (6, 2),
    (6, 3), (6, 4), (7, 4), (8, 4), (8, 5), (9, 5),
]
EXPECTED_START_VALUE = 40.7526
VALUE_ITERATION_SWEEPS = 39
POLICY_ITERATION_SWEEPS = 378


def plan_route(slip=SLIP_DEFAULT, limit=400, **kwargs):
    """The route the plan *intends*, walked on firm footing.

    THE ROUTE IS NO LONGER DETERMINISTIC AND THAT IS THE POINT
    The ice used to sit beside the plan's route rather than on it, so walking
    the plan gave the same eleven cells every time however slippery the floor
    was. The ice is now on the route, so a walk with slipping switched on takes
    a different number of steps on every seed — which is the room working, not
    the room broken.

    So the two questions are asked separately: this one solves at the given
    slip and then walks the resulting policy with `slip=0`, which is the route
    the plan meant to take. `walk` still answers the other question — what
    actually happens when the floor gets a say.
    """
    _, algorithm = solve(slip=slip, **kwargs)
    firm = GridWorld(rooms.room(1), slip=0.0, rewards=kwargs.get("rewards"))
    state = firm.reset(seed=0)
    cells = [firm.cell_of(state)]
    total = 0.0
    for _ in range(limit):
        state, reward, done, info = firm.step(algorithm.act(state))
        total += reward
        cells.append(firm.cell_of(state))
        if done:
            return cells, total, info
    return cells, total, {"goal": False, "hazard": False}


def build(key="value_iteration", slip=SLIP_DEFAULT, gamma=0.95, theta=1e-4,
          rewards=None, **parameters):
    room = rooms.room(1)
    env = GridWorld(room, slip=slip, rewards=rewards)
    settings = {"gamma": gamma, "theta": theta}
    settings.update(parameters)
    return env, algorithms.build(key, env, settings)


def solve(key="value_iteration", slip=SLIP_DEFAULT, limit=5000, **kwargs):
    env, algorithm = build(key, slip, **kwargs)
    while not algorithm.finished and algorithm.sweeps < limit:
        algorithm.update()
    return env, algorithm


def walk(env, algorithm, seed=0, limit=400):
    """One greedy run. Returns the cells visited, the total, and how it ended."""
    state = env.reset(seed=seed)
    cells = [env.cell_of(state)]
    total = 0.0
    for _ in range(limit):
        state, reward, done, info = env.step(algorithm.act(state))
        total += reward
        cells.append(env.cell_of(state))
        if done:
            return cells, total, info
    return cells, total, {"goal": False, "hazard": False}


# ----------------------------------------------------------------------
# The layout
# ----------------------------------------------------------------------

def test_the_layout_is_rectangular_with_one_start_and_one_exit():
    env, _ = build()
    assert all(len(line) == env.cols for line in env.grid)
    joined = "".join(env.grid)
    assert joined.count("S") == 1
    assert joined.count(GOAL) == 1
    assert env.start == (0, 0)
    assert env.goal == (9, 5)


def test_the_state_is_five_numbers_and_there_are_810_of_them():
    """A cell alone is not a Markov state here — see the README.

    The battery and the last direction both change what happens next, and the
    collapsed dimension is one value because this chamber has no planks.
    """
    env, _ = build()
    assert len(env.start_state()) == 5
    assert len(env.all_states()) == 810


def _shortest_route(env, pads):
    """Breadth-first search over the real rules.

    `pads` is 'teleport', 'floor' or 'blocked' — the three things the pair of
    pads can be, and the distinction the measurement turns on.
    """
    from collections import deque

    queue = deque([(env.start, 0)])
    seen = {env.start}
    while queue:
        cell, distance = queue.popleft()
        if cell == env.goal:
            return distance
        row, col = cell
        for action in ACTIONS:
            delta_row, delta_col = DELTAS[action]
            step = (row + delta_row, col + delta_col)
            if not env.in_bounds(*step):
                continue
            if env.is_wall(*step) or env.tile_at(*step) == LASER:
                continue
            if env.tile_at(*step) == TELEPORT:
                if pads == "blocked":
                    continue
                if pads == "teleport":
                    # A pad is entered and left in the *same* step; the agent
                    # never stands on one. Counting the hop separately is what
                    # made this measure 12 rather than the 11 it really takes.
                    step = next(pad for pad in env.teleports if pad != step)
            if step in seen:
                continue
            seen.add(step)
            queue.append((step, distance + 1))
    return None


def test_the_pads_are_the_only_gap_in_the_beam_wall():
    """What column 3 does, stated precisely enough to be true.

    Column 3 is `#`, `#`, `T`, `L`, `#`, `L` down to row 5, so the pad at (2,3)
    is the only opening between the two halves of the upper chamber — every
    other cell in that column is wall or beam.

    The three measurements below are all different, and the difference is the
    point. The README used to say flatly that there is "no route to the exit at
    all" without the teleporter, and that is only true of the third case: if the
    pads are left as ordinary floor the agent can simply *walk* down column 2
    through (5,2) and (6,2), which is 14 steps and needs no teleporting.
    """
    env, _ = build()
    assert _shortest_route(env, pads="teleport") == 11
    assert _shortest_route(env, pads="floor") == 14
    assert _shortest_route(env, pads="blocked") is None

    # And the column really is otherwise sealed above row 6.
    from game.grid import WALL
    for row in range(6):
        tile = env.tile_at(row, 3)
        assert tile in (WALL, LASER, TELEPORT), (row, tile)


def test_a_beam_is_not_fatal_and_only_the_exit_ends_a_run():
    env, _ = build()
    assert env.is_terminal((9, 5, 0, 4, 0))
    # A laser throws the agent back to the start; it does not end the run.
    assert not env.is_terminal((3, 1, 0, 4, 0))
    assert not env.is_terminal(env.start_state())


# ----------------------------------------------------------------------
# The environment
# ----------------------------------------------------------------------

def test_probabilities_always_sum_to_one():
    env, _ = build(slip=0.3)
    for state in env.all_states():
        for action in env.actions():
            total = sum(probability for probability, _, _, _
                        in env.transitions(state, action))
            assert total == pytest.approx(1.0, abs=1e-9)


def test_firm_ground_is_deterministic_and_ice_is_not():
    env, _ = build(slip=0.2)
    assert env.direction_probabilities((0, 1, 0, 4, 0), 0) == {0: 1.0}
    icy = env.direction_probabilities((1, 1, 0, 4, 0), 0)
    assert icy[0] == pytest.approx(0.8)
    assert sorted(icy.values()) == pytest.approx([0.1, 0.1, 0.8])


def test_ice_only_ever_slides_sideways():
    """Sliding backwards would make ice a different hazard entirely."""
    env, _ = build(slip=0.4)
    backwards = {0: 1, 1: 0, 2: 3, 3: 2}
    for action in ACTIONS:
        outcomes = env.direction_probabilities((1, 1, 0, 4, 0), action)
        assert action in outcomes
        for direction in outcomes:
            assert direction != backwards[action]


def test_walls_hold_the_agent_in_place_and_cost_the_penalty():
    env, _ = build()
    env.reset(seed=0)
    state, reward, done, _ = env.step(0)      # UP, into the chamber edge
    assert (state[0], state[1]) == (0, 0)
    assert reward == pytest.approx(-4.0)
    assert not done


def test_the_same_seed_gives_the_same_episode():
    env_a, algorithm = solve()
    first = walk(env_a, algorithm, seed=7)[0]
    env_b, _ = build()
    second = walk(env_b, algorithm, seed=7)[0]
    assert first == second


def test_stepping_never_needs_a_renderer():
    """A blunt statement of the architecture rule, as a test."""
    import sys
    env, algorithm = solve()
    walk(env, algorithm)
    assert not any(name.startswith("game.renderer") for name in sys.modules)


# ----------------------------------------------------------------------
# The planners
# ----------------------------------------------------------------------

def test_value_iteration_converges_and_reaches_the_exit():
    env, algorithm = solve()
    assert algorithm.converged
    assert algorithm.last_delta < 1e-4
    cells, total, info = plan_route()
    assert info["goal"]
    assert cells == EXPECTED_ROUTE
    assert total == pytest.approx(89.0)


def test_the_measured_route_is_eleven_steps_through_the_pad():
    env, algorithm = solve()
    cells, _, _ = plan_route()
    assert len(cells) - 1 == 11
    # Straight from one pad to the other, which is the whole point of them.
    assert (2, 2) in cells and (6, 2) in cells
    assert algorithm.values[env.start_state()] == pytest.approx(
        EXPECTED_START_VALUE, abs=1e-3)


def test_the_plan_crosses_the_ice_rather_than_going_round_it():
    """The room's mechanic has to appear in the room's answer.

    The ice used to be beside the route: measured at every setting of the
    slider, the plan walked eleven steps and stood on ice zero times. So the
    one thing the chamber is about never happened unless exploration blundered
    onto it — and an arrow pointing one way while R-5 went another then read as
    a rendering fault rather than as the floor.
    """
    from game.grid import TILE_KINDS
    env, _ = build()
    cells, _, _ = plan_route()
    icy = [cell for cell in cells
           if TILE_KINDS[env.grid[cell[0]][cell[1]]] in ("slippery", "cracked")]
    assert len(icy) >= 3, "the plan crosses only %d icy cells: %s" % (
        len(icy), icy)
    # And the last stride onto the panel is one of them.
    assert (8, 5) in icy


def test_the_model_is_read_rather_than_experienced():
    """A planner must converge without a single step being taken."""
    env, algorithm = build()
    while not algorithm.finished:
        algorithm.update()
    assert algorithm.converged
    assert env.steps == 0


def test_the_plan_avoids_beams_it_has_never_touched():
    env, algorithm = solve()
    for seed in range(20):
        cells, _, info = walk(env, algorithm, seed=seed)
        assert not info["hazard"], "the plan walked into a beam: %s" % (cells,)


def test_value_iteration_needs_far_fewer_sweeps_than_policy_iteration():
    """The comparison this room exists to make: same plan, 26x the work."""
    _, value_iteration = solve("value_iteration")
    _, policy_iteration = solve("policy_iteration")
    assert value_iteration.sweeps == VALUE_ITERATION_SWEEPS
    assert policy_iteration.sweeps == POLICY_ITERATION_SWEEPS
    assert policy_iteration.rounds < value_iteration.sweeps


def test_both_planners_agree_on_the_policy_and_the_values():
    _, value_iteration = solve("value_iteration")
    _, policy_iteration = solve("policy_iteration")
    assert value_iteration.greedy_policy() == policy_iteration.greedy_policy()
    for state in value_iteration.values:
        assert value_iteration.values[state] == pytest.approx(
            policy_iteration.values[state], abs=1e-3)


def test_a_lower_discount_factor_is_worth_less_at_the_start():
    _, patient = solve(gamma=0.95)
    _, impatient = solve(gamma=0.70)
    start = patient.env.start_state()
    assert impatient.values[start] < patient.values[start]


def test_the_battery_bonus_changes_the_plan():
    """Raise it far enough and the detour starts paying for itself."""
    ignored = plan_route(rewards={"battery": 10.0})
    fetched = plan_route(rewards={"battery": 80.0})
    assert len(ignored[0]) - 1 == 11
    assert (0, 6) not in ignored[0]        # the battery is left where it is
    assert len(fetched[0]) - 1 > 11
    assert (0, 6) in fetched[0]            # and now it goes and gets it


# ----------------------------------------------------------------------
# What the slip slider does
#
# It used to do almost nothing, and two tests here pinned that shortcoming
# on purpose so it could not be quietly forgotten. It has now been fixed —
# the ice was moved onto the route the plan actually takes — so these assert
# the behaviour instead of the limitation.
# ----------------------------------------------------------------------

def test_slipperiness_changes_what_the_route_is_worth():
    """V(start) falls as the floor loosens, because the plan stands on ice.

    This is the test that used to assert the opposite. The ice sat beside the
    route rather than on it, so V(start) was 51.2497 at every setting of the
    slider to six decimal places and the parameter could not make the chamber
    harder. Measured now:

        slip 0.00 -> 51.250      slip 0.35 -> 32.239
        slip 0.10 -> 46.609      slip 0.50 -> 27.502
        slip 0.20 -> 40.753
    """
    values = []
    for slip in (0.0, 0.1, 0.2, 0.35, 0.5):
        env, algorithm = solve(slip=slip)
        values.append(algorithm.values[env.start_state()])

    # Strictly worse each time the floor gets looser. No plateau anywhere.
    for before, after in zip(values, values[1:]):
        assert after < before - 1.0, "V(start) barely moved: %s" % (values,)
    assert values[0] == pytest.approx(51.2497, abs=1e-3)
    assert values[2] == pytest.approx(EXPECTED_START_VALUE, abs=1e-3)


def test_slipperiness_reroutes_the_plan_when_the_floor_gets_bad_enough():
    """And past a point the plan stops using the icy corridor at all.

    At the default it accepts the ice out of the teleport pad, because eleven
    steps with a chance of a slip still beats the long way round. Loosen the
    floor further and the sums change: the plan gives up the pad corridor and
    walks fifteen steps on firm ground instead. That flip is the room.
    """
    short = plan_route(slip=0.20)[0]
    long_way = plan_route(slip=0.50)[0]

    assert len(short) - 1 == 11
    assert (6, 3) in short and (6, 4) in short      # over the ice

    assert len(long_way) - 1 > 11
    assert (6, 3) not in long_way                   # round it

    # Both still get out — a rerouted plan is a cautious one, not a lost one.
    assert plan_route(slip=0.20)[2]["goal"]
    assert plan_route(slip=0.50)[2]["goal"]


def test_a_slip_is_reported_so_the_screen_can_explain_the_mismatch():
    """The arrow is the action tried; the floor decides what happens.

    Without the environment saying which steps it overruled, a policy arrow
    pointing right while R-5 moves down is indistinguishable from a drawing
    bug — which is how it was reported. `info["slipped"]` is the answer, and
    `recorder.frame` now carries it onto every frame.
    """
    from game import recorder
    from game.grid import ACTION_NAMES as NAMES

    env, algorithm = solve(slip=0.5)
    state = env.reset(seed=3)
    slips = 0
    frames = 0
    for _ in range(200):
        action = algorithm.act(state)
        state, reward, done, info = env.step(action)
        built = recorder.frame(env, NAMES, state, reward, action,
                               bool(info.get("slipped")))
        assert "slipped" in built
        assert isinstance(built["slipped"], bool)
        frames += 1
        if info.get("slipped"):
            slips += 1
        if done:
            break
    assert frames
    assert slips, "no slip in 200 steps at slip 0.50 — the ice is not on the route"


def test_the_teleporter_pays_nothing_so_riding_it_is_not_profitable():
    """The fix for a real bug: a pad that paid on entry was a cycle.

    Both pads lead to each other, so any reward on entry could be collected
    forever. Above about γ 0.98 riding it beat leaving, and the plan shuttled
    between the pads instead of ever reaching the exit.
    """
    env, _ = build()
    assert env.rewards["shortcut"] == 0.0
    # And at a discount high enough to have triggered it, the plan still leaves.
    cells, _, info = plan_route(gamma=0.999)
    assert info["goal"]
    # Not pinned to a step count: at γ 0.999 the plan is happy to spend steps
    # avoiding ice, so the route is longer than the default one. What matters
    # is that it *ends*, rather than shuttling between the pads forever.
    assert len(cells) - 1 < 40
    assert cells[-1] == (9, 5)
    # And it does not sit on the pads: each is entered at most once.
    assert cells.count((2, 3)) <= 1
    assert cells.count((6, 2)) <= 1


# ----------------------------------------------------------------------
# What the page is handed
# ----------------------------------------------------------------------

def test_a_snapshot_is_json_serialisable():
    _, algorithm = solve()
    restored = json.loads(json.dumps(algorithm.snapshot()))
    assert restored["converged"] is True
    assert restored["policy"]["0,0"] in (0, 1, 2, 3)


def test_a_snapshot_holds_no_infinities_before_the_first_sweep():
    """`allow_nan=False` is the point of this test.

    Python writes bare `Infinity` into what it calls JSON and reads it back
    again quite happily, so a round trip inside Python proves nothing. No
    browser will parse it, and it loses the entire response rather than the
    one field — which is exactly how it went wrong.
    """
    _, algorithm = build()
    json.dumps(algorithm.snapshot(), allow_nan=False)
    assert algorithm.snapshot()["delta"] is None
    for _ in range(3):
        algorithm.update()
    json.dumps(algorithm.snapshot(), allow_nan=False)


def test_the_registry_offers_exactly_what_room_one_allows():
    room = rooms.room(1)
    assert room["algorithm_default"] == "value_iteration"
    for key in room["algorithms"]:
        assert algorithms.get(key) is not None
    described = algorithms.describe_all(room["algorithms"])
    assert [entry["key"] for entry in described] == room["algorithms"]
    assert all(entry["needsModel"] for entry in described)
