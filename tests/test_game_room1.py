"""The room 1 simulation core, tested without a canvas.

Everything here runs the environment and the planners directly. Nothing
imports a renderer, a server or a browser, which is the property that
matters: if the simulation ever needs rendering in order to advance, these
tests stop passing.
"""

import pytest

from game import algorithms, rooms
from game.grid import ACTIONS, GOAL, HAZARD, GridWorld

SLIP_DEFAULT = 0.20


def build(key="value_iteration", slip=SLIP_DEFAULT, gamma=0.95, theta=1e-4):
    room = rooms.room(1)
    env = GridWorld(room, slip=slip)
    algorithm = algorithms.build(key, env, {"gamma": gamma, "theta": theta})
    return env, algorithm


def solve(key="value_iteration", slip=SLIP_DEFAULT, limit=4000, **kwargs):
    env, algorithm = build(key, slip, **kwargs)
    while not algorithm.finished and algorithm.sweeps < limit:
        algorithm.update()
    return env, algorithm


def walk(env, algorithm, seed=0, limit=80):
    """One greedy run. Returns the cells visited and how it ended."""
    state = env.reset(seed=seed)
    path = [state]
    for _ in range(limit):
        state, _, done, info = env.step(algorithm.act(state))
        path.append(state)
        if done:
            return path, info
    return path, {"goal": False, "hazard": False}


def route(path):
    """Which of the two ways across the chamber a run took."""
    cells = set(path)
    if cells & {(1, 5), (2, 5)}:
        return "shaft"
    if cells & {(9, 3), (9, 5)}:
        return "ring"
    return "unclear"


# ----------------------------------------------------------------------
# The layout
# ----------------------------------------------------------------------

def test_the_layout_is_rectangular_and_has_one_start_and_one_goal():
    env, _ = build()
    assert all(len(line) == env.cols for line in env.grid)
    joined = "".join(env.grid)
    assert joined.count("S") == 1
    assert joined.count("G") == 1


def test_both_routes_exist_and_the_shaft_is_the_shorter_one():
    """Measured by breadth-first search, not asserted from the comment."""
    from collections import deque

    env, _ = build()

    def shortest(blocked):
        start = env.start_state()
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            (row, col), distance = queue.popleft()
            if (row, col) == env.goal:
                return distance
            for action in ACTIONS:
                from game.grid import DELTAS
                delta_row, delta_col = DELTAS[action]
                cell = (row + delta_row, col + delta_col)
                if cell in seen or cell in blocked:
                    continue
                if env.is_wall(*cell):
                    continue
                if env.tile_at(*cell) == HAZARD:
                    continue
                seen.add(cell)
                queue.append((cell, distance + 1))
        return None

    shaft = shortest(blocked={(9, 0)})            # the ring closed off
    ring = shortest(blocked={(1, 5), (2, 5)})     # the shaft closed off
    assert shaft == 13
    assert ring == 23


def test_the_icy_shaft_cells_have_a_hazard_on_either_side():
    """This is the whole reason the room has a decision in it."""
    env, _ = build()
    for row in (1, 2):
        assert env.tile_at(row, 4) == HAZARD
        assert env.tile_at(row, 6) == HAZARD
        assert env.tile_at(row, 5) == "~"


def test_the_ring_has_ice_but_never_beside_a_hazard():
    env, _ = build()
    assert env.tile_at(9, 2) == "~"
    for row, col in ((8, 2), (9, 1), (9, 3)):
        assert env.tile_at(row, col) != HAZARD


# ----------------------------------------------------------------------
# The environment
# ----------------------------------------------------------------------

def test_probabilities_always_sum_to_one():
    env, _ = build(slip=0.3)
    for state in env.all_states():
        for action in env.actions():
            total = sum(probability
                        for probability, _, _, _ in env.transitions(state, action))
            assert total == pytest.approx(1.0, abs=1e-9)


def test_firm_ground_is_deterministic_and_ice_is_not():
    env, _ = build(slip=0.2)
    assert env.direction_probabilities((0, 0), 1) == {1: 1.0}
    icy = env.direction_probabilities((1, 5), 1)
    assert icy[1] == pytest.approx(0.8)
    assert sorted(icy.values()) == pytest.approx([0.1, 0.1, 0.8])


def test_ice_only_ever_slides_sideways():
    """Sliding backwards would make the shaft harmless."""
    env, _ = build(slip=0.4)
    for action in ACTIONS:
        outcomes = env.direction_probabilities((1, 5), action)
        assert action in outcomes
        for direction in outcomes:
            assert direction != {0: 1, 1: 0, 2: 3, 3: 2}[action]


def test_a_hazard_ends_the_run_and_the_goal_ends_it_too():
    env, _ = build()
    assert env.is_terminal((1, 4))
    assert env.is_terminal(env.goal)
    assert not env.is_terminal(env.start_state())


def test_walls_hold_the_agent_in_place_and_cost_the_penalty():
    env, _ = build()
    env.reset(seed=0)
    state, reward, done, _ = env.step(0)      # UP, into the chamber edge
    assert state == (0, 0)
    assert reward == pytest.approx(-4.0)
    assert not done


def test_the_same_seed_gives_the_same_episode():
    env_a, algorithm = solve()
    first, _ = walk(env_a, algorithm, seed=7)
    env_b, _ = build()
    second, _ = walk(env_b, algorithm, seed=7)
    assert first == second


def test_stepping_never_needs_a_renderer():
    """A blunt statement of the architecture rule, as a test."""
    import sys
    env, algorithm = solve()
    walk(env, algorithm)
    assert "game.renderer" not in sys.modules


# ----------------------------------------------------------------------
# The planners
# ----------------------------------------------------------------------

def test_value_iteration_converges_and_reaches_the_goal():
    env, algorithm = solve()
    assert algorithm.converged
    assert algorithm.last_delta < 1e-4
    _, info = walk(env, algorithm)
    assert info["goal"]


def test_the_model_is_read_rather_than_experienced():
    """A planner must converge without a single step being taken."""
    env, algorithm = build()
    while not algorithm.finished:
        algorithm.update()
    assert algorithm.converged
    assert env.steps == 0


def test_the_plan_avoids_hazards_it_has_never_touched():
    env, algorithm = solve()
    for _ in range(20):
        path, info = walk(env, algorithm, seed=_)
        assert not info["hazard"], "the plan walked into a beam: %s" % (path,)


def test_the_route_flips_from_the_shaft_to_the_ring_as_the_ice_worsens():
    """The headline result of the room, and nothing is learned to get it."""
    env, algorithm = solve(slip=0.04)
    assert route(walk(env, algorithm)[0]) == "shaft"

    env, algorithm = solve(slip=0.30)
    assert route(walk(env, algorithm)[0]) == "ring"


def test_a_lower_discount_factor_is_worth_less_at_the_start():
    _, patient = solve(gamma=0.95)
    _, impatient = solve(gamma=0.70)
    start = (0, 0)
    assert impatient.values[start] < patient.values[start]


def test_both_planners_agree_on_the_policy_and_the_values():
    value_env, value_iteration = solve("value_iteration")
    _, policy_iteration = solve("policy_iteration")

    first = value_iteration.greedy_policy()
    second = policy_iteration.greedy_policy()
    assert first == second

    for state in value_iteration.values:
        assert value_iteration.values[state] == pytest.approx(
            policy_iteration.values[state], abs=1e-3)


def test_policy_iteration_needs_fewer_rounds_but_more_sweeps():
    """Documented, because it is the reason to have both available."""
    _, value_iteration = solve("value_iteration")
    _, policy_iteration = solve("policy_iteration")
    assert policy_iteration.rounds < value_iteration.sweeps
    assert policy_iteration.sweeps > value_iteration.sweeps


def test_a_snapshot_is_json_serialisable():
    import json
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
    import json
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
