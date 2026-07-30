"""Tests for the Dynamic Programming solvers — required checks 22, 23, 24 and 28.

Numbered comments refer to the checklist in the project brief.
"""

import pytest

from core import actions
from rooms.room1 import dp_solver, simulate
from rooms.room1.environment import Room1Env

GAMMA = 0.95
THETA = 1e-6


@pytest.fixture(scope="module")
def env():
    return Room1Env()


@pytest.fixture(scope="module")
def value_result(env):
    return dp_solver.value_iteration(env, gamma=GAMMA, theta=THETA, max_sweeps=2000)


@pytest.fixture(scope="module")
def policy_result(env):
    return dp_solver.policy_iteration(env, gamma=GAMMA, theta=THETA,
                                      max_iterations=200)


# ----------------------------------------------------------------------
# Convergence
# ----------------------------------------------------------------------

def test_value_iteration_converges(value_result):
    """22. Value Iteration converges."""
    assert value_result["converged"]
    assert value_result["delta_history"][-1] < THETA
    assert value_result["iterations"] < 2000
    # The deltas must actually be shrinking, not just end small.
    assert value_result["delta_history"][0] > value_result["delta_history"][-1]
    assert value_result["bellman_residual"] < 1e-4


def test_policy_iteration_converges(policy_result):
    """23. Policy Iteration converges."""
    assert policy_result["converged"]
    assert policy_result["iterations"] < 200
    # It converges when the improvement step changes nothing at all.
    assert policy_result["policy_change_history"][-1] == 0
    assert policy_result["bellman_residual"] < 1e-4


def test_both_methods_record_the_histories_the_graphs_need(value_result,
                                                          policy_result):
    for result in (value_result, policy_result):
        assert len(result["delta_history"]) == result["iterations"]
        assert len(result["start_value_history"]) == result["iterations"]
        assert len(result["policy_change_history"]) == result["iterations"]
        assert result["runtime_seconds"] > 0


def test_terminal_states_keep_a_value_of_zero(env, value_result):
    for state in env.all_states():
        if env.is_terminal(state):
            assert value_result["values"][state] == 0.0


def test_policy_covers_every_non_terminal_state(env, value_result):
    for state in env.all_states():
        action = value_result["policy"][state]
        if env.is_terminal(state):
            assert action is None
        else:
            assert action in env.actions()


# ----------------------------------------------------------------------
# Agreement between the two methods
# ----------------------------------------------------------------------

def test_the_two_methods_agree_or_the_difference_is_a_tie(env, value_result,
                                                          policy_result):
    """24. The two policies are similar, or differences are explained as ties.

    Floating point values are compared with a tolerance, and any state where the
    two chosen actions are worth the same is counted as a tie rather than a
    disagreement.
    """
    comparison = dp_solver.compare_policies(env, value_result, policy_result,
                                            GAMMA, tolerance=1e-6)
    assert comparison["states_compared"] > 0
    assert comparison["different"] == 0, \
        "genuine disagreements: %s" % comparison["different_details"][:3]
    assert comparison["agreement_with_ties_percent"] == pytest.approx(100.0)


def test_the_two_methods_find_the_same_start_value(value_result, policy_result):
    assert value_result["start_state_value"] == \
        pytest.approx(policy_result["start_state_value"], abs=1e-4)


def test_the_two_value_tables_match_everywhere(env, value_result, policy_result):
    for state in env.all_states():
        assert value_result["values"][state] == \
            pytest.approx(policy_result["values"][state], abs=1e-3), \
            "values differ at %s" % (state,)


# ----------------------------------------------------------------------
# The policy is actually good
# ----------------------------------------------------------------------

def test_the_policy_escapes_reliably(env, value_result):
    """A plan that cannot reach the exit is not a plan."""
    batch = simulate.run_many(env, value_result["policy"], episodes=30)
    assert batch["success_rate"] >= 0.9
    assert batch["mean_return"] > 0


def test_the_policy_leaves_the_start_platform(value_result):
    start = (0, 0, False, actions.NONE)
    assert value_result["policy"][start] is not None


# ----------------------------------------------------------------------
# Changing the risk changes the plan
# ----------------------------------------------------------------------

def test_more_slippery_ice_changes_the_policy():
    """28. Changing slipping probability changes the policy.

    With safe ice the plan takes the short frozen shaft.  With slippery ice it
    abandons the shaft for the long safe ring.  This is the headline result of
    Room 1, so a regression here matters.
    """
    safe_env = Room1Env(strong_ice_intended=0.95)
    risky_env = Room1Env(strong_ice_intended=0.35)

    safe = dp_solver.value_iteration(safe_env, gamma=GAMMA, theta=THETA,
                                     max_sweeps=2000)
    risky = dp_solver.value_iteration(risky_env, gamma=GAMMA, theta=THETA,
                                      max_sweeps=2000)

    differing = [state for state in safe_env.all_states()
                 if safe["policy"][state] != risky["policy"][state]]
    assert differing, "the policy did not change at all"

    # And the change is visible in the route the robot actually walks.
    safe_route = simulate.describe_route(
        simulate.run_episode(safe_env, safe["policy"], seed=1)["frames"])
    risky_route = simulate.describe_route(
        simulate.run_episode(risky_env, risky["policy"], seed=1)["frames"])
    assert "Frozen shaft" in safe_route
    assert "Frozen shaft" not in risky_route


def test_a_bigger_laser_penalty_never_increases_the_start_value():
    """Making the hazard worse cannot make the plan look better."""
    mild = dp_solver.value_iteration(Room1Env(laser_penalty=-5), gamma=GAMMA,
                                     theta=THETA, max_sweeps=2000)
    harsh = dp_solver.value_iteration(Room1Env(laser_penalty=-120), gamma=GAMMA,
                                      theta=THETA, max_sweeps=2000)
    assert harsh["start_state_value"] <= mild["start_state_value"] + 1e-6


def test_q_value_matches_a_hand_computed_expectation(env, value_result):
    """The one line of maths everything else is built on."""
    state = (9, 0, False, actions.NONE)
    action = actions.RIGHT
    expected = 0.0
    for probability, next_state, reward, done in env.transitions(state, action):
        future = 0.0 if done else value_result["values"][next_state]
        expected += probability * (reward + GAMMA * future)
    assert dp_solver.q_value(env, state, action, value_result["values"], GAMMA) == \
        pytest.approx(expected)
