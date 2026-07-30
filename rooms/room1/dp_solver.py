"""Dynamic Programming for Room 1: Value Iteration and Policy Iteration.

Both algorithms are written out by hand.  No reinforcement learning library is
used anywhere.  The style is deliberately plain so the maths is easy to follow:

    Q(s, a)  =  sum over s'  of  P(s'|s,a) * ( R(s,a,s') + gamma * V(s') )

Value Iteration repeatedly replaces V(s) with the best Q(s, a).
Policy Iteration alternates between measuring a policy and improving it.

Both need the full transition model, which `Room1Env.transitions` provides.
That is the whole point of Room 1: because the model is known, the robot can
plan a route around lasers it has never touched.
"""

import time


def q_value(env, state, action, values, gamma):
    """The expected value of taking `action` in `state`.

    This is the one line of maths that both algorithms are built from.
    """
    total = 0.0
    for probability, next_state, reward, done in env.transitions(state, action):
        # A finished episode has no future, so V(next_state) counts as 0.
        future_value = 0.0 if done else values[next_state]
        total += probability * (reward + gamma * future_value)
    return total


def best_action(env, state, values, gamma):
    """The action with the highest Q value in this state.

    Ties are broken by the fixed order of `env.actions()`, so the same value
    table always produces exactly the same policy.
    """
    chosen_action = None
    chosen_value = None
    for action in env.actions():
        value = q_value(env, state, action, values, gamma)
        if chosen_value is None or value > chosen_value:
            chosen_action = action
            chosen_value = value
    return chosen_action, chosen_value


def greedy_policy(env, values, gamma):
    """Read a policy off a value table: pi(s) = argmax_a Q(s, a)."""
    policy = {}
    for state in env.all_states():
        if env.is_terminal(state):
            policy[state] = None
            continue
        action, _ = best_action(env, state, values, gamma)
        policy[state] = action
    return policy


def bellman_residual(env, values, gamma):
    """The largest remaining error in the value table.

    A converged value table satisfies V(s) = max_a Q(s, a) for every state, so
    this number tells us how close we really are.
    """
    worst = 0.0
    for state in env.all_states():
        if env.is_terminal(state):
            continue
        _, value = best_action(env, state, values, gamma)
        worst = max(worst, abs(value - values[state]))
    return worst


def count_policy_changes(old_policy, new_policy):
    """How many states changed their preferred action."""
    changes = 0
    for state in new_policy:
        if old_policy.get(state) != new_policy[state]:
            changes += 1
    return changes


# ----------------------------------------------------------------------
# Value Iteration
# ----------------------------------------------------------------------

def value_iteration(env, gamma=0.95, theta=1e-6, max_sweeps=1000):
    """Solve the room by repeatedly applying the Bellman optimality update.

        V_new(s) = max_a  sum_s'  P(s'|s,a) * ( R + gamma * V(s') )

    Stops when the largest change in one sweep is smaller than `theta`, or when
    `max_sweeps` sweeps have been done.
    """
    states = env.all_states()
    values = {state: 0.0 for state in states}
    start_state = env.start_state()

    delta_history = []
    start_value_history = []
    policy_change_history = []
    previous_policy = {}

    started_at = time.perf_counter()
    sweeps = 0

    for _ in range(max_sweeps):
        sweeps += 1
        delta = 0.0

        for state in states:
            if env.is_terminal(state):
                # A terminal state has no future reward, so its value stays 0.
                continue
            _, new_value = best_action(env, state, values, gamma)
            delta = max(delta, abs(new_value - values[state]))
            values[state] = new_value

        # Record the three curves the interface plots.
        current_policy = greedy_policy(env, values, gamma)
        delta_history.append(delta)
        start_value_history.append(values[start_state])
        policy_change_history.append(count_policy_changes(previous_policy, current_policy))
        previous_policy = current_policy

        if delta < theta:
            break

    runtime = time.perf_counter() - started_at
    policy = greedy_policy(env, values, gamma)

    return {
        "algorithm": "Value Iteration",
        "values": values,
        "policy": policy,
        "iterations": sweeps,
        "runtime_seconds": runtime,
        "start_state_value": values[start_state],
        "final_delta": delta_history[-1] if delta_history else 0.0,
        "bellman_residual": bellman_residual(env, values, gamma),
        "converged": bool(delta_history) and delta_history[-1] < theta,
        "delta_history": delta_history,
        "start_value_history": start_value_history,
        "policy_change_history": policy_change_history,
        "gamma": gamma,
        "theta": theta,
        "max_sweeps": max_sweeps,
    }


# ----------------------------------------------------------------------
# Policy Iteration
# ----------------------------------------------------------------------

def policy_evaluation(env, policy, values, gamma, theta, max_sweeps=1000):
    """Work out how good the given policy is.

    Repeatedly applies  V(s) = Q(s, pi(s))  until the values stop moving.
    Returns (values, number_of_sweeps).
    """
    sweeps = 0
    for _ in range(max_sweeps):
        sweeps += 1
        delta = 0.0
        for state in env.all_states():
            if env.is_terminal(state):
                continue
            new_value = q_value(env, state, policy[state], values, gamma)
            delta = max(delta, abs(new_value - values[state]))
            values[state] = new_value
        if delta < theta:
            break
    return values, sweeps


def policy_improvement(env, values, gamma):
    """Pick the best action everywhere, given the current values."""
    return greedy_policy(env, values, gamma)


def policy_iteration(env, gamma=0.95, theta=1e-6, max_iterations=100):
    """Solve the room by alternating evaluation and improvement.

    1. Start from an arbitrary policy (always UP).
    2. Evaluate it.
    3. Improve it.
    4. Stop when the improvement step changes nothing.
    """
    states = env.all_states()
    values = {state: 0.0 for state in states}
    start_state = env.start_state()

    # Step 1: an arbitrary starting policy.  The first action of the action list
    # is used for every non-terminal state.
    first_action = env.actions()[0]
    policy = {}
    for state in states:
        policy[state] = None if env.is_terminal(state) else first_action

    delta_history = []
    start_value_history = []
    policy_change_history = []
    evaluation_sweeps_total = 0

    started_at = time.perf_counter()
    iterations = 0
    stable = False

    for _ in range(max_iterations):
        iterations += 1

        # Step 2: evaluate the current policy.
        values_before = dict(values)
        values, evaluation_sweeps = policy_evaluation(env, policy, values, gamma, theta)
        evaluation_sweeps_total += evaluation_sweeps

        biggest_value_move = 0.0
        for state in states:
            biggest_value_move = max(biggest_value_move,
                                     abs(values[state] - values_before[state]))

        # Step 3: improve it.
        new_policy = policy_improvement(env, values, gamma)
        changes = count_policy_changes(policy, new_policy)

        delta_history.append(biggest_value_move)
        start_value_history.append(values[start_state])
        policy_change_history.append(changes)

        policy = new_policy

        # Step 4: nothing changed, so the policy is optimal.
        if changes == 0:
            stable = True
            break

    runtime = time.perf_counter() - started_at

    return {
        "algorithm": "Policy Iteration",
        "values": values,
        "policy": policy,
        "iterations": iterations,
        "evaluation_sweeps": evaluation_sweeps_total,
        "runtime_seconds": runtime,
        "start_state_value": values[start_state],
        "final_delta": delta_history[-1] if delta_history else 0.0,
        "bellman_residual": bellman_residual(env, values, gamma),
        "converged": stable,
        "delta_history": delta_history,
        "start_value_history": start_value_history,
        "policy_change_history": policy_change_history,
        "gamma": gamma,
        "theta": theta,
        "max_sweeps": max_iterations,
    }


# ----------------------------------------------------------------------
# Comparing the two algorithms
# ----------------------------------------------------------------------

def compare_policies(env, first_result, second_result, gamma, tolerance=1e-6):
    """Compare two policies state by state.

    Two policies can disagree and still both be optimal, when two actions have
    the same Q value.  Those cases are counted separately as "ties", because a
    tie is normal maths and not a bug.
    """
    first_policy = first_result["policy"]
    second_policy = second_result["policy"]
    values = first_result["values"]

    identical = 0
    tied = 0
    genuinely_different = []
    compared = 0

    for state in env.all_states():
        if env.is_terminal(state):
            continue
        compared += 1
        first_action = first_policy[state]
        second_action = second_policy[state]

        if first_action == second_action:
            identical += 1
            continue

        # The actions differ.  Are they worth the same?
        first_value = q_value(env, state, first_action, values, gamma)
        second_value = q_value(env, state, second_action, values, gamma)
        if abs(first_value - second_value) < tolerance:
            tied += 1
        else:
            genuinely_different.append({
                "state": state,
                "first_action": first_action,
                "second_action": second_action,
                "first_value": first_value,
                "second_value": second_value,
            })

    agreement = 100.0 * identical / compared if compared else 0.0
    effective_agreement = 100.0 * (identical + tied) / compared if compared else 0.0

    return {
        "states_compared": compared,
        "identical": identical,
        "tied": tied,
        "different": len(genuinely_different),
        "different_details": genuinely_different,
        "agreement_percent": agreement,
        "agreement_with_ties_percent": effective_agreement,
    }


def solve(env, algorithm, gamma=0.95, theta=1e-6, max_sweeps=1000):
    """Run one of the two algorithms by name."""
    if algorithm == "Policy Iteration":
        return policy_iteration(env, gamma=gamma, theta=theta, max_iterations=max_sweeps)
    return value_iteration(env, gamma=gamma, theta=theta, max_sweeps=max_sweeps)
