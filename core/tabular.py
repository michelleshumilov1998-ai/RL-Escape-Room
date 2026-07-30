"""The pieces every tabular room shares.

Rooms 2 and 3 both keep a Q-table and both pick actions with epsilon-greedy, so
those parts live here rather than being written out twice. What is *not* here is
the update rule or the training loop: those are the interesting part of each room
and belong in the room that teaches them.

A Q-table is a dictionary of dictionaries: `q[state][action] -> value`. Plain,
readable, and easy to print while debugging.
"""


def new_q_table(states, action_list, initial_value=0.0):
    """A fresh table with every entry set to `initial_value`.

    Setting it above any reward the room can pay is called *optimistic
    initialisation*: it makes an untried action look better than a tried one, so
    the agent works through all of them before settling on a favourite.
    """
    return {state: {action: initial_value for action in action_list}
            for state in states}


def best_action(q, state, action_list):
    """The action with the highest value in this state.

    Ties go to whichever comes first in `action_list`, so reading a policy off a
    finished table always gives the same answer.
    """
    chosen = action_list[0]
    for action in action_list[1:]:
        if q[state][action] > q[state][chosen]:
            chosen = action
    return chosen


def best_value(q, state, action_list):
    """The highest value available in this state — the `max` in Q-Learning."""
    return q[state][best_action(q, state, action_list)]


def best_actions(q, state, action_list, tolerance=1e-12):
    """Every action tied for the highest value in this state."""
    highest = best_value(q, state, action_list)
    return [action for action in action_list
            if q[state][action] >= highest - tolerance]


def epsilon_greedy(q, state, action_list, epsilon, rng):
    """Mostly the best known action, but `epsilon` of the time a random one.

    Ties are broken *randomly*. A fresh table is all equal, so every action ties
    at the start; always taking the first one would send the robot the same way on
    every early episode and whole parts of the room would never be explored.
    """
    if rng.random() < epsilon:
        return rng.choice(action_list)
    tied = best_actions(q, state, action_list)
    if len(tied) == 1:
        return tied[0]
    return rng.choice(tied)


def greedy_policy(q, states, action_list, is_terminal):
    """Read the learned policy off the table, with no exploration left in it."""
    policy = {}
    for state in states:
        policy[state] = None if is_terminal(state) else best_action(
            q, state, action_list)
    return policy


def mean_absolute_q(q):
    """The average size of the entries in the table.

    Used as a convergence signal: while learning is still moving this climbs, and
    once the values settle it goes flat.
    """
    total = 0.0
    count = 0
    for actions_for_state in q.values():
        for value in actions_for_state.values():
            total += abs(value)
            count += 1
    return total / count if count else 0.0


def moving_average(values, window):
    """A simple trailing average, used by most of the training graphs."""
    if window <= 1 or not values:
        return list(values)
    averaged = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= window:
            running -= values[index - window]
        averaged.append(running / min(index + 1, window))
    return averaged
