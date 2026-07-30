"""SARSA for Room 2, written out by hand.

SARSA is named after the five things one update needs:

    State, Action, Reward, next State, next Action

and that last item is the whole point. After acting and seeing the reward, the
agent picks the action it is *actually going to take next* — using the same
epsilon-greedy policy, exploration and all — and uses that action's value as the
target:

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s',a') - Q(s,a) ]

That makes SARSA **on-policy**: it learns how good the policy it is following
really is. Q-Learning instead uses `max_a' Q(s',a')`, the value of behaving
perfectly from the next state onwards, and so is **off-policy**.

In this room the difference is visible. The lower span is worth more than the
upper walkway when walked flawlessly, so Q-Learning has every reason to prefer
it. But an epsilon-greedy walk along the span occasionally steps off it into the
shaft, and SARSA's target contains exactly those steps, so the span's value comes
out lower and SARSA takes the long way round.

Q-Learning is implemented here too, purely so that claim can be measured on the
same room rather than asserted. Room 3 is where Q-Learning is the subject.

No reinforcement learning library is used. The style is deliberately plain:
`q[state][action]` is a dictionary of dictionaries, and the update is one line.
"""

import time

from core import actions
from rooms.room2 import map_data

MAX_STEPS_DEFAULT = 200

# Default hyperparameters. All adjustable from the interface.
ALPHA_DEFAULT = 0.10
GAMMA_DEFAULT = 0.99
EPSILON_START_DEFAULT = 1.00
EPSILON_MIN_DEFAULT = 0.10
EPSILON_DECAY_DEFAULT = 0.995
EPISODES_DEFAULT = 1500
Q_INIT_DEFAULT = 120.0

SARSA = "SARSA"
Q_LEARNING = "Q-Learning"


def new_q_table(env, initial_value=0.0):
    """A fresh table: q[state][action], every entry set to `initial_value`.

    Starting the table high is called *optimistic initialisation*, and this room
    needs it. The safe upper walkway can be walked by accident, so an agent that
    starts from zeros finds it almost immediately, locks on, and then never has a
    reason to try the lower span again — the span was entered 17 times in 2000
    episodes in testing, far too few for the exit reward to travel back along it.

    Setting every entry above any reward the room can actually pay makes each
    untried action look better than any tried one, so the agent works its way
    through all of them before settling. Both routes then get evaluated properly,
    and the two methods can genuinely disagree about which is better.
    """
    return {state: {action: initial_value for action in env.actions()}
            for state in env.all_states()}


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


def best_actions(q, state, action_list, tolerance=1e-12):
    """Every action tied for the highest value in this state."""
    highest = q[state][best_action(q, state, action_list)]
    return [action for action in action_list
            if q[state][action] >= highest - tolerance]


def epsilon_greedy(q, state, action_list, epsilon, rng):
    """Mostly the best known action, but `epsilon` of the time a random one.

    The random steps are what make this room dangerous, and what SARSA takes into
    account.

    Ties are broken *randomly*, which matters more here than it looks. A fresh
    table is all zeros, so every action is tied at the start; always taking the
    first one would march the robot straight up the west wall on every early
    episode and the lower span would never be explored at all. Choosing randomly
    among equals keeps the search honest.
    """
    if rng.random() < epsilon:
        return rng.choice(action_list)
    tied = best_actions(q, state, action_list)
    if len(tied) == 1:
        return tied[0]
    return rng.choice(tied)


def greedy_policy(q, env):
    """Read the learned policy off the table, with no exploration left in it."""
    action_list = env.actions()
    policy = {}
    for state in env.all_states():
        policy[state] = None if env.is_terminal(state) else best_action(
            q, state, action_list)
    return policy


def mean_absolute_q(q):
    """The average size of the entries in the table.

    Used as the convergence signal: while learning is still moving, this climbs;
    once the values settle, it flattens out.
    """
    total = 0.0
    count = 0
    for actions_for_state in q.values():
        for value in actions_for_state.values():
            total += abs(value)
            count += 1
    return total / count if count else 0.0


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(env, algorithm=SARSA, alpha=ALPHA_DEFAULT, gamma=GAMMA_DEFAULT,
          epsilon_start=EPSILON_START_DEFAULT, epsilon_min=EPSILON_MIN_DEFAULT,
          epsilon_decay=EPSILON_DECAY_DEFAULT, episodes=EPISODES_DEFAULT,
          max_steps=MAX_STEPS_DEFAULT, q_init=Q_INIT_DEFAULT, seed=0,
          progress=None):
    """Train for `episodes` episodes and record everything the graphs need.

    `algorithm` is SARSA or Q_LEARNING. The two differ in exactly one line — the
    target used in the update — which is marked below.
    """
    import random

    rng = random.Random(seed)
    q = new_q_table(env, initial_value=q_init)
    action_list = env.actions()

    history = {
        "reward": [], "length": [], "success": [], "epsilon": [],
        "mean_abs_q": [], "pit_fall": [], "used_span": [], "keycard": [],
        "elapsed": [],
    }
    example_update = None
    epsilon = epsilon_start
    started_at = time.perf_counter()

    for episode in range(episodes):
        state = env.reset(seed=rng.randrange(1_000_000_000))
        action = epsilon_greedy(q, state, action_list, epsilon, rng)

        total_reward = 0.0
        steps = 0
        fell_in_pit = False
        crossed_span = False
        took_keycard = False
        escaped = False

        for _ in range(max_steps):
            steps += 1
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            if info["pit_fall"]:
                fell_in_pit = True
            if info["bridge_collapsed"]:
                crossed_span = True
            if info["keycard_collected"]:
                took_keycard = True
            if info["reached_exit"]:
                escaped = True

            # ---- the one line that separates the two methods ----------
            if done:
                # A finished episode has no next action and no future value.
                next_action = None
                target = reward
            elif algorithm == Q_LEARNING:
                # Off-policy: assume the best possible action comes next.
                next_action = epsilon_greedy(q, next_state, action_list, epsilon, rng)
                target = reward + gamma * q[next_state][
                    best_action(q, next_state, action_list)]
            else:
                # On-policy: use the action actually going to be taken next.
                next_action = epsilon_greedy(q, next_state, action_list, epsilon, rng)
                target = reward + gamma * q[next_state][next_action]

            before = q[state][action]
            q[state][action] = before + alpha * (target - before)

            # Keep one worked example from the last episode, so the interface can
            # show a real update with real numbers instead of a made-up one.
            if episode == episodes - 1 and example_update is None:
                example_update = {
                    "state": state, "action": action, "reward": reward,
                    "next_state": next_state, "next_action": next_action,
                    "q_before": before, "q_after": q[state][action],
                    "target": target, "alpha": alpha, "gamma": gamma,
                    "next_q": 0.0 if done else q[next_state][next_action or action_list[0]],
                    "done": done, "algorithm": algorithm,
                }

            if done:
                break
            state, action = next_state, next_action

        # Decay exploration towards its floor, once per episode.
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        history["reward"].append(total_reward)
        history["length"].append(steps)
        history["success"].append(1.0 if escaped else 0.0)
        history["epsilon"].append(epsilon)
        history["mean_abs_q"].append(mean_absolute_q(q))
        history["pit_fall"].append(1.0 if fell_in_pit else 0.0)
        history["used_span"].append(1.0 if crossed_span else 0.0)
        history["keycard"].append(1.0 if took_keycard else 0.0)
        history["elapsed"].append(time.perf_counter() - started_at)

        if progress is not None and episodes >= 20:
            if (episode + 1) % max(1, episodes // 20) == 0:
                progress((episode + 1) / episodes)

    runtime = time.perf_counter() - started_at

    return {
        "algorithm": algorithm,
        "q": q,
        "policy": greedy_policy(q, env),
        "episodes": episodes,
        "runtime_seconds": runtime,
        "history": history,
        "example_update": example_update,
        "final_epsilon": epsilon,
        "mean_abs_q": mean_absolute_q(q),
        "hyperparameters": {
            "alpha": alpha, "gamma": gamma, "epsilon_start": epsilon_start,
            "epsilon_min": epsilon_min, "epsilon_decay": epsilon_decay,
            "episodes": episodes, "max_steps": max_steps, "q_init": q_init,
            "seed": seed,
        },
        "environment_parameters": env.parameter_summary(),
        "map_hash": map_data.map_hash(),
    }


# ----------------------------------------------------------------------
# Reading the result
# ----------------------------------------------------------------------

def describe_learned_route(q, env):
    """Which way the learned policy actually goes.

    Walks the greedy policy from the start and reports the route by name, which
    is the clearest single summary of what the agent decided.
    """
    policy = greedy_policy(q, env)
    state = env.reset()
    visited = set()
    used_span = False

    for _ in range(MAX_STEPS_DEFAULT):
        action = policy.get(state)
        if action is None:
            break
        state, _, done, info = env.step(action)
        cell = (state[0], state[1])
        if cell in visited and not info["blocked"]:
            break                      # going round in circles
        visited.add(cell)
        if info["bridge_collapsed"]:
            used_span = True
        if done:
            if info["reached_exit"]:
                return "Lower span" if used_span else "Upper walkway"
            if info["pit_fall"]:
                return "Fell into the shaft"
            break

    return "No route found"


def moving_average(values, window):
    """A simple trailing average, used by several of the graphs."""
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
