"""Q-Learning for Room 3, written out by hand.

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

The `max` is the whole difference from SARSA. Q-Learning bootstraps from the best
action available in the next state, whether or not that is the action it is
actually going to take. It therefore learns the value of behaving optimally while
still exploring, which is what **off-policy** means.

That is the right tool for this chamber. The reward for escaping sits 23 steps away
from the first move that leads towards it, behind three generators that only count
in the right order. Every one of those steps has to be worth something because of
what it eventually leads to, and bootstrapping from the best next action is how
that value travels backwards through the sequence.

SARSA is implemented here too — the same loop with one different line — so the
comparison in the educational section is a measurement rather than a claim.

No reinforcement learning library is used. The shared table helpers live in
`core/tabular.py`.
"""

import random
import time

from core import tabular
from rooms.room3 import map_data
from rooms.room3.environment import STAGE_UNLOCKED

Q_LEARNING = "Q-Learning"
SARSA = "SARSA"

MAX_STEPS_DEFAULT = 300

# Defaults. All adjustable from the interface.
ALPHA_DEFAULT = 0.15
GAMMA_DEFAULT = 0.98
EPSILON_START_DEFAULT = 1.00
EPSILON_MIN_DEFAULT = 0.05
EPSILON_DECAY_DEFAULT = 0.999
EPISODES_DEFAULT = 5000
Q_INIT_DEFAULT = 0.0

# Convenience re-exports, so the room reads as one module.
epsilon_greedy = tabular.epsilon_greedy
best_action = tabular.best_action
best_value = tabular.best_value
mean_absolute_q = tabular.mean_absolute_q
moving_average = tabular.moving_average


def new_q_table(env, initial_value=Q_INIT_DEFAULT):
    return tabular.new_q_table(env.all_states(), env.actions(), initial_value)


def greedy_policy(q, env):
    return tabular.greedy_policy(q, env.all_states(), env.actions(),
                                env.is_terminal)


def train(env, algorithm=Q_LEARNING, alpha=ALPHA_DEFAULT, gamma=GAMMA_DEFAULT,
          epsilon_start=EPSILON_START_DEFAULT, epsilon_min=EPSILON_MIN_DEFAULT,
          epsilon_decay=EPSILON_DECAY_DEFAULT, episodes=EPISODES_DEFAULT,
          max_steps=MAX_STEPS_DEFAULT, q_init=Q_INIT_DEFAULT, seed=0,
          progress=None):
    """Train for `episodes` episodes and record everything the graphs need."""
    rng = random.Random(seed)
    action_list = env.actions()
    q = new_q_table(env, q_init)

    history = {
        "reward": [], "length": [], "success": [], "epsilon": [],
        "mean_abs_q": [], "elapsed": [], "stage": [], "generators_done": [],
        "guard_caught": [], "hazards": [], "door_passed": [], "waits": [],
    }
    example_update = None
    epsilon = epsilon_start
    started_at = time.perf_counter()

    for episode in range(episodes):
        state = env.reset(seed=rng.randrange(1_000_000_000))
        total_reward = 0.0
        steps = 0
        best_stage = 0
        hazards = 0
        waits = 0
        caught = False
        escaped = False
        passed_door = False

        for _ in range(max_steps):
            steps += 1
            action = tabular.epsilon_greedy(q, state, action_list, epsilon, rng)
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            best_stage = max(best_stage, next_state[2])
            if info["hazard"]:
                hazards += 1
            if info["waited"]:
                waits += 1
            if info["caught"]:
                caught = True
            if info["reached_exit"]:
                escaped = True
            if (next_state[0], next_state[1]) == map_data.reactor_door_cell():
                passed_door = True

            # ---- the one line that separates the two methods -----------
            if done:
                # No next state to bootstrap from.
                target = reward
            elif algorithm == SARSA:
                # On-policy: the action actually going to be taken next.
                next_action = tabular.epsilon_greedy(q, next_state, action_list,
                                                     epsilon, rng)
                target = reward + gamma * q[next_state][next_action]
            else:
                # Off-policy: the best action available next, taken or not.
                target = reward + gamma * tabular.best_value(q, next_state,
                                                             action_list)

            before = q[state][action]
            q[state][action] = before + alpha * (target - before)

            if episode == episodes - 1 and example_update is None:
                example_update = {
                    "state": state, "action": action, "reward": reward,
                    "next_state": next_state,
                    "q_before": before, "q_after": q[state][action],
                    "target": target, "alpha": alpha, "gamma": gamma,
                    "max_next_q": (0.0 if done else
                                   tabular.best_value(q, next_state, action_list)),
                    "best_next_action": (None if done else
                                         tabular.best_action(q, next_state,
                                                             action_list)),
                    "done": done, "algorithm": algorithm,
                }

            if done:
                break
            state = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        history["reward"].append(total_reward)
        history["length"].append(steps)
        history["success"].append(1.0 if escaped else 0.0)
        history["epsilon"].append(epsilon)
        history["mean_abs_q"].append(tabular.mean_absolute_q(q))
        history["stage"].append(float(best_stage))
        history["generators_done"].append(1.0 if best_stage >= STAGE_UNLOCKED else 0.0)
        history["guard_caught"].append(1.0 if caught else 0.0)
        history["hazards"].append(float(hazards))
        history["door_passed"].append(1.0 if passed_door else 0.0)
        history["waits"].append(float(waits))
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
        "mean_abs_q": tabular.mean_absolute_q(q),
        "hyperparameters": {
            "alpha": alpha, "gamma": gamma, "epsilon_start": epsilon_start,
            "epsilon_min": epsilon_min, "epsilon_decay": epsilon_decay,
            "episodes": episodes, "max_steps": max_steps, "q_init": q_init,
            "seed": seed,
        },
        "environment_parameters": env.parameter_summary(),
        "map_hash": map_data.map_hash(),
    }
