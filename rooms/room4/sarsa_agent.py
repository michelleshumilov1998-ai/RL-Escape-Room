"""Semi-Gradient SARSA with linear function approximation, written by hand.

--------------------------------------------------------------------------
THE VALUE FUNCTION
--------------------------------------------------------------------------
    Q(s, a) = w · x(s, a)

`x(s, a)` is the sparse tile-coded feature vector from `tile_coder.py`. It has one
entry per weight but only `num_tilings` of them are 1, so the dot product is just a
sum over those few indices — no dense vectors are ever built.

--------------------------------------------------------------------------
THE UPDATE
--------------------------------------------------------------------------
    delta = r + gamma * Q(s', a') - Q(s, a)          (or  r - Q(s,a)  if terminal)
    w    += alpha_effective * delta * x(s, a)

Because the features are 0 or 1, `alpha * delta * x` only touches the active
indices, and each one moves by exactly `alpha_effective * delta`.

--------------------------------------------------------------------------
WHY "SEMI-GRADIENT"
--------------------------------------------------------------------------
The target `r + gamma * Q(s', a')` contains the weights too, so a true gradient
would have to differentiate through it as well. Semi-gradient methods deliberately
do not: the target is treated as a fixed number and only the gradient of the
current estimate `Q(s,a)` is used — which, for a linear model, is just `x(s,a)`.
That is what makes the update the simple line above, and it is the standard
approach for this family of methods.

--------------------------------------------------------------------------
SARSA OR Q-LEARNING
--------------------------------------------------------------------------
Semi-Gradient SARSA is this room's method. Semi-Gradient Q-Learning is implemented
too — the same code with `max_a' Q(s',a')` in place of `Q(s',a')` — so the two can
be compared on identical features, rewards, episodes and seeds.

No reinforcement learning library is used anywhere.
"""

import math
import random
import time

from rooms.room4 import chamber
from rooms.room4.environment import ACTIONS, Room4Env
from rooms.room4.tile_coder import TileCoder

SARSA = "Semi-Gradient SARSA"
Q_LEARNING = "Semi-Gradient Q-Learning"

# Defaults, all adjustable from the interface.
ALPHA_DEFAULT = 0.30
GAMMA_DEFAULT = 0.99
EPSILON_START_DEFAULT = 1.00
EPSILON_MIN_DEFAULT = 0.02
EPSILON_DECAY_DEFAULT = 0.995
EPISODES_DEFAULT = 1200
NUM_TILINGS_DEFAULT = 8
TILES_PER_DIMENSION_DEFAULT = 8
MAX_STEPS_DEFAULT = 220


class SemiGradientAgent:
    """A linear value function over tile-coded features."""

    def __init__(self, env, num_tilings=NUM_TILINGS_DEFAULT,
                 tiles_per_dimension=TILES_PER_DIMENSION_DEFAULT,
                 alpha=ALPHA_DEFAULT, gamma=GAMMA_DEFAULT):
        self.coder = TileCoder(env.state_bounds(), num_tilings,
                               tiles_per_dimension, len(env.actions()))
        self.actions = list(env.actions())
        self.gamma = gamma
        # Every active feature pushes the same prediction, so the step size is
        # shared out between them. See the note in tile_coder.py.
        self.alpha = alpha
        self.effective_alpha = alpha / self.coder.num_tilings

        self.weights = [0.0] * self.coder.total_features
        # The norm is wanted once per episode over 160k weights, which is far too
        # slow to recompute. Since an update touches only a handful of them, the
        # sum of squares is kept up to date as they change instead.
        self._sum_squares = 0.0

    # ------------------------------------------------------------------

    def q_value(self, state, action):
        """w · x(s, a), summed over the active tiles only."""
        total = 0.0
        for index in self.coder.active_tiles(state, action):
            total += self.weights[index]
        return total

    def all_q_values(self, state):
        return [self.q_value(state, action) for action in self.actions]

    def best_action(self, state):
        """The action with the highest estimate; ties go to the first."""
        values = self.all_q_values(state)
        best = 0
        for index in range(1, len(values)):
            if values[index] > values[best]:
                best = index
        return self.actions[best]

    def max_q_value(self, state):
        return max(self.all_q_values(state))

    def epsilon_greedy(self, state, epsilon, rng):
        if rng.random() < epsilon:
            return rng.choice(self.actions)
        values = self.all_q_values(state)
        highest = max(values)
        tied = [self.actions[index] for index, value in enumerate(values)
                if value >= highest - 1e-12]
        return tied[0] if len(tied) == 1 else rng.choice(tied)

    # ------------------------------------------------------------------

    def update(self, state, action, target):
        """Move the active weights towards `target`. Returns the TD error."""
        indices = self.coder.active_tiles(state, action)
        current = 0.0
        for index in indices:
            current += self.weights[index]

        delta = target - current
        change = self.effective_alpha * delta
        for index in indices:
            before = self.weights[index]
            after = before + change
            self.weights[index] = after
            self._sum_squares += after * after - before * before
        return delta

    def weight_norm(self):
        """||w||, kept up to date rather than recomputed."""
        return math.sqrt(max(0.0, self._sum_squares))

    def active_feature_count(self):
        """How many weights are no longer zero — how much of the space is used."""
        return sum(1 for weight in self.weights if weight != 0.0)

    def configuration(self):
        return self.coder.configuration()

    def is_finite(self):
        """No NaN or infinity anywhere, which divergence would produce."""
        return all(math.isfinite(weight) for weight in self.weights)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(env=None, algorithm=SARSA, alpha=ALPHA_DEFAULT, gamma=GAMMA_DEFAULT,
          epsilon_start=EPSILON_START_DEFAULT, epsilon_min=EPSILON_MIN_DEFAULT,
          epsilon_decay=EPSILON_DECAY_DEFAULT, episodes=EPISODES_DEFAULT,
          num_tilings=NUM_TILINGS_DEFAULT,
          tiles_per_dimension=TILES_PER_DIMENSION_DEFAULT,
          max_steps=MAX_STEPS_DEFAULT, seed=0, environment_parameters=None,
          training_starts=None, progress=None):
    """Train for `episodes` episodes and record everything the graphs need.

    `training_starts` is the set of release points episodes are drawn from. The
    generalisation test evaluates on points that are deliberately not in it.
    """
    env = env or Room4Env(**(environment_parameters or {}))
    starts = tuple(training_starts or chamber.TRAINING_STARTS)

    rng = random.Random(seed)
    agent = SemiGradientAgent(env, num_tilings, tiles_per_dimension, alpha, gamma)

    history = {
        "reward": [], "length": [], "success": [], "epsilon": [],
        "final_distance": [], "landing_speed": [], "collisions": [],
        "weight_norm": [], "td_error": [], "active_features": [],
        "hard_landing": [], "crash": [], "timeout": [], "elapsed": [],
    }
    example_update = None
    epsilon = epsilon_start
    started_at = time.perf_counter()

    for episode in range(episodes):
        start = starts[rng.randrange(len(starts))]
        state = env.reset(seed=rng.randrange(1_000_000_000), start_position=start)
        action = agent.epsilon_greedy(state, epsilon, rng)

        total_reward = 0.0
        steps = 0
        collisions = 0
        td_sum = 0.0
        landed = False
        hard = False
        crashed = False
        timed_out = False
        landing_speed = None

        for _ in range(max_steps):
            steps += 1
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            if info["collision"] or info["boundary"]:
                collisions += 1
            if info["hard_landing"]:
                hard = True
                landing_speed = info["speed"]
            if info["safe_landing"]:
                landed = True
                landing_speed = info["speed"]
            if info["crashed"]:
                crashed = True
                landing_speed = info["speed"]
            if info["timeout"]:
                timed_out = True

            # ---- the target, and the one line that differs -------------
            if done:
                # Terminal: no bootstrap term at all.
                target = reward
                next_action = None
            elif algorithm == Q_LEARNING:
                next_action = agent.epsilon_greedy(next_state, epsilon, rng)
                target = reward + gamma * agent.max_q_value(next_state)
            else:
                next_action = agent.epsilon_greedy(next_state, epsilon, rng)
                target = reward + gamma * agent.q_value(next_state, next_action)

            if episode == episodes - 1 and example_update is None:
                indices = agent.coder.active_tiles(state, action)
                before = agent.q_value(state, action)
                example_update = {
                    "state": state, "action": action, "reward": reward,
                    "next_state": next_state, "next_action": next_action,
                    "q_before": before,
                    "next_q": (0.0 if done else
                               (agent.max_q_value(next_state)
                                if algorithm == Q_LEARNING
                                else agent.q_value(next_state, next_action))),
                    "target": target, "td_error": target - before,
                    "active_tiles": list(indices),
                    "alpha": alpha,
                    "effective_alpha": agent.effective_alpha,
                    "weight_change": agent.effective_alpha * (target - before),
                    "gamma": gamma, "done": done, "algorithm": algorithm,
                }

            td_sum += abs(agent.update(state, action, target))

            if done:
                break
            state, action = next_state, next_action

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        history["reward"].append(total_reward)
        history["length"].append(steps)
        history["success"].append(1.0 if landed else 0.0)
        history["epsilon"].append(epsilon)
        history["final_distance"].append(
            chamber.distance_to_goal(env.state[0], env.state[1]))
        history["landing_speed"].append(
            landing_speed if landing_speed is not None else float("nan"))
        history["collisions"].append(float(collisions))
        history["weight_norm"].append(agent.weight_norm())
        history["td_error"].append(td_sum / max(1, steps))
        history["active_features"].append(float(agent.active_feature_count()))
        history["hard_landing"].append(1.0 if hard else 0.0)
        history["crash"].append(1.0 if crashed else 0.0)
        history["timeout"].append(1.0 if timed_out else 0.0)
        history["elapsed"].append(time.perf_counter() - started_at)

        if progress is not None and episodes >= 20:
            if (episode + 1) % max(1, episodes // 20) == 0:
                progress((episode + 1) / episodes)

    runtime = time.perf_counter() - started_at

    return {
        "algorithm": algorithm,
        "agent": agent,
        "episodes": episodes,
        "runtime_seconds": runtime,
        "history": history,
        "example_update": example_update,
        "final_epsilon": epsilon,
        "weight_norm": agent.weight_norm(),
        "active_features": agent.active_feature_count(),
        "tile_coder": agent.configuration(),
        "hyperparameters": {
            "alpha": alpha, "gamma": gamma, "epsilon_start": epsilon_start,
            "epsilon_min": epsilon_min, "epsilon_decay": epsilon_decay,
            "episodes": episodes, "num_tilings": num_tilings,
            "tiles_per_dimension": tiles_per_dimension, "max_steps": max_steps,
            "seed": seed,
        },
        "environment_parameters": env.parameter_summary(),
        "training_starts": [list(start) for start in starts],
    }


def moving_average(values, window):
    """A trailing average that tolerates the NaNs in the landing-speed series."""
    if window <= 1 or not values:
        return list(values)
    averaged = []
    kept = []
    for value in values:
        if value == value:          # NaN fails this comparison
            kept.append(value)
        if len(kept) > window:
            kept.pop(0)
        averaged.append(sum(kept) / len(kept) if kept else float("nan"))
    return averaged
