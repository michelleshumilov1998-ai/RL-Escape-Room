"""Semi-Gradient Q-Learning with linear function approximation, written by hand.

    Q(s,a) = w · x(s,a)

    target = r                                    if the episode ended
           = r + gamma * max_a' Q(s',a')          otherwise
    delta  = target - Q(s,a)
    w     += alpha * delta * x(s,a)

The `max` makes it **off-policy**: the target uses the best action available next,
whether or not that is the action about to be taken, so the values learned are the
values of behaving greedily even while the behaviour is still exploring.

The difference from Room 4 is what the features describe. There, they were a
position and a velocity in one fixed hall. Here they are *local* — what the radar
can see and where the objective is — with nothing about which warehouse this is. So
the same weights mean something in a layout the agent has never entered, which is
the point of the room.

Semi-Gradient SARSA is included for comparison, on identical features.

No reinforcement learning library is used.
"""

import math
import random
import time

from rooms.room5 import layout as L
from rooms.room5.environment import ACTIONS, Room5Env
from rooms.room5.features import DEFAULT_FEATURE_SET, FEATURE_SETS, FeatureExtractor

Q_LEARNING = "Semi-Gradient Q-Learning"
SARSA = "Semi-Gradient SARSA"

ALPHA_DEFAULT = 0.10
GAMMA_DEFAULT = 0.97
EPSILON_START_DEFAULT = 1.00
EPSILON_MIN_DEFAULT = 0.05
EPSILON_DECAY_DEFAULT = 0.999
EPISODES_DEFAULT = 2500
TRAINING_LAYOUTS_DEFAULT = 20
VALIDATION_LAYOUTS_DEFAULT = 8
TEST_LAYOUTS_DEFAULT = 12
VALIDATION_EVERY_DEFAULT = 250


class LinearAgent:
    """A linear value function over the sparse local features."""

    def __init__(self, extractor=None):
        self.extractor = extractor or FeatureExtractor()
        self.actions = list(ACTIONS)
        self.weights = [0.0] * self.extractor.total_features
        self._sum_squares = 0.0

    def q_value(self, observation, action):
        total = 0.0
        for index in self.extractor.active_features(observation, action):
            total += self.weights[index]
        return total

    def all_q_values(self, observation):
        return [self.q_value(observation, action) for action in self.actions]

    def best_action(self, observation):
        values = self.all_q_values(observation)
        best = 0
        for index in range(1, len(values)):
            if values[index] > values[best]:
                best = index
        return self.actions[best]

    def max_q_value(self, observation):
        return max(self.all_q_values(observation))

    def epsilon_greedy(self, observation, epsilon, rng):
        if rng.random() < epsilon:
            return rng.choice(self.actions)
        values = self.all_q_values(observation)
        highest = max(values)
        tied = [self.actions[index] for index, value in enumerate(values)
                if value >= highest - 1e-12]
        return tied[0] if len(tied) == 1 else rng.choice(tied)

    def update(self, observation, action, target, alpha):
        """Move the active weights towards `target`. Returns the TD error."""
        indices = self.extractor.active_features(observation, action)
        current = 0.0
        for index in indices:
            current += self.weights[index]
        delta = target - current
        # Shared out between the active features, as in Room 4.
        change = alpha * delta / max(1, len(indices))
        for index in indices:
            before = self.weights[index]
            after = before + change
            self.weights[index] = after
            self._sum_squares += after * after - before * before
        return delta

    def weight_norm(self):
        return math.sqrt(max(0.0, self._sum_squares))

    def active_feature_count(self):
        return sum(1 for weight in self.weights if weight != 0.0)

    def is_finite(self):
        return all(math.isfinite(weight) for weight in self.weights)

    def configuration(self):
        return self.extractor.configuration()


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate(agent, layouts, environment_parameters=None, epsilon=0.0):
    """Run the greedy policy once on each layout and summarise.

    Used for the validation curve during training and for the final
    train/validation/test comparison. No learning happens here.
    """
    import random as _random
    rng = _random.Random(0)
    parameters = dict(environment_parameters or {})

    successes = 0
    terminals = 0
    collisions = 0
    caught = 0
    timeouts = 0
    returns = []
    efficiencies = []

    for layout in layouts:
        env = Room5Env(layout=layout, **parameters)
        observation = env.reset(layout)
        total = 0.0
        steps = 0
        for _ in range(env.max_steps):
            action = (rng.choice(env.actions()) if epsilon and rng.random() < epsilon
                      else agent.best_action(observation))
            observation, reward, done, info = env.step(action)
            total += reward
            steps += 1
            if info["collision"]:
                collisions += 1
            if info["terminal_activated"]:
                terminals += 1
            if done:
                if info["reached_exit"]:
                    successes += 1
                    reference = layout.shortest_mission_length()
                    if reference:
                        efficiencies.append(steps / reference)
                if info["caught"]:
                    caught += 1
                if info["timeout"]:
                    timeouts += 1
                break
        returns.append(total)

    count = max(1, len(layouts))
    return {
        "layouts": len(layouts),
        "success_rate": successes / count,
        "terminal_rate": terminals / count,
        "collision_rate": collisions / count,
        "caught_rate": caught / count,
        "timeout_rate": timeouts / count,
        "mean_return": sum(returns) / count if returns else 0.0,
        "path_efficiency": (sum(efficiencies) / len(efficiencies)
                            if efficiencies else float("nan")),
    }


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(algorithm=Q_LEARNING, alpha=ALPHA_DEFAULT, gamma=GAMMA_DEFAULT,
          epsilon_start=EPSILON_START_DEFAULT, epsilon_min=EPSILON_MIN_DEFAULT,
          epsilon_decay=EPSILON_DECAY_DEFAULT, episodes=EPISODES_DEFAULT,
          training_layouts=TRAINING_LAYOUTS_DEFAULT,
          validation_layouts=VALIDATION_LAYOUTS_DEFAULT,
          difficulty=L.DEFAULT_DIFFICULTY, feature_set=DEFAULT_FEATURE_SET,
          environment_parameters=None, validation_every=VALIDATION_EVERY_DEFAULT,
          seed=0, progress=None, layout_overrides=None):
    """Train across a pool of layouts, checking held-out ones as it goes.

    Each episode picks a training layout at random. Validation layouts come from a
    different seed range and are never trained on.
    """
    rng = random.Random(seed)
    extractor = FeatureExtractor(groups=FEATURE_SETS[feature_set])
    agent = LinearAgent(extractor)
    parameters = dict(environment_parameters or {})

    training_pool = L.generate_pool("training", training_layouts, difficulty,
                                    layout_overrides)
    validation_pool = L.generate_pool("validation", validation_layouts, difficulty,
                                      layout_overrides)

    env = Room5Env(layout=training_pool[0], difficulty=difficulty, **parameters)

    history = {
        "reward": [], "length": [], "success": [], "epsilon": [], "terminal": [],
        "collisions": [], "caught": [], "timeout": [], "conveyor": [],
        "td_error": [], "weight_norm": [], "active_features": [], "elapsed": [],
        "validation_episode": [], "validation_success": [],
    }
    example_update = None
    epsilon = epsilon_start
    started_at = time.perf_counter()

    for episode in range(episodes):
        layout = training_pool[rng.randrange(len(training_pool))]
        observation = env.reset(layout)
        action = agent.epsilon_greedy(observation, epsilon, rng)

        total_reward = 0.0
        steps = 0
        collisions = 0
        conveyors = 0
        td_sum = 0.0
        activated = False
        escaped = False
        caught = False
        timed_out = False

        for _ in range(env.max_steps):
            steps += 1
            next_observation, reward, done, info = env.step(action)
            total_reward += reward

            if info["collision"]:
                collisions += 1
            if info["conveyor"]:
                conveyors += 1
            if info["terminal_activated"]:
                activated = True
            if info["reached_exit"]:
                escaped = True
            if info["caught"]:
                caught = True
            if info["timeout"]:
                timed_out = True

            # ---- the target, and the one line that differs -------------
            if done:
                target = reward
                next_action = None
            elif algorithm == SARSA:
                next_action = agent.epsilon_greedy(next_observation, epsilon, rng)
                target = reward + gamma * agent.q_value(next_observation,
                                                        next_action)
            else:
                next_action = agent.epsilon_greedy(next_observation, epsilon, rng)
                target = reward + gamma * agent.max_q_value(next_observation)

            if episode == episodes - 1 and example_update is None:
                indices = extractor.active_features(observation, action)
                before = agent.q_value(observation, action)
                example_update = {
                    "observation": observation, "action": action,
                    "reward": reward,
                    "q_before": before,
                    "max_next_q": (0.0 if done
                                   else agent.max_q_value(next_observation)),
                    "target": target, "td_error": target - before,
                    "active_features": list(indices),
                    "alpha": alpha,
                    "weight_change": alpha * (target - before) / max(1, len(indices)),
                    "gamma": gamma, "done": done, "algorithm": algorithm,
                }

            td_sum += abs(agent.update(observation, action, target, alpha))

            if done:
                break
            observation, action = next_observation, next_action

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        history["reward"].append(total_reward)
        history["length"].append(steps)
        history["success"].append(1.0 if escaped else 0.0)
        history["epsilon"].append(epsilon)
        history["terminal"].append(1.0 if activated else 0.0)
        history["collisions"].append(float(collisions))
        history["conveyor"].append(float(conveyors))
        history["caught"].append(1.0 if caught else 0.0)
        history["timeout"].append(1.0 if timed_out else 0.0)
        history["td_error"].append(td_sum / max(1, steps))
        history["weight_norm"].append(agent.weight_norm())
        history["active_features"].append(float(agent.active_feature_count()))
        history["elapsed"].append(time.perf_counter() - started_at)

        # Periodic check on layouts that are never trained on.
        if validation_every and ((episode + 1) % validation_every == 0
                                 or episode == episodes - 1):
            outcome = evaluate(agent, validation_pool, parameters)
            history["validation_episode"].append(episode + 1)
            history["validation_success"].append(outcome["success_rate"])

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
        "features": extractor.configuration(),
        "feature_set": feature_set,
        "difficulty": difficulty,
        "training_seeds": [layout.seed for layout in training_pool],
        "validation_seeds": [layout.seed for layout in validation_pool],
        "hyperparameters": {
            "alpha": alpha, "gamma": gamma, "epsilon_start": epsilon_start,
            "epsilon_min": epsilon_min, "epsilon_decay": epsilon_decay,
            "episodes": episodes, "training_layouts": training_layouts,
            "validation_layouts": validation_layouts, "seed": seed,
            "feature_set": feature_set, "difficulty": difficulty,
        },
        "environment_parameters": Room5Env(layout=training_pool[0],
                                          difficulty=difficulty,
                                          **parameters).parameter_summary(),
    }


def moving_average(values, window):
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
