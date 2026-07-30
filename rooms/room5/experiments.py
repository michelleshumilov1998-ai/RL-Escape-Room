"""Experiments for Room 5, all of them about generalisation.

The central measurement is train / validation / **unseen test**, on layouts from
separate seed ranges. Nothing in training ever touches the test seeds, and the
tests assert it.

Everything is reported as mean ± standard deviation across seeds, and a sweep is
only ever described as the best-performing configuration among those tested.
"""

from rooms.room5 import layout as L, q_agent, simulate
from rooms.room5.features import FEATURE_SETS

SWEEPABLE_PARAMETERS = {
    "alpha": [0.03, 0.10, 0.30],
    "gamma": [0.90, 0.97, 1.00],
    "epsilon_decay": [0.995, 0.999, 0.9995],
    "training_layouts": [5, 20, 50],
}

READABLE_NAMES = {
    "alpha": "Learning rate (alpha)",
    "gamma": "Discount factor (gamma)",
    "epsilon_decay": "Exploration decay",
    "training_layouts": "Number of training layouts",
}

RADAR_RANGES = {"Short radar (2)": 2, "Medium radar (4)": 4, "Long radar (7)": 7}
LAYOUT_COUNTS = (5, 20, 50)
ROBOT_COUNTS = (0, 1, 2, 3)

REPEATS_DEFAULT = 2
EPISODES_DEFAULT = 2000
TEST_LAYOUTS = 12
VALIDATION_LAYOUTS = 8
TAIL_FRACTION = 0.2


def _tail_mean(values):
    numbers = [value for value in values if value == value]
    if not numbers:
        return 0.0
    count = max(1, int(len(numbers) * TAIL_FRACTION))
    return sum(numbers[-count:]) / len(numbers[-count:])


def run_configuration(hyperparameters=None, environment_parameters=None,
                      algorithm=q_agent.Q_LEARNING, episodes=EPISODES_DEFAULT,
                      repeats=REPEATS_DEFAULT, difficulty=L.DEFAULT_DIFFICULTY,
                      feature_set=None, test_difficulty=None,
                      layout_overrides=None):
    """Train one configuration `repeats` times and evaluate all three splits."""
    hyperparameters = dict(hyperparameters or {})
    environment_parameters = dict(environment_parameters or {})
    collected = {key: [] for key in
                 ("train_reward", "train_success", "train_caught", "runtime",
                  "weight_norm", "training_success", "validation_success",
                  "test_success", "test_return", "test_efficiency",
                  "test_collisions", "test_caught", "test_timeout")}

    for repeat in range(repeats):
        result = q_agent.train(
            algorithm=algorithm, episodes=episodes, seed=repeat,
            difficulty=difficulty,
            feature_set=feature_set or q_agent.DEFAULT_FEATURE_SET,
            environment_parameters=environment_parameters,
            validation_every=0, layout_overrides=layout_overrides,
            **hyperparameters)
        history = result["history"]
        collected["train_reward"].append(_tail_mean(history["reward"]))
        collected["train_success"].append(_tail_mean(history["success"]))
        collected["train_caught"].append(_tail_mean(history["caught"]))
        collected["runtime"].append(result["runtime_seconds"])
        collected["weight_norm"].append(result["weight_norm"])

        # The evaluation difficulty may differ from the training one, which is what
        # the stress test uses.
        evaluation_difficulty = test_difficulty or difficulty
        pools = {
            "training_success": L.generate_pool(
                "training", hyperparameters.get("training_layouts",
                                                q_agent.TRAINING_LAYOUTS_DEFAULT),
                difficulty, layout_overrides),
            "validation_success": L.generate_pool("validation",
                                                  VALIDATION_LAYOUTS,
                                                  evaluation_difficulty,
                                                  layout_overrides),
            "test_success": L.generate_pool("test", TEST_LAYOUTS,
                                            evaluation_difficulty,
                                            layout_overrides),
        }
        for key, pool in pools.items():
            outcome = simulate.run_many(result["agent"], pool,
                                        environment_parameters)
            collected[key].append(outcome["success_rate"])
            if key == "test_success":
                collected["test_return"].append(outcome["mean_return"])
                collected["test_efficiency"].append(outcome["path_efficiency"])
                collected["test_collisions"].append(outcome["mean_collisions"])
                collected["test_caught"].append(outcome["caught_rate"])
                collected["test_timeout"].append(outcome["timeout_rate"])

    summary = {"algorithm": algorithm, "repeats": repeats, "episodes": episodes,
               "difficulty": difficulty,
               "feature_set": feature_set or q_agent.DEFAULT_FEATURE_SET}
    for key, values in collected.items():
        summary["mean_" + key] = _mean(values)
        summary["std_" + key] = _std(values)
    return summary


def parameter_experiment(parameter_name, values=None, **kwargs):
    values = values if values is not None else SWEEPABLE_PARAMETERS[parameter_name]
    rows = []
    for value in values:
        hyperparameters = dict(kwargs.pop("base_hyperparameters", None) or {})
        hyperparameters[parameter_name] = value
        row = run_configuration(hyperparameters=hyperparameters, **kwargs)
        row["parameter"] = READABLE_NAMES.get(parameter_name, parameter_name)
        row["value"] = value
        rows.append(row)
    return rows


def feature_set_experiment(episodes=EPISODES_DEFAULT, repeats=1, **kwargs):
    """Which features actually matter, from target-only to the full set."""
    rows = []
    for name in FEATURE_SETS:
        row = run_configuration(feature_set=name, episodes=episodes,
                                repeats=repeats, **kwargs)
        row["parameter"] = "feature set"
        row["value"] = name
        rows.append(row)
    return rows


def radar_range_experiment(episodes=EPISODES_DEFAULT, repeats=1, **kwargs):
    """More information against more features to learn about."""
    rows = []
    for label, radar_range in RADAR_RANGES.items():
        row = run_configuration(
            environment_parameters={"radar_range": radar_range},
            layout_overrides={"radar_range": radar_range},
            episodes=episodes, repeats=repeats, **kwargs)
        row["parameter"] = "radar range"
        row["value"] = label
        rows.append(row)
    return rows


def layout_count_experiment(episodes=EPISODES_DEFAULT, repeats=1, **kwargs):
    """The headline experiment: how layout variety affects generalisation.

    Every agent is evaluated on the same unseen test layouts, so the only thing
    changing is how much variety it saw while training.
    """
    rows = []
    for count in LAYOUT_COUNTS:
        row = run_configuration(hyperparameters={"training_layouts": count},
                                episodes=episodes, repeats=repeats, **kwargs)
        row["parameter"] = "training layouts"
        row["value"] = count
        rows.append(row)
    return rows


def robot_count_experiment(episodes=EPISODES_DEFAULT, repeats=1, **kwargs):
    rows = []
    for count in ROBOT_COUNTS:
        row = run_configuration(layout_overrides={"robots": count},
                                episodes=episodes, repeats=repeats, **kwargs)
        row["parameter"] = "maintenance robots"
        row["value"] = count
        rows.append(row)
    return rows


def difficulty_experiment(episodes=EPISODES_DEFAULT, repeats=1, **kwargs):
    rows = []
    for difficulty in L.DIFFICULTIES:
        row = run_configuration(difficulty=difficulty, episodes=episodes,
                                repeats=repeats, **kwargs)
        row["parameter"] = "difficulty"
        row["value"] = difficulty
        rows.append(row)
    return rows


def generalisation_stress_test(episodes=EPISODES_DEFAULT, repeats=1, **kwargs):
    """Train on Easy, then evaluate on Medium and Hard layouts it never saw.

    This is deliberately a distribution shift, not just unseen layouts, and it is
    labelled as such. If the success rate collapses, that is the result.
    """
    rows = []
    for evaluation in L.DIFFICULTIES:
        row = run_configuration(difficulty="Easy", test_difficulty=evaluation,
                                episodes=episodes, repeats=repeats, **kwargs)
        row["parameter"] = "evaluation difficulty"
        row["value"] = "trained Easy, tested %s" % evaluation
        row["distribution_shift"] = evaluation != "Easy"
        rows.append(row)
    return rows


def compare_algorithms(episodes=EPISODES_DEFAULT, repeats=REPEATS_DEFAULT,
                       **kwargs):
    rows = []
    for algorithm in (q_agent.Q_LEARNING, q_agent.SARSA):
        row = run_configuration(algorithm=algorithm, episodes=episodes,
                                repeats=repeats, **kwargs)
        row["parameter"] = "algorithm"
        row["value"] = algorithm
        rows.append(row)
    return rows


def baseline_policies(layouts=None, difficulty=L.DEFAULT_DIFFICULTY):
    """Two reference policies, to show what the learned one is beating.

      * random           — uniform over the five actions
      * greedy target    — always step towards the objective, ignoring obstacles
    """
    import random as _random

    pool = layouts or L.generate_pool("test", TEST_LAYOUTS, difficulty)

    class RandomPolicy:
        def __init__(self):
            self.rng = _random.Random(0)

        def best_action(self, observation):
            return self.rng.randrange(5)

    class GreedyTargetPolicy:
        """Steps whichever way reduces the gap most. No obstacle avoidance."""

        def best_action(self, observation):
            delta_row, delta_col, _ = observation["target"]
            if abs(delta_row) >= abs(delta_col):
                return 1 if delta_row > 0 else 0        # DOWN / UP
            return 3 if delta_col > 0 else 2            # RIGHT / LEFT

    rows = []
    for label, policy in (("Random policy", RandomPolicy()),
                          ("Greedy target policy", GreedyTargetPolicy())):
        outcome = simulate.run_many(policy, pool)
        rows.append({
            "parameter": "baseline", "value": label,
            "mean_test_success": outcome["success_rate"],
            "std_test_success": 0.0,
            "mean_test_return": outcome["mean_return"],
            "mean_test_efficiency": outcome["path_efficiency"],
            "mean_test_caught": outcome["caught_rate"],
            "mean_test_timeout": outcome["timeout_rate"],
            "mean_training_success": float("nan"),
            "mean_validation_success": float("nan"),
        })
    return rows


def best_performing_row(rows, metric="mean_test_success"):
    if not rows:
        return None
    best = rows[0]
    for row in rows[1:]:
        if row.get(metric, float("-inf")) > best.get(metric, float("-inf")):
            best = row
    return best


def _mean(numbers):
    numbers = [value for value in numbers if value == value]
    return sum(numbers) / len(numbers) if numbers else 0.0


def _std(numbers):
    numbers = [value for value in numbers if value == value]
    if len(numbers) < 2:
        return 0.0
    average = _mean(numbers)
    return (sum((number - average) ** 2 for number in numbers) / len(numbers)) ** 0.5
