"""Parameter experiments and the generalisation test for Room 4.

The headline experiment here is **generalisation**: train from one release point,
then fly from points the agent has never started from. Nothing in the training loop
ever sees the test starts — the split is enforced in `chamber.py` and asserted by
the tests.

Everything is reported as mean ± standard deviation across seeds, and a sweep is
only ever described as the best-performing configuration among those tested.
"""

from rooms.room4 import chamber, sarsa_agent, simulate

SWEEPABLE_PARAMETERS = {
    "alpha": [0.05, 0.15, 0.30, 0.60],
    "gamma": [0.95, 0.98, 0.99, 1.00],
    "epsilon_decay": [0.99, 0.995, 0.998],
    "num_tilings": [4, 8, 12],
    "tiles_per_dimension": [6, 8, 10],
}

READABLE_NAMES = {
    "alpha": "Learning rate (alpha)",
    "gamma": "Discount factor (gamma)",
    "epsilon_decay": "Exploration decay",
    "num_tilings": "Number of tilings",
    "tiles_per_dimension": "Tiles per dimension",
}

# The three tile-coding configurations the brief asks to compare.
TILE_CONFIGURATIONS = (
    {"num_tilings": 4, "tiles_per_dimension": 6},
    {"num_tilings": 8, "tiles_per_dimension": 8},
    {"num_tilings": 12, "tiles_per_dimension": 10},
)

WIND_LEVELS = {"Low wind": 0.5, "Medium wind": 1.0, "High wind": 1.6}
LANDING_THRESHOLDS = {"Strict (0.5 m/s)": 0.5, "Default (0.7 m/s)": 0.7,
                      "Relaxed (1.0 m/s)": 1.0}

REPEATS_DEFAULT = 2
EPISODES_DEFAULT = 800  # below ~700 the drone has not learned to land at all
EVALUATION_EPISODES = 10
TAIL_FRACTION = 0.25


def _tail_mean(values):
    numbers = [value for value in values if value == value]     # drop NaNs
    if not numbers:
        return 0.0
    count = max(1, int(len(numbers) * TAIL_FRACTION))
    tail = numbers[-count:]
    return sum(tail) / len(tail)


def run_configuration(hyperparameters=None, environment_parameters=None,
                      algorithm=sarsa_agent.SARSA, episodes=EPISODES_DEFAULT,
                      repeats=REPEATS_DEFAULT, first_seed=0,
                      evaluation_starts=None):
    """Train one configuration `repeats` times and summarise it."""
    hyperparameters = dict(hyperparameters or {})
    environment_parameters = dict(environment_parameters or {})

    collected = {key: [] for key in
                 ("reward", "success", "steps", "final_distance", "landing_speed",
                  "collisions", "crash", "hard_landing", "timeout", "runtime",
                  "weight_norm", "eval_success", "eval_return",
                  "eval_landing_speed", "eval_collisions")}

    for repeat in range(repeats):
        result = sarsa_agent.train(algorithm=algorithm, episodes=episodes,
                                   seed=first_seed + repeat,
                                   environment_parameters=environment_parameters,
                                   **hyperparameters)
        history = result["history"]

        collected["reward"].append(_tail_mean(history["reward"]))
        collected["success"].append(_tail_mean(history["success"]))
        collected["steps"].append(_tail_mean(history["length"]))
        collected["final_distance"].append(_tail_mean(history["final_distance"]))
        collected["landing_speed"].append(_tail_mean(history["landing_speed"]))
        collected["collisions"].append(_tail_mean(history["collisions"]))
        collected["crash"].append(_tail_mean(history["crash"]))
        collected["hard_landing"].append(_tail_mean(history["hard_landing"]))
        collected["timeout"].append(_tail_mean(history["timeout"]))
        collected["runtime"].append(result["runtime_seconds"])
        collected["weight_norm"].append(result["weight_norm"])

        batch = simulate.run_many(result["agent"], episodes=EVALUATION_EPISODES,
                                  starts=evaluation_starts,
                                  environment_parameters=environment_parameters)
        collected["eval_success"].append(batch["success_rate"])
        collected["eval_return"].append(batch["mean_return"])
        collected["eval_landing_speed"].append(batch["mean_landing_speed"])
        collected["eval_collisions"].append(batch["mean_collisions"])

    summary = {"algorithm": algorithm, "repeats": repeats, "episodes": episodes}
    for key, values in collected.items():
        summary["mean_" + key] = _mean(values)
        summary["std_" + key] = _std(values)
    return summary


def parameter_experiment(parameter_name, values=None, algorithm=sarsa_agent.SARSA,
                         episodes=EPISODES_DEFAULT, repeats=REPEATS_DEFAULT,
                         base_hyperparameters=None, environment_parameters=None):
    values = values if values is not None else SWEEPABLE_PARAMETERS[parameter_name]
    rows = []
    for value in values:
        hyperparameters = dict(base_hyperparameters or {})
        hyperparameters[parameter_name] = value
        row = run_configuration(hyperparameters=hyperparameters,
                                environment_parameters=environment_parameters,
                                algorithm=algorithm, episodes=episodes,
                                repeats=repeats)
        row["parameter"] = READABLE_NAMES.get(parameter_name, parameter_name)
        row["value"] = value
        rows.append(row)
    return rows


def tile_resolution_experiment(episodes=EPISODES_DEFAULT, repeats=REPEATS_DEFAULT):
    """Coarse, default and fine tile coding, with the memory cost of each.

    The trade-off: coarse tilings generalise widely but cannot represent a precise
    approach to the pad; fine ones are precise but need far more experience to fill
    in, and far more memory.
    """
    rows = []
    for configuration in TILE_CONFIGURATIONS:
        row = run_configuration(hyperparameters=configuration, episodes=episodes,
                                repeats=repeats)
        row["parameter"] = "tile coding"
        row["value"] = "%d x %d" % (configuration["num_tilings"],
                                    configuration["tiles_per_dimension"])
        # 4 state dimensions, 5 actions.
        row["weights"] = (configuration["num_tilings"]
                          * configuration["tiles_per_dimension"] ** 4 * 5)
        rows.append(row)
    return rows


def wind_strength_experiment(episodes=EPISODES_DEFAULT, repeats=REPEATS_DEFAULT):
    rows = []
    for label, multiplier in WIND_LEVELS.items():
        row = run_configuration(
            environment_parameters={"wind_multiplier": multiplier},
            episodes=episodes, repeats=repeats)
        row["parameter"] = "wind strength"
        row["value"] = label
        rows.append(row)
    return rows


def landing_threshold_experiment(episodes=EPISODES_DEFAULT,
                                 repeats=REPEATS_DEFAULT):
    rows = []
    for label, speed in LANDING_THRESHOLDS.items():
        row = run_configuration(
            environment_parameters={"safe_landing_speed": speed},
            episodes=episodes, repeats=repeats)
        row["parameter"] = "safe landing speed"
        row["value"] = label
        rows.append(row)
    return rows


def compare_algorithms(episodes=EPISODES_DEFAULT, repeats=REPEATS_DEFAULT):
    """SARSA against Q-Learning on identical features, rewards and seeds."""
    rows = []
    for algorithm in (sarsa_agent.SARSA, sarsa_agent.Q_LEARNING):
        row = run_configuration(algorithm=algorithm, episodes=episodes,
                                repeats=repeats)
        row["parameter"] = "algorithm"
        row["value"] = algorithm
        rows.append(row)
    return rows


# ----------------------------------------------------------------------
# The generalisation test
# ----------------------------------------------------------------------

def generalisation_test(episodes=EPISODES_DEFAULT, repeats=REPEATS_DEFAULT,
                        hyperparameters=None, environment_parameters=None):
    """Train from the training start only, then fly from unseen ones.

    Three groups, and the agent has flown from exactly one of them:

      * training starts    — the release points episodes were drawn from
      * validation starts  — nearby, used while settling on the configuration
      * unseen test starts — never used for either

    A tile-coded linear model shares features between nearby states, so if it has
    generalised at all it should still land from a point it has never left.
    """
    groups = (("Training starts", chamber.TRAINING_STARTS),
              ("Validation starts", chamber.VALIDATION_STARTS),
              ("Unseen test starts", chamber.TEST_STARTS))

    rows = []
    per_seed = {label: [] for label, _ in groups}

    for repeat in range(repeats):
        result = sarsa_agent.train(
            episodes=episodes, seed=repeat,
            training_starts=chamber.TRAINING_STARTS,
            environment_parameters=environment_parameters,
            **(hyperparameters or {}))
        for label, starts in groups:
            batch = simulate.run_many(
                result["agent"], episodes=max(len(starts) * 4, 8),
                starts=starts, environment_parameters=environment_parameters)
            per_seed[label].append(batch)

    for label, starts in groups:
        batches = per_seed[label]
        rows.append({
            "parameter": "start group",
            "value": label,
            "starts": [list(start) for start in starts],
            "mean_eval_success": _mean([b["success_rate"] for b in batches]),
            "std_eval_success": _std([b["success_rate"] for b in batches]),
            "mean_eval_return": _mean([b["mean_return"] for b in batches]),
            "std_eval_return": _std([b["mean_return"] for b in batches]),
            "mean_final_distance": _mean([b["mean_final_distance"]
                                          for b in batches]),
            "mean_eval_landing_speed": _mean([b["mean_landing_speed"]
                                              for b in batches]),
            "mean_eval_collisions": _mean([b["mean_collisions"] for b in batches]),
            "mean_crash": _mean([b["crash_rate"] for b in batches]),
            "seen_during_training": label == "Training starts",
        })
    return rows


def best_performing_row(rows, metric="mean_reward"):
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
