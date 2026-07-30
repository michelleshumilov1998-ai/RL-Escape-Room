"""Parameter experiments for Room 3.

Same recipe as the earlier rooms: train a configuration several times with
different seeds, measure the settled part of training, evaluate the greedy policy,
and report mean ± standard deviation.

A sweep can only say which of the values *tested* did best. It is reported as the
"best-performing configuration among the tested settings", never as optimal.
"""

from rooms.room3 import q_agent, simulate
from rooms.room3.environment import Room3Env

SWEEPABLE_PARAMETERS = {
    "alpha": [0.05, 0.10, 0.15, 0.30, 0.50],
    "gamma": [0.85, 0.90, 0.95, 0.98, 1.00],
    "epsilon_decay": [0.995, 0.998, 0.999, 0.9995],
    "epsilon_min": [0.01, 0.05, 0.10, 0.20],
}

READABLE_NAMES = {
    "alpha": "Learning rate (alpha)",
    "gamma": "Discount factor (gamma)",
    "epsilon_decay": "Exploration decay",
    "epsilon_min": "Minimum epsilon",
}

REPEATS_DEFAULT = 3
EPISODES_DEFAULT = 3000  # below ~3000 the greedy policy does not finish the sequence
TAIL_FRACTION = 0.2


def _tail_mean(values):
    if not values:
        return 0.0
    count = max(1, int(len(values) * TAIL_FRACTION))
    tail = values[-count:]
    return sum(tail) / len(tail)


def run_configuration(hyperparameters=None, environment_parameters=None,
                      algorithm=q_agent.Q_LEARNING, episodes=EPISODES_DEFAULT,
                      repeats=REPEATS_DEFAULT, first_seed=0):
    """Train one configuration `repeats` times and summarise the results."""
    hyperparameters = dict(hyperparameters or {})
    environment_parameters = dict(environment_parameters or {})

    collected = {key: [] for key in
                 ("reward", "success", "steps", "stage", "generators",
                  "caught", "hazards", "runtime", "greedy_reward",
                  "greedy_success")}
    routes = []

    for repeat in range(repeats):
        env = Room3Env(**environment_parameters)
        result = q_agent.train(env, algorithm=algorithm, episodes=episodes,
                               seed=first_seed + repeat, **hyperparameters)
        history = result["history"]

        collected["reward"].append(_tail_mean(history["reward"]))
        collected["success"].append(_tail_mean(history["success"]))
        collected["steps"].append(_tail_mean(history["length"]))
        collected["stage"].append(_tail_mean(history["stage"]))
        collected["generators"].append(_tail_mean(history["generators_done"]))
        collected["caught"].append(_tail_mean(history["guard_caught"]))
        collected["hazards"].append(_tail_mean(history["hazards"]))
        collected["runtime"].append(result["runtime_seconds"])

        # How the learned policy does with exploration switched off.
        run = simulate.run_episode(env, result["policy"], seed=0)
        collected["greedy_reward"].append(run["total_reward"])
        collected["greedy_success"].append(1.0 if run["success"] else 0.0)
        routes.append(simulate.describe_route(run["frames"]))

    summary = {"algorithm": algorithm, "repeats": repeats, "episodes": episodes,
               "route": _most_common(routes)}
    for key, values in collected.items():
        summary["mean_" + key] = _mean(values)
        summary["std_" + key] = _std(values)
    return summary


def parameter_experiment(parameter_name, values=None, environment_parameters=None,
                         algorithm=q_agent.Q_LEARNING, episodes=EPISODES_DEFAULT,
                         repeats=REPEATS_DEFAULT, base_hyperparameters=None):
    """Sweep one hyperparameter across several values."""
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


def compare_algorithms(environment_parameters=None, episodes=EPISODES_DEFAULT,
                       repeats=REPEATS_DEFAULT, base_hyperparameters=None):
    """Q-Learning against SARSA on the same chamber with the same settings."""
    rows = []
    for algorithm in (q_agent.Q_LEARNING, q_agent.SARSA):
        row = run_configuration(hyperparameters=base_hyperparameters,
                                environment_parameters=environment_parameters,
                                algorithm=algorithm, episodes=episodes,
                                repeats=repeats)
        row["parameter"] = "algorithm"
        row["value"] = algorithm
        rows.append(row)
    return rows


def best_performing_row(rows, metric="mean_reward"):
    if not rows:
        return None
    best = rows[0]
    for row in rows[1:]:
        if row[metric] > best[metric]:
            best = row
    return best


def _mean(numbers):
    return sum(numbers) / len(numbers) if numbers else 0.0


def _std(numbers):
    if len(numbers) < 2:
        return 0.0
    average = _mean(numbers)
    return (sum((number - average) ** 2 for number in numbers) / len(numbers)) ** 0.5


def _most_common(values):
    if not values:
        return "—"
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get)
