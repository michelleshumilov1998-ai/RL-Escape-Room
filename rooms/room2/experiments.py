"""Parameter experiments for Room 2.

Same recipe as Room 1, adapted to a room that learns instead of planning:

  1. Build an environment.
  2. Train an agent with one hyperparameter changed.
  3. Measure the last slice of training, and run a batch of greedy episodes.
  4. Report the averages with the standard deviation next to them.

Training is stochastic, so nothing here is ever reported from a single run: every
configuration is trained `repeats` times with different seeds.

As in Room 1, a sweep can only say which of the values *tested* did best. It is
reported as the "best-performing value among the tested configurations", never as
an optimal hyperparameter.
"""

from rooms.room2 import sarsa_agent, simulate
from rooms.room2.environment import Room2Env

# Which hyperparameters the interface can sweep, and the values offered.
SWEEPABLE_PARAMETERS = {
    "alpha": [0.05, 0.10, 0.30, 0.50, 0.80],
    "gamma": [0.80, 0.90, 0.95, 0.99, 1.00],
    "epsilon_decay": [0.980, 0.990, 0.995, 0.999],
    "epsilon_start": [0.20, 0.50, 1.00],
    "epsilon_min": [0.01, 0.05, 0.10, 0.20],
}

READABLE_NAMES = {
    "alpha": "Learning rate (alpha)",
    "gamma": "Discount factor (gamma)",
    "epsilon_decay": "Epsilon decay",
    "epsilon_start": "Initial epsilon",
    "epsilon_min": "Minimum epsilon",
}

REPEATS_DEFAULT = 3
EPISODES_DEFAULT = 1200
EVALUATION_EPISODES = 20
TAIL_FRACTION = 0.2      # the share of training counted as "settled"


def _tail_mean(values):
    """The average over the last fifth of training.

    Early episodes are mostly exploration, so averaging the whole run would hide
    what the agent actually ended up doing.
    """
    if not values:
        return 0.0
    count = max(1, int(len(values) * TAIL_FRACTION))
    tail = values[-count:]
    return sum(tail) / len(tail)


def run_configuration(hyperparameters=None, environment_parameters=None,
                      algorithm=sarsa_agent.SARSA, episodes=EPISODES_DEFAULT,
                      repeats=REPEATS_DEFAULT, first_seed=0):
    """Train one configuration `repeats` times and summarise the results."""
    hyperparameters = dict(hyperparameters or {})
    environment_parameters = dict(environment_parameters or {})

    rewards, successes, steps, pit_falls, span_use, runtimes = [], [], [], [], [], []
    keycards, greedy_returns, routes = [], [], []

    for repeat in range(repeats):
        env = Room2Env(**environment_parameters)
        result = sarsa_agent.train(env, algorithm=algorithm, episodes=episodes,
                                   seed=first_seed + repeat, **hyperparameters)
        history = result["history"]

        rewards.append(_tail_mean(history["reward"]))
        successes.append(_tail_mean(history["success"]))
        steps.append(_tail_mean(history["length"]))
        pit_falls.append(_tail_mean(history["pit_fall"]))
        span_use.append(_tail_mean(history["used_span"]))
        keycards.append(_tail_mean(history["keycard"]))
        runtimes.append(result["runtime_seconds"])

        # How good the learned policy is when it stops exploring.
        batch = simulate.run_many(env, result["policy"], episodes=EVALUATION_EPISODES)
        greedy_returns.append(batch["mean_return"])
        routes.append(simulate.describe_route(batch["runs"][0]["frames"]))

    return {
        "algorithm": algorithm,
        "repeats": repeats,
        "episodes": episodes,
        "mean_reward": _mean(rewards), "std_reward": _std(rewards),
        "mean_success": _mean(successes), "std_success": _std(successes),
        "mean_steps": _mean(steps), "std_steps": _std(steps),
        "mean_pit_falls": _mean(pit_falls), "std_pit_falls": _std(pit_falls),
        "mean_span_use": _mean(span_use), "std_span_use": _std(span_use),
        "mean_keycard": _mean(keycards),
        "mean_greedy_return": _mean(greedy_returns),
        "std_greedy_return": _std(greedy_returns),
        "mean_runtime": _mean(runtimes), "std_runtime": _std(runtimes),
        "route": _most_common(routes),
    }


def parameter_experiment(parameter_name, values=None, environment_parameters=None,
                         algorithm=sarsa_agent.SARSA, episodes=EPISODES_DEFAULT,
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
    """Train SARSA and Q-Learning on the same room with the same settings.

    This is what turns "SARSA is safer here" from a claim into a measurement.
    """
    results = {}
    for algorithm in (sarsa_agent.SARSA, sarsa_agent.Q_LEARNING):
        results[algorithm] = run_configuration(
            hyperparameters=base_hyperparameters,
            environment_parameters=environment_parameters,
            algorithm=algorithm, episodes=episodes, repeats=repeats)
    return results


def compare_state_representations(episodes=EPISODES_DEFAULT, repeats=REPEATS_DEFAULT,
                                  base_hyperparameters=None):
    """The headline experiment of Room 2.

    Trains both methods twice: once with the assignment's state, and once with the
    collapsed bridges added to it. It shows that with the assignment's state the
    two methods behave identically, and that once the state is complete enough for
    the problem to be Markovian, Q-Learning takes the dangerous bridge and SARSA
    does not.
    """
    table = []
    for include_bridge_state in (False, True):
        label = ("(row, col, keycard)" if not include_bridge_state
                 else "(row, col, keycard, collapsed)")
        for algorithm in (sarsa_agent.SARSA, sarsa_agent.Q_LEARNING):
            row = run_configuration(
                hyperparameters=base_hyperparameters,
                environment_parameters={"include_bridge_state": include_bridge_state},
                algorithm=algorithm, episodes=episodes, repeats=repeats)
            row["state_representation"] = label
            row["include_bridge_state"] = include_bridge_state
            row["parameter"] = "state representation"
            row["value"] = "%s / %s" % (label, algorithm)
            table.append(row)
    return table


def best_performing_row(rows, metric="mean_reward"):
    """The row that scored highest on a metric.

    Named after what it really is: the best of the values that were tried.
    """
    if not rows:
        return None
    best = rows[0]
    for row in rows[1:]:
        if row[metric] > best[metric]:
            best = row
    return best


def _mean(numbers):
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def _std(numbers):
    if len(numbers) < 2:
        return 0.0
    average = _mean(numbers)
    return (sum((number - average) ** 2 for number in numbers) / len(numbers)) ** 0.5


def _most_common(values):
    """The value that appears most often, for reporting which route was chosen."""
    if not values:
        return "—"
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get)
