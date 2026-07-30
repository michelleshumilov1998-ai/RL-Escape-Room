"""Parameter experiments for Room 1.

Every experiment follows the same recipe:

  1. Build an environment with one setting changed.
  2. Solve it with Dynamic Programming.
  3. Run a batch of simulations with different seeds.
  4. Report the averages, with the standard deviation next to them.

A single simulation of a stochastic room tells you almost nothing, so nothing in
this file ever reports a result from one episode.

The wording matters too: a sweep can only tell us which value did best out of
the ones we tried.  It is reported as the "best-performing value among the
tested configurations", never as an "optimal hyperparameter".
"""

from rooms.room1 import dp_solver, simulate
from rooms.room1.environment import Room1Env

# The three risk settings used by the slipping experiment.
RISK_LEVELS = {
    "Low risk": {"weak_ice_intended": 0.90, "strong_ice_intended": 0.85},
    "Medium risk": {"weak_ice_intended": 0.80, "strong_ice_intended": 0.60},
    "High risk": {"weak_ice_intended": 0.60, "strong_ice_intended": 0.35},
}

# The discount factors used by the gamma experiment.
GAMMA_VALUES = [0.70, 0.85, 0.95, 0.99]

# Which settings the interface is allowed to sweep, and the values offered.
SWEEPABLE_PARAMETERS = {
    "gamma": [0.70, 0.85, 0.95, 0.99],
    "weak_ice_intended": [0.60, 0.70, 0.80, 0.90, 1.00],
    "strong_ice_intended": [0.30, 0.45, 0.60, 0.75, 0.90],
    "oil_momentum": [0.20, 0.35, 0.55, 0.70, 0.85],
    "laser_penalty": [-5, -15, -30, -60, -120],
}

EPISODES_DEFAULT = 30


def run_configuration(environment_parameters, gamma, theta=1e-6, max_sweeps=2000,
                      algorithm="Value Iteration", episodes=EPISODES_DEFAULT,
                      first_seed=0):
    """Solve one configuration and measure it over a batch of episodes."""
    env = Room1Env(**environment_parameters)
    solution = dp_solver.solve(env, algorithm, gamma=gamma, theta=theta,
                               max_sweeps=max_sweeps)
    batch = simulate.run_many(env, solution["policy"], episodes=episodes,
                              first_seed=first_seed)
    example = simulate.run_episode(env, solution["policy"], seed=first_seed)

    return {
        "iterations": solution["iterations"],
        "runtime_seconds": solution["runtime_seconds"],
        "start_state_value": solution["start_state_value"],
        "bellman_residual": solution["bellman_residual"],
        "success_rate": batch["success_rate"],
        "mean_return": batch["mean_return"],
        "std_return": batch["std_return"],
        "mean_steps": batch["mean_steps"],
        "std_steps": batch["std_steps"],
        "mean_laser_hits": batch["mean_laser_hits"],
        "std_laser_hits": batch["std_laser_hits"],
        "teleport_use_rate": batch["teleport_use_rate"],
        "battery_rate": batch["battery_rate"],
        "route": simulate.describe_route(example["frames"]),
    }


def parameter_experiment(parameter_name, values, base_parameters=None, gamma=0.95,
                         theta=1e-6, max_sweeps=2000, algorithm="Value Iteration",
                         episodes=EPISODES_DEFAULT):
    """Sweep one setting across several values and report every result.

    `gamma` belongs to the solver rather than to the environment, so it is
    handled separately from the environment settings.
    """
    base_parameters = dict(base_parameters or {})
    rows = []

    for value in values:
        if parameter_name == "gamma":
            environment_parameters = base_parameters
            solver_gamma = value
        else:
            environment_parameters = dict(base_parameters)
            environment_parameters[parameter_name] = value
            solver_gamma = gamma

        row = run_configuration(environment_parameters, solver_gamma, theta=theta,
                                max_sweeps=max_sweeps, algorithm=algorithm,
                                episodes=episodes)
        row["parameter"] = parameter_name
        row["value"] = value
        rows.append(row)

    return rows


def gamma_experiment(base_parameters=None, episodes=EPISODES_DEFAULT, **kwargs):
    """How the discount factor changes the route the robot plans.

    A low discount factor makes the robot impatient, so it tends to accept a
    risky shortcut.  A high one makes it willing to walk the long safe ring.
    """
    return parameter_experiment("gamma", GAMMA_VALUES,
                                base_parameters=base_parameters,
                                episodes=episodes, **kwargs)


def slipping_experiment(base_parameters=None, gamma=0.95, episodes=EPISODES_DEFAULT,
                        **kwargs):
    """How the amount of ice changes the route the robot plans.

    This is the headline result of Room 1: as the floor gets more slippery the
    plan moves off the short frozen shaft and onto the long safe ring, purely
    because the model says so.  Nothing is learned from experience.
    """
    rows = []
    for level_name, overrides in RISK_LEVELS.items():
        environment_parameters = dict(base_parameters or {})
        environment_parameters.update(overrides)
        row = run_configuration(environment_parameters, gamma, episodes=episodes,
                                **kwargs)
        row["parameter"] = "ice risk"
        row["value"] = level_name
        row["weak_ice_intended"] = overrides["weak_ice_intended"]
        row["strong_ice_intended"] = overrides["strong_ice_intended"]
        rows.append(row)
    return rows


def compare_algorithms(environment_parameters=None, gamma=0.95, theta=1e-6,
                       max_sweeps=2000, episodes=EPISODES_DEFAULT):
    """Run both algorithms on the same room with the same settings.

    Returns both solutions, a state-by-state policy comparison, and a batch of
    simulations for each policy so the comparison covers behaviour and not just
    convergence speed.
    """
    env = Room1Env(**(environment_parameters or {}))

    value_result = dp_solver.value_iteration(env, gamma=gamma, theta=theta,
                                             max_sweeps=max_sweeps)
    policy_result = dp_solver.policy_iteration(env, gamma=gamma, theta=theta,
                                               max_iterations=max_sweeps)
    comparison = dp_solver.compare_policies(env, value_result, policy_result, gamma)

    value_batch = simulate.run_many(env, value_result["policy"], episodes=episodes)
    policy_batch = simulate.run_many(env, policy_result["policy"], episodes=episodes)

    return {
        "value_iteration": value_result,
        "policy_iteration": policy_result,
        "comparison": comparison,
        "value_iteration_batch": value_batch,
        "policy_iteration_batch": policy_batch,
        "gamma": gamma,
        "episodes": episodes,
    }


def best_performing_row(rows, metric="mean_return"):
    """The row that scored highest on a metric.

    Deliberately named after what it really is: the best of the values that were
    tested, which is not the same thing as an optimal value.
    """
    if not rows:
        return None
    best = rows[0]
    for row in rows[1:]:
        if row[metric] > best[metric]:
            best = row
    return best
