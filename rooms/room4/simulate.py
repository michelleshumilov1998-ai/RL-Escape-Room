"""Flying the learned policy and recording it for replay.

Every frame stores the actual numbers the step produced — position, velocity,
thrust, wind and, when it is switched on, the turbulence that was drawn. Replay
therefore walks the stored list and never recomputes anything, which matters here
because turbulence is random: re-simulating a turbulent episode would give a
different flight.
"""

import math

from rooms.room4 import chamber
from rooms.room4.environment import (ACTION_ARROWS, ACTION_NAMES, MAX_STEPS_DEFAULT,
                                     Room4Env)

STATUS_FLYING = "Flying"
STATUS_LANDED = "Safely landed"
STATUS_CRASHED = "Crashed"
STATUS_TIMEOUT = "Out of time"


def _frame(step, state, previous_state, action, info, reward, cumulative_reward,
           done, status):
    x, y, velocity_x, velocity_y = state
    return {
        "step": step,
        "state": [x, y, velocity_x, velocity_y],
        "x": x, "y": y, "vx": velocity_x, "vy": velocity_y,
        "from_x": previous_state[0], "from_y": previous_state[1],
        "speed": info.get("speed", math.hypot(velocity_x, velocity_y)),
        "action": action,
        "action_name": ACTION_NAMES[action] if action is not None else "-",
        "action_arrow": ACTION_ARROWS[action] if action is not None else "·",
        "thrust_x": info.get("thrust_x", 0.0),
        "thrust_y": info.get("thrust_y", 0.0),
        "wind_x": info.get("wind_x", 0.0),
        "wind_y": info.get("wind_y", 0.0),
        "turbulence_x": info.get("turbulence_x", 0.0),
        "turbulence_y": info.get("turbulence_y", 0.0),
        "reward": reward,
        "cumulative_reward": cumulative_reward,
        "distance_to_goal": info.get("distance_to_goal",
                                     chamber.distance_to_goal(x, y)),
        "collision": bool(info.get("collision", False)),
        "boundary": bool(info.get("boundary", False)),
        "hard_landing": bool(info.get("hard_landing", False)),
        "safe_landing": bool(info.get("safe_landing", False)),
        "crashed": bool(info.get("crashed", False)),
        "timeout": bool(info.get("timeout", False)),
        "done": done,
        "event": info.get("description", ""),
        "status": status,
    }


def run_episode(agent, env=None, epsilon=0.0, seed=0, start_position=None,
                max_steps=MAX_STEPS_DEFAULT, environment_parameters=None):
    """Fly one episode with `agent` and record every frame."""
    import random

    env = env or Room4Env(**(environment_parameters or {}))
    rng = random.Random(seed)
    state = env.reset(seed=seed, start_position=start_position)

    cumulative_reward = 0.0
    start_info = {"description": "Drone link established — rotors up",
                  "speed": 0.0,
                  "distance_to_goal": chamber.distance_to_goal(state[0], state[1])}
    frames = [_frame(0, state, state, None, start_info, 0.0, 0.0, False,
                     STATUS_FLYING)]

    collisions = 0
    hard_landings = 0
    landed = False
    crashed = False
    timed_out = False
    landing_speed = None
    status = STATUS_TIMEOUT

    for step in range(1, max_steps + 1):
        action = (rng.choice(env.actions()) if rng.random() < epsilon
                  else agent.best_action(state))
        previous_state = state
        state, reward, done, info = env.step(action)
        cumulative_reward += reward

        if info["collision"] or info["boundary"]:
            collisions += 1
        if info["hard_landing"]:
            hard_landings += 1
            landing_speed = info["speed"]
        if info["safe_landing"]:
            landed = True
            landing_speed = info["speed"]
            status = STATUS_LANDED
        elif info["crashed"]:
            crashed = True
            landing_speed = info["speed"]
            status = STATUS_CRASHED
        elif info["timeout"]:
            timed_out = True
            status = STATUS_TIMEOUT
        elif not done:
            status = STATUS_FLYING

        frames.append(_frame(step, state, previous_state, action, info, reward,
                             cumulative_reward, done, status))
        if done:
            break

    frames[-1]["status"] = status

    return {
        "frames": frames,
        "seed": seed,
        "epsilon": epsilon,
        "start_position": list(frames[0]["state"][:2]),
        "success": landed,
        "status": status,
        "steps": len(frames) - 1,
        "total_reward": cumulative_reward,
        "collisions": collisions,
        "hard_landings": hard_landings,
        "crashed": crashed,
        "timeout": timed_out,
        "landing_speed": landing_speed,
        "final_distance": chamber.distance_to_goal(state[0], state[1]),
    }


def run_many(agent, episodes=20, first_seed=0, epsilon=0.0, starts=None,
             environment_parameters=None):
    """Fly several episodes and average the results.

    `starts` cycles through a list of release points, which is how the
    generalisation test evaluates on positions never trained on.
    """
    starts = tuple(starts or chamber.TRAINING_STARTS)
    runs = []
    for index in range(episodes):
        runs.append(run_episode(agent, epsilon=epsilon, seed=first_seed + index,
                                start_position=starts[index % len(starts)],
                                environment_parameters=environment_parameters))

    landed = [run for run in runs if run["success"]]
    speeds = [run["landing_speed"] for run in runs
              if run["landing_speed"] is not None]

    return {
        "episodes": episodes,
        "starts": [list(start) for start in starts],
        "success_rate": len(landed) / episodes if episodes else 0.0,
        "mean_return": _mean([run["total_reward"] for run in runs]),
        "std_return": _std([run["total_reward"] for run in runs]),
        "mean_steps": _mean([run["steps"] for run in runs]),
        "mean_final_distance": _mean([run["final_distance"] for run in runs]),
        "mean_landing_speed": _mean(speeds),
        "collision_rate": _mean([1.0 if run["collisions"] else 0.0
                                 for run in runs]),
        "mean_collisions": _mean([run["collisions"] for run in runs]),
        "crash_rate": _mean([1.0 if run["crashed"] else 0.0 for run in runs]),
        "hard_landing_rate": _mean([1.0 if run["hard_landings"] else 0.0
                                    for run in runs]),
        "timeout_rate": _mean([1.0 if run["timeout"] else 0.0 for run in runs]),
        "runs": runs,
    }


def _mean(numbers):
    return sum(numbers) / len(numbers) if numbers else 0.0


def _std(numbers):
    if len(numbers) < 2:
        return 0.0
    average = _mean(numbers)
    return (sum((number - average) ** 2 for number in numbers) / len(numbers)) ** 0.5
