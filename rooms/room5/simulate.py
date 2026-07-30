"""Running a mission and recording it for replay.

Room 5 has more moving parts than any other room — a procedural layout, patrolling
robots, conveyors and a fog of war — so a replay cannot be reconstructed from the
action list alone. Every frame therefore stores the layout seed, the robot
positions, the radar reading, the visible cells and the stage, and playback walks
that list without re-generating or re-simulating anything.
"""

from rooms.room5 import layout as L
from rooms.room5.environment import ACTION_ARROWS, ACTION_NAMES, Room5Env

STATUS_RUNNING = "Running"
STATUS_ESCAPED = "Escaped"
STATUS_CAUGHT = "Struck by a maintenance robot"
STATUS_TIMEOUT = "Out of time"


def _frame(step, env, from_cell, action, observation, info, reward,
           cumulative_reward, done, status):
    row, col = env.position
    target_row, target_col = env.current_target()
    return {
        "step": step,
        "row": row, "col": col,
        "from_row": from_cell[0], "from_col": from_cell[1],
        "agent_position": [row, col],
        "stage": env.stage,
        "current_target": [target_row, target_col],
        "action": action,
        "action_name": ACTION_NAMES[action] if action is not None else "-",
        "action_arrow": ACTION_ARROWS[action] if action is not None else "·",
        "reward": reward,
        "cumulative_reward": cumulative_reward,
        "radar_observation": list(observation["static_radar"]),
        "dynamic_radar": list(observation["dynamic_radar"]),
        "nearest_robot": observation["nearest_robot"],
        "target_features": list(observation["target"]),
        "visible_cells": info.get("visible_cells", []),
        "seen_cells": [list(cell) for cell in sorted(env.seen_cells)],
        "moving_robot_positions": info.get("robot_cells", []),
        "moving_robot_directions": info.get("robot_directions", []),
        "conveyor_event": bool(info.get("conveyor", False)),
        "collision": bool(info.get("collision", False)),
        "terminal_activated": bool(info.get("terminal_activated", False)),
        "exit_unlocked": env.stage >= 1,
        "early_exit": bool(info.get("early_exit", False)),
        "charged": bool(info.get("charged", False)),
        "caught": bool(info.get("caught", False)),
        "reached_exit": bool(info.get("reached_exit", False)),
        "timeout": bool(info.get("timeout", False)),
        # The engine's generic names, so one renderer serves every room.
        "has_item": env.stage >= 1,
        "item_collected": bool(info.get("terminal_activated", False)),
        "done": done,
        "success": bool(info.get("reached_exit", False)),
        "event": info.get("description", ""),
        "status": status,
    }


def run_episode(agent, layout, environment_parameters=None, epsilon=0.0, seed=0):
    """Run one mission on one layout and record every frame."""
    import random

    rng = random.Random(seed)
    env = Room5Env(layout=layout, **(environment_parameters or {}))
    observation = env.reset(layout)

    start_info = {"description": "Radar online — warehouse layout unknown",
                  "robot_cells": [list(robot.cell()) for robot in env.robots],
                  "robot_directions": [robot.pattern for robot in env.robots],
                  "visible_cells": [list(cell) for cell in env.visible_cells()]}
    frames = [_frame(0, env, env.position, None, observation, start_info, 0.0, 0.0,
                     False, STATUS_RUNNING)]

    cumulative_reward = 0.0
    collisions = 0
    conveyors = 0
    waits = 0
    activated = False
    escaped = False
    caught = False
    timed_out = False
    status = STATUS_TIMEOUT

    for step in range(1, env.max_steps + 1):
        action = (rng.choice(env.actions()) if epsilon and rng.random() < epsilon
                  else agent.best_action(observation))
        from_cell = env.position
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward

        if info["collision"]:
            collisions += 1
        if info["conveyor"]:
            conveyors += 1
        if action == 4:
            waits += 1
        if info["terminal_activated"]:
            activated = True
        if info["reached_exit"]:
            escaped = True
            status = STATUS_ESCAPED
        elif info["caught"]:
            caught = True
            status = STATUS_CAUGHT
        elif info["timeout"]:
            timed_out = True
            status = STATUS_TIMEOUT
        elif not done:
            status = STATUS_RUNNING

        frames.append(_frame(step, env, from_cell, action, observation, info,
                             reward, cumulative_reward, done, status))
        if done:
            break

    frames[-1]["status"] = status
    reference = layout.shortest_mission_length()
    steps = len(frames) - 1

    return {
        "frames": frames,
        "layout_seed": layout.seed,
        "layout_split": layout.split,
        "difficulty": layout.difficulty,
        "seed": seed,
        "success": escaped,
        "status": status,
        "steps": steps,
        "total_reward": cumulative_reward,
        "collisions": collisions,
        "conveyor_events": conveyors,
        "waits": waits,
        "terminal_activated": activated,
        "caught": caught,
        "timeout": timed_out,
        "stage": frames[-1]["stage"],
        "reference_path": reference,
        "path_efficiency": (steps / reference) if (reference and escaped) else None,
    }


def run_many(agent, layouts, environment_parameters=None, epsilon=0.0):
    """Run one mission per layout and average the results."""
    runs = [run_episode(agent, layout, environment_parameters, epsilon,
                        seed=index)
            for index, layout in enumerate(layouts)]
    successes = [run for run in runs if run["success"]]
    efficiencies = [run["path_efficiency"] for run in runs
                    if run["path_efficiency"] is not None]

    return {
        "layouts": len(runs),
        "success_rate": len(successes) / len(runs) if runs else 0.0,
        "terminal_rate": _mean([1.0 if run["terminal_activated"] else 0.0
                                for run in runs]),
        "mean_return": _mean([run["total_reward"] for run in runs]),
        "std_return": _std([run["total_reward"] for run in runs]),
        "mean_steps": _mean([run["steps"] for run in runs]),
        "caught_rate": _mean([1.0 if run["caught"] else 0.0 for run in runs]),
        "timeout_rate": _mean([1.0 if run["timeout"] else 0.0 for run in runs]),
        "mean_collisions": _mean([run["collisions"] for run in runs]),
        "path_efficiency": _mean(efficiencies) if efficiencies else float("nan"),
        "runs": runs,
    }


def _mean(numbers):
    return sum(numbers) / len(numbers) if numbers else 0.0


def _std(numbers):
    if len(numbers) < 2:
        return 0.0
    average = _mean(numbers)
    return (sum((number - average) ** 2 for number in numbers) / len(numbers)) ** 0.5
