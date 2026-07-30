"""Running the robot through Room 1 and recording it for replay.

A simulation produces a list of "frames".  Each frame describes the robot AFTER
one transition, together with what caused it, so the animation can simply walk
through the list.  Frame 0 is the starting position.

Because every frame stores the direction that actually happened, replay never
has to sample anything again: playing a recording back always shows exactly the
same episode.
"""

from core import actions
from rooms.room1 import map_data

MAX_STEPS_DEFAULT = 200

STATUS_RUNNING = "Running"
STATUS_ESCAPED = "Escaped"
STATUS_MAX_STEPS = "Max steps reached"


def _frame(step, state, from_cell, action, info, reward, cumulative_reward, done, status):
    """Build one replay frame.

    Every field the replay viewer and the HUD need is stored explicitly, so a
    saved recording never depends on the environment being re-created.
    """
    row, col, has_battery, previous_direction = state
    from_row, from_col = from_cell
    return {
        "step": step,
        "state": [row, col, has_battery, previous_direction],
        "row": row,
        "col": col,
        "has_battery": has_battery,
        "previous_direction": previous_direction,
        "previous_direction_name": actions.direction_name(previous_direction),
        "from_row": from_row,
        "from_col": from_col,
        "action": action,
        "action_name": actions.direction_name(action) if action is not None else "-",
        "actual_direction": info.get("actual_direction"),
        "actual_direction_name": (actions.direction_name(info["actual_direction"])
                                  if info.get("actual_direction") is not None else "-"),
        "reward": reward,
        "cumulative_reward": cumulative_reward,
        "event": info.get("description", ""),
        "laser_hit": bool(info.get("laser_hit", False)),
        "slipped": bool(info.get("slipped", False)),
        "momentum": bool(info.get("momentum", False)),
        "blocked": bool(info.get("blocked", False)),
        "used_teleport": bool(info.get("used_teleport", False)),
        "battery_collected": bool(info.get("battery_collected", False)),
        "reached_exit": bool(info.get("reached_exit", False)),
        "done": done,
        "status": status,
    }


def run_episode(env, policy, seed=0, max_steps=MAX_STEPS_DEFAULT):
    """Run one episode following `policy` and record every frame.

    The seed makes the episode reproducible: the same seed and the same policy
    always give the same recording.
    """
    state = env.reset(seed=seed)
    cumulative_reward = 0.0

    start_info = {"description": "Systems online — episode started"}
    frames = [_frame(0, state, (state[0], state[1]), None, start_info,
                     0.0, 0.0, False, STATUS_RUNNING)]

    laser_hits = 0
    wall_collisions = 0
    teleports_used = 0
    slips = 0
    battery_collected = False
    success = False
    status = STATUS_MAX_STEPS

    for step in range(1, max_steps + 1):
        action = policy.get(state)
        if action is None:
            # Either the episode already ended or the policy has no entry here.
            break

        from_cell = (state[0], state[1])
        state, reward, done, info = env.step(action)
        cumulative_reward += reward

        if info["laser_hit"]:
            laser_hits += 1
        if info["blocked"]:
            wall_collisions += 1
        if info["used_teleport"]:
            teleports_used += 1
        if info["slipped"]:
            slips += 1
        if info["battery_collected"]:
            battery_collected = True

        if done:
            success = True
            status = STATUS_ESCAPED
        elif step == max_steps:
            status = STATUS_MAX_STEPS
        else:
            status = STATUS_RUNNING

        frames.append(_frame(step, state, from_cell, action, info,
                             reward, cumulative_reward, done, status))

        if done:
            break

    # Make sure the last frame carries the final verdict.
    frames[-1]["status"] = STATUS_ESCAPED if success else STATUS_MAX_STEPS

    return {
        "frames": frames,
        "seed": seed,
        "success": success,
        "status": STATUS_ESCAPED if success else STATUS_MAX_STEPS,
        "steps": len(frames) - 1,
        "total_reward": cumulative_reward,
        "laser_hits": laser_hits,
        "wall_collisions": wall_collisions,
        "teleports_used": teleports_used,
        "slips": slips,
        "used_teleport": teleports_used > 0,
        "battery_collected": battery_collected,
        "map_hash": map_data.map_hash(),
    }


def run_many(env, policy, episodes=30, first_seed=0, max_steps=MAX_STEPS_DEFAULT):
    """Run several episodes with different seeds and average the results.

    A single episode says almost nothing about a stochastic room, so every
    number reported in the interface comes from a batch like this one.
    """
    runs = []
    for index in range(episodes):
        runs.append(run_episode(env, policy, seed=first_seed + index, max_steps=max_steps))

    successes = [run for run in runs if run["success"]]

    return {
        "episodes": episodes,
        "success_rate": len(successes) / episodes if episodes else 0.0,
        "mean_return": _mean([run["total_reward"] for run in runs]),
        "std_return": _standard_deviation([run["total_reward"] for run in runs]),
        "mean_steps": _mean([run["steps"] for run in runs]),
        "std_steps": _standard_deviation([run["steps"] for run in runs]),
        "mean_steps_when_successful": _mean([run["steps"] for run in successes]),
        "mean_laser_hits": _mean([run["laser_hits"] for run in runs]),
        "std_laser_hits": _standard_deviation([run["laser_hits"] for run in runs]),
        "mean_wall_collisions": _mean([run["wall_collisions"] for run in runs]),
        "teleport_use_rate": _mean([1.0 if run["used_teleport"] else 0.0 for run in runs]),
        "battery_rate": _mean([1.0 if run["battery_collected"] else 0.0 for run in runs]),
        "runs": runs,
    }


def describe_route(frames):
    """Name the route an episode took, by looking at which landmarks it visited.

    Used by the experiment tables, where "which way did the plan send the robot"
    is the interesting result rather than any single number.
    """
    visited = {(frame["row"], frame["col"]) for frame in frames}
    parts = []

    if visited & set(map_data.SHAFT_CELLS):
        parts.append("Frozen shaft")
    if any(frame["used_teleport"] for frame in frames):
        parts.append("Teleporter")
    if visited & {(9, 6), (9, 7)}:
        parts.append("South ring")
    if not parts:
        parts.append("Unclear")
    if any(frame["has_battery"] for frame in frames):
        parts.append("+ battery")

    return " ".join(parts)


def _mean(numbers):
    """Plain average, returning 0 for an empty list."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def _standard_deviation(numbers):
    """Population standard deviation, returning 0 for fewer than two values."""
    if len(numbers) < 2:
        return 0.0
    average = _mean(numbers)
    squared_gaps = [(number - average) ** 2 for number in numbers]
    return (sum(squared_gaps) / len(numbers)) ** 0.5
