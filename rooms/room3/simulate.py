"""Running the learned policy through Room 3 and recording it for replay.

Same frame-per-step shape as Rooms 1 and 2. Room 3's frames additionally carry the
guard's position, the mission stage and whether the sliding doors are open, so the
animation can redraw the whole chamber at any point in the recording without
re-running anything.
"""

from core import actions
from rooms.room3 import map_data
from rooms.room3.environment import STAGE_UNLOCKED

MAX_STEPS_DEFAULT = 300

STATUS_RUNNING = "Running"
STATUS_ESCAPED = "Escaped"
STATUS_CAUGHT = "Detected by the security robot"
STATUS_MAX_STEPS = "Max steps reached"


def _frame(step, state, from_cell, action, info, reward, cumulative_reward, done,
           status):
    row, col, stage, guard_index = state
    from_row, from_col = from_cell
    guard_row, guard_col = info.get("guard_cell") or map_data.guard_cell(guard_index)

    return {
        "step": step,
        "state": [row, col, stage, guard_index],
        "row": row,
        "col": col,
        "stage": stage,
        "guard_index": guard_index,
        "guard_row": guard_row,
        "guard_col": guard_col,
        "doors_open": bool(map_data.door_is_open(guard_index)),
        # The renderer speaks in generic names, so one engine serves every room.
        "has_item": stage >= STAGE_UNLOCKED,
        "item_collected": bool(info.get("unlocked", False)),
        "from_row": from_row,
        "from_col": from_col,
        "action": action,
        "action_name": actions.direction_name(action) if action is not None else "-",
        "actual_direction": action,
        "actual_direction_name": (actions.direction_name(action)
                                  if action is not None else "-"),
        "reward": reward,
        "cumulative_reward": cumulative_reward,
        "event": info.get("description", ""),
        "blocked": bool(info.get("blocked", False)),
        "door_blocked": bool(info.get("door_blocked", False)),
        "waited": bool(info.get("waited", False)),
        "hazard": bool(info.get("hazard", False)),
        "caught": bool(info.get("caught", False)),
        "generator": info.get("generator"),
        "unlocked": bool(info.get("unlocked", False)),
        "wrong_order": bool(info.get("wrong_order", False)),
        "reached_exit": bool(info.get("reached_exit", False)),
        "done": done,
        "status": status,
    }


def run_episode(env, policy, epsilon=0.0, seed=0, max_steps=MAX_STEPS_DEFAULT):
    """Follow `policy` for one episode and record every frame."""
    import random

    rng = random.Random(seed)
    state = env.reset(seed=seed)
    cumulative_reward = 0.0

    start_info = {"description": "Reactor offline — R-5 entering the chamber",
                  "guard_cell": list(env.guard_cell())}
    frames = [_frame(0, state, (state[0], state[1]), None, start_info, 0.0, 0.0,
                     False, STATUS_RUNNING)]

    hazards = 0
    wall_collisions = 0
    waits = 0
    generators = []
    caught = False
    success = False
    status = STATUS_MAX_STEPS

    for step in range(1, max_steps + 1):
        action = _choose(policy, state, env, epsilon, rng)
        if action is None:
            break

        from_cell = (state[0], state[1])
        state, reward, done, info = env.step(action)
        cumulative_reward += reward

        if info["hazard"]:
            hazards += 1
        if info["blocked"]:
            wall_collisions += 1
        if info["waited"]:
            waits += 1
        if info["generator"]:
            generators.append(info["generator"])
        if info["caught"]:
            caught = True

        if done and info["reached_exit"]:
            success = True
            status = STATUS_ESCAPED
        elif done and info["caught"]:
            status = STATUS_CAUGHT
        elif step == max_steps:
            status = STATUS_MAX_STEPS
        else:
            status = STATUS_RUNNING

        frames.append(_frame(step, state, from_cell, action, info, reward,
                             cumulative_reward, done, status))
        if done:
            break

    frames[-1]["status"] = status

    return {
        "frames": frames,
        "seed": seed,
        "epsilon": epsilon,
        "success": success,
        "status": status,
        "steps": len(frames) - 1,
        "total_reward": cumulative_reward,
        "hazards": hazards,
        "wall_collisions": wall_collisions,
        "waits": waits,
        "generators": generators,
        "stage": frames[-1]["stage"],
        "generators_done": frames[-1]["stage"] >= STAGE_UNLOCKED,
        "caught": caught,
        "map_hash": map_data.map_hash(),
    }


def _choose(policy, state, env, epsilon, rng):
    if rng.random() < epsilon:
        return rng.choice(env.actions())
    return policy.get(state)


def run_many(env, policy, episodes=30, first_seed=0, epsilon=0.0,
             max_steps=MAX_STEPS_DEFAULT):
    """Run several episodes and average the results.

    The chamber is deterministic apart from the agent's own exploration, so with
    `epsilon` at 0 every episode is identical. The batch still matters when
    exploration is switched on, and it keeps the reporting consistent with the
    other rooms.
    """
    runs = [run_episode(env, policy, epsilon=epsilon, seed=first_seed + index,
                        max_steps=max_steps)
            for index in range(episodes)]
    successes = [run for run in runs if run["success"]]

    return {
        "episodes": episodes,
        "success_rate": len(successes) / episodes if episodes else 0.0,
        "mean_return": _mean([run["total_reward"] for run in runs]),
        "std_return": _std([run["total_reward"] for run in runs]),
        "mean_steps": _mean([run["steps"] for run in runs]),
        "std_steps": _std([run["steps"] for run in runs]),
        "generator_rate": _mean([1.0 if run["generators_done"] else 0.0
                                 for run in runs]),
        "mean_stage": _mean([run["stage"] for run in runs]),
        "caught_rate": _mean([1.0 if run["caught"] else 0.0 for run in runs]),
        "mean_hazards": _mean([run["hazards"] for run in runs]),
        "mean_waits": _mean([run["waits"] for run in runs]),
        "runs": runs,
    }


def describe_route(frames):
    """Whether the run used the service shaft or went round the ring."""
    visited = {(frame["row"], frame["col"]) for frame in frames}
    shaft = set(map_data.shaft_cells())
    parts = []
    if visited & shaft:
        parts.append("Service shaft")
    else:
        parts.append("Service ring")
    if any(frame["hazard"] for frame in frames):
        parts.append("(took shocks)")
    if any(frame["caught"] for frame in frames):
        parts.append("(caught)")
    return " ".join(parts)


def _mean(numbers):
    return sum(numbers) / len(numbers) if numbers else 0.0


def _std(numbers):
    if len(numbers) < 2:
        return 0.0
    average = _mean(numbers)
    return (sum((number - average) ** 2 for number in numbers) / len(numbers)) ** 0.5
