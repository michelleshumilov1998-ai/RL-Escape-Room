"""Running the learned policy through Room 2 and recording it for replay.

Same shape as Room 1: a run produces a list of frames, each describing the robot
AFTER one step together with what caused it, so the animation only has to walk
through the list. Frame 0 is the starting position.

Room 2 adds one thing Room 1 did not need: the layout changes during an episode.
Every frame therefore carries the full list of bridges that have collapsed so far,
which is what lets the replay redraw the sector correctly at any point without
re-running anything.
"""

from core import actions
from rooms.room2 import map_data

MAX_STEPS_DEFAULT = 200

STATUS_RUNNING = "Running"
STATUS_ESCAPED = "Escaped"
STATUS_FELL = "Fell into the shaft"
STATUS_MAX_STEPS = "Max steps reached"


def _frame(step, observation, from_cell, action, info, reward, cumulative_reward,
           done, status, collapsed):
    """Build one replay frame.

    Every field the replay viewer and the HUD need is stored explicitly, so a
    saved recording never depends on the environment still existing.
    """
    row, col, has_keycard = observation[0], observation[1], observation[2]
    from_row, from_col = from_cell
    return {
        "step": step,
        "state": list(observation),
        "row": row,
        "col": col,
        "has_keycard": has_keycard,
        # The renderer uses one generic name for "the thing this room collects",
        # so the same JavaScript serves Room 1's battery and Room 2's keycard.
        "has_item": has_keycard,
        "item_collected": bool(info.get("keycard_collected", False)),
        "from_row": from_row,
        "from_col": from_col,
        "action": action,
        "action_name": actions.direction_name(action) if action is not None else "-",
        # Room 2 is deterministic, so the robot always goes where it aimed.
        "actual_direction": action,
        "actual_direction_name": (actions.direction_name(action)
                                  if action is not None else "-"),
        "reward": reward,
        "cumulative_reward": cumulative_reward,
        "event": info.get("description", ""),
        "blocked": bool(info.get("blocked", False)),
        "door_locked": bool(info.get("door_locked", False)),
        "pit_fall": bool(info.get("pit_fall", False)),
        "bridge_collapsed": bool(info.get("bridge_collapsed", False)),
        "collapsed_cell": list(info["collapsed_cell"]) if info.get("collapsed_cell") else None,
        "keycard_collected": bool(info.get("keycard_collected", False)),
        "reached_exit": bool(info.get("reached_exit", False)),
        # Every bridge that has gone by this point in the episode.
        "collapsed": [list(cell) for cell in collapsed],
        "done": done,
        "status": status,
    }


def run_episode(env, policy, epsilon=0.0, seed=0, max_steps=MAX_STEPS_DEFAULT):
    """Follow `policy` for one episode and record every frame.

    `epsilon` is normally 0, so what gets recorded is the policy the agent
    actually learned. Raising it shows the behaviour policy instead, exploration
    mistakes included — which is what SARSA was valuing all along.
    """
    import random

    rng = random.Random(seed)
    observation = env.reset(seed=seed)
    cumulative_reward = 0.0

    start_info = {"description": "Powering up on the start platform"}
    frames = [_frame(0, observation, (observation[0], observation[1]), None,
                     start_info, 0.0, 0.0, False, STATUS_RUNNING, [])]

    pit_falls = 0
    wall_collisions = 0
    bridges_collapsed = 0
    keycard_collected = False
    success = False
    status = STATUS_MAX_STEPS

    for step in range(1, max_steps + 1):
        action = _choose(policy, observation, env, epsilon, rng)
        if action is None:
            break

        from_cell = (observation[0], observation[1])
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward

        if info["pit_fall"]:
            pit_falls += 1
        if info["blocked"]:
            wall_collisions += 1
        if info["bridge_collapsed"]:
            bridges_collapsed += 1
        if info["keycard_collected"]:
            keycard_collected = True

        if done and info["reached_exit"]:
            success = True
            status = STATUS_ESCAPED
        elif done and info["pit_fall"]:
            status = STATUS_FELL
        elif step == max_steps:
            status = STATUS_MAX_STEPS
        else:
            status = STATUS_RUNNING

        frames.append(_frame(step, observation, from_cell, action, info, reward,
                             cumulative_reward, done, status,
                             env.collapsed_cells()))
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
        "pit_falls": pit_falls,
        "wall_collisions": wall_collisions,
        "bridges_collapsed": bridges_collapsed,
        "keycard_collected": keycard_collected,
        "used_span": bridges_collapsed > 0,
        "map_hash": map_data.map_hash(),
    }


def _choose(policy, observation, env, epsilon, rng):
    """Pick an action from the policy, with `epsilon` chance of a random one."""
    if rng.random() < epsilon:
        return rng.choice(env.actions())
    return policy.get(observation)


def run_many(env, policy, episodes=30, first_seed=0, epsilon=0.0,
             max_steps=MAX_STEPS_DEFAULT):
    """Run several episodes and average the results.

    A single episode of a room where one wrong step is fatal says very little, so
    every number reported in the interface comes from a batch like this.
    """
    runs = [run_episode(env, policy, epsilon=epsilon, seed=first_seed + index,
                        max_steps=max_steps)
            for index in range(episodes)]
    successes = [run for run in runs if run["success"]]

    return {
        "episodes": episodes,
        "success_rate": len(successes) / episodes if episodes else 0.0,
        "mean_return": _mean([run["total_reward"] for run in runs]),
        "std_return": _standard_deviation([run["total_reward"] for run in runs]),
        "mean_steps": _mean([run["steps"] for run in runs]),
        "std_steps": _standard_deviation([run["steps"] for run in runs]),
        "mean_steps_when_successful": _mean([run["steps"] for run in successes]),
        "pit_fall_rate": _mean([1.0 if run["pit_falls"] else 0.0 for run in runs]),
        "keycard_rate": _mean([1.0 if run["keycard_collected"] else 0.0
                               for run in runs]),
        "span_use_rate": _mean([1.0 if run["used_span"] else 0.0 for run in runs]),
        "mean_wall_collisions": _mean([run["wall_collisions"] for run in runs]),
        "runs": runs,
    }


def describe_route(frames):
    """Name the route an episode took, from the landmarks it visited."""
    visited = {(frame["row"], frame["col"]) for frame in frames}
    used_span = any(frame["bridge_collapsed"] for frame in frames)
    fell = any(frame["pit_fall"] for frame in frames)

    parts = []
    if used_span or visited & set(map_data.collapsing_cells()):
        parts.append("Lower bridge")
    if visited & {(1, col) for col in range(map_data.GRID_COLS)}:
        parts.append("Upper walkway")
    if not parts:
        parts.append("Unclear")
    if fell:
        parts.append("(fell)")
    return " ".join(parts)


def _mean(numbers):
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def _standard_deviation(numbers):
    if len(numbers) < 2:
        return 0.0
    average = _mean(numbers)
    return (sum((number - average) ** 2 for number in numbers) / len(numbers)) ** 0.5
