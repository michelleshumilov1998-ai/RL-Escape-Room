"""Saving and loading trained Room 4 models.

Room 4 stores a weight vector rather than a Q-table, so loading has more to check:
the weights only mean anything alongside the exact tile coder that produced them.
A file is refused unless the shape, the action count, the tile-coder configuration
and the state bounds all match.
"""

import json
import os

from rooms.room4 import chamber

FORMAT_VERSION = 1
ENVIRONMENT_VERSION = "room4-continuous-1"

SAVE_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "saves")


class ModelFileError(Exception):
    """Raised when a saved model cannot be used with the current room."""


def save_path(name):
    if not name.endswith(".json"):
        name = name + ".json"
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    return os.path.join(SAVE_DIRECTORY, name)


def save_model(name, result):
    """Write one trained model to disk and return the path it went to."""
    agent = result["agent"]
    payload = {
        "format_version": FORMAT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "room": 4,
        "algorithm": result["algorithm"],
        "tile_coder": result["tile_coder"],
        "num_actions": agent.coder.num_actions,
        "weight_count": len(agent.weights),
        "state_bounds": [list(pair) for pair in agent.coder.bounds],
        "hyperparameters": result["hyperparameters"],
        "environment_parameters": result["environment_parameters"],
        "training_starts": result["training_starts"],
        "chamber": {
            "width": chamber.WIDTH, "height": chamber.HEIGHT,
            "goal": list(chamber.GOAL_POSITION),
            "goal_radius": chamber.GOAL_RADIUS,
            "obstacles": [obstacle.as_dict() for obstacle in chamber.OBSTACLES],
            "fans": [fan.as_dict() for fan in chamber.FANS],
        },
        "episodes": result["episodes"],
        "runtime_seconds": result["runtime_seconds"],
        "weight_norm": result["weight_norm"],
        "active_features": result["active_features"],
        "history": result["history"],
        # Only the non-zero weights are written. The vector is mostly empty and
        # storing all 160k floats would make the file needlessly enormous.
        "weights": {str(index): value
                    for index, value in enumerate(agent.weights)
                    if value != 0.0},
    }

    path = save_path(name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def load_into(name, agent):
    """Read a saved model back into `agent`, checking it is compatible.

    Returns the file's payload. Raises ModelFileError, without touching the
    agent, if anything about the file does not match.
    """
    path = save_path(name)
    if not os.path.isfile(path):
        raise ModelFileError("No saved model named '%s'." % name)

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as problem:
        raise ModelFileError("'%s' could not be read: %s" % (name, problem))

    for required in ("format_version", "weights", "tile_coder", "algorithm"):
        if required not in payload:
            raise ModelFileError("'%s' is missing the field '%s'." % (name, required))

    if payload["format_version"] != FORMAT_VERSION:
        raise ModelFileError(
            "'%s' uses file format %s, but this version reads format %d."
            % (name, payload["format_version"], FORMAT_VERSION))
    if payload.get("room") != 4:
        raise ModelFileError("'%s' is not a Room 4 model." % name)
    if payload.get("environment_version") != ENVIRONMENT_VERSION:
        raise ModelFileError(
            "'%s' was trained against environment %s, but this is %s."
            % (name, payload.get("environment_version"), ENVIRONMENT_VERSION))
    if payload.get("num_actions") != agent.coder.num_actions:
        raise ModelFileError(
            "'%s' has %s actions but this room has %d."
            % (name, payload.get("num_actions"), agent.coder.num_actions))
    if payload.get("weight_count") != len(agent.weights):
        raise ModelFileError(
            "'%s' holds %s weights but this tile coder needs %d — the tile "
            "configuration must match." % (name, payload.get("weight_count"),
                                           len(agent.weights)))
    if not agent.coder.matches(payload["tile_coder"]):
        raise ModelFileError(
            "'%s' was built with a different tile coder, so its weights would "
            "mean something else here." % name)

    weights = [0.0] * len(agent.weights)
    for index, value in payload["weights"].items():
        position = int(index)
        if not 0 <= position < len(weights):
            raise ModelFileError("'%s' has a weight index outside the vector."
                                 % name)
        weights[position] = value

    agent.weights = weights
    agent._sum_squares = sum(value * value for value in weights)
    return payload
