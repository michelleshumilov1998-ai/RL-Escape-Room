"""Saving and loading trained Room 5 models.

The weights only mean anything alongside the exact feature extractor that produced
them, so a file is refused unless the feature groups, the bin counts, the radar
configuration, the action count and the vector length all match.
"""

import json
import os

FORMAT_VERSION = 1
ENVIRONMENT_VERSION = "room5-warehouse-1"

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
    agent = result["agent"]
    payload = {
        "format_version": FORMAT_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "room": 5,
        "algorithm": result["algorithm"],
        "features": result["features"],
        "feature_set": result["feature_set"],
        "num_actions": agent.extractor.num_actions,
        "weight_count": len(agent.weights),
        "radar_range": result["environment_parameters"]["radar_range"],
        "difficulty": result["difficulty"],
        "hyperparameters": result["hyperparameters"],
        "environment_parameters": result["environment_parameters"],
        "training_seeds": result["training_seeds"],
        "validation_seeds": result["validation_seeds"],
        "episodes": result["episodes"],
        "runtime_seconds": result["runtime_seconds"],
        "weight_norm": result["weight_norm"],
        "active_features": result["active_features"],
        "history": result["history"],
        "weights": {str(index): value for index, value in enumerate(agent.weights)
                    if value != 0.0},
    }
    path = save_path(name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def load_into(name, agent, radar_range=None):
    """Read a saved model into `agent`, validating everything first.

    The agent is left untouched if the file is incompatible.
    """
    path = save_path(name)
    if not os.path.isfile(path):
        raise ModelFileError("No saved model named '%s'." % name)

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as problem:
        raise ModelFileError("'%s' could not be read: %s" % (name, problem))

    for required in ("format_version", "weights", "features", "algorithm"):
        if required not in payload:
            raise ModelFileError("'%s' is missing the field '%s'."
                                 % (name, required))

    if payload["format_version"] != FORMAT_VERSION:
        raise ModelFileError(
            "'%s' uses file format %s, but this version reads format %d."
            % (name, payload["format_version"], FORMAT_VERSION))
    if payload.get("room") != 5:
        raise ModelFileError("'%s' is not a Room 5 model." % name)
    if payload.get("environment_version") != ENVIRONMENT_VERSION:
        raise ModelFileError(
            "'%s' was trained against environment %s, but this is %s."
            % (name, payload.get("environment_version"), ENVIRONMENT_VERSION))
    if payload.get("num_actions") != agent.extractor.num_actions:
        raise ModelFileError("'%s' has %s actions but this room has %d."
                             % (name, payload.get("num_actions"),
                                agent.extractor.num_actions))
    if payload.get("weight_count") != len(agent.weights):
        raise ModelFileError(
            "'%s' holds %s weights but this feature set needs %d."
            % (name, payload.get("weight_count"), len(agent.weights)))
    if not agent.extractor.matches(payload["features"]):
        raise ModelFileError(
            "'%s' was built with a different feature representation, so its "
            "weights would mean something else here." % name)
    if radar_range is not None and payload.get("radar_range") != radar_range:
        raise ModelFileError(
            "'%s' was trained with radar range %s, but this room is set to %s."
            % (name, payload.get("radar_range"), radar_range))

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
