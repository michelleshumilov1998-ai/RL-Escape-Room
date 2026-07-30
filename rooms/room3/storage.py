"""Saving and loading trained Room 3 models.

Same approach as the other rooms: plain JSON, a format version, a fingerprint of
the map, and a check that the room and the state format match before anything is
loaded.

Room 3's state is four integers — row, column, stage and the guard's patrol index
— written as a string like "7,1,2,15" because JSON has no tuples.
"""

import json
import os

from rooms.room3 import map_data

FORMAT_VERSION = 1

SAVE_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "saves")


class ModelFileError(Exception):
    """Raised when a saved model cannot be used with the current room."""


def _state_to_key(state):
    return ",".join(str(int(component)) for component in state)


def _key_to_state(key):
    return tuple(int(part) for part in key.split(","))


def save_path(name):
    if not name.endswith(".json"):
        name = name + ".json"
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    return os.path.join(SAVE_DIRECTORY, name)


def save_model(name, env, result):
    """Write one trained model to disk and return the path it went to."""
    flat = {}
    for state, actions_for_state in result["q"].items():
        for action, value in actions_for_state.items():
            flat["%s|%d" % (_state_to_key(state), action)] = value

    payload = {
        "format_version": FORMAT_VERSION,
        "room": 3,
        "room_name": map_data.ROOM_NAME,
        "map_hash": map_data.map_hash(),
        "map": list(map_data.ROOM_MAP),
        "algorithm": result["algorithm"],
        "state_format": "row,col,stage,guard_index",
        "patrol_length": map_data.PATROL_LENGTH,
        "state_space_size": len(result["q"]),
        "episodes": result["episodes"],
        "runtime_seconds": result["runtime_seconds"],
        "final_epsilon": result["final_epsilon"],
        "mean_abs_q": result["mean_abs_q"],
        "hyperparameters": result["hyperparameters"],
        "environment_parameters": result["environment_parameters"],
        "history": result["history"],
        "q": flat,
    }

    path = save_path(name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    return path


def load_model(name):
    """Read a saved model back, checking that it still fits this room."""
    path = save_path(name)
    if not os.path.isfile(path):
        raise ModelFileError("No saved model named '%s'." % name)

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as problem:
        raise ModelFileError("'%s' could not be read: %s" % (name, problem))

    for required in ("format_version", "map_hash", "q", "algorithm"):
        if required not in payload:
            raise ModelFileError("'%s' is missing the field '%s'." % (name, required))

    if payload["format_version"] != FORMAT_VERSION:
        raise ModelFileError(
            "'%s' uses file format %s, but this version reads format %d."
            % (name, payload["format_version"], FORMAT_VERSION))
    if payload["map_hash"] != map_data.map_hash():
        raise ModelFileError(
            "'%s' was trained on a different version of the map, so its policy "
            "would not make sense here." % name)
    if payload.get("room") != 3:
        raise ModelFileError("'%s' is not a Room 3 model." % name)
    if payload.get("patrol_length") != map_data.PATROL_LENGTH:
        raise ModelFileError(
            "'%s' was trained against a %s-cell patrol, but this room has %d."
            % (name, payload.get("patrol_length"), map_data.PATROL_LENGTH))

    q = {}
    for combined, value in payload["q"].items():
        state_part, action_part = combined.rsplit("|", 1)
        q.setdefault(_key_to_state(state_part), {})[int(action_part)] = value

    model = dict(payload)
    model["q"] = q
    model["policy"] = None      # rebuilt by the caller against a live environment
    return model
