"""Saving and loading trained Room 2 models.

The same approach as Room 1: plain JSON, a format version, a fingerprint of the
map, and a check that the state format matches before anything is loaded.

Room 2 stores a Q-table rather than a value table. Its keys are tuples of three
or four numbers depending on the state representation, and JSON has no tuples, so
each state is written as a string like "5,4,1" or "5,4,1,3", and the action is
folded into the key as well.
"""

import json
import os

from rooms.room2 import map_data

FORMAT_VERSION = 1

SAVE_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "saves")


class ModelFileError(Exception):
    """Raised when a saved model cannot be used with the current room."""


def _state_to_key(state):
    """Turn a state tuple into a string JSON can use as a dictionary key."""
    parts = [str(int(component)) if isinstance(component, bool) else str(component)
             for component in state]
    return ",".join(parts)


def _key_to_state(key):
    """Turn the string back into the original state tuple.

    The third component is the keycard flag and has to come back as a bool, or
    dictionary lookups against a live environment would silently miss.
    """
    numbers = [int(part) for part in key.split(",")]
    state = [numbers[0], numbers[1], bool(numbers[2])]
    state.extend(numbers[3:])
    return tuple(state)


def save_path(name):
    if not name.endswith(".json"):
        name = name + ".json"
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    return os.path.join(SAVE_DIRECTORY, name)


def list_saved_files():
    if not os.path.isdir(SAVE_DIRECTORY):
        return []
    return sorted(name for name in os.listdir(SAVE_DIRECTORY)
                  if name.endswith(".json"))


def save_model(name, env, result):
    """Write one trained model to disk and return the path it went to."""
    q_as_strings = {}
    for state, actions_for_state in result["q"].items():
        for action, value in actions_for_state.items():
            q_as_strings["%s|%d" % (_state_to_key(state), action)] = value

    payload = {
        "format_version": FORMAT_VERSION,
        "room": 2,
        "room_name": map_data.ROOM_NAME,
        "map_hash": map_data.map_hash(),
        "map": list(map_data.ROOM_MAP),
        "algorithm": result["algorithm"],
        "state_format": ("row,col,has_keycard,collapsed_mask"
                         if env.include_bridge_state else "row,col,has_keycard"),
        "include_bridge_state": env.include_bridge_state,
        "state_space_size": len(result["q"]),
        "episodes": result["episodes"],
        "runtime_seconds": result["runtime_seconds"],
        "final_epsilon": result["final_epsilon"],
        "mean_abs_q": result["mean_abs_q"],
        "hyperparameters": result["hyperparameters"],
        "environment_parameters": result["environment_parameters"],
        "history": result["history"],
        "q": q_as_strings,
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
            raise ModelFileError("'%s' is missing the field '%s'."
                                 % (name, required))

    if payload["format_version"] != FORMAT_VERSION:
        raise ModelFileError(
            "'%s' uses file format %s, but this version reads format %d."
            % (name, payload["format_version"], FORMAT_VERSION))

    if payload["map_hash"] != map_data.map_hash():
        raise ModelFileError(
            "'%s' was trained on a different version of the map, so its policy "
            "would not make sense here." % name)

    if payload.get("room") != 2:
        raise ModelFileError("'%s' is not a Room 2 model." % name)

    # Rebuild the nested Q-table.
    q = {}
    for combined_key, value in payload["q"].items():
        state_part, action_part = combined_key.rsplit("|", 1)
        state = _key_to_state(state_part)
        q.setdefault(state, {})[int(action_part)] = value

    model = dict(payload)
    model["q"] = q
    model["policy"] = None      # rebuilt by the caller against a live environment
    return model
