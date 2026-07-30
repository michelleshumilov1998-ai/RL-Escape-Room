"""Saving and loading solved Room 1 results.

Results are stored as plain JSON so they can be inspected in any text editor.

The state of Room 1 is a tuple, and JSON has no tuples, so every state is turned
into a short string like "4,9,1,2" and back again.

Every file records a format version, a fingerprint of the map and the size of
the state space.  Loading checks all three, so an old result can never be shown
on top of a map or a state representation that has changed since.
"""

import json
import os

from rooms.room1 import map_data

FORMAT_VERSION = 1

SAVE_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "saves")


class SolutionFileError(Exception):
    """Raised when a saved file cannot be used with the current room."""


def _state_to_key(state):
    """Turn the state tuple into a string JSON can use as a dictionary key."""
    row, col, has_battery, previous_direction = state
    return "%d,%d,%d,%d" % (row, col, int(has_battery), previous_direction)


def _key_to_state(key):
    """Turn the string back into the original state tuple."""
    row, col, has_battery, previous_direction = key.split(",")
    return (int(row), int(col), bool(int(has_battery)), int(previous_direction))


def save_path(name):
    """Full path of a save file, created inside the project's saves folder."""
    if not name.endswith(".json"):
        name = name + ".json"
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    return os.path.join(SAVE_DIRECTORY, name)


def list_saved_files():
    """Every saved result currently on disk, newest name order last."""
    if not os.path.isdir(SAVE_DIRECTORY):
        return []
    names = [name for name in os.listdir(SAVE_DIRECTORY) if name.endswith(".json")]
    return sorted(names)


def save_solution(name, env, solution):
    """Write one solved result to disk and return the path it went to."""
    payload = {
        "format_version": FORMAT_VERSION,
        "room": 1,
        "room_name": map_data.ROOM_NAME,
        "map_hash": map_data.map_hash(),
        "map": list(map_data.ROOM_MAP),
        "state_space_size": len(solution["values"]),
        "state_format": "row,col,has_battery,previous_direction",
        "algorithm": solution["algorithm"],
        "environment_parameters": env.parameter_summary(),
        "gamma": solution["gamma"],
        "theta": solution["theta"],
        "iterations": solution["iterations"],
        "runtime_seconds": solution["runtime_seconds"],
        "start_state_value": solution["start_state_value"],
        "bellman_residual": solution["bellman_residual"],
        "converged": solution["converged"],
        "delta_history": solution["delta_history"],
        "start_value_history": solution["start_value_history"],
        "policy_change_history": solution["policy_change_history"],
        "values": {_state_to_key(state): value
                   for state, value in solution["values"].items()},
        "policy": {_state_to_key(state): action
                   for state, action in solution["policy"].items()},
    }

    path = save_path(name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    return path


def load_solution(name):
    """Read a saved result back, checking that it still fits this room.

    Raises SolutionFileError with a readable message when it does not.
    """
    path = save_path(name)
    if not os.path.isfile(path):
        raise SolutionFileError("No saved result named '%s'." % name)

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as problem:
        raise SolutionFileError("'%s' could not be read: %s" % (name, problem))

    for required in ("format_version", "map_hash", "values", "policy", "algorithm"):
        if required not in payload:
            raise SolutionFileError("'%s' is missing the field '%s'." % (name, required))

    if payload["format_version"] != FORMAT_VERSION:
        raise SolutionFileError(
            "'%s' uses file format %s, but this version reads format %d."
            % (name, payload["format_version"], FORMAT_VERSION))

    if payload["map_hash"] != map_data.map_hash():
        raise SolutionFileError(
            "'%s' was solved on a different version of the map, so its policy "
            "would not make sense here." % name)

    solution = dict(payload)
    solution["values"] = {_key_to_state(key): value
                          for key, value in payload["values"].items()}
    solution["policy"] = {_key_to_state(key): action
                          for key, action in payload["policy"].items()}

    expected_states = payload.get("state_space_size")
    if expected_states is not None and len(solution["values"]) != expected_states:
        raise SolutionFileError(
            "'%s' holds %d states but says it should hold %d."
            % (name, len(solution["values"]), expected_states))

    return solution
