"""Turning Room 1 into a scene dictionary the renderers can draw.

This module is pure data preparation.  It imports the map and the theme, but
never an agent, a solver or a policy, and it never changes anything it is given.

The palette, the default view options and the fingerprints are shared with the
other rooms and live in `renderers/scene_common.py`.
"""

from core import actions, story, tiles
from renderers import scene_common
from renderers.scene_common import DEFAULT_OPTIONS, PALETTE  # noqa: F401 (re-export)
from rooms.room1 import map_data
from ui.theme import COLORS


def _tile_kinds():
    """The 10x10 grid of machine-readable tile names."""
    grid = []
    for row in range(map_data.GRID_ROWS):
        line = []
        for col in range(map_data.GRID_COLS):
            line.append(tiles.tile_kind(map_data.tile_at(row, col)))
        grid.append(line)
    return grid


def _door_directions():
    """Which way each one-way door may be crossed, keyed by "row,col"."""
    doors = {}
    for row in range(map_data.GRID_ROWS):
        for col in range(map_data.GRID_COLS):
            tile = map_data.tile_at(row, col)
            if tiles.is_one_way_door(tile):
                doors["%d,%d" % (row, col)] = tiles.door_direction(tile)
    return doors


def _legend():
    """The legend shown under the chamber."""
    return [[tiles.tile_kind(tile), name] for tile, name in map_data.LEGEND]


def _render_frames(frames):
    """Copies of the frames with the render-only fields the engine expects.

    The recorded frames belong to the replay format and are never modified; these
    are shallow copies with two generic names added, so one JavaScript engine can
    serve Room 1's battery and Room 2's keycard without knowing about either.
    """
    prepared = []
    for frame in frames:
        copy = dict(frame)
        copy["has_item"] = frame["has_battery"]
        copy["item_collected"] = frame["battery_collected"]
        prepared.append(copy)
    return prepared


def build_scene(frames=None, options=None, status="ACTIVE", overlays=None):
    """Build the scene dictionary for Room 1.

    `frames` is a recorded episode from `rooms.room1.simulate`.  When it is None
    the chamber is drawn with R-5 standing on the start platform.
    """
    record = story.room(1)
    start_row, start_col = map_data.start_cell()

    if not frames:
        # A single resting frame, so the chamber is never drawn empty.
        frames = [{
            "step": 0,
            "row": start_row, "col": start_col,
            "from_row": start_row, "from_col": start_col,
            "has_battery": False,
            "previous_direction": actions.NONE,
            "previous_direction_name": "NONE",
            "action": None, "action_name": "-",
            "actual_direction": None, "actual_direction_name": "-",
            "reward": 0.0, "cumulative_reward": 0.0,
            "event": "Standing by on the start platform",
            "laser_hit": False, "slipped": False, "momentum": False,
            "blocked": False, "used_teleport": False,
            "battery_collected": False, "reached_exit": False,
            "done": False, "status": "Standing by",
        }]

    scene = {
        "meta": {
            "sector": record["sector"],
            "title": record["name"].upper(),
            "subtitle": record["subtitle"],
            "systemStatus": status.upper(),
            "roomNumber": 1,
            "accent": COLORS["cyan"],
            "robotLabel": "R-5",
            # Labels for the three counters in the heads-up display.
            "itemLabel": "Battery",
            "failLabel": "Laser hits",
            "extraLabel": "Teleports",
        },
        "grid": {
            "rows": map_data.GRID_ROWS,
            "cols": map_data.GRID_COLS,
            "tiles": _tile_kinds(),
            "doors": _door_directions(),
        },
        "landmarks": {
            "start": list(map_data.start_cell()),
            "exit": list(map_data.exit_cell()),
            "pads": [list(pad) for pad in map_data.teleport_cells()],
            "battery": list(map_data.battery_cell()),
            "lasers": [list(cell) for cell in map_data.find_tiles(tiles.LASER)],
        },
        "frames": _render_frames(frames),
        "options": scene_common.merge_options(options),
        "palette": PALETTE,
        "deltas": scene_common.DELTAS,
        "directionNames": scene_common.DIRECTION_NAMES,
        "legend": _legend(),
        "overlays": overlays or {},
    }

    return scene_common.finalise(scene, 1, map_data.map_hash())
