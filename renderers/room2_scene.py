"""Turning Room 2 into a scene dictionary the renderers can draw.

Exactly the same contract as Room 1's scene builder — see
`renderers/base_renderer.py` — so both rooms are drawn by the same engine and
share every colour, camera and control.

Room 2 adds one field the engine understands: each frame carries `collapsed`, the
list of bridges that have already fallen at that point in the episode. The engine
draws those cells as holes, which is how the layout changing mid-episode survives
into the replay.
"""

from core import actions, story
from renderers import scene_common
from renderers.scene_common import PALETTE
from rooms.room2 import map_data
from ui.theme import COLORS


def _tile_kinds():
    """The 10x10 grid of machine-readable tile names."""
    grid = []
    for row in range(map_data.GRID_ROWS):
        line = []
        for col in range(map_data.GRID_COLS):
            line.append(map_data.tile_kind(map_data.tile_at(row, col)))
        grid.append(line)
    return grid


def _legend():
    return [[map_data.tile_kind(tile), name] for tile, name in map_data.LEGEND]


def build_scene(frames=None, options=None, status="DAMAGED", overlays=None):
    """Build the scene dictionary for Room 2.

    `frames` is a recorded episode from `rooms.room2.simulate`.  When it is None
    the sector is drawn with R-5 standing on the start platform and every bridge
    still intact.
    """
    record = story.room(2)
    start_row, start_col = map_data.start_cell()

    if not frames:
        frames = [{
            "step": 0,
            "row": start_row, "col": start_col,
            "from_row": start_row, "from_col": start_col,
            "has_keycard": False, "has_item": False, "item_collected": False,
            "action": None, "action_name": "-",
            "actual_direction": None, "actual_direction_name": "-",
            "previous_direction": actions.NONE, "previous_direction_name": "NONE",
            "reward": 0.0, "cumulative_reward": 0.0,
            "event": "Powering up on the start platform",
            "blocked": False, "door_locked": False, "pit_fall": False,
            "bridge_collapsed": False, "collapsed_cell": None,
            "keycard_collected": False, "reached_exit": False,
            "collapsed": [], "done": False, "status": "Standing by",
        }]

    scene = {
        "meta": {
            "sector": record["sector"],
            "title": record["name"].upper(),
            "subtitle": record["subtitle"],
            "systemStatus": status.upper(),
            "roomNumber": 2,
            "accent": COLORS["warning"],
            "robotLabel": "R-5",
            # Labels for the three counters in the heads-up display.
            "itemLabel": "Keycard",
            "failLabel": "Pit falls",
            "extraLabel": "Bridges lost",
        },
        "grid": {
            "rows": map_data.GRID_ROWS,
            "cols": map_data.GRID_COLS,
            "tiles": _tile_kinds(),
            # Room 2 has no one-way doors, but the field is part of the contract.
            "doors": {},
        },
        "landmarks": {
            "start": list(map_data.start_cell()),
            "exit": list(map_data.exit_cell()),
            "keycard": list(map_data.keycard_cell()),
            "collapsing": [list(cell) for cell in map_data.collapsing_cells()],
            "pits": [list(cell) for cell in map_data.pit_cells()],
        },
        "frames": [dict(frame) for frame in frames],
        "options": scene_common.merge_options(options),
        "palette": PALETTE,
        "deltas": scene_common.DELTAS,
        "directionNames": scene_common.DIRECTION_NAMES,
        "legend": _legend(),
        "overlays": overlays or {},
    }

    return scene_common.finalise(scene, 2, map_data.map_hash())
