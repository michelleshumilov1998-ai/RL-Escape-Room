"""Turning Room 3 into a scene dictionary the renderers can draw.

Same contract as the other rooms — see `renderers/base_renderer.py` — so the same
engine draws it with the same camera, controls and palette.

Room 3 adds two things the engine understands:

  * `stage` on every frame, which decides how each generator is lit and whether the
    blast door is open;
  * `guard_row` / `guard_col` on every frame, so the security robot is animated from
    the recording rather than being re-simulated.
"""

from core import actions, story
from renderers import scene_common
from renderers.scene_common import PALETTE
from rooms.room3 import map_data
from ui.theme import COLORS


def _tile_kinds():
    return [[map_data.tile_kind(map_data.tile_at(row, col))
             for col in range(map_data.GRID_COLS)]
            for row in range(map_data.GRID_ROWS)]


def _legend():
    return [[map_data.tile_kind(tile), name] for tile, name in map_data.LEGEND]


def _prepare(frames):
    """Room 3 has no previous direction in its state, so that HUD row is reused.

    The engine draws one generic row labelled by `meta.speedLabel`; here it shows
    whether the sliding doors are open at this point in the patrol, which is the
    thing a viewer actually needs while watching R-5 wait beside one.
    """
    prepared = []
    for frame in frames:
        copy = dict(frame)
        copy["previous_direction_name"] = ("OPEN" if frame.get("doors_open")
                                           else "SHUT")
        prepared.append(copy)
    return prepared


def build_scene(frames=None, options=None, status="OFFLINE", overlays=None):
    """Build the scene dictionary for Room 3."""
    record = story.room(3)
    start_row, start_col = map_data.start_cell()
    guard_row, guard_col = map_data.guard_cell(0)

    if not frames:
        frames = [{
            "step": 0,
            "row": start_row, "col": start_col,
            "from_row": start_row, "from_col": start_col,
            "stage": 0, "guard_index": 0,
            "guard_row": guard_row, "guard_col": guard_col,
            "doors_open": bool(map_data.door_is_open(0)),
            "has_item": False, "item_collected": False,
            "action": None, "action_name": "-",
            "actual_direction": None, "actual_direction_name": "-",
            "previous_direction": actions.NONE, "previous_direction_name": "NONE",
            "reward": 0.0, "cumulative_reward": 0.0,
            "event": "Reactor offline — R-5 entering the chamber",
            "blocked": False, "door_blocked": False, "waited": False,
            "hazard": False, "caught": False, "generator": None,
            "unlocked": False, "wrong_order": False, "reached_exit": False,
            "done": False, "status": "Standing by",
        }]

    scene = {
        "meta": {
            "sector": record["sector"],
            "title": record["name"].upper(),
            "subtitle": record["subtitle"],
            "systemStatus": status.upper(),
            "roomNumber": 3,
            "accent": COLORS["danger"],
            "robotLabel": "R-5",
            "itemLabel": "Reactor",
            "failLabel": "Detections",
            "extraLabel": "Hazards",
            # Room 3 draws a second robot, and the engine needs telling.
            "hasGuard": True,
            "guardLabel": "SEC-1",
            "speedLabel": "Doors",
        },
        "grid": {
            "rows": map_data.GRID_ROWS,
            "cols": map_data.GRID_COLS,
            "tiles": _tile_kinds(),
            "doors": {},
        },
        "landmarks": {
            "start": list(map_data.start_cell()),
            "exit": list(map_data.exit_cell()),
            "generators": {
                "a": list(map_data.generator_cell(map_data.GENERATOR_A)),
                "b": list(map_data.generator_cell(map_data.GENERATOR_B)),
                "c": list(map_data.generator_cell(map_data.GENERATOR_C)),
            },
            "reactorDoor": list(map_data.reactor_door_cell()),
            "hazards": [list(cell) for cell in map_data.hazard_cells()],
            "slidingDoors": [list(cell) for cell in map_data.sliding_door_cells()],
            "patrol": [list(cell) for cell in map_data.PATROL],
        },
        "frames": _prepare(frames),
        "options": scene_common.merge_options(options),
        "palette": PALETTE,
        "deltas": scene_common.DELTAS,
        "directionNames": scene_common.DIRECTION_NAMES,
        "legend": _legend(),
        "overlays": overlays or {},
    }

    return scene_common.finalise(scene, 3, map_data.map_hash())
