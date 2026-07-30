"""Turning Room 5 into a scene dictionary the renderers can draw.

Same contract as the other rooms. Room 5 adds three things the engine understands:

  * `fogOfWar` on the meta — in gameplay mode, cells the radar has never reached are
    hidden and cells it has seen but cannot currently see are dimmed. That matches
    what the agent actually knows.
  * `robots` on every frame, so the maintenance robots animate from the recording.
  * `radar` on every frame, so the scan rays can be drawn from the real readings.
"""

from core import story
from renderers import scene_common
from renderers.scene_common import PALETTE
from rooms.room5 import layout as L
from rooms.room5.environment import ACTION_ARROWS, ACTION_NAMES, RADAR_DIRECTIONS
from ui.theme import COLORS


def _tile_kinds(layout):
    return [[L.TILE_KINDS[layout.tile_at(row, col)]
             for col in range(layout.size)]
            for row in range(layout.size)]


def _conveyor_directions(layout):
    """Which way each conveyor pushes, keyed by "row,col"."""
    directions = {}
    for row in range(layout.size):
        for col in range(layout.size):
            delta = layout.conveyor_at(row, col)
            if delta is not None:
                directions["%d,%d" % (row, col)] = list(delta)
    return directions


def _prepare(frames):
    """Fill in the HUD's previous-action row from the frame before each one.

    Room 5's observation really does include the previous action — it is five of
    the twenty-two numbers the agent sees — so the row is worth showing properly
    rather than leaving it reading NONE for the whole mission.
    """
    prepared = []
    previous_name = "NONE"
    for frame in frames:
        copy = dict(frame)
        copy["previous_direction_name"] = previous_name
        previous_name = frame.get("action_name") or "NONE"
        prepared.append(copy)
    return prepared


def build_scene(layout, frames=None, options=None, status="RECONFIGURING",
                overlays=None, fog_of_war=True):
    """Build the scene dictionary for Room 5."""
    record = story.room(5)

    if not frames:
        start_row, start_col = layout.start
        frames = [{
            "step": 0, "row": start_row, "col": start_col,
            "from_row": start_row, "from_col": start_col,
            "stage": 0, "current_target": list(layout.terminal),
            "action": None, "action_name": "-", "action_arrow": "·",
            "reward": 0.0, "cumulative_reward": 0.0,
            "radar_observation": [1.0] * 8, "dynamic_radar": [1.0] * 4,
            "nearest_robot": 1.0, "target_features": [0.0, 0.0, 0.0],
            "visible_cells": [], "seen_cells": [],
            "moving_robot_positions": [list(robot.cell())
                                       for robot in layout.robot_templates],
            "moving_robot_directions": [robot.pattern
                                        for robot in layout.robot_templates],
            "conveyor_event": False, "collision": False,
            "terminal_activated": False, "exit_unlocked": False,
            "early_exit": False, "charged": False, "caught": False,
            "reached_exit": False, "timeout": False,
            "has_item": False, "item_collected": False,
            "done": False, "success": False,
            "event": "Radar online — warehouse layout unknown",
            "status": "Standing by",
        }]

    scene = {
        "meta": {
            "sector": record["sector"],
            "title": record["name"].upper(),
            "subtitle": record["subtitle"],
            "systemStatus": status.upper(),
            "roomNumber": 5,
            "accent": COLORS["success"],
            "robotLabel": "R-5",
            "itemLabel": "Terminal",
            "failLabel": "Collisions",
            "extraLabel": "Conveyors",
            "guardLabel": "MNT",
            "hasGuard": False,          # Room 5 draws its own robot list instead
            "hasRobots": True,
            "fogOfWar": bool(fog_of_war),
            "radarRange": layout.radar_range,
            "layoutSeed": layout.seed,
            "layoutSplit": layout.split,
        },
        "grid": {
            "rows": layout.size,
            "cols": layout.size,
            "tiles": _tile_kinds(layout),
            "doors": {},
            "conveyors": _conveyor_directions(layout),
        },
        "landmarks": {
            "start": list(layout.start),
            "exit": list(layout.exit_cell),
            "terminal": list(layout.terminal),
            "robotRoutes": [[list(cell) for cell in robot.route]
                            for robot in layout.robot_templates],
        },
        "frames": _prepare(frames),
        "options": scene_common.merge_options(options),
        "palette": PALETTE,
        "deltas": scene_common.DELTAS,
        "directionNames": scene_common.DIRECTION_NAMES,
        "radarDirections": [list(direction) for direction in RADAR_DIRECTIONS],
        "legend": [[L.TILE_KINDS[tile], name] for tile, name in L.LEGEND],
        "actionNames": {str(key): value for key, value in ACTION_NAMES.items()},
        "actionArrows": {str(key): value for key, value in ACTION_ARROWS.items()},
        "overlays": overlays or {},
    }

    return scene_common.finalise(scene, 5, "layout-%s" % layout.seed)
