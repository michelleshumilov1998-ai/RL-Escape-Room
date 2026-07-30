"""Turning Room 4 into a scene dictionary the renderers can draw.

Room 4 has no grid, but it is still drawn by the same isometric engine. The trick
is that the chamber is 10 x 10 metres, so **one metre is treated as one cell**:

    col = x                     row = height - y

That reuses the whole projection, every camera mode, the zoom, the pan and the
rotation, while the drone, the obstacles and the wind all sit at fractional
positions between cells. The 10 x 10 grid of plain floor tiles that comes with it
is exactly the metal decking the hall should have anyway.

`meta.sceneKind` is set to "continuous", which tells the engine to draw the fans,
the airflow, the obstacles, the landing pad and the drone instead of running the
per-tile switch.
"""

from core import story
from renderers import scene_common
from renderers.scene_common import PALETTE
from rooms.room4 import chamber
from rooms.room4.environment import ACTION_ARROWS, ACTION_NAMES
from ui.theme import COLORS


def world_to_cell(x, y):
    """Metres to the engine's cell coordinates."""
    return chamber.HEIGHT - y, x


def _floor_grid():
    """A plain deck, which is what the continuous hall stands on."""
    rows = int(round(chamber.HEIGHT))
    cols = int(round(chamber.WIDTH))
    return [["floor" for _ in range(cols)] for _ in range(rows)]


def _render_frames(frames):
    """Frames with the engine's generic fields added.

    `row` and `col` are derived from the metre position so the shared plumbing —
    the scene validator, the episode fingerprint, the follow camera — all keep
    working without knowing this room is continuous.
    """
    prepared = []
    for frame in frames:
        copy = dict(frame)
        row, col = world_to_cell(frame["x"], frame["y"])
        from_row, from_col = world_to_cell(frame.get("from_x", frame["x"]),
                                           frame.get("from_y", frame["y"]))
        copy["row"] = row
        copy["col"] = col
        copy["from_row"] = from_row
        copy["from_col"] = from_col
        copy["has_item"] = bool(frame.get("safe_landing", False))
        copy["item_collected"] = bool(frame.get("safe_landing", False))
        copy["action_name"] = frame.get("action_name", "-")
        copy["actual_direction_name"] = frame.get("action_name", "-")
        copy["previous_direction_name"] = "%.2f m/s" % frame.get("speed", 0.0)
        prepared.append(copy)
    return prepared


def build_scene(frames=None, options=None, status="TESTING", overlays=None,
                wind_multiplier=1.0, safe_landing_speed=0.7):
    """Build the scene dictionary for Room 4."""
    record = story.room(4)

    if not frames:
        start_x, start_y = chamber.START_POSITION
        frames = [{
            "step": 0, "x": start_x, "y": start_y, "vx": 0.0, "vy": 0.0,
            "from_x": start_x, "from_y": start_y, "speed": 0.0,
            "action": None, "action_name": "-", "action_arrow": "·",
            "thrust_x": 0.0, "thrust_y": 0.0, "wind_x": 0.0, "wind_y": 0.0,
            "turbulence_x": 0.0, "turbulence_y": 0.0,
            "reward": 0.0, "cumulative_reward": 0.0,
            "distance_to_goal": chamber.distance_to_goal(start_x, start_y),
            "collision": False, "boundary": False, "hard_landing": False,
            "safe_landing": False, "crashed": False, "timeout": False,
            "done": False, "event": "Drone link established — rotors up",
            "status": "Standing by",
        }]

    scene = {
        "meta": {
            "sector": record["sector"],
            "title": record["name"].upper(),
            "subtitle": record["subtitle"],
            "systemStatus": status.upper(),
            "roomNumber": 4,
            "accent": COLORS["purple"],
            "robotLabel": "R-5 DRONE LINK",
            "itemLabel": "Landing",
            "failLabel": "Collisions",
            "extraLabel": "Hard landings",
            "sceneKind": "continuous",
            "speedLabel": "Speed",
            "safeLandingSpeed": safe_landing_speed,
        },
        "grid": {
            "rows": int(round(chamber.HEIGHT)),
            "cols": int(round(chamber.WIDTH)),
            "tiles": _floor_grid(),
            "doors": {},
        },
        "continuous": chamber.geometry_as_dict(wind_multiplier),
        "landmarks": {
            "start": list(world_to_cell(*chamber.START_POSITION)),
            "exit": list(world_to_cell(*chamber.GOAL_POSITION)),
        },
        "frames": _render_frames(frames),
        "options": scene_common.merge_options(options),
        "palette": PALETTE,
        "deltas": scene_common.DELTAS,
        "directionNames": scene_common.DIRECTION_NAMES,
        "legend": [
            ["start", "Release point"],
            ["exit_door", "Landing pad"],
            ["wall", "Chamber wall"],
            ["floor", "Metal decking"],
            ["obstacle", "Obstacle"],
            ["fan", "Ventilation fan"],
            ["wind", "Airflow"],
        ],
        "actionNames": {str(key): value for key, value in ACTION_NAMES.items()},
        "actionArrows": {str(key): value for key, value in ACTION_ARROWS.items()},
        "overlays": overlays or {},
    }

    return scene_common.finalise(scene, 4, "continuous-%s" % chamber.WIDTH)
