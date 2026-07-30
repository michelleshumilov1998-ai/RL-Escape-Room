"""The tile alphabet used by the grid rooms of PROJECT R-5.

Every room map is a list of strings.  Each character is one tile.  This module
gives those characters names, so the rest of the code never has to compare
against a raw character like '%'.
"""

from core import actions

START = "S"
EXIT = "E"
WALL = "#"
FLOOR = "."
WEAK_ICE = "~"
STRONG_ICE = "%"
OIL = "O"
LASER = "L"
BATTERY = "B"
TELEPORT = "T"

# The four one-way doors.  Each one may only be crossed in its own direction.
DOOR_RIGHT = ">"
DOOR_LEFT = "<"
DOOR_UP = "^"
DOOR_DOWN = "v"

# Which direction each one-way door allows.
DOOR_DIRECTIONS = {
    DOOR_RIGHT: actions.RIGHT,
    DOOR_LEFT: actions.LEFT,
    DOOR_UP: actions.UP,
    DOOR_DOWN: actions.DOWN,
}

ONE_WAY_DOORS = tuple(DOOR_DIRECTIONS.keys())

# Every character a valid map may contain.
VALID_TILES = (
    START,
    EXIT,
    WALL,
    FLOOR,
    WEAK_ICE,
    STRONG_ICE,
    OIL,
    LASER,
    BATTERY,
    TELEPORT,
) + ONE_WAY_DOORS

# Names shown in the interface legend.
TILE_NAMES = {
    START: "Start platform",
    EXIT: "Control panel / Exit",
    WALL: "Metallic wall",
    FLOOR: "Normal floor",
    WEAK_ICE: "Wet floor (weak ice)",
    STRONG_ICE: "Frozen floor (strong ice)",
    OIL: "Oil spill",
    LASER: "Laser beam",
    BATTERY: "Battery",
    TELEPORT: "Maintenance teleporter",
    DOOR_RIGHT: "One-way door (rightwards)",
    DOOR_LEFT: "One-way door (leftwards)",
    DOOR_UP: "One-way door (upwards)",
    DOOR_DOWN: "One-way door (downwards)",
}

# Short machine-readable names, used by the renderer so the JavaScript side
# never has to deal with characters like '%' or '~'.
TILE_KINDS = {
    START: "start",
    EXIT: "exit",
    WALL: "wall",
    FLOOR: "floor",
    WEAK_ICE: "weak_ice",
    STRONG_ICE: "strong_ice",
    OIL: "oil",
    LASER: "laser",
    BATTERY: "battery",
    TELEPORT: "teleport",
    DOOR_RIGHT: "door",
    DOOR_LEFT: "door",
    DOOR_UP: "door",
    DOOR_DOWN: "door",
}


def is_wall(tile):
    """True if the robot can never stand on this tile."""
    return tile == WALL


def is_one_way_door(tile):
    """True if this tile is one of the four one-way doors."""
    return tile in DOOR_DIRECTIONS


def door_direction(tile):
    """The only direction in which this one-way door may be crossed."""
    return DOOR_DIRECTIONS[tile]


def tile_name(tile):
    """Readable name of a tile, for the legend and tooltips."""
    return TILE_NAMES[tile]


def tile_kind(tile):
    """Machine-readable name of a tile, for the renderer."""
    return TILE_KINDS[tile]
