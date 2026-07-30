"""The map of Room 1 — the Laser Security Chamber.

R-5 wakes up on the start platform in the top-left corner of a 10x10 security
chamber.  The control panel (the exit) is set into the right-hand wall.

An important detail about lasers: stepping into a beam always throws the robot
back to the start platform, so a laser is never something to walk through.  The
real danger is therefore *slippery floor next to a beam*, where the robot can be
pushed sideways into a laser it never aimed at.  The whole map is built around
that idea, which gives three genuinely different ways to cross the chamber:

  * THE FROZEN SHAFT — 13 steps, the short route.  Two cells of frozen floor in
    a narrow vertical shaft with laser emitters on both walls.  Each icy step
    has a real chance of sliding into a beam, so the fast route is also the one
    most likely to send the robot back to the start.

  * THE MAINTENANCE TELEPORTER — 17 steps, the middle route.  Reaching the pad
    means crossing one wet cell flanked by two beams, and the landing pad drops
    the robot onto frozen floor with another beam beside it.

  * THE SOUTH RING — 23 steps, the long route.  It runs down the west wall,
    along the bottom of the chamber and up the east wall.  It has wet floor and
    an oil spill, but not a single laser next to them, so it can never cost a
    laser hit — only time.

A battery sits at the top of a dead-end maintenance alcove off the south ring.
Collecting it is optional: it is worth a one-time bonus but costs six extra
steps and crosses a wet cell next to a beam.

Because every one of these probabilities is known in advance, Room 1 can be
solved with Dynamic Programming instead of learning from experience.  Which
route the robot chooses depends entirely on the parameters: safer ice makes the
frozen shaft worth the gamble, and a lower discount factor makes the impatient
teleporter route look better than the slow, safe ring.
"""

import hashlib

from core import actions, tiles

# Identification shown in the interface (the laboratory frame around the room).
SECTOR_CODE = "LAB SECTOR A-01"
ROOM_TITLE = "LASER SECURITY CHAMBER"
ROOM_SUBTITLE = "SECURITY PROTOCOL ACTIVE"
SYSTEM_STATUS = "ACTIVE"
ROOM_NUMBER = 1
ROOM_NAME = "Laser Security Chamber"
ALGORITHM_NAME = "Dynamic Programming (Value Iteration)"
MISSION = "Reach the control panel and disable the security system"
REWARD_COMPONENT = "Navigation Chart"

# The map itself.  Row 0 is the top row, so UP means a smaller row number.
#
#         c0  c1  c2  c3  c4  c5  c6  c7  c8  c9
ROOM_MAP = [
    "S.........",  # r0  north gallery; the arm east of c6 is a dead end
    ".###L%L###",  # r1  frozen shaft, laser emitters on both walls
    ".###L%L##T",  # r2  frozen shaft continues; teleporter landing pad at c9
    ".####.##L%",  # r3  shaft exit at c5; icy final step to the exit at c9
    ".####..>.E",  # r4  one-way security gate at c7, control panel at c9
    ".########.",  # r5
    ".#B######.",  # r6  battery, at the top of the maintenance alcove
    ".#O#T####.",  # r7  oil in the alcove; teleporter entry pad at c4
    ".#~L~L###.",  # r8  wet cells flanked by beams: alcove and teleporter gate
    "..~...OO..",  # r9  south ring: wet floor and oil, but no beams beside them
]

GRID_ROWS = len(ROOM_MAP)
GRID_COLS = len(ROOM_MAP[0])

# How long each of the three routes is, in steps, ignoring any slipping.
# The test suite checks these numbers with a breadth-first search, so the room
# cannot silently lose the short / medium / long structure it is built around.
ROUTE_LENGTHS = {
    "shaft": 13,  # through the frozen shaft: shortest, most likely to fail
    "teleport": 17,  # down the west wall to the teleporter pad, then the icy step
    "ring": 23,  # the full south ring: slowest, but never beside a beam
}


def tile_at(row, col):
    """The tile character at a cell."""
    return ROOM_MAP[row][col]


def in_bounds(row, col):
    """True if the cell is inside the 10x10 chamber."""
    return 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS


def find_tiles(tile):
    """Every (row, col) holding the given tile, scanned top-left to bottom-right."""
    found = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            if ROOM_MAP[row][col] == tile:
                found.append((row, col))
    return found


def start_cell():
    """The one start cell."""
    return find_tiles(tiles.START)[0]


def exit_cell():
    """The one exit cell (the control panel)."""
    return find_tiles(tiles.EXIT)[0]


def teleport_cells():
    """The two teleporter pads, as a list of two cells."""
    return find_tiles(tiles.TELEPORT)


def battery_cell():
    """The one battery cell."""
    return find_tiles(tiles.BATTERY)[0]


def walkable_cells():
    """Every cell the robot could ever stand on (everything except walls)."""
    cells = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            if not tiles.is_wall(ROOM_MAP[row][col]):
                cells.append((row, col))
    return cells


# Landmarks referred to by the interface and by the tests.
SHAFT_CELLS = ((1, 5), (2, 5))  # the two frozen cells of the short route
TELEPORT_GATE_CELL = (8, 4)  # the wet cell guarding the teleporter pad
ALCOVE_GATE_CELL = (8, 2)  # the wet cell guarding the battery alcove


def shortest_route_length(blocked_cells=(), use_teleporter=True):
    """Steps needed to walk from the start to the exit, avoiding every laser.

    A breadth-first search over the walkable cells.  Laser cells are treated as
    impassable, because entering one always throws the robot back to the start.
    One-way doors may only be crossed in their own direction.

    `blocked_cells` lets the caller close a route off in order to measure the
    others, which is how the three route lengths in ROUTE_LENGTHS are checked.
    Returns None when no route exists.
    """
    from collections import deque

    blocked = set(blocked_cells)
    pads = teleport_cells()
    partner = {pads[0]: pads[1], pads[1]: pads[0]}
    goal = exit_cell()

    queue = deque([(start_cell(), 0)])
    visited = {start_cell()}

    while queue:
        (row, col), distance = queue.popleft()
        if (row, col) == goal:
            return distance

        standing_on = tile_at(row, col)
        for direction in (actions.UP, actions.DOWN, actions.LEFT, actions.RIGHT):
            # A one-way door may only be left in its own direction.
            if tiles.is_one_way_door(standing_on):
                if direction != tiles.door_direction(standing_on):
                    continue

            next_row, next_col = actions.move(row, col, direction)
            if not in_bounds(next_row, next_col):
                continue
            target = tile_at(next_row, next_col)
            if tiles.is_wall(target) or target == tiles.LASER:
                continue
            if (next_row, next_col) in blocked:
                continue
            # A one-way door may only be entered in its own direction.
            if tiles.is_one_way_door(target) and direction != tiles.door_direction(target):
                continue
            if target == tiles.TELEPORT:
                if not use_teleporter:
                    continue
                next_row, next_col = partner[(next_row, next_col)]

            if (next_row, next_col) not in visited:
                visited.add((next_row, next_col))
                queue.append(((next_row, next_col), distance + 1))

    return None


def route_lengths():
    """Measure the three routes, each one with the other two closed off."""
    return {
        "shaft": shortest_route_length(blocked_cells=teleport_cells(),
                                       use_teleporter=False),
        "teleport": shortest_route_length(blocked_cells=SHAFT_CELLS),
        "ring": shortest_route_length(blocked_cells=SHAFT_CELLS,
                                      use_teleporter=False),
    }


def reachable_cells():
    """Every cell the robot can actually get to from the start platform.

    Lasers count as passable here, because a beam does not block the robot: it
    throws it back to the start.  Used by the tests to prove that no part of the
    chamber is sealed off by accident.
    """
    from collections import deque

    pads = teleport_cells()
    partner = {pads[0]: pads[1], pads[1]: pads[0]}

    queue = deque([start_cell()])
    visited = {start_cell()}

    while queue:
        row, col = queue.popleft()
        standing_on = tile_at(row, col)
        for direction in (actions.UP, actions.DOWN, actions.LEFT, actions.RIGHT):
            if tiles.is_one_way_door(standing_on):
                if direction != tiles.door_direction(standing_on):
                    continue
            next_row, next_col = actions.move(row, col, direction)
            if not in_bounds(next_row, next_col):
                continue
            target = tile_at(next_row, next_col)
            if tiles.is_wall(target):
                continue
            if tiles.is_one_way_door(target) and direction != tiles.door_direction(target):
                continue
            if target == tiles.TELEPORT:
                next_row, next_col = partner[(next_row, next_col)]
            if (next_row, next_col) not in visited:
                visited.add((next_row, next_col))
                queue.append((next_row, next_col))

    return visited


def map_hash():
    """A short fingerprint of the map.

    Saved solver results store this value.  When a file is loaded back we
    compare the fingerprints, so an old result can never be shown on top of a
    map that has since been edited.
    """
    joined = "\n".join(ROOM_MAP).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


# The legend shown next to the map in the interface.
LEGEND = [
    (tiles.START, tiles.tile_name(tiles.START)),
    (tiles.EXIT, tiles.tile_name(tiles.EXIT)),
    (tiles.WALL, tiles.tile_name(tiles.WALL)),
    (tiles.FLOOR, tiles.tile_name(tiles.FLOOR)),
    (tiles.WEAK_ICE, tiles.tile_name(tiles.WEAK_ICE)),
    (tiles.STRONG_ICE, tiles.tile_name(tiles.STRONG_ICE)),
    (tiles.OIL, tiles.tile_name(tiles.OIL)),
    (tiles.LASER, tiles.tile_name(tiles.LASER)),
    (tiles.BATTERY, tiles.tile_name(tiles.BATTERY)),
    (tiles.TELEPORT, tiles.tile_name(tiles.TELEPORT)),
    (tiles.DOOR_RIGHT, tiles.tile_name(tiles.DOOR_RIGHT)),
]
