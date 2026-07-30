"""The map of Room 2 — the Broken Bridge Sector.

R-5 enters the damaged industrial sector. The emergency shutdown dropped several
bridges into the maintenance shaft below, and the shaft runs right through the
middle of the room. To get out, R-5 has to pick up the security keycard and reach
the exit door on the far side.

--------------------------------------------------------------------------
THE TILE ALPHABET
--------------------------------------------------------------------------
Room 2 has its own alphabet, because it reuses two letters that mean something
else in Room 1: here `B` is a bridge (not a battery) and `T` is a maintenance
platform (not a teleporter). Each room therefore owns its own character map, and
only the four shared tiles — start, exit, wall, floor — come from `core.tiles`.

--------------------------------------------------------------------------
WHY THE ROOM IS SHAPED LIKE THIS
--------------------------------------------------------------------------
This is a bridge over a pit, built deliberately in the shape of the classic
cliff-walking problem, because that is the situation where SARSA and Q-Learning
visibly disagree.

  * THE LOWER SPAN — 11 steps, the short route. It runs straight across the top
    of the shaft, so all eight of its middle cells have a drop directly beneath
    them, and the four in the centre are collapsing bridges. Walking it is
    perfectly safe once the robot is confident. The danger is that a single
    exploratory step downwards, anywhere along it, is fatal.

  * THE UPPER WALKWAY — 21 steps, the long route. It climbs the west wall,
    crosses the top of the sector and comes back down the east wall. Not one cell
    on it has a drop beside it, so an exploratory step is never worse than a bump
    into a wall.

Under the default rewards the short route is worth **more** when walked perfectly
(+106 against +104), so a method that values behaving perfectly prefers it. A
method that values the policy it is actually following — exploration mistakes and
all — should prefer the long one, because roughly one episode in five along the
span ends in the shaft. That disagreement is the whole point of the room, and it
is the classic cliff-walking result.

Two consequences worth knowing:

  * The default discount factor is high (0.99). The whole argument rests on a
    rare, very expensive outcome, and a robot that discounts the future heavily
    stops caring about falling.
  * The default minimum epsilon is 0.10 rather than something near zero. SARSA
    and Q-Learning agree completely once exploration stops, so the difference
    only exists while the agent is still taking the occasional random step.

The keycard sits one step above the start platform, on the way up, so neither
route is penalised for collecting it.
"""

import hashlib

from core import actions, tiles

# Identification shown in the interface.
SECTOR_CODE = "LAB SECTOR B-04"
ROOM_TITLE = "BROKEN BRIDGE SECTOR"
ROOM_SUBTITLE = "STRUCTURAL FAILURE DETECTED"
SYSTEM_STATUS = "DAMAGED"
ROOM_NUMBER = 2
ROOM_NAME = "Broken Bridge Sector"
ALGORITHM_NAME = "SARSA"
MISSION = "Collect the keycard and escape through the industrial bridge network"
REWARD_COMPONENT = "Security Keycard"

# ----------------------------------------------------------------------
# Room 2's own tile alphabet
# ----------------------------------------------------------------------

START = tiles.START            # "S"
EXIT = tiles.EXIT              # "E"
WALL = tiles.WALL              # "#"
FLOOR = tiles.FLOOR            # "."
BRIDGE = "B"                   # a sound bridge; behaves like floor
COLLAPSING = "C"               # collapses the moment it is crossed
PIT = "P"                      # entering one ends the episode
KEYCARD = "K"                  # the security keycard
PLATFORM = "T"                 # a maintenance platform; behaves like floor

VALID_TILES = (START, EXIT, WALL, FLOOR, BRIDGE, COLLAPSING, PIT, KEYCARD,
               PLATFORM)

# Tiles the robot can stand on without dying. A pit is enterable but fatal, and
# a wall cannot be entered at all, so neither counts as walkable.
WALKABLE_TILES = (START, EXIT, FLOOR, BRIDGE, COLLAPSING, KEYCARD, PLATFORM)

TILE_NAMES = {
    START: "Start platform",
    EXIT: "Exit door",
    WALL: "Metallic wall",
    FLOOR: "Metal floor",
    BRIDGE: "Bridge",
    COLLAPSING: "Collapsing bridge",
    PIT: "Maintenance shaft (pit)",
    KEYCARD: "Security keycard",
    PLATFORM: "Maintenance platform",
}

# Machine-readable names handed to the renderer, so the JavaScript side never
# has to know about the letters.
TILE_KINDS = {
    START: "start",
    EXIT: "exit_door",
    WALL: "wall",
    FLOOR: "floor",
    BRIDGE: "bridge",
    COLLAPSING: "collapsing_bridge",
    PIT: "pit",
    KEYCARD: "keycard",
    PLATFORM: "platform",
}

# The map itself. Row 0 is the top row, so UP means a smaller row number.
#
#         c0  c1  c2  c3  c4  c5  c6  c7  c8  c9
ROOM_MAP = [
    "##########",  # r0
    "#TBBBBBBT#",  # r1  the upper walkway: the long way over the top of the shaft
    "#B######B#",  # r2  a capped deck, so nothing on the walkway above is exposed
    "#B#PPP##B#",  # r3  the maintenance shaft
    "#B#PPP##B#",  # r4
    "#BBCCCBBB#",  # r5  the span: three collapsing bridges straight across the shaft
    "#B#PPP##B#",  # r6
    "#K#PPP##B#",  # r7  the keycard, on the way up from the start platform
    "#S#PPP##E#",  # r8  start platform, shaft floor, exit door
    "##########",  # r9
]

GRID_ROWS = len(ROOM_MAP)
GRID_COLS = len(ROOM_MAP[0])

# Steps from the start platform to the exit door, measured by breadth-first
# search and ignoring the keycard. Checked by the tests, so the room cannot
# silently lose the trade-off it is built around.
ROUTE_LENGTHS = {
    "span": 13,      # across the bridge: short, but the shaft is on both sides
    "walkway": 21,   # over the top: long, and nothing to fall into anywhere on it
}

# The same two routes counted properly, including collecting the keycard.
# The keycard is one step above the start platform, so the span has to make a
# two-step detour for it while the walkway simply passes over it on its way up.
ROUTE_STEPS_WITH_KEYCARD = {
    "span": 13,
    "walkway": 21,
}

# How many collapsing bridges each route crosses.
ROUTE_COLLAPSING_CROSSINGS = {
    "span": 3,
    "walkway": 0,
}


def tile_at(row, col):
    """The tile character at a cell."""
    return ROOM_MAP[row][col]


def in_bounds(row, col):
    """True if the cell is inside the 10x10 sector."""
    return 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS


def is_wall(tile):
    return tile == WALL


def is_pit(tile):
    return tile == PIT


def is_collapsing(tile):
    return tile == COLLAPSING


def tile_name(tile):
    return TILE_NAMES[tile]


def tile_kind(tile):
    return TILE_KINDS[tile]


def find_tiles(tile):
    """Every (row, col) holding the given tile."""
    found = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            if ROOM_MAP[row][col] == tile:
                found.append((row, col))
    return found


def start_cell():
    return find_tiles(START)[0]


def exit_cell():
    return find_tiles(EXIT)[0]


def keycard_cell():
    return find_tiles(KEYCARD)[0]


def pit_cells():
    return find_tiles(PIT)


def collapsing_cells():
    return find_tiles(COLLAPSING)


def walkable_cells():
    """Every cell the robot can stand on: not a wall and not a pit."""
    cells = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            if ROOM_MAP[row][col] in WALKABLE_TILES:
                cells.append((row, col))
    return cells


def enterable_pits():
    """The pits the robot can actually fall into.

    Most of the shaft is walled off and is only ever scenery. These are the pits
    with a walkable cell next to them, which are the ones that can really end an
    episode.
    """
    walkable = set(walkable_cells())
    reachable = []
    for row, col in pit_cells():
        for direction in actions.ACTIONS:
            neighbour = actions.move(row, col, direction)
            if neighbour in walkable:
                reachable.append((row, col))
                break
    return reachable


def exposed_cells():
    """Walkable cells with a pit directly next to them.

    One exploratory step from any of these can kill the robot, which is exactly
    the risk SARSA is supposed to notice and Q-Learning is supposed to ignore.
    """
    exposed = []
    for row, col in walkable_cells():
        for direction in actions.ACTIONS:
            next_row, next_col = actions.move(row, col, direction)
            if in_bounds(next_row, next_col) and is_pit(tile_at(next_row, next_col)):
                exposed.append((row, col))
                break
    return exposed


def reachable_cells():
    """Every walkable cell R-5 can get to from the start platform.

    Pits are not traversed: falling into one ends the episode, so a pit is never
    a step on the way to anywhere.
    """
    from collections import deque

    walkable = set(walkable_cells())
    queue = deque([start_cell()])
    visited = {start_cell()}

    while queue:
        row, col = queue.popleft()
        for direction in actions.ACTIONS:
            neighbour = actions.move(row, col, direction)
            if neighbour in walkable and neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


def shortest_route_length(blocked_cells=()):
    """Steps from the start to the exit, avoiding pits.

    `blocked_cells` closes a route off so the others can be measured, which is
    how the two numbers in ROUTE_LENGTHS are checked.
    """
    from collections import deque

    walkable = set(walkable_cells()) - set(blocked_cells)
    goal = exit_cell()
    queue = deque([(start_cell(), 0)])
    visited = {start_cell()}

    while queue:
        (row, col), distance = queue.popleft()
        if (row, col) == goal:
            return distance
        for direction in actions.ACTIONS:
            neighbour = actions.move(row, col, direction)
            if neighbour in walkable and neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, distance + 1))
    return None


def route_lengths():
    """Measure both routes, each one with the other closed off.

    The keycard sits next to the start platform, so the detour to pick it up
    costs both routes the same two steps and is left out of these numbers.
    """
    span = collapsing_cells()
    walkway = [(1, col) for col in range(GRID_COLS)]
    return {
        "span": shortest_route_length(blocked_cells=walkway),
        "walkway": shortest_route_length(blocked_cells=span),
    }


def perfect_return(route, step_cost=-1, collapsing_penalty=-2,
                   keycard_bonus=25, exit_reward=100):
    """What a route pays out when it is walked without a single mistake.

    This is the number that makes the room work: the short route is worth *more*
    than the long one when nothing goes wrong, so only a method that accounts for
    its own exploration has any reason to avoid it.
    """
    steps = ROUTE_STEPS_WITH_KEYCARD[route]
    crossings = ROUTE_COLLAPSING_CROSSINGS[route]
    return (steps * step_cost + crossings * collapsing_penalty
            + keycard_bonus + exit_reward)


def map_hash():
    """A short fingerprint of the map, stored inside saved models."""
    joined = "\n".join(ROOM_MAP).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


# The legend shown under the sector view.
LEGEND = [
    (START, TILE_NAMES[START]),
    (EXIT, TILE_NAMES[EXIT]),
    (WALL, TILE_NAMES[WALL]),
    (BRIDGE, TILE_NAMES[BRIDGE]),
    (COLLAPSING, TILE_NAMES[COLLAPSING]),
    (PIT, TILE_NAMES[PIT]),
    (KEYCARD, TILE_NAMES[KEYCARD]),
    (PLATFORM, TILE_NAMES[PLATFORM]),
]
