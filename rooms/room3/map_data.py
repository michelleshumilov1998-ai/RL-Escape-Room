"""The map of Room 3 — the Reactor Control Chamber.

R-5 reaches the heart of the laboratory. The emergency shutdown has taken the
central reactor offline, and until it is running again the sectors above cannot be
reached. Three generators have to be brought up **in order** — A, then B, then C —
after which the reactor blast door unlocks.

--------------------------------------------------------------------------
THE SHAPE OF THE ROOM
--------------------------------------------------------------------------
The chamber is a square service ring around the reactor core, with the three
generators spaced around it and the blast door on the south side:

    A is the north-east corner, B the south-east, C the south-west.

Because they must be activated in that order, the shortest possible mission is a
full clockwise circuit of the ring — the reward for reaching the exit is a long way
away from the first move that leads towards it. That delay is the point of the
room, and it is what Q-Learning is good at.

Two things get in the way:

  * **The security robot** patrols the same ring, anticlockwise, one cell per step.
    Meeting it head-on ends the episode. R-5 cannot outrun it and cannot pass it on
    the ring, so it has to duck out of the way.
  * **The reactor service shaft** runs straight through the middle from the north
    corridor to the south corridor. It is six steps instead of twelve, it has no
    electrical hazards on it, and — because the guard never leaves the ring — it is
    the only safe place to stand. Both ends are sliding doors that open and close on
    a fixed cycle, so getting in means arriving at the right moment or waiting for
    it.

The electrical hazards sit on the east and west sides of the ring, which is what
makes the shaft worth using at all: going round costs twelve steps and two shocks,
going through costs six steps and nothing, if the door is open.

--------------------------------------------------------------------------
KEEPING IT MARKOVIAN
--------------------------------------------------------------------------
The sliding doors are on a timer, which would normally make the room
non-Markovian: the same cell means different things at different moments. Rather
than add a separate clock to the state, the door cycle is derived from the guard's
position on its patrol, which is already part of the state. The patrol is 24 cells
long and the door cycle is 4 steps, so the pattern repeats cleanly and the agent
can always work out the door state from what it can see.
"""

import hashlib

from core import actions, tiles

# Identification shown in the interface.
SECTOR_CODE = "LAB SECTOR C-07"
ROOM_TITLE = "REACTOR CONTROL CHAMBER"
ROOM_SUBTITLE = "POWER GRID OFFLINE"
ROOM_NUMBER = 3
ROOM_NAME = "Reactor Control Chamber"
ALGORITHM_NAME = "Q-Learning"

# ----------------------------------------------------------------------
# Room 3's tile alphabet
# ----------------------------------------------------------------------

START = tiles.START            # "S"
EXIT = tiles.EXIT              # "E"
WALL = tiles.WALL              # "#"
FLOOR = tiles.FLOOR            # "."
GENERATOR_A = "A"
GENERATOR_B = "B"
GENERATOR_C = "C"
REACTOR_DOOR = "R"             # opens only once all three generators are running
HAZARD = "X"                   # electrical arc; costly but not fatal
SLIDING_DOOR = "D"             # opens and closes on a fixed cycle

GENERATORS = (GENERATOR_A, GENERATOR_B, GENERATOR_C)

VALID_TILES = (START, EXIT, WALL, FLOOR, GENERATOR_A, GENERATOR_B, GENERATOR_C,
               REACTOR_DOOR, HAZARD, SLIDING_DOOR)

# Everything except a wall can be stood on. The reactor door and the sliding doors
# can only be entered at the right moment, which the environment decides.
WALKABLE_TILES = (START, EXIT, FLOOR, GENERATOR_A, GENERATOR_B, GENERATOR_C,
                  REACTOR_DOOR, HAZARD, SLIDING_DOOR)

TILE_NAMES = {
    START: "Start platform",
    EXIT: "Reactor exit",
    WALL: "Reinforced wall",
    FLOOR: "Metal floor",
    GENERATOR_A: "Generator A",
    GENERATOR_B: "Generator B",
    GENERATOR_C: "Generator C",
    REACTOR_DOOR: "Reactor blast door",
    HAZARD: "Electrical hazard",
    SLIDING_DOOR: "Sliding door",
}

TILE_KINDS = {
    START: "start",
    EXIT: "exit_door",
    WALL: "wall",
    FLOOR: "floor",
    GENERATOR_A: "generator_a",
    GENERATOR_B: "generator_b",
    GENERATOR_C: "generator_c",
    REACTOR_DOOR: "reactor_door",
    HAZARD: "hazard",
    SLIDING_DOOR: "sliding_door",
}

# The map itself. Row 0 is the top row.
#
#         c0  c1  c2  c3  c4  c5  c6  c7  c8  c9
ROOM_MAP = [
    "##########",  # r0
    "#S.....A##",  # r1  north corridor, Generator A in the north-east corner
    "#.##D##.##",  # r2  the north sliding door into the service shaft
    "#X##.##X##",  # r3  electrical hazards on the ring's east and west sides
    "#.##.##.##",  # r4  the service shaft runs down the middle
    "#X##.##X##",  # r5
    "#.##D##.##",  # r6  the south sliding door
    "#C.....B##",  # r7  south corridor, C south-west and B south-east
    "####R#####",  # r8  the reactor blast door
    "####E#####",  # r9  the exit, behind the blast door
]

GRID_ROWS = len(ROOM_MAP)
GRID_COLS = len(ROOM_MAP[0])

# ----------------------------------------------------------------------
# The security robot's patrol
# ----------------------------------------------------------------------

def _build_patrol():
    """The service ring, listed anticlockwise starting from the north-west.

    Anticlockwise matters: R-5's mission takes it clockwise (A, then B, then C), so
    the two meet head-on once per lap and R-5 has to get out of the way.
    """
    clockwise = []
    clockwise.extend((1, col) for col in range(1, 8))       # north, west -> east
    clockwise.extend((row, 7) for row in range(2, 8))       # east, north -> south
    clockwise.extend((7, col) for col in range(6, 0, -1))   # south, east -> west
    clockwise.extend((row, 1) for row in range(6, 1, -1))   # west, south -> north
    # Reversed, so the guard runs against the direction the mission takes R-5.
    return tuple(reversed(clockwise))


PATROL = _build_patrol()
PATROL_LENGTH = len(PATROL)

# The sliding doors are open for this many steps, then shut for the same again.
DOOR_HALF_PERIOD = 2
DOOR_PERIOD = DOOR_HALF_PERIOD * 2


def door_is_open(guard_index):
    """Whether the sliding doors are open, given where the guard is.

    Deriving this from the guard's patrol position rather than from a separate
    clock is what keeps the room Markovian: everything the agent needs in order to
    predict the doors is already in its state.
    """
    return (guard_index // DOOR_HALF_PERIOD) % 2 == 0


def guard_cell(guard_index):
    """Where the security robot is standing."""
    return PATROL[guard_index % PATROL_LENGTH]


def next_guard_index(guard_index):
    """The guard advances one cell along its patrol every step."""
    return (guard_index + 1) % PATROL_LENGTH


# ----------------------------------------------------------------------
# Looking things up on the map
# ----------------------------------------------------------------------

def tile_at(row, col):
    return ROOM_MAP[row][col]


def in_bounds(row, col):
    return 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS


def is_wall(tile):
    return tile == WALL


def tile_name(tile):
    return TILE_NAMES[tile]


def tile_kind(tile):
    return TILE_KINDS[tile]


def find_tiles(tile):
    return [(row, col)
            for row in range(GRID_ROWS)
            for col in range(GRID_COLS)
            if ROOM_MAP[row][col] == tile]


def start_cell():
    return find_tiles(START)[0]


def exit_cell():
    return find_tiles(EXIT)[0]


def generator_cell(letter):
    return find_tiles(letter)[0]


def reactor_door_cell():
    return find_tiles(REACTOR_DOOR)[0]


def hazard_cells():
    return find_tiles(HAZARD)


def sliding_door_cells():
    return find_tiles(SLIDING_DOOR)


def walkable_cells():
    return [(row, col)
            for row in range(GRID_ROWS)
            for col in range(GRID_COLS)
            if ROOM_MAP[row][col] in WALKABLE_TILES]


def shaft_cells():
    """The service shaft: the cells the guard can never reach.

    Standing in here is the only way to let the patrol go past.
    """
    ring = set(PATROL)
    return [cell for cell in walkable_cells()
            if cell not in ring and tile_at(*cell) != EXIT]


def reachable_cells(doors_open=True):
    """Every cell R-5 can get to from the start, ignoring the guard.

    With `doors_open` false the service shaft is sealed, which is what the ring has
    to be able to cope with — the mission must still be possible.
    """
    from collections import deque

    walkable = set(walkable_cells())
    queue = deque([start_cell()])
    visited = {start_cell()}

    while queue:
        row, col = queue.popleft()
        for action in actions.ACTIONS:
            neighbour = actions.move(row, col, action)
            if neighbour not in walkable or neighbour in visited:
                continue
            tile = tile_at(*neighbour)
            if tile == SLIDING_DOOR and not doors_open:
                continue
            visited.add(neighbour)
            queue.append(neighbour)
    return visited


def mission_route_length():
    """Steps for the shortest legal mission, ignoring the guard and the doors.

    Start, then A, then B, then C, then the blast door and the exit. Used by the
    tests and shown in the interface as a reference.
    """
    from collections import deque

    def walk(origin, target):
        walkable = set(walkable_cells())
        queue = deque([(origin, 0)])
        visited = {origin}
        while queue:
            cell, distance = queue.popleft()
            if cell == target:
                return distance
            for action in actions.ACTIONS:
                neighbour = actions.move(*cell, action)
                if neighbour in walkable and neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, distance + 1))
        return None

    legs = [(start_cell(), generator_cell(GENERATOR_A)),
            (generator_cell(GENERATOR_A), generator_cell(GENERATOR_B)),
            (generator_cell(GENERATOR_B), generator_cell(GENERATOR_C)),
            (generator_cell(GENERATOR_C), exit_cell())]
    total = 0
    for origin, target in legs:
        steps = walk(origin, target)
        if steps is None:
            return None
        total += steps
    return total


def map_hash():
    joined = "\n".join(ROOM_MAP).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


LEGEND = [
    (START, TILE_NAMES[START]),
    (GENERATOR_A, TILE_NAMES[GENERATOR_A]),
    (GENERATOR_B, TILE_NAMES[GENERATOR_B]),
    (GENERATOR_C, TILE_NAMES[GENERATOR_C]),
    (REACTOR_DOOR, TILE_NAMES[REACTOR_DOOR]),
    (EXIT, TILE_NAMES[EXIT]),
    (HAZARD, TILE_NAMES[HAZARD]),
    (SLIDING_DOOR, TILE_NAMES[SLIDING_DOOR]),
    (WALL, TILE_NAMES[WALL]),
    (FLOOR, TILE_NAMES[FLOOR]),
]
