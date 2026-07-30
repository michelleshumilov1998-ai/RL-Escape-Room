"""Procedural warehouse layouts for Room 5, with strict train/test separation.

Every layout is generated from a seed, so a seed *is* a layout: reproducible, and
easy to keep apart. That matters more here than anywhere else in the game, because
Room 5's whole claim is that R-5 can navigate warehouses it has never seen. If the
evaluation layouts were ever trained on, the claim would be empty.

    Training seeds     1000 ...          used to train
    Validation seeds   2000 ...          used while settling the configuration
    Test seeds         3000 ...          never used for either

The three ranges cannot overlap and the tests assert it.

--------------------------------------------------------------------------
GENERATION
--------------------------------------------------------------------------
Shelves are laid down in aisles, then crates, conveyors and patrol routes are
scattered around them. A layout is only accepted once breadth-first search
confirms both legs of the mission are walkable:

    start -> terminal        and        terminal -> exit

If either fails the layout is thrown away and another seed offset is tried, so an
impossible warehouse can never reach the agent.
"""

import random
from collections import deque

# --- the tile alphabet -----------------------------------------------------
START = "S"
EXIT = "E"
TERMINAL = "T"
SHELF = "#"
CRATE = "C"
CONVEYOR_UP = "^"
CONVEYOR_DOWN = "v"
CONVEYOR_LEFT = "<"
CONVEYOR_RIGHT = ">"
CHARGER = "F"
FLOOR = "."

CONVEYORS = {CONVEYOR_UP: (-1, 0), CONVEYOR_DOWN: (1, 0),
             CONVEYOR_LEFT: (0, -1), CONVEYOR_RIGHT: (0, 1)}

# Everything that blocks movement outright.
BLOCKING = (SHELF, CRATE)

VALID_TILES = (START, EXIT, TERMINAL, SHELF, CRATE, CHARGER, FLOOR) + \
    tuple(CONVEYORS.keys())

TILE_NAMES = {
    START: "Start", EXIT: "Final exit", TERMINAL: "Access terminal",
    SHELF: "Storage shelf", CRATE: "Static crate", CHARGER: "Charging station",
    FLOOR: "Open floor", CONVEYOR_UP: "Conveyor (up)",
    CONVEYOR_DOWN: "Conveyor (down)", CONVEYOR_LEFT: "Conveyor (left)",
    CONVEYOR_RIGHT: "Conveyor (right)",
}

TILE_KINDS = {
    START: "start", EXIT: "exit_door", TERMINAL: "terminal", SHELF: "shelf",
    CRATE: "crate", CHARGER: "charger", FLOOR: "floor",
    CONVEYOR_UP: "conveyor", CONVEYOR_DOWN: "conveyor",
    CONVEYOR_LEFT: "conveyor", CONVEYOR_RIGHT: "conveyor",
}

# --- the seed ranges -------------------------------------------------------
TRAINING_SEED_BASE = 1000
VALIDATION_SEED_BASE = 2000
TEST_SEED_BASE = 3000

SPLIT_NAMES = ("training", "validation", "test")

# --- difficulty presets ----------------------------------------------------
DIFFICULTIES = {
    "Easy": {"size": 12, "shelf_aisles": 2, "crates": 6, "robots": 1,
             "conveyors": 1, "radar_range": 6, "max_steps": 300},
    "Medium": {"size": 12, "shelf_aisles": 3, "crates": 12, "robots": 2,
               "conveyors": 4, "radar_range": 4, "max_steps": 300},
    "Hard": {"size": 14, "shelf_aisles": 4, "crates": 18, "robots": 3,
             "conveyors": 6, "radar_range": 3, "max_steps": 260},
}

# Easy is the default because it is the setting at which the agent demonstrably
# learns and generalises. Medium and Hard are selectable and the agent does
# measurably worse on them, which is the point of the stress test.
DEFAULT_DIFFICULTY = "Easy"


class Robot:
    """A maintenance robot on a fixed patrol.

    Two patterns, both entirely deterministic so a replay is exact:

      * "loop"    — round and round the route
      * "bounce"  — to the end and back again
    """

    def __init__(self, route, pattern="loop", index=0, forward=True):
        self.route = tuple(route)
        self.pattern = pattern
        self.index = index
        self.forward = forward

    def cell(self):
        return self.route[self.index]

    def advance(self):
        """Move one step along the route and return the new cell."""
        if len(self.route) <= 1:
            return self.cell()
        if self.pattern == "bounce":
            if self.forward:
                if self.index + 1 >= len(self.route):
                    self.forward = False
                    self.index -= 1
                else:
                    self.index += 1
            else:
                if self.index - 1 < 0:
                    self.forward = True
                    self.index += 1
                else:
                    self.index -= 1
        else:
            self.index = (self.index + 1) % len(self.route)
        return self.cell()

    def copy(self):
        return Robot(self.route, self.pattern, self.index, self.forward)

    def as_dict(self):
        return {"route": [list(cell) for cell in self.route],
                "pattern": self.pattern, "index": self.index,
                "forward": self.forward}


class Layout:
    """One generated warehouse."""

    def __init__(self, seed, split, grid, start, terminal, exit_cell, robots,
                 difficulty, radar_range, max_steps):
        self.seed = seed
        self.split = split
        self.grid = grid
        self.size = len(grid)
        self.start = start
        self.terminal = terminal
        self.exit_cell = exit_cell
        self.robot_templates = robots
        self.difficulty = difficulty
        self.radar_range = radar_range
        self.max_steps = max_steps

    # ------------------------------------------------------------------

    def tile_at(self, row, col):
        return self.grid[row][col]

    def in_bounds(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size

    def blocks(self, row, col):
        if not self.in_bounds(row, col):
            return True
        return self.grid[row][col] in BLOCKING

    def conveyor_at(self, row, col):
        return CONVEYORS.get(self.grid[row][col])

    def fresh_robots(self):
        """A new set of robots at their starting patrol positions."""
        return [robot.copy() for robot in self.robot_templates]

    def obstacle_density(self):
        blocked = sum(1 for row in self.grid for tile in row if tile in BLOCKING)
        return blocked / (self.size * self.size)

    def shortest_mission_length(self):
        """Start to terminal to exit, ignoring the robots.

        For analysis only — path efficiency is measured against it. It is never
        given to the agent.
        """
        first = _path_length(self, self.start, self.terminal)
        second = _path_length(self, self.terminal, self.exit_cell)
        if first is None or second is None:
            return None
        return first + second

    def as_dict(self):
        return {
            "seed": self.seed, "split": self.split, "size": self.size,
            "grid": ["".join(row) for row in self.grid],
            "start": list(self.start), "terminal": list(self.terminal),
            "exit": list(self.exit_cell),
            "robots": [robot.as_dict() for robot in self.robot_templates],
            "difficulty": self.difficulty, "radarRange": self.radar_range,
            "obstacleDensity": self.obstacle_density(),
        }


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------

def _path_length(layout, origin, target):
    """Breadth-first search, treating shelves and crates as walls."""
    if origin == target:
        return 0
    queue = deque([(origin, 0)])
    visited = {origin}
    while queue:
        (row, col), distance = queue.popleft()
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            cell = (row + delta_row, col + delta_col)
            if cell in visited or layout.blocks(*cell):
                continue
            if cell == target:
                return distance + 1
            visited.add(cell)
            queue.append((cell, distance + 1))
    return None


def _blank(size):
    return [[FLOOR for _ in range(size)] for _ in range(size)]


def _build_candidate(seed, split, settings):
    """One attempt at a layout. May be unwalkable; the caller checks."""
    rng = random.Random(seed)
    size = settings["size"]
    grid = _blank(size)

    # Storage shelves, in aisles with a gap so corridors always exist.
    for aisle in range(settings["shelf_aisles"]):
        col = 2 + aisle * 3
        if col >= size - 1:
            break
        gap = rng.randrange(2, size - 2)
        for row in range(1, size - 1):
            if row == gap or row == gap + 1:
                continue
            grid[row][col] = SHELF

    # Scattered crates on open floor.
    placed = 0
    attempts = 0
    while placed < settings["crates"] and attempts < 400:
        attempts += 1
        row = rng.randrange(1, size - 1)
        col = rng.randrange(1, size - 1)
        if grid[row][col] == FLOOR:
            grid[row][col] = CRATE
            placed += 1

    # Conveyor cells, which shove the robot one extra square.
    for _ in range(settings["conveyors"]):
        for _ in range(40):
            row = rng.randrange(1, size - 1)
            col = rng.randrange(1, size - 1)
            if grid[row][col] == FLOOR:
                grid[row][col] = rng.choice(list(CONVEYORS.keys()))
                break

    # The three fixed points: start bottom-left, terminal middle-ish, exit
    # top-right. Each is forced clear.
    start = (size - 2, 1)
    exit_cell = (1, size - 2)
    terminal = None
    for _ in range(200):
        row = rng.randrange(2, size - 2)
        col = rng.randrange(2, size - 2)
        if (row, col) not in (start, exit_cell):
            terminal = (row, col)
            break
    terminal = terminal or (size // 2, size // 2)

    for cell, tile in ((start, START), (terminal, TERMINAL), (exit_cell, EXIT)):
        grid[cell[0]][cell[1]] = tile
        # Clear a little space so nothing is walled in.
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            row, col = cell[0] + delta_row, cell[1] + delta_col
            if 0 < row < size - 1 and 0 < col < size - 1:
                if grid[row][col] in BLOCKING:
                    grid[row][col] = FLOOR

    # A charging station, worth a small one-off bonus.
    for _ in range(40):
        row = rng.randrange(1, size - 1)
        col = rng.randrange(1, size - 1)
        if grid[row][col] == FLOOR:
            grid[row][col] = CHARGER
            break

    # Patrol routes for the maintenance robots, along open corridors.
    robots = []
    for number in range(settings["robots"]):
        route = _patrol_route(grid, size, rng)
        if route:
            pattern = "loop" if number % 2 == 0 else "bounce"
            robots.append(Robot(route, pattern,
                                index=rng.randrange(len(route))))

    return Layout(seed, split, grid, start, terminal, exit_cell, robots,
                  settings.get("difficulty", DEFAULT_DIFFICULTY),
                  settings["radar_range"], settings["max_steps"])


def _patrol_route(grid, size, rng):
    """A short straight run of open cells for a robot to walk up and down."""
    for _ in range(60):
        horizontal = rng.random() < 0.5
        length = rng.randrange(3, max(4, size // 2))
        row = rng.randrange(1, size - 1)
        col = rng.randrange(1, size - 1)
        cells = []
        for step in range(length):
            cell = (row, col + step) if horizontal else (row + step, col)
            if not (0 < cell[0] < size - 1 and 0 < cell[1] < size - 1):
                break
            if grid[cell[0]][cell[1]] in BLOCKING:
                break
            if grid[cell[0]][cell[1]] in (START, TERMINAL, EXIT):
                break
            cells.append(cell)
        if len(cells) >= 3:
            return cells
    return None


def is_valid(layout):
    """Both legs of the mission have to be walkable."""
    if _path_length(layout, layout.start, layout.terminal) is None:
        return False
    if _path_length(layout, layout.terminal, layout.exit_cell) is None:
        return False
    return True


def generate(seed, split="training", difficulty=DEFAULT_DIFFICULTY,
             overrides=None):
    """A validated layout for one seed.

    If the first attempt is unwalkable the seed is nudged and tried again, so this
    always returns a layout the mission can actually be completed in.
    """
    settings = dict(DIFFICULTIES[difficulty])
    settings["difficulty"] = difficulty
    settings.update(overrides or {})

    for attempt in range(60):
        layout = _build_candidate(seed + attempt * 7919, split, settings)
        if is_valid(layout):
            layout.seed = seed          # report the seed asked for
            return layout
    raise RuntimeError("could not generate a walkable layout for seed %s" % seed)


def seed_for(split, index):
    """The seed at `index` within one split."""
    base = {"training": TRAINING_SEED_BASE, "validation": VALIDATION_SEED_BASE,
            "test": TEST_SEED_BASE}[split]
    return base + index


def seeds(split, count):
    return [seed_for(split, index) for index in range(count)]


def generate_pool(split, count, difficulty=DEFAULT_DIFFICULTY, overrides=None):
    """A list of validated layouts for one split."""
    return [generate(seed_for(split, index), split, difficulty, overrides)
            for index in range(count)]


def splits_overlap(training_count, validation_count, test_count):
    """True if any two splits would share a seed. Should always be False."""
    training = set(seeds("training", training_count))
    validation = set(seeds("validation", validation_count))
    test = set(seeds("test", test_count))
    return bool(training & validation or training & test or validation & test)


LEGEND = [
    (START, TILE_NAMES[START]),
    (TERMINAL, TILE_NAMES[TERMINAL]),
    (EXIT, TILE_NAMES[EXIT]),
    (SHELF, TILE_NAMES[SHELF]),
    (CRATE, TILE_NAMES[CRATE]),
    (CONVEYOR_RIGHT, "Conveyor belt"),
    (CHARGER, TILE_NAMES[CHARGER]),
    (FLOOR, TILE_NAMES[FLOOR]),
]
