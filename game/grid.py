"""The grid environment. Pure logic — no canvas, no DOM, no timers.

One class serves every grid room. A room hands it a layout and a reward
table and nothing else; the environment has no idea which algorithm is
running against it, which is what lets the tabular methods be swapped
without the room being touched.

    state   (row, col, battery, last_direction)
    action  0 UP, 1 DOWN, 2 LEFT, 3 RIGHT

Given a seed it is fully deterministic, and `transitions()` exposes the
whole model, which is what Dynamic Programming needs and what the
model-free methods are deliberately not given.

WHY THE STATE IS FOUR NUMBERS
A cell alone is not a Markov state in this room any more.

  * A battery can only be collected once, so whether it is already in hand
    changes what entering its cell is worth.  Without that, the model says
    the battery pays out every time it is stepped on and the best plan is
    to stand there collecting it forever.
  * Oil carries the agent on in the direction it was already going, so
    what happens next depends on what happened last.  A model that cannot
    see the previous direction cannot express that.

Both are cheap: ten cells by ten, two battery states, five directions
counting "none" — a thousand states, which Value Iteration sweeps in
milliseconds.  The page still draws one grid, so the display projects
each cell onto its best state; see `Planner.snapshot`.

THE SURFACES
The surface *under* the agent decides what a step does, because that is
what it pushes off from.

    ~  ice          goes where it aimed with probability 1 - slip,
                    otherwise slides to one of the two sides, evenly
    =  cracked ice  the same, at twice the slip: this is the surface that
                    makes a short route genuinely frightening
    o  oil          no grip at all: with probability `slip` it carries on
                    the way it was already travelling, whatever it aimed

Everything else is firm.
"""

import random

# The four actions, and how each one changes (row, col).
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS = (UP, DOWN, LEFT, RIGHT)

# A fifth value for "has not moved yet", so the first step on oil has
# something defined to carry on from.
NO_DIRECTION = 4

ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}
ACTION_ARROWS = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}

DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# Sliding always means sideways, never backwards.
PERPENDICULAR = {UP: (LEFT, RIGHT), DOWN: (LEFT, RIGHT),
                 LEFT: (UP, DOWN), RIGHT: (UP, DOWN)}

# The tile alphabet. Each maps to one of the entity kinds in config.ENTITIES.
WALL = "#"
FLOOR = "."
START = "S"
GOAL = "E"
LASER = "L"
SLIPPERY = "~"
CRACKED = "="
OIL = "o"
BATTERY = "B"
TELEPORT = "T"
ONE_WAY = "D"

TILE_KINDS = {
    WALL: "wall",
    FLOOR: "floor",
    START: "start",
    GOAL: "goal",
    LASER: "hazard",
    SLIPPERY: "slippery",
    CRACKED: "cracked",
    OIL: "oil",
    BATTERY: "battery",
    TELEPORT: "teleport",
    ONE_WAY: "oneway",
}

# The surfaces that do not simply do as they are told.
LOOSE = (SLIPPERY, CRACKED, OIL)

# Which way a one-way door may be entered. Down only: it is a drop, and
# what makes it interesting is that the choice to take it cannot be undone.
ONE_WAY_DIRECTION = DOWN


class GridWorld:
    """A grid of tiles with a start, an exit, and everything between."""

    def __init__(self, room, slip=0.2, seed=0):
        self.room = room
        self.grid = list(room["grid"])
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.slip = slip
        self.rewards = dict(room["rewards"])

        self.start = self._find(START)
        self.goal = self._find(GOAL)
        self.teleports = self._find_all(TELEPORT)
        self.has_battery_tile = bool(self._find_all(BATTERY))

        self.rng = random.Random(seed)
        self.state = self.start_state()
        self.steps = 0

    # ------------------------------------------------------------------
    # The layout
    # ------------------------------------------------------------------

    def _find(self, tile):
        found = self._find_all(tile)
        if not found:
            raise ValueError("the layout has no %r tile" % tile)
        return found[0]

    def _find_all(self, tile):
        return [(row, col)
                for row in range(self.rows)
                for col in range(self.cols)
                if self.grid[row][col] == tile]

    def tile_at(self, row, col):
        return self.grid[row][col]

    def in_bounds(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_wall(self, row, col):
        if not self.in_bounds(row, col):
            return True
        return self.grid[row][col] == WALL

    def cell_of(self, state):
        """The two numbers the page draws. Everything else is bookkeeping."""
        return (state[0], state[1])

    def all_states(self):
        """Every state the agent could be in, in a stable order.

        A cell for every battery and direction it could be in it — which is
        what makes the battery collectable once and the oil have memory.
        """
        batteries = (0, 1) if self.has_battery_tile else (0,)
        directions = list(ACTIONS) + [NO_DIRECTION]

        states = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col] == WALL:
                    continue
                for battery in batteries:
                    for last in directions:
                        states.append((row, col, battery, last))
        return states

    def actions(self):
        return list(ACTIONS)

    def start_state(self):
        return (self.start[0], self.start[1], 0, NO_DIRECTION)

    def is_terminal(self, state):
        """Only the control panel ends a run.

        A laser does not: it throws the agent back to the door it came in
        by, which costs it everything it had walked and is a good deal more
        interesting to plan around than simply dying.
        """
        return self.grid[state[0]][state[1]] == GOAL

    # ------------------------------------------------------------------
    # One transition, in two parts
    # ------------------------------------------------------------------

    def slip_of(self, tile):
        """How little grip a surface has, as a probability."""
        if tile == SLIPPERY:
            return self.slip
        if tile == CRACKED:
            # Twice as treacherous, and capped so a direction is always
            # possible at all.
            return min(0.95, self.slip * 2.0)
        if tile == OIL:
            return self.slip
        return 0.0

    def direction_probabilities(self, state, action):
        """Which direction the agent actually goes, and how likely each is."""
        row, col, _, last = state
        surface = self.grid[row][col]
        loose = self.slip_of(surface)
        if surface not in LOOSE or loose <= 0:
            return {action: 1.0}

        if surface == OIL:
            # No grip: it carries on the way it was already going. With
            # nothing to carry on from, it simply goes where it aimed.
            if last == NO_DIRECTION or last == action:
                return {action: 1.0}
            return {last: loose, action: 1.0 - loose}

        sideways = loose / 2.0
        left_slip, right_slip = PERPENDICULAR[action]
        return {
            action: 1.0 - loose,
            left_slip: sideways,
            right_slip: sideways,
        }

    def _other_teleport(self, cell):
        """Where a pad sends you: the next one round, so a pair swaps."""
        if len(self.teleports) < 2:
            return cell
        index = self.teleports.index(cell)
        return self.teleports[(index + 1) % len(self.teleports)]

    def resolve(self, state, direction):
        """Move one step and report (next_state, reward, done)."""
        row, col, battery, _ = state
        delta_row, delta_col = DELTAS[direction]
        target_row, target_col = row + delta_row, col + delta_col
        step = self.rewards["step"]

        blocked = self.is_wall(target_row, target_col)
        if not blocked and self.grid[target_row][target_col] == ONE_WAY:
            # A one-way door is a wall from every side but one.
            blocked = direction != ONE_WAY_DIRECTION

        if blocked:
            return ((row, col, battery, direction),
                    step + self.rewards["wall"], False)

        tile = self.grid[target_row][target_col]

        if tile == LASER:
            # Straight back to the start, with nothing kept but the battery
            # — it is bolted on, and the beam does not take it off again.
            return ((self.start[0], self.start[1], battery, NO_DIRECTION),
                    step + self.rewards["laser"], False)

        if tile == TELEPORT:
            landing = self._other_teleport((target_row, target_col))
            # Arriving by pad leaves no momentum to carry on with.
            return ((landing[0], landing[1], battery, NO_DIRECTION),
                    step + self.rewards["shortcut"], False)

        if tile == BATTERY and battery == 0:
            return ((target_row, target_col, 1, direction),
                    step + self.rewards["battery"], False)

        if tile == GOAL:
            return ((target_row, target_col, battery, direction),
                    step + self.rewards["goal"], True)

        return ((target_row, target_col, battery, direction), step, False)

    def transitions(self, state, action):
        """Every outcome of an action, as (probability, next, reward, done).

        This is the model. Dynamic Programming reads it; the model-free
        methods are never handed the environment at all, so they cannot.
        """
        if self.is_terminal(state):
            return [(1.0, state, 0.0, True)]

        merged = {}
        probabilities = self.direction_probabilities(state, action)
        for direction, probability in probabilities.items():
            outcome = self.resolve(state, direction)
            merged[outcome] = merged.get(outcome, 0.0) + probability

        return [(probability, next_state, reward, done)
                for (next_state, reward, done), probability in merged.items()]

    # ------------------------------------------------------------------
    # Being stepped through
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.state = self.start_state()
        self.steps = 0
        return self.state

    def step(self, action):
        """Take one action, sampling the outcome. Returns the usual four."""
        state = self.state
        probabilities = self.direction_probabilities(state, action)

        # Sorted, so a given seed always draws the same way regardless of
        # dictionary ordering.
        directions = sorted(probabilities.keys())
        weights = [probabilities[direction] for direction in directions]
        actual = self.rng.choices(directions, weights=weights)[0]

        next_state, reward, done = self.resolve(state, actual)
        tile = self.grid[next_state[0]][next_state[1]]
        # Where it was aiming, which is what says a laser was walked into
        # rather than that the agent is merely standing at the start again.
        aimed_row = state[0] + DELTAS[actual][0]
        aimed_col = state[1] + DELTAS[actual][1]
        burnt = (self.in_bounds(aimed_row, aimed_col)
                 and self.grid[aimed_row][aimed_col] == LASER)

        self.state = next_state
        self.steps += 1

        info = {
            "action": action,
            "direction": actual,
            "slipped": actual != action,
            "hazard": burnt,
            "battery": next_state[2] > state[2],
            "goal": done and tile == GOAL,
        }
        return next_state, reward, done, info

    # ------------------------------------------------------------------
    # What the renderer is given
    # ------------------------------------------------------------------

    def layout_snapshot(self):
        """The static part of the scene: sent once, never changes."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "tiles": [[TILE_KINDS[self.grid[row][col]]
                       for col in range(self.cols)]
                      for row in range(self.rows)],
            "start": list(self.start),
            "goal": list(self.goal),
        }

    def snapshot(self):
        """The moving part: where the agent is, and what it is carrying."""
        return {
            "agent": list(self.cell_of(self.state)),
            "battery": bool(self.state[2]),
            "steps": self.steps,
            "terminal": self.is_terminal(self.state),
        }
