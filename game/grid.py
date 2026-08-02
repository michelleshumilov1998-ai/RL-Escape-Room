"""The grid environment. Pure logic — no canvas, no DOM, no timers.

One class serves every grid room. A room hands it a layout and a reward
table and nothing else; the environment has no idea which algorithm is
running against it, which is what lets the tabular methods be swapped
without the room being touched.

    state   (row, col, battery, last_direction, collapsed)
    action  0 UP, 1 DOWN, 2 LEFT, 3 RIGHT

Given a seed it is fully deterministic, and `transitions()` exposes the
whole model, which is what Dynamic Programming needs and what the
model-free methods are deliberately not given.

WHY THE STATE IS FIVE NUMBERS
A cell alone is not a Markov state in these rooms.

  * A battery can only be collected once, so whether it is already in hand
    changes what entering its cell is worth.  Without that, the model says
    the battery pays out every time it is stepped on and the best plan is
    to stand there collecting it forever.
  * Oil carries the agent on in the direction it was already going, so
    what happens next depends on what happened last.  A model that cannot
    see the previous direction cannot express that.
  * A collapsing plank is a bridge early in an episode and a hole later.
    Without knowing which have already gone, the value of stepping onto one
    is an average of "fine" and "fatal" — and an average is not a fact
    about the state.  Measured on the old build, that average put the span
    below the safe route, *both* methods avoided it, and the on-policy /
    off-policy difference the room exists to show never appeared at all.

All three are cheap.  Only the rooms that contain the thing pay for it: the
battery dimension collapses to one value where there is no battery, and the
mask to one value where there are no planks, so room 1 is no larger than
before.  The page still draws one grid, so the display projects each cell
onto its best state; see `Planner.snapshot`.

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

THE TWO HAZARDS, WHICH ARE NOT THE SAME KIND OF THING
Both cost the agent dearly; only one ends the run, and the difference is
the whole reason two rooms can share this class.

    L  a beam       throws the agent back to the start and carries on.  The
                    episode survives, so the cost is everything it had
                    walked — which is harder to plan around than dying and
                    is what makes room 1's short route a gamble rather than
                    a coin flip.
    H  a shaft      ends the run then and there.  Nothing is recovered and
                    there is no second attempt within the episode, which is
                    what puts a real price on room 2's ledge.
"""

import random

# The four actions, and how each one changes (row, col).
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS = (UP, DOWN, LEFT, RIGHT)

# Standing still. Not offered in every room — a room says which actions it
# has, and only one where waiting can be the right move includes it. Room 3's
# doors are on a cycle, so arriving early and holding is a real tactic there;
# in a room with nothing on a timer it would only be a way to waste a step.
WAIT = 5

# A fifth value for "has not moved yet", so the first step on oil has
# something defined to carry on from.
NO_DIRECTION = 4

ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT",
                WAIT: "WAIT"}
ACTION_ARROWS = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→", WAIT: "·"}

DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1),
          WAIT: (0, 0)}

# Sliding always means sideways, never backwards.
PERPENDICULAR = {UP: (LEFT, RIGHT), DOWN: (LEFT, RIGHT),
                 LEFT: (UP, DOWN), RIGHT: (UP, DOWN)}

# The tile alphabet. Each maps to one of the entity kinds in config.ENTITIES.
WALL = "#"
FLOOR = "."
START = "S"
GOAL = "E"
LASER = "L"
PIT = "H"
# A sound bridge behaves exactly like floor; it is a separate character only so
# it can be drawn as a span over the shaft rather than as ground.
BRIDGE = "G"
COLLAPSING = "C"
# Room 3's furniture. The three generators have to be brought up in order, so
# they are numbered rather than named — 'A', 'B' and 'C' would have collided
# with the battery and the collapsing plank.
GENERATORS = ("1", "2", "3")
# One key per generator, and only its own generator: key 'a' starts generator
# '1', 'b' starts '2', 'c' starts '3'. Paired by position in these two tuples.
KEYS = ("a", "b", "c")
SLIDING = "d"          # a door on a cycle: open some steps, shut the others
BLAST = "R"            # the reactor door: shut until every generator is up
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
    PIT: "pit",
    BRIDGE: "bridge",
    COLLAPSING: "collapsing",
    GENERATORS[0]: "generator",
    GENERATORS[1]: "generator",
    GENERATORS[2]: "generator",
    KEYS[0]: "key",
    KEYS[1]: "key",
    KEYS[2]: "key",
    SLIDING: "sliding",
    BLAST: "blast",
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

    def __init__(self, room, slip=0.2, seed=0, rewards=None, collapse=0.0):
        self.room = room
        self.grid = list(room["grid"])
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.slip = slip
        # How likely a plank is to give way under the step that lands on it.
        self.collapse = collapse
        # The room's table, with any entry a parameter is driving laid over
        # the top. Editing the table edits the *model*, which is why every
        # such parameter is reset-scope.
        self.rewards = dict(room["rewards"])
        if rewards:
            self.rewards.update(rewards)

        self.start = self._find(START)
        self.goal = self._find(GOAL)
        self.teleports = self._find_all(TELEPORT)
        self.has_battery_tile = bool(self._find_all(BATTERY))

        # One bit per collapsing plank, in map order, so a set of already-gone
        # planks packs into a single integer the state can carry.
        self.collapsing_order = self._find_all(COLLAPSING)
        self.collapsing_bit = {cell: index
                               for index, cell in
                               enumerate(self.collapsing_order)}

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

    def world_position(self, state):
        """Where the agent is, in the units the room screen draws in.

        That screen has one coordinate system for every room, continuous or
        not: a cell's *centre*, so that a position means the same thing for a
        wall, a drone and an agent. A grid room is therefore not a special
        case of drawing — it is a room whose agent happens to move between
        lattice points, which is why these come out on half-integers.
        """
        return {"x": state[1] + 0.5, "y": state[0] + 0.5}

    def all_states(self):
        """Every state the agent could be in, in a stable order.

        A cell for every battery, direction and set of collapsed planks it
        could be in it — which is what makes the battery collectable once, the
        oil have memory, and a plank stay broken.

        Each of the three extra dimensions collapses to a single value in a
        room that does not contain the thing, so a room pays only for what it
        actually has.
        """
        batteries = (0, 1) if self.has_battery_tile else (0,)
        directions = list(ACTIONS) + [NO_DIRECTION]
        masks = range(1 << len(self.collapsing_order))

        states = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col] == WALL:
                    continue
                for battery in batteries:
                    for last in directions:
                        for collapsed in masks:
                            states.append((row, col, battery, last, collapsed))
        return states

    def actions(self):
        return list(ACTIONS)

    def start_state(self):
        return (self.start[0], self.start[1], 0, NO_DIRECTION, 0)

    def is_collapsed(self, cell, mask):
        """Whether that plank has already gone, according to the mask."""
        bit = self.collapsing_bit.get(cell)
        return bit is not None and bool(mask & (1 << bit))

    def _with_collapsed(self, cell, mask):
        """The mask with that plank marked gone. Unchanged if it is not one."""
        bit = self.collapsing_bit.get(cell)
        return mask if bit is None else mask | (1 << bit)

    def is_terminal(self, state):
        """The exit ends a run, and so does a shaft.

        A laser does not: it throws the agent back to the door it came in
        by, which costs it everything it had walked and is a good deal more
        interesting to plan around than simply dying.
        """
        return self.grid[state[0]][state[1]] in (GOAL, PIT)

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
        # Indexed rather than unpacked: a room may carry more than five
        # numbers — room 3 adds a patrol position and a generator stage — and
        # this only ever needs the first two and the direction.
        row, col, last = state[0], state[1], state[3]
        surface = self.grid[row][col]
        loose = self.slip_of(surface)
        # Standing still cannot be sent sideways, whatever it is standing on.
        if action == WAIT or surface not in LOOSE or loose <= 0:
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

    def lands_on_sound_plank(self, state, direction):
        """Whether this move is the one that risks the crossing.

        THE PROBABILITY IS PER CROSSING, NOT PER PLANK.
        It used to be drawn again for every plank the agent stepped on, so a
        span of four planks failed with probability 1 - (1 - p)^4 while the
        control beside it was labelled "bridge collapse probability". At the
        0.10 default that is a 34% chance of dying, not a 10% one, and the
        number on the slider meant nothing a player could reason with.

        Now it is drawn once, on the step that takes the agent OFF solid
        ground and ON to the span. Survive that and the whole crossing is
        made; the planks in the middle are structure, not four more dice.
        `p` is therefore exactly the chance of losing a crossing, which is
        what the label has always claimed.

        THIS IS STILL MARKOV, AND NEEDS NO EXTRA STATE
        "Am I already on the span" is not history — it is the tile the agent
        is standing on, which is in the state. A step from deck or plank onto
        a plank continues a crossing that has already been paid for; a step
        from anywhere else begins one.
        """
        row, col, collapsed = state[0], state[1], state[4]
        delta_row, delta_col = DELTAS[direction]
        target = (row + delta_row, col + delta_col)
        if self.is_wall(*target):
            return False
        if self.grid[target[0]][target[1]] != COLLAPSING:
            return False
        if self.is_collapsed(target, collapsed):
            return False
        # Already out on the planks: this crossing has been paid for.
        #
        # The test is the PLANKS, not the whole span. The sound deck sections
        # at either end are solid ground -- they are what the agent steps off
        # from -- so treating them as "already crossing" meant the only way on
        # to the span never drew at all and the collapse never fired.
        return self.grid[row][col] != COLLAPSING

    def outcomes(self, state, direction):
        """Every way one direction can turn out, as (probability, result).

        A step is settled in two stages, and they are different in kind. The
        first is *where the agent goes* — the floor it pushed off from may send
        it sideways, which is `direction_probabilities`. The second is *whether
        what it landed on holds*, which is this. Only planks have a second
        stage; everywhere else there is one outcome with probability one.
        """
        if self.collapse > 0 and self.lands_on_sound_plank(state, direction):
            return [(self.collapse, self.resolve(state, direction, True)),
                    (1.0 - self.collapse, self.resolve(state, direction, False))]
        return [(1.0, self.resolve(state, direction))]

    def resolve(self, state, direction, gives_way=False):
        """Move one step and report (next_state, reward, done).

        `gives_way` is the second stage already decided: True means the plank
        this step lands on chose now to go. Deterministic given both.
        """
        row, col, battery, _, collapsed = state
        delta_row, delta_col = DELTAS[direction]
        target_row, target_col = row + delta_row, col + delta_col
        step = self.rewards["step"]

        blocked = self.is_wall(target_row, target_col)
        if not blocked and self.grid[target_row][target_col] == ONE_WAY:
            # A one-way door is a wall from every side but one.
            blocked = direction != ONE_WAY_DIRECTION

        if blocked:
            return ((row, col, battery, direction, collapsed),
                    step + self.rewards["wall"], False)

        tile = self.grid[target_row][target_col]

        # Crossing a plank breaks it. The break is booked on the way *off*,
        # which is what makes the span one-way: the agent is never dropped by
        # the plank it is standing on, only by one it comes back to.
        leaving = self._with_collapsed((row, col), collapsed)

        if tile == LASER:
            # Straight back to the start, with nothing kept but the battery
            # — it is bolted on, and the beam does not take it off again.
            return ((self.start[0], self.start[1], battery, NO_DIRECTION,
                     leaving), step + self.rewards["laser"], False)

        if tile == PIT:
            # The one thing here that ends a run badly. A beam is survivable
            # and this is not, which is what the reward has to say too.
            return ((target_row, target_col, battery, direction, leaving),
                    step + self.rewards["hazard"], True)

        if tile == COLLAPSING:
            gone = self.is_collapsed((target_row, target_col), collapsed)
            if gone or gives_way:
                # Two ways to be falling and no way to tell them apart from
                # down there: either there was no plank left, or this one
                # chose now to go. The plank is marked broken either way, so
                # a replay shows it give rather than only showing the fall.
                broken = self._with_collapsed((target_row, target_col), leaving)
                return ((target_row, target_col, battery, direction, broken),
                        step + self.rewards["hazard"], True)
            return ((target_row, target_col, battery, direction, leaving),
                    step, False)

        if tile == TELEPORT:
            landing = self._other_teleport((target_row, target_col))
            # Arriving by pad leaves no momentum to carry on with.
            return ((landing[0], landing[1], battery, NO_DIRECTION, leaving),
                    step + self.rewards["shortcut"], False)

        if tile == BATTERY and battery == 0:
            return ((target_row, target_col, 1, direction, leaving),
                    step + self.rewards["battery"], False)

        if tile == GOAL:
            return ((target_row, target_col, battery, direction, leaving),
                    step + self.rewards["goal"], True)

        return ((target_row, target_col, battery, direction, leaving),
                step, False)

    def transitions(self, state, action):
        """Every outcome of an action, as (probability, next, reward, done).

        This is the model. Dynamic Programming reads it; the model-free
        methods are never handed the environment at all, so they cannot.
        """
        if self.is_terminal(state):
            return [(1.0, state, 0.0, True)]

        # Both stages, multiplied out: where it goes, and whether that held.
        merged = {}
        probabilities = self.direction_probabilities(state, action)
        for direction, probability in probabilities.items():
            for share, outcome in self.outcomes(state, direction):
                merged[outcome] = merged.get(outcome, 0.0) + probability * share

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

        # The second stage. Drawn only when there is a plank to draw it for,
        # so a room without planks consumes exactly the randomness it used to
        # and its runs are unchanged.
        gives_way = (self.collapse > 0
                     and self.lands_on_sound_plank(state, actual)
                     and self.rng.random() < self.collapse)

        next_state, reward, done = self.resolve(state, actual, gives_way)
        tile = self.grid[next_state[0]][next_state[1]]
        # Where it was aiming, which is what says a laser was walked into
        # rather than that the agent is merely standing at the start again.
        aimed_row = state[0] + DELTAS[actual][0]
        aimed_col = state[1] + DELTAS[actual][1]
        burnt = (self.in_bounds(aimed_row, aimed_col)
                 and self.grid[aimed_row][aimed_col] == LASER)
        # A shaft needs no such care: the agent is left standing in it, so
        # the tile it ended on says outright that it fell. Stepping onto a
        # plank that has already gone is the same fall by a different route,
        # and the agent is left standing on the gap in exactly the same way.
        fell = tile == PIT or (tile == COLLAPSING and done)

        self.state = next_state
        self.steps += 1

        info = {
            "action": action,
            "direction": actual,
            "slipped": actual != action,
            "hazard": burnt or fell,
            "battery": next_state[2] > state[2],
            "goal": done and tile == GOAL,
            # Which planks gave way on this step, so a replay can show them
            # going rather than only their consequences.
            "collapsed": self.collapsed_cells(next_state[4] & ~state[4]),
        }
        self.augment_info(info, state, next_state, done)
        return next_state, reward, done, info

    def augment_info(self, info, state, next_state, done):
        """A room's chance to add to what a step reports. Nothing here.

        Room 3 uses it to say that being caught by the guard is a failure of
        the same kind as falling, so the readout counts it.
        """

    def collapsed_cells(self, mask):
        """The cells named by the set bits of a mask, in map order."""
        return [cell for cell, bit in self.collapsing_bit.items()
                if mask & (1 << bit)]

    def loose_entities(self):
        """Entities that are not tiles. None in a room where everything is.

        A patrolling guard has no square of the map to belong to, so it cannot
        come out of the layout scan; it is declared once here and then moved by
        every step. See `ReactorWorld`.
        """
        return []

    def frame_extras(self, state):
        """What a frame carries beyond the agent, for this room.

        Returns (entityStates, entityPositions) as the room screen's contract
        spells them, or (None, None). Kept on the environment because what
        moves and what changes appearance is the room's business, not the
        recorder's — see `ReactorWorld` for the room that has both.

        Every state here is *absolute* rather than a change since the last
        step. That matters because the same method feeds two things: a
        recorded frame, which is walked in order, and the live view, which
        can be joined at any moment. A diff would only work for the first.
        """
        if not self.collapsing_order:
            return None, None
        states = [
            {"id": "r%dc%d" % cell,
             "state": "collapsed" if self.is_collapsed(cell, state[4])
                      else "sound"}
            for cell in self.collapsing_order
        ]
        return states, None

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
            # Which planks have gone, so the grid can be drawn as it now is
            # rather than as it started.
            "collapsed": [list(cell)
                          for cell in self.collapsed_cells(self.state[4])],
        }
