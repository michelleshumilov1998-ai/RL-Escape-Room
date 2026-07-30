"""Room 3 — the Reactor Control Chamber.  A grid with things that move.

Rooms 1 and 2 are static: a tile means the same thing on every step of every
episode.  This room is not.  A guard walks a fixed patrol, and the two sliding
doors on the central shaft open and shut on a cycle, so the same cell is safe
at one moment and fatal at the next and a door is a wall half the time.

That is the whole reason the room exists.  Q-Learning is mandated here, and
what it has to cope with is a reward a very long way from the first move that
leads towards it.  Three generators must be brought up **in order**, and each
one needs **its own key** fetched first — six errands before the reactor door
will even open, and the exit pays out only after the last of them.

--------------------------------------------------------------------------
THE STATE
--------------------------------------------------------------------------
    (row, col, battery, last_direction, collapsed, stage, guard, keys)

The first five belong to the base grid and three of them are dead weight
here — no battery, no planks, and nothing loose to carry momentum — but they
cost one value each and keeping the layout identical means every algorithm,
the cell projection and the renderer all work unchanged.  `last_direction` is
pinned to "none" for exactly that reason: it would otherwise multiply the
table by six to describe something no rule in this room reads.

The three that matter:

  * stage   how far through the generator sequence: 0 none, 3 all of them and
            the reactor door open.  They only count in order, so standing on
            the third generator first does nothing whatever.
  * guard   where the patrol has got to, as an index into a fixed loop.
  * keys    which keys are in hand, one bit each.  A generator needs its own
            key and no other, so this cannot be folded into `stage`: the
            errands and the sequence advance independently.

--------------------------------------------------------------------------
WHY THE DOORS ARE DRIVEN BY THE GUARD
--------------------------------------------------------------------------
A door on a timer would make the room non-Markovian: the same cell means
different things at different moments, and nothing in the state would say
which.  The obvious fix is to put a clock in the state, which works and costs
a whole dimension.

Instead the door cycle is *derived* from the patrol index, which is in the
state already.  The patrol length is a whole multiple of the door period, so
the pattern repeats cleanly, and the agent can always work out whether a door
is open from something it can see.  One dimension, two moving things.

--------------------------------------------------------------------------
THE ORDER OF EVENTS IN A STEP
--------------------------------------------------------------------------
    1. R-5 acts.  A wall, a shut sliding door and the reactor door before the
       reactor is up all block it.
    2. Where it landed takes effect: a key, a generator, the exit.
    3. The guard advances one cell along its patrol.
    4. If they are now on the same cell — or swapped past each other — the
       run ends.

Step 4 has to consider the swap as well as the collision.  Two things moving
one cell per step towards each other pass *through* each other otherwise, and
a guard you can walk through is not a guard.
"""

from game.grid import (ACTIONS, BLAST, DELTAS, GENERATORS, KEYS, NO_DIRECTION,
                       SLIDING, WAIT, GridWorld)


class ReactorWorld(GridWorld):
    """The grid, plus a patrol, doors on a cycle, and a keyed sequence."""

    def __init__(self, room, slip=0.0, seed=0, rewards=None, collapse=0.0):
        GridWorld.__init__(self, room, slip=slip, seed=seed, rewards=rewards,
                           collapse=collapse)

        self.patrol = [tuple(cell) for cell in room["patrol"]]
        self.door_period = room["door_period"]
        self.door_open_for = room["door_open_for"]
        if len(self.patrol) % self.door_period:
            raise ValueError(
                "the patrol (%d) must be a whole number of door cycles (%d), "
                "or the door pattern does not repeat and the guard index "
                "stops being enough to know the doors from"
                % (len(self.patrol), self.door_period))

        # Generator n and key n are the same errand seen from two ends, so
        # they are found in the same order and indexed by the same number.
        self.generators = [self._find(tile) for tile in GENERATORS]
        self.keys = [self._find(tile) for tile in KEYS]
        self.stages = len(self.generators)
        self.state = self.start_state()

    # ------------------------------------------------------------------
    # The moving parts
    # ------------------------------------------------------------------

    def guard_cell(self, guard):
        return self.patrol[guard % len(self.patrol)]

    def next_guard(self, guard):
        return (guard + 1) % len(self.patrol)

    def doors_open(self, guard):
        """Whether the sliding doors are open at this point in the cycle."""
        return (guard % self.door_period) < self.door_open_for

    def holds_key(self, keys, index):
        return bool(keys & (1 << index))

    def actions(self):
        """The four directions, and waiting.

        Waiting is only worth having where something else is on a clock, and
        here it is the difference between arriving at a shut door and being
        able to do anything about it.
        """
        return list(ACTIONS) + [WAIT]

    # ------------------------------------------------------------------
    # The state
    # ------------------------------------------------------------------

    def start_state(self):
        return (self.start[0], self.start[1], 0, NO_DIRECTION, 0, 0, 0, 0)

    def all_states(self):
        states = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.is_wall(row, col):
                    continue
                for stage in range(self.stages + 1):
                    for guard in range(len(self.patrol)):
                        for keys in range(1 << len(self.keys)):
                            # `last_direction` is pinned: nothing here reads
                            # it, and enumerating it would multiply the table
                            # by six to no purpose.
                            states.append((row, col, 0, NO_DIRECTION, 0,
                                           stage, guard, keys))
        return states

    def is_terminal(self, state):
        """Only the exit ends a run well.

        Being caught is terminal too, but a caught state is never handed out
        as somewhere the agent is standing — `resolve` reports the run over on
        the step it happens, so there is no such state to ask about.
        """
        return (state[0], state[1]) == self.goal

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------

    def _blocked(self, state, target):
        row, col = target
        if self.is_wall(row, col):
            return True
        tile = self.grid[row][col]
        if tile == SLIDING and not self.doors_open(state[6]):
            # A shut door is a wall, and it is a wall the agent can see is
            # going to open again two steps later.
            return True
        if tile == BLAST and state[5] < self.stages:
            return True
        return False

    def resolve(self, state, direction, gives_way=False):
        """Move one step and report (next_state, reward, done)."""
        row, col, _, _, _, stage, guard, keys = state
        delta_row, delta_col = DELTAS[direction]
        target = (row + delta_row, col + delta_col)
        step = self.rewards["step"]
        moved_guard = self.next_guard(guard)

        if self._blocked(state, target):
            landed = (row, col)
            reward = step + self.rewards["wall"]
        else:
            landed = target
            reward = step
            # A key, if it is still there to be picked up.
            if landed in self.keys:
                index = self.keys.index(landed)
                if not self.holds_key(keys, index):
                    keys |= 1 << index
                    reward += self.rewards["key"]

            # A generator, if it is the next in the sequence and its own key
            # is in hand. Either condition missing and the cell is just floor.
            if stage < self.stages and landed == self.generators[stage] \
                    and self.holds_key(keys, stage):
                stage += 1
                reward += self.rewards["generator"]

            if landed == self.goal:
                return ((landed[0], landed[1], 0, NO_DIRECTION, 0, stage,
                         moved_guard, keys),
                        reward + self.rewards["goal"], True)

        # The guard moves after the agent, and catching it ends the run. The
        # swap has to count as a catch: two things stepping one cell towards
        # each other would otherwise pass straight through.
        caught = (landed == self.guard_cell(moved_guard)
                  or (landed == self.guard_cell(guard)
                      and (row, col) == self.guard_cell(moved_guard)))
        if caught:
            return ((landed[0], landed[1], 0, NO_DIRECTION, 0, stage,
                     moved_guard, keys),
                    reward + self.rewards["caught"], True)

        return ((landed[0], landed[1], 0, NO_DIRECTION, 0, stage, moved_guard,
                 keys), reward, False)

    # ------------------------------------------------------------------
    # What the page is told
    # ------------------------------------------------------------------

    def augment_info(self, info, state, next_state, done):
        """Being caught is a failure of the same kind as falling.

        The readout counts one number for "ended badly", and without this a
        run that walked into the guard every time would report a hundred per
        cent of episodes ending in nothing in particular.
        """
        caught = done and not info["goal"]
        info["hazard"] = caught
        info["caught"] = caught
        info["guard"] = list(self.guard_cell(next_state[6]))
        info["doorsOpen"] = self.doors_open(next_state[6])
        info["stage"] = next_state[5]
        info["keys"] = next_state[7]

    def loose_entities(self):
        """The guard, declared once at the start of its patrol."""
        start = self.patrol[0]
        return [{
            "id": "guard",
            "type": "guard",
            "position": {"x": start[1] + 0.5, "y": start[0] + 0.5},
            "size": {"width": 1.0, "height": 1.0},
        }]

    def frame_extras(self, state):
        """Where the guard is, which doors are open, which keys are gone.

        The guard is declared once as an entity and moved per step; the doors
        and the keys are entities whose *appearance* changes. Two different
        things, which is why the contract keeps them in separate fields.
        """
        guard = self.guard_cell(state[6])
        positions = [{"id": "guard",
                      "position": {"x": guard[1] + 0.5, "y": guard[0] + 0.5}}]

        shown = "open" if self.doors_open(state[6]) else "shut"
        states = [{"id": "r%dc%d" % cell, "state": shown}
                  for cell in self._find_all(SLIDING)]
        # A generator is running once the sequence has passed it. `stage` is
        # exactly that count, so this needs nothing else to work it out.
        for index, cell in enumerate(self.generators):
            states.append({"id": "r%dc%d" % cell,
                           "state": "on" if index < state[5] else "off"})
        # A key already in hand is no longer lying on the floor.
        for index, cell in enumerate(self.keys):
            states.append({
                "id": "r%dc%d" % cell,
                "state": "taken" if self.holds_key(state[7], index) else "there",
            })
        return states, positions

    def snapshot(self):
        """The moving part: the guard, the doors, and how far through it is."""
        state = self.state
        return {
            "agent": list(self.cell_of(state)),
            "battery": False,
            "steps": self.steps,
            "terminal": self.is_terminal(state),
            "collapsed": [],
            "guard": list(self.guard_cell(state[6])),
            "doorsOpen": self.doors_open(state[6]),
            "stage": state[5],
            "stages": self.stages,
            "keys": [self.holds_key(state[7], index)
                     for index in range(len(self.keys))],
            "generatorsOn": [index < state[5]
                             for index in range(len(self.generators))],
        }
