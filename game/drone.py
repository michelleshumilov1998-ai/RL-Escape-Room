"""Room 4's world: a continuous flight chamber, with no grid under it.

The first three chambers are grids. A state there is a cell plus a few flags,
there are finitely many of them, and a table with one row per state is a
perfectly good way to hold what has been learned. This room is the case where
that stops working.

WHAT MAKES IT DIFFERENT
The drone's state is four real numbers — where it is and how fast it is
going — and there are infinitely many of those. No table can have a row per
state, and two states that differ in the sixth decimal place are the same
situation and must not be learned about separately. That is the whole reason
function approximation exists, and `algorithms/tile_coding.py` is the answer
this room is built to demonstrate.

    state = (x, y, vx, vy)      metres, and metres per second

Position is continuous inside a 10 x 10 m room. Velocity is held inside
[-1, 1] m/s on each axis, which is the range the assignment fixes. An action
is a thrust, so the agent steers an acceleration and never sets its position
directly: what it does now is felt several steps later, and that delay is most
of the difficulty.

WHY THERE IS NO MOVING GATE
An earlier sketch of this room had a barrier that rose and fell on a cycle.
It is not here, and the reason is the same one that shapes room 3. A thing
that moves on a timer makes the room non-Markov: the same four numbers would
mean "safe" at one moment and "about to be crushed" at the next, with nothing
in the state saying which. Room 3 could afford its sliding doors because it
derives them from the guard's patrol index, which *is* part of the state.
Here the state is fixed at four numbers by the assignment, so there is no
free dimension to hang a phase on, and adding a fifth would be answering a
different question from the one this room asks.

Everything in the chamber is therefore *static in time* and interesting in
space: the forces depend on where the drone is, never on when it got there.
Wind is constant within a run for the same reason — it is a reset-scope
parameter, so it can be changed between runs and studied, without ever being
a hidden variable inside one.

THE TASK
Reach the landing pad *slowly*. Arriving inside the pad with either speed
component above the landing threshold is a crash rather than a landing, which
is what stops the answer from being "point at the pad and hold full thrust".
"""

import math
import random

# The actions. Five of them: hold, and a push along each axis. Diagonals are
# deliberately absent — they are reachable by alternating pushes, and leaving
# them out keeps the action set the same size as room 3's.
HOLD = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

ACTIONS = (HOLD, UP, DOWN, LEFT, RIGHT)

ACTION_NAMES = {HOLD: "HOLD", UP: "UP", DOWN: "DOWN", LEFT: "LEFT",
                RIGHT: "RIGHT"}

# y increases downwards, matching the canvas and the way every other room in
# this project is written down. "UP" therefore pushes towards smaller y.
THRUST = {
    HOLD: (0.0, 0.0),
    UP: (0.0, -1.0),
    DOWN: (0.0, 1.0),
    LEFT: (-1.0, 0.0),
    RIGHT: (1.0, 0.0),
}


class DroneWorld:
    """A continuous 10 x 10 m chamber flown by thrust.

    The same four-method surface every room offers — `reset`, `step`,
    `actions`, `world_position` — so the session loop, the recorder and the
    replay never learn that this room is not a grid.
    """

    # What the rest of the code asks instead of checking the class.
    is_grid = False

    def __init__(self, room, slip=0.0, seed=0, rewards=None, collapse=0.0):
        # `slip` and `collapse` are accepted and ignored: they are the grid
        # rooms' parameters, and `Session._build` hands the same arguments to
        # every world rather than growing a branch per room.
        self.room = room
        self.rng = random.Random(seed)

        self.width = float(room["size"][0])
        self.height = float(room["size"][1])
        self.dt = float(room["dt"])
        self.thrust = float(room["thrust"])
        self.drag = float(room["drag"])
        self.speed_limit = float(room["speed_limit"])

        self.start_position = tuple(room["start"])
        self.pad = dict(room["pad"])
        self.landing_speed = float(room.get("landing_speed",
                                            room["pad"]["landing_speed"]))

        # The furniture. Each is a plain dict of numbers; nothing here is
        # markup and nothing here draws.
        self.pillars = [dict(pillar) for pillar in room.get("pillars", [])]
        self.zones = [dict(zone) for zone in room.get("zones", [])]

        self.rewards = dict(room["rewards"])
        if rewards:
            self.rewards.update(rewards)

        # How hard the wind blows, as a multiplier on every wind zone's own
        # vector. A parameter rather than a constant so the room can be
        # studied at several strengths; reset-scope, because it is part of the
        # model and everything learned was learned against the old value.
        self.wind = float(room.get("wind", 1.0))

        self.state = self.start_state()
        self.steps = 0
        # Which danger zones have already been charged for. The entry cost is
        # paid once per visit, not once per step: at 0.02 s a step, a per-step
        # charge would make a single crossing cost hundreds.
        self._charged = set()

    # ------------------------------------------------------------------
    # The state
    # ------------------------------------------------------------------

    def start_state(self):
        return (self.start_position[0], self.start_position[1], 0.0, 0.0)

    def actions(self):
        return ACTIONS

    def all_states(self):
        """There is no such list, and that is the point of this room.

        Raised rather than returned empty so a planner asked to run here fails
        with something that explains itself instead of quietly converging on
        an empty value table.
        """
        raise NotImplementedError(
            "room 4's state is continuous: there is no enumerable state space, "
            "which is why it is solved with function approximation rather than "
            "with a table or a planner")

    def transitions(self, state, action):
        """The model. Not given here — see `all_states`."""
        raise NotImplementedError(
            "room 4 does not hand out a model: the only way to find out what "
            "a thrust does is to apply it")

    def world_position(self, state):
        """Where the drone is, in the units the screen draws in.

        A continuous room needs no translation at all: the state already *is*
        a position in world units. Grid rooms have to add a half to reach a
        cell's centre; here there are no cells to be at the centre of.
        """
        return {"x": state[0], "y": state[1]}

    def world_velocity(self, state):
        """How fast it is going. Grid rooms answer None to this."""
        return {"x": state[2], "y": state[3]}

    def is_terminal(self, state):
        return self._landing(state)[0] is not None or self._crashed(state)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _in_pad(self, x, y):
        pad = self.pad
        return (abs(x - pad["x"]) <= pad["width"] / 2.0
                and abs(y - pad["y"]) <= pad["height"] / 2.0)

    def _landing(self, state):
        """(outcome, speed) for a state on the pad, (None, speed) otherwise.

        The distinction this draws is the room: touching the pad is not the
        task, touching it *slowly* is. `outcome` is "landed" or "crashed".
        """
        x, y, vx, vy = state
        speed = math.hypot(vx, vy)
        if not self._in_pad(x, y):
            return None, speed
        gentle = (abs(vx) <= self.landing_speed
                  and abs(vy) <= self.landing_speed)
        return ("landed" if gentle else "crashed"), speed

    def _crashed(self, state):
        """Outside the chamber, or inside something solid."""
        x, y = state[0], state[1]
        if x <= 0.0 or x >= self.width or y <= 0.0 or y >= self.height:
            return True
        for pillar in self.pillars:
            if math.hypot(x - pillar["x"], y - pillar["y"]) <= pillar["radius"]:
                return True
        return False

    def _zones_at(self, x, y):
        """Every zone containing a point, in declaration order."""
        found = []
        for zone in self.zones:
            if (abs(x - zone["x"]) <= zone["width"] / 2.0
                    and abs(y - zone["y"]) <= zone["height"] / 2.0):
                found.append(zone)
        return found

    def _distance_to_pad(self, x, y):
        """Straight-line distance to the pad's centre.

        Deliberately the centre and not the nearest edge: the shaped reward
        below is a difference of two of these, and measuring to an edge makes
        that difference go flat the moment the drone is anywhere over the pad,
        exactly where the last and most delicate part of the approach is.
        """
        return math.hypot(x - self.pad["x"], y - self.pad["y"])

    # ------------------------------------------------------------------
    # One step of flight
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.state = self.start_state()
        self.steps = 0
        self._charged = set()
        return self.state

    def step(self, action):
        """Integrate one 0.02 s tick and report what happened.

        Semi-implicit Euler: the velocity is updated first and the position
        moved with the *new* velocity. At this step size the difference from
        the explicit form is small, but it is the stable one of the two and it
        costs nothing.
        """
        x, y, vx, vy = self.state
        before = self._distance_to_pad(x, y)

        zones = self._zones_at(x, y)

        # Thrust, scaled by any zone that changes what a push is worth.
        push_x, push_y = THRUST[action]
        thrust = self.thrust
        for zone in zones:
            thrust *= zone.get("thrust_scale", 1.0)
        vx += push_x * thrust * self.dt
        vy += push_y * thrust * self.dt

        # Wind: an acceleration that depends on where the drone is and on
        # nothing else, so it stays a function of the state.
        for zone in zones:
            wind = zone.get("wind")
            if wind:
                vx += wind[0] * self.wind * self.dt
                vy += wind[1] * self.wind * self.dt

        # Drag, and any zone that adds to it. Applied as a decay over the tick
        # rather than subtracted, so it can never push the drone backwards
        # however large the coefficient is set.
        drag = self.drag
        for zone in zones:
            drag += zone.get("extra_drag", 0.0)
        decay = math.exp(-drag * self.dt)
        vx *= decay
        vy *= decay

        # The assignment fixes the velocity range, so it is a hard clamp and
        # not a soft penalty.
        limit = self.speed_limit
        vx = max(-limit, min(limit, vx))
        vy = max(-limit, min(limit, vy))

        x += vx * self.dt
        y += vy * self.dt

        next_state = (x, y, vx, vy)
        self.state = next_state
        self.steps += 1

        # ---- what it earned -------------------------------------------

        reward = self.rewards["step"]

        # Shaped by progress: closing on the pad pays, drifting away costs.
        # Written as a difference of distances so that the total over any
        # round trip is zero — a shaping term that paid for approach without
        # charging for retreat would make circling the pad profitable.
        after = self._distance_to_pad(x, y)
        reward += (before - after) * self.rewards["progress"]

        # The danger zones, charged once per entry.
        for zone in self._zones_at(x, y):
            if zone.get("danger") and zone["id"] not in self._charged:
                self._charged.add(zone["id"])
                reward += self.rewards["danger"]

        outcome, speed = self._landing(next_state)
        crashed = self._crashed(next_state)
        done = False
        landed = False

        if outcome == "landed":
            reward += self.rewards["goal"]
            done = True
            landed = True
        elif outcome == "crashed":
            # On the pad but far too fast. A crash-landing, and the run is
            # over: letting it bounce and try again would make speed free.
            reward += self.rewards["hard_landing"]
            done = True
        elif crashed:
            reward += self.rewards["wall"]
            done = True

        info = {
            "action": action,
            # Nothing here slips: the room is deterministic given the state.
            # The key is reported because every other room reports it and the
            # recorder does not branch on which room it is recording.
            "slipped": False,
            "hazard": (crashed or outcome == "crashed"),
            "goal": landed,
            "speed": speed,
            "landed": landed,
            "hardLanding": outcome == "crashed",
            "distance": after,
        }
        return next_state, reward, done, info

    # ------------------------------------------------------------------
    # What the screen is given
    # ------------------------------------------------------------------

    def entities(self):
        """The chamber's furniture, in the room screen's contract shape.

        A grid room's entities are read off its layout by `definition.build`,
        one per cell. There is no layout here, so the room states its own
        entities and this is where they come from. Positions are centres in
        world units, which is the one coordinate system the contract has.
        """
        entities = [{
            "id": "pad",
            "type": "pad",
            "position": {"x": self.pad["x"], "y": self.pad["y"]},
            "size": {"width": self.pad["width"], "height": self.pad["height"]},
        }, {
            "id": "launch",
            "type": "start",
            "position": {"x": self.start_position[0],
                         "y": self.start_position[1]},
            "size": {"width": 0.6, "height": 0.6},
        }]

        for index, pillar in enumerate(self.pillars):
            # A circle described as a square of the same span: the renderer is
            # told which *shape* to draw by the entity type, and "pillar"
            # draws a disc inside the box it is given.
            entities.append({
                "id": "pillar%d" % index,
                "type": "pillar",
                "position": {"x": pillar["x"], "y": pillar["y"]},
                "size": {"width": pillar["radius"] * 2,
                         "height": pillar["radius"] * 2},
            })

        for zone in self.zones:
            entity = {
                "id": zone["id"],
                "type": zone["type"],
                "position": {"x": zone["x"], "y": zone["y"]},
                "size": {"width": zone["width"], "height": zone["height"]},
            }
            # A wind zone carries its direction, so the renderer can draw the
            # arrows the right way round without knowing what wind is. Scaled
            # by the run's wind strength, so the arrows describe the run.
            wind = zone.get("wind")
            if wind:
                entity["appearance"] = {
                    "vector": {"x": wind[0] * self.wind,
                               "y": wind[1] * self.wind},
                }
            entities.append(entity)

        return entities

    def loose_entities(self):
        """Nothing moves in this room but the drone. See the module docstring."""
        return []

    def frame_extras(self, state):
        """Nothing in this chamber changes appearance or position."""
        return None, None

    def layout_snapshot(self):
        """There is no layout. Kept so every world answers the same questions."""
        return {"rows": None, "cols": None, "tiles": None,
                "start": list(self.start_position),
                "goal": [self.pad["x"], self.pad["y"]]}

    def snapshot(self):
        outcome, speed = self._landing(self.state)
        return {
            "agent": [self.state[0], self.state[1]],
            "velocity": [self.state[2], self.state[3]],
            "speed": speed,
            "steps": self.steps,
            "terminal": self.is_terminal(self.state),
            "onPad": self._in_pad(self.state[0], self.state[1]),
            "landed": outcome == "landed",
        }
