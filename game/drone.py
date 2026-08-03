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


# How thick the masonry around a continuous chamber is drawn, in metres. One
# metre, because that is one cell of rooms 1 to 3 and the brick coursing has to
# match theirs exactly for the two to read as the same building.
CHAMBER_WALL = 1.0


def chamber_wall_entities(width, height, thickness=CHAMBER_WALL):
    """The laboratory wall around a continuous chamber, as ordinary tiles.

    THE SAME WALL AS ROOMS 1 TO 3, AND WHY IT IS STATED THIS WAY
    Rooms 1 to 3 are enclosed by `wall` tiles one cell square, drawn by the
    masonry recipe. Rooms 4 and 5 used a single `tunnelWall` entity spanning
    the whole chamber whose recipe drew a thin stroked frame with brackets and
    measurement ticks on it — a pressure vessel, not a laboratory, and the one
    thing that made the two halves of the building look like different games.
    So the wall is now a ring of the very same tiles, at the very same size,
    and the renderer draws them with the very same recipe. Nothing about the
    look is decided here or there twice.

    THE RING IS OUTSIDE THE WORLD, WHICH IS THE HONEST PLACE FOR IT
    A wall drawn *inside* the boundary would show the drone flying through
    masonry: the run ends when the agent's centre comes within its own radius
    of the edge, so the drone's hull touches the boundary exactly as it dies,
    and there is no room inside for a wall of any thickness. Drawn outside, the
    masonry's inner face sits on the boundary and the hull meets it at the
    instant of the collision — which is what a wall looks like when it is real.

    The renderer reserves the margin for it; see `wallMargin` in
    `definition.py`. Tiles are one unit square and laid on the unit grid, so
    the courses line up with the floor tiling and with the walls of rooms 1
    to 3 that they are meant to be continuous with.
    """
    tiles = []
    across = int(round(width / thickness))
    down = int(round(height / thickness))
    half = thickness / 2.0

    def tile(column, row):
        tiles.append({
            "id": "wall%d_%d" % (column, row),
            "type": "wall",
            "position": {"x": column * thickness + half,
                         "y": row * thickness + half},
            "size": {"width": thickness, "height": thickness},
        })

    # The two long runs, corners included, then the two sides between them.
    for column in range(-1, across + 1):
        tile(column, -1)
        tile(column, down)
    for row in range(0, down):
        tile(-1, row)
        tile(across, row)
    return tiles


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
        # THE TEST IS ON THE SPEED, NOT ON EACH COMPONENT.
        # It used to be `abs(vx) <= limit and abs(vy) <= limit`, which was a
        # real gradation while the velocity was continuous. With the velocity
        # discrete every component is already within 1, so that form is true
        # for every possible arrival -- the landing rule would be a no-op and
        # `hard_landing` could never fire. The magnitude keeps three regimes:
        #
        #   limit < 1.0        only a full stop counts as a landing
        #   1.0 <= limit < 1.42  arriving along one axis is a landing,
        #                        arriving diagonally is a crash
        #   limit >= 1.42      any arrival counts
        gentle = speed <= self.landing_speed
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

    def zones_at(self, x, y):
        """Every zone containing a point, in declaration order."""
        found = []
        for zone in self.zones:
            if (abs(x - zone["x"]) <= zone["width"] / 2.0
                    and abs(y - zone["y"]) <= zone["height"] / 2.0):
                found.append(zone)
        return found

    def distance_to_pad(self, x, y):
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

    # How likely a zone effect is to fire on one tick, per unit of strength.
    # `wind * dt` at wind = 1 is 0.02, so a zone shoves the drone about once a
    # second, which is the magnitude the continuous version had: it used to add
    # `wind * dt` to the velocity every tick, reaching one whole unit after
    # fifty ticks. Keeping the rate identical is what lets the wind slider mean
    # the same thing it meant before.
    ZONE_RATE = 1.0

    def _stepped(self, value, push):
        """One unit of change, clamped to the three velocities that exist."""
        limit = self.speed_limit
        return max(-limit, min(limit, value + push))

    def step(self, action):
        """Integrate one 0.02 s tick and report what happened.

        THE VELOCITY IS DISCRETE. THE MOVEMENT IS NOT.
        The assignment fixes `Vx, Vy` to {-1, 0, 1}, so an action steps a
        velocity component by exactly one unit and the component is clamped to
        those three values -- it is never 0.37. The *position* is still
        continuous: `x += vx * dt` moves the drone a fraction of a metre each
        tick, and dt is 0.02 s, so a flight is a smooth path and not a walk
        between cells.

        WHAT THAT COST, AND WHAT REPLACED IT
        Three things in the old continuous form produce fractional velocities
        by construction and therefore cannot survive: drag as an exponential
        decay, wind as a fractional acceleration, and a thrust magnitude. Each
        zone that used them is given the discrete equivalent instead, so no
        zone becomes decoration:

          wind zones   push the velocity one whole unit in the wind direction,
                       with probability `wind * dt` per tick -- the same
                       expected effect per second as the old acceleration.
          slow zone    pulls the velocity one unit TOWARDS zero, at a rate set
                       by its own `extra_drag`. This is drag, discretised.
          boost zone   makes a thrust reach full speed in one press instead of
                       one step at a time, which is what "overcharge" now
                       means when a step is already the whole range.

        The draws come from `self.rng`, which is seeded, so a flight is still
        reproducible from its seed.
        """
        x, y, vx, vy = self.state
        before = self.distance_to_pad(x, y)

        zones = self.zones_at(x, y)

        # 1. The action. One unit per press, or straight to full in a boost.
        push_x, push_y = THRUST[action]
        overcharged = any(zone.get("thrust_scale", 1.0) > 1.0 for zone in zones)
        if overcharged:
            limit = self.speed_limit
            if push_x:
                vx = math.copysign(limit, push_x)
            if push_y:
                vy = math.copysign(limit, push_y)
        else:
            vx = self._stepped(vx, push_x)
            vy = self._stepped(vy, push_y)

        # 2. Wind: a whole-unit shove, so the velocity stays one of three.
        for zone in zones:
            wind = zone.get("wind")
            if not wind:
                continue
            chance = self.wind * self.dt * self.ZONE_RATE
            if wind[0] and self.rng.random() < chance * abs(wind[0]):
                vx = self._stepped(vx, math.copysign(1.0, wind[0]))
            if wind[1] and self.rng.random() < chance * abs(wind[1]):
                vy = self._stepped(vy, math.copysign(1.0, wind[1]))

        # 3. Drag, discretised: a pull of one unit towards rest. Only the zones
        #    that declare it -- there is no global drag any more, because a
        #    decay applied every tick cannot leave a velocity in {-1, 0, 1}.
        for zone in zones:
            extra = zone.get("extra_drag", 0.0)
            if not extra:
                continue
            chance = extra * self.dt * self.ZONE_RATE
            if vx and self.rng.random() < chance:
                vx = self._stepped(vx, -math.copysign(1.0, vx))
            if vy and self.rng.random() < chance:
                vy = self._stepped(vy, -math.copysign(1.0, vy))

        # The three values are all that exist, so this is exact rather than a
        # tolerance. Rounding guards against a copysign leaving -0.0.
        vx = float(round(vx))
        vy = float(round(vy))

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
        after = self.distance_to_pad(x, y)
        reward += (before - after) * self.rewards["progress"]

        # The danger zones, charged once per entry.
        for zone in self.zones_at(x, y):
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

    @staticmethod
    def _blows(vector):
        """A vector turned into the word the drawing recipes want.

        `stream` and `vent` in `shapes.js` point their chevrons by a compass
        word rather than by an angle, and that translation is done here on
        purpose: the wind model lives in this file, and the renderer must not
        be in a position to disagree with it about which way the air moves.
        The magnitude goes over as well, so the drawing can show strength
        without recomputing anything.
        """
        if abs(vector[0]) >= abs(vector[1]):
            return "right" if vector[0] > 0 else "left"
        return "down" if vector[1] > 0 else "up"

    def entities(self):
        """The chamber's furniture, in the room screen's contract shape.

        A grid room's entities are read off its layout by `definition.build`,
        one per cell. There is no layout here, so the room states its own
        entities and this is where they come from. Positions are centres in
        world units, which is the one coordinate system the contract has.

        NOTHING HERE IS PHYSICS
        Every entity below is a description of something the model already
        does. The fans are the clearest case: they apply no force at all. The
        wind zone applies the force, and a fan is drawn at the mouth of that
        zone so the force has a visible cause. Deleting every fan would change
        how the chamber reads and not one number in `step`.
        """
        pad = self.pad
        entities = chamber_wall_entities(self.width, self.height)
        entities += [{
            "id": "pad",
            "type": "pad",
            "position": {"x": pad["x"], "y": pad["y"]},
            "size": {"width": pad["width"], "height": pad["height"]},
            # The platform draws its own guidance marks, and it needs to know
            # how slow is slow enough to say so.
            "appearance": {"landingSpeed": self.landing_speed},
        }, {
            "id": "launch",
            "type": "start",
            "position": {"x": self.start_position[0],
                         "y": self.start_position[1]},
            "size": {"width": 0.7, "height": 0.7},
        }]

        for index, pillar in enumerate(self.pillars):
            # A disc described as the square that bounds it. The recipe draws
            # the disc to the full width of that box, so what is on screen is
            # the circle `_crashed` actually tests against — the radius is not
            # written down twice.
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
            wind = zone.get("wind")
            if wind:
                strength = self.wind
                blowing = (wind[0] * strength, wind[1] * strength)
                entity["appearance"] = {
                    "blows": self._blows(wind),
                    # Both are handed over so the drawing can show *how hard*
                    # without owning any part of the model.
                    "vector": {"x": blowing[0], "y": blowing[1]},
                    "strength": strength,
                }
            entities.append(entity)

        # The fans, at the mouth of each wind zone: decoration with a job,
        # which is to make it obvious where the air comes from and which way
        # it goes. Placed against whichever chamber edge the zone blows away
        # from, so a band blowing down gets fans along its top.
        for zone in self.zones:
            wind = zone.get("wind")
            if not wind:
                continue
            entities.extend(self._fans_for(zone, wind))

        # Two lamps flanking the landing platform, and one over the
        # overcharge. Both are the sort of thing a real test chamber would
        # have and both mark something that matters.
        entities.append({
            "id": "lamp-pad",
            "type": "warningLight",
            "position": {"x": pad["x"] - pad["width"] / 2.0 - 0.32,
                         "y": pad["y"]},
            "size": {"width": 0.3, "height": 0.3},
            "appearance": {"color": "--goal"},
        })
        for zone in self.zones:
            if zone.get("danger"):
                entities.append({
                    "id": "lamp-" + zone["id"],
                    "type": "warningLight",
                    "position": {"x": zone["x"],
                                 "y": zone["y"] - zone["height"] / 2.0 - 0.3},
                    "size": {"width": 0.3, "height": 0.3},
                })

        return entities

    def _fans_for(self, zone, wind):
        """The bank of fans driving one wind zone.

        Positioned just outside the zone on its upwind edge and clamped inside
        the chamber, so they read as the thing pushing the air through it.
        """
        blows = self._blows(wind)
        vertical = blows in ("up", "down")
        span = zone["width"] if vertical else zone["height"]
        # As many fans as fit across the mouth without crowding, at least one.
        count = max(1, min(3, int(span / 0.9)))
        size = min(0.95, span / count * 0.85)
        depth = 0.62

        if vertical:
            edge = (zone["y"] - zone["height"] / 2.0 if blows == "down"
                    else zone["y"] + zone["height"] / 2.0)
            offset = -depth / 2.0 if blows == "down" else depth / 2.0
            along = zone["x"]
        else:
            edge = (zone["x"] - zone["width"] / 2.0 if blows == "right"
                    else zone["x"] + zone["width"] / 2.0)
            offset = -depth / 2.0 if blows == "right" else depth / 2.0
            along = zone["y"]

        fans = []
        for index in range(count):
            share = (index - (count - 1) / 2.0) * (span / count)
            centre = along + share
            if vertical:
                x, y = centre, edge + offset
            else:
                x, y = edge + offset, centre
            fans.append({
                "id": "%s-fan%d" % (zone["id"], index),
                "type": "fan",
                # Clamped so a fan never straddles the chamber wall.
                "position": {"x": min(max(x, size / 2.0),
                                      self.width - size / 2.0),
                             "y": min(max(y, size / 2.0),
                                      self.height - size / 2.0)},
                "size": {"width": size, "height": size},
                "appearance": {"blows": blows, "strength": self.wind,
                               # Fans in one bank are given different phases so
                               # the bank does not turn as a single object.
                               "phase": index * 0.37},
            })
        return fans

    def loose_entities(self):
        """Nothing moves in this room but the drone. See the module docstring."""
        return []

    # How near the platform counts as "on approach", in metres. Only the
    # warning light uses it; nothing in the model reads it.
    APPROACH_RANGE = 1.8

    def frame_extras(self, state):
        """What the landing platform looks like at this moment.

        Nothing in this chamber moves and nothing changes colour except the
        platform, which reports how the approach is going. It is here rather
        than in the renderer for the reason the whole contract exists: the
        screen must never be in a position to announce a landing the
        environment did not agree to. The four words below are the only way it
        can say so, and only `_landing` decides which one it is.

        A pure function of the state, like every other frame field, so a
        recorded step and the live view cannot describe the same moment
        differently — and so scrubbing a replay backwards shows the approach
        going wrong again rather than staying red.
        """
        outcome, speed = self._landing(state)
        if outcome == "landed":
            look = "landed"
        elif outcome == "crashed":
            look = "crashed"
        # THE WARNING USES THE SAME RULE AS THE LANDING.
        # It tested each component, which the landing no longer does -- and
        # with a discrete velocity no component can exceed 1, so at the default
        # limit the warning could never fire while a diagonal arrival was still
        # a crash. The screen would have stayed calm right up to the wreck.
        elif (self.distance_to_pad(state[0], state[1]) <= self.APPROACH_RANGE
              and speed > self.landing_speed):
            # Near, and too fast to land if it arrived now.
            look = "fast"
        else:
            look = "clear"
        return [{"id": "pad", "state": look}], None

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
