"""Room 5's world: an automated warehouse that rearranges itself.

Rooms 1 to 4 each have *one* layout. The agent may take hundreds of episodes to
solve it, but it is solving the same room every time, and nothing it learns has
to be worth anything anywhere else. This room is the one where that stops being
enough.

    Every episode is a different warehouse, drawn from a seed.
    The agent never sees the map. It sees what its sensors reach.
    The final score is measured on layouts it was never trained on.

THE STATE, AND WHY IT HAS A PHASE IN IT

    state = (x, y, vx, vy, stage, phase)

`x, y, vx, vy` are room 4's continuous flight state, unchanged in kind: an
action is a thrust, drag bleeds speed off, velocity is clamped, position
integrates with the new velocity.

`stage` is the mission: 0 while the control terminal is still to be reached, 1
once it has been. The exit does not open in stage 0, so the room cannot be
finished by walking to the door.

`phase` is what keeps the room Markov with things moving in it. Every dynamic
obstacle walks a patrol whose period *divides* `PHASE_PERIOD`, so where all of
them are is a pure function of one bounded integer — exactly the trick room 3
uses to derive its sliding doors from the guard's patrol index. Obstacles on a
wall clock would make the same six numbers mean "clear" at one moment and
"about to be run over" at the next, with nothing in the state saying which.

    `ReactorWorld` raises if a room breaks that divisibility. So does this.

WHAT THE AGENT ACTUALLY GETS, WHICH IS NOT THE STATE
The environment knows where every obstacle is. The agent does not. It is handed
fourteen numbers — where it is, how fast, where its current objective is, and
three forward sensor rays that reach `sensor_range` metres and no further. So
the *environment* is Markov and the *observation* is not: two different
warehouses can present the same fourteen numbers, and an obstacle four metres
behind a shelf is invisible until it is not. That gap is the point of the room
and it is stated rather than papered over — see `observation`.
"""

import math
import random

from game.drone import (ACTION_NAMES, ACTIONS, HOLD, THRUST,
                        chamber_wall_entities)

# How long the whole warehouse takes to return to the arrangement it started
# in. Every patrol period divides this, so `phase` is bounded and the obstacle
# positions are a function of it alone.
#
# 3600 ticks is 72 seconds of flight, which is longer than the 60 seconds an
# episode lasts at the default step limit — so in practice the warehouse does
# not repeat within one episode at all. It was 480, which is 9.6 seconds, and
# that was far too short to hold a patrol slow enough to be dodged: the whole
# lap had to fit inside it, so a 12 m circuit could not be walked at less than
# 1.25 m/s however the speed was set. A drone capped at 1 m/s cannot avoid
# traffic that is quicker than it is.
PHASE_PERIOD = 3600

# The periods the decorative cargo lines may cycle at. Divisors of
# PHASE_PERIOD, so the crates on them repeat with everything else.
PATROL_PERIODS = (600, 900, 1200, 1800)

# Every divisor of PHASE_PERIOD that is long enough to be a patrol rather than
# a blur. A speed setting picks the nearest of *these* to the period it wants,
# which is what keeps obstacle position a function of the phase alone at every
# speed rather than only at the two that happened to divide evenly.
#
# THIS WAS A REAL BUG AND A SUBTLE ONE
# The period used to be `int(base / speed_scale)`, which produced 171, 320 and
# 685 at the ordinary speed settings — none of them divisors of 480. The phase
# then wrapped mid-patrol and every obstacle in the room jumped to a different
# place, so the same six state numbers meant "clear" on one lap and "occupied"
# on the next. The room stopped being Markov, and it did so only at some
# settings of a slider, which is the hardest kind of bug to see.
PHASE_DIVISORS = tuple(period for period in range(1, PHASE_PERIOD + 1)
                       if PHASE_PERIOD % period == 0 and period >= 60)

# Half the agent, and half a dynamic obstacle, in metres. The assignment fixes
# obstacle width at about 0.5 m, so the radius is 0.25 and the collision test
# below is circle-to-circle against exactly that.
AGENT_RADIUS = 0.22
OBSTACLE_WIDTH = 0.5
OBSTACLE_RADIUS = OBSTACLE_WIDTH / 2.0

# How long a laser's on/off rhythm lasts, in ticks. Divisors of PHASE_PERIOD,
# so a beam's state stays a function of the phase alone.
LASER_CYCLES = (180, 240, 300)

# How long a warning lamp's blink lasts. Decoration, and a divisor for the
# same reason: a lamp that jumped at the wrap would be the one thing on screen
# contradicting the claim that this room repeats exactly.
BEACON_CYCLE = 240
LASER_HALF_WIDTH = 0.06

# Mission stages.
STAGE_TERMINAL = 0
STAGE_EXIT = 1


class InvalidLayout(Exception):
    """A generated warehouse that failed validation and must be redrawn."""


# ----------------------------------------------------------------------
# Procedural generation
# ----------------------------------------------------------------------

def _rectangles_overlap(one, two):
    return not (one["x"] + one["width"] / 2 <= two["x"] - two["width"] / 2
                or two["x"] + two["width"] / 2 <= one["x"] - one["width"] / 2
                or one["y"] + one["height"] / 2 <= two["y"] - two["height"] / 2
                or two["y"] + two["height"] / 2 <= one["y"] - one["height"] / 2)


def _point_clear_of_rectangle(point, rectangle, margin):
    """True when a disc of `margin` at `point` misses the rectangle."""
    nearest_x = max(rectangle["x"] - rectangle["width"] / 2,
                    min(point[0], rectangle["x"] + rectangle["width"] / 2))
    nearest_y = max(rectangle["y"] - rectangle["height"] / 2,
                    min(point[1], rectangle["y"] + rectangle["height"] / 2))
    return math.hypot(point[0] - nearest_x, point[1] - nearest_y) > margin


def _ray_hits_segment(ox, oy, dx, dy, one, two):
    """Where a ray meets a segment, or None. Standard 2-D cross-product form."""
    ex, ey = two[0] - one[0], two[1] - one[1]
    denominator = dx * ey - dy * ex
    if abs(denominator) < 1e-12:
        return None
    fx, fy = one[0] - ox, one[1] - oy
    along = (fx * ey - fy * ex) / denominator      # distance along the ray
    across = (fx * dy - fy * dx) / denominator     # 0..1 along the segment
    if along < 0.0 or not (0.0 <= across <= 1.0):
        return None
    return along


def _distance_to_segment(point, one, two):
    """How far `point` is from the segment `one`-`two`."""
    dx, dy = two[0] - one[0], two[1] - one[1]
    length_squared = dx * dx + dy * dy
    if length_squared < 1e-12:
        return math.hypot(point[0] - one[0], point[1] - one[1])
    share = ((point[0] - one[0]) * dx + (point[1] - one[1]) * dy) / length_squared
    share = max(0.0, min(1.0, share))
    return math.hypot(point[0] - (one[0] + dx * share),
                      point[1] - (one[1] + dy * share))


def _segments_clear(one, two, other_one, other_two, margin):
    """True when two straight runs stay at least `margin` apart.

    Sampled along the first run, like `_segment_clear_of_rectangle` and for the
    same reason: generation happens once per layout and is cached, so an exact
    segment-to-segment solve would buy nothing but a harder line to read.
    """
    length = math.hypot(two[0] - one[0], two[1] - one[1])
    steps = max(2, int(length / (margin / 3.0))) if margin > 0 else 2
    for index in range(steps + 1):
        share = index / float(steps)
        point = (one[0] + (two[0] - one[0]) * share,
                 one[1] + (two[1] - one[1]) * share)
        if _distance_to_segment(point, other_one, other_two) <= margin:
            return False
    return True


def _segment_clear_of_rectangle(one, two, rectangle, margin):
    """True when the straight run from `one` to `two` misses the rectangle.

    Sampled along the segment rather than solved: the rectangle is inflated by
    `margin` and the run is checked every third of a margin, which cannot step
    over anything as wide as the margin itself. Generation runs once per layout
    and is cached, so the cost of sampling here is paid forty times in total.
    """
    length = math.hypot(two[0] - one[0], two[1] - one[1])
    if length < 1e-9:
        return _point_clear_of_rectangle(one, rectangle, margin)
    steps = max(2, int(length / (margin / 3.0)))
    for index in range(steps + 1):
        share = index / float(steps)
        point = (one[0] + (two[0] - one[0]) * share,
                 one[1] + (two[1] - one[1]) * share)
        if not _point_clear_of_rectangle(point, rectangle, margin):
            return False
    return True


def generate_layout(seed, settings):
    """One warehouse, from one seed. Reproducible, and validated.

    The same seed must always produce the same warehouse — the replay, the
    train/test split and every experiment depend on it — so everything random
    here comes from a generator seeded with nothing but `seed` and the
    settings, and never from the global one.

    Raises `InvalidLayout` if it cannot draw a valid one, which the caller
    answers by trying the next sub-seed. Validation is in
    `_validate_layout`; the path it finds is thrown away and never reaches the
    agent.
    """
    width = settings["width"]
    height = settings["height"]

    for attempt in range(24):
        rng = random.Random((seed * 7919) + attempt)
        layout = _draw_layout(rng, seed, settings)
        try:
            _validate_layout(layout, settings)
        except InvalidLayout:
            continue
        layout["attempts"] = attempt + 1
        return layout

    raise InvalidLayout(
        "no valid warehouse for seed %r after 24 attempts; the settings are "
        "probably too crowded (obstacles=%r)" % (seed, settings["obstacles"]))


# The floor, cut into nine regions. An objective is placed inside one of them,
# and no two objectives share a region — which is what guarantees they are
# spread across the building rather than clustered in one corner of it.
_REGIONS = tuple((column, row) for row in range(3) for column in range(3))

# How far apart the objectives have to be, in metres. The mission has to be a
# journey: without this, three regions can still put the terminal a metre from
# the start if both land on the shared edge.
_MIN_SEPARATION = 3.2


def _place_objectives(rng, width, height):
    """Start, terminal and exit — anywhere valid, and well apart.

    Rejection sampling over the nine regions. It converges in a handful of
    attempts and the fallback is a wide spread rather than an exception, because
    a layout that cannot place its objectives is better redrawn by the caller
    than raised from here.
    """
    margin = 1.05
    for _ in range(80):
        regions = rng.sample(_REGIONS, 3)
        points = []
        for column, row in regions:
            low_x = margin + (width - 2 * margin) * column / 3.0
            low_y = margin + (height - 2 * margin) * row / 3.0
            points.append((low_x + rng.uniform(0, (width - 2 * margin) / 3.0),
                           low_y + rng.uniform(0, (height - 2 * margin) / 3.0)))

        one, two, three = points
        if (math.hypot(one[0] - two[0], one[1] - two[1]) >= _MIN_SEPARATION
                and math.hypot(two[0] - three[0], two[1] - three[1]) >= _MIN_SEPARATION
                and math.hypot(one[0] - three[0], one[1] - three[1]) >= 2.6):
            return one, two, three

    # Opposite corners, chosen at random, as a last resort.
    corners = rng.sample([(margin, margin), (width - margin, margin),
                          (margin, height - margin),
                          (width - margin, height - margin)], 3)
    return corners[0], (width / 2, height / 2), corners[2]


def _draw_layout(rng, seed, settings):
    """One candidate warehouse. May be invalid; the caller checks."""
    width = settings["width"]
    height = settings["height"]

    # The three objectives, placed anywhere in the building.
    #
    # THIS IS THE ROOM'S WHOLE POINT AND THE FIRST VERSION GOT IT WRONG
    # It pinned the start to the bottom-left corner, the terminal to the middle
    # and the exit to the top-right, each with about 1.3 m of jitter. Measured
    # over the forty training layouts, the bearing from start to terminal
    # spanned 44 degrees and the bearing from terminal to exit 48 — every
    # episode was the *same diagonal* across the same room. The shelves moved
    # and the traffic moved, but the navigation problem did not, and an agent
    # can memorise "head up and right" without generalising to anything.
    #
    # Now the three are drawn from nine regions of the floor, a different
    # region each, with a minimum separation between them. The mission runs a
    # different way across the building every episode: sometimes right to left,
    # sometimes bottom to top, sometimes a dog-leg back on itself.
    start, terminal, exit_position = _place_objectives(rng, width, height)

    # Storage shelves: axis-aligned blocks, laid down one at a time and only
    # kept if they clear everything already there. Shelves are what make one
    # warehouse a different navigation problem from another.
    shelves = []
    wanted = settings["shelves"]
    for _ in range(wanted * 6):
        if len(shelves) >= wanted:
            break
        upright = rng.random() < 0.5
        long_side = rng.uniform(1.8, 3.2)
        candidate = {
            "x": rng.uniform(1.6, width - 1.6),
            "y": rng.uniform(1.6, height - 1.6),
            "width": 0.62 if upright else long_side,
            "height": long_side if upright else 0.62,
        }
        # Clear of the three fixed points by more than the agent's radius, and
        # not touching another shelf.
        clearance = AGENT_RADIUS + 0.55
        if not all(_point_clear_of_rectangle(point, candidate, clearance)
                   for point in (start, terminal, exit_position)):
            continue

        # And clear of the two straight runs between consecutive objectives.
        #
        # WHY THIS RULE EXISTS, MEASURED
        # Without it, a shelf lands across the direct line about half the time,
        # and a reactive policy cannot get round one: skirting a 3 m shelf means
        # moving *away* from the objective for twenty decisions, and the progress
        # shaping charges for every one of them. It is a potential-well local
        # minimum, and it is a real limitation of a linear policy over local
        # features rather than a bug. Measured: with shelves allowed on the line,
        # a single fixed layout trained to 0% escape; with one shelf and none on
        # the line, to 100%.
        #
        # So the straight runs stay flyable and the shelves shape everything
        # else. They are not decoration — the agent still meets them whenever it
        # drifts, overshoots, or dodges a cart, which is most episodes, and they
        # still differ in every layout. What this rule buys is that the room is
        # solvable by the class of policy the assignment requires.
        if not all(_segment_clear_of_rectangle(one, two, candidate,
                                               AGENT_RADIUS + 0.30)
                   for one, two in ((start, terminal), (terminal, exit_position))):
            continue
        padded = dict(candidate, width=candidate["width"] + 0.7,
                      height=candidate["height"] + 0.7)
        if any(_rectangles_overlap(padded, other) for other in shelves):
            continue
        shelves.append(candidate)

    # The moving traffic.
    #
    # HOW MANY, AND WHY IT IS NOT SIMPLY THE SLIDER
    # The assignment requires the *quantity* of obstacles to be dynamic, not
    # only their positions, and the first version drew exactly `obstacles` of
    # them in every warehouse — so the count was a setting rather than a
    # property of the layout, and two consecutive episodes always met the same
    # number of drones. The slider now names the middle of a range and each
    # layout draws its own count from the seed, so an agent cannot rely on
    # there being a fixed amount of traffic any more than on where it is.
    #
    # `obstacle_variation` is the half-width of that range, and 0 turns the
    # variation off — which is what makes "quantity varies when configured to
    # vary" something that can be tested both ways.
    nominal = settings["obstacles"]
    spread = settings["obstacle_variation"]
    low = max(0, nominal - spread)
    high = max(low, nominal + spread)
    wanted_movers = rng.randint(low, high) if high > low else nominal

    movers = []
    for index in range(wanted_movers):
        mover = _draw_mover(rng, index, settings, shelves,
                            (start, terminal, exit_position))
        if mover is not None:
            movers.append(mover)

    lasers = _draw_lasers(rng, settings, shelves,
                          (start, terminal, exit_position))

    # The machinery that makes the place a working warehouse rather than a room
    # with two enemies in it. NONE of it is physics: no conveyor, crate or
    # beacon appears in any collision test or in the observation. They are
    # generated with the layout so they differ between warehouses like
    # everything else, and they move off the same phase, so they are
    # deterministic and they replay exactly.
    #
    # Kept decorative on purpose. The hazard budget is deliberately small — two
    # security drones and two beams — because the room has to stay readable, and
    # anything else that could kill would spend that budget without adding to
    # the timing puzzle.
    conveyors = []
    edges = rng.sample(("top", "bottom", "left", "right"), 2)
    for index, edge in enumerate(edges):
        horizontal = edge in ("top", "bottom")
        if horizontal:
            near = rng.uniform(0.9, 1.7)
            lane = {"x": width / 2,
                    "y": near if edge == "top" else height - near,
                    "width": width - 1.2, "height": 0.34}
        else:
            near = rng.uniform(0.9, 1.7)
            lane = {"x": near if edge == "left" else width - near,
                    "y": height / 2, "width": 0.34, "height": height - 1.2}
        lane.update(id="conveyor%d" % index, horizontal=horizontal,
                    period=rng.choice(PATROL_PERIODS),
                    crates=rng.randint(2, 3),
                    offset=rng.randrange(PHASE_PERIOD))
        conveyors.append(lane)

    corners = rng.sample([(0.55, 0.55), (width - 0.55, 0.55),
                          (0.55, height - 0.55),
                          (width - 0.55, height - 0.55)], 2)
    beacons = [{"id": "beacon%d" % index, "x": point[0], "y": point[1],
                "cycle": BEACON_CYCLE, "offset": index * (BEACON_CYCLE // 2)}
               for index, point in enumerate(corners)]

    return {
        "seed": seed,
        "start": start,
        "terminal": terminal,
        "exit": exit_position,
        "shelves": shelves,
        "movers": movers,
        # How many this layout meant to have, as against how many could be
        # placed. `_validate_layout` compares the two rather than comparing
        # against the slider, which is no longer the per-layout count.
        "movers_wanted": wanted_movers,
        "lasers": lasers,
        "conveyors": conveyors,
        "beacons": beacons,
        "width": width,
        "height": height,
    }


#: How long a patrol's side may be, in metres.
PATROL_SIDE = (1.8, 3.0)


def _patrol_options(speed, dt):
    """The (period, side) pairs that give exactly `speed` on a legal period.

    THE OBVIOUS WAY ROUND IS THE WRONG WAY ROUND
    The first version drew a side length first and then snapped the period to
    the nearest divisor of PHASE_PERIOD. The period has to divide it — that is
    what keeps obstacle position a function of the phase — but the divisors are
    sparse at the long end (… 900, 1200, 1800, 3600), so at slow settings the
    snap was brutal: asked for 0.2 m/s the drones actually flew at 0.30, half
    again as fast as the slider said, and nothing on screen admitted it.

    Choosing the period *first* and then sizing the square to suit inverts the
    problem and removes it. Every option below travels at exactly the requested
    speed; the only thing quantised is how big the circuit is, which nobody is
    reading off a slider. A speed with no option at all falls back to the
    nearest achievable, and the mover carries the figure it really flies at.
    """
    low, high = PATROL_SIDE
    options = []
    for period in PHASE_DIVISORS:
        side = speed * period * dt / 4.0
        if low <= side <= high:
            options.append((period, side))
    if options:
        return options
    # Nothing fits: the speed is outside what any legal period can express at
    # a sensible circuit size. Take the period whose best side is least wrong
    # and clamp — the mover's own `speed` field then reports the truth.
    def error(period):
        side = speed * period * dt / 4.0
        return abs(side - min(high, max(low, side)))
    period = min(PHASE_DIVISORS, key=error)
    return [(period, min(high, max(low, speed * period * dt / 4.0)))]


def _draw_mover(rng, index, settings, shelves, fixed):
    """One security drone, on a closed square patrol.

    Square only, and few of them. The first version drew from three patterns —
    horizontal, vertical and a rectangle — with as many as six on the floor, and
    the result read as random traffic rather than as something a player could
    time. A square circuit is the one shape whose whole route is obvious from
    watching a corner of it.

    `index` puts the patrols in different bands of the room: even-numbered
    drones keep to the upper half and odd-numbered ones to the lower, so their
    circuits do not overlap and each is a separate piece of timing to solve.

    THE PATROL IS A LOOP, AND IT USED NOT TO BE
    `_mover_at` walked `path + path[-2:0:-1]`, which is out-and-back —
    A→B→C→D→C→B — and then wrapped from B straight to A. Measured on seed 1000,
    that wrap moved the drone 2.23 m in one 0.02 s tick: 111 m/s, once every
    lap, through whatever happened to be in the way. It is the single largest
    reason the trained agent was colliding in 92% of episodes, because a hazard
    that teleports cannot be avoided by any policy, learned or otherwise. The
    circuit is now closed — A→B→C→D→A — and every leg is walked.
    """
    width = settings["width"]
    height = settings["height"]
    speed = settings["obstacle_speed"]
    dt = settings["dt"]

    # Which band of the room this drone patrols in.
    if index % 2 == 0:
        low, high = 1.3, height / 2 - 0.2
    else:
        low, high = height / 2 + 0.2, height - 1.3

    # Which circuit sizes travel at exactly the requested speed on a period
    # that divides the warehouse phase. One is picked per attempt, so the
    # squares still vary between warehouses.
    options = _patrol_options(speed, dt)

    for _ in range(40):
        period, side = rng.choice(options)
        left = rng.uniform(1.3, width - 1.3 - side)
        top = rng.uniform(low, max(low, high - side))
        path = [(left, top), (left + side, top),
                (left + side, top + side), (left, top + side)]
        if any(point[1] > height - 1.0 or point[1] < 0.8 for point in path):
            continue

        if any(not _point_clear_of_rectangle(point, shelf, OBSTACLE_RADIUS + 0.1)
               for point in path for shelf in shelves):
            continue
        if any(math.hypot(point[0] - target[0], point[1] - target[1])
               < OBSTACLE_RADIUS + AGENT_RADIUS + 0.7
               for point in path for target in fixed):
            continue

        return {
            "id": "drone%d" % index,
            "pattern": "square",
            "path": path,
            "side": side,
            "period": period,
            # What it really travels at. Equal to the setting for any speed a
            # legal period can express, and honest about it when it is not.
            "speed": side * 4.0 / (period * dt),
            "offset": rng.randrange(PHASE_PERIOD),
        }
    return None


def _draw_lasers(rng, settings, shelves, fixed):
    """Two diagonal beams that blink on and off, as in room 1.

    A callback to the Laser Security Chamber, and the one hazard in this room
    whose timing is a *rhythm* rather than a position: each beam is lit for part
    of a cycle and dark for the rest, and both the cycle and the phase come out
    of the state, so it stays deterministic and stays Markov.

    Diagonal on purpose. Every other edge in the warehouse is axis-aligned, so
    a diagonal reads instantly as something that does not belong to the
    architecture.
    """
    width = settings["width"]
    height = settings["height"]
    lasers = []

    for index in range(settings.get("lasers", 2)):
        for _ in range(40):
            cycle = rng.choice(LASER_CYCLES)
            # A diagonal chord across one quadrant, well inside the walls.
            span = rng.uniform(2.2, 3.4)
            x = rng.uniform(1.2, width - 1.2 - span)
            y = rng.uniform(1.2, height - 1.2 - span)
            downhill = rng.random() < 0.5
            one = (x, y + span) if downhill else (x, y)
            two = (x + span, y) if downhill else (x + span, y + span)

            # Never lit across an objective, or the mission would be a lottery.
            if any(_distance_to_segment(target, one, two)
                   < AGENT_RADIUS + LASER_HALF_WIDTH + 0.55 for target in fixed):
                continue

            # And never across the direct run between two objectives — the
            # same rule the shelves obey, for the same measured reason.
            #
            # The beams used to be exempt from it because they blinked: a gap
            # came round every few seconds, so a beam on the route was a wait
            # rather than a wall. Now that they are lit throughout (see the
            # note in the entry below) a beam across the route is a wall, and
            # a wall on the direct line is the potential-well local minimum
            # that a reactive linear policy cannot escape — skirting it means
            # moving away from the objective for twenty decisions with the
            # progress shaping charging for every one.
            #
            # Measured: with the beams made permanent and this rule absent,
            # train escape fell from 27% to 9% and beams alone accounted for a
            # third of all episodes. The rule is what makes "always on" an
            # improvement rather than a harder room.
            start, terminal, exit_position = fixed
            if not all(_segments_clear(route_one, route_two, one, two,
                                       AGENT_RADIUS + LASER_HALF_WIDTH + 0.30)
                       for route_one, route_two in ((start, terminal),
                                                    (terminal, exit_position))):
                continue
            if any(not _point_clear_of_rectangle(one, shelf, 0.2)
                   or not _point_clear_of_rectangle(two, shelf, 0.2)
                   for shelf in shelves):
                continue

            lasers.append({
                "id": "laser%d" % index,
                "from": one,
                "to": two,
                "cycle": cycle,
                # LIT THROUGHOUT, AND THIS USED TO BE 0.42
                #
                # The beams blinked: lit for a bit under half of each cycle,
                # dark for the rest, on the reasoning that timing a gap was a
                # puzzle worth having. It is not a puzzle the agent can solve,
                # and measuring the trained policy is what showed it: beams
                # killed 32% of evaluation episodes, more than the drones and
                # the shelves together.
                #
                # The reason is that a beam's state is a function of `phase`,
                # and `phase` is deliberately NOT in the observation — the
                # agent is given fourteen local readings and no clock. So a dark
                # beam is invisible to the sensors, indistinguishable from open
                # floor, and can light up while the drone is halfway across it.
                # No policy over this observation can do better than chance
                # there, and a hazard that kills at random is not difficulty,
                # it is noise: it teaches the value function that perfectly
                # good routes are fatal.
                #
                # Lit throughout, the beams are ordinary solid geometry. The
                # rays see them, the agent can learn to go round, and they
                # still go dark the moment the terminal is reached — which is
                # what disarming the security system means and is now the only
                # thing that changes them. The blink machinery is left in place
                # (`cycle`, `duty`, `laser_states`) so it can be turned back on
                # if the phase is ever added to the observation.
                "duty": 1.0,
                "offset": rng.randrange(cycle),
            })
            break
    return lasers


def _validate_layout(layout, settings):
    """Refuse a warehouse that cannot be flown, before anything trains on it.

    Continuous space cannot be searched exhaustively, so the check is an
    approximation and says so: the room is cut into a grid of `cell` metres,
    every shelf is inflated by the agent's radius, and breadth-first search
    looks for a route start -> terminal and terminal -> exit. A route the grid
    can find is one the continuous agent can fly; a route it cannot find might
    still exist through a gap narrower than a cell, which is why the cell is
    small relative to the gaps the generator leaves.

    The path found here is used for this test and for the efficiency metric in
    analysis, and is never handed to the learner.
    """
    from collections import deque

    width, height = layout["width"], layout["height"]
    for name in ("start", "terminal", "exit"):
        point = layout[name]
        if not (0 < point[0] < width and 0 < point[1] < height):
            raise InvalidLayout("%s is outside the warehouse" % name)

    if len(layout["shelves"]) < max(1, settings["shelves"] - 2):
        raise InvalidLayout("could not place enough shelves")
    # Against what this layout drew, not against the slider: the count is a
    # property of the layout now. A warehouse that wanted three drones and
    # could only fit two is a failed draw and is redrawn.
    if len(layout["movers"]) < layout["movers_wanted"]:
        raise InvalidLayout("could not place every dynamic obstacle")

    # Every patrol has to stay inside the building and off the static geometry
    # for its whole lap, not merely at its corners. A corner-only check passes
    # a square whose edge clips the end of a shelf.
    for mover in layout["movers"]:
        circuit = mover["path"] + [mover["path"][0]]
        for one, two in zip(circuit, circuit[1:]):
            for point in (one, two):
                if not (OBSTACLE_RADIUS <= point[0] <= width - OBSTACLE_RADIUS
                        and OBSTACLE_RADIUS <= point[1] <= height - OBSTACLE_RADIUS):
                    raise InvalidLayout("a patrol leaves the warehouse")
            for shelf in layout["shelves"]:
                if not _segment_clear_of_rectangle(one, two, shelf,
                                                   OBSTACLE_RADIUS):
                    raise InvalidLayout("a patrol crosses a shelf")

    cell = 0.25
    columns = int(width / cell)
    rows = int(height / cell)

    def blocked(column, row):
        x = (column + 0.5) * cell
        y = (row + 0.5) * cell
        for shelf in layout["shelves"]:
            if not _point_clear_of_rectangle((x, y), shelf, AGENT_RADIUS):
                return True
        return False

    grid = [[blocked(column, row) for row in range(rows)]
            for column in range(columns)]

    def cell_of(point):
        return (min(columns - 1, max(0, int(point[0] / cell))),
                min(rows - 1, max(0, int(point[1] / cell))))

    def reachable(source, target):
        origin, goal = cell_of(source), cell_of(target)
        if grid[origin[0]][origin[1]] or grid[goal[0]][goal[1]]:
            return None
        queue = deque([(origin, 0)])
        seen = {origin}
        while queue:
            (column, row), distance = queue.popleft()
            if (column, row) == goal:
                return distance * cell
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                step = (column + dc, row + dr)
                if not (0 <= step[0] < columns and 0 <= step[1] < rows):
                    continue
                if step in seen or grid[step[0]][step[1]]:
                    continue
                seen.add(step)
                queue.append((step, distance + 1))
        return None

    first = reachable(layout["start"], layout["terminal"])
    if first is None:
        raise InvalidLayout("no route from the start to the terminal")
    second = reachable(layout["terminal"], layout["exit"])
    if second is None:
        raise InvalidLayout("no route from the terminal to the exit")

    # Kept for the path-efficiency metric in analysis. Not an input anywhere.
    layout["reference_path"] = first + second
    return layout


# ----------------------------------------------------------------------
# The world
# ----------------------------------------------------------------------

class WarehouseWorld:
    """A continuous warehouse whose layout changes between episodes."""

    is_grid = False

    # This room's furniture is not static for the session's lifetime: every
    # episode is a different warehouse. Anything that caches the entity list
    # from the opening definition will draw the wrong building — see
    # `Session._scene`, which is where the current one is sent from.
    layout_varies = True

    def __init__(self, room, slip=0.0, seed=0, rewards=None, collapse=0.0):
        self.room = room
        self.seed = seed
        self.rng = random.Random(seed)

        self.width = float(room["size"][0])
        self.height = float(room["size"][1])
        self.dt = float(room["dt"])
        self.thrust = float(room["thrust"])
        self.drag = float(room["drag"])
        self.speed_limit = float(room["speed_limit"])

        self.sensor_range = float(room.get("sensor_range",
                                           room["sensor_range_default"]))
        self.sensor_spread = float(room.get("sensor_spread", 0.55))
        self.reach = float(room["reach"])
        # How many ticks of physics one decision lasts. See `step`.
        self.action_repeat = max(1, int(room.get("action_repeat", 1)))
        # How many decisions an episode may last. The world truncates itself
        # rather than leaving it to the session, because running out of time
        # carries a penalty and only the thing that applies rewards can apply
        # that one. The session's own limit is a backstop above this.
        self.max_steps = int(room.get("max_steps", 300))
        # What share of training episodes begin at the terminal in stage 1.
        # See `reset`.
        self.stage_one_share = float(room.get("stage_one_share", 0.0))

        self.rewards = dict(room["rewards"])
        if rewards:
            self.rewards.update(rewards)

        self.settings = {
            "width": self.width,
            "height": self.height,
            "dt": self.dt,
            "shelves": int(room.get("shelves", 4)),
            "obstacles": int(room.get("obstacles", room["obstacles_default"])),
            # In metres per second, and a real speed rather than a multiplier.
            # See `_period_for` for how it becomes a patrol period.
            "obstacle_speed": float(room.get("obstacle_speed",
                                             room["obstacle_speed_default"])),
            "obstacle_variation": int(room.get("obstacle_variation", 1)),
            "lasers": int(room.get("lasers", 2)),
        }

        # The three pools of layouts, and the rule that keeps them honest. A
        # test layout the agent trained on is not an unseen layout, so the
        # pools are disjoint by construction and checked here rather than
        # trusted.
        # Each pool starts at its own base seed and runs for as many layouts as
        # the user asked for. The bases are far enough apart (1000 / 2000 /
        # 3000) that the pools cannot meet at any legal size, and the check
        # below proves it rather than trusting the arithmetic.
        def pool(name, base_key):
            base = room[base_key][0]
            count = int(room.get("%s_layouts" % name,
                                 room[base_key][1] - base))
            return list(range(base, base + max(1, count)))

        self.pools = {
            "train": pool("train", "train_seeds"),
            "validation": pool("validation", "validation_seeds"),
            "test": pool("test", "test_seeds"),
        }
        for one, two in (("train", "validation"), ("train", "test"),
                         ("validation", "test")):
            shared = set(self.pools[one]) & set(self.pools[two])
            if shared:
                raise ValueError(
                    "the %s and %s layout pools share %d seeds; a test layout "
                    "that was trained on is not an unseen layout"
                    % (one, two, len(shared)))

        # Everything that moves off the phase has to repeat with it, or the
        # same six state numbers would mean different things on different laps.
        for period in PATROL_PERIODS + LASER_CYCLES + (BEACON_CYCLE,):
            if PHASE_PERIOD % period:
                raise ValueError(
                    "cycle %d does not divide the warehouse period %d, so "
                    "obstacle positions would stop being a function of the "
                    "phase and the room would not be Markov"
                    % (period, PHASE_PERIOD))
        if not PHASE_DIVISORS:
            raise ValueError("no legal patrol periods for phase %d"
                             % PHASE_PERIOD)

        self._layouts = {}
        self.split = "train"
        self.layout = self.layout_for(self.pools["train"][0])
        self.state = self.start_state()
        self.steps = 0
        self._scored_terminal = False

    # ------------------------------------------------------------------
    # Layouts
    # ------------------------------------------------------------------

    def layout_for(self, seed):
        """One layout, generated once and remembered.

        Cached because generation runs a breadth-first search and an episode
        asks for its layout on every reset; regenerating per frame is the
        performance trap this exists to avoid.
        """
        if seed not in self._layouts:
            self._layouts[seed] = generate_layout(seed, self.settings)
        return self._layouts[seed]

    def choose_layout(self, split=None, index=None):
        """Pick the next episode's warehouse out of a pool."""
        split = split or self.split
        pool = self.pools[split]
        seed = pool[index % len(pool)] if index is not None \
            else self.rng.choice(pool)
        self.split = split
        self.layout = self.layout_for(seed)
        # Shelf edges as plain tuples, so the ray test is arithmetic on floats
        # rather than four dictionary lookups per shelf per ray.
        self._shelf_bounds = tuple(
            (shelf["x"] - shelf["width"] / 2, shelf["x"] + shelf["width"] / 2,
             shelf["y"] - shelf["height"] / 2, shelf["y"] + shelf["height"] / 2)
            for shelf in self.layout["shelves"])
        # Both caches belong to a layout and must go with it.
        self._movers_at = {}
        self._observations = {}
        return self.layout

    # ------------------------------------------------------------------
    # The state
    # ------------------------------------------------------------------

    def start_state(self):
        start = self.layout["start"]
        return (start[0], start[1], 0.0, 0.0, STAGE_TERMINAL, 0)

    def actions(self):
        return ACTIONS

    def all_states(self):
        raise NotImplementedError(
            "room 5's state is continuous and its layout changes between "
            "episodes: there is no enumerable state space, and no single map "
            "to plan over")

    def transitions(self, state, action):
        raise NotImplementedError(
            "room 5 hands out no model: the agent is given what its sensors "
            "reach and nothing else")

    def world_position(self, state):
        # Rounded for the same reason the obstacle positions are: this pair is
        # written into every recorded frame. A tenth of a millimetre is well
        # inside what the canvas can draw and what the collision tests care
        # about — the drone's radius is 0.22 m.
        return {"x": round(state[0], 4), "y": round(state[1], 4)}

    def world_velocity(self, state):
        return {"x": round(state[2], 4), "y": round(state[3], 4)}

    def is_terminal(self, state):
        return (self._reached_exit(state) or self._hit_anything(state)
                or self._outside(state))

    # ------------------------------------------------------------------
    # Where the traffic is, as a function of the phase alone
    # ------------------------------------------------------------------

    def mover_positions(self, phase):
        """Every dynamic obstacle's position and velocity at one phase.

        A pure function of `phase`, which is why it is safe for the recorder,
        the renderer and the replay to call it: they all get the same answer
        for the same step, and nothing here advances anything.

        Being a pure function of one bounded integer is also what makes it
        cacheable — there are only `PHASE_PERIOD` possible answers, and one
        observation asks for the same one four times over.
        """
        found = self._movers_at.get(phase)
        if found is None:
            found = tuple(self._mover_at(mover, phase)
                          for mover in self.layout["movers"])
            self._movers_at[phase] = found
        return found

    def _mover_at(self, mover, phase):
        path = mover["path"]
        period = mover["period"]
        # A closed circuit: back to where it started, so the lap joins up and
        # the wrap from the last leg to the first is a step like any other.
        #
        #     A → B
        #     ↑   ↓
        #     D ← C
        #
        # See the note in `_draw_mover` for what walking it out-and-back did.
        circuit = path + [path[0]]
        legs = len(circuit) - 1
        through = ((phase + mover["offset"]) % period) / float(period)
        along = through * legs
        leg = min(legs - 1, int(along))
        share = along - leg
        one, two = circuit[leg], circuit[leg + 1]
        x = one[0] + (two[0] - one[0]) * share
        y = one[1] + (two[1] - one[1]) * share
        # The velocity of the leg it is on, in metres per second.
        length = math.hypot(two[0] - one[0], two[1] - one[1])
        seconds = period * self.dt / legs
        speed = length / seconds if seconds else 0.0
        heading = math.atan2(two[1] - one[1], two[0] - one[0])
        return {
            "id": mover["id"],
            "x": x, "y": y,
            "vx": math.cos(heading) * speed,
            "vy": math.sin(heading) * speed,
        }

    def conveyor_crates(self, phase):
        """Where the cargo on each conveyor is, at one phase.

        Decoration, and a pure function of the phase like everything else that
        moves here — so it animates live, animates in a replay, and appears in
        no collision test anywhere. Crates run the length of the lane and wrap
        round to the start, which is what a loop of belt does.
        """
        found = []
        for lane in self.layout.get("conveyors", []):
            span = (lane["width"] if lane["horizontal"] else lane["height"])
            start = (lane["x"] - lane["width"] / 2 if lane["horizontal"]
                     else lane["y"] - lane["height"] / 2)
            through = ((phase + lane["offset"]) % lane["period"]) / lane["period"]
            for index in range(lane["crates"]):
                share = (through + index / float(lane["crates"])) % 1.0
                along = start + span * share
                found.append({
                    "id": "%s-crate%d" % (lane["id"], index),
                    "x": along if lane["horizontal"] else lane["x"],
                    "y": lane["y"] if lane["horizontal"] else along,
                })
        return found

    def beacon_states(self, phase):
        """Which warning lamps are lit. Decoration, from the phase."""
        found = []
        for beacon in self.layout.get("beacons", []):
            through = (phase + beacon["offset"]) % beacon["cycle"]
            found.append({"id": beacon["id"],
                          "on": through < beacon["cycle"] * 0.5})
        return found

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------

    def _outside(self, state):
        x, y = state[0], state[1]
        return (x <= AGENT_RADIUS or x >= self.width - AGENT_RADIUS
                or y <= AGENT_RADIUS or y >= self.height - AGENT_RADIUS)

    def _hit_shelf(self, state):
        for shelf in self.layout["shelves"]:
            if not _point_clear_of_rectangle((state[0], state[1]), shelf,
                                             AGENT_RADIUS):
                return shelf
        return None

    def _hit_mover(self, state):
        """Circle against circle, at the obstacle's real 0.5 m width."""
        touching = AGENT_RADIUS + OBSTACLE_RADIUS
        for mover in self.mover_positions(state[5]):
            if math.hypot(state[0] - mover["x"], state[1] - mover["y"]) <= touching:
                return mover
        return None

    def laser_states(self, phase):
        """Every beam and whether it is lit, at one phase.

        A pure function of the phase, like the drone positions, so the recorder,
        the renderer and the replay all agree and nothing here advances a clock.
        """
        found = []
        for laser in self.layout.get("lasers", []):
            through = (phase + laser["offset"]) % laser["cycle"]
            found.append({
                "id": laser["id"],
                "from": laser["from"],
                "to": laser["to"],
                "on": through < laser["cycle"] * laser["duty"],
            })
        return found

    def _hit_laser(self, state):
        """A lit beam is solid; a dark one is not there at all.

        And once the control terminal has been reached the whole security system
        is down, so no beam is lit at all. That is what disarming it means: the
        reward for stage 1 is not only an unlocked door but a warehouse that has
        stopped trying to kill you. It also makes the mission order matter for a
        second reason beyond the lock.
        """
        if state[4] == STAGE_EXIT:
            return None
        for laser in self.laser_states(state[5]):
            if not laser["on"]:
                continue
            gap = _distance_to_segment((state[0], state[1]),
                                       laser["from"], laser["to"])
            if gap <= AGENT_RADIUS + LASER_HALF_WIDTH:
                return laser
        return None

    def _hit_anything(self, state):
        return (self._hit_shelf(state) is not None
                or self._hit_mover(state) is not None
                or self._hit_laser(state) is not None)

    # ------------------------------------------------------------------
    # The objectives
    # ------------------------------------------------------------------

    def target_of(self, state):
        """Where the agent is trying to get to, which depends on the stage."""
        if state[4] == STAGE_TERMINAL:
            return self.layout["terminal"]
        return self.layout["exit"]

    def _at(self, state, point):
        return math.hypot(state[0] - point[0], state[1] - point[1]) <= self.reach

    def _reached_exit(self, state):
        return state[4] == STAGE_EXIT and self._at(state, self.layout["exit"])

    # ------------------------------------------------------------------
    # The sensors — all the agent knows about the warehouse
    # ------------------------------------------------------------------

    SENSOR_RAYS = ("left", "centre", "right")

    # ------------------------------------------------------------------
    # THE VISIBILITY RULE, STATED ONCE AND OBEYED EVERYWHERE
    #
    # A dynamic obstacle is visible to R-5 when BOTH of these hold:
    #
    #   RANGE    the Euclidean distance from the CENTRE OF THE AGENT to the
    #            CENTRE OF THE OBSTACLE is less than or equal to
    #            `sensor_range` metres.
    #
    #            Centre to centre, exactly. Not edge to edge, and the
    #            obstacle's 0.25 m radius is NOT subtracted from the range
    #            first — an obstacle whose centre is at 3.01 m with the range
    #            set to 3.0 is invisible, even though its near edge is at
    #            2.76 m. Subtracting the radius is the easy mistake here
    #            because a ray naturally solves for the surface, and it would
    #            quietly widen every range setting by a quarter of a metre.
    #
    #   BEARING  the obstacle's centre lies within the FORWARD CONE: the angle
    #            between R-5's facing direction and the bearing to that centre
    #            is at most `sensor_spread` radians.
    #
    #            "Ahead" is therefore a cone, and it is exactly the cone the
    #            three sensor rays span — the outer rays sit on its two edges
    #            at ±`sensor_spread`. That is what lets the drawing be the
    #            sensor rather than an illustration of it.
    #
    # FACING, WHEN THERE IS NO VELOCITY TO READ
    # The cone points the way the drone is travelling. Below 0.05 m/s there is
    # no meaningful direction of travel, so it points at the current objective
    # instead — the direction it is about to travel in. That rule is a pure
    # function of the state, which matters: "the last non-zero heading" would
    # be a hidden variable the observation does not carry, and two visits to
    # the same state would then sense differently.
    # ------------------------------------------------------------------

    def visible_obstacles(self, state):
        """Every dynamic obstacle R-5 can currently see, nearest first.

        The one authority on the rule above. `nearest_mover`, the observation,
        the ray solver and the sensor drawing all go through this or apply the
        identical test, so there is no second definition to drift.
        """
        heading = self._heading(state)
        found = []
        for mover in self.mover_positions(state[5]):
            offset_x = mover["x"] - state[0]
            offset_y = mover["y"] - state[1]
            # Centre to centre. No radius anywhere in this line.
            distance = math.hypot(offset_x, offset_y)
            if distance > self.sensor_range:
                continue
            bearing = math.atan2(offset_y, offset_x)
            # Wrapped into (-pi, pi] before comparing, or a cone straddling
            # due west would reject everything in it.
            difference = (bearing - heading + math.pi) % (2 * math.pi) - math.pi
            if abs(difference) > self.sensor_spread:
                continue
            found.append({"mover": mover, "distance": distance,
                          "bearing": bearing, "offset": difference})
        found.sort(key=lambda entry: entry["distance"])
        return found

    def sees(self, state, mover):
        """Whether one obstacle passes the rule. For tests and the drawing."""
        return any(entry["mover"]["id"] == mover["id"]
                   for entry in self.visible_obstacles(state))

    def sense(self, state):
        """Three forward rays, as normalised free distances in [0, 1].

        0.0 is "something is on top of me" and 1.0 is "nothing within range".
        The rays point where the agent is *travelling*, because a drone with no
        heading of its own has nothing else to aim them by; when it is barely
        moving they default to pointing at the current objective, which is the
        direction it is about to travel in.

        The drawing of these rays in `shapes.js` uses these very numbers, so
        what is on screen is what the agent is given — a wide cone drawn over a
        single line check would be a picture of a sensor the code does not have.
        """
        heading = self._heading(state)
        found = {}
        for name, offset in zip(self.SENSOR_RAYS,
                                (-self.sensor_spread, 0.0, self.sensor_spread)):
            found[name] = self._ray(state, heading + offset)
        return found

    def _heading(self, state):
        if math.hypot(state[2], state[3]) > 0.05:
            return math.atan2(state[3], state[2])
        target = self.target_of(state)
        return math.atan2(target[1] - state[1], target[0] - state[0])

    def _ray(self, state, heading):
        """One ray's free distance, normalised, solved rather than marched.

        Analytic on purpose. The first version of this marched the ray in steps
        of a tenth of the agent's radius and tested every shelf at every
        sample, which came to 3.9 million rectangle tests for 8,000 environment
        steps and made the room six times slower to train than room 4 — slow
        enough that it could not be tuned at all. The slab test below is one
        comparison per shelf and one quadratic per cart, exactly, and it is
        also *more* accurate than sampling: a marched ray can step straight
        over a thin shelf.
        """
        dx, dy = math.cos(heading), math.sin(heading)
        limit = self.sensor_range
        nearest = limit

        # The chamber wall, which is always somewhere along the ray.
        for span, origin, direction in ((self.width, state[0], dx),
                                        (self.height, state[1], dy)):
            if direction > 1e-12:
                nearest = min(nearest, (span - origin) / direction)
            elif direction < -1e-12:
                nearest = min(nearest, -origin / direction)

        # Shelves: the slab method for an axis-aligned box.
        for shelf in self._shelf_bounds:
            left, right, top, bottom = shelf
            # Inside one already, for the same reason as the carts below.
            if left <= state[0] <= right and top <= state[1] <= bottom:
                return 0.0
            if abs(dx) < 1e-12:
                if not (left <= state[0] <= right):
                    continue
                near_x, far_x = -math.inf, math.inf
            else:
                t1 = (left - state[0]) / dx
                t2 = (right - state[0]) / dx
                near_x, far_x = (t1, t2) if t1 <= t2 else (t2, t1)

            if abs(dy) < 1e-12:
                if not (top <= state[1] <= bottom):
                    continue
                near_y, far_y = -math.inf, math.inf
            else:
                t1 = (top - state[1]) / dy
                t2 = (bottom - state[1]) / dy
                near_y, far_y = (t1, t2) if t1 <= t2 else (t2, t1)

            enter = max(near_x, near_y)
            leave = min(far_x, far_y)
            if leave >= max(enter, 0.0) and 0.0 <= enter < nearest:
                nearest = enter

        # Security drones: the smaller positive root of |O + tD - C|² = r².
        for mover in self.mover_positions(state[5]):
            ox = state[0] - mover["x"]
            oy = state[1] - mover["y"]
            # THE RANGE GATE, APPLIED TO THE CENTRE AND NOT TO THE SURFACE
            # The quadratic below solves for where the ray meets the drone's
            # *edge*, which is up to 0.25 m nearer than its centre. Without
            # this line an obstacle centred at 3.2 m would be reported by a
            # 3.0 m sensor as a hit at 2.95 m — the range quietly widened by
            # the obstacle's radius, which is exactly what the rule above
            # forbids. Gate on the centre first, then solve.
            if math.hypot(ox, oy) > self.sensor_range:
                continue
            c = ox * ox + oy * oy - OBSTACLE_RADIUS * OBSTACLE_RADIUS
            # Already inside one. Nothing is free in any direction, and the
            # far root — the way *out* — would read as clear space ahead. Only
            # reachable on a state that has already collided, but a sensor that
            # reports 3 m of clear road from inside a cart is wrong whatever
            # the caller does with it.
            if c <= 0.0:
                return 0.0
            b = ox * dx + oy * dy
            discriminant = b * b - c
            if discriminant < 0:
                continue
            hit = -b - math.sqrt(discriminant)
            if 0.0 <= hit < nearest:
                nearest = hit

        # Lit beams, as segments. A dark beam is not sensed because it is not
        # there — which is the whole timing puzzle: the agent has to learn that
        # the gap is coming rather than that the beam is gone.
        for laser in self.laser_states(state[5]):
            if not laser["on"]:
                continue
            hit = _ray_hits_segment(state[0], state[1], dx, dy,
                                    laser["from"], laser["to"])
            if hit is not None and 0.0 <= hit < nearest:
                nearest = hit

        return max(0.0, min(1.0, nearest / limit))

    def nearest_mover(self, state):
        """The closest *visible* obstacle as (distance, mover), or None.

        Visible by the rule above — inside the range and inside the forward
        cone — and not merely nearby. An agent that could always name its
        nearest obstacle, whatever direction it was in, would have a sense of
        the room the sensors do not give it, and the room would stop being
        about partial observability.

        The distance returned is centre to centre, which is the number the
        observation normalises and the readout prints.
        """
        visible = self.visible_obstacles(state)
        if not visible:
            return None
        return (visible[0]["distance"], visible[0]["mover"])

    # ------------------------------------------------------------------
    # The observation, and the features built from it
    # ------------------------------------------------------------------

    # Which axes are tile-coded together, and how finely. Grouped because a
    # Cartesian grid over fourteen axes cannot be built — see `GroupedTileCoder`.
    # The groupings are the modelling decision: position with position, the two
    # velocity components together, the target offset together, the three rays
    # together, and the nearest obstacle's distance with how fast it is closing.
    # WHY THESE GROUPS, AND NOT THE OBVIOUS ONES
    # The first version grouped by *kind* — position with position, velocity
    # with velocity, the rays together. It was stable at a low discount and
    # never escaped once, and the reason is that the single most useful thing
    # this agent can know is a *conjunction* across two kinds: "the terminal is
    # ahead and to the right, and the ray pointing that way is short". Grouped
    # by kind, that conjunction is not representable — the target direction and
    # the rays only ever add.
    #
    # So the first group is deliberately mixed: where the objective is, crossed
    # with what the middle ray sees. That is the group that has to carry
    # obstacle avoidance, and everything else refines it.
    OBSERVATION_GROUPS = (
        (3, 6),   # 0-2   target dx, target dy, centre ray   <- the important one
        (3, 5),   # 3-5   the three rays together
        (2, 6),   # 6-7   where in the warehouse
        (2, 6),   # 8-9   velocity
        (2, 6),   # 10-11 alignment with the target, distance to it
        (2, 5),   # 12-13 nearest cart: gap and how fast it is closing
    )
    OBSERVATION_SIZE = 14

    # WHAT WAS TRIED HERE AND REVERTED, SO IT IS NOT TRIED TWICE
    # A seventh group carrying the *direction* of the nearest cart — its world
    # bearing, alongside the gap — on the reasoning that `gap` and `closing`
    # say "a cart is near and getting nearer" without saying from where, so a
    # policy given only those can brake but cannot pick a side to dodge to.
    #
    # It half worked, and the half that worked is not the half that mattered:
    #
    #     features        escape   terminal   collide    reward
    #     14 (this one)      10%       45%       75%      +8.4
    #     17 (with bearing)   0%        8%       20%     -91.7
    #
    # Collisions fell by more than half, so the bearing genuinely does buy
    # avoidance. But the policy then avoided *instead of* travelling: -91.7 is
    # essentially the timeout penalty, and terminal activation collapsed. The
    # extra features shifted the balance between "get there" and "stay alive"
    # far enough that staying alive won.
    #
    # Reinstating it needs the reward rebalanced at the same time — a stronger
    # progress term, or a smaller collision penalty — rather than the features
    # alone. Left out until that is done and measured.

    def observation(self, state):
        """The fourteen numbers the agent gets, each already in [0, 1].

        NOT THE STATE. The environment knows every obstacle's position; this
        carries three range-limited rays and one nearest-obstacle reading. Two
        different warehouses can produce identical numbers here, and an
        obstacle behind a shelf is absent from it entirely. The room is
        therefore Markov and the observation is not, which is the honest
        description and the reason a policy learned here has to generalise
        rather than memorise.

        Cached by state. The learner asks for the same state's observation two
        or three times per step — once to choose an action and again to evaluate
        the update — and each one costs three ray solves.
        """
        found = self._observations.get(state)
        if found is None:
            found = self._observe(state)
            # Bounded, so one pathological run cannot grow it without limit.
            # It is cleared with the layout in any case.
            if len(self._observations) < 200000:
                self._observations[state] = found
        return found

    def _observe(self, state):
        x, y, vx, vy, stage, phase = state
        limit = self.speed_limit
        target = self.target_of(state)

        rays = self.sense(state)
        nearest = self.nearest_mover(state)
        if nearest is None:
            # Nothing within range: "far away", "not closing", and no direction
            # to speak of. The neutral 0.5s put it in the middle tile of each
            # axis rather than at an edge, so "no cart" is its own region of the
            # feature space instead of looking like a cart due north.
            gap, closing = 1.0, 0.5
        else:
            distance, mover = nearest
            gap = distance / self.sensor_range
            # How fast it is closing, as a share of the worst case. Positive
            # means the gap is shrinking.
            towards = 0.0
            if distance > 1e-6:
                towards = ((mover["vx"] - vx) * (state[0] - mover["x"])
                           + (mover["vy"] - vy) * (state[1] - mover["y"])) / distance
            closing = max(0.0, min(1.0, 0.5 + towards / (4.0 * limit)))

        # Where the objective is. As a *direction* and a *distance* rather than
        # as an offset in metres: a direction means the same thing in every
        # warehouse and at every range, which is exactly the property a policy
        # has to transfer between layouts. An offset in metres does not — "3.4
        # metres east" is a different feature in a room where the target is
        # near than in one where it is far.
        to_x = target[0] - x
        to_y = target[1] - y
        distance = math.hypot(to_x, to_y)
        span = math.hypot(self.width, self.height)
        if distance > 1e-9:
            bearing_x, bearing_y = to_x / distance, to_y / distance
        else:
            bearing_x, bearing_y = 0.0, 0.0

        # How much of the current speed is going the right way. Negative means
        # it is heading away, which is the one thing a reactive policy most
        # needs to notice about its own motion.
        speed = math.hypot(vx, vy)
        if speed > 1e-9 and distance > 1e-9:
            align = (vx * bearing_x + vy * bearing_y) / limit
        else:
            align = 0.0

        return (
            # Group 0 — the objective's direction, with the ray that points
            # along the way the agent is going. The conjunction that matters.
            0.5 + bearing_x / 2,
            0.5 + bearing_y / 2,
            rays["centre"],
            # Group 1 — the three rays together.
            rays["left"], rays["centre"], rays["right"],
            # Group 2 — where in the warehouse.
            x / self.width,
            y / self.height,
            # Group 3 — velocity.
            (vx + limit) / (2 * limit),
            (vy + limit) / (2 * limit),
            # Group 4 — alignment and how far there is to go.
            max(0.0, min(1.0, 0.5 + align / 2)),
            min(1.0, distance / span),
            # Group 5 — the nearest cart within range: how near, and how fast
            # it is closing. Deliberately without a direction; see the note on
            # OBSERVATION_GROUPS for what happened when one was added.
            gap, closing,
        )

    # Stage 0 and stage 1 are different tasks with different objectives, so
    # they get their own blocks of weights rather than sharing one. The
    # learner reads these two hooks; a room without them has one context.
    context_count = 2

    def context(self, state):
        return int(state[4])

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------

    def reset(self, seed=None, split=None, index=None, evaluating=False):
        """A new episode, in a warehouse drawn from a pool.

        This is where the room rearranges itself. Every reset picks a layout,
        which is what stops one memorised route from being worth anything.

        THE SECOND STAGE STARVES ITSELF, SO TRAINING SOMETIMES STARTS IN IT
        Stage 0 and stage 1 have separate blocks of weights, because they are
        different tasks. But stage 1 is only ever *entered* by finishing stage
        0, so early on it gets almost no experience: measured, the agent reached
        the terminal in 22% of episodes and escaped in 0%, and the second block
        was still nearly untrained after ten thousand episodes.

        So a share of *training* episodes begin at the terminal, already in
        stage 1, with the exit unlocked. This is a curriculum over start
        states and nothing more — the agent is given no path, no layout and no
        hidden information, only a different distribution of situations to
        practise. `evaluating=True` switches it off, so every reported number
        is measured from the real start in stage 0 with the door shut.
        """
        if seed is not None:
            self.rng = random.Random(seed)
        self.choose_layout(split=split, index=index)

        if not evaluating and self.rng.random() < self.stage_one_share:
            terminal = self.layout["terminal"]
            self.state = (terminal[0], terminal[1], 0.0, 0.0, STAGE_EXIT, 0)
            # Already spent: this episode does not collect the activation
            # bonus, because it did not do the activating.
            self._scored_terminal = True
        else:
            self.state = self.start_state()
            self._scored_terminal = False

        self.steps = 0
        return self.state

    def step(self, action):
        """One *decision*, which is `action_repeat` ticks of physics.

        WHY THE AGENT DOES NOT DECIDE EVERY TICK
        The physics runs at dt = 0.02 s because the assignment fixes it there,
        and crossing the warehouse takes some 400 of those. A discount of 0.95
        gives an effective horizon of 1/(1-0.95) = 20 steps — four tenths of a
        second. The control terminal is twenty times further away than the
        agent can see, so the terminal reward is worth 0.95^400 ~ 1e-9 at the
        start and is invisible.

        Measured, before this existed: the greedy policy oscillated in place —
        431 UP against 425 DOWN over 1200 steps, moving half a metre — because
        with a 0.4 s horizon the only thing it could optimise was the immediate
        shaping term, and thrusting into the shaping gradient and back out of it
        looks locally fine.

        Raising the discount instead is the obvious alternative and it does not
        work here: at 0.99 semi-gradient Q-Learning diverges in this room, with
        Q(start) reaching +1200 against a true value near 250. Holding each
        action for several ticks fixes the horizon without touching either the
        discount or the physics: the state still evolves every 0.02 s, and the
        agent chooses every 0.2 s -- ten ticks -- which is a reasonable rate
        for a drone to change its mind and makes the discount a horizon of a
        few hundred physics steps rather than a few dozen decisions.
        """
        total = 0.0
        # Mission events happen on a tick, and a decision is ten of them.
        #
        # WHY THESE HAVE TO BE GATHERED RATHER THAN READ OFF THE LAST TICK
        # This returned the final tick's `info` and nothing else, so an event
        # that did not also end the episode was overwritten by the ticks after
        # it. Traced on seed 1000: the terminal was activated on tick 8 of a
        # ten-tick decision, and `step` reported `event: None`. Terminal
        # activation — the one mission event this room is built around — never
        # reached the recorder, the history or the outcome classifier at all.
        #
        # Ending events are unaffected either way, because the loop breaks on
        # `done` and the last tick is then the one that ended it. That is why
        # the collisions always looked right and this did not.
        mission = []
        for tick in range(self.action_repeat):
            state, reward, done, info = self._tick(action)
            total += reward
            if info["event"] in ("terminal", "locked"):
                mission.append(info["event"])
            if done:
                break

        info["reward"] = total
        # Every mission event of this decision, in order. A decision can carry
        # one and still end in a collision, and both are worth having.
        info["missionEvents"] = mission
        info["activatedThisStep"] = "terminal" in mission
        # An ending event always wins the single `event` slot — it is what the
        # episode's outcome is classified from. A mission event fills it only
        # when nothing ended the decision.
        if info["event"] is None and mission:
            info["event"] = mission[-1]
        return state, total, done, info

    def _tick(self, action):
        x, y, vx, vy, stage, phase = self.state
        target = self.target_of(self.state)
        before = math.hypot(x - target[0], y - target[1])

        push_x, push_y = THRUST[action]
        vx += push_x * self.thrust * self.dt
        vy += push_y * self.thrust * self.dt

        decay = math.exp(-self.drag * self.dt)
        vx *= decay
        vy *= decay

        limit = self.speed_limit
        vx = max(-limit, min(limit, vx))
        vy = max(-limit, min(limit, vy))

        x += vx * self.dt
        y += vy * self.dt
        phase = (phase + 1) % PHASE_PERIOD

        next_state = (x, y, vx, vy, stage, phase)

        reward = self.rewards["step"]
        event = None
        done = False
        success = False

        # ---- the mission ------------------------------------------------
        activated = False
        if stage == STAGE_TERMINAL and self._at(next_state, self.layout["terminal"]):
            # Once, and only once. Paying it every step inside the terminal
            # would make standing on it more profitable than leaving.
            if not self._scored_terminal:
                self._scored_terminal = True
                reward += self.rewards["terminal"]
                activated = True
                event = "terminal"
            stage = STAGE_EXIT
            next_state = (x, y, vx, vy, stage, phase)

        elif stage == STAGE_TERMINAL and self._at(next_state, self.layout["exit"]):
            # The door is shut in stage 0, and trying it costs a little so the
            # agent learns the order rather than discovering it by accident.
            reward += self.rewards["locked"]
            event = "locked"

        # Shaping, measured against whichever objective is current now.
        after_target = self.target_of(next_state)
        after = math.hypot(x - after_target[0], y - after_target[1])
        if not activated:
            reward += (before - after) * self.rewards["progress"]

        # ---- what ends it ----------------------------------------------
        shelf = self._hit_shelf(next_state)
        mover = self._hit_mover(next_state)
        laser = self._hit_laser(next_state)
        if self._outside(next_state):
            reward += self.rewards["boundary"]
            done, event = True, "boundary"
        elif mover is not None:
            reward += self.rewards["dynamic"]
            done, event = True, "dynamic"
        elif laser is not None:
            reward += self.rewards["laser"]
            done, event = True, "laser"
        elif shelf is not None:
            reward += self.rewards["static"]
            done, event = True, "static"
        elif self._reached_exit(next_state):
            reward += self.rewards["goal"]
            done, success, event = True, True, "escaped"
        elif self.steps + 1 >= self.max_steps * self.action_repeat:
            # Out of time. Charged, so that standing still is the worst outcome
            # rather than the safest one — see the note in `rooms.py`.
            #
            # `steps` counts *ticks* of physics, because that is the clock the
            # obstacle phase and the renderer run on, while `max_steps` is in
            # decisions. Comparing the two directly — which this did — ended
            # every episode after 60 decisions instead of 300, on a mission that
            # needs about 165. It read as "the agent never escapes".
            reward += self.rewards.get("timeout", 0.0)
            done, event = True, "timeout"

        self.state = next_state
        self.steps += 1

        info = {
            "action": action,
            "slipped": False,
            "hazard": event in ("dynamic", "static", "boundary", "laser"),
            "goal": success,
            "event": event,
            "stage": stage,
            "collision": event if event in ("dynamic", "static", "boundary",
                                           "laser") else None,
            "terminalActivated": stage == STAGE_EXIT,
            "exitUnlocked": stage == STAGE_EXIT,
            "layoutSeed": self.layout["seed"],
            "split": self.split,
            "speed": math.hypot(vx, vy),
            "distance": after,
        }
        return next_state, reward, done, info

    # ------------------------------------------------------------------
    # What the screen is given
    # ------------------------------------------------------------------

    def entities(self):
        layout = self.layout
        # The laboratory wall, in the same masonry tiles as rooms 1 to 3. See
        # `chamber_wall_entities` for why the ring sits outside the boundary.
        entities = chamber_wall_entities(self.width, self.height)
        entities += [{
            "id": "terminal",
            "type": "terminal",
            "position": {"x": layout["terminal"][0], "y": layout["terminal"][1]},
            "size": {"width": self.reach * 2, "height": self.reach * 2},
        }, {
            "id": "exit",
            "type": "blastdoor",
            "position": {"x": layout["exit"][0], "y": layout["exit"][1]},
            "size": {"width": self.reach * 2, "height": self.reach * 2},
        }, {
            "id": "launch",
            "type": "start",
            "position": {"x": layout["start"][0], "y": layout["start"][1]},
            "size": {"width": 0.7, "height": 0.7},
        }]

        for index, shelf in enumerate(layout["shelves"]):
            entities.append({
                "id": "shelf%d" % index,
                "type": "shelf",
                "position": {"x": shelf["x"], "y": shelf["y"]},
                "size": {"width": shelf["width"], "height": shelf["height"]},
            })

        # --- the working machinery, none of it physics -------------------
        for lane in layout.get("conveyors", []):
            entities.append({
                "id": lane["id"],
                "type": "conveyor",
                "position": {"x": lane["x"], "y": lane["y"]},
                "size": {"width": lane["width"], "height": lane["height"]},
                "appearance": {"horizontal": lane["horizontal"]},
            })
        for crate in self.conveyor_crates(0):
            entities.append({
                "id": crate["id"],
                "type": "crate",
                "position": {"x": crate["x"], "y": crate["y"]},
                "size": {"width": 0.34, "height": 0.34},
            })
        for beacon in layout.get("beacons", []):
            entities.append({
                "id": beacon["id"],
                "type": "beacon",
                "position": {"x": beacon["x"], "y": beacon["y"]},
                "size": {"width": 0.4, "height": 0.4},
            })

        # --- the beams. A box that bounds the diagonal, with the endpoints
        # carried on the entity so the recipe draws the line the collision test
        # actually uses rather than a line of its own.
        for laser in layout.get("lasers", []):
            one, two = laser["from"], laser["to"]
            entities.append({
                "id": laser["id"],
                "type": "laser",
                "position": {"x": (one[0] + two[0]) / 2,
                             "y": (one[1] + two[1]) / 2},
                "size": {"width": max(0.2, abs(two[0] - one[0])),
                         "height": max(0.2, abs(two[1] - one[1]))},
                # Which diagonal of the box the beam runs along.
                "appearance": {"downhill": (two[1] - one[1]) * (two[0] - one[0]) > 0},
            })

        # Declared once, wherever the patrol starts; every recorded step gives
        # them a new position. Exactly how room 3's guard is handled.
        for mover in layout["movers"]:
            first = mover["path"][0]
            entities.append({
                "id": mover["id"],
                "type": "cart",
                "position": {"x": first[0], "y": first[1]},
                "size": {"width": OBSTACLE_WIDTH, "height": OBSTACLE_WIDTH},
                "appearance": {"pattern": mover["pattern"]},
            })

        return entities

    def loose_entities(self):
        return []

    def frame_extras(self, state):
        """Where the traffic is, and what the terminal and the door look like.

        Positions rather than states for the carts, because they move; states
        for the terminal and the door, because they change appearance. Both are
        pure functions of the state — the carts of its phase — so a recorded
        frame and the live view cannot disagree, and a replay scrubbed
        backwards shows the door locking again.
        """
        phase = state[5]
        unlocked = state[4] == STAGE_EXIT
        states = [
            {"id": "terminal", "state": "used" if unlocked else "armed"},
            {"id": "exit", "state": "open" if unlocked else "locked"},
        ]
        # A lit beam and a dark one are different things, and only the room may
        # say which — the same rule as room 4's landing platform. Once the
        # security system is disarmed the beams go down, which is what the
        # terminal is *for* and the clearest possible signal that it worked.
        for laser in self.laser_states(phase):
            states.append({"id": laser["id"],
                           "state": "off" if unlocked
                                    else ("on" if laser["on"] else "off")})
        for beacon in self.beacon_states(phase):
            # Red while the security system is armed, green once it is not.
            states.append({"id": beacon["id"],
                           "state": ("clear" if unlocked else "alarm")
                                    if beacon["on"] else "dim"})

        # The velocity rides along with the position. The renderer does not
        # read it, but a recorded frame is required to carry it — an obstacle's
        # motion is part of what a replay has to reproduce, and reconstructing
        # it by differencing two frames would be a guess rather than a record.
        #
        # Rounded, and to different depths for different things. Every one of
        # these numbers is written into every recorded frame, and a warehouse
        # frame carries up to nine of them; at full double precision that is
        # about 600 bytes a frame and roughly a megabyte across a batch, spent
        # on digits far below what a 10 m room drawn on a 900-pixel canvas can
        # express. Four decimals is a tenth of a millimetre for the hazards,
        # and the decorative cargo does not need even that.
        positions = [{"id": mover["id"],
                      "position": {"x": round(mover["x"], 4),
                                   "y": round(mover["y"], 4)},
                      "velocity": {"x": round(mover["vx"], 4),
                                   "y": round(mover["vy"], 4)}}
                     for mover in self.mover_positions(phase)]
        positions.extend({"id": crate["id"],
                          "position": {"x": round(crate["x"], 2),
                                       "y": round(crate["y"], 2)}}
                         for crate in self.conveyor_crates(phase))
        return states, positions

    def frame_detail(self, state):
        """Everything a recorded frame needs beyond where the drone is.

        WHAT THIS IS FOR
        A replay of this room has to be inspectable, not merely watchable: the
        question it answers is "what did the agent know, and what did it do
        about it". So each frame carries the mission stage, the objective it
        was heading for, the three sensor rays as the agent received them, and
        which obstacles passed the visibility rule at that instant — the last
        of these being the only way to see, after the fact, that a drone it
        flew into was genuinely outside its cone.

        Every field is derived from the state alone, so a frame reproduced from
        a recording and a frame taken live are identical by construction. The
        replay recomputes nothing; it reads these.
        """
        rays = self.sense(state)
        target = self.target_of(state)
        visible = self.visible_obstacles(state)

        # What ended the run here, if anything did. Read off the state the same
        # way `_tick` reads it, so a frame cannot disagree with the transition
        # that produced it.
        event = None
        if self._outside(state):
            event = "boundary"
        elif self._hit_mover(state) is not None:
            event = "dynamic"
        elif self._hit_laser(state) is not None:
            event = "laser"
        elif self._hit_shelf(state) is not None:
            event = "static"
        elif self._reached_exit(state):
            event = "escaped"

        # Deliberately no sensor range, layout seed or split in here. All three
        # are constant for the whole episode and are carried once on the
        # episode and once on the room definition; repeating them on each of
        # three hundred frames made the recorded batch 3.2 MB, and the page
        # refetches that batch every couple of seconds while training runs.
        return {
            "stage": int(state[4]),
            "phase": int(state[5]),
            # Which way the sensors point, worked out by `_heading` — the same
            # call `sense` and `visible_obstacles` make. Sent rather than left
            # for the browser to infer from the velocity, because the fallback
            # when the drone is nearly still is "point at the objective", and a
            # renderer that guessed "point along the velocity" would draw a
            # cone at a different angle from the one the agent actually used.
            "heading": round(self._heading(state), 4),
            "target": {"x": round(target[0], 4), "y": round(target[1], 4)},
            "targetDistance": round(math.hypot(state[0] - target[0],
                                               state[1] - target[1]), 4),
            "sensors": {name: round(rays[name], 4) for name in self.SENSOR_RAYS},
            # Centre-to-centre distances, by the rule in `visible_obstacles`.
            "visible": [{"id": entry["mover"]["id"],
                         "distance": round(entry["distance"], 3),
                         "bearing": round(entry["offset"], 3)}
                        for entry in visible],
            "event": event,
        }

    def layout_snapshot(self):
        return {"rows": None, "cols": None, "tiles": None,
                "start": list(self.layout["start"]),
                "goal": list(self.layout["exit"])}

    def snapshot(self):
        state = self.state
        rays = self.sense(state)
        nearest = self.nearest_mover(state)
        return {
            "agent": [state[0], state[1]],
            "velocity": [state[2], state[3]],
            "speed": math.hypot(state[2], state[3]),
            "steps": self.steps,
            "terminal": self.is_terminal(state),
            "stage": int(state[4]),
            "phase": int(state[5]),
            "layoutSeed": self.layout["seed"],
            "split": self.split,
            "sensors": rays,
            "sensorRange": self.sensor_range,
            "nearestObstacle": None if nearest is None else round(nearest[0], 3),
            "target": list(self.target_of(state)),
        }
