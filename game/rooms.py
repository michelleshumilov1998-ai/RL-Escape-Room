"""The rooms, as data.

A room is a layout, a reward table, which algorithms it allows, and the
words the sidebar shows.  Nothing here is markup and nothing here is
behaviour: the page builds its LEVEL INFO section from these fields, so
changing what a room says is a change to this file alone.

Only room 1 is built.  Rooms 2 to 5 keep their entries so the chamber
select can name them, with `built` false until they exist.
"""

# ----------------------------------------------------------------------
# Room 1 — the Laser Security Chamber
# ----------------------------------------------------------------------
#
# The room of the known map: every rule below is in the model before the
# agent has moved, which is what lets Dynamic Programming route round a
# beam it has never touched.
#
# THE TWO ROUTES
#   the pads    about 14 steps. Six icy strides along row 2 onto a
#               teleport pad that drops the agent past the beam wall
#               entirely. The corridor is sealed above and beamed below on
#               purpose: every stride is taken *on* ice, and a sideways
#               slip on ice goes up into the wall or down into the beams.
#               It also passes nowhere near the battery.
#   the door    about 22 steps. Along the top, down the east side and
#               through the one-way door, which is safe and cannot be
#               taken back.
#
# Which one the plan takes depends on how much grip the floor has, and it
# decides before the agent has moved at all. That flip is the room.
#
# WHAT IS IN IT
#   ~  ice, the ordinary kind          =  cracked ice, twice as loose
#   o  oil, which has no grip at all and carries the agent on the way it
#      was already going
#   L  a beam. Walking into one throws the agent back to the door it came
#      in by, which costs it everything it had walked.
#   B  a battery, worth collecting once and never again
#   T  a pair of teleport pads, each of which sends the agent to the other
#   D  a one-way door: it may be entered going down and no other way
#
#        c0 c1 c2 c3 c4 c5 c6 c7 c8 c9
ROOM1_GRID = [
    "S..#..B...",   # r0  start, wall, battery
    ".~.#.###..",   # r1  weak ice and scattered walls
    ".~.T...#..",   # r2  first teleporter
    ".L.LL..#..",   # r3  separated laser hazards
    "...#..o...",   # r4  oil in an important junction
    "##.L###...",   # r5  central barrier with a dangerous crossing
    "..T..=....",   # r6  second teleporter and strong ice
    ".###.D.##.",   # r7  one-way door toward the lower section
    "...o......",   # r8  oil near the final approach
    "....#E....",   # r9  control panel / exit
]

ROOM1 = {
    "key": "room1",
    "number": 1,
    "built": True,
    "name": "Laser Security Chamber",
    "sector": "LAB SECTOR A-01",
    "grid": ROOM1_GRID,

    # Every transition pays the step cost; event rewards are added on top,
    # so a beam is -1 + -30 and the panel is -1 + 100.
    #
    # A pad pays nothing, deliberately. It is entered unconditionally and
    # both pads lead to each other, so any bonus on entry is a cycle the
    # agent can ride: two steps apart via (5,2), it netted +6 every four
    # steps. Above about γ 0.98 that beat leaving outright — the plan
    # shuttled between the pads 198 times in 400 steps and never reached the
    # panel. The reward for a shortcut is the steps it saves.
    "rewards": {"step": -1.0, "wall": -3.0, "laser": -30.0,
                "battery": 10.0, "shortcut": 0.0, "goal": 100.0},

    # The assignment fixes Dynamic Programming here. Policy Iteration is
    # offered beside it because it is nearly free and the comparison is
    # worth seeing; the default is what the assignment asks for.
    "algorithms": ["value_iteration", "policy_iteration"],
    "algorithm_default": "value_iteration",

    "parameters": ["gamma", "theta", "slip", "battery_reward"],

    # Value Iteration and Policy Iteration reach the same plan by different
    # routes, and how many sweeps each needs is the thing worth seeing side
    # by side.
    "comparison": True,

    # What the status strip's one metric is.
    "metric": {"key": "startValue", "label": "V(start)", "format": "%.2f"},

    "info": {
        "objective": "Cross the beam wall to the control panel and shut the "
                     "security grid down. Every rule of this room is known "
                     "before the first move.",
        "obstacles": [
            "A wall of beams cuts the chamber in two. Walking into one "
            "throws R-5 back to the door it came in by, which costs it "
            "everything it had walked.",
            "Ice sends it sideways instead of where it aimed. Cracked ice "
            "does the same, twice as often.",
            "Oil has no grip at all: on oil it carries on the way it was "
            "already going, whatever it tried.",
            "A pair of teleport pads skips the wall entirely — but the way "
            "to the first pad crosses ice directly above the beams.",
            "A one-way door at the east end may be entered going down and "
            "no other way, so taking it cannot be undone.",
        ],
        "actions": "Up, down, left, right. The action is what the agent "
                   "tries; the floor it is standing on decides what happens.",
        "rewards": [
            ["Each step", "-1"],
            ["Walking into a wall", "-1 + (-3) = -4"],
            ["Walking into a beam", "-1 + (-30) = -31, and back to the start"],
            # Left without a number: the battery bonus is a slider, so any
            # figure written here would be a lie at every other setting.
            ["Collecting the battery", "-1 + the battery bonus"],
            ["Taking a teleport pad", "-1, like any other step"],
            ["Reaching the panel", "-1 + 100 = +99"],
        ],
        "termination": "Only the control panel ends a run. A beam does not "
                       "— it sends R-5 back to the start, which is worse to "
                       "plan around than simply ending it.",
        "note": "This is the only room whose model is known in advance. "
                "Nothing is learned from experience here — the plan is "
                "computed from the model before the agent moves at all, "
                "which is why it can avoid a beam it has never touched. The "
                "state is more than a square: whether the battery is "
                "already in hand, and which way it was last pushed, both "
                "change what happens next.",
    },
}

# ----------------------------------------------------------------------
# Room 2 — the Broken Bridge Sector
# ----------------------------------------------------------------------
#
# A shaft through the middle of the sector, and two ways round it.
#
#   the span    7 steps along the bottom: two sound bridge sections with four
#               collapsing planks between them. Every plank has a chance of
#               giving way under the step that lands on it, and that is a
#               fall. Crossed cleanly it is worth +93.
#   the lap     21 steps: up the west wall, along the safe corridor at the
#               top, down the east wall. Nothing on it can drop the agent.
#               Worth +79.
#
# WHERE THE RISK IS, AND WHY IT IS THERE RATHER THAN ANYWHERE ELSE
# A plank fails under the step that arrives on it, so the danger is the
# crossing itself and it compounds: four planks at chance p get across with
# probability (1-p)^4. At the default 0.10 that is about two attempts in
# three. Survive a plank and it is gone behind you, so the span cannot be
# walked back — a step returning onto a gap is the same fall.
#
# That gives the room two different reasons to fear the span, which is the
# point of it. The first is the plank failing, which no policy can avoid. The
# second is the exploratory step: every cell of the span has the shaft
# directly above it, so a random step upwards is fatal, and stepping back is
# fatal once a plank has gone. A method that prices the route it would walk
# perfectly sees only the first. A method that prices the route it is
# actually walking pays for both.
#
# THE MARGIN
# The span is 14 ahead of the lap when it works (+93 against +79), which has
# to be wide enough that the plank risk alone does not close it, or every
# method takes the lap and the room demonstrates nothing:
#
#     plank risk  <  14  <  plank risk + exploration cost
#
# The failure chance is a slider, so the point where that stops holding can
# be found rather than taken on trust.
#
#        c0 c1 c2 c3 c4 c5 c6 c7 c8 c9
ROOM2_GRID = [
    "##########",
    "#........#",
    "#.######.#",
    "#.#....#.#",
    "#.#....#.#",
    "#.#....#.#",
    "#.#....#.#",
    "#.#....#.#",
    "#SGCCCCGE#",
    "##########",
]

ROOM2 = {
    "key": "room2",
    "number": 2,
    "built": True,
    "name": "Broken Bridge Sector",
    "sector": "LAB SECTOR B-04",
    "grid": ROOM2_GRID,

    "rewards": {"step": -1.0, "wall": -2.0, "hazard": -100.0, "goal": 100.0},

    # The assignment fixes SARSA here. The others are offered because this
    # room is not "the SARSA room" internally — it is a model-free grid that
    # ships with SARSA selected, and the comparison is the lesson.
    "algorithms": ["sarsa", "expected_sarsa", "q_learning",
                   "double_q_learning"],
    "algorithm_default": "sarsa",

    # No `slip` here: there is not a loose tile in this sector, so the control
    # would move nothing. The risk in this room is the planks, and that has a
    # control of its own.
    "parameters": ["alpha", "gamma", "epsilon", "epsilon_min",
                   "epsilon_decay", "collapse_chance", "q_init", "episodes"],

    "parameter_defaults": {"q_init": 90.0},

    "metric": {"key": "meanReward", "label": "Mean reward", "format": "%.1f"},

    # Rooms 2 and 3 support running several methods against the same grid and
    # the same seed, and drawing both learned routes at once.
    "comparison": True,

    "info": {
        "objective": "Cross the sector to the exit door. The model is not "
                     "given: the only way to find out what a step does is to "
                     "take it.",
        "obstacles": [
            "A shaft fills the middle of the sector. Entering it anywhere "
            "ends the run.",
            "The span along the bottom is the short way across: two sound "
            "bridge sections with four collapsing planks between them.",
            "A plank can give way under the step that lands on it, and that "
            "is a fall. Four of them in a row, so the risk compounds.",
            "Survive a plank and it is gone behind you — the span cannot be "
            "walked back, and a step returning onto a gap is the same fall.",
            "Every cell of the span has the shaft directly above it, so one "
            "exploratory step upwards ends the run wherever you are on it.",
        ],
        "actions": "Up, down, left, right. Nothing else.",
        # This room is *about* the choice between these two, so it is stated
        # rather than left to be inferred from the layout. `best` is what a
        # flawless walk earns; the margin between them is the whole design.
        "routes": [
            {"name": "The span", "steps": 7, "best": 93,
             "risk": "Four planks that can give way, and the shaft above "
                     "every step of it."},
            {"name": "The lap", "steps": 21, "best": 79,
             "risk": "None. Nothing on it can drop the agent."},
        ],
        "rewards": [
            ["Each step", "-1"],
            ["Walking into a wall", "-1 + (-2) = -3"],
            ["Falling into the shaft", "-1 + (-100) = -101"],
            ["A plank giving way underfoot", "-1 + (-100) = -101"],
            ["Reaching the exit", "-1 + 100 = +99"],
        ],
        "termination": "The run ends at the exit door, in the shaft, or under "
                       "a plank that gave way, and otherwise when the step "
                       "limit is reached.",
        "note": "The span pays more than the lap when it works: +93 against "
                "+79. So a method that learns the value of behaving perfectly "
                "has every reason to take it, and one that learns the value "
                "of what it is actually doing — random steps and failed "
                "planks included — does not. Which plank has already gone is "
                "part of the state here; without that, stepping towards the "
                "span would be worth an average of \"fine\" and \"fatal\", "
                "and an average is not a fact about where you are.",
    },
}

# ----------------------------------------------------------------------
# Room 3 — the Reactor Control Chamber
# ----------------------------------------------------------------------
#
# A service ring around the reactor core, with a shaft straight through the
# middle of it. Three generators have to be brought up IN ORDER — 1, then 2,
# then 3 — and only then does the blast door at the south-east unlock.
#
# WHY THE ORDER IS THE POINT
# The generators are placed so that taking them in order is one clockwise lap
# of the ring. The exit reward is therefore about two dozen steps away from
# the first move that leads towards it, with nothing paid out in between
# except the generators themselves. That distance is what the room is for:
# it is the case where a method has to carry value a long way back.
#
# WHAT GETS IN THE WAY
#   the guard   walks the ring anticlockwise, one cell per step, forever. It
#               cannot be outrun on the ring and cannot be passed, so meeting
#               it means having somewhere else to be.
#   the shaft   runs from the top of the ring to the bottom through the
#               middle. Six steps instead of going round, no live panels on
#               it, and the guard never leaves the ring — so it is the only
#               place that is always safe to stand.
#   the doors   both ends of the shaft are on a cycle: open for two steps of
#               every four. Getting in means arriving on the right step or
#               waiting for it, which is why this is the only room with a
#               WAIT action.
#
#        c0 c1 c2 c3 c4 c5 c6 c7 c8 c9
#   1 2 3  the generators, which come up in that order and no other
#   a b c  their keys: 'a' starts '1', 'b' starts '2', 'c' starts '3'
#   d      a sliding door, open two steps in every four
#   R      the reactor door, shut until all three generators are up
#
# Each key sits well away from the generator it belongs to, so every errand is
# a journey rather than a detour of one square: key 'a' is on the south run and
# generator '1' in the north-east corner, key 'b' is on the north run and
# generator '2' in the south-east, and key 'c' is in the shaft itself.
#
#        c0 c1 c2 c3 c4 c5 c6 c7 c8 c9
ROOM3_GRID = [
    "##########",   # r0
    "#S.b...1##",   # r1  north run; key b, and generator 1 in the corner
    "#.##d##.##",   # r2  the shaft's north door
    "#.##.##.##",   # r3
    "#.##c##.##",   # r4  the shaft, with key c inside it
    "#.##.##.##",   # r5
    "#.##d##.##",   # r6  the shaft's south door
    "#3.a...2##",   # r7  south run; generator 3, key a, generator 2
    "####R#####",   # r8  the reactor door
    "####E#####",   # r9  the exit, behind it
]

# The service ring, listed anticlockwise, which is the direction the guard
# walks. R-5's errands run clockwise, so the two meet head-on once a lap and
# R-5 has to have somewhere to be. The guard never enters the shaft, which is
# what makes the shaft a refuge and not merely a shortcut.
_ROOM3_CLOCKWISE = (
    [(1, col) for col in range(1, 8)]            # north run, west to east
    + [(row, 7) for row in range(2, 8)]          # east run, north to south
    + [(7, col) for col in range(6, 0, -1)]      # south run, east to west
    + [(row, 1) for row in range(6, 1, -1)]      # west run, south to north
)
ROOM3_PATROL = list(reversed(_ROOM3_CLOCKWISE))

ROOM3 = {
    "key": "room3",
    "number": 3,
    "built": True,
    "name": "Reactor Control Chamber",
    "sector": "LAB SECTOR C-07",
    "grid": ROOM3_GRID,

    # This room has things that move on their own, so it is built on the
    # environment that knows about them rather than on the plain grid.
    "world": "reactor",

    "patrol": ROOM3_PATROL,
    # Open for two steps out of every four. The patrol length has to be a
    # whole number of these or the pattern stops repeating and the guard index
    # is no longer enough to know the doors from — `ReactorWorld` checks it.
    "door_period": 4,
    "door_open_for": 2,

    "rewards": {"step": -1.0, "wall": -2.0, "key": 15.0, "generator": 20.0,
                "caught": -100.0, "goal": 100.0},

    # The assignment fixes Q-Learning here. The others are offered for the
    # same reason as in room 2: the comparison is worth more than the label.
    "algorithms": ["q_learning", "double_q_learning", "sarsa",
                   "expected_sarsa"],
    "algorithm_default": "q_learning",

    # No `slip`: nothing in this chamber is loose. The difficulty here is
    # timing and sequence, not grip.
    "parameters": ["alpha", "gamma", "epsilon", "epsilon_min",
                   "epsilon_decay", "q_init", "episodes"],

    # A long delayed reward needs a discount close to one to reach the start
    # at all, and enough episodes to get there.
    "parameter_defaults": {"gamma": 0.99, "episodes": 4000, "q_init": 0.0},

    "metric": {"key": "meanReward", "label": "Mean reward", "format": "%.1f"},
    "episode_metric": "reward",
    "comparison": True,

    "info": {
        "objective": "Fetch each generator's key, bring the three generators "
                     "up in order, and leave through the reactor door. "
                     "Nothing is given in advance; the only way to learn what "
                     "a step does is to take it.",
        "obstacles": [
            "Every generator needs its own key, and no other key will do. "
            "Six errands before the exit will open at all.",
            "The generators only count in order. Standing on the third one "
            "first does nothing, key in hand or not.",
            "Each key is kept well away from the generator it belongs to.",
            "The reactor door stays shut until all three are running.",
            "A security robot walks the ring one cell per step, against the "
            "way the errands run. Meeting it ends the run, and it cannot be "
            "outrun or passed.",
            "A generator is white until it is running, and green once it is. "
            "How far through the sequence the reactor is can be read off the "
            "chamber itself.",
            "The shaft through the middle is shorter, and the robot never "
            "enters it — but both ends are doors that are open only two steps "
            "in every four.",
        ],
        "actions": "Up, down, left, right, and wait. Waiting is worth having "
                   "here: a shut door opens again two steps later.",
        "rewards": [
            ["Each step", "-1"],
            ["Walking into a wall or a shut door", "-1 + (-2) = -3"],
            ["Picking up a key", "-1 + 15 = +14"],
            ["Bringing a generator up, in order and keyed", "-1 + 20 = +19"],
            ["Being caught by the robot", "-1 + (-100) = -101"],
            ["Leaving through the reactor door", "-1 + 100 = +99"],
        ],
        "termination": "The run ends at the exit or in the robot's hands, and "
                       "otherwise when the step limit is reached.",
        "note": "The exit is a long way from the first move that leads to it — "
                "six errands, in a fixed order, before the door will open — "
                "and that distance is the difficulty this room is built "
                "around. The state carries three things beyond the square: "
                "which keys are in hand, how far through the sequence the "
                "reactor is, and where the patrol has got to. The doors are "
                "worked out from the patrol rather than from a clock of their "
                "own: the cycle divides into the patrol exactly, so one "
                "number covers both moving things.",
    },
}

# ----------------------------------------------------------------------
# The rest, named but not built
# ----------------------------------------------------------------------

PLACEHOLDERS = [
    {"key": "room4", "number": 4, "built": False,
     "name": "Drone Wind Tunnel", "sector": "LAB SECTOR D-07",
     "algorithms": ["tile_coding_sarsa"],
     "algorithm_default": "tile_coding_sarsa"},
    {"key": "room5", "number": 5, "built": False,
     "name": "Adaptive Storage Facility", "sector": "LAB SECTOR E-12",
     "algorithms": ["tile_coding_q"],
     "algorithm_default": "tile_coding_q"},
]

ROOMS = {ROOM1["number"]: ROOM1, ROOM2["number"]: ROOM2,
         ROOM3["number"]: ROOM3}
for placeholder in PLACEHOLDERS:
    ROOMS[placeholder["number"]] = placeholder

ROOM_NUMBERS = sorted(ROOMS.keys())


def room(number):
    if number not in ROOMS:
        raise KeyError("no room numbered %r" % number)
    return ROOMS[number]


def is_built(number):
    return bool(ROOMS.get(number, {}).get("built"))
