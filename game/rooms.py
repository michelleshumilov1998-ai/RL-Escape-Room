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
    "S...B.....",   # r0  the door in, and a battery on the way past
    ".#######..",   # r1  sealed the length of the ice, so it must be walked
    ".~~~~~~T#.",   # r2  six icy steps to the pad, the beams directly below
    "LLLLLLLLLD",   # r3  the beam wall, and the one-way door at the end
    ".........o",   # r4  oil, right where the door drops you
    "..........",   # r5
    "...T......",   # r6  the other pad
    ".....=....",   # r7  cracked ice on the last approach
    "..........",   # r8
    ".....E....",   # r9  the control panel
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
    "rewards": {"step": -1.0, "wall": -3.0, "laser": -30.0,
                "battery": 10.0, "shortcut": 5.0, "goal": 100.0},

    # The assignment fixes Dynamic Programming here. Policy Iteration is
    # offered beside it because it is nearly free and the comparison is
    # worth seeing; the default is what the assignment asks for.
    "algorithms": ["value_iteration", "policy_iteration"],
    "algorithm_default": "value_iteration",

    "parameters": ["gamma", "theta", "slip"],

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
            ["Collecting the battery", "-1 + 10 = +9"],
            ["Taking a teleport pad", "-1 + 5 = +4"],
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
# A shaft, and a ledge along the top of it.
#
#   the ledge   9 steps, straight across, with the shaft directly below and
#               three icy patches on the way. Walked flawlessly it is worth
#               +91.
#   the gantry  21 steps: up the west wall, along the roof, down the east
#               wall. Dry the whole way, and nothing to fall into. Worth +79.
#
# The ledge is dangerous to travel *along* and safe to *cross*: a horizontal
# move on ice slides up or down, and down is the shaft, whereas a vertical
# move slides left or right and costs nothing. Both routes step onto the
# ledge at each end; only one walks its length.
#
# THE MARGIN IS THE DESIGN, AND IT WAS MEASURED
# The two routes have to be far enough apart in value that a small amount of
# ice cannot close the gap, or every method takes the safe route and the room
# demonstrates nothing. An earlier version put the gantry one row up, a
# margin of +2, and any ice at all was enough to send Q-Learning round the
# long way too. Walling the middle off makes the margin +12, which leaves a
# usable window:
#
#     ice cost  <  12  <  ice cost + exploration cost
#
# Q-Learning prices the ledge at what a flawless walk earns, so it stays on
# it. SARSA also pays for the ε-greedy steps that walk off it, which lands
# above the margin, so it goes round. Above roughly slip 0.08 the ice alone
# closes the gap and both take the gantry — worth trying with the slider,
# but it is why the default is 0.05.
#
#        c0 c1 c2 c3 c4 c5 c6 c7 c8 c9
ROOM2_GRID = [
    "##########",   # r0
    "#........#",   # r1  the roof: the long way round runs along here
    "#.######.#",   # r2
    "#.######.#",   # r3  the middle is sealed, so the detour is a full lap
    "#.######.#",   # r4
    "#.######.#",   # r5
    "#.######.#",   # r6
    "#.~.~.~..#",   # r7  the ledge, with three icy patches along it
    "#SHHHHHHG#",   # r8  start, the open shaft, the exit door
    "##########",   # r9
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

    "parameters": ["alpha", "gamma", "epsilon", "epsilon_min",
                   "epsilon_decay", "slip", "q_init", "episodes"],

    # Where this room needs something other than the global default. The slip
    # figure is the measured one: above about 0.08 the ice alone closes the
    # gap between the two routes and every method takes the safe one.
    "parameter_defaults": {"slip": 0.05, "epsilon_min": 0.10, "q_init": 90.0},

    "metric": {"key": "meanReward", "label": "Mean reward", "format": "%.1f"},

    # Rooms 2 and 3 support running several methods against the same grid and
    # the same seed, and drawing both learned routes at once.
    "comparison": True,

    "info": {
        "objective": "Cross the sector to the exit door. The model is not "
                     "given: the only way to find out what a step does is to "
                     "take it.",
        "obstacles": [
            "The open shaft along the bottom row ends the run immediately.",
            "Three icy patches sit on the ledge above it. Walking *along* "
            "ice slides you up or down — and down is the shaft.",
            "Crossing the ledge vertically is safe, because a vertical move "
            "only ever slides left or right.",
            "The middle of the sector is sealed, so the safe route is a full "
            "lap: 21 steps against the ledge's 9.",
        ],
        "actions": "Up, down, left, right. Nothing else.",
        "rewards": [
            ["Each step", "-1"],
            ["Walking into a wall", "-1 + (-2) = -3"],
            ["Falling into the shaft", "-1 + (-100) = -101"],
            ["Reaching the exit", "-1 + 100 = +99"],
        ],
        "termination": "The run ends at the exit door or in the shaft, and "
                       "otherwise when the step limit is reached.",
        "note": "The ledge pays more than the lap when it is walked "
                "perfectly: +91 against +79. So a method that learns the "
                "value of behaving perfectly has every reason to take it, "
                "and one that learns the value of what it is actually doing "
                "— random steps and slips included — does not. Run the "
                "comparison to see both routes drawn at once.",
    },
}

# ----------------------------------------------------------------------
# The rest, named but not built
# ----------------------------------------------------------------------

PLACEHOLDERS = [
    {"key": "room3", "number": 3, "built": False,
     "name": "Reactor Control Chamber", "sector": "LAB SECTOR C-07",
     "algorithms": ["sarsa", "expected_sarsa", "q_learning",
                    "double_q_learning"],
     "algorithm_default": "q_learning"},
    {"key": "room4", "number": 4, "built": False,
     "name": "Drone Wind Tunnel", "sector": "LAB SECTOR D-07",
     "algorithms": ["tile_coding_sarsa"],
     "algorithm_default": "tile_coding_sarsa"},
    {"key": "room5", "number": 5, "built": False,
     "name": "Adaptive Storage Facility", "sector": "LAB SECTOR E-12",
     "algorithms": ["tile_coding_q"],
     "algorithm_default": "tile_coding_q"},
]

ROOMS = {ROOM1["number"]: ROOM1, ROOM2["number"]: ROOM2}
for placeholder in PLACEHOLDERS:
    ROOMS[placeholder["number"]] = placeholder

ROOM_NUMBERS = sorted(ROOMS.keys())


def room(number):
    if number not in ROOMS:
        raise KeyError("no room numbered %r" % number)
    return ROOMS[number]


def is_built(number):
    return bool(ROOMS.get(number, {}).get("built"))
