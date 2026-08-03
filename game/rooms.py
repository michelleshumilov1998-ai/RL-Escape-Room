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
# WHERE THE ICE IS, AND WHY IT MOVED
# It used to sit at (1,1) and (2,1) with cracked ice at (6,5) — all three
# *beside* the route rather than on it. Measured: at every slip setting the plan
# walked S → (0,1) → (0,2) → (1,2) → (2,2) → the pad, then out of the far pad
# along r6 c3–c4 and down to the panel, eleven steps, crossing ice zero times.
# So the room's central mechanic never appeared in the answer, and the only time
# a player saw a slip was during exploration — where an arrow pointing one way
# and a robot going another reads as a rendering fault rather than as the floor.
#
# The ice is now on the route the plan actually takes, in the two places the
# route cannot avoid:
#
#   r6 c3–c4   the corridor out of the second teleport pad. The only way off
#              that pad, and a sideways slip there goes up into the beam at
#              (5,3) — which throws R-5 back to the start. Slipping here is
#              expensive without being a lottery.
#   r8 c4–c5   the final approach, directly above the panel, plus r9 c6 so the
#              approach cannot simply be taken from the east instead. A slip
#              here costs a step or two and nothing worse, which is what makes
#              the last stride a matter of grip rather than of luck.
#
#        c0 c1 c2 c3 c4 c5 c6 c7 c8 c9
ROOM1_GRID = [
    "S..#..B...",   # r0  start, wall, battery
    ".~.#.###..",   # r1  weak ice and scattered walls
    ".~.T...#..",   # r2  first teleporter
    ".L.LL..#..",   # r3  separated laser hazards
    "...#..o...",   # r4  oil in an important junction
    "##.L###...",   # r5  central barrier with a dangerous crossing
    "..T~~=....",   # r6  second teleporter, and the icy corridor out of it
    ".###.D.##.",   # r7  one-way door toward the lower section
    "...o~~....",   # r8  oil, then ice on the final approach
    "....#E~...",   # r9  control panel / exit, iced from the east too
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

    # THE PLANNER'S GRAPH, WHICH IS NOT AN EPISODE GRAPH.
    #
    # A room that names no charts gets the shared four: reward per episode,
    # steps per episode, exploration rate, convergence measure. Every one of
    # those is a property of an *episode*, and Value Iteration does not have
    # episodes. It sweeps the state space, and it has no exploration rate at
    # all. So this room drew four graphs that were empty, permanently, under
    # four labels describing quantities the algorithm never produces.
    #
    # What it does measure is the largest change to any state's value in a
    # sweep, which is exactly the quantity the stopping rule tests: it halts
    # when that falls below theta. The far end has always sent it as
    # `snapshot.curve`; nothing was drawing it. One graph, of the thing the
    # algorithm actually computes, with the stopping threshold on it.
    #
    # `source: sweeps` is what tells the charts panel to read the planner
    # curve rather than the episode history, and `thresholdKey` makes the
    # dashed line follow the theta slider.
    "charts": [
        {"key": "delta", "source": "sweeps", "scale": "log",
         "thresholdKey": "theta",
         "label": "Largest value change per sweep"},
    ],

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
# A lethal shaft through the middle of the sector, and two ways past it.
#
#   the span    9 steps: a steel gantry across the shaft. Two sound deck
#               tiles ('G') carry the agent on and off, with four collapsing
#               planks ('C') between them. Crossed cleanly it is worth +29.
#   the way     21 steps: up the west wall, along the top, down the east.
#   round       Nothing on it can drop the agent, but four cells of it are
#               iced, so it is not free of chance either. Worth +17.
#
# THE RISK IS ONE DRAW PER CROSSING, NOT ONE PER PLANK
# `lands_on_sound_plank` returns true only for the step that takes the agent
# from off the span on to a plank. So the collapse is sampled once, on the way
# in; survive it and the whole gantry is crossed. The number on the slider is
# therefore exactly the chance of losing a crossing:
#
#     P(a crossing fails) = collapse_chance
#
# Measured over 2000 attempts per setting: 0.10 -> 0.101, 0.25 -> 0.253,
# 0.50 -> 0.524. It is NOT 1-(1-p)^4; the four planks do not each roll a die.
# This was the other way round once, and it made the control unreadable — at
# the 0.10 default the true risk was 34% under a label that said 10%.
#
# THE MARGIN, AND WHAT IT BUYS
# The span is 12 ahead of the way round when it works (+29 against +17). A
# fall forfeits the exit as well as paying the penalty, so the span is worth
# taking only while
#
#     crossing risk  <  steps saved / (exit + fall penalty + steps walked in)
#
# At the documented +100/-100 that threshold is about 3% — the span is
# irrational almost immediately and no 10x10 map can fix it. At +38/-15 it is
# about 25%, which is where a slider is worth having. Measured across five
# seeds: span at 0.00, 0.05, 0.10 and 0.20; mixed 2/5 at 0.30; the way round
# at 0.40 and 0.60.
#
# THERE IS NO WALKABLE SPACE BESIDE THE SPAN
# Every plank has shaft directly above and directly below it, so stepping
# sideways off the gantry is impossible rather than merely discouraged. A test
# enforces that each plank's four neighbours are shaft, wall or span.
#
# THE ICE
# Four slippery cells, at the two corners of the way round: one before a turn,
# one after it, and one against a wall on each side. Each has one side open and
# one side solid, so a slip costs progress or bumps stone — never two
# identical outcomes, and never a fall, because nothing beside that corridor
# is lethal.
#
# 'H' is the shaft; 'C' the collapsing planks; 'G' the sound deck either side;
# '~' the ice.
#
#        c0 c1 c2 c3 c4 c5 c6 c7 c8 c9
ROOM2_GRID = [
    "##########",
    "#.~....~.#",
    "#~.####.~#",
    "#.#HHHH#.#",
    "#.#HHHH#.#",
    "#.#HHHH#.#",
    "#.#HHHH#.#",
    "#.GCCCCG.#",
    "#S#HHHH#E#",
    "##########",
]

ROOM2 = {
    "key": "room2",
    "number": 2,
    "built": True,
    "name": "Broken Bridge Sector",
    "sector": "LAB SECTOR B-04",
    "grid": ROOM2_GRID,

    # RESTORED TO THE DOCUMENTED NUMBERS
    #
    # These were changed to step -4 / hazard -40 with two planks, to make the
    # span a live decision across a 30-50% collapse band. That was tuning, not
    # compliance: the assignment fixes the grid size, SARSA and an unknown
    # model, and says nothing about reward magnitudes. It also put the code out
    # of step with the README, which documents +93 against +79 and a -101 fall.
    #
    # So the simplest implementation that satisfies the assignment is the
    # documented one, and that is what is here.
    #
    # WHAT THAT COSTS, STATED PLAINLY
    # On these numbers the span is worth taking only when the collapse chance
    # is very near zero: a 14-point margin (+93 against +79) cannot pay for a
    # 100-point fall, and with four planks a 30% chance leaves 24% survival.
    # The slider therefore moves the policy only at the very bottom of its
    # range. That is a property of the documented design, not a defect
    # introduced here, and it is left alone because changing it is a redesign.
    "rewards": {"step": -1.0, "wall": -2.0, "hazard": -15.0, "goal": 38.0},

    # The assignment fixes SARSA here. The others are offered because this
    # room is not "the SARSA room" internally — it is a model-free grid that
    # ships with SARSA selected, and the comparison is the lesson.
    "algorithms": ["sarsa", "expected_sarsa", "q_learning",
                   "double_q_learning"],
    "algorithm_default": "sarsa",

    # Two sources of chance, and a control for each: the ice on the lap and
    # the planks on the span.
    "parameters": ["alpha", "gamma", "epsilon", "epsilon_min",
                   "epsilon_decay", "slip", "collapse_chance", "q_init",
                   "episodes"],

    # Optimistic initialisation, as documented: above the best return the
    # room can pay, so every action is tried before one is settled on.
    "parameter_defaults": {"q_init": 30.0},

    # THE TRAINING DASHBOARD.
    #
    # Named explicitly rather than falling back to the shared four, so that
    # every title says what the series actually is:
    #
    #   "Episode Return"  is the sum of every reward in the episode, G = Sum r,
    #                     not the last reward. `session._log_episode` writes it
    #                     from a running total.
    #   "Smoothed Return" is a 20-episode trailing mean of that same series,
    #                     computed on the FULL history before any downsampling.
    #   "Loss / Mean |TD Error|" is mean(|delta|) over the episode, where delta
    #                     is the temporal-difference error of whichever method
    #                     is running. There is no neural network in this
    #                     project and this is NOT a network loss -- it is the
    #                     loss-like convergence diagnostic the method itself
    #                     produces. See `session._log_episode`.
    #   "Success rate"    is a 20-episode mean of a 0/1 indicator taken from
    #                     `info["goal"]`, not from a reward threshold.
    "charts": [
        {"key": "reward", "label": "Episode Return (total reward)"},
        {"key": "reward", "smoothOnly": True,
         "label": "Smoothed Return (20-episode moving average)"},
        {"key": "convergence", "label": "Loss / Mean |TD Error|"},
        {"key": "epsilon", "label": "Exploration rate \u03b5"},
        {"key": "steps", "label": "Steps per episode"},
        {"key": "success", "smoothOnly": True,
         "label": "Success rate (20-episode moving average)"},
    ],

    "metric": {"key": "meanReward", "label": "Mean reward", "format": "%.1f"},

    # DECOR IS SCENERY. IT IS NOT PART OF THE ROOM.
    #
    # Read only by `definition.py`, which turns it into entities for the
    # renderer. `GridWorld` never sees this key: the cells below stay ordinary
    # floor, with the same transitions, the same -1 step and the same place in
    # the state space they always had.
    #
    # What it buys is the middle of the sector looking like the sealed
    # maintenance void the briefing describes, rather than like a flat black
    # rectangle. Rows 3-7, columns 3-6 — the pocket enclosed by the walls of
    # r2 and columns 2 and 7, open only downwards onto the planks.

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
            "One steel span crosses it, and it is the only way over: a "
            "gantry of four planks with solid deck at either end.",
            "The span is risked once, as R-5 steps on to it. Survive that "
            "and the whole crossing is made.",
            "The way round is safe from the span, but four cells of it are "
            "iced and a step on those can carry the agent sideways.",
        ],
        "actions": "Up, down, left, right. Nothing else.",
        "routes": [
            {"name": "The span", "steps": 9, "best": 29,
             "risk": "One roll of the dice, taken on the step that puts R-5 "
                     "on the planks."},
            {"name": "The way round", "steps": 21, "best": 17,
             "risk": "Nothing can drop the agent. Four iced cells at the two "
                     "corners can cost it steps."},
        ],
        "rewards": [
            ["Each step", "-1"],
            ["Walking into a wall", "-1 + (-2) = -3"],
            ["Falling into the shaft", "-1 + (-15) = -16"],
            ["The span giving way underfoot", "-1 + (-15) = -16"],
            ["Reaching the exit", "-1 + 38 = +37"],
        ],
        "termination": "The run ends at the exit door, in the shaft, under a "
                       "span that gave way, and otherwise when the step "
                       "limit is reached.",
        "note": "The span pays more than the way round when it works: +29 "
                "against +17. The whole crossing is risked once, on the step "
                "that puts R-5 on the planks, so the number on the slider is "
                "the chance of losing a crossing and nothing has to be "
                "compounded to read it. SARSA learns the value of the route "
                "it is actually walking \u2014 exploratory steps beside a "
                "lethal shaft included \u2014 which is why it gives the span "
                "up earlier than a method that prices only a perfect walk.",
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

    # THE TRAINING DASHBOARD.
    #
    # Named explicitly rather than falling back to the shared four, so that
    # every title says what the series actually is:
    #
    #   "Episode Return"  is the sum of every reward in the episode, G = Sum r,
    #                     not the last reward. `session._log_episode` writes it
    #                     from a running total.
    #   "Smoothed Return" is a 20-episode trailing mean of that same series,
    #                     computed on the FULL history before any downsampling.
    #   "Loss / Mean |TD Error|" is mean(|delta|) over the episode, where delta
    #                     is the temporal-difference error of whichever method
    #                     is running. There is no neural network in this
    #                     project and this is NOT a network loss -- it is the
    #                     loss-like convergence diagnostic the method itself
    #                     produces. See `session._log_episode`.
    #   "Success rate"    is a 20-episode mean of a 0/1 indicator taken from
    #                     `info["goal"]`, not from a reward threshold.
    "charts": [
        {"key": "reward", "label": "Episode Return (total reward)"},
        {"key": "reward", "smoothOnly": True,
         "label": "Smoothed Return (20-episode moving average)"},
        {"key": "convergence", "label": "Loss / Mean |TD Error|"},
        {"key": "epsilon", "label": "Exploration rate \u03b5"},
        {"key": "steps", "label": "Steps per episode"},
        {"key": "success", "smoothOnly": True,
         "label": "Success rate (20-episode moving average)"},
    ],

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
# Room 4 — the Drone Test Chamber
# ----------------------------------------------------------------------
#
# The room with no grid under it. Rooms 1 to 3 have finitely many states and a
# table with a row for each; here the state is four real numbers and there is
# no table to be had. That is the whole reason this room exists.
#
#     state = (x, y, vx, vy)      metres, metres per second
#     10 x 10 m, integrated every 0.02 s, speed held inside [-1, 1] m/s
#
# THE TASK IS THE LANDING, NOT THE ARRIVAL
# Touching the pad is easy. Touching it at under 0.22 m/s on both axes is the
# problem, and it is what stops the answer being "aim at the pad and hold full
# thrust". A fast arrival is a crash and ends the run at -30.
#
# WHY THE FURNITURE IS ALL STATIC
# Every force in this chamber depends on where the drone is and on nothing
# else, so it stays a function of the state. A gate rising and falling on a
# timer would break that: the same four numbers would mean "clear" at one
# moment and "blocked" at the next with nothing in the state saying which.
# Room 3 can afford its sliding doors because it derives them from the guard's
# patrol index, which is part of its state; the assignment fixes this room's
# state at four numbers, so there is no free dimension to hang a phase on.
# See the note in `game/drone.py`.
#
# THE LAYOUT, IN METRES
#     launch pad   (1.5, 8.2)   bottom left
#     landing pad  (8.2, 1.6)   top right, 1.6 x 0.9 m
#     two pillars  in the middle, which the direct line runs into
#     a wind band  across the top left, blowing down — against the approach
#     a slow zone  which costs time and makes a gentle arrival easier
#     a boost zone which shortens the route and costs 5 to enter
#
# y increases downwards, as everywhere else in this project.

ROOM4 = {
    "key": "room4",
    "number": 4,
    "built": True,
    "name": "Drone Wind Tunnel",
    "sector": "LAB SECTOR D-07",
    "subtitle": "FLIGHT STABILISATION TEST",

    # This room is built on the continuous world rather than the grid.
    "world": "drone",

    "size": (10.0, 10.0),
    # The assignment's figure. At this step size a 10 m room is several
    # hundred steps across, which is why this room has a step limit of its own.
    "dt": 0.02,
    # Thrust, in m/s^2. A third of a second of it reaches the speed limit,
    # which is responsive enough to steer and still slow enough that stopping
    # has to be planned for rather than done on arrival.
    #
    # Measured while setting these: an action pushes along *one* axis, so only
    # one axis can be decelerated per step, and a drone that has to lose speed
    # on both alternates and brakes at half the rate. At 2.0 m/s^2 against a
    # 0.22 m/s limit the best hand-flown approach arrived at 0.242 — a near
    # miss, and a room whose task is very slightly out of reach is not a
    # harder room, it is a broken one. These are the numbers that leave the
    # landing demanding and reachable.
    "thrust": 3.0,
    "drag": 0.7,
    "speed_limit": 1.0,

    "start": (1.5, 8.2),
    # The velocity is discrete, so the only speeds that exist are 0, 1 and
    # sqrt(2). At 1.0 an arrival along one axis lands and a diagonal one is a
    # crash; below 1.0 nothing but a full stop counts. See `config.PARAMETERS`.
    "pad": {"x": 8.2, "y": 1.6, "width": 1.6, "height": 0.9,
            "landing_speed": 1.00},

    "pillars": [
        {"x": 4.3, "y": 5.5, "radius": 0.95},
        {"x": 6.5, "y": 3.5, "radius": 0.80},
    ],

    "zones": [
        # Blows downwards across the top left, so the last part of the climb
        # towards the pad is made against it.
        {"id": "wind1", "type": "wind", "x": 3.6, "y": 2.6,
         "width": 3.0, "height": 3.0, "wind": (0.0, 1.4)},
        # A second bank, wide and shallow, across the bottom right — and the
        # reason it is there is that the first one was not in the way.
        #
        # The route the agent actually learns runs right along the floor and
        # then climbs the east wall, which passes nowhere near the top-left
        # band; the chamber was solvable without ever meeting a fan. This one
        # sits exactly on the corner of that L, so the turn upwards is made
        # into a downdraught and the climb has to be flown rather than
        # coasted. Its fans stand in a row along its top edge, which is what
        # `_fans_for` does with a band that blows down.
        #
        # 1.1 m/s² against 3.0 m/s² of thrust: enough to be a real hurdle,
        # and well short of pinning the drone to the floor. Measured below.
        {"id": "wind2", "type": "wind", "x": 7.6, "y": 7.8,
         "width": 3.6, "height": 1.6, "wind": (0.0, 1.1)},
        # Thick air: costs time, and makes arriving slowly much easier. The
        # trade this room is built around is that it is not on the short way.
        {"id": "slow1", "type": "slow", "x": 7.6, "y": 5.4,
         "width": 2.4, "height": 2.0, "extra_drag": 3.0},
        # A thruster overcharge: doubles the push, shortens the route, and is
        # far harder to arrive slowly out of. Charged once per entry.
        {"id": "boost1", "type": "boost", "x": 2.4, "y": 5.0,
         "width": 1.6, "height": 2.4, "thrust_scale": 2.0, "danger": True},
    ],

    # Continuous rewards, unlike the three grid rooms. `progress` multiplies
    # the metres closed on the pad this step, so it is a rate rather than an
    # event: at the speed limit one step closes 0.02 m and pays 0.1.
    "rewards": {"step": -0.01, "progress": 5.0, "wall": -100.0,
                "danger": -5.0, "goal": 200.0, "hard_landing": -30.0},

    # The assignment asks for function approximation here. The discretised
    # table is offered beside it because the comparison *is* the lesson: it is
    # the obvious thing to try, and watching it fail is what shows why the
    # approximation is needed.
    "algorithms": ["semi_gradient_sarsa", "semi_gradient_q", "discretised_q"],
    "algorithm_default": "semi_gradient_sarsa",

    "parameters": ["alpha", "gamma", "epsilon", "epsilon_min",
                   "epsilon_decay", "tilings", "buckets", "wind",
                   "landing_speed", "episodes"],

    # γ close to one: the landing is several hundred steps from the launch, so
    # a discount that would do for a ten-cell grid leaves the pad worth
    # nothing at all by the time the value reaches the start.
    "parameter_defaults": {"gamma": 0.995, "alpha": 0.30, "epsilon_min": 0.02,
                           "epsilon_decay": 0.997, "episodes": 1200},

    # An episode here is hundreds of steps rather than tens, so the shared
    # 400-step limit would end every run in mid-air before it reached the pad.
    #
    # Set against measurement rather than guessed. Hand-flown reference routes
    # land in 891 steps at their quickest and 1821 at their most cautious, and
    # the trade between the two is the room; a limit that only admitted the
    # quickest would be deciding that trade in advance. Early training
    # episodes are far shorter than any of these — a random thrust reaches a
    # wall in about ninety steps — so the cost of the generous limit is paid
    # only once the agent is competent enough to be worth watching.
    "max_steps": 1800,

    # How many environment steps of this room are worth one step of the
    # interface's speed tiers.
    #
    # The tiers are "steps per second", and that unit does not mean the same
    # thing here as it does in a grid room. One step in rooms 1 to 3 moves the
    # agent a whole cell — a tenth of the room. One step here is 0.02 s of
    # flight, so at the speed cap it moves 0.02 m: a five-hundredth of the
    # chamber. Measured at the "Normal" tier of 9 steps a second, that is
    # 11 pixels a second and 56 seconds to cross the chamber, against 569
    # pixels a second for a grid agent on the same tier. It reads as a drone
    # that does not move at all, which is exactly how it was reported.
    #
    # 12 puts Normal at about 4.6 seconds to cross under full thrust and Slow
    # at about 14, which is the pace the grid rooms feel like. Full
    # distance-parity with a grid step would be 50 — one tier-step per world
    # unit — and crosses the chamber in a second, too fast to watch the
    # control that this room is about.
    #
    # The same mismatch, in the replay rather than the live view, is what
    # `playback.stepsPerSecond` in `definition.py` already exists to fix.
    "live_step_scale": 12,

    # THE TRAINING DASHBOARD.
    #
    # Named explicitly rather than falling back to the shared four, so that
    # every title says what the series actually is:
    #
    #   "Episode Return"  is the sum of every reward in the episode, G = Sum r,
    #                     not the last reward. `session._log_episode` writes it
    #                     from a running total.
    #   "Smoothed Return" is a 20-episode trailing mean of that same series,
    #                     computed on the FULL history before any downsampling.
    #   "Loss / Mean |TD Error|" is mean(|delta|) over the episode, where delta
    #                     is the temporal-difference error of whichever method
    #                     is running. There is no neural network in this
    #                     project and this is NOT a network loss -- it is the
    #                     loss-like convergence diagnostic the method itself
    #                     produces. See `session._log_episode`.
    #   "Success rate"    is a 20-episode mean of a 0/1 indicator taken from
    #                     `info["goal"]`, not from a reward threshold.
    "charts": [
        {"key": "reward", "label": "Episode Return (total reward)"},
        {"key": "reward", "smoothOnly": True,
         "label": "Smoothed Return (20-episode moving average)"},
        {"key": "convergence", "label": "Loss / Mean |TD Error|"},
        {"key": "epsilon", "label": "Exploration rate \u03b5"},
        {"key": "steps", "label": "Steps per episode (physics steps of 0.02 s)"},
        {"key": "success", "smoothOnly": True,
         "label": "Success rate (20-episode moving average)"},
        {"key": "weightNorm", "label": "Weight norm \u2016w\u2016"},
    ],

    "metric": {"key": "meanReward", "label": "Mean reward", "format": "%.1f"},
    "episode_metric": "reward",
    "comparison": True,

    "info": {
        "story": "With the reactor back up, R-5 finds the ground corridor to "
                 "the final sector collapsed. The only way on is through the "
                 "aerospace bay: it couples itself to an experimental drone "
                 "frame and lifts off the launch platform into the wind "
                 "tunnel.",
        "objective": "Fly the drone frame across the tunnel and set it down on "
                     "the landing platform. Reaching the platform is not "
                     "enough — arriving faster than the landing speed on "
                     "either axis is a crash, not a landing.",
        "obstacles": [
            "There is no grid. The drone moves continuously and is steered by "
            "thrust, so an action changes where it is going rather than where "
            "it is — and it is felt several steps later.",
            "Speed is capped at 1 m/s on each axis, and the tunnel is 10 m "
            "across, so crossing it takes hundreds of 0.02 s steps.",
            "The chamber wall and the turbine housings are solid. Touching "
            "either ends the run.",
            "A bank of ventilation fans drives air down through the middle of "
            "the tunnel, across the last part of the approach.",
            "A second bank, wide and shallow, drives air down across the "
            "bottom right — directly over the turn from the floor onto the "
            "climb, so the climb has to be flown into a downdraught.",
            "A stabilisation field on the right is dense air: it costs time "
            "and makes a gentle arrival far easier. It is not on the short way.",
            "A thruster overcharge field on the left doubles the push and "
            "shortens the route, at 5 to enter — and it is much harder to "
            "arrive slowly out of it.",
        ],
        "actions": "Hold, or thrust up, down, left or right. A thrust steps the matching velocity component by one unit within {-1, 0, +1}; it is not a move from one square to the next. Two presses are needed to reverse a direction, which is what momentum means here."
                   "acceleration, not a move: nothing here steps from one "
                   "square to the next.",
        "rewards": [
            ["Each step", "-0.01"],
            ["Closing on the pad", "+5 per metre closed"],
            ["Drifting away from it", "-5 per metre lost"],
            ["Entering the overcharge", "-5, once per visit"],
            ["Hitting a wall or a pillar", "-100"],
            ["Reaching the pad too fast", "-30"],
            ["Landing gently", "+200"],
        ],
        "termination": "The run ends on a landing, on a crash-landing, "
                       "against a wall or a pillar, and otherwise when the "
                       "step limit is reached.",
        # STATED EXPLICITLY BECAUSE IT IS THE THING MOST OFTEN MISREAD.
        # The state is four continuous numbers. There is no angle in it and no
        # angular velocity: the tilt the renderer draws is atan2(vy, vx), a
        # display value computed from the velocity for the picture's sake, and
        # the learner never sees it.
        "state": "(x, y, vx, vy). x and y are the position in metres and are "
                 "continuous. vx and vy are the velocity components and are "
                 "DISCRETE: each is one of -1, 0 or +1 metres per second. The "
                 "world advances in steps of dt = 0.02 s, so the position "
                 "moves a fraction of a metre per tick and the flight path is "
                 "smooth even though the velocity is not. There is no angle "
                 "and no angular velocity in the state \u2014 the tilt drawn "
                 "on screen is atan2(vy, vx), a display value only.",
        "note": "This is the room where a table stops working. The state is "
                "four real numbers, so there are infinitely many states and "
                "no row can be kept for each; two states differing in the "
                "sixth decimal are the same situation and must not be learned "
                "about separately. The actions are discrete \u2014 hold, up, "
                "down, left, right \u2014 but each one applies an "
                "acceleration rather than moving the drone to the next "
                "square: thrust changes the velocity, drag decays it, and the "
                "position is carried by the new velocity. The two "
                "function-approximation methods cover the space with eight "
                "overlapping grids of tiles and learn a weight per tile, so "
                "what is learned about one place carries to the places around "
                "it. The third method rounds the state into buckets and uses "
                "an ordinary table; it is here to be compared against, and "
                "the bucket count is a slider so that both ways of failing "
                "can be watched.",
    },
}

# ----------------------------------------------------------------------
# Room 5 — the Adaptive Storage Facility
# ----------------------------------------------------------------------
#
# The final chamber, and the only one that is not a chamber: it is a *pool* of
# warehouses, and the agent gets a different one every episode.
#
# WHAT MAKES IT THE LAST ROOM
# Rooms 1 to 4 each have one layout. However hard they are, an agent that
# memorises the one map has solved them, and nothing it learns has to be worth
# anything anywhere else. Here the shelves move between episodes, the traffic
# patrols different routes, and the score that counts is measured on layouts
# that were never trained on. A memorised route is worth nothing by
# construction.
#
#     state       (x, y, vx, vy, stage, phase)   — what the world knows
#     observation 14 numbers, three of them range-limited sensor rays and one
#                 pair describing the nearest obstacle that passes the
#                 visibility rule            — what the agent knows
#
# The gap between those two lines is the room. See `game/warehouse.py`.
#
# WHAT COUNTS AS SEEING SOMETHING
# An obstacle is visible when the distance from the CENTRE of the agent to the
# CENTRE of the obstacle is at most `sensor_range` metres AND the obstacle's
# centre lies inside the forward cone of half-angle `sensor_spread`. Centre to
# centre exactly: the obstacle's radius is never subtracted from the range
# first. The rule lives in one method — `WarehouseWorld.visible_obstacles` —
# and the observation, the ray solver and the drawing all go through it.
#
# THE MISSION IS IN TWO STAGES
# The blast door is shut until the control terminal has been reached, so the
# room cannot be finished by flying at the exit. Stage 0 is "find the
# terminal", stage 1 is "now leave", and the two get separate blocks of weights
# because they are different tasks with different objectives.
#
# THE SPLIT IS THE MEASUREMENT
# Three disjoint pools of seeds. Training samples from the first, and the
# numbers that matter come from the third — which no episode ever trains on.
# `WarehouseWorld` raises if they overlap, because a test layout that was
# trained on is not an unseen layout and reporting it as one would be the
# single most misleading thing this project could do.

ROOM5 = {
    "key": "room5",
    "number": 5,
    "built": True,
    "name": "Adaptive Storage Facility",
    "sector": "LAB SECTOR E-12",
    "subtitle": "AUTONOMOUS GENERALISATION TEST",

    "world": "warehouse",

    # The last chamber there is. The room screen reads this to know that
    # escaping here ends the story rather than unlocking the next door, and
    # shows the ending sequence instead of returning to the chamber select.
    # A flag on the room rather than a number compared in the interface, so
    # "which room is last" stays a fact about the rooms.
    "final": True,

    "size": (10.0, 10.0),
    "dt": 0.02,
    # Room 4's flight model, unchanged in kind — R-5 stays in the drone frame,
    # which is both the narrative continuity and the reason the physics is
    # already known to work.
    "thrust": 3.0,
    "drag": 0.7,
    "speed_limit": 1.0,
    # How close counts as reaching an objective. Generous on purpose: measured
    # below, a naive direct-to-target controller escaped 1 layout in 12 at the
    # original 0.55, and a room whose objective a perfect controller misses is
    # not a hard room.
    "reach": 0.85,

    # How far the sensors see, in metres. The room's central parameter: the
    # whole difficulty is that this is not 10.
    "sensor_range_default": 3.0,
    "sensor_spread": 0.55,

    # How much furniture a generated warehouse gets.
    #
    # Cut from 4 shelves and 3 carts, against measurement. A direct-to-target
    # controller — which is roughly the best a reactive linear policy can hope
    # to become — escaped only 1 of 12 layouts at the old defaults, dying on a
    # shelf almost every time. A default the required agent cannot solve is not
    # a difficulty setting, it is a broken room. The sliders go up from here.
    "shelves": 3,
    "obstacles_default": 2,
    # Two diagonal beams, as in room 1. Kept to two for the same reason the
    # drones are: the room has to stay readable.
    "lasers": 2,

    # In METRES PER SECOND, and a real speed rather than the multiplier it used
    # to be. Measured before this changed: the drones patrolled at between 2.3
    # and 6.2 m/s against a drone capped at 1.0, and once a lap they jumped
    # 2.2 m in a single 0.02 s tick. Nothing a policy could learn would avoid
    # that, and the 92% collision rate said so.
    #
    # 0.35 m/s is roughly a third of R-5's top speed: slow enough to watch a
    # circuit and time a crossing, quick enough that the timing is a real
    # constraint. A drone crosses its own width in about 1.4 seconds.
    "obstacle_speed_default": 0.35,

    # How much the number of drones varies between warehouses, either side of
    # the setting above. The assignment requires the *quantity* to be dynamic
    # and not only the positions, so at the default each layout draws one, two
    # or three of them. 0 pins it, which is what makes the variation testable
    # in both directions.
    "obstacle_variation": 1,

    # The three pools, as [from, to) ranges. Disjoint, and checked on build.
    # 120 training layouts, not 40. Measured: the generalisation gap closes
    # monotonically as the training pool grows, and at 40 the agent was plainly
    # memorising them.
    #
    #     layouts   train escape   test escape   gap
    #        10          20%            5%       +15
    #        40          30%            5%       +25
    #       120          18%           10%        +8
    #       300           5%           10%        -5
    #
    # At 300 the gap inverts — test beats train — which is what a policy that
    # has stopped memorising looks like, but each layout is then seen so few
    # times that both numbers fall. 120 is where variety is enough to transfer
    # and each warehouse is still seen often enough to learn from.
    "train_seeds": (1000, 1120),
    "validation_seeds": (2000, 2010),
    "test_seeds": (3000, 3020),

    # Per tick of physics, not per decision, so the step cost does not change
    # if `action_repeat` does.
    #
    # THE NUMBERS ARE SET SO THAT TRYING BEATS STANDING STILL
    # The first version had no timeout penalty at all, and it taught the agent
    # to hover. Worked out arithmetically, on the old numbers:
    #
    #     hover the whole episode      -30.0
    #     set off and crash halfway    -88.0
    #     complete the mission        +302.0
    #
    # Hovering beat crashing by 58, so an agent that could not yet complete the
    # mission was *correct* to stand still, and epsilon-greedy over 300
    # decisions essentially never stumbles through the whole mission to find
    # the +302. It was a reward-design bug wearing the costume of an algorithm
    # bug: 0% terminal, 5% collisions, and a policy that had learned to survive
    # by not moving.
    #
    # On these numbers:
    #
    #     hover the whole episode     -110.0   (step cost plus the timeout)
    #     set off and crash halfway    -48.0
    #     complete the mission        +382.0
    #
    # so setting off is worth more than standing still even when the chance of
    # success is zero, which is what gives exploration somewhere to start from.
    # Collisions are still much the worst thing that can happen on the way.
    "rewards": {"step": -0.02, "progress": 4.0, "static": -50.0,
                "dynamic": -60.0, "boundary": -50.0, "terminal": 80.0,
                "locked": -5.0, "goal": 250.0, "timeout": -80.0,
                "laser": -60.0},

    # The assignment asks for feature-based approximation here, and semi-
    # gradient Q-Learning is what it names. Both are offered: this room is a
    # far better test of the on-policy/off-policy difference than room 2's map
    # managed, because a cart really does kill and exploration really does
    # reach one.
    "algorithms": ["semi_gradient_q", "semi_gradient_sarsa"],
    "algorithm_default": "semi_gradient_q",

    # Everything the assignment asks to be under the user's control, in three
    # groups: the learning method, the run, and the warehouse itself.
    "parameters": [
        # the method
        "alpha", "gamma", "epsilon", "epsilon_min", "epsilon_decay", "tilings",
        # the run
        "episodes", "max_steps", "seed",
        # the warehouse
        "sensor_range", "obstacles", "obstacle_variation", "obstacle_speed",
        "shelves",
        # the split
        "train_layouts", "validation_layouts", "test_layouts",
    ],

    # gamma 0.95, not 0.99. Measured: at 0.99 semi-gradient Q-Learning diverges
    # here exactly as it does in room 4 — Q(start) reached +1200 against a true
    # value of at most ~250, the weight norm passed 3600 and the mean |TD error|
    # was 67. At 0.95 it is stable (Q(start) ~10, |TD| ~1). Off-policy
    # bootstrapping with an approximator has no convergence guarantee, and this
    # is the room's second demonstration of it.
    "parameter_defaults": {"gamma": 0.97, "alpha": 0.20, "epsilon_min": 0.05,
                           "epsilon_decay": 0.9985, "episodes": 4000},

    # One decision is held for TEN ticks of physics. The tick is dt = 0.02 s,
    # so a decision lasts 10 x 0.02 = 0.2 s. See `WarehouseWorld.step` for the
    # measurement that forced holding an action at all.
    "action_repeat": 10,

    # A share of training episodes start at the terminal in stage 1, so the
    # second half of the mission gets practised instead of starving. Training
    # only — evaluation always starts at the real start with the door shut.
    # See `WarehouseWorld.reset`.
    "stage_one_share": 0.4,

    # In *decisions*, not ticks. 300 decisions x 10 ticks x 0.02 s = 3000
    # ticks and 60 seconds of simulated flight.
    "max_steps": 300,

    # ------------------------------------------------------------------
    # HOW FAST THIS ROOM IS *WATCHED*. Neither number touches the physics.
    # ------------------------------------------------------------------
    #
    # A recorded frame here is one decision — `action_repeat` ticks of dt —
    # which is 10 * 0.02 = 0.2 seconds of flight. So replaying at N frames a
    # second shows the flight at N * 0.2 times real speed, and the honest
    # figure for real time is 1 / 0.2 = 5.
    #
    # It was inheriting the shared 50, which is right for room 4 (whose frame
    # is one tick, so 50 a second is real time) and ten times too fast here.
    # Measured on a 1440x900 window: R-5 crossed the warehouse at about 830
    # pixels a second, which is faster than the eye tracks — the direction it
    # chose, the obstacle it dodged and the moment it reached the terminal all
    # went past in well under a second.
    #
    # 5.0 is exactly real time: R-5 at its 1 m/s cap crosses the 10 m warehouse
    # in ten seconds, at about 83 pixels a second on the same window. The speed
    # chips scale it from there, so Slow is a third of that and Fast about three
    # times it.
    "replay_steps_per_second": 5.0,

    # How many decisions one unit of the interface's speed tiers is worth while
    # training is being watched. At the Normal tier of 9, a scale of 1 gives 9
    # decisions a second — 1.8 times real time, which still reads clearly and
    # keeps a watched run making visible progress. It was 3, which put live
    # training at 5.4 times real speed.
    #
    # Turbo is unaffected by either number and remains the way to train a full
    # run quickly; these two are only about what can be followed by eye.
    "live_step_scale": 1,

    # How often training stops to measure the frozen policy on held-out
    # layouts, and over how many of them. This is what draws the train /
    # validation / unseen-test curve the assignment asks for; no weight is
    # updated by it — see `Session._checkpoint`. Eight layouts is noisy per
    # point and cheap enough to take sixteen times over a run, which is the
    # right trade for a trend line.
    "checkpoint_every": 250,
    "checkpoint_layouts": 8,

    # Fewer than the shared 40. A frame here carries every shelf, drone, beam
    # and beacon in a generated warehouse plus the sensor detail, and an
    # episode is up to 300 of them — measured at 40 episodes the batch came to
    # 3.2 MB, which the screen then refetched every two seconds during
    # training. 24 is enough to browse and a third of the weight.
    "episodes_recorded": 24,

    # The graphs this room puts up, in order. Read by `definition.build` and
    # drawn by `charts.js`; a room that names none gets the shared four.
    #
    # The 0/1 series are rates: each chart draws the raw series faintly and a
    # rolling average over it in the accent colour, so an indicator that is 1
    # on escapes and 0 otherwise becomes an escape *rate* without anything
    # having to smooth it first.
    "charts": [
        {"key": "reward", "label": "Episode Return (total reward)"},
        {"key": "reward", "smoothOnly": True,
         "label": "Smoothed Return (20-episode moving average)"},
        {"key": "steps",
         "label": "Decisions per episode (0.2 s each)"},
        {"key": "success", "label": "Complete escape rate"},
        {"key": "terminalReached", "label": "Terminal activation rate"},
        {"key": "collision", "label": "Collision rate"},
        {"key": "timeout", "label": "Timeout rate"},
        {"key": "epsilon", "label": "Exploration rate ε"},
                {"key": "convergence", "label": "Loss / Mean |TD Error|"},
        {"key": "weightNorm", "label": "Weight norm ‖w‖"},
        # The one series that does not come from the training episodes: three
        # lines from the periodic held-out checkpoints.
        {"keys": ["train", "validation", "test"], "source": "checkpoints",
         "x": "episode",
         "label": "Escape rate — train vs validation vs unseen test"},
    ],

    "metric": {"key": "meanReward", "label": "Mean reward", "format": "%.1f"},
    "episode_metric": "reward",
    "comparison": True,

    # What the screen calls the objective at each mission stage, indexed by
    # `stage`. Kept here rather than in the interface for the same reason every
    # other word about a room is: the room owns what it is asking for, and a
    # screen that wrote these itself would be free to disagree with the state
    # the environment is actually rewarding against.
    "objectives": ["OBJECTIVE: REACH CONTROL TERMINAL",
                   "OBJECTIVE: REACH UNLOCKED EXIT"],

    "info": {
        "story": "Past the wind tunnel is the last sector: an automated store "
                 "that rearranges itself between missions. The laboratory "
                 "built it to settle one question — whether R-5 learned to "
                 "adapt, or only memorised four rooms.",
        "objective": "Reach the central control terminal to disarm the "
                     "adaptive security system, then leave through the blast "
                     "door. The door is shut until the terminal is reached.",
        "obstacles": [
            "The warehouse is different every episode. The shelves move, the "
            "objectives move, the number of security drones changes, and the "
            "route that worked last time is not the route this time.",
            "R-5 cannot see the map. It has a forward sensor cone — three "
            "rays reaching the sensor range and no further — so an obstacle "
            "behind a shelf, or simply off to one side, does not exist until "
            "it does.",
            "A security drone is visible only when its centre is within the "
            "sensor range of R-5's centre AND inside that forward cone. "
            "Nothing outside both is in the observation at all.",
            "Security drones are half a metre wide and patrol closed square "
            "circuits at about a third of R-5's top speed. Touching one ends "
            "the run.",
            "Two diagonal security beams cut across the floor while the "
            "system is armed. They go dark the moment the terminal is "
            "reached, which is what disarming it means.",
            "The storage shelves are solid, and so is the chamber wall.",
            "The blast door does not open until the terminal has been "
            "reached, and trying it early costs something.",
        ],
        "actions": "Hold, or thrust up, down, left or right — the same drone "
                   "frame as the wind tunnel. Thrust is an acceleration, not "
                   "a move.",
        # These figures are the `rewards` table above, written out. They were
        # wrong in every row for a while — the table said 80 and this said 50 —
        # which is worse than saying nothing, because the sidebar is where
        # anyone reads the rules from.
        "rewards": [
            ["Each step", "-0.02"],
            ["Closing on the current objective", "+4 per metre closed"],
            ["Drifting away from it", "-4 per metre lost"],
            ["Reaching the control terminal", "+80, once and once only"],
            ["Trying the blast door while it is locked", "-5"],
            ["Hitting a shelf", "-50"],
            ["Hitting the chamber wall", "-50"],
            ["Hitting a security drone", "-60"],
            ["Touching an armed security beam", "-60"],
            ["Running out of time", "-80"],
            ["Escaping through the blast door", "+250"],
        ],
        "termination": "The run ends on an escape, on any collision, and on "
                       "running out of time — which is charged for, so that "
                       "hovering is the worst outcome rather than the safest.",
        "note": "This room measures generalisation, not skill at one map. "
                "Training samples 120 warehouse layouts; the score that counts "
                "is measured on 20 more that were never trained on, and the "
                "two pools are checked to be disjoint. The agent is given "
                "fourteen numbers — where it is, how fast, the direction and "
                "distance to its objective, three range-limited sensor rays, "
                "and how near the closest visible drone is and how fast it is "
                "closing — and never the layout. Two different warehouses can "
                "produce identical numbers, so the observation is not Markov "
                "even though the world is: that is partial observability, and "
                "it is why a linear model over local features can transfer "
                "between layouts at all.",
    },
}

# ----------------------------------------------------------------------
# The rest, named but not built
# ----------------------------------------------------------------------

PLACEHOLDERS = []

ROOMS = {ROOM1["number"]: ROOM1, ROOM2["number"]: ROOM2,
         ROOM3["number"]: ROOM3, ROOM4["number"]: ROOM4,
         ROOM5["number"]: ROOM5}
for placeholder in PLACEHOLDERS:
    ROOMS[placeholder["number"]] = placeholder

ROOM_NUMBERS = sorted(ROOMS.keys())


def room(number):
    if number not in ROOMS:
        raise KeyError("no room numbered %r" % number)
    return ROOMS[number]


def is_built(number):
    return bool(ROOMS.get(number, {}).get("built"))
