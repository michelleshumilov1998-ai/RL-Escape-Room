"""Every tunable value in the game, in one place.

This is the single config object the whole application reads.  The browser
does not keep its own copy: `serve.py` hands this out as JSON at
`/api/config`, so a speed tier or a parameter range is written down exactly
once and the two halves of the application can never disagree about it.

The one thing that is *not* here is the palette.  Colours live in the
stylesheet as CSS custom properties, because that is where the rest of the
interface already reads them from; what lives here instead is which
property each entity uses (`ENTITIES` below), which is the part the
simulation actually owns.
"""

# ----------------------------------------------------------------------
# Simulation rate
# ----------------------------------------------------------------------

# Speed tiers.  The three animated tiers are a fixed number of environment
# steps per rendered frame, so movement can be interpolated between cells.
# Turbo is different in kind: it runs as many steps as fit inside a time
# budget and renders only the result, because tabular methods need
# thousands of episodes and one step per frame would never get there.
SPEEDS = {
    "slow": {"label": "Slow", "steps_per_frame": 1, "animated": True},
    "normal": {"label": "Normal", "steps_per_frame": 4, "animated": True},
    "fast": {"label": "Fast", "steps_per_frame": 16, "animated": True},
    "turbo": {"label": "Turbo", "steps_per_frame": None, "animated": False},
}

SPEED_DEFAULT = "normal"

# A planner is a different problem from a learner. Room 1 converges in
# under thirty sweeps, so "one sweep per frame" would be over in half a
# second and the speed control would mean nothing. The animated tiers
# therefore pace a planner in sweeps per *second*, which is slow enough to
# watch value spread outwards from the goal.
PLANNER_SWEEPS_PER_SECOND = {"slow": 2.0, "normal": 8.0, "fast": 30.0}

# Turbo's budget, in milliseconds of simulation per request.  Kept well
# under a frame so the interface never stops answering.
TURBO_BUDGET_MS = 12.0

# A hard ceiling on one batch regardless of the clock, so a pathologically
# fast machine cannot run away with the episode counter between renders.
TURBO_MAX_STEPS = 40000

# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

RENDER = {
    # The grid is scaled to fit whatever space it has; this is only the
    # size it is drawn at before that scaling, and the floor of it.
    "cell_size": 44,
    "cell_size_min": 14,
    "grid_padding": 28,
    # How long an agent takes to slide from one cell to the next, as a
    # share of the time between steps. 1.0 means it is always moving.
    "move_tween": 0.85,
    "replay_steps_per_second": 4.0,
    "replay_pause_seconds": 1.1,
    "sidebar_ms": 200,
}

# ----------------------------------------------------------------------
# What can appear on a grid
# ----------------------------------------------------------------------

# One definition per entity: the renderer draws from this and the legend is
# generated from it, so a new entity cannot appear on the canvas without
# also appearing in the legend.
#
#   colour  the CSS custom property holding its colour
#   shape   which drawing recipe the renderer uses.  A thing on the grid is
#           meant to look like what it is rather than like a coloured
#           square, and this is where a room says which it is: the
#           browser's `shapes.js` knows how to draw a laser, and this file
#           decides that a hazard in this chamber *is* one.  The recipes
#           are shared with the room screen, so a wall looks the same in
#           every room.
#   role    what kind of thing it is, out of "static", "hazard", "goal" and
#           "agent".  The renderer draws in role order and skips the agent's
#           own entity, so it never has to recognise particular type names to
#           work out what goes underneath what.
ENTITIES = {
    "floor": {
        "label": "Floor", "colour": "--cell-floor", "shape": "floor",
        "in_legend": False, "role": "static",
    },
    "wall": {
        "label": "Wall", "colour": "--cell-wall", "shape": "wall",
        "in_legend": True, "role": "static",
    },
    "slippery": {
        "label": "Ice", "colour": "--cell-slippery", "shape": "ice",
        "in_legend": True, "role": "static",
    },
    # This chamber's hazard is a laser grid, and the beams are what the
    # room is named after, so they are drawn as beams with emitters rather
    # than as red cells.
    "hazard": {
        "label": "Laser", "colour": "--hazard", "shape": "laser",
        "in_legend": True, "role": "hazard",
    },
    # Room 2's hazard, and a different kind of thing from room 1's beam: it
    # ends the run rather than sending the agent back to the door, so it is
    # drawn as a hole in the floor rather than as something switched on.
    "pit": {
        "label": "Shaft", "colour": "--hazard", "shape": "abyss",
        "in_legend": True, "role": "hazard",
    },
    # PURELY DECORATIVE, AND NOT A TILE AT ALL.
    #
    # Room 2's middle is walled off on three sides and open only downwards
    # onto the planks: to the environment it is ordinary floor, and a step up
    # off a plank lands on it, costs -1 and leads nowhere. It rendered as a
    # flat black hole in the room, which was the weakest thing on the screen
    # and said nothing about what the space is.
    #
    # This draws it as what the fiction says it is: a deep maintenance void
    # with the machinery visible a long way down — SEEN THROUGH A STEEL
    # GRATING, so it reads as depth without ever suggesting the agent would
    # fall through it. That distinction is the whole reason it is not simply
    # drawn as the `pit` above: a pit ends the run and this does not.
    #
    # It reaches the screen through `room["decor"]`, which only
    # `definition.py` reads. No tile, transition, reward or state is involved.
    "void": {
        "label": "Maintenance void", "colour": "--cell-wall",
        "shape": "maintenanceVoid", "in_legend": True, "role": "static",
    },
    # A sound span over the shaft. Behaves exactly like floor; it is a tile of
    # its own only so it can be drawn as a bridge rather than as ground.
    "bridge": {
        "label": "Bridge", "colour": "--cell-start", "shape": "bridge",
        "in_legend": True, "role": "static",
    },
    # A plank that gives way once it has been crossed. `states` is how the
    # room says what "collapsed" looks like — the renderer treats the word as
    # opaque and only uses it to pick an appearance, so the drawing code never
    # has to know what collapsing means.
    "collapsing": {
        "label": "Collapsing plank", "colour": "--cell-slippery",
        # `bridgeWeak` rather than `bridge`: the same steel gantry with hazard
        # chevrons and a stress crack on it, so a section that may give way is
        # tellable from a sound one at a glance. Which is which is the whole
        # decision this chamber is about, and a tint alone did not carry it.
        "shape": "bridgeWeak", "in_legend": True, "role": "static",
        "states": {
            "collapsed": {"shape": "bridgeBroken", "colour": "--hazard"},
            # Named explicitly so an episode can declare every plank intact on
            # its first frame. A replay is watched on its own and must not
            # inherit the wreckage of the episode before it.
            "sound": {"shape": "bridgeWeak", "colour": "--cell-slippery"},
        },
    },
    # Room 3's furniture.
    # A generator is white while it is off and green once it is running, so
    # progress through the sequence is readable off the chamber itself rather
    # than only from the sidebar.
    "generator": {
        "label": "Generator", "colour": "--generator-off",
        "shape": "generator", "in_legend": True, "role": "static",
        "states": {
            "off": {"shape": "generator", "colour": "--generator-off"},
            "on": {"shape": "generator", "colour": "--goal"},
        },
    },
    # A door on a cycle. Which of the two it is showing comes from the step,
    # so the room says what each looks like and the renderer stays ignorant.
    "sliding": {
        "label": "Sliding door", "colour": "--cell-start", "shape": "door",
        "in_legend": True, "role": "static",
        "states": {
            "open": {"shape": "door", "colour": "--cell-start"},
            "shut": {"shape": "wall", "colour": "--cell-wall"},
        },
    },
    "key": {
        "label": "Generator key", "colour": "--accent", "shape": "key",
        "in_legend": True, "role": "static",
        # A key in hand is no longer lying on the floor.
        "states": {
            "there": {"shape": "key", "colour": "--accent"},
            "taken": {"shape": "floor", "colour": "--cell-floor"},
        },
    },
    "blast": {
        "label": "Reactor door", "colour": "--goal", "shape": "door",
        "in_legend": True, "role": "static",
    },
    # Not a tile: it walks a patrol, so it is one entity that is given a new
    # position every step rather than a square of the map.
    "guard": {
        "label": "Security robot", "colour": "--hazard", "shape": "guard",
        "in_legend": True, "role": "hazard",
    },
    "start": {
        "label": "Start", "colour": "--cell-start", "shape": "start",
        "in_legend": True, "role": "static",
    },
    "goal": {
        "label": "Control panel", "colour": "--goal", "shape": "exit",
        "in_legend": True, "role": "goal",
    },
    "agent": {
        "label": "R-5", "colour": "--accent", "shape": "agent",
        "in_legend": True, "role": "agent",
    },

    # The rest of room 1's floor. Each is a different way of not doing what
    # it was told, which is the part of the model worth being able to see.
    "cracked": {
        "label": "Cracked ice", "colour": "--cell-slippery", "shape": "ice",
        "in_legend": True, "role": "static",
    },
    "oil": {
        "label": "Oil", "colour": "--cell-wall", "shape": "oil",
        "in_legend": True, "role": "static",
    },
    "battery": {
        "label": "Battery", "colour": "--goal", "shape": "battery",
        "in_legend": True, "role": "static",
    },
    # "Pad" on its own said nothing about what the thing does, and room 4 has
    # a landing platform that is also a pad. The legend is the only place the
    # teleport is ever explained, so it says what it is.
    "teleport": {
        "label": "Teleport pad", "colour": "--accent", "shape": "teleport",
        "in_legend": True, "role": "static",
    },
    "oneway": {
        "label": "One-way door", "colour": "--cell-start", "shape": "oneway",
        "in_legend": True, "role": "static",
    },

    # ---- room 4's furniture -------------------------------------------
    # None of these is a tile. The chamber has no grid, so the room hands its
    # entities over directly (`DroneWorld.entities`) and each carries a real
    # size in metres rather than being one cell square.
    #
    # Three of the five are things the model does: the landing platform, the
    # turbine housings and the two fields. The other two — the chamber shell
    # and the fans — are drawn explanations of rules that are already there,
    # and neither applies a force of its own.
    "pad": {
        "label": "Landing platform", "colour": "--goal", "shape": "platform",
        "in_legend": True, "role": "goal",
        # What the platform looks like as the approach goes right or wrong. The
        # words are opaque to the renderer, exactly as room 2's planks are: the
        # room says what each state looks like and the drawing code only picks.
        # Only the environment ever reports one of these — see `frame_extras`.
        "states": {
            "clear": {"shape": "platform", "colour": "--goal"},
            "fast": {"shape": "platformWarning", "colour": "--hazard"},
            "landed": {"shape": "platformLanded", "colour": "--goal"},
            "crashed": {"shape": "platformCrashed", "colour": "--hazard"},
        },
    },
    "pillar": {
        "label": "Turbine housing", "colour": "--cell-wall", "shape": "turbine",
        "in_legend": True, "role": "static",
    },
    # The wind band. `stream` draws a lane of chevrons pointing downwind; which
    # way that is comes over on the entity as `blows`, worked out from the wind
    # vector in `drone.py` so the two halves cannot disagree about it.
    "wind": {
        "label": "Airflow", "colour": "--cell-slippery", "shape": "stream",
        "in_legend": True, "role": "static",
    },
    "slow": {
        "label": "Stabilisation field", "colour": "--cell-start",
        "shape": "stabiliser", "in_legend": True, "role": "static",
    },
    "boost": {
        "label": "Thruster overcharge", "colour": "--hazard",
        "shape": "overcharge", "in_legend": True, "role": "hazard",
    },
    # An industrial fan at the mouth of a wind zone. Decorative: the zone
    # applies the force, and this is the visible reason it exists.
    "fan": {
        "label": "Ventilation fan", "colour": "--cell-wall", "shape": "fan",
        "in_legend": True, "role": "static",
    },
    # No `tunnelWall` any more. Rooms 4 and 5 were enclosed by one entity
    # spanning the whole chamber whose recipe drew a thin stroked frame with
    # corner brackets and measurement ticks — a pressure vessel, and the single
    # thing that most made them look like a different game from rooms 1 to 3.
    # They now state a ring of ordinary `wall` tiles instead, so the masonry,
    # the legend entry and the brick coursing are literally the grid rooms'.
    # See `chamber_wall_entities` in `game/drone.py`.
    "warningLight": {
        "label": "Status lamp", "colour": "--hazard", "shape": "warningLight",
        "in_legend": False, "role": "static",
    },

    # ---- room 5's furniture -------------------------------------------
    # The warehouse. Every one of these has real collision geometry except the
    # chamber shell, which is reused from room 4 and draws the boundary the
    # environment actually ends a run at.
    "shelf": {
        "label": "Storage shelf", "colour": "--cell-wall", "shape": "shelf",
        "in_legend": True, "role": "static",
    },
    # A patrolling cart: not a tile and not fixed either. Declared once at the
    # start of its patrol and given a new position by every recorded step,
    # exactly as room 3's guard is.
    "cart": {
        "label": "Security drone", "colour": "--hazard", "shape": "cart",
        "in_legend": True, "role": "hazard",
    },
    # A diagonal beam, back from room 1. Lit or dark, and the room says which:
    # dark beams are drawn as an unpowered emitter line so the player can see
    # where one is about to appear, which is what makes the timing learnable
    # rather than a surprise.
    "laser": {
        "label": "Security beam", "colour": "--hazard", "shape": "laser",
        "in_legend": True, "role": "hazard",
        "states": {
            "on": {"shape": "laser", "colour": "--hazard"},
            "off": {"shape": "laserIdle", "colour": "--cell-wall"},
        },
    },

    # ---- the working warehouse -----------------------------------------
    # None of the three below has any collision geometry, appears in the
    # observation, or is read by a single line of `step`. They are here because
    # a warehouse with two enemies in it and nothing else moving reads as a
    # room, not as a functioning building. `role: static` and out of the hazard
    # legend, so nothing about them suggests they can hurt you.
    "conveyor": {
        "label": "Cargo line", "colour": "--cell-start", "shape": "conveyor",
        "in_legend": True, "role": "static",
    },
    "crate": {
        "label": "Cargo", "colour": "--cell-wall", "shape": "crate",
        "in_legend": False, "role": "static",
    },
    "beacon": {
        "label": "Alarm beacon", "colour": "--hazard", "shape": "beacon",
        "in_legend": False, "role": "static",
        "states": {
            "alarm": {"shape": "beacon", "colour": "--hazard"},
            "clear": {"shape": "beacon", "colour": "--goal"},
            "dim": {"shape": "beaconDim", "colour": "--cell-wall"},
        },
    },
    # The first objective. Armed until it is reached, then spent — and the
    # environment is the only thing that says which.
    "terminal": {
        "label": "Control terminal", "colour": "--accent", "shape": "terminal",
        "in_legend": True, "role": "goal",
        "states": {
            "armed": {"shape": "terminal", "colour": "--accent"},
            "used": {"shape": "terminalUsed", "colour": "--goal"},
        },
    },
    # The way out, shut until the terminal has been reached.
    "blastdoor": {
        "label": "Blast door", "colour": "--goal", "shape": "blastdoor",
        "in_legend": True, "role": "goal",
        "states": {
            "locked": {"shape": "blastdoorLocked", "colour": "--hazard"},
            "open": {"shape": "blastdoor", "colour": "--goal"},
        },
    },
}

# ----------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------

# Every parameter the interface may expose, with the range it is valid in,
# the value it resets to, whether it can be changed mid-run, and one line
# of plain English.
#
#   scope   "live"  safe to change while training; applies at once
#           "reset" changing it invalidates what has been learned, so the
#                   run is marked stale until Reset is pressed
PARAMETERS = {
    "gamma": {
        "label": "Discount factor",
        "symbol": "γ",
        "minimum": 0.50, "maximum": 0.999, "step": 0.005, "default": 0.95,
        "scope": "reset",
        "explanation": "How much a reward in the future is worth now. Low "
                       "values make the agent impatient.",
    },
    "theta": {
        "label": "Stopping threshold",
        "symbol": "θ",
        "minimum": 1e-6, "maximum": 1e-1, "step": None, "default": 1e-4,
        "choices": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "scope": "live",
        "explanation": "A sweep that changes every value by less than this "
                       "counts as converged.",
    },
    "slip": {
        "label": "Ice slipperiness",
        "symbol": "",
        "minimum": 0.0, "maximum": 0.5, "step": 0.02, "default": 0.20,
        "scope": "reset",
        "explanation": "Chance of sliding sideways instead of going where "
                       "it aimed, when standing on ice.",
    },
    # A reward rather than a rate, and the only parameter that edits the model
    # itself: it changes what the reward table says, which is the thing a
    # planner reads instead of experience. Reset-scope for that reason — every
    # value already computed was computed against the old table.
    "battery_reward": {
        "label": "Battery bonus",
        "symbol": "",
        "minimum": 0.0, "maximum": 80.0, "step": 2.0, "default": 10.0,
        "scope": "reset",
        "explanation": "What collecting the battery is worth. Raise it far "
                       "enough and the detour to fetch it starts paying for "
                       "itself; the plan then goes out of its way.",
    },
    # Room 2's central control, and the one the chamber exists to demonstrate.
    #
    # THE NAME THE PLAYER SEES IS THE THING IT DOES
    # It was labelled "Plank failure chance", which is the mechanism rather
    # than the decision — the room is about whether to risk the bridge, so the
    # control is named after that. The backend field is unchanged
    # (`collapse_chance` -> `GridWorld.collapse`); only the label, the range
    # and the step moved. There is deliberately no second parameter.
    #
    # The full 0 to 1 range, because both ends are worth being able to see: at
    # 0.00 the bridge never gives way and the short route is simply correct, and
    # at 1.00 every plank fails on the step that lands on it, so the bridge is
    # certain death and the long way round is the only route. A slider that
    # stopped at 0.5 could show neither end.
    "collapse_chance": {
        "label": "Bridge collapse probability",
        "symbol": "",
        # The default is unchanged at 0.10 on purpose: re-ranging a control is
        # a presentation change, and moving its default would quietly alter
        # what the room does out of the box.
        "minimum": 0.0, "maximum": 1.0, "step": 0.05, "default": 0.10,
        "scope": "reset",
        # MEASURED, NOT ASSUMED. `lands_on_sound_plank` returns true only for
        # the step that moves the agent from off the span on to a plank, so the
        # draw is made once per crossing attempt. Verified over 2000 attempts
        # per setting: p=0.10 fails 0.101 of the time, p=0.25 fails 0.253,
        # p=0.50 fails 0.524 -- the slider value itself, not 1-(1-p)^4.
        "explanation": "The probability that one attempted bridge crossing "
                       "fails. The random draw is made once, when the agent "
                       "first enters the collapsing span. If the draw "
                       "succeeds, the crossing fails; otherwise the agent may "
                       "cross the entire span. At 0.10, each attempted "
                       "crossing has a 10% failure probability. Changing this "
                       "value changes the environment, so the current learned "
                       "policy must be discarded and retrained.",
    },
    "alpha": {
        "label": "Learning rate",
        "symbol": "α",
        "minimum": 0.01, "maximum": 1.0, "step": 0.01, "default": 0.10,
        "scope": "live",
        "explanation": "How much of each new estimate to keep.",
    },
    "epsilon": {
        "label": "Exploration rate",
        "symbol": "ε",
        "minimum": 0.0, "maximum": 1.0, "step": 0.01, "default": 1.00,
        "scope": "live",
        "explanation": "How often the agent takes a random action instead "
                       "of its best one.",
    },
    "epsilon_min": {
        "label": "Minimum exploration",
        "symbol": "",
        "minimum": 0.0, "maximum": 0.5, "step": 0.01, "default": 0.05,
        "scope": "live",
        "explanation": "Exploration never decays below this.",
    },
    "q_init": {
        "label": "Initial Q value",
        "symbol": "",
        "minimum": 0.0, "maximum": 150.0, "step": 5.0, "default": 0.0,
        "scope": "reset",
        "explanation": "What every untried action is worth to begin with. "
                       "Setting it high makes the agent try everything once "
                       "before it settles on a favourite.",
    },
    "episodes": {
        "label": "Episodes to train",
        "symbol": "",
        "minimum": 100, "maximum": 8000, "step": 100, "default": 1500,
        "scope": "live",
        "explanation": "How many attempts to learn from before the run is "
                       "counted as finished.",
    },
    "epsilon_decay": {
        "label": "Exploration decay",
        "symbol": "",
        "minimum": 0.90, "maximum": 1.0, "step": 0.001, "default": 0.995,
        "scope": "live",
        "explanation": "Exploration is multiplied by this after every "
                       "episode.",
    },

    # ---- room 4's own -------------------------------------------------
    # How many overlapping grids of tiles the function approximation covers
    # the state space with. More resolves position and speed more finely and
    # costs proportionally more memory; the step size is divided by this
    # number, so raising it does not also raise the effective learning rate.
    "tilings": {
        "label": "Tilings",
        "symbol": "",
        "minimum": 1, "maximum": 16, "step": 1, "default": 8,
        "scope": "reset",
        "explanation": "How many offset grids of tiles the state space is "
                       "covered with. One is a single coarse grid and "
                       "generalises badly; eight overlap so that what is "
                       "learned in one place carries to the places near it.",
    },
    # The control method's resolution, and the whole point of exposing it: the
    # discretised table fails at both ends of this slider, for opposite
    # reasons. See `algorithms/discretised_q.py`.
    "buckets": {
        "label": "Buckets per axis",
        "symbol": "",
        "minimum": 3, "maximum": 20, "step": 1, "default": 8,
        "scope": "reset",
        "explanation": "Only for the discretised table: how finely each axis "
                       "is rounded off. Low values cannot tell a gentle "
                       "approach from a fast one; high values never see the "
                       "same bucket twice.",
    },
    "wind": {
        "label": "Wind strength",
        "symbol": "",
        "minimum": 0.0, "maximum": 2.0, "step": 0.1, "default": 1.0,
        "scope": "reset",
        "explanation": "A multiplier on the wind band. Constant within a run "
                       "on purpose: wind that changed between episodes would "
                       "be a force the state cannot see, and the room would "
                       "stop being Markov.",
    },
    # ---- room 5's own -------------------------------------------------
    "sensor_range": {
        "label": "Sensor range",
        "symbol": "",
        "minimum": 1.0, "maximum": 10.0, "step": 0.5, "default": 3.0,
        "scope": "reset",
        "explanation": "How far ahead R-5 can see, in metres. The whole "
                       "difficulty of the room is that this is not 10: an "
                       "obstacle further off than this is not in the "
                       "observation at all. Raise it to the room's width and "
                       "the partial observability disappears.",
    },
    "obstacles": {
        "label": "Security drones",
        "symbol": "",
        "minimum": 0, "maximum": 6, "step": 1, "default": 2,
        "scope": "reset",
        "explanation": "How many patrolling drones a generated warehouse gets, "
                       "on average. Each layout draws its own count either "
                       "side of this, so the quantity varies between episodes "
                       "as well as the positions.",
    },
    "obstacle_variation": {
        "label": "Drone count variation",
        "symbol": "",
        "minimum": 0, "maximum": 3, "step": 1, "default": 1,
        "scope": "reset",
        "explanation": "How far the number of drones may vary either side of "
                       "the setting above, per warehouse. Zero pins every "
                       "layout to the same count.",
    },
    "obstacle_speed": {
        "label": "Drone speed",
        "symbol": "",
        "minimum": 0.1, "maximum": 1.5, "step": 0.05, "default": 0.35,
        "scope": "reset",
        "explanation": "How fast the drones patrol, in metres per second. R-5 "
                       "itself is capped at 1 m/s, so anything above that is "
                       "traffic it cannot outrun — which is a difficulty "
                       "setting and not a bug.",
    },
    "shelves": {
        "label": "Storage shelves",
        "symbol": "",
        "minimum": 0, "maximum": 8, "step": 1, "default": 3,
        "scope": "reset",
        "explanation": "How many solid shelf blocks each warehouse is "
                       "furnished with. More shelves means tighter corridors "
                       "and more layouts rejected as unflyable.",
    },
    "max_steps": {
        "label": "Episode length",
        "symbol": "",
        "minimum": 100, "maximum": 800, "step": 25, "default": 300,
        "scope": "reset",
        "explanation": "How many agent decisions an episode may last before "
                       "it times out. Each decision is held for 10 physics "
                       "ticks of 0.02 seconds, so one decision lasts 0.2 "
                       "seconds and 300 decisions represent at most 60 "
                       "seconds of simulated flight.",
    },
    "train_layouts": {
        "label": "Training layouts",
        "symbol": "",
        "minimum": 10, "maximum": 400, "step": 10, "default": 120,
        "scope": "reset",
        "explanation": "How many different warehouses training draws from. Too "
                       "few and the agent memorises them; too many and each is "
                       "seen too rarely to learn anything from.",
    },
    "validation_layouts": {
        "label": "Validation layouts",
        "symbol": "",
        "minimum": 5, "maximum": 100, "step": 5, "default": 10,
        "scope": "reset",
        "explanation": "How many warehouses are kept back to check parameters "
                       "against. Measured on, never trained on — and separate "
                       "from the test pool so that tuning against these does "
                       "not spend the final score.",
    },
    "seed": {
        "label": "Random seed",
        "symbol": "",
        "minimum": 0, "maximum": 999, "step": 1, "default": 0,
        "scope": "reset",
        "explanation": "What every random draw in the run starts from — which "
                       "warehouse comes up when, and which way exploration "
                       "goes. The same seed reproduces the whole run exactly.",
    },
    "test_layouts": {
        "label": "Unseen test layouts",
        "symbol": "",
        "minimum": 5, "maximum": 100, "step": 5, "default": 20,
        "scope": "reset",
        "explanation": "How many held-out warehouses the final score is "
                       "measured on. None of them is ever trained on.",
    },
    "landing_speed": {
        "label": "Landing speed limit",
        "symbol": "",
        "minimum": 0.05, "maximum": 1.5, "step": 0.05, "default": 1.0,
        "scope": "reset",
        # THREE REGIMES, BECAUSE THE VELOCITY HAS THREE VALUES.
        # The test is on the speed magnitude, and with Vx, Vy in {-1, 0, 1}
        # the only speeds that exist are 0, 1 and sqrt(2) ~ 1.41. So:
        #   below 1.0   only a full stop counts as a landing
        #   1.0 to 1.41 arriving along one axis lands, diagonally crashes
        #   above 1.41  any arrival lands
        # Measured over 1200 episodes: at 1.0 the agent lands 100% of the
        # time, at 1.5 it lands 98%, and below 1.0 it learns to hover instead
        # of risking the crash penalty and never lands at all.
        "explanation": "How fast the drone may be moving when it touches the "
                       "platform. The velocity components are discrete, so the "
                       "only speeds that exist are 0, 1 and about 1.41 "
                       "diagonally. Below 1.0 only a full stop counts as a "
                       "landing; from 1.0 an arrival along one axis lands and "
                       "a diagonal one crashes; above 1.41 any arrival lands.",
    },
}

# ----------------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------------

SESSION = {
    # An episode is abandoned after this many steps so a lost agent cannot
    # stall a run forever.
    "max_steps_per_episode": 400,
    # How many recent episodes the metrics average over.
    "metric_window": 50,
    # How many points the learning curve keeps. Longer runs are downsampled
    # into this many buckets rather than sending every episode to the page.
    "curve_points": 220,
    "seed_default": 0,

    # How many whole episodes are kept step by step, so a finished run can be
    # replayed rather than only summarised. Every episode contributes to the
    # graphs; this is the far smaller number that can be watched back.
    #
    # It is a budget rather than a preference. An episode early in a run is
    # ε-random and routinely hits the step limit, so the cost of one is closer
    # to `max_steps_per_episode` than to the length of a good route — and the
    # whole batch is sent to the page in one response.
    "episodes_recorded": 40,
}


def as_dict():
    """The whole config, ready to be handed to the browser as JSON."""
    return {
        "speeds": SPEEDS,
        "speedDefault": SPEED_DEFAULT,
        "plannerSweepsPerSecond": PLANNER_SWEEPS_PER_SECOND,
        "turboBudgetMs": TURBO_BUDGET_MS,
        "turboMaxSteps": TURBO_MAX_STEPS,
        "render": RENDER,
        "entities": ENTITIES,
        "parameters": PARAMETERS,
        "session": SESSION,
    }
