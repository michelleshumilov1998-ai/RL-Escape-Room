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
        "shape": "bridge", "in_legend": True, "role": "static",
        "states": {
            "collapsed": {"shape": "bridgeBroken", "colour": "--hazard"},
            # Named explicitly so an episode can declare every plank intact on
            # its first frame. A replay is watched on its own and must not
            # inherit the wreckage of the episode before it.
            "sound": {"shape": "bridge", "colour": "--cell-slippery"},
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
    "teleport": {
        "label": "Pad", "colour": "--accent", "shape": "teleport",
        "in_legend": True, "role": "static",
    },
    "oneway": {
        "label": "One-way door", "colour": "--cell-start", "shape": "oneway",
        "in_legend": True, "role": "static",
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
    "collapse_chance": {
        "label": "Plank failure chance",
        "symbol": "",
        "minimum": 0.0, "maximum": 0.5, "step": 0.01, "default": 0.10,
        "scope": "reset",
        "explanation": "Chance that a plank gives way under the step that "
                       "lands on it, which ends the run. Four planks at 0.10 "
                       "means about two crossings in three get across.",
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
