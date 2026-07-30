"""Directions and actions shared by every room of PROJECT R-5.

A "direction" is one of five values.  Four of them are real movements and one
of them (NONE) means "the robot has no movement history yet".

An "action" is what the agent chooses: always one of the four real movements.
A direction can also be the *result* of an action, which is not always the same
thing, because ice and oil can push the robot somewhere it did not intend to go.

The code is kept intentionally plain: plain integers, plain dictionaries, and
no classes.  Everything here should be readable by someone new to Python.
"""

# The five direction values.
NONE = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

# Standing still on purpose.  Rooms 3 and 5 need it, because they contain moving
# hazards and timed doors where waiting is sometimes the best thing to do.  It is
# deliberately NOT in ALL_DIRECTIONS: that list describes the direction a robot
# last *moved*, which Room 1 keeps in its state, and adding a sixth value there
# would change Room 1's state space for no reason.
WAIT = 5

# All five movement-history values, used when we build a state space.
ALL_DIRECTIONS = [NONE, UP, DOWN, LEFT, RIGHT]

# The four movement actions. Rooms 1 and 2 use exactly these.
ACTIONS = [UP, DOWN, LEFT, RIGHT]

# The same four plus waiting. Rooms 3 and 5 use these.
ACTIONS_WITH_WAIT = [UP, DOWN, LEFT, RIGHT, WAIT]

# How much each direction changes (row, col).
# Row 0 is the top row, so UP means row - 1.  Waiting changes nothing.
DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
    WAIT: (0, 0),
}

# The two directions "sideways" from a given direction.
# Slipping on ice always means sliding sideways instead of forward.
PERPENDICULAR = {
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (UP, DOWN),
}

# Short names, used in the interface and in the replay log.
DIRECTION_NAMES = {
    NONE: "NONE",
    UP: "UP",
    DOWN: "DOWN",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
    WAIT: "WAIT",
}

# Arrow characters, used for the policy view and the HUD.
DIRECTION_ARROWS = {
    NONE: "·",
    UP: "↑",
    DOWN: "↓",
    LEFT: "←",
    RIGHT: "→",
    WAIT: "◦",
}


def direction_name(direction):
    """Return the readable name of a direction, e.g. UP -> 'UP'."""
    return DIRECTION_NAMES[direction]


def direction_arrow(direction):
    """Return the arrow character of a direction, e.g. UP -> '^'."""
    return DIRECTION_ARROWS[direction]


def move(row, col, direction):
    """Return the cell you reach by taking one step in `direction`.

    This only does the arithmetic.  It does not know about walls, so the
    result may be outside the grid; the environment checks that.
    """
    delta_row, delta_col = DELTAS[direction]
    return row + delta_row, col + delta_col
