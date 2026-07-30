"""The geometry of Room 4 — the Drone Wind Tunnel.

Unlike Rooms 1 to 3 there is no grid here. The chamber is a continuous 10 x 10
metre hall, so it is described by shapes and force fields instead of a map of
characters:

  * **Fans** create the wind. Each one is a position, a direction, a strength and a
    falloff radius, and `wind_at` adds them all up.
  * **Obstacles** are circles and axis-aligned rectangles.
  * **The landing pad** is a position and a radius.

Everything here is deterministic and depends only on position, which is what keeps
the environment Markovian without needing a clock in the state.
"""

import math

# The hall, in metres.
WIDTH = 10.0
HEIGHT = 10.0

# Where the drone is released, and where it has to end up.
START_POSITION = (1.2, 1.2)
START_VELOCITY = (0.0, 0.0)
GOAL_POSITION = (8.6, 8.6)
GOAL_RADIUS = 0.6

# Nearby release points, used to check that the learned policy generalises rather
# than memorising one trajectory. The training start is never in the test list.
TRAINING_STARTS = ((1.2, 1.2),)
VALIDATION_STARTS = ((1.6, 1.0), (0.9, 1.7))
TEST_STARTS = ((2.0, 0.9), (0.8, 2.2), (1.9, 2.1), (0.7, 0.8), (2.4, 1.5))


class Fan:
    """One industrial fan, pushing air in a fixed direction.

    The force falls off smoothly with distance so the field has no hard edges for
    the tile coder to trip over:

        force = strength * max(0, 1 - distance / radius) ** 2
    """

    def __init__(self, name, x, y, direction_x, direction_y, strength, radius):
        self.name = name
        self.x = x
        self.y = y
        length = math.hypot(direction_x, direction_y) or 1.0
        self.direction_x = direction_x / length
        self.direction_y = direction_y / length
        self.strength = strength
        self.radius = radius

    def force_at(self, x, y):
        """The push this fan applies at a point."""
        distance = math.hypot(x - self.x, y - self.y)
        if distance >= self.radius:
            return 0.0, 0.0
        falloff = (1.0 - distance / self.radius) ** 2
        push = self.strength * falloff
        return self.direction_x * push, self.direction_y * push

    def as_dict(self):
        return {"name": self.name, "x": self.x, "y": self.y,
                "dx": self.direction_x, "dy": self.direction_y,
                "strength": self.strength, "radius": self.radius}


# The four fans in the hall. Between them they make the direct diagonal route
# across the middle expensive and reward going round the quieter edges.
FANS = (
    # A wall fan on the west side pushing the drone back the way it came.
    Fan("west intake", 0.0, 5.0, 1.0, 0.0, 1.5, 4.5),
    # The tunnel fan: a strong updraught up the middle of the hall.
    Fan("tunnel", 5.0, 4.2, 0.0, 1.0, 3.4, 3.0),
    # A downdraught guarding the approach to the landing pad.
    Fan("pad guard", 8.0, 6.4, 0.0, -1.0, 2.2, 2.6),
    # A crosswind that pushes sideways near the pad itself.
    Fan("crosswind", 7.2, 8.8, -1.0, 0.0, 1.6, 2.4),
)


class Circle:
    """A round obstacle — a turbine housing or a support column."""

    kind = "circle"

    def __init__(self, name, x, y, radius):
        self.name = name
        self.x = x
        self.y = y
        self.radius = radius

    def contains(self, x, y, margin=0.0):
        return math.hypot(x - self.x, y - self.y) < self.radius + margin

    def push_out(self, x, y, margin):
        """Move a point to the nearest spot outside the circle."""
        offset_x, offset_y = x - self.x, y - self.y
        distance = math.hypot(offset_x, offset_y) or 1e-9
        wanted = self.radius + margin
        scale = wanted / distance
        return self.x + offset_x * scale, self.y + offset_y * scale

    def normal_at(self, x, y):
        offset_x, offset_y = x - self.x, y - self.y
        length = math.hypot(offset_x, offset_y) or 1e-9
        return offset_x / length, offset_y / length

    def as_dict(self):
        return {"kind": "circle", "name": self.name, "x": self.x, "y": self.y,
                "radius": self.radius}


class Rectangle:
    """An axis-aligned obstacle — a maintenance platform or a hanging barrier."""

    kind = "rect"

    def __init__(self, name, x0, y0, x1, y1):
        self.name = name
        self.x0, self.y0 = min(x0, x1), min(y0, y1)
        self.x1, self.y1 = max(x0, x1), max(y0, y1)

    def contains(self, x, y, margin=0.0):
        return (self.x0 - margin < x < self.x1 + margin
                and self.y0 - margin < y < self.y1 + margin)

    def push_out(self, x, y, margin):
        """Move a point out through whichever face is nearest."""
        left = abs(x - (self.x0 - margin))
        right = abs((self.x1 + margin) - x)
        below = abs(y - (self.y0 - margin))
        above = abs((self.y1 + margin) - y)
        nearest = min(left, right, below, above)
        if nearest == left:
            return self.x0 - margin, y
        if nearest == right:
            return self.x1 + margin, y
        if nearest == below:
            return x, self.y0 - margin
        return x, self.y1 + margin

    def normal_at(self, x, y):
        left = abs(x - self.x0)
        right = abs(self.x1 - x)
        below = abs(y - self.y0)
        above = abs(self.y1 - y)
        nearest = min(left, right, below, above)
        if nearest == left:
            return -1.0, 0.0
        if nearest == right:
            return 1.0, 0.0
        if nearest == below:
            return 0.0, -1.0
        return 0.0, 1.0

    def as_dict(self):
        return {"kind": "rect", "name": self.name, "x0": self.x0, "y0": self.y0,
                "x1": self.x1, "y1": self.y1}


# The obstacles. Placed so the straight line from the release point to the pad is
# blocked, and so the drone has to commit to going above or below the middle.
OBSTACLES = (
    Circle("turbine housing", 4.6, 6.6, 0.85),
    Circle("support column", 6.4, 3.0, 0.7),
    Rectangle("maintenance platform", 2.6, 4.2, 4.0, 4.9),
    Rectangle("hanging barrier", 6.9, 5.6, 7.6, 7.0),
)


def wind_at(x, y, strength_multiplier=1.0):
    """The total wind force at a point, summed over every fan."""
    total_x = 0.0
    total_y = 0.0
    for fan in FANS:
        force_x, force_y = fan.force_at(x, y)
        total_x += force_x
        total_y += force_y
    return total_x * strength_multiplier, total_y * strength_multiplier


def blocking_obstacle(x, y, margin=0.0):
    """The first obstacle containing this point, or None."""
    for obstacle in OBSTACLES:
        if obstacle.contains(x, y, margin):
            return obstacle
    return None


def inside_bounds(x, y, margin=0.0):
    return margin <= x <= WIDTH - margin and margin <= y <= HEIGHT - margin


def distance_to_goal(x, y):
    return math.hypot(x - GOAL_POSITION[0], y - GOAL_POSITION[1])


def inside_goal(x, y):
    return distance_to_goal(x, y) <= GOAL_RADIUS


def start_is_clear():
    """The release points must not be inside anything."""
    for start in TRAINING_STARTS + VALIDATION_STARTS + TEST_STARTS:
        if blocking_obstacle(*start) is not None:
            return False
        if not inside_bounds(*start):
            return False
    return True


def geometry_as_dict(strength_multiplier=1.0, samples=13):
    """Everything the renderer needs, including a sampled wind field."""
    field = []
    for row in range(samples):
        for col in range(samples):
            x = WIDTH * (col + 0.5) / samples
            y = HEIGHT * (row + 0.5) / samples
            if blocking_obstacle(x, y) is not None:
                continue
            wind_x, wind_y = wind_at(x, y, strength_multiplier)
            if abs(wind_x) < 1e-3 and abs(wind_y) < 1e-3:
                continue
            field.append({"x": x, "y": y, "dx": wind_x, "dy": wind_y})

    return {
        "width": WIDTH, "height": HEIGHT,
        "start": list(START_POSITION),
        "goal": list(GOAL_POSITION), "goalRadius": GOAL_RADIUS,
        "fans": [fan.as_dict() for fan in FANS],
        "obstacles": [obstacle.as_dict() for obstacle in OBSTACLES],
        "windField": field,
    }
