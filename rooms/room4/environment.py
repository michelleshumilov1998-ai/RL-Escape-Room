"""The Room 4 environment — the Drone Wind Tunnel.

--------------------------------------------------------------------------
THE STATE
--------------------------------------------------------------------------
    (x, y, vx, vy)      all real-valued

This is the point of the room. There is no grid and no finite list of states, so
the Q-table the first three rooms relied on is not available: between any two
positions there is another one. R-5 has to *generalise* between similar states
instead, which is what `tile_coder.py` and `sarsa_agent.py` are for.

--------------------------------------------------------------------------
THE DYNAMICS
--------------------------------------------------------------------------
Thrust changes velocity, not position. Each step, in this order:

    wind_x, wind_y = wind_at(x, y)                  # depends only on position
    vx = (vx + thrust_x + wind_x + turbulence_x) * drag
    vy = (vy + thrust_y + wind_y + turbulence_y) * drag
    vx, vy = clip(vx, vy, v_max)
    x, y  = x + vx * dt, y + vy * dt                # walked in substeps

The move is walked in substeps so a fast drone cannot tunnel straight through a
thin obstacle between one frame and the next.

The wind depends only on where the drone is, never on the time, which is what
keeps the environment Markovian without a clock in the state. Turbulence is
optional, off by default, and drawn from a seeded generator so a replay of a
turbulent episode is still exact.

--------------------------------------------------------------------------
LANDING
--------------------------------------------------------------------------
Reaching the pad is not enough. A landing counts only when

    distance_to_goal <= goal_radius   AND   speed <= safe_landing_speed

Arriving too fast is a hard landing: it costs a lot and the episode carries on, so
the drone gets the chance to come round and try again. Arriving very fast is a
crash and ends the episode. That is what forces R-5 to learn to slow down rather
than simply aim at the pad.
"""

import math
import random

from rooms.room4 import chamber

# --- actions ---------------------------------------------------------------
NO_THRUST = 0
THRUST_UP = 1
THRUST_DOWN = 2
THRUST_LEFT = 3
THRUST_RIGHT = 4

ACTIONS = (NO_THRUST, THRUST_UP, THRUST_DOWN, THRUST_LEFT, THRUST_RIGHT)

ACTION_NAMES = {
    NO_THRUST: "COAST",
    THRUST_UP: "THRUST UP",
    THRUST_DOWN: "THRUST DOWN",
    THRUST_LEFT: "THRUST LEFT",
    THRUST_RIGHT: "THRUST RIGHT",
}

ACTION_ARROWS = {
    NO_THRUST: "·", THRUST_UP: "↑", THRUST_DOWN: "↓",
    THRUST_LEFT: "←", THRUST_RIGHT: "→",
}

# Which way each action pushes, before the thrust strength is applied.
ACTION_DIRECTIONS = {
    NO_THRUST: (0.0, 0.0),
    THRUST_UP: (0.0, 1.0),
    THRUST_DOWN: (0.0, -1.0),
    THRUST_LEFT: (-1.0, 0.0),
    THRUST_RIGHT: (1.0, 0.0),
}

# --- defaults, all adjustable from the interface ---------------------------
DT_DEFAULT = 0.25
THRUST_DEFAULT = 1.1
DRAG_DEFAULT = 0.90
V_MAX_DEFAULT = 3.0
WIND_MULTIPLIER_DEFAULT = 1.0
TURBULENCE_DEFAULT = 0.0
SAFE_LANDING_SPEED_DEFAULT = 0.7
CRASH_SPEED_DEFAULT = 2.4

STEP_COST_DEFAULT = -0.5
PROGRESS_SCALE_DEFAULT = 2.0
THRUST_COST_DEFAULT = -0.05
OBSTACLE_PENALTY_DEFAULT = -30.0
BOUNDARY_PENALTY_DEFAULT = -25.0
HARD_LANDING_PENALTY_DEFAULT = -40.0
CRASH_PENALTY_DEFAULT = -100.0
LANDING_REWARD_DEFAULT = 150.0

MAX_STEPS_DEFAULT = 220

# How many pieces each move is walked in, and how far the drone is kept from a
# surface it has just hit.
SUBSTEPS = 4
COLLISION_MARGIN = 0.06


class Room4Env:
    """The drone wind tunnel.

    There is no model to read: like Rooms 2 and 3, the only way to find out what
    an action does is to take it.
    """

    def __init__(self,
                 dt=DT_DEFAULT, thrust=THRUST_DEFAULT, drag=DRAG_DEFAULT,
                 v_max=V_MAX_DEFAULT, wind_multiplier=WIND_MULTIPLIER_DEFAULT,
                 turbulence=TURBULENCE_DEFAULT,
                 safe_landing_speed=SAFE_LANDING_SPEED_DEFAULT,
                 crash_speed=CRASH_SPEED_DEFAULT,
                 step_cost=STEP_COST_DEFAULT,
                 progress_scale=PROGRESS_SCALE_DEFAULT,
                 thrust_cost=THRUST_COST_DEFAULT,
                 obstacle_penalty=OBSTACLE_PENALTY_DEFAULT,
                 boundary_penalty=BOUNDARY_PENALTY_DEFAULT,
                 hard_landing_penalty=HARD_LANDING_PENALTY_DEFAULT,
                 crash_penalty=CRASH_PENALTY_DEFAULT,
                 landing_reward=LANDING_REWARD_DEFAULT,
                 start_position=None):
        self.dt = dt
        self.thrust = thrust
        self.drag = drag
        self.v_max = v_max
        self.wind_multiplier = wind_multiplier
        self.turbulence = turbulence
        self.safe_landing_speed = safe_landing_speed
        self.crash_speed = crash_speed
        self.step_cost = step_cost
        self.progress_scale = progress_scale
        self.thrust_cost = thrust_cost
        self.obstacle_penalty = obstacle_penalty
        self.boundary_penalty = boundary_penalty
        self.hard_landing_penalty = hard_landing_penalty
        self.crash_penalty = crash_penalty
        self.landing_reward = landing_reward
        self.start_position = start_position or chamber.START_POSITION

        self.state = None
        self.rng = random.Random()
        self.steps = 0

    # ------------------------------------------------------------------

    def actions(self):
        return list(ACTIONS)

    def state_bounds(self):
        """The interval each state variable lives in, for the tile coder."""
        return ((0.0, chamber.WIDTH), (0.0, chamber.HEIGHT),
                (-self.v_max, self.v_max), (-self.v_max, self.v_max))

    def reset(self, seed=None, start_position=None):
        if seed is not None:
            self.rng = random.Random(seed)
        start_x, start_y = start_position or self.start_position
        velocity_x, velocity_y = chamber.START_VELOCITY
        self.state = (start_x, start_y, velocity_x, velocity_y)
        self.steps = 0
        return self.state

    def is_terminal(self, state):
        """A landing is decided by the transition, not by the position alone.

        Sitting on the pad at speed is not a terminal state, which is exactly the
        distinction the room is about, so this only ever returns False and the
        `done` flag comes back from `step`.
        """
        return False

    # ------------------------------------------------------------------

    def step(self, action):
        """Apply one thrust action and move the drone."""
        x, y, velocity_x, velocity_y = self.state
        self.steps += 1
        previous_distance = chamber.distance_to_goal(x, y)

        thrust_x, thrust_y = ACTION_DIRECTIONS[action]
        thrust_x *= self.thrust
        thrust_y *= self.thrust

        wind_x, wind_y = chamber.wind_at(x, y, self.wind_multiplier)

        turbulence_x = turbulence_y = 0.0
        if self.turbulence > 0:
            turbulence_x = self.rng.gauss(0.0, self.turbulence)
            turbulence_y = self.rng.gauss(0.0, self.turbulence)

        velocity_x = (velocity_x + thrust_x + wind_x + turbulence_x) * self.drag
        velocity_y = (velocity_y + thrust_y + wind_y + turbulence_y) * self.drag
        velocity_x = _clip(velocity_x, self.v_max)
        velocity_y = _clip(velocity_y, self.v_max)

        # --- walk the move in substeps, so nothing is tunnelled through ---
        collided = False
        hit_boundary = False
        reached_pad = False
        for _ in range(SUBSTEPS):
            x += velocity_x * self.dt / SUBSTEPS
            y += velocity_y * self.dt / SUBSTEPS

            # The pad is checked inside the walk, not only at the end. A drone
            # arriving fast enough covers more than the pad's diameter in one
            # step, and testing only the final position would let it fly straight
            # through without the landing ever being noticed.
            if chamber.inside_goal(x, y):
                reached_pad = True
                break

            obstacle = chamber.blocking_obstacle(x, y, COLLISION_MARGIN)
            if obstacle is not None:
                collided = True
                x, y = obstacle.push_out(x, y, COLLISION_MARGIN)
                normal_x, normal_y = obstacle.normal_at(x, y)
                # Bounce back along the surface normal, losing most of the speed.
                along = velocity_x * normal_x + velocity_y * normal_y
                velocity_x = (velocity_x - 1.6 * along * normal_x) * 0.35
                velocity_y = (velocity_y - 1.6 * along * normal_y) * 0.35

            if not chamber.inside_bounds(x, y, COLLISION_MARGIN):
                hit_boundary = True
                x = min(max(x, COLLISION_MARGIN), chamber.WIDTH - COLLISION_MARGIN)
                y = min(max(y, COLLISION_MARGIN), chamber.HEIGHT - COLLISION_MARGIN)
                if x <= COLLISION_MARGIN or x >= chamber.WIDTH - COLLISION_MARGIN:
                    velocity_x = -velocity_x * 0.3
                if y <= COLLISION_MARGIN or y >= chamber.HEIGHT - COLLISION_MARGIN:
                    velocity_y = -velocity_y * 0.3

        speed = math.hypot(velocity_x, velocity_y)
        distance = chamber.distance_to_goal(x, y)

        # --- rewards ---------------------------------------------------
        reward = self.step_cost
        reward += self.progress_scale * (previous_distance - distance)
        if action != NO_THRUST:
            reward += self.thrust_cost

        event = "Flying"
        done = False
        safe_landing = False
        hard_landing = False
        crashed = False

        if collided:
            reward += self.obstacle_penalty
            event = "Struck an obstacle"
        if hit_boundary:
            reward += self.boundary_penalty
            event = "Hit the chamber wall"

        # --- the landing pad -------------------------------------------
        if reached_pad or chamber.inside_goal(x, y):
            if speed <= self.safe_landing_speed:
                reward += self.landing_reward
                safe_landing = True
                done = True
                event = "Safe landing — %.2f m/s" % speed
            elif speed >= self.crash_speed:
                reward += self.crash_penalty
                crashed = True
                done = True
                event = "Crashed into the pad at %.2f m/s" % speed
            else:
                # Too fast to count, not fast enough to break: bounce off and
                # carry on, so the drone can come round and try again.
                reward += self.hard_landing_penalty
                hard_landing = True
                velocity_x *= -0.4
                velocity_y *= -0.4
                event = "Too fast to land — %.2f m/s" % speed

        if not done and self.steps >= MAX_STEPS_DEFAULT:
            done = True
            event = "Out of time"

        self.state = (x, y, velocity_x, velocity_y)

        info = {
            "description": event,
            "thrust_x": thrust_x, "thrust_y": thrust_y,
            "wind_x": wind_x, "wind_y": wind_y,
            "turbulence_x": turbulence_x, "turbulence_y": turbulence_y,
            "speed": speed,
            "distance_to_goal": distance,
            "collision": collided,
            "boundary": hit_boundary,
            "safe_landing": safe_landing,
            "hard_landing": hard_landing,
            "crashed": crashed,
            "timeout": event == "Out of time",
        }
        return self.state, reward, done, info

    # ------------------------------------------------------------------

    def parameter_summary(self):
        return {
            "dt": self.dt, "thrust": self.thrust, "drag": self.drag,
            "v_max": self.v_max, "wind_multiplier": self.wind_multiplier,
            "turbulence": self.turbulence,
            "safe_landing_speed": self.safe_landing_speed,
            "crash_speed": self.crash_speed, "step_cost": self.step_cost,
            "progress_scale": self.progress_scale,
            "thrust_cost": self.thrust_cost,
            "obstacle_penalty": self.obstacle_penalty,
            "boundary_penalty": self.boundary_penalty,
            "hard_landing_penalty": self.hard_landing_penalty,
            "crash_penalty": self.crash_penalty,
            "landing_reward": self.landing_reward,
            "start_position": list(self.start_position),
        }


def _clip(value, limit):
    return max(-limit, min(limit, value))
