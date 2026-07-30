"""The Room 1 environment — a 10x10 security chamber with a fully known model.

This file is written to be read.  There is no numpy, no vectorisation and no
clever tricks: just loops, dictionaries and small helper functions.

--------------------------------------------------------------------------
THE STATE
--------------------------------------------------------------------------
A state is a plain tuple of four values:

    (row, col, has_battery, previous_direction)

  * row, col            -> where the robot stands
  * has_battery         -> True after the battery has been picked up.  It is
                           part of the state because the battery bonus may only
                           be paid once.
  * previous_direction  -> the direction the robot actually moved last.  It is
                           part of the state because oil makes the robot keep
                           sliding the way it was already going.  Without it the
                           environment would not be Markovian: the same cell
                           would behave differently depending on hidden history.

There are 54 walkable cells x 2 battery values x 5 directions = 540 states.

--------------------------------------------------------------------------
THE ACTIONS
--------------------------------------------------------------------------
Four actions: UP, DOWN, LEFT, RIGHT.  The action is what the robot *tries* to
do.  The floor decides what actually happens.

--------------------------------------------------------------------------
HOW THE FLOOR DECIDES  (the stochastic part)
--------------------------------------------------------------------------
The surface the robot is *standing on* decides the outcome, because that is the
surface it pushes off from.

  Normal floor      100% the chosen action.
  Wet floor  '~'    p_weak the chosen action, the rest split evenly between the
                    two sideways directions.
  Frozen floor '%'  p_strong the chosen action, the rest split evenly sideways.
  Oil  'O'          p_oil keep sliding in previous_direction,
                    10% split evenly sideways,
                    everything left over for the chosen action.
                    If there is no previous direction (or it equals the chosen
                    action) the momentum share is given to the chosen action
                    instead, so the numbers always add up to exactly 1.

--------------------------------------------------------------------------
WHAT HAPPENS WHEN THE ROBOT ARRIVES  (the deterministic part)
--------------------------------------------------------------------------
Once a direction has been picked, `resolve_direction` decides the result.  Every
transition pays the step cost (-1), and event rewards are ADDED on top of it.
So a laser hit is -1 + -30 = -31, and reaching the exit is -1 + 100 = +99.
This one convention is used everywhere, with no exceptions.
"""

import random

from core import actions, tiles
from rooms.room1 import map_data

# Default reward values.  All of them can be overridden from the interface.
STEP_COST_DEFAULT = -1
WALL_PENALTY_DEFAULT = -3
LASER_PENALTY_DEFAULT = -30
BATTERY_BONUS_DEFAULT = 10
TELEPORT_BONUS_DEFAULT = 5
EXIT_REWARD_DEFAULT = 100

# Default slipping probabilities.
WEAK_ICE_INTENDED_DEFAULT = 0.80
STRONG_ICE_INTENDED_DEFAULT = 0.60
OIL_MOMENTUM_DEFAULT = 0.55

# On oil, this much probability is always reserved for a sideways slip.
OIL_SIDEWAYS_TOTAL = 0.10

# How close to 1.0 a probability distribution has to be.
PROBABILITY_TOLERANCE = 1e-9


class Room1Env:
    """The Laser Security Chamber.

    The full transition model is available through `transitions`, which is what
    lets Dynamic Programming plan a route without ever touching a laser.
    """

    def __init__(self,
                 weak_ice_intended=WEAK_ICE_INTENDED_DEFAULT,
                 strong_ice_intended=STRONG_ICE_INTENDED_DEFAULT,
                 oil_momentum=OIL_MOMENTUM_DEFAULT,
                 step_cost=STEP_COST_DEFAULT,
                 wall_penalty=WALL_PENALTY_DEFAULT,
                 laser_penalty=LASER_PENALTY_DEFAULT,
                 battery_bonus=BATTERY_BONUS_DEFAULT,
                 teleport_bonus=TELEPORT_BONUS_DEFAULT,
                 exit_reward=EXIT_REWARD_DEFAULT,
                 require_battery=False):
        self.weak_ice_intended = weak_ice_intended
        self.strong_ice_intended = strong_ice_intended
        self.oil_momentum = oil_momentum
        self.step_cost = step_cost
        self.wall_penalty = wall_penalty
        self.laser_penalty = laser_penalty
        self.battery_bonus = battery_bonus
        self.teleport_bonus = teleport_bonus
        self.exit_reward = exit_reward
        self.require_battery = require_battery

        # Fixed facts about the map, looked up once.
        self.start_row, self.start_col = map_data.start_cell()
        self.exit_row, self.exit_col = map_data.exit_cell()
        self.teleport_pads = map_data.teleport_cells()

        # Live episode data, filled in by reset() and step().
        self.state = None
        self.rng = random.Random()

    # ------------------------------------------------------------------
    # The state space
    # ------------------------------------------------------------------

    def all_states(self):
        """Every state of the environment, in a stable order.

        The order never changes, which matters for saving results to a file.
        """
        states = []
        for row, col in map_data.walkable_cells():
            for has_battery in (False, True):
                for previous_direction in actions.ALL_DIRECTIONS:
                    states.append((row, col, has_battery, previous_direction))
        return states

    def actions(self):
        """The four actions the agent may choose."""
        return list(actions.ACTIONS)

    def start_state(self):
        """The state the robot starts every episode in."""
        return (self.start_row, self.start_col, False, actions.NONE)

    def is_terminal(self, state):
        """True if the episode ends in this state.

        Only the control panel ends the episode.  A laser hit sends the robot
        back to the start but the episode keeps going.
        """
        row, col, has_battery, _ = state
        if (row, col) != (self.exit_row, self.exit_col):
            return False
        if self.require_battery and not has_battery:
            return False
        return True

    # ------------------------------------------------------------------
    # Step 1 of a transition: which direction does the robot actually go?
    # ------------------------------------------------------------------

    def direction_probabilities(self, state, action):
        """The chance of each actual direction, given the chosen action.

        Returns a dictionary {direction: probability} that always sums to 1.
        The surface under the robot decides the numbers.
        """
        row, col, _, previous_direction = state
        surface = map_data.tile_at(row, col)

        if surface == tiles.WEAK_ICE:
            return self._slippery_probabilities(action, self.weak_ice_intended)

        if surface == tiles.STRONG_ICE:
            return self._slippery_probabilities(action, self.strong_ice_intended)

        if surface == tiles.OIL:
            return self._oil_probabilities(action, previous_direction)

        # Every other surface is firm: the robot goes exactly where it wants.
        return {action: 1.0}

    def _slippery_probabilities(self, action, intended_probability):
        """Ice: mostly the chosen action, otherwise a sideways slide."""
        left_slip, right_slip = actions.PERPENDICULAR[action]
        slip_each = (1.0 - intended_probability) / 2.0
        return {
            action: intended_probability,
            left_slip: slip_each,
            right_slip: slip_each,
        }

    def _oil_probabilities(self, action, previous_direction):
        """Oil: the robot tends to keep sliding the way it was already going."""
        left_slip, right_slip = actions.PERPENDICULAR[action]
        slip_each = OIL_SIDEWAYS_TOTAL / 2.0

        # No momentum to carry, so the momentum share goes to the chosen action.
        # This also covers the case where the momentum points the same way as
        # the action, which would otherwise be the same outcome twice.
        if previous_direction == actions.NONE or previous_direction == action:
            return {
                action: 1.0 - OIL_SIDEWAYS_TOTAL,
                left_slip: slip_each,
                right_slip: slip_each,
            }

        # The previous direction is often one of the two sideways directions, so
        # the shares are added together instead of overwriting each other.
        shares = [
            (previous_direction, self.oil_momentum),
            (action, 1.0 - self.oil_momentum - OIL_SIDEWAYS_TOTAL),
            (left_slip, slip_each),
            (right_slip, slip_each),
        ]
        probabilities = {}
        for direction, probability in shares:
            probabilities[direction] = probabilities.get(direction, 0.0) + probability
        return probabilities

    # ------------------------------------------------------------------
    # Step 2 of a transition: what happens once the direction is known?
    # ------------------------------------------------------------------

    def resolve_direction(self, state, direction):
        """Move one step in `direction` and report the result.

        Returns (next_state, reward, done, event) where `event` is a small
        dictionary describing what happened, used by the animation and the log.
        """
        row, col, has_battery, _ = state
        standing_on = map_data.tile_at(row, col)
        target_row, target_col = actions.move(row, col, direction)

        # A one-way door may only be LEFT in its own direction.
        if tiles.is_one_way_door(standing_on):
            if direction != tiles.door_direction(standing_on):
                return self._blocked(state, "Blocked by the one-way door")

        # Walls and the edge of the chamber.
        if not map_data.in_bounds(target_row, target_col):
            return self._blocked(state, "Bumped into the chamber wall")
        target = map_data.tile_at(target_row, target_col)
        if tiles.is_wall(target):
            return self._blocked(state, "Bumped into a metallic wall")

        # A one-way door may only be ENTERED in its own direction.
        if tiles.is_one_way_door(target):
            if direction != tiles.door_direction(target):
                return self._blocked(state, "The one-way door refused entry")

        # The control panel: the episode ends here.
        if target == tiles.EXIT:
            if self.require_battery and not has_battery:
                return self._blocked(state, "The control panel needs the battery")
            next_state = (target_row, target_col, has_battery, direction)
            event = self._event("Control panel reached — security disabled",
                                reached_exit=True)
            return next_state, self.step_cost + self.exit_reward, True, event

        # A laser beam: heavy penalty and back to the start platform.  The
        # battery stays collected, and the episode continues.
        if target == tiles.LASER:
            next_state = (self.start_row, self.start_col, has_battery, actions.NONE)
            event = self._event("Laser hit — returned to the start platform",
                                laser_hit=True)
            return next_state, self.step_cost + self.laser_penalty, False, event

        # A teleporter pad: jump to the other pad.  This happens exactly once
        # per transition, because we never look at the destination tile again,
        # so two pads can never bounce the robot back and forth forever.
        if target == tiles.TELEPORT:
            other_row, other_col = self._paired_pad(target_row, target_col)
            next_state = (other_row, other_col, has_battery, actions.NONE)
            event = self._event("Teleporter used", used_teleport=True)
            return next_state, self.step_cost + self.teleport_bonus, False, event

        # The battery: the bonus is paid the first time only.
        if target == tiles.BATTERY and not has_battery:
            next_state = (target_row, target_col, True, direction)
            event = self._event("Battery collected", battery_collected=True)
            return next_state, self.step_cost + self.battery_bonus, False, event

        # An ordinary step.
        next_state = (target_row, target_col, has_battery, direction)
        return next_state, self.step_cost, False, self._event("Moved")

    def _blocked(self, state, description):
        """The robot could not move: it stays put and pays the wall penalty.

        The previous direction is cleared, because a robot that has been stopped
        is no longer carrying any momentum.
        """
        row, col, has_battery, _ = state
        next_state = (row, col, has_battery, actions.NONE)
        event = self._event(description, blocked=True)
        return next_state, self.step_cost + self.wall_penalty, False, event

    def _paired_pad(self, row, col):
        """Given one teleporter pad, return the other one."""
        first, second = self.teleport_pads
        if (row, col) == first:
            return second
        return first

    def _event(self, description, blocked=False, laser_hit=False,
               used_teleport=False, battery_collected=False, reached_exit=False):
        """Build the small event dictionary carried by every transition."""
        return {
            "description": description,
            "blocked": blocked,
            "laser_hit": laser_hit,
            "used_teleport": used_teleport,
            "battery_collected": battery_collected,
            "reached_exit": reached_exit,
        }

    # ------------------------------------------------------------------
    # The full transition model — this is what Dynamic Programming needs
    # ------------------------------------------------------------------

    def transitions(self, state, action):
        """Every possible outcome of taking `action` in `state`.

        Returns a list of (probability, next_state, reward, done).  The
        probabilities always add up to 1.

        Outcomes that are completely identical are merged into one entry, so a
        bonus such as the +5 for the teleporter can never be counted twice.
        """
        if self.is_terminal(state):
            # Nothing happens after the episode has ended.
            return [(1.0, state, 0.0, True)]

        merged = {}
        for direction, probability in self.direction_probabilities(state, action).items():
            next_state, reward, done, _ = self.resolve_direction(state, direction)
            key = (next_state, reward, done)
            merged[key] = merged.get(key, 0.0) + probability

        outcomes = []
        for (next_state, reward, done), probability in merged.items():
            outcomes.append((probability, next_state, reward, done))
        return outcomes

    # ------------------------------------------------------------------
    # Running one episode (used by the animation and the experiments)
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        """Put the robot back on the start platform.

        Passing a seed makes the whole episode reproducible.
        """
        if seed is not None:
            self.rng = random.Random(seed)
        self.state = self.start_state()
        return self.state

    def step(self, action):
        """Take one action, sampling the outcome from the model.

        Returns (next_state, reward, done, info).  `info` describes what
        happened, including whether the robot slipped, so the animation and the
        replay log can show it.
        """
        state = self.state
        probabilities = self.direction_probabilities(state, action)

        # Sample one direction.  Sorting keeps the draw reproducible for a given
        # seed, because dictionary order alone is not something to rely on.
        directions = sorted(probabilities.keys())
        weights = [probabilities[direction] for direction in directions]
        actual_direction = self.rng.choices(directions, weights=weights)[0]

        next_state, reward, done, event = self.resolve_direction(state, actual_direction)

        _, _, _, previous_direction = state
        info = dict(event)
        info["action"] = action
        info["actual_direction"] = actual_direction
        info["slipped"] = actual_direction != action
        info["momentum"] = (actual_direction != action
                            and actual_direction == previous_direction)
        if info["momentum"]:
            info["description"] = "Momentum carried the robot"
        elif info["slipped"] and not event["laser_hit"] and not event["blocked"]:
            info["description"] = "Slipped"

        self.state = next_state
        return next_state, reward, done, info

    # ------------------------------------------------------------------
    # A self-check used by the tests and by the interface
    # ------------------------------------------------------------------

    def worst_probability_error(self):
        """How far the worst state-action pair is from summing to exactly 1."""
        worst = 0.0
        for state in self.all_states():
            for action in self.actions():
                total = sum(probability
                            for probability, _, _, _ in self.transitions(state, action))
                worst = max(worst, abs(total - 1.0))
        return worst

    def parameter_summary(self):
        """The settings of this environment, for display and for saved files."""
        return {
            "weak_ice_intended": self.weak_ice_intended,
            "strong_ice_intended": self.strong_ice_intended,
            "oil_momentum": self.oil_momentum,
            "step_cost": self.step_cost,
            "wall_penalty": self.wall_penalty,
            "laser_penalty": self.laser_penalty,
            "battery_bonus": self.battery_bonus,
            "teleport_bonus": self.teleport_bonus,
            "exit_reward": self.exit_reward,
            "require_battery": self.require_battery,
        }
