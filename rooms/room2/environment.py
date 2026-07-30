"""The Room 2 environment — the Broken Bridge Sector.

Written in the same plain style as Room 1: loops, dictionaries and small
functions, no numpy and no clever tricks.

--------------------------------------------------------------------------
THE STATE
--------------------------------------------------------------------------
    (row, col, has_keycard)

  * row, col      -> where the robot stands
  * has_keycard   -> True once the security keycard has been picked up. It is
                     part of the state because the exit door stays locked until
                     the keycard is held, so the same cell means two different
                     things depending on whether R-5 has it.

--------------------------------------------------------------------------
NO MODEL IS OFFERED
--------------------------------------------------------------------------
Room 1 exposed `transitions(state, action)`, because Dynamic Programming needs
the model. Room 2 deliberately does **not** have that method. SARSA has to learn
from experience, and leaving the model out means the agent structurally cannot
cheat by reading it. The only way to find out what an action does is to take it.

--------------------------------------------------------------------------
WHERE THE DANGER COMES FROM
--------------------------------------------------------------------------
The transitions are **deterministic**: an action always moves the robot the way
it was aimed. Nothing slips. What makes this room dangerous is that the agent
explores — an epsilon-greedy step taken at the wrong moment walks straight off
the span into the shaft. So the risk belongs to the *policy being followed*, not
to the environment, which is precisely the distinction between SARSA (which
learns the value of the policy it actually follows) and Q-Learning (which learns
the value of behaving perfectly).

--------------------------------------------------------------------------
THE COLLAPSING BRIDGES
--------------------------------------------------------------------------
Crossing a collapsing bridge breaks it. From that moment on, for the rest of the
episode, that cell behaves as a pit. The layout therefore changes while the
episode is running, and `reset` rebuilds it.

The set of already-collapsed bridges is held by the environment. Whether the agent
gets to *see* it is a switch:

  * `include_bridge_state=False` (the default) gives the state the assignment
    asks for, `(row, col, has_keycard)`. The collapse is then invisible to the
    agent, and the room is no longer Markovian from where it is standing: the cell
    ahead can be a sound bridge early in an episode and a hole later, and nothing
    in the state tells the two apart. The measured consequence is that the value
    of stepping towards the bridge becomes an average of "+82 if it is still
    there" and "-101 if it is not", which drags it below the safe route. Both
    SARSA and Q-Learning therefore avoid the bridge, and for the same reason.

  * `include_bridge_state=True` adds which bridges have already gone, as
    `(row, col, has_keycard, collapsed_mask)`. That restores the Markov property,
    and the classic on-policy/off-policy split appears: Q-Learning takes the
    bridge and falls in about one episode in five, while SARSA walks over the top.

The default is the assignment's representation. The switch exists because the
difference between the two is the clearest lesson in this room: it shows that
*what the agent is allowed to observe* can matter more than which update rule it
uses. Both results are reported in the README.

--------------------------------------------------------------------------
REWARDS
--------------------------------------------------------------------------
Every transition pays the step cost, and event rewards are ADDED on top of it —
the same convention Room 1 uses, applied here without exception. So falling into
a pit is -1 + -100 = -101, and reaching the exit is -1 + 100 = +99.
"""

import random

from core import actions
from rooms.room2 import map_data

# Default rewards. All of them can be changed from the interface.
STEP_COST_DEFAULT = -1
WALL_PENALTY_DEFAULT = -5
PIT_PENALTY_DEFAULT = -100
KEYCARD_BONUS_DEFAULT = 25
EXIT_REWARD_DEFAULT = 100
COLLAPSING_PENALTY_DEFAULT = -2

MAX_STEPS_DEFAULT = 200


class Room2Env:
    """The Broken Bridge Sector.

    Use it the way an agent has to: `reset()`, then `step(action)` over and over,
    reading the reward that comes back. There is no model to inspect.
    """

    def __init__(self,
                 include_bridge_state=False,
                 step_cost=STEP_COST_DEFAULT,
                 wall_penalty=WALL_PENALTY_DEFAULT,
                 pit_penalty=PIT_PENALTY_DEFAULT,
                 keycard_bonus=KEYCARD_BONUS_DEFAULT,
                 exit_reward=EXIT_REWARD_DEFAULT,
                 collapsing_penalty=COLLAPSING_PENALTY_DEFAULT):
        self.include_bridge_state = include_bridge_state
        self.step_cost = step_cost
        self.wall_penalty = wall_penalty
        self.pit_penalty = pit_penalty
        self.keycard_bonus = keycard_bonus
        self.exit_reward = exit_reward
        self.collapsing_penalty = collapsing_penalty

        # Fixed facts about the map, looked up once.
        self.start_row, self.start_col = map_data.start_cell()
        self.exit_row, self.exit_col = map_data.exit_cell()
        # Each collapsing bridge gets one bit in the mask, in map order.
        self.collapsing_order = tuple(map_data.collapsing_cells())
        self.bridge_bit = {cell: index
                           for index, cell in enumerate(self.collapsing_order)}

        # Live episode data, rebuilt by reset().
        self.state = None
        self.collapsed = set()
        self.rng = random.Random()

    # ------------------------------------------------------------------
    # The state space
    # ------------------------------------------------------------------

    def collapsed_mask(self):
        """The already-collapsed bridges packed into one integer.

        Bit 0 is the first collapsing bridge on the map, bit 1 the second, and so
        on. Only used when `include_bridge_state` is on.
        """
        mask = 0
        for cell in self.collapsed:
            if cell in self.bridge_bit:
                mask |= 1 << self.bridge_bit[cell]
        return mask

    def observation(self):
        """What the agent gets to see of the current situation.

        Either `(row, col, has_keycard)` or, with `include_bridge_state` on,
        `(row, col, has_keycard, collapsed_mask)`.
        """
        row, col, has_keycard = self.state
        if self.include_bridge_state:
            return (row, col, has_keycard, self.collapsed_mask())
        return (row, col, has_keycard)

    def all_states(self):
        """Every state the agent can be in, in a stable order.

        Includes the pits that can actually be fallen into, so a Q-table built
        over this list never has a missing entry when an episode ends in one.
        """
        cells = list(map_data.walkable_cells()) + list(map_data.enterable_pits())
        masks = range(1 << len(self.collapsing_order)) if self.include_bridge_state \
            else [None]
        states = []
        for row, col in sorted(set(cells)):
            for has_keycard in (False, True):
                for mask in masks:
                    if mask is None:
                        states.append((row, col, has_keycard))
                    else:
                        states.append((row, col, has_keycard, mask))
        return states

    def actions(self):
        """The four actions the agent may choose."""
        return list(actions.ACTIONS)

    def start_state(self):
        if self.include_bridge_state:
            return (self.start_row, self.start_col, False, 0)
        return (self.start_row, self.start_col, False)

    def is_terminal(self, state):
        """True if the episode ends in this state, judging by the map alone.

        A bridge that collapsed during the episode is not covered here, because
        it is not a property of the map. Those endings arrive through the `done`
        flag that `step` returns.
        """
        row, col, has_keycard = state[0], state[1], state[2]
        if map_data.is_pit(map_data.tile_at(row, col)):
            return True
        if (row, col) == (self.exit_row, self.exit_col) and has_keycard:
            return True
        return False

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        """Start a new episode: every bridge is rebuilt and the keycard is back."""
        if seed is not None:
            self.rng = random.Random(seed)
        self.collapsed = set()
        self.state = (self.start_row, self.start_col, False)
        return self.observation()

    def step(self, action):
        """Take one action and report what happened.

        Returns (next_state, reward, done, info). `info` describes the event, so
        the animation and the replay log can show it.
        """
        row, col, has_keycard = self.state
        target_row, target_col = actions.move(row, col, action)

        # --- the edge of the sector, and walls -------------------------
        if not map_data.in_bounds(target_row, target_col):
            return self._blocked("Bumped into the sector wall")
        target = map_data.tile_at(target_row, target_col)
        if map_data.is_wall(target):
            return self._blocked("Bumped into a metallic wall")

        # --- the exit door, which stays locked without the keycard -----
        if target == map_data.EXIT and not has_keycard:
            return self._blocked("The exit door is locked — the keycard is missing",
                                 door_locked=True)

        # --- a bridge that has already collapsed is now a hole ---------
        if (target_row, target_col) in self.collapsed:
            return self._fall(target_row, target_col, has_keycard,
                              "Stepped onto a collapsed bridge and fell")

        # --- an open pit ------------------------------------------------
        if map_data.is_pit(target):
            return self._fall(target_row, target_col, has_keycard,
                              "Fell into the maintenance shaft")

        # Everything from here on is a successful move.
        reward = self.step_cost
        event = self._event("Moved")
        collapsed_cell = None

        # --- crossing a collapsing bridge breaks it behind the robot ----
        if map_data.is_collapsing(target):
            reward += self.collapsing_penalty
            self.collapsed.add((target_row, target_col))
            collapsed_cell = (target_row, target_col)
            event = self._event("The bridge gave way after crossing",
                                bridge_collapsed=True)

        # --- the keycard, which is only ever paid for once -------------
        if target == map_data.KEYCARD and not has_keycard:
            has_keycard = True
            reward += self.keycard_bonus
            event = self._event("Security keycard collected — the exit unlocks",
                                keycard_collected=True)

        # --- the exit door, now that the keycard is held ---------------
        if target == map_data.EXIT:
            reward += self.exit_reward
            event = self._event("Exit door reached — sector cleared",
                                reached_exit=True)
            self.state = (target_row, target_col, has_keycard)
            event["collapsed_cell"] = collapsed_cell
            return self.observation(), reward, True, event

        self.state = (target_row, target_col, has_keycard)
        event["collapsed_cell"] = collapsed_cell
        return self.observation(), reward, False, event

    # ------------------------------------------------------------------
    # The three ways a step can turn out
    # ------------------------------------------------------------------

    def _blocked(self, description, door_locked=False):
        """The robot could not move: it stays put and pays the wall penalty."""
        event = self._event(description, blocked=True, door_locked=door_locked)
        event["collapsed_cell"] = None
        return self.observation(), self.step_cost + self.wall_penalty, False, event

    def _fall(self, row, col, has_keycard, description):
        """The robot went into the shaft. The episode ends here."""
        self.state = (row, col, has_keycard)
        event = self._event(description, pit_fall=True)
        event["collapsed_cell"] = None
        return self.observation(), self.step_cost + self.pit_penalty, True, event

    def _event(self, description, blocked=False, pit_fall=False,
               keycard_collected=False, bridge_collapsed=False,
               reached_exit=False, door_locked=False):
        """The small event dictionary carried by every transition."""
        return {
            "description": description,
            "blocked": blocked,
            "pit_fall": pit_fall,
            "keycard_collected": keycard_collected,
            "bridge_collapsed": bridge_collapsed,
            "reached_exit": reached_exit,
            "door_locked": door_locked,
        }

    # ------------------------------------------------------------------
    # Odds and ends
    # ------------------------------------------------------------------

    def collapsed_cells(self):
        """The bridges that have broken so far this episode."""
        return sorted(self.collapsed)

    def parameter_summary(self):
        """The settings of this environment, for display and for saved files."""
        return {
            "include_bridge_state": self.include_bridge_state,
            "step_cost": self.step_cost,
            "wall_penalty": self.wall_penalty,
            "pit_penalty": self.pit_penalty,
            "keycard_bonus": self.keycard_bonus,
            "exit_reward": self.exit_reward,
            "collapsing_penalty": self.collapsing_penalty,
        }
