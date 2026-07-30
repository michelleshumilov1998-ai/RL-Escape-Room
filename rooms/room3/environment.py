"""The Room 3 environment — the Reactor Control Chamber.

--------------------------------------------------------------------------
THE STATE
--------------------------------------------------------------------------
    (row, col, stage, guard_index)

  * row, col      -> where R-5 stands
  * stage         -> how far through the generator sequence it is:
                       0  nothing running
                       1  Generator A running
                       2  A and B running
                       3  all three running, the blast door is unlocked
  * guard_index   -> where the security robot is on its 24-cell patrol

The guard's position is in the state rather than tracked on the side, which is
what makes the room a proper MDP: everything needed to predict the next step is
visible. The sliding doors are derived from `guard_index` too, so they are
predictable without adding a separate clock.

31 walkable cells x 4 stages x 24 patrol positions = 2976 states, x 5 actions.

--------------------------------------------------------------------------
WHAT MAKES THIS A Q-LEARNING ROOM
--------------------------------------------------------------------------
The reward for escaping is a long way from the move that starts earning it. The
generators only count in order, so the shortest possible mission is a full lap of
the service ring — 23 steps before the +150 arrives. Every step of that lap has to
be valued by what it eventually leads to, which is exactly what bootstrapping from
`max Q(s', a')` does.

--------------------------------------------------------------------------
THE ORDER OF A STEP
--------------------------------------------------------------------------
Used identically in training, evaluation and replay:

    1. R-5 acts. Walls, a shut sliding door and a locked blast door all block it.
    2. The cell it landed on takes effect: hazard, generator, or the exit.
    3. If the exit was reached the episode ends here.
    4. The security robot advances one cell along its patrol.
    5. A collision is checked — same cell, or the two swapping places.

--------------------------------------------------------------------------
REWARDS
--------------------------------------------------------------------------
Every transition pays the step cost and event rewards are ADDED on top, the same
convention as Rooms 1 and 2. So a hazard is -1 + -25 = -26 and the exit is
-1 + 150 = +149.
"""

import random

from core import actions
from rooms.room3 import map_data

# Default rewards, all adjustable from the interface.
STEP_COST_DEFAULT = -1
WALL_PENALTY_DEFAULT = -5
HAZARD_PENALTY_DEFAULT = -25
GUARD_PENALTY_DEFAULT = -100
GENERATOR_BONUS_DEFAULT = 20
FINAL_GENERATOR_BONUS_DEFAULT = 40
EXIT_REWARD_DEFAULT = 150

MAX_STEPS_DEFAULT = 300

STAGE_NONE = 0
STAGE_A = 1
STAGE_AB = 2
STAGE_UNLOCKED = 3

# Which generator each stage is waiting for.
GENERATOR_FOR_STAGE = {
    STAGE_NONE: map_data.GENERATOR_A,
    STAGE_A: map_data.GENERATOR_B,
    STAGE_AB: map_data.GENERATOR_C,
}

STAGE_NAMES = {
    STAGE_NONE: "No generators running",
    STAGE_A: "Generator A running",
    STAGE_AB: "Generators A and B running",
    STAGE_UNLOCKED: "All three running — blast door unlocked",
}


class Room3Env:
    """The Reactor Control Chamber.

    Learned from experience: there is no `transitions()` method to read.
    """

    def __init__(self,
                 step_cost=STEP_COST_DEFAULT,
                 wall_penalty=WALL_PENALTY_DEFAULT,
                 hazard_penalty=HAZARD_PENALTY_DEFAULT,
                 guard_penalty=GUARD_PENALTY_DEFAULT,
                 generator_bonus=GENERATOR_BONUS_DEFAULT,
                 final_generator_bonus=FINAL_GENERATOR_BONUS_DEFAULT,
                 exit_reward=EXIT_REWARD_DEFAULT,
                 guard_start_index=0):
        self.step_cost = step_cost
        self.wall_penalty = wall_penalty
        self.hazard_penalty = hazard_penalty
        self.guard_penalty = guard_penalty
        self.generator_bonus = generator_bonus
        self.final_generator_bonus = final_generator_bonus
        self.exit_reward = exit_reward
        self.guard_start_index = guard_start_index

        self.start_row, self.start_col = map_data.start_cell()
        self.exit_row, self.exit_col = map_data.exit_cell()

        self.state = None
        self.rng = random.Random()

    # ------------------------------------------------------------------
    # The state space
    # ------------------------------------------------------------------

    def all_states(self):
        """Every state, in a stable order."""
        states = []
        for row, col in map_data.walkable_cells():
            for stage in (STAGE_NONE, STAGE_A, STAGE_AB, STAGE_UNLOCKED):
                for guard_index in range(map_data.PATROL_LENGTH):
                    states.append((row, col, stage, guard_index))
        return states

    def actions(self):
        """Four moves plus waiting.

        Waiting matters here: it is how R-5 lets the patrol go past and how it
        holds position until a sliding door opens.
        """
        return list(actions.ACTIONS_WITH_WAIT)

    def start_state(self):
        return (self.start_row, self.start_col, STAGE_NONE, self.guard_start_index)

    def is_terminal(self, state):
        """Only the exit ends an episode by virtue of the state alone.

        Being caught by the guard also ends one, but that depends on how the state
        was arrived at, so it comes back through `step`'s `done` flag.
        """
        row, col, stage, _ = state
        return (row, col) == (self.exit_row, self.exit_col) and stage == STAGE_UNLOCKED

    # ------------------------------------------------------------------
    # Reading the situation
    # ------------------------------------------------------------------

    def guard_cell(self, state=None):
        state = state or self.state
        return map_data.guard_cell(state[3])

    def doors_open(self, state=None):
        state = state or self.state
        return map_data.door_is_open(state[3])

    def stage_name(self, state=None):
        state = state or self.state
        return STAGE_NAMES[state[2]]

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.state = self.start_state()
        return self.state

    def step(self, action):
        """Take one action and report what happened."""
        row, col, stage, guard_index = self.state
        guard_before = map_data.guard_cell(guard_index)

        reward = self.step_cost
        event = self._event("Moved")
        target_row, target_col = actions.move(row, col, action)

        # ---- 1. resolve the move --------------------------------------
        if action == actions.WAIT:
            new_row, new_col = row, col
            event = self._event("Held position", waited=True)
        elif not map_data.in_bounds(target_row, target_col) or \
                map_data.is_wall(map_data.tile_at(target_row, target_col)):
            new_row, new_col = row, col
            reward += self.wall_penalty
            event = self._event("Bumped into a reinforced wall", blocked=True)
        else:
            target = map_data.tile_at(target_row, target_col)

            if target == map_data.SLIDING_DOOR and not map_data.door_is_open(guard_index):
                new_row, new_col = row, col
                reward += self.wall_penalty
                event = self._event("The sliding door is shut", blocked=True,
                                    door_blocked=True)
            elif target == map_data.REACTOR_DOOR and stage != STAGE_UNLOCKED:
                new_row, new_col = row, col
                reward += self.wall_penalty
                event = self._event(
                    "The blast door is locked — the reactor is still offline",
                    blocked=True, door_blocked=True)
            else:
                new_row, new_col = target_row, target_col

        # ---- 2. what is on the cell R-5 ended up on -------------------
        landed_on = map_data.tile_at(new_row, new_col)
        activated = None

        if landed_on == map_data.HAZARD and (new_row, new_col) != (row, col):
            reward += self.hazard_penalty
            event = self._event("Electrical arc — systems jolted", hazard=True)

        elif landed_on in map_data.GENERATORS:
            wanted = GENERATOR_FOR_STAGE.get(stage)
            if landed_on == wanted:
                activated = landed_on
                stage += 1
                if stage == STAGE_UNLOCKED:
                    reward += self.final_generator_bonus
                    event = self._event(
                        "Generator %s online — reactor door unlocked" % landed_on,
                        generator=landed_on, unlocked=True)
                else:
                    reward += self.generator_bonus
                    event = self._event("Generator %s online" % landed_on,
                                        generator=landed_on)
            elif (new_row, new_col) != (row, col):
                # Out of order, so nothing happens. Worth saying so, because this
                # is the sequencing rule the room is built around.
                event = self._event(
                    "Generator %s will not start — %s has to come first"
                    % (landed_on, wanted if wanted else "the sequence"),
                    wrong_order=True)

        elif landed_on == map_data.EXIT and stage == STAGE_UNLOCKED:
            reward += self.exit_reward
            self.state = (new_row, new_col, stage, guard_index)
            event = self._event("Reactor exit reached — sector cleared",
                                reached_exit=True)
            event["guard_cell"] = list(guard_before)
            return self.state, reward, True, event

        # ---- 3. the guard advances ------------------------------------
        new_guard_index = map_data.next_guard_index(guard_index)
        guard_after = map_data.guard_cell(new_guard_index)

        # ---- 4. collision --------------------------------------------
        same_cell = (new_row, new_col) == guard_after
        crossing = (new_row, new_col) == guard_before and (row, col) == guard_after
        if same_cell or crossing:
            self.state = (new_row, new_col, stage, new_guard_index)
            event = self._event("Detected by the security robot", caught=True)
            event["guard_cell"] = list(guard_after)
            event["activated"] = activated
            return self.state, reward + self.guard_penalty, True, event

        self.state = (new_row, new_col, stage, new_guard_index)
        event["guard_cell"] = list(guard_after)
        event["activated"] = activated
        return self.state, reward, False, event

    # ------------------------------------------------------------------

    def _event(self, description, blocked=False, waited=False, hazard=False,
               caught=False, generator=None, unlocked=False, reached_exit=False,
               wrong_order=False, door_blocked=False):
        return {
            "description": description,
            "blocked": blocked,
            "waited": waited,
            "hazard": hazard,
            "caught": caught,
            "generator": generator,
            "unlocked": unlocked,
            "reached_exit": reached_exit,
            "wrong_order": wrong_order,
            "door_blocked": door_blocked,
        }

    def parameter_summary(self):
        return {
            "step_cost": self.step_cost,
            "wall_penalty": self.wall_penalty,
            "hazard_penalty": self.hazard_penalty,
            "guard_penalty": self.guard_penalty,
            "generator_bonus": self.generator_bonus,
            "final_generator_bonus": self.final_generator_bonus,
            "exit_reward": self.exit_reward,
            "guard_start_index": self.guard_start_index,
        }
