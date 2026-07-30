"""The Room 5 environment — the Adaptive Storage Facility.

--------------------------------------------------------------------------
WHAT THE AGENT SEES
--------------------------------------------------------------------------
Not the warehouse. R-5 gets a **local observation** only:

  * eight static radar rays — normalised distance to the nearest shelf, crate or
    wall in each of eight compass directions
  * four dynamic radar rays — the same for maintenance robots, up/right/down/left
  * where the current objective is, relative to R-5, and how far
  * the mission stage
  * the previous action

That is deliberately a **partially observable** problem, not a full MDP: two
different spots in two different warehouses can produce identical readings. It is
also the only representation that can transfer, because it never mentions an
absolute position or a layout identity. The full map exists in the environment and
is shown in the analysis view, but `observation()` never exposes it.

--------------------------------------------------------------------------
THE MISSION
--------------------------------------------------------------------------
    stage 0   reach the access terminal
    stage 1   terminal activated, the final exit is unlocked

Reaching the exit at stage 0 does nothing but cost a little.

--------------------------------------------------------------------------
THE ORDER OF A STEP
--------------------------------------------------------------------------
Fixed, and identical in training, evaluation and replay:

    1. R-5 picks an action.
    2. R-5 moves; a shelf or crate blocks it and costs the collision penalty.
    3. A conveyor under R-5 shoves it one more square, if that square is free.
    4. The maintenance robots each advance one step along their patrol.
    5. Collisions are checked — same cell, or the two swapping places.
    6. The terminal, the charger and the exit are processed.
"""

import math

from rooms.room5 import layout as L

# --- actions ---------------------------------------------------------------
MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3
WAIT = 4

ACTIONS = (MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, WAIT)

ACTION_NAMES = {MOVE_UP: "UP", MOVE_DOWN: "DOWN", MOVE_LEFT: "LEFT",
                MOVE_RIGHT: "RIGHT", WAIT: "WAIT"}
ACTION_ARROWS = {MOVE_UP: "↑", MOVE_DOWN: "↓", MOVE_LEFT: "←",
                 MOVE_RIGHT: "→", WAIT: "◦"}
ACTION_DELTAS = {MOVE_UP: (-1, 0), MOVE_DOWN: (1, 0), MOVE_LEFT: (0, -1),
                 MOVE_RIGHT: (0, 1), WAIT: (0, 0)}

# The eight radar directions, clockwise from north.
RADAR_DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1),
                    (0, -1), (-1, -1))
RADAR_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
# The four directions the dynamic radar covers.
DYNAMIC_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))

# --- defaults --------------------------------------------------------------
STEP_COST_DEFAULT = -1.0
COLLISION_PENALTY_DEFAULT = -5.0
CONVEYOR_PENALTY_DEFAULT = -1.0
TERMINAL_REWARD_DEFAULT = 40.0
EARLY_EXIT_PENALTY_DEFAULT = -2.0
ROBOT_PENALTY_DEFAULT = -100.0
TIMEOUT_PENALTY_DEFAULT = -30.0
EXIT_REWARD_DEFAULT = 200.0
CHARGER_REWARD_DEFAULT = 5.0
PROGRESS_SCALE_DEFAULT = 1.5


class Room5Env:
    """One warehouse mission on one generated layout."""

    def __init__(self, layout=None, difficulty=L.DEFAULT_DIFFICULTY,
                 step_cost=STEP_COST_DEFAULT,
                 collision_penalty=COLLISION_PENALTY_DEFAULT,
                 conveyor_penalty=CONVEYOR_PENALTY_DEFAULT,
                 terminal_reward=TERMINAL_REWARD_DEFAULT,
                 early_exit_penalty=EARLY_EXIT_PENALTY_DEFAULT,
                 robot_penalty=ROBOT_PENALTY_DEFAULT,
                 timeout_penalty=TIMEOUT_PENALTY_DEFAULT,
                 exit_reward=EXIT_REWARD_DEFAULT,
                 charger_reward=CHARGER_REWARD_DEFAULT,
                 progress_scale=PROGRESS_SCALE_DEFAULT,
                 radar_range=None, max_steps=None):
        self.layout = layout or L.generate(L.seed_for("training", 0), "training",
                                           difficulty)
        self.difficulty = difficulty
        self.step_cost = step_cost
        self.collision_penalty = collision_penalty
        self.conveyor_penalty = conveyor_penalty
        self.terminal_reward = terminal_reward
        self.early_exit_penalty = early_exit_penalty
        self.robot_penalty = robot_penalty
        self.timeout_penalty = timeout_penalty
        self.exit_reward = exit_reward
        self.charger_reward = charger_reward
        self.progress_scale = progress_scale
        self.radar_range = radar_range or self.layout.radar_range
        self.max_steps = max_steps or self.layout.max_steps

        self.position = None
        self.stage = 0
        self.robots = []
        self.steps = 0
        self.previous_action = WAIT
        self.charger_used = False
        self.seen_cells = set()

    # ------------------------------------------------------------------

    def actions(self):
        return list(ACTIONS)

    def set_layout(self, layout):
        """Swap in a different warehouse. The agent is unchanged."""
        self.layout = layout
        self.radar_range = layout.radar_range
        self.max_steps = layout.max_steps

    def reset(self, layout=None):
        if layout is not None:
            self.set_layout(layout)
        self.position = self.layout.start
        self.stage = 0
        self.robots = self.layout.fresh_robots()
        self.steps = 0
        self.previous_action = WAIT
        self.charger_used = False
        self.seen_cells = set(self.visible_cells())
        return self.observation()

    # ------------------------------------------------------------------
    # What R-5 is aiming at
    # ------------------------------------------------------------------

    def current_target(self):
        return self.layout.terminal if self.stage == 0 else self.layout.exit_cell

    def target_distance(self):
        row, col = self.position
        target_row, target_col = self.current_target()
        return math.hypot(target_row - row, target_col - col)

    # ------------------------------------------------------------------
    # The radar
    # ------------------------------------------------------------------

    def _cast(self, direction, hits_robot=False):
        """Distance to the first blocker along one direction, normalised.

        1.0 means nothing within range; small values mean something close. Robot
        positions are only consulted when `hits_robot` is set, which keeps the
        static and dynamic channels separate.
        """
        row, col = self.position
        delta_row, delta_col = direction
        robot_cells = {robot.cell() for robot in self.robots} if hits_robot else set()

        for step in range(1, self.radar_range + 1):
            cell = (row + delta_row * step, col + delta_col * step)
            if not self.layout.in_bounds(*cell):
                return step / self.radar_range
            if hits_robot:
                if cell in robot_cells:
                    return step / self.radar_range
            elif self.layout.blocks(*cell):
                return step / self.radar_range
        return 1.0

    def static_radar(self):
        return [self._cast(direction) for direction in RADAR_DIRECTIONS]

    def dynamic_radar(self):
        return [self._cast(direction, hits_robot=True)
                for direction in DYNAMIC_DIRECTIONS]

    def nearest_robot_distance(self):
        """Normalised distance to the closest maintenance robot."""
        if not self.robots:
            return 1.0
        row, col = self.position
        best = min(abs(robot.cell()[0] - row) + abs(robot.cell()[1] - col)
                   for robot in self.robots)
        return min(1.0, best / max(1, self.radar_range * 2))

    def visible_cells(self):
        """What the radar can currently see, used for the fog of war.

        Purely a display concern — the agent never receives this list.
        """
        row, col = self.position
        visible = {(row, col)}
        for direction in RADAR_DIRECTIONS:
            for step in range(1, self.radar_range + 1):
                cell = (row + direction[0] * step, col + direction[1] * step)
                if not self.layout.in_bounds(*cell):
                    break
                visible.add(cell)
                if self.layout.blocks(*cell):
                    break
        return visible

    # ------------------------------------------------------------------
    # The observation
    # ------------------------------------------------------------------

    def observation(self):
        """The local view the agent learns from. Never the whole map."""
        row, col = self.position
        target_row, target_col = self.current_target()
        size = self.layout.size

        delta_row = (target_row - row) / size
        delta_col = (target_col - col) / size
        distance = self.target_distance() / (size * 1.42)

        previous = [0.0] * len(ACTIONS)
        previous[self.previous_action] = 1.0

        return {
            "static_radar": self.static_radar(),
            "dynamic_radar": self.dynamic_radar(),
            "target": [delta_row, delta_col, distance],
            "stage": float(self.stage),
            "nearest_robot": self.nearest_robot_distance(),
            "previous_action": previous,
        }

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------

    def step(self, action):
        """Take one action, following the documented order exactly."""
        self.steps += 1
        previous_distance = self.target_distance()
        row, col = self.position

        reward = self.step_cost
        event = "Moved"
        collided = False
        conveyed = False
        activated = False
        early_exit = False
        charged = False

        # --- 2. move ---------------------------------------------------
        delta_row, delta_col = ACTION_DELTAS[action]
        target = (row + delta_row, col + delta_col)
        if action == WAIT:
            new_position = (row, col)
            event = "Held position"
        elif self.layout.blocks(*target):
            new_position = (row, col)
            reward += self.collision_penalty
            collided = True
            event = "Blocked by warehouse racking"
        else:
            new_position = target

        # The final exit is shut until the terminal has been used.
        if new_position == self.layout.exit_cell and self.stage == 0:
            reward += self.early_exit_penalty
            early_exit = True
            event = "The final gate is locked — the terminal is still offline"
            new_position = (row, col)

        # --- 3. conveyor ------------------------------------------------
        conveyor = self.layout.conveyor_at(*new_position)
        if conveyor is not None:
            pushed = (new_position[0] + conveyor[0], new_position[1] + conveyor[1])
            if (not self.layout.blocks(*pushed)
                    and not (pushed == self.layout.exit_cell and self.stage == 0)):
                new_position = pushed
                reward += self.conveyor_penalty
                conveyed = True
                event = "Carried along by a conveyor"

        moved_from = (row, col)
        self.position = new_position

        # --- 4. the robots advance --------------------------------------
        robot_before = [robot.cell() for robot in self.robots]
        for robot in self.robots:
            robot.advance()
        robot_after = [robot.cell() for robot in self.robots]

        # --- 5. collisions, including a straight swap -------------------
        caught = False
        for before, after in zip(robot_before, robot_after):
            if after == self.position:
                caught = True
                break
            if after == moved_from and before == self.position:
                caught = True
                break
        if caught:
            self.previous_action = action
            return (self.observation(), reward + self.robot_penalty, True,
                    self._info("Struck by a maintenance robot", collided,
                               conveyed, caught=True, robot_cells=robot_after))

        # --- 6. the terminal, the charger and the exit ------------------
        tile = self.layout.tile_at(*self.position)

        if tile == L.CHARGER and not self.charger_used:
            self.charger_used = True
            reward += self.charger_reward
            charged = True
            event = "Topped up at the charging station"

        if self.position == self.layout.terminal and self.stage == 0:
            self.stage = 1
            reward += self.terminal_reward
            activated = True
            event = "Access terminal activated — the final gate unlocks"

        elif self.position == self.layout.exit_cell and self.stage == 1:
            reward += self.exit_reward
            self.previous_action = action
            self.seen_cells |= self.visible_cells()
            return (self.observation(), reward, True,
                    self._info("Final gate reached — facility cleared", collided,
                               conveyed, reached_exit=True,
                               robot_cells=robot_after))

        # Shaping, against whichever objective is current. Deliberately small so
        # it can never outweigh the terminal or the exit.
        if not activated:
            reward += self.progress_scale * (previous_distance
                                             - self.target_distance())

        self.previous_action = action
        self.seen_cells |= self.visible_cells()

        done = self.steps >= self.max_steps
        if done:
            reward += self.timeout_penalty
            event = "Out of time"

        return (self.observation(), reward, done,
                self._info(event, collided, conveyed, activated=activated,
                           early_exit=early_exit, charged=charged,
                           timeout=done, robot_cells=robot_after))

    # ------------------------------------------------------------------

    def _info(self, description, collided, conveyed, activated=False,
              early_exit=False, charged=False, caught=False, reached_exit=False,
              timeout=False, robot_cells=None):
        return {
            "description": description,
            "collision": collided,
            "conveyor": conveyed,
            "terminal_activated": activated,
            "early_exit": early_exit,
            "charged": charged,
            "caught": caught,
            "reached_exit": reached_exit,
            "timeout": timeout,
            "stage": self.stage,
            "robot_cells": [list(cell) for cell in
                            (robot_cells or [robot.cell()
                                             for robot in self.robots])],
            "robot_directions": [robot.pattern for robot in self.robots],
            "visible_cells": [list(cell) for cell in self.visible_cells()],
        }

    def parameter_summary(self):
        return {
            "difficulty": self.difficulty,
            "step_cost": self.step_cost,
            "collision_penalty": self.collision_penalty,
            "conveyor_penalty": self.conveyor_penalty,
            "terminal_reward": self.terminal_reward,
            "early_exit_penalty": self.early_exit_penalty,
            "robot_penalty": self.robot_penalty,
            "timeout_penalty": self.timeout_penalty,
            "exit_reward": self.exit_reward,
            "charger_reward": self.charger_reward,
            "progress_scale": self.progress_scale,
            "radar_range": self.radar_range,
            "max_steps": self.max_steps,
        }
