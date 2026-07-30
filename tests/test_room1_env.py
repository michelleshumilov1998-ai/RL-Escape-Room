"""Tests for the Room 1 environment — required checks 9 to 21.

Numbered comments refer to the checklist in the project brief.
"""

import pytest

from core import actions, tiles
from rooms.room1 import map_data
from rooms.room1.environment import Room1Env


@pytest.fixture
def env():
    return Room1Env()


# ----------------------------------------------------------------------
# The state space
# ----------------------------------------------------------------------

def test_every_valid_state_appears_in_all_states(env):
    """9. Every valid state appears in all_states."""
    states = set(env.all_states())
    expected = len(map_data.walkable_cells()) * 2 * len(actions.ALL_DIRECTIONS)
    assert len(states) == expected
    assert len(env.all_states()) == expected, "all_states contains duplicates"

    for row, col in map_data.walkable_cells():
        for has_battery in (False, True):
            for direction in actions.ALL_DIRECTIONS:
                assert (row, col, has_battery, direction) in states

    # Walls are never states.
    for row in range(map_data.GRID_ROWS):
        for col in range(map_data.GRID_COLS):
            if tiles.is_wall(map_data.tile_at(row, col)):
                assert (row, col, False, actions.NONE) not in states


def test_battery_status_is_part_of_the_state(env):
    """10. Battery status is included in the state space."""
    values = {state[2] for state in env.all_states()}
    assert values == {False, True}


def test_previous_direction_is_part_of_the_state(env):
    """11. Previous direction is included in the state space."""
    values = {state[3] for state in env.all_states()}
    assert values == set(actions.ALL_DIRECTIONS)


def test_all_states_order_is_stable(env):
    """Saved files rely on the state order never changing."""
    assert env.all_states() == Room1Env().all_states()


# ----------------------------------------------------------------------
# The transition model
# ----------------------------------------------------------------------

def test_all_transition_probabilities_sum_to_one(env):
    """12. All transition probabilities sum to 1, for every state and action."""
    for state in env.all_states():
        for action in env.actions():
            outcomes = env.transitions(state, action)
            total = sum(probability for probability, _, _, _ in outcomes)
            assert abs(total - 1.0) < 1e-9, \
                "state %s action %s sums to %r" % (state, action, total)
            for probability, _, _, _ in outcomes:
                assert probability > 0.0


def test_worst_probability_error_is_zero(env):
    assert env.worst_probability_error() < 1e-9


def test_transitions_are_merged_so_no_outcome_repeats(env):
    """Identical outcomes must be merged into a single entry."""
    for state in env.all_states():
        for action in env.actions():
            keys = [(next_state, reward, done)
                    for _, next_state, reward, done in env.transitions(state, action)]
            assert len(keys) == len(set(keys)), \
                "duplicate outcome for state %s action %s" % (state, action)


@pytest.mark.parametrize("intended", [0.2, 0.5, 0.8, 1.0])
def test_probabilities_sum_to_one_for_other_slip_settings(intended):
    """The interface can move the sliders, so the model has to stay valid."""
    env = Room1Env(weak_ice_intended=intended, strong_ice_intended=intended,
                   oil_momentum=min(intended, 0.85))
    assert env.worst_probability_error() < 1e-9


def test_ice_slips_sideways_only(env):
    """On ice the robot either goes where it aimed, or exactly sideways."""
    state = (8, 4, False, actions.NONE)          # a wet cell
    assert map_data.tile_at(8, 4) == tiles.WEAK_ICE
    distribution = env.direction_probabilities(state, actions.UP)
    assert distribution[actions.UP] == pytest.approx(0.80)
    assert distribution[actions.LEFT] == pytest.approx(0.10)
    assert distribution[actions.RIGHT] == pytest.approx(0.10)
    assert actions.DOWN not in distribution


def test_oil_carries_the_previous_direction(env):
    """Oil mostly keeps the robot going the way it was already going.

    The action here is the opposite of the momentum, so none of the four shares
    land on the same direction and each one is visible on its own.
    """
    state = (9, 7, False, actions.RIGHT)         # an oil cell, moving right
    assert map_data.tile_at(9, 7) == tiles.OIL
    distribution = env.direction_probabilities(state, actions.LEFT)
    assert distribution[actions.RIGHT] == pytest.approx(0.55)   # momentum
    assert distribution[actions.LEFT] == pytest.approx(0.35)    # chosen action
    assert distribution[actions.UP] == pytest.approx(0.05)      # sideways
    assert distribution[actions.DOWN] == pytest.approx(0.05)
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_oil_shares_are_added_together_when_they_collide(env):
    """When the momentum points sideways, the two shares are added, not lost.

    Momentum is RIGHT and the chosen action is UP, so RIGHT is both the momentum
    direction and one of the two sideways directions: 0.55 + 0.05 = 0.60.  Adding
    them is what keeps the distribution summing to exactly 1.
    """
    state = (9, 7, False, actions.RIGHT)
    distribution = env.direction_probabilities(state, actions.UP)
    assert distribution[actions.RIGHT] == pytest.approx(0.60)
    assert distribution[actions.UP] == pytest.approx(0.35)
    assert distribution[actions.LEFT] == pytest.approx(0.05)
    assert actions.DOWN not in distribution
    assert sum(distribution.values()) == pytest.approx(1.0)


def test_oil_without_a_previous_direction_still_sums_to_one(env):
    """With no momentum to carry, the momentum share goes to the action."""
    state = (9, 7, False, actions.NONE)
    distribution = env.direction_probabilities(state, actions.UP)
    assert distribution[actions.UP] == pytest.approx(0.90)
    assert sum(distribution.values()) == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Walls
# ----------------------------------------------------------------------

def test_walking_into_a_wall_keeps_the_robot_in_place(env):
    """13. Entering a wall keeps the robot in place."""
    state = (0, 1, False, actions.NONE)
    assert tiles.is_wall(map_data.tile_at(1, 1))
    next_state, reward, done, event = env.resolve_direction(state, actions.DOWN)
    assert next_state[:2] == (0, 1)
    assert reward == env.step_cost + env.wall_penalty
    assert not done
    assert event["blocked"]


def test_walking_off_the_edge_keeps_the_robot_in_place(env):
    state = (0, 0, False, actions.NONE)
    next_state, reward, done, _ = env.resolve_direction(state, actions.UP)
    assert next_state[:2] == (0, 0)
    assert reward == env.step_cost + env.wall_penalty


def test_a_blocked_move_clears_the_previous_direction(env):
    """A robot that has been stopped is no longer carrying momentum."""
    state = (0, 1, False, actions.RIGHT)
    next_state, _, _, _ = env.resolve_direction(state, actions.DOWN)
    assert next_state[3] == actions.NONE


# ----------------------------------------------------------------------
# Lasers
# ----------------------------------------------------------------------

def test_a_laser_returns_the_robot_to_the_start(env):
    """14. A laser returns the robot to the start."""
    state = (1, 5, False, actions.DOWN)        # in the shaft, beam to the left
    assert map_data.tile_at(1, 4) == tiles.LASER
    next_state, _, done, event = env.resolve_direction(state, actions.LEFT)
    assert next_state[:2] == map_data.start_cell()
    assert next_state[3] == actions.NONE
    assert not done, "a laser hit must not end the episode"
    assert event["laser_hit"]


def test_a_laser_gives_the_correct_penalty(env):
    """15. A laser gives the correct penalty (step cost plus the penalty)."""
    state = (1, 5, False, actions.DOWN)
    _, reward, _, _ = env.resolve_direction(state, actions.LEFT)
    assert reward == env.step_cost + env.laser_penalty
    assert reward == -31


def test_a_laser_keeps_a_collected_battery(env):
    state = (1, 5, True, actions.DOWN)
    next_state, _, _, _ = env.resolve_direction(state, actions.LEFT)
    assert next_state[2] is True


def test_the_laser_penalty_is_configurable():
    env = Room1Env(laser_penalty=-90)
    _, reward, _, _ = env.resolve_direction((1, 5, False, actions.DOWN), actions.LEFT)
    assert reward == -91


# ----------------------------------------------------------------------
# The battery
# ----------------------------------------------------------------------

def test_the_battery_reward_is_given_only_once(env):
    """16. The battery reward is given only once."""
    battery_row, battery_col = map_data.battery_cell()
    approach = (battery_row + 1, battery_col, False, actions.UP)

    first_state, first_reward, _, event = env.resolve_direction(approach, actions.UP)
    assert first_state[:2] == (battery_row, battery_col)
    assert first_state[2] is True
    assert first_reward == env.step_cost + env.battery_bonus
    assert event["battery_collected"]

    # Arriving again with the battery already collected pays only the step cost.
    approach_again = (battery_row + 1, battery_col, True, actions.UP)
    _, second_reward, _, second_event = env.resolve_direction(approach_again,
                                                              actions.UP)
    assert second_reward == env.step_cost
    assert not second_event["battery_collected"]


# ----------------------------------------------------------------------
# The teleporter
# ----------------------------------------------------------------------

def test_the_teleporter_moves_the_robot_to_its_pair(env):
    """17. The teleporter moves the robot to the paired cell."""
    first_pad, second_pad = map_data.teleport_cells()
    approach = (first_pad[0] + 1, first_pad[1], False, actions.UP)
    assert map_data.tile_at(*first_pad) == tiles.TELEPORT

    next_state, reward, done, event = env.resolve_direction(approach, actions.UP)
    assert next_state[:2] == second_pad
    assert reward == env.step_cost + env.teleport_bonus
    assert not done
    assert event["used_teleport"]


def test_the_teleporter_does_not_loop(env):
    """18. The teleporter does not create an infinite loop.

    It resolves exactly once per transition: the robot lands on the paired pad
    and stays there, rather than being bounced straight back.
    """
    first_pad, second_pad = map_data.teleport_cells()
    approach = (first_pad[0] + 1, first_pad[1], False, actions.UP)
    next_state, _, _, _ = env.resolve_direction(approach, actions.UP)
    assert next_state[:2] == second_pad
    assert next_state[:2] != first_pad
    # The previous direction is cleared, so oil momentum cannot carry across.
    assert next_state[3] == actions.NONE


def test_the_teleport_bonus_is_never_paid_twice(env):
    """A merged transition must not stack the bonus."""
    first_pad, _ = map_data.teleport_cells()
    approach = (first_pad[0] + 1, first_pad[1], False, actions.UP)
    for probability, _, reward, _ in env.transitions(approach, actions.UP):
        assert reward <= env.step_cost + env.teleport_bonus


# ----------------------------------------------------------------------
# The one-way door
# ----------------------------------------------------------------------

def test_the_one_way_door_blocks_the_wrong_direction(env):
    """19. The one-way door blocks movement in the wrong direction."""
    door_row, door_col = 4, 7
    assert map_data.tile_at(door_row, door_col) == tiles.DOOR_RIGHT

    # Trying to enter it from the right, moving left, is refused.
    from_the_right = (door_row, door_col + 1, False, actions.LEFT)
    next_state, reward, _, event = env.resolve_direction(from_the_right, actions.LEFT)
    assert next_state[:2] == (door_row, door_col + 1)
    assert reward == env.step_cost + env.wall_penalty
    assert event["blocked"]

    # Standing on the door, only the arrow direction may be taken.
    on_the_door = (door_row, door_col, False, actions.RIGHT)
    blocked_state, blocked_reward, _, _ = env.resolve_direction(on_the_door,
                                                               actions.LEFT)
    assert blocked_state[:2] == (door_row, door_col)
    assert blocked_reward == env.step_cost + env.wall_penalty


def test_the_one_way_door_allows_the_right_direction(env):
    """20. The one-way door allows movement in the correct direction."""
    door_row, door_col = 4, 7
    approach = (door_row, door_col - 1, False, actions.RIGHT)
    next_state, reward, done, event = env.resolve_direction(approach, actions.RIGHT)
    assert next_state[:2] == (door_row, door_col)
    assert reward == env.step_cost
    assert not event["blocked"]

    onwards = (door_row, door_col, False, actions.RIGHT)
    after_state, after_reward, _, _ = env.resolve_direction(onwards, actions.RIGHT)
    assert after_state[:2] == (door_row, door_col + 1)
    assert after_reward == env.step_cost


# ----------------------------------------------------------------------
# The exit
# ----------------------------------------------------------------------

def test_reaching_the_exit_ends_the_episode(env):
    """21. The exit terminates the episode."""
    exit_row, exit_col = map_data.exit_cell()
    approach = (exit_row, exit_col - 1, False, actions.RIGHT)
    next_state, reward, done, event = env.resolve_direction(approach, actions.RIGHT)
    assert next_state[:2] == (exit_row, exit_col)
    assert done
    assert reward == env.step_cost + env.exit_reward
    assert reward == 99
    assert event["reached_exit"]
    assert env.is_terminal(next_state)


def test_a_terminal_state_has_no_outgoing_transitions(env):
    exit_row, exit_col = map_data.exit_cell()
    terminal = (exit_row, exit_col, False, actions.RIGHT)
    assert env.is_terminal(terminal)
    for action in env.actions():
        outcomes = env.transitions(terminal, action)
        assert outcomes == [(1.0, terminal, 0.0, True)]


def test_requiring_the_battery_locks_the_exit():
    """The optional toggle turns the exit into a locked door."""
    env = Room1Env(require_battery=True)
    exit_row, exit_col = map_data.exit_cell()

    without = (exit_row, exit_col - 1, False, actions.RIGHT)
    next_state, reward, done, event = env.resolve_direction(without, actions.RIGHT)
    assert next_state[:2] == (exit_row, exit_col - 1)
    assert not done
    assert reward == env.step_cost + env.wall_penalty
    assert event["blocked"]

    with_battery = (exit_row, exit_col - 1, True, actions.RIGHT)
    open_state, open_reward, open_done, _ = env.resolve_direction(with_battery,
                                                                 actions.RIGHT)
    assert open_state[:2] == (exit_row, exit_col)
    assert open_done
    assert open_reward == env.step_cost + env.exit_reward


# ----------------------------------------------------------------------
# reset and step
# ----------------------------------------------------------------------

def test_reset_puts_the_robot_on_the_start_platform_with_no_momentum(env):
    state = env.reset(seed=0)
    assert state == (0, 0, False, actions.NONE)
    assert state == env.start_state()


def test_step_is_reproducible_for_a_given_seed(env):
    def walk():
        env.reset(seed=1234)
        return [env.step(actions.DOWN) for _ in range(12)]

    first = walk()
    second = walk()
    assert [(s, r, d) for s, r, d, _ in first] == [(s, r, d) for s, r, d, _ in second]


def test_step_reports_slipping(env):
    """A step that did not go the intended way is reported as a slip."""
    env.reset(seed=3)
    env.state = (8, 4, False, actions.NONE)      # a wet cell
    seen_slip = False
    for _ in range(60):
        env.state = (8, 4, False, actions.NONE)
        _, _, _, info = env.step(actions.UP)
        if info["slipped"]:
            seen_slip = True
            assert info["actual_direction"] != actions.UP
            break
    assert seen_slip, "60 attempts on wet floor produced no slip at all"
