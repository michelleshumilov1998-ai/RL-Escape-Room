"""Tests for the Room 2 map, environment and SARSA agent.

Covers the checks the Room 2 brief asks for: map size, one start, one exit, one
keycard, bridges collapsing and becoming pits, pits ending the episode, the door
staying locked without the keycard and opening with it, and SARSA updating
correctly.
"""

from collections import Counter

import pytest

from core import actions
from rooms.room2 import map_data, sarsa_agent
from rooms.room2.environment import Room2Env


@pytest.fixture
def env():
    return Room2Env()


def _counts():
    counter = Counter()
    for row in map_data.ROOM_MAP:
        counter.update(row)
    return counter


# ----------------------------------------------------------------------
# The map
# ----------------------------------------------------------------------

def test_map_size_is_ten_by_ten():
    assert len(map_data.ROOM_MAP) == 10
    assert map_data.GRID_ROWS == 10 and map_data.GRID_COLS == 10
    for index, row in enumerate(map_data.ROOM_MAP):
        assert len(row) == 10, "row %d has %d cells" % (index, len(row))


def test_exactly_one_start_one_exit_one_keycard():
    counts = _counts()
    assert counts[map_data.START] == 1
    assert counts[map_data.EXIT] == 1
    assert counts[map_data.KEYCARD] == 1
    assert map_data.start_cell() == (8, 1)
    assert map_data.exit_cell() == (8, 8)
    assert map_data.keycard_cell() == (7, 1)


def test_the_map_has_the_pieces_the_brief_asks_for():
    counts = _counts()
    assert counts[map_data.BRIDGE] >= 3, "several normal bridges"
    assert counts[map_data.COLLAPSING] >= 3, "several collapsing bridges"
    assert len(map_data.enterable_pits()) >= 5, "at least five deadly pits"
    assert counts[map_data.WALL] >= 4, "metallic walls"


def test_every_character_is_valid():
    for row in map_data.ROOM_MAP:
        for character in row:
            assert character in map_data.VALID_TILES, "unknown tile %r" % character


def test_every_walkable_cell_is_reachable():
    walkable = set(map_data.walkable_cells())
    unreachable = walkable - map_data.reachable_cells()
    assert unreachable == set(), "unreachable cells: %s" % sorted(unreachable)


def test_there_are_two_routes_and_the_short_one_is_the_risky_one():
    """The trade-off the whole room is built on."""
    measured = map_data.route_lengths()
    assert measured == map_data.ROUTE_LENGTHS, "route lengths changed: %s" % measured
    assert measured["span"] < measured["walkway"]

    # Only the short route has anything to fall off.
    exposed = set(map_data.exposed_cells())
    assert exposed, "the short route has no exposure at all"
    walkway = {(1, col) for col in range(map_data.GRID_COLS)}
    assert not exposed & walkway, "the long route should be safe"


def test_the_short_route_pays_more_when_walked_perfectly():
    """If this stops being true, the room stops posing a dilemma."""
    assert map_data.perfect_return("span") > map_data.perfect_return("walkway")


def test_map_hash_is_stable():
    fingerprint = map_data.map_hash()
    assert len(fingerprint) == 16
    assert map_data.map_hash() == fingerprint


# ----------------------------------------------------------------------
# The state space
# ----------------------------------------------------------------------

def test_the_state_is_row_col_and_keycard(env):
    state = env.reset()
    assert state == (8, 1, False)
    assert len(state) == 3
    keycard_values = {state[2] for state in env.all_states()}
    assert keycard_values == {False, True}


def test_the_environment_offers_no_model(env):
    """SARSA has to learn, so the model must not be there to read."""
    assert not hasattr(env, "transitions")


def test_every_observation_is_covered_by_all_states(env):
    states = set(env.all_states())
    env.reset(seed=0)
    assert env.observation() in states
    for action in (actions.UP, actions.UP, actions.UP, actions.RIGHT, actions.RIGHT):
        env.step(action)
        assert env.observation() in states


def test_the_optional_bridge_state_makes_the_state_longer():
    env = Room2Env(include_bridge_state=True)
    state = env.reset()
    assert len(state) == 4
    assert state == (8, 1, False, 0)
    assert env.observation() in set(env.all_states())


# ----------------------------------------------------------------------
# Collapsing bridges
# ----------------------------------------------------------------------

def test_crossing_a_collapsing_bridge_collapses_it(env):
    """Bridge collapse works."""
    first_bridge = map_data.collapsing_cells()[0]
    env.reset()
    _walk_to_the_bridge(env)

    state, reward, done, info = env.step(actions.RIGHT)
    assert (state[0], state[1]) == first_bridge
    assert info["bridge_collapsed"]
    assert tuple(info["collapsed_cell"]) == first_bridge
    assert first_bridge in env.collapsed
    assert reward == env.step_cost + env.collapsing_penalty
    assert not done


def test_a_collapsed_bridge_becomes_a_pit(env):
    """Bridge becomes pit: going back onto it ends the episode."""
    env.reset()
    _walk_to_the_bridge(env)
    env.step(actions.RIGHT)       # onto the first bridge, which collapses
    env.step(actions.RIGHT)       # onto the second

    state, reward, done, info = env.step(actions.LEFT)   # back onto the first
    assert done, "a collapsed bridge must end the episode"
    assert info["pit_fall"]
    assert reward == env.step_cost + env.pit_penalty


def test_reset_rebuilds_every_bridge(env):
    env.reset()
    _walk_to_the_bridge(env)
    env.step(actions.RIGHT)
    assert env.collapsed
    env.reset()
    assert env.collapsed == set()


def test_a_bridge_that_has_not_been_crossed_is_safe(env):
    env.reset()
    _walk_to_the_bridge(env)
    state, reward, done, info = env.step(actions.RIGHT)
    assert not done
    assert not info["pit_fall"]


def _walk_to_the_bridge(env):
    """Walk from the start platform to the cell just left of the first bridge."""
    for _ in range(3):
        env.step(actions.UP)          # (7,1) keycard, (6,1), (5,1)
    env.step(actions.RIGHT)           # (5,2), the cell before the first bridge
    assert (env.state[0], env.state[1]) == (5, 2)


# ----------------------------------------------------------------------
# Pits
# ----------------------------------------------------------------------

def test_a_pit_ends_the_episode(env):
    """Pits terminate the episode."""
    env.reset()
    _walk_to_the_bridge(env)
    env.step(actions.RIGHT)                      # standing on a bridge
    state, reward, done, info = env.step(actions.DOWN)   # step off it
    assert done
    assert info["pit_fall"]
    assert reward == env.step_cost + env.pit_penalty
    assert env.is_terminal(state)


def test_the_pit_penalty_is_configurable():
    env = Room2Env(pit_penalty=-250)
    env.reset()
    _walk_to_the_bridge(env)
    env.step(actions.RIGHT)
    _, reward, _, _ = env.step(actions.DOWN)
    assert reward == -251


# ----------------------------------------------------------------------
# The keycard and the door
# ----------------------------------------------------------------------

def test_the_keycard_is_collected_once_and_only_once(env):
    env.reset()
    state, reward, done, info = env.step(actions.UP)
    assert (state[0], state[1]) == map_data.keycard_cell()
    assert state[2] is True
    assert info["keycard_collected"]
    assert reward == env.step_cost + env.keycard_bonus

    # Leaving and coming back pays only the step cost.
    env.step(actions.DOWN)
    _, second_reward, _, second_info = env.step(actions.UP)
    assert second_reward == env.step_cost
    assert not second_info["keycard_collected"]


def test_the_door_stays_locked_without_the_keycard(env):
    """Door remains locked without keycard."""
    env.reset()
    env.state = (8, 7, False)                 # next to the door, no keycard
    state, reward, done, info = env.step(actions.RIGHT)
    assert (state[0], state[1]) == (8, 7), "the robot must not get through"
    assert not done
    assert info["door_locked"]
    assert reward == env.step_cost + env.wall_penalty


def test_the_door_opens_once_the_keycard_is_held(env):
    """Door opens after keycard."""
    env.reset()
    env.state = (8, 7, True)
    state, reward, done, info = env.step(actions.RIGHT)
    assert (state[0], state[1]) == map_data.exit_cell()
    assert done
    assert info["reached_exit"]
    assert reward == env.step_cost + env.exit_reward
    assert env.is_terminal(state)


def test_walls_keep_the_robot_in_place(env):
    env.reset()
    state, reward, done, info = env.step(actions.LEFT)   # (8,0) is a wall
    assert (state[0], state[1]) == (8, 1)
    assert reward == env.step_cost + env.wall_penalty
    assert info["blocked"]
    assert not done


def test_the_environment_is_deterministic(env):
    """The danger comes from exploring, not from the environment."""
    def walk():
        env.reset(seed=1)
        return [env.step(action) for action in
                (actions.UP, actions.UP, actions.UP, actions.RIGHT, actions.RIGHT)]
    first = [(s, r, d) for s, r, d, _ in walk()]
    second = [(s, r, d) for s, r, d, _ in walk()]
    assert first == second


# ----------------------------------------------------------------------
# SARSA
# ----------------------------------------------------------------------

def test_the_sarsa_update_moves_q_by_alpha_times_the_error():
    """SARSA updates correctly: one update, checked by hand."""
    env = Room2Env()
    q = sarsa_agent.new_q_table(env, initial_value=0.0)
    state = (8, 1, False)
    action = actions.UP
    alpha, gamma = 0.5, 0.9

    q[state][action] = 2.0
    next_state = (7, 1, True)
    next_action = actions.UP
    q[next_state][next_action] = 10.0

    reward = 4.0
    target = reward + gamma * q[next_state][next_action]      # 4 + 0.9*10 = 13
    expected = 2.0 + alpha * (target - 2.0)                   # 2 + 0.5*11 = 7.5

    q[state][action] = q[state][action] + alpha * (target - q[state][action])
    assert q[state][action] == pytest.approx(7.5)
    assert q[state][action] == pytest.approx(expected)


def test_sarsa_learns_to_escape():
    env = Room2Env()
    result = sarsa_agent.train(env, algorithm=sarsa_agent.SARSA, episodes=800,
                               seed=0)
    tail = result["history"]["success"][-200:]
    assert sum(tail) / len(tail) >= 0.8, "SARSA did not learn to reach the exit"
    assert result["runtime_seconds"] > 0


def test_training_records_everything_the_graphs_need():
    env = Room2Env()
    result = sarsa_agent.train(env, episodes=100, seed=0)
    for key in ("reward", "length", "success", "epsilon", "mean_abs_q",
                "pit_fall", "used_span", "keycard", "elapsed"):
        assert key in result["history"], "history is missing %r" % key
        assert len(result["history"][key]) == 100


def test_epsilon_decays_to_its_floor_and_no_further():
    env = Room2Env()
    result = sarsa_agent.train(env, episodes=400, seed=0, epsilon_start=1.0,
                               epsilon_min=0.10, epsilon_decay=0.99)
    epsilons = result["history"]["epsilon"]
    assert epsilons[0] < 1.0
    assert epsilons[-1] == pytest.approx(0.10, abs=1e-9)
    assert all(value >= 0.10 - 1e-12 for value in epsilons)


def test_the_greedy_policy_covers_every_non_terminal_state():
    env = Room2Env()
    result = sarsa_agent.train(env, episodes=100, seed=0)
    for state in env.all_states():
        action = result["policy"][state]
        if env.is_terminal(state):
            assert action is None
        else:
            assert action in env.actions()


def test_training_is_reproducible_from_a_seed():
    def run():
        return sarsa_agent.train(Room2Env(), episodes=150, seed=42)["history"]["reward"]
    assert run() == run()


def test_a_worked_example_update_is_recorded():
    """The interface shows one real update, so one has to be captured."""
    result = sarsa_agent.train(Room2Env(), episodes=50, seed=0)
    example = result["example_update"]
    assert example is not None
    for key in ("state", "action", "reward", "next_state", "q_before", "q_after",
                "target", "alpha", "gamma"):
        assert key in example
    # The stored arithmetic must actually hold.
    expected = example["q_before"] + example["alpha"] * (
        example["target"] - example["q_before"])
    assert example["q_after"] == pytest.approx(expected)


def test_q_learning_is_available_for_comparison():
    result = sarsa_agent.train(Room2Env(), algorithm=sarsa_agent.Q_LEARNING,
                               episodes=200, seed=0)
    assert result["algorithm"] == sarsa_agent.Q_LEARNING
    assert result["history"]["reward"]


def test_with_the_markov_state_q_learning_takes_the_bridge_and_sarsa_does_not():
    """The headline result of Room 2.

    Only holds once the agent can see which bridges have already fallen; with the
    assignment's state both methods avoid the bridge, which is the point of the
    state-representation experiment. Trained long enough to be stable, and checked
    loosely so it is not brittle.
    """
    episodes = 2500
    sarsa = sarsa_agent.train(Room2Env(include_bridge_state=True),
                              algorithm=sarsa_agent.SARSA, episodes=episodes, seed=0)
    q_learning = sarsa_agent.train(Room2Env(include_bridge_state=True),
                                   algorithm=sarsa_agent.Q_LEARNING,
                                   episodes=episodes, seed=0)

    def tail(result, key):
        values = result["history"][key][-400:]
        return sum(values) / len(values)

    assert tail(q_learning, "used_span") > 0.5, "Q-Learning should take the bridge"
    assert tail(sarsa, "used_span") < 0.2, "SARSA should avoid the bridge"
    assert tail(sarsa, "reward") > tail(q_learning, "reward"), \
        "SARSA should collect more reward online"
    assert tail(q_learning, "pit_fall") > tail(sarsa, "pit_fall")
