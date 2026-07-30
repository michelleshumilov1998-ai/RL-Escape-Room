"""The algorithm registry.

Each algorithm lives in its own file and is listed here once. A room says
which keys it allows and which one is the assignment's default; it never
imports an algorithm class directly, so adding one is a change to this file
and nothing else.
"""

from game.algorithms.double_q_learning import DoubleQLearning
from game.algorithms.expected_sarsa import ExpectedSarsa
from game.algorithms.policy_iteration import PolicyIteration
from game.algorithms.q_learning import QLearning
from game.algorithms.sarsa import Sarsa
from game.algorithms.value_iteration import ValueIteration

ALGORITHMS = {
    ValueIteration.key: ValueIteration,
    PolicyIteration.key: PolicyIteration,
    Sarsa.key: Sarsa,
    ExpectedSarsa.key: ExpectedSarsa,
    QLearning.key: QLearning,
    DoubleQLearning.key: DoubleQLearning,
}


def get(key):
    """The class for one key."""
    if key not in ALGORITHMS:
        raise KeyError("no algorithm named %r" % key)
    return ALGORITHMS[key]


def build(key, env, parameters, rng=None):
    """A fresh algorithm instance, wired to an environment."""
    return get(key)(env, parameters, rng)


def describe_all(keys):
    """The metadata for a room's allowed algorithms, in the order given."""
    return [get(key).describe() for key in keys]
