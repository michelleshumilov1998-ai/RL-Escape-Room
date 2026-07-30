"""The one interface every algorithm implements.

Three methods, and nothing else the room is allowed to care about:

    act(state)            what to do now, under the behaviour policy
    update(transition)    one unit of learning
    greedy_policy()       the policy with the exploration taken out

The point of keeping it this narrow is that swapping one algorithm for
another must never require touching a room.  A room builds an environment,
is handed an algorithm object, and drives the loop; it does not know
whether the thing it is driving bootstraps from the action it is about to
take or from the best one available.

`update` means slightly different things to the two families, and
deliberately so:

  * A **learner** is given a transition it just lived through.
  * A **planner** already has the model and ignores the argument entirely;
    one call is one sweep over the state space.

Both are still "one unit of learning", which is all the session loop needs
to know in order to count progress and decide when a run is finished.
"""

from collections import namedtuple

# What a learner is told about the step that just happened.  `next_action`
# is filled in by the session before the update, which is what lets SARSA
# and Q-Learning differ in exactly one line: SARSA uses it, Q-Learning
# ignores it.
Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "next_action", "done"))


class Algorithm:
    """The contract. Subclasses override the three methods and the metadata."""

    # Identity, used by the registry and shown in the selector.
    key = ""
    label = ""
    family = ""
    # Whether it reads `env.transitions()`. Planners do; learners must not.
    needs_model = False
    # Which entries of config.PARAMETERS this algorithm exposes.
    parameters = ()
    # One line for the sidebar's algorithm section.
    summary = ""
    update_rule = ""
    watch_for = ""

    def __init__(self, env, parameters, rng=None):
        self.env = env
        self.parameters = dict(parameters)
        self.rng = rng

    # -- the interface --------------------------------------------------

    def act(self, state):
        raise NotImplementedError

    def update(self, transition=None):
        """One unit of learning. Returns a small dict of progress numbers."""
        raise NotImplementedError

    def greedy_policy(self):
        """{state: action}, with no exploration left in it."""
        raise NotImplementedError

    # -- shared behaviour ----------------------------------------------

    def set_parameter(self, name, value):
        """Change a live parameter. Reset-scope ones are handled by the
        session, which rebuilds the algorithm instead of mutating it."""
        self.parameters[name] = value
        setattr(self, name, value)

    def snapshot(self):
        """Whatever the renderer and the status strip need. Serialisable."""
        return {}

    @property
    def finished(self):
        """True once there is no point continuing to train."""
        return False

    @classmethod
    def describe(cls):
        return {
            "key": cls.key,
            "label": cls.label,
            "family": cls.family,
            "needsModel": cls.needs_model,
            "summary": cls.summary,
            "updateRule": cls.update_rule,
            "watchFor": cls.watch_for,
        }


# ----------------------------------------------------------------------
# Shared by the planners
# ----------------------------------------------------------------------

def q_value(env, state, action, values, gamma):
    """The expected value of one action, read straight off the model.

        Q(s,a) = Σ P(s'|s,a) [ R + γ V(s') ]

    Both Dynamic Programming methods are built from this one line.
    """
    total = 0.0
    for probability, next_state, reward, done in env.transitions(state, action):
        future = 0.0 if done else values[next_state]
        total += probability * (reward + gamma * future)
    return total


def best_action(env, state, values, gamma):
    """The highest-valued action, and its value.

    Ties go to the first action in the environment's own order, so the same
    value table always reads back the same policy.
    """
    chosen = None
    chosen_value = None
    for action in env.actions():
        value = q_value(env, state, action, values, gamma)
        if chosen_value is None or value > chosen_value:
            chosen = action
            chosen_value = value
    return chosen, chosen_value


def project_to_cells(values, policy):
    """Collapse a state table onto the one grid the page actually draws.

    A room's state carries more than a cell — a battery in hand, the
    direction it was last pushed — so several states share one square. Each
    square therefore shows its best state, and the arrow shown is that same
    state's, so the colour and the arrow can never disagree about which
    state they are describing.

    Both families project through here on purpose. A planner's values and a
    learner's are meant to be read against each other, and they could not be
    if one collapsed the extra dimensions differently from the other.

    Returns (values by "row,col", actions by "row,col").
    """
    best = {}
    for state, value in values.items():
        key = "%d,%d" % (state[0], state[1])
        if key not in best or value > best[key][0]:
            best[key] = (value, state)

    arrows = {}
    for key, (_, state) in best.items():
        action = policy.get(state)
        if action is not None:
            arrows[key] = action

    return {key: value for key, (value, _) in best.items()}, arrows


class Planner(Algorithm):
    """Common ground for the methods that are handed the model.

    They share a value table, how a policy is read off it, and what a
    snapshot looks like; they differ only in what one sweep does.
    """

    needs_model = True
    family = "Dynamic Programming"

    def __init__(self, env, parameters, rng=None):
        Algorithm.__init__(self, env, parameters, rng)
        self.gamma = parameters["gamma"]
        self.theta = parameters["theta"]
        self.states = env.all_states()
        self.values = {state: 0.0 for state in self.states}
        self.sweeps = 0
        self.last_delta = float("inf")
        self.converged = False

    def act(self, state):
        """Planners have no exploration: acting is always greedy."""
        action, _ = best_action(self.env, state, self.values, self.gamma)
        return action

    def greedy_policy(self):
        policy = {}
        for state in self.states:
            if self.env.is_terminal(state):
                policy[state] = None
                continue
            action, _ = best_action(self.env, state, self.values, self.gamma)
            policy[state] = action
        return policy

    @property
    def finished(self):
        return self.converged

    def snapshot(self):
        policy = self.greedy_policy()
        values, arrows = project_to_cells(self.values, policy)

        return {
            "values": values,
            "policy": arrows,
            "sweeps": self.sweeps,
            # Before the first sweep there is no measured change. None rather
            # than infinity, because infinity is not a value JSON can carry.
            "delta": None if self.last_delta == float("inf") else self.last_delta,
            "converged": self.converged,
            "startValue": self.values[self.env.start_state()],
        }
