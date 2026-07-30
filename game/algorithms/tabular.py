"""What every tabular, model-free method shares.

The Q-table, epsilon-greedy action selection, how a policy is read off, and
what a snapshot looks like — all identical across the four methods.  What is
*not* here is the target the update moves towards, because that single
expression is the entire difference between them:

    SARSA              r + γ Q(s', a')            the action actually next
    Expected SARSA     r + γ Σ π(a|s') Q(s',a)    the average over the policy
    Q-Learning         r + γ max_a Q(s',a)        the best action available
    Double Q-Learning  r + γ Q_b(s', argmax Q_a)  chosen by one, valued by other

Each subclass therefore overrides one method.  If implementing the second one
had required touching the room, the layering would be wrong.

None of them is given the environment's model. `env.transitions()` exists on
the grid, but nothing below ever calls it — the only way these methods find
out what an action does is to take it.
"""

from game.algorithms.base import Algorithm, project_to_cells


class TabularLearner(Algorithm):
    """A Q-table learned from experience."""

    needs_model = False
    family = "Temporal-Difference Learning"
    parameters = ("alpha", "gamma", "epsilon", "epsilon_min", "epsilon_decay",
                  "slip", "q_init", "episodes")

    def __init__(self, env, parameters, rng=None):
        Algorithm.__init__(self, env, parameters, rng)
        self.alpha = parameters["alpha"]
        self.gamma = parameters["gamma"]
        self.epsilon = parameters["epsilon"]
        self.epsilon_min = parameters["epsilon_min"]
        self.epsilon_decay = parameters["epsilon_decay"]
        self.episode_target = int(parameters.get("episodes", 500))

        self.action_list = env.actions()
        # Optimistic initialisation. A table of zeros makes an untried action
        # look no better than a tried one, so a route that fails early gets
        # abandoned and never reconsidered — which is exactly how this room
        # loses the comparison it exists to make. Starting above any reward
        # the room can pay makes the agent evaluate both routes properly.
        self.q_init = float(parameters.get("q_init", 0.0))
        self.q = {state: {action: self.q_init for action in self.action_list}
                  for state in env.all_states()}
        self.episodes = 0

    # ------------------------------------------------------------------
    # Reading the table
    # ------------------------------------------------------------------

    def _table(self, state):
        """The table `act` and `greedy_policy` consult.

        Double Q-Learning overrides this to use the sum of its two tables;
        everything else is unaffected by that.
        """
        return self.q[state]

    def best_actions(self, state, tolerance=1e-12):
        values = self._table(state)
        highest = max(values[action] for action in self.action_list)
        return [action for action in self.action_list
                if values[action] >= highest - tolerance]

    def best_action(self, state):
        return self.best_actions(state)[0]

    def act(self, state):
        """Mostly the best known action, ε of the time a random one.

        Ties are broken *randomly*. A fresh table is all zeros, so every
        action ties at the start; always taking the first would send the
        agent the same way on every early episode and whole parts of the
        room would never be seen at all.
        """
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.action_list)
        tied = self.best_actions(state)
        if len(tied) == 1:
            return tied[0]
        return self.rng.choice(tied)

    def action_probabilities(self, state):
        """The ε-greedy policy's distribution — what Expected SARSA averages
        over, and worth having in one place so it cannot drift from `act`."""
        share = self.epsilon / len(self.action_list)
        probabilities = {action: share for action in self.action_list}
        tied = self.best_actions(state)
        for action in tied:
            probabilities[action] += (1.0 - self.epsilon) / len(tied)
        return probabilities

    def greedy_policy(self):
        policy = {}
        for state in self.q:
            terminal = self.env.is_terminal(state)
            policy[state] = None if terminal else self.best_action(state)
        return policy

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def target(self, transition):
        """Where the estimate should move to. The one thing that differs."""
        raise NotImplementedError

    def update(self, transition):
        """One step of experience, folded into the table."""
        target = self.target(transition)
        before = self.q[transition.state][transition.action]
        error = target - before
        self.q[transition.state][transition.action] = before + self.alpha * error
        return {"tdError": error, "target": target, "qBefore": before,
                "qAfter": self.q[transition.state][transition.action]}

    def end_episode(self):
        """Called once an episode finishes: decay exploration."""
        self.episodes += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @property
    def finished(self):
        return self.episodes >= self.episode_target

    # ------------------------------------------------------------------

    def state_values(self):
        """max_a Q(s,a) per state — what the heatmap draws."""
        return {state: max(self._table(state)[action]
                           for action in self.action_list)
                for state in self.q}

    def mean_absolute_q(self):
        total = 0.0
        count = 0
        for state in self.q:
            for action in self.action_list:
                total += abs(self.q[state][action])
                count += 1
        return total / count if count else 0.0

    def snapshot(self):
        values = self.state_values()
        policy = self.greedy_policy()
        # Several states share one square, so the table is projected onto the
        # grid exactly the way a planner's is — see `project_to_cells`.
        projected, arrows = project_to_cells(values, policy)
        return {
            "values": projected,
            "policy": arrows,
            "episodes": self.episodes,
            "episodeTarget": self.episode_target,
            "epsilon": self.epsilon,
            "meanAbsQ": self.mean_absolute_q(),
            "converged": self.finished,
            "startValue": values[self.env.start_state()],
        }

    def set_parameter(self, name, value):
        Algorithm.set_parameter(self, name, value)
        if name == "episodes":
            self.episode_target = int(value)
