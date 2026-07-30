"""Value Iteration — room 1's assignment default.

One sweep applies the Bellman optimality update to every state:

    V(s) <- max_a  Σ P(s'|s,a) [ R + γ V(s') ]

and the run is finished when a whole sweep moves no value by more than θ.
The policy is read off at the end with argmax_a Q(s,a).

Everything it needs comes from the model, so it never takes a step in the
environment.  That is the whole point of the room: the agent can route
around a hazard it has never touched, because the hazard is already in the
model.  Something learning from experience would have to be hit first.
"""

from game.algorithms.base import Planner, best_action


class ValueIteration(Planner):

    key = "value_iteration"
    label = "Value Iteration"
    parameters = ("gamma", "theta", "slip")
    summary = ("Sweeps the whole grid over and over, each time replacing a "
               "cell's value with the value of its best action, until the "
               "numbers stop moving.")
    update_rule = "V(s) ← max_a Σ P(s'|s,a) [ R + γ V(s') ]"
    watch_for = ("Value spreads outwards from the goal, one sweep at a "
                 "time, and flows around the hazards rather than through "
                 "them. Watch the arrows settle before the numbers do.")

    def update(self, transition=None):
        """One sweep over every state. The transition argument is unused:
        a planner reads the model instead of living through anything."""
        delta = 0.0

        for state in self.states:
            if self.env.is_terminal(state):
                # A terminal state has no future, so its value stays 0.
                continue
            _, new_value = best_action(self.env, state, self.values, self.gamma)
            change = abs(new_value - self.values[state])
            if change > delta:
                delta = change
            self.values[state] = new_value

        self.sweeps += 1
        self.last_delta = delta
        self.converged = delta < self.theta

        return {"sweeps": self.sweeps, "delta": delta,
                "converged": self.converged}
