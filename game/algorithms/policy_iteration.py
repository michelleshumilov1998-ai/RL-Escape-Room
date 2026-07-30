"""Policy Iteration — room 1's alternative.

Instead of improving the values and reading a policy off at the end, this
alternates two steps:

    evaluate    V(s) <- Q(s, π(s))   until the values settle
    improve     π(s) <- argmax_a Q(s,a)

and stops when an improvement step changes no action anywhere.  One call to
`update` is one full round of both, so the interface still means the same
thing it does for Value Iteration: one unit of learning.

It reaches the same answer as Value Iteration but does considerably more
total sweeps, because each evaluation is run to convergence rather than
truncated after one pass.  Having both available is what makes that
visible rather than a claim.
"""

from game.algorithms.base import Planner, q_value


class PolicyIteration(Planner):

    key = "policy_iteration"
    label = "Policy Iteration"
    parameters = ("gamma", "theta", "slip")
    summary = ("Measures how good its current plan is, then improves the "
               "plan, and repeats until improving it changes nothing.")
    update_rule = ("evaluate: V(s) ← Q(s, π(s))   ·   "
                   "improve: π(s) ← argmax_a Q(s,a)")
    watch_for = ("The arrows change in whole blocks rather than drifting: "
                 "each round the plan is re-cut all at once. It settles in "
                 "very few rounds, but each round costs many sweeps.")

    # A single evaluation is capped so a near-1 discount factor cannot spin.
    EVALUATION_SWEEPS_MAX = 1000

    def __init__(self, env, parameters, rng=None):
        Planner.__init__(self, env, parameters, rng)
        # Step 1: an arbitrary starting policy — the first action everywhere.
        first_action = env.actions()[0]
        self.policy = {}
        for state in self.states:
            terminal = env.is_terminal(state)
            self.policy[state] = None if terminal else first_action
        self.rounds = 0
        self.evaluation_sweeps = 0

    def _evaluate(self):
        """Run V(s) = Q(s, π(s)) until it settles. Returns the biggest move."""
        biggest_move = 0.0

        for _ in range(self.EVALUATION_SWEEPS_MAX):
            self.evaluation_sweeps += 1
            self.sweeps += 1
            delta = 0.0
            for state in self.states:
                if self.env.is_terminal(state):
                    continue
                new_value = q_value(self.env, state, self.policy[state],
                                    self.values, self.gamma)
                change = abs(new_value - self.values[state])
                if change > delta:
                    delta = change
                self.values[state] = new_value
            if delta > biggest_move:
                biggest_move = delta
            if delta < self.theta:
                break

        return biggest_move

    def update(self, transition=None):
        """One round: evaluate the current plan, then improve it."""
        biggest_move = self._evaluate()

        # Step 3: improve. `greedy_policy` is already argmax over the model.
        improved = self.greedy_policy()
        changes = 0
        for state in self.states:
            if improved[state] != self.policy[state]:
                changes += 1
        self.policy = improved

        self.rounds += 1
        self.last_delta = biggest_move
        # Step 4: nothing changed, so the plan is optimal.
        self.converged = changes == 0

        return {"sweeps": self.sweeps, "rounds": self.rounds,
                "delta": biggest_move, "policyChanges": changes,
                "converged": self.converged}

    def snapshot(self):
        state = Planner.snapshot(self)
        state["rounds"] = self.rounds
        state["evaluationSweeps"] = self.evaluation_sweeps
        return state
