"""Expected SARSA.

    Q(s,a) <- Q(s,a) + α [ r + γ Σ_a' π(a'|s') Q(s',a') - Q(s,a) ]

Between the other two. Where SARSA uses the one action it happened to draw
and Q-Learning uses the best one, this uses the *average over the policy* —
every action weighted by how likely the ε-greedy policy is to pick it.

It is still on-policy, so like SARSA it accounts for exploration; but it
removes the variance that comes from which action the draw happened to
land on, which usually makes its learning curve visibly smoother.
"""

from game.algorithms.tabular import TabularLearner


class ExpectedSarsa(TabularLearner):

    key = "expected_sarsa"
    label = "Expected SARSA"
    summary = ("Like SARSA, but averages over every action the policy might "
               "take next instead of using the single one it drew.")
    update_rule = "Q(s,a) ← Q(s,a) + α [ r + γ Σ π(a'|s') Q(s',a') − Q(s,a) ]"
    watch_for = ("A smoother reward curve than SARSA's for the same route. "
                 "Averaging over the policy takes out the noise of which "
                 "action the draw happened to pick.")

    def target(self, transition):
        if transition.done:
            return transition.reward
        # The same distribution `act` samples from, so the two cannot drift.
        probabilities = self.action_probabilities(transition.next_state)
        values = self.q[transition.next_state]
        expected = sum(probabilities[action] * values[action]
                       for action in self.action_list)
        return transition.reward + self.gamma * expected
