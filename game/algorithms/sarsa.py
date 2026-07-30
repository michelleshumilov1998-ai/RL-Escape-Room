"""SARSA — room 2's assignment default.

    Q(s,a) <- Q(s,a) + α [ r + γ Q(s',a') - Q(s,a) ]

`a'` is the action the agent is *actually going to take next*, drawn from the
same ε-greedy policy it is following. That makes SARSA **on-policy**: it
learns how good the policy it is really following is, exploration mistakes
and all.

In this room that has a visible consequence. The ledge above the cliff is the
shorter way across, and it pays more when walked perfectly. But an ε-greedy
walk along it occasionally steps off, and the ice occasionally pushes the
agent off even when it does not. SARSA's target contains exactly those
outcomes, so the ledge is valued lower than it would be under a perfect
policy — and it takes the longer way round instead.
"""

from game.algorithms.tabular import TabularLearner


class Sarsa(TabularLearner):

    key = "sarsa"
    label = "SARSA"
    summary = ("Learns the value of the policy it is actually following, "
               "including the exploratory steps that go wrong.")
    update_rule = "Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') − Q(s,a) ]"
    watch_for = ("It should settle on the longer route away from the cliff. "
                 "It is paying for its own mistakes, so the short ledge "
                 "never looks as good to it as it does on paper.")

    def target(self, transition):
        if transition.done:
            # Nothing follows a terminal state, so there is nothing to
            # bootstrap from.
            return transition.reward
        # The action that will genuinely be taken next — that is the whole
        # of the difference from Q-Learning.
        following = self.q[transition.next_state][transition.next_action]
        return transition.reward + self.gamma * following
