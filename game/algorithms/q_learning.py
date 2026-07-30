"""Q-Learning.

    Q(s,a) <- Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]

The `max` is the entire difference from SARSA. It bootstraps from the best
action available in the next state whether or not that is the action it is
about to take, so it learns the value of behaving greedily while it is still
exploring. That is what **off-policy** means.

Here that means it prices the ledge at what a flawless walk along it would
earn, and takes it — then falls into the cliff whenever exploration or the
ice pushes it off. Its learned route is the better one; its behaviour while
learning is worse. Both halves of that are the point.
"""

from game.algorithms.tabular import TabularLearner


class QLearning(TabularLearner):

    key = "q_learning"
    label = "Q-Learning"
    summary = ("Learns the value of behaving perfectly, even while it is "
               "still exploring — so it never charges itself for its own "
               "random steps.")
    update_rule = "Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]"
    watch_for = ("It should settle on the short ledge right beside the "
                 "cliff, and keep falling in while it trains. The reward "
                 "curve stays lower than SARSA's for exactly that reason.")

    def target(self, transition):
        if transition.done:
            return transition.reward
        values = self.q[transition.next_state]
        best = max(values[action] for action in self.action_list)
        return transition.reward + self.gamma * best
