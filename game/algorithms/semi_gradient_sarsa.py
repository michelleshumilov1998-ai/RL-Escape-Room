"""Semi-gradient SARSA: on-policy control with function approximation.

    target = r + γ Q(s', a')

`a'` is the action the ε-greedy policy actually drew for the next state, so the
value learned is the value of *flying the way it is currently flying* —
exploratory thrusts included. In a room where a wall costs -100 that is the
cautious estimate, and it is the difference this room's method selector is for.
"""

from game.algorithms.linear import LinearLearner


class SemiGradientSarsa(LinearLearner):

    key = "semi_gradient_sarsa"
    label = "Semi-gradient SARSA"
    summary = ("On-policy control with tile-coded features. Learns the value "
               "of the way it is actually flying, exploratory thrusts and "
               "all.")
    update_rule = "w ← w + (α/tilings)[r + γQ(s′,a′) − Q(s,a)]∇Q"
    watch_for = ("It keeps its distance from the walls, because the value it "
                 "is learning includes the chance of a random thrust into "
                 "one.")

    def target(self, transition):
        if transition.done or transition.next_action is None:
            return transition.reward
        return transition.reward + self.gamma * self.q(
            transition.next_state, transition.next_action)
