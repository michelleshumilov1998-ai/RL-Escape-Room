"""Semi-gradient Q-Learning: off-policy control with function approximation.

    target = r + γ max_a Q(s', a)

The maximum rather than the action taken, so what is learned is the value of
flying *perfectly* from here on.

THIS METHOD DOES NOT SOLVE THIS ROOM, AND THAT IS THE MEASURED RESULT
It is kept because what it does instead is the single most important caveat
about function approximation, and this chamber demonstrates it outright rather
than describing it. Off-policy bootstrapping with an approximator has no
convergence guarantee — the tabular Q-Learning of rooms 2 and 3 has one, and it
does not carry over. Measured here, 600 episodes, landings in the last 100:

    γ       max |weight|     landings
    0.950          9.9            0%
    0.980         22.6            0%
    0.990        6.4e+76          0%
    0.995        1.1e+168         0%

At the room's default γ the weights diverge outright: 10^168 is not a large
number, it is a broken run. Lowering α slows it without stopping it — at
α 0.02 the weights still reach 5.7e+05.

Below γ 0.99 it stays bounded and *still* does not land, for a separate reason.
Its episodes end 43% against a wall and 57% at the step limit: the max operator
prices every approach as though the braking will be done perfectly, so it never
learns to brake, and the crash penalty then teaches it to hover instead. Semi-
gradient SARSA on the same chamber, same γ, same seed, lands 71% of the time.

Both halves of that are worth having on screen. The first is the deadly triad —
off-policy updates, bootstrapping and function approximation together — and the
second is room 2's on-policy/off-policy contrast appearing here with much more
force than it managed on room 2's own map, because here the walls really are
fatal and exploration really does reach them.
"""

from game.algorithms.linear import LinearLearner


class SemiGradientQLearning(LinearLearner):

    key = "semi_gradient_q"
    label = "Semi-gradient Q-Learning"
    summary = ("Off-policy control with tile-coded features — and the room's "
               "cautionary case: it does not land, and the reason is the "
               "point of it.")
    update_rule = "w ← w + (α/tilings)[r + γ max_a Q(s′,a) − Q(s,a)]∇Q"
    watch_for = ("It fails, on purpose. At the default γ the weights diverge "
                 "outright; below 0.99 they stay bounded and it still will not "
                 "brake, ending against a wall or at the step limit. Off-policy "
                 "updates plus approximation have no convergence guarantee.")

    def target(self, transition):
        if transition.done:
            return transition.reward
        values = self.all_q(transition.next_state)
        return transition.reward + self.gamma * max(values.values())
