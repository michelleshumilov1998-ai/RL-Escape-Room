"""Double Q-Learning.

Q-Learning takes a `max` over estimates that are themselves noisy, and a max
over noise is biased upwards: it systematically over-values actions that got
lucky early. On a room where one route is fast but occasionally catastrophic,
that bias is precisely the wrong way round.

The fix is two tables. One picks the action, the other says what it is worth,
and which does which is decided by a coin flip on every update:

    with probability ½
        a* = argmax_a Q_a(s', a)
        Q_a(s,a) <- Q_a(s,a) + α [ r + γ Q_b(s', a*) - Q_a(s,a) ]
    otherwise the same with the two swapped

Because the table doing the choosing is never the table doing the valuing,
a lucky estimate in one cannot inflate its own target.

This is the one method here that needs more than a different target
expression, so it overrides `update` and the table `act` reads. Everything
else — ε-greedy, how a policy is read off, the snapshot — is inherited
unchanged.
"""

from game.algorithms.tabular import TabularLearner


class DoubleQLearning(TabularLearner):

    key = "double_q_learning"
    label = "Double Q-Learning"
    summary = ("Q-Learning with two tables. One chooses the action, the "
               "other prices it, which removes the upward bias that taking "
               "a maximum over noisy estimates introduces.")
    update_rule = ("Q_a(s,a) ← Q_a(s,a) + α [ r + γ Q_b(s', argmax_a' "
                   "Q_a(s',a')) − Q_a(s,a) ]   (tables swap at random)")
    watch_for = ("Much the same route as Q-Learning, reached with less "
                 "wild over-estimation on the way. Compare mean |Q| against "
                 "plain Q-Learning — this one stays lower.")

    def __init__(self, env, parameters, rng=None):
        TabularLearner.__init__(self, env, parameters, rng)
        # `self.q` is inherited and used as the first table; the second
        # mirrors it. Both start at zero.
        self.q_second = {state: {action: self.q_init
                                 for action in self.action_list}
                         for state in env.all_states()}
        self._combined = {}

    def _table(self, state):
        """Both tables added together.

        Acting on the sum uses twice the experience and is the standard
        choice; neither table alone is what the agent believes.
        """
        row = self._combined.get(state)
        if row is None:
            row = {}
            self._combined[state] = row
        for action in self.action_list:
            row[action] = self.q[state][action] + self.q_second[state][action]
        return row

    def target(self, transition):
        """Unused: this method's update does not fit the single-target shape."""
        raise NotImplementedError(
            "Double Q-Learning updates one of two tables; see `update`")

    def update(self, transition):
        first, second = self.q, self.q_second
        if self.rng.random() < 0.5:
            first, second = self.q_second, self.q

        if transition.done:
            target = transition.reward
        else:
            row = first[transition.next_state]
            highest = max(row[action] for action in self.action_list)
            # Chosen by one table, valued by the other. Ties go to the first
            # action so the choice stays reproducible for a given seed.
            chosen = None
            for action in self.action_list:
                if row[action] >= highest - 1e-12:
                    chosen = action
                    break
            target = (transition.reward
                      + self.gamma * second[transition.next_state][chosen])

        before = first[transition.state][transition.action]
        error = target - before
        first[transition.state][transition.action] = before + self.alpha * error
        return {"tdError": error, "target": target, "qBefore": before,
                "qAfter": first[transition.state][transition.action]}

    def mean_absolute_q(self):
        total = 0.0
        count = 0
        for table in (self.q, self.q_second):
            for state in table:
                for action in self.action_list:
                    total += abs(table[state][action])
                    count += 1
        return total / count if count else 0.0
