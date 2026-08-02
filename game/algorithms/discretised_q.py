"""Tabular Q-Learning on a rounded-off state. The control case.

This is the obvious thing to try when the state turns out to be continuous: cut
each axis into buckets, treat the bucket as the state, and use the same table
that worked in rooms 2 and 3.

    bucket(x, y, vx, vy)  ->  a tuple of four integers  ->  a row of the table

WHAT IT ACTUALLY COSTS — MEASURED, NOT ASSUMED
It was built expecting it to fail outright, and that is not what happens. Given
enough experience a fine enough table lands about as often as the tile coder
does. What it costs is *how much* experience, and how much memory. Landings in
the last 100 episodes of a run of N, same chamber and same seed:

    method                     N=300   N=600   N=1200   memory
    tile coding, 8 tilings        2%     86%      97%    10,368 weights
    table, 8 buckets an axis      0%      0%      38%     4,096 rows
    table, 20 buckets an axis     0%      2%      95%   160,000 rows

So the honest claim is about sample efficiency and not about capability: the
tile coder is competent at 600 episodes where the best table is at 2%, and it
gets there with a fifteenth of the memory. The table catches up by 1200.

WHY THE BUCKET COUNT CANNOT BE SET WELL
The two ends fail for opposite reasons, and the table above shows both:

  * **Coarse** buckets put genuinely different situations in one row. At eight
    buckets an axis one row spans a 1.25 m square and a quarter of the speed
    range, so "drifting gently towards the pad" and "closing on it far too
    fast" share a row and no policy written in this table can tell them apart.
    38% at 1200 episodes is that ceiling, not slow learning.
  * **Fine** buckets tell them apart and give up generalisation to do it.
    Nothing learned about one row transfers to the row beside it, so every
    bucket must be visited on its own account — which is why 20 buckets is
    still at 2% after 600 episodes while the tile coder is at 86%.

The tile coder escapes the choice rather than settling it: overlapping offset
grids resolve finely *and* share what they learn between neighbours, which one
grid of buckets cannot do at any resolution.

The bucket count is exposed as a parameter so both ends can be watched instead
of taken on trust.
"""

from game.algorithms.base import Algorithm


class DiscretisedQLearning(Algorithm):
    """Q-Learning over a bucketed copy of a continuous state."""

    key = "discretised_q"
    label = "Q-Learning (discretised)"
    family = "Temporal-Difference Learning"
    needs_model = False
    parameters = ("alpha", "gamma", "epsilon", "epsilon_min", "epsilon_decay",
                  "buckets", "episodes")
    summary = ("The state rounded into buckets and learned in an ordinary "
               "table. The control case for the two that approximate.")
    update_rule = "Q(s,a) ← Q(s,a) + α[r + γ max Q(s′,·) − Q(s,a)]"
    watch_for = ("How much longer it takes. At 8 buckets it cannot tell a "
                 "gentle approach from a fast one and stalls near 38%; at 20 "
                 "it gets there, but needs twice the episodes the tile coder "
                 "does and fifteen times the memory.")

    def __init__(self, env, parameters, rng=None):
        Algorithm.__init__(self, env, parameters, rng)
        self.alpha = parameters["alpha"]
        self.gamma = parameters["gamma"]
        self.epsilon = parameters["epsilon"]
        self.epsilon_min = parameters["epsilon_min"]
        self.epsilon_decay = parameters["epsilon_decay"]
        self.episode_target = int(parameters.get("episodes", 1500))
        self.buckets = int(parameters.get("buckets", 8))

        self.action_list = env.actions()
        self.q_init = float(parameters.get("q_init", 0.0))
        # Grown on demand. The table's full size is buckets^4 rows, which at
        # the upper end of the slider is more rows than a run will ever reach —
        # allocating them all would spend most of the memory on rows that are
        # never read, and how few get visited is part of what this shows.
        self.q = {}
        self.episodes = 0
        self._error_total = 0.0
        self._error_count = 0

    # ------------------------------------------------------------------

    def bucket(self, state):
        """The four integers that stand in for the state."""
        env = self.env
        limit = env.speed_limit
        shares = (
            state[0] / env.width,
            state[1] / env.height,
            (state[2] + limit) / (2.0 * limit),
            (state[3] + limit) / (2.0 * limit),
        )
        indexed = []
        for share in shares:
            index = int(share * self.buckets)
            # The top of the range belongs in the last bucket rather than in a
            # bucket of its own one past the end.
            indexed.append(max(0, min(self.buckets - 1, index)))
        return tuple(indexed)

    def _row(self, key):
        row = self.q.get(key)
        if row is None:
            row = {action: self.q_init for action in self.action_list}
            self.q[key] = row
        return row

    def best_actions(self, key, tolerance=1e-12):
        row = self._row(key)
        highest = max(row[action] for action in self.action_list)
        return [action for action in self.action_list
                if row[action] >= highest - tolerance]

    def greedy_action(self, state):
        # Ties by action order when there is no generator — see the note on
        # `LinearLearner.greedy_action`.
        tied = self.best_actions(self.bucket(state))
        if len(tied) == 1 or self.rng is None:
            return tied[0]
        return self.rng.choice(tied)

    def act(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.action_list)
        return self.greedy_action(state)

    def greedy_policy(self):
        """No table over states — only over buckets. See `LinearLearner`."""
        return {}

    def update(self, transition):
        key = self.bucket(transition.state)
        row = self._row(key)
        before = row[transition.action]

        if transition.done:
            target = transition.reward
        else:
            following = self._row(self.bucket(transition.next_state))
            best = max(following[action] for action in self.action_list)
            target = transition.reward + self.gamma * best

        error = target - before
        row[transition.action] = before + self.alpha * error

        self._error_total += abs(error)
        self._error_count += 1
        return {"tdError": error, "target": target, "qBefore": before,
                "qAfter": row[transition.action]}

    def end_episode(self):
        self.episodes += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @property
    def finished(self):
        return self.episodes >= self.episode_target

    def snapshot(self):
        return {
            "values": {},
            "policy": {},
            "episodes": self.episodes,
            "episodeTarget": self.episode_target,
            "epsilon": self.epsilon,
            "meanAbsQ": (self._error_total / self._error_count
                         if self._error_count else 0.0),
            "converged": self.finished,
            "startValue": max(self._row(self.bucket(
                self.env.start_state())).values()),
            # How much of the table has been reached at all, against how much
            # of it there is. This is the number that makes the fine-bucket
            # failure visible rather than merely asserted.
            "visited": len(self.q),
            "features": self.buckets ** 4,
        }

    def set_parameter(self, name, value):
        Algorithm.set_parameter(self, name, value)
        if name == "episodes":
            self.episode_target = int(value)
