"""What the two function-approximation methods share.

Q is no longer looked up. It is *computed*, as a weighted sum of the features
the tile coder switches on:

    Q(s, a) = Σ w[a][i]   for each active feature i

Because the tile coder returns exactly one tile per tiling and every active
feature contributes with weight 1, the sum is a plain addition of `tilings`
numbers and the gradient with respect to the weights is 1 on those features
and 0 everywhere else. That is what makes the update a single line:

    w[a][i] += (α / tilings) * error      for each active i

Dividing α by the number of tilings is not cosmetic. Without it the effective
step size is `tilings` times what the slider says, because every one of them
moves by the full amount and they are then summed again on the next lookup;
at eight tilings and α 0.5 the update overshoots and the weights diverge
within a few hundred steps. With it, α means what it says: the fraction of the
error corrected in one step.

THE ONE DIFFERENCE BETWEEN THE TWO METHODS
Exactly as with the tabular four, it is the target and nothing else:

    semi-gradient SARSA        r + γ Q(s', a')      the action actually next
    semi-gradient Q-Learning   r + γ max_a Q(s',a)  the best action available

"Semi-gradient" is the honest name: the target depends on the weights too, and
the derivative of *that* dependence is ignored. Following the full gradient is
a different and worse-behaved algorithm; this is the one that works.
"""

from game.algorithms.base import Algorithm
from game.algorithms.tile_coding import GroupedTileCoder, TileCoder


class LinearLearner(Algorithm):
    """Action values as a linear function of tile-coded features."""

    needs_model = False
    family = "Function Approximation"
    parameters = ("alpha", "gamma", "epsilon", "epsilon_min", "epsilon_decay",
                  "tilings", "episodes")

    def __init__(self, env, parameters, rng=None):
        Algorithm.__init__(self, env, parameters, rng)
        self.alpha = parameters["alpha"]
        self.gamma = parameters["gamma"]
        self.epsilon = parameters["epsilon"]
        self.epsilon_min = parameters["epsilon_min"]
        self.epsilon_decay = parameters["epsilon_decay"]
        self.episode_target = int(parameters.get("episodes", 1500))

        self.action_list = env.actions()
        tilings = int(parameters.get("tilings", 8))

        # Which coder the room needs, asked of the room rather than assumed.
        #
        # Room 4's observation is its four state numbers and one coder over all
        # four is affordable. Room 5's is fourteen, where a single Cartesian grid
        # is 6^14 tiles and cannot be built — it declares `OBSERVATION_GROUPS`
        # and gets a grouped coder instead. Everything below is identical
        # either way, which is the point of asking.
        groups = getattr(env, "OBSERVATION_GROUPS", None)
        if groups:
            self.coder = GroupedTileCoder(groups, tilings=tilings)
        else:
            self.coder = TileCoder(
                dimensions=4,
                tilings=tilings,
                tiles_per_dimension=int(
                    parameters.get("tiles_per_dimension", 6)))

        # How many separate blocks of weights the room wants, and which one a
        # state falls in. Room 5 has two — before and after the control
        # terminal are different tasks with different objectives, and sharing
        # one set of weights between them would ask a linear model to represent
        # both with the same features. Rooms without the hook have one block
        # and are unaffected.
        self.context_count = int(getattr(env, "context_count", 1))
        self.block = self.coder.features
        self.features = self.block * self.context_count

        # One weight vector per action. A flat list rather than a dict: it is
        # read several times per step and indexed by integers throughout.
        self.weights = {action: [0.0] * self.features
                        for action in self.action_list}

        self.episodes = 0
        # Kept for the readout: the mean size of the corrections made this
        # episode is this room's "something that ought to fall towards zero".
        self._error_total = 0.0
        self._error_count = 0
        self._last_error = 0.0

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def scale(self, state):
        """The observation, with every axis mapped into [0, 1].

        A room that limits what its agent may see owns that decision, so if it
        offers an `observation` it is used verbatim — room 5's is fourteen numbers
        of which only three are sensor rays, and it is emphatically *not* its
        state. Otherwise this is the only place that knows what the four numbers
        of a flight state mean: position over the room's size, velocity over its
        limit, so the axes reach the coder on the same footing. A coder handed
        raw metres beside raw metres per second would cut one axis a hundred
        times more finely than the other for no reason but the units.
        """
        env = self.env
        if hasattr(env, "observation"):
            return env.observation(state)
        return (
            state[0] / env.width,
            state[1] / env.height,
            (state[2] + env.speed_limit) / (2.0 * env.speed_limit),
            (state[3] + env.speed_limit) / (2.0 * env.speed_limit),
        )

    def active_features(self, state):
        """The feature indices a state switches on, in its own block."""
        indices = self.coder.active(self.scale(state))
        if self.context_count == 1:
            return indices
        # Shifted into this state's block, so the two stages of room 5's
        # mission cannot overwrite each other's weights.
        offset = self.env.context(state) * self.block
        return [offset + index for index in indices]

    def q(self, state, action, active=None):
        if active is None:
            active = self.active_features(state)
        weights = self.weights[action]
        total = 0.0
        for index in active:
            total += weights[index]
        return total

    def all_q(self, state, active=None):
        if active is None:
            active = self.active_features(state)
        return {action: self.q(state, action, active)
                for action in self.action_list}

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def best_actions(self, state, active=None, tolerance=1e-12):
        values = self.all_q(state, active)
        highest = max(values.values())
        return [action for action in self.action_list
                if values[action] >= highest - tolerance]

    def greedy_action(self, state):
        """The best action for one state, with no exploration in it.

        This is the primitive the continuous rooms use. A whole-policy table is
        not available here and never will be — see `greedy_policy`.

        Ties are broken randomly when there is a generator to do it with, and
        by action order when there is not. Reading a policy must not *require*
        randomness: a session always supplies a generator, but a caller that
        builds a learner to inspect it has no reason to, and crashing on that
        turns a missing convenience into a missing method.
        """
        tied = self.best_actions(state)
        if len(tied) == 1 or self.rng is None:
            return tied[0]
        return self.rng.choice(tied)

    def act(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.action_list)
        return self.greedy_action(state)

    def action_probabilities(self, state):
        share = self.epsilon / len(self.action_list)
        probabilities = {action: share for action in self.action_list}
        tied = self.best_actions(state)
        for action in tied:
            probabilities[action] += (1.0 - self.epsilon) / len(tied)
        return probabilities

    def greedy_policy(self):
        """There is no policy table in a continuous room.

        Returned empty rather than raised, because the screens ask for this
        while painting and an exception there would take the page down. What
        replaces it is `greedy_action`, which answers for one state at a time —
        which is all a replay or a renderer ever actually needs.
        """
        return {}

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def target(self, transition):
        """Where the estimate should move to. The one thing that differs."""
        raise NotImplementedError

    def update(self, transition):
        active = self.active_features(transition.state)
        before = self.q(transition.state, transition.action, active)
        error = self.target(transition) - before

        # The gradient is 1 on each active feature, so the correction is shared
        # equally between them. Divided by how many are active rather than by
        # the tiling count, because a grouped coder switches on `tilings` per
        # group: dividing by `tilings` alone would multiply the effective step
        # size by the number of groups. See the module docstring.
        step = self.alpha / len(active) * error if active else 0.0
        weights = self.weights[transition.action]
        for index in active:
            weights[index] += step

        self._error_total += abs(error)
        self._error_count += 1
        self._last_error = error
        return {"tdError": error, "target": before + error, "qBefore": before,
                "qAfter": self.q(transition.state, transition.action, active)}

    def end_episode(self):
        self.episodes += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @property
    def finished(self):
        return self.episodes >= self.episode_target

    # ------------------------------------------------------------------

    def mean_absolute_error(self):
        if not self._error_count:
            return 0.0
        return self._error_total / self._error_count

    def snapshot(self):
        # No value heatmap and no policy arrows: both are grid overlays, and
        # there are no cells here to lay them over. The room screen already
        # guards every overlay on `room.isGrid`.
        return {
            "values": {},
            "policy": {},
            "episodes": self.episodes,
            "episodeTarget": self.episode_target,
            "epsilon": self.epsilon,
            # Named as the tabular rooms name it so the readout needs no
            # branch, and honest about what it is: the mean correction size.
            "meanAbsQ": self.mean_absolute_error(),
            "converged": self.finished,
            "startValue": max(self.all_q(self.env.start_state()).values()),
            "features": self.features,
            "tilings": self.coder.tilings,
        }

    def set_parameter(self, name, value):
        Algorithm.set_parameter(self, name, value)
        if name == "episodes":
            self.episode_target = int(value)
