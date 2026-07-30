"""Tile coding, written out by hand.

--------------------------------------------------------------------------
THE PROBLEM
--------------------------------------------------------------------------
Room 4's state is four real numbers. There is no list of states to keep a table
over, so the value of an action has to be *computed* from the state rather than
looked up. The standard way to do that with a linear model is to turn the state
into a long, mostly-zero feature vector and learn one weight per feature:

    Q(s, a) = w · x(s, a)

--------------------------------------------------------------------------
WHAT TILE CODING DOES
--------------------------------------------------------------------------
Lay a grid over the state space and the state falls into exactly one cell. Give
that cell a feature and nearby states share it, which is generalisation. But a
single grid has hard edges: two states either side of a boundary share nothing.

So instead lay down several grids, each shifted by a fraction of a cell. Every
state now activates one cell per grid — `num_tilings` features out of a very large
number — and two nearby states share most, but not all, of them. The overlap is
what makes the generalisation smooth rather than blocky.

    8 tilings x 8 tiles per dimension x 4 dimensions x 5 actions
      = 8 * 8^4 * 5 = 163,840 weights, of which 8 are touched per update.

--------------------------------------------------------------------------
WHY THE LEARNING RATE IS DIVIDED
--------------------------------------------------------------------------
Every active feature contributes to the same prediction, so `num_tilings` weights
each move by the full step size would overshoot by that factor. The effective step
size is therefore

    effective_alpha = alpha / num_tilings

which is what `SemiGradientAgent` uses, and what makes `alpha` mean roughly the
same thing whatever the tiling count.
"""

import math


class TileCoder:
    """A tile coder over a fixed number of continuous dimensions.

    Each action gets its own block of the weight vector, so the features for one
    action can never be confused with another's — `action_offset + tile_index`.
    """

    def __init__(self, bounds, num_tilings=8, tiles_per_dimension=8,
                 num_actions=5):
        if num_tilings < 1:
            raise ValueError("there has to be at least one tiling")
        if tiles_per_dimension < 2:
            raise ValueError("a tiling needs at least two tiles per dimension")

        self.bounds = tuple(tuple(pair) for pair in bounds)
        self.dimensions = len(self.bounds)
        self.num_tilings = num_tilings
        self.tiles_per_dimension = tiles_per_dimension
        self.num_actions = num_actions

        # How many tiles one tiling holds, and the whole coder per action.
        self.tiles_per_tiling = tiles_per_dimension ** self.dimensions
        self.features_per_action = self.num_tilings * self.tiles_per_tiling
        self.total_features = self.features_per_action * self.num_actions

        # Each tiling is shifted by a reproducible fraction of a tile. The classic
        # choice is odd multiples of 1/(2 * num_tilings) per dimension, which
        # spreads the offsets out instead of lining them up.
        self.offsets = []
        for tiling in range(num_tilings):
            offset = []
            for dimension in range(self.dimensions):
                odd = 2 * dimension + 1
                offset.append((tiling * odd) / (2.0 * num_tilings) % 1.0)
            self.offsets.append(tuple(offset))
        self.offsets = tuple(self.offsets)

    # ------------------------------------------------------------------

    def normalise(self, state):
        """Each state variable mapped onto 0..1, clamped at the edges."""
        normalised = []
        for value, (low, high) in zip(state, self.bounds):
            span = high - low
            if span <= 0:
                normalised.append(0.0)
                continue
            share = (value - low) / span
            normalised.append(min(1.0, max(0.0, share)))
        return normalised

    def active_tiles(self, state, action):
        """The indices this state and action switch on — one per tiling.

        Every index is guaranteed to be inside the weight vector, which the tests
        check across the whole state space.
        """
        if not 0 <= action < self.num_actions:
            raise ValueError("action %r is outside the action set" % (action,))

        normalised = self.normalise(state)
        action_offset = action * self.features_per_action
        indices = []

        for tiling, offset in enumerate(self.offsets):
            flat = 0
            for dimension in range(self.dimensions):
                # Shift, scale, and take the whole part: which tile of this tiling.
                shifted = normalised[dimension] * self.tiles_per_dimension \
                    + offset[dimension]
                tile = int(math.floor(shifted))
                tile = min(self.tiles_per_dimension - 1, max(0, tile))
                flat = flat * self.tiles_per_dimension + tile
            indices.append(action_offset + tiling * self.tiles_per_tiling + flat)

        return indices

    # ------------------------------------------------------------------

    def configuration(self):
        """Everything needed to rebuild an identical coder, for saved models."""
        return {
            "bounds": [list(pair) for pair in self.bounds],
            "num_tilings": self.num_tilings,
            "tiles_per_dimension": self.tiles_per_dimension,
            "num_actions": self.num_actions,
            "total_features": self.total_features,
        }

    def matches(self, configuration):
        """Whether a saved configuration describes this coder."""
        mine = self.configuration()
        for key in ("num_tilings", "tiles_per_dimension", "num_actions",
                    "total_features"):
            if configuration.get(key) != mine[key]:
                return False
        saved_bounds = configuration.get("bounds")
        if saved_bounds is None or len(saved_bounds) != len(mine["bounds"]):
            return False
        for saved, own in zip(saved_bounds, mine["bounds"]):
            if abs(saved[0] - own[0]) > 1e-9 or abs(saved[1] - own[1]) > 1e-9:
                return False
        return True
