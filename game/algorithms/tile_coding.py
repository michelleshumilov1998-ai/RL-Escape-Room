"""A tile coder, written out by hand.

THE PROBLEM IT SOLVES
Room 4's state is four real numbers. A table needs one row per state and there
are infinitely many, so the table cannot be built at all. Rounding the numbers
off and using a table of the rounded values — which is what room 4's comparison
panel actually does, for contrast — trades one failure for another: coarse
buckets make genuinely different situations share a row, and fine buckets make
every visit land somewhere new, so nothing learned in one cell is ever of any
use in the cell beside it.

WHAT IT DOES INSTEAD
The state space is covered several times over by grids of tiles, each grid
offset from the last by a fraction of a tile. A state falls in exactly one tile
of each grid, so it is described by a handful of overlapping features rather
than by one bucket:

    tilings = 8, tiles per axis = 6   ->   8 active features out of 8 * 6^4

Two nearby states share most of their active tiles, so learning about one
teaches most of what there is to know about the other — that is generalisation,
and it is the whole point. Two distant states share none, so they stay
separate. The offsets are what buys the resolution: eight coarse grids laid at
different offsets resolve position far more finely than one coarse grid, at a
fraction of the cost of one fine one.

WHY ASYMMETRIC OFFSETS
Each tiling is displaced by an odd multiple of 1/tilings along every axis —
1, 3, 5, 7 … — rather than by the same fraction on all of them. Offsetting all
axes equally lays the tilings along a diagonal of the space, so states that
differ diagonally are resolved well and states differing along one axis alone
are barely resolved at all. The odd-multiple rule is the standard fix and it
costs one line.

Nothing here knows what x, y, vx or vy mean. It is handed a state already
scaled to the unit interval on each axis, and returns integers.
"""


class GroupedTileCoder:
    """Several small tile coders side by side, over disjoint slices of one
    observation.

    WHY ROOM 5 CANNOT USE ONE CODER
    Room 4's observation is four numbers, and one coder over all four is 6^4 =
    1296 tiles a tiling — perfectly affordable. Room 5's observation is fourteen,
    and the same construction is 6^14, which is 78 billion tiles a tiling. It
    is not a tuning problem; a full Cartesian grid over fourteen axes cannot be
    built at any resolution worth having.

    The fix is to stop insisting that every axis be resolved jointly with every
    other. What actually needs to interact is *within* a group — where the drone
    is with where it is going, or the three sensor rays with each other — and
    across groups an additive approximation is enough:

        Q(s,a) = Σ over groups Σ over that group's active tiles  w[a][i]

    So each group gets its own coder over two or three axes, the blocks are laid
    end to end in one weight vector, and a state switches on `tilings` features
    per group instead of `tilings` in total. That is the standard answer, it
    keeps the whole thing linear, and it is what makes fourteen dimensions cost
    about two thousand weights rather than billions.

    WHAT IS GIVEN UP, SAID PLAINLY
    Cross-group interactions the groups do not contain. The agent cannot
    represent "this sensor reading matters only in the bottom-left corner"
    unless position and sensors share a group. Which axes go together is
    therefore a real modelling decision, not a formality — it is made in
    `warehouse.py`, where the meaning of the axes is known.
    """

    def __init__(self, groups, tilings=8):
        # `groups` is a list of (axes, tiles_per_axis): `axes` is how many of
        # the observation's numbers this group takes, in order.
        self.tilings = int(tilings)
        self.groups = []
        self.features = 0
        start = 0
        for axes, tiles in groups:
            coder = TileCoder(dimensions=axes, tilings=self.tilings,
                              tiles_per_dimension=tiles)
            self.groups.append({
                "coder": coder,
                # Which slice of the observation this group reads, and where
                # its block of features begins in the shared vector.
                "from": start,
                "to": start + axes,
                "offset": self.features,
            })
            start += axes
            self.features += coder.features
        self.dimensions = start

        # A bias feature: on for every state, so the weight vector can carry an
        # overall offset instead of having to spend one tile per group on it.
        # Cheap, and it is what lets the value of "being anywhere at all" be
        # learned separately from the value of being in any particular place.
        self.bias = self.features
        self.features += 1
        # How many features one state switches on, over all the groups plus the
        # bias. The step size is divided by this rather than by `tilings`, for
        # exactly the reason `linear.py` divides by `tilings`: every active
        # feature moves and they are all summed again on the next lookup.
        self.active_count = self.tilings * len(self.groups) + 1

    def active(self, scaled):
        """The feature indices one observation switches on."""
        indices = []
        for group in self.groups:
            part = scaled[group["from"]:group["to"]]
            offset = group["offset"]
            for index in group["coder"].active(part):
                indices.append(offset + index)
        indices.append(self.bias)
        return indices


class TileCoder:
    """Overlapping grids of tiles over a box in n dimensions."""

    def __init__(self, dimensions, tilings=8, tiles_per_dimension=6):
        self.dimensions = int(dimensions)
        self.tilings = int(tilings)
        self.tiles = int(tiles_per_dimension)

        # One tiling's worth of tiles, and the whole coder's worth. A feature
        # index is (which tiling) * tiles_per_tiling + (which tile in it), so
        # the tilings occupy disjoint blocks and cannot alias onto each other.
        self.tiles_per_tiling = self.tiles ** self.dimensions
        self.features = self.tilings * self.tiles_per_tiling

        # The displacement of each tiling along each axis, as a share of one
        # tile. Odd multiples of 1/tilings, per the module docstring.
        self.offsets = []
        for tiling in range(self.tilings):
            share = tiling / float(self.tilings)
            self.offsets.append(
                tuple((share * (2 * axis + 1)) % 1.0
                      for axis in range(self.dimensions)))

    def active(self, scaled):
        """The feature indices a state switches on: one per tiling.

        `scaled` is the state with every axis already mapped into [0, 1];
        anything outside is clamped, so a drone momentarily past a wall is
        described by the edge tiles rather than by an index off the end of the
        weight vector.
        """
        indices = []
        for tiling in range(self.tilings):
            offset = self.offsets[tiling]
            flat = 0
            for axis in range(self.dimensions):
                value = scaled[axis]
                if value < 0.0:
                    value = 0.0
                elif value > 1.0:
                    value = 1.0
                # The offset shifts the grid, so the coordinate is shifted the
                # other way before being cut into tiles.
                shifted = value * self.tiles + offset[axis]
                cell = int(shifted) % self.tiles
                flat = flat * self.tiles + cell
            indices.append(tiling * self.tiles_per_tiling + flat)
        return indices
