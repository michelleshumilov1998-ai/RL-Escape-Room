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
