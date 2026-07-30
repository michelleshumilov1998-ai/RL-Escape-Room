"""Turning Room 5's local observation into sparse linear features.

--------------------------------------------------------------------------
WHY NOT ONE BIG TILE CODER
--------------------------------------------------------------------------
The observation has sixteen-odd numbers in it. A single joint grid over all of them
would need `bins ** 16` cells, which is astronomically large and would never be
filled. So the observation is split into small **groups** and each group gets its
own little tile coder. The active indices are then concatenated.

    bias                 1 feature, always on
    stage                which half of the mission
    target direction     2D: where the objective is, relative to R-5
    target distance      1D: how far
    static radar         eight 1D coders, one per compass direction
    dynamic radar        four 1D coders, plus the nearest-robot distance
    previous action      one feature per action

Every group is offset into its own slice of the index space so they cannot collide,
and the whole thing is then repeated per action, so the features for MOVE_UP can
never be confused with the features for WAIT.

--------------------------------------------------------------------------
WHY THIS GENERALISES
--------------------------------------------------------------------------
Nothing here mentions an absolute position, a layout, or a seed. Every feature is
either "what is near me" or "where is my objective" — both of which mean the same
thing in a warehouse the agent has never seen. That is what makes it possible to
train on one set of layouts and be evaluated on another.

The groups can be switched off individually, which is what the feature-set
experiment uses.
"""

from rooms.room5.environment import ACTIONS

# The named feature sets the experiment compares.
FEATURE_SETS = {
    "A. Target only": ("bias", "stage", "target_direction", "target_distance"),
    "B. Target + static radar": ("bias", "stage", "target_direction",
                                 "target_distance", "static_radar"),
    "C. Target + both radars": ("bias", "stage", "target_direction",
                                "target_distance", "static_radar",
                                "dynamic_radar"),
    "D. Full representation": ("bias", "stage", "target_direction",
                              "target_distance", "static_radar",
                              "dynamic_radar", "previous_action"),
}

DEFAULT_FEATURE_SET = "D. Full representation"

BINS_DEFAULT = 6
TILINGS_DEFAULT = 3


def _tile_1d(value, bins, tilings, low=0.0, high=1.0):
    """Which bin `value` falls in, once per tiling, each slightly offset."""
    span = (high - low) or 1.0
    share = min(1.0, max(0.0, (value - low) / span))
    indices = []
    for tiling in range(tilings):
        offset = tiling / tilings
        bin_index = int(share * bins + offset)
        indices.append(tiling * bins + min(bins - 1, max(0, bin_index)))
    return indices, bins * tilings


def _tile_2d(first, second, bins, tilings, low=-1.0, high=1.0):
    """The same idea over two dimensions at once."""
    span = (high - low) or 1.0
    a = min(1.0, max(0.0, (first - low) / span))
    b = min(1.0, max(0.0, (second - low) / span))
    indices = []
    for tiling in range(tilings):
        offset = tiling / tilings
        row = min(bins - 1, max(0, int(a * bins + offset)))
        col = min(bins - 1, max(0, int(b * bins + offset)))
        indices.append(tiling * bins * bins + row * bins + col)
    return indices, bins * bins * tilings


class FeatureExtractor:
    """Observation and action to a small list of active feature indices."""

    def __init__(self, groups=None, bins=BINS_DEFAULT, tilings=TILINGS_DEFAULT,
                 radar_directions=8, dynamic_directions=4):
        self.groups = tuple(groups or FEATURE_SETS[DEFAULT_FEATURE_SET])
        self.bins = bins
        self.tilings = tilings
        self.radar_directions = radar_directions
        self.dynamic_directions = dynamic_directions
        self.num_actions = len(ACTIONS)

        # Work out where each group's slice starts.
        self.offsets = {}
        cursor = 0
        for name, size in self._group_sizes():
            self.offsets[name] = cursor
            cursor += size
        self.features_per_action = cursor
        self.total_features = cursor * self.num_actions

    def _group_sizes(self):
        """Each active group and how many indices it owns."""
        per_1d = self.bins * self.tilings
        per_2d = self.bins * self.bins * self.tilings
        sizes = []
        for name in self.groups:
            if name == "bias":
                sizes.append((name, 1))
            elif name == "stage":
                sizes.append((name, 2))
            elif name == "target_direction":
                sizes.append((name, per_2d))
            elif name == "target_distance":
                sizes.append((name, per_1d))
            elif name == "static_radar":
                sizes.append((name, per_1d * self.radar_directions))
            elif name == "dynamic_radar":
                # the four rays, plus the nearest-robot distance
                sizes.append((name, per_1d * (self.dynamic_directions + 1)))
            elif name == "previous_action":
                sizes.append((name, self.num_actions))
            else:
                raise ValueError("unknown feature group: %r" % (name,))
        return sizes

    # ------------------------------------------------------------------

    def active_features(self, observation, action):
        """The indices this observation and action switch on."""
        if not 0 <= action < self.num_actions:
            raise ValueError("action %r is outside the action set" % (action,))

        indices = []
        for name in self.groups:
            base = self.offsets[name]
            if name == "bias":
                indices.append(base)
            elif name == "stage":
                indices.append(base + int(observation["stage"]))
            elif name == "target_direction":
                local, _ = _tile_2d(observation["target"][0],
                                    observation["target"][1],
                                    self.bins, self.tilings)
                indices.extend(base + index for index in local)
            elif name == "target_distance":
                local, _ = _tile_1d(observation["target"][2], self.bins,
                                    self.tilings)
                indices.extend(base + index for index in local)
            elif name == "static_radar":
                per = self.bins * self.tilings
                for direction, value in enumerate(observation["static_radar"]):
                    local, _ = _tile_1d(value, self.bins, self.tilings)
                    indices.extend(base + direction * per + index
                                   for index in local)
            elif name == "dynamic_radar":
                per = self.bins * self.tilings
                values = list(observation["dynamic_radar"]) + \
                    [observation["nearest_robot"]]
                for direction, value in enumerate(values):
                    local, _ = _tile_1d(value, self.bins, self.tilings)
                    indices.extend(base + direction * per + index
                                   for index in local)
            elif name == "previous_action":
                for position, value in enumerate(observation["previous_action"]):
                    if value > 0.5:
                        indices.append(base + position)

        action_offset = action * self.features_per_action
        return [action_offset + index for index in indices]

    # ------------------------------------------------------------------

    def configuration(self):
        return {"groups": list(self.groups), "bins": self.bins,
                "tilings": self.tilings,
                "radar_directions": self.radar_directions,
                "dynamic_directions": self.dynamic_directions,
                "num_actions": self.num_actions,
                "features_per_action": self.features_per_action,
                "total_features": self.total_features}

    def matches(self, configuration):
        mine = self.configuration()
        for key in ("groups", "bins", "tilings", "radar_directions",
                    "dynamic_directions", "num_actions", "total_features"):
            if configuration.get(key) != mine[key]:
                return False
        return True
