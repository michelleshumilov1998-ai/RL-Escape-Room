"""The contract between the game logic and the visual layer.

The renderers never touch an environment, an agent or a policy.  They are given
a plain dictionary — a "scene" — and they draw it.  That keeps all the
reinforcement learning maths on one side of the wall and all the pixels on the
other, and it means a replay recorded months ago still draws correctly.

A scene is JSON-serialisable and looks like this:

    {
      "signature": str,        a short id that changes when the scene changes
      "meta":      {...},      sector code, room title, subtitle, status
      "grid":      {...},      rows, cols, tile kinds, one-way door directions
      "landmarks": {...},      start, exit, teleporter pads, battery, lasers
      "frames":    [...],      the recorded episode, one entry per step
      "options":   {...},      camera, quality, speed, follow, muted
      "palette":   {...},      the colours from ui/theme.py
      "deltas":    {...},      direction id -> (row change, col change)
      "legend":    [...],      (tile kind, human readable name) pairs
    }

Rules every renderer must follow:

  * Never change the scene it was given.
  * Never sample randomness.  A replay must show exactly what was recorded, so
    everything the picture needs is already in the frames.
  * Never compute a reward, a value or an action.
"""

SCENE_KEYS = ("signature", "meta", "grid", "landmarks", "frames", "options",
              "palette", "deltas", "legend")

CAMERA_MODES = ("iso", "tactical", "top")
QUALITY_LEVELS = ("Low", "Medium", "High")


def validate_scene(scene):
    """Check a scene has everything a renderer needs.

    Returns a list of problems, empty when the scene is fine.  Used by the tests
    so a broken scene is caught before it reaches a browser.
    """
    problems = []

    for key in SCENE_KEYS:
        if key not in scene:
            problems.append("missing key: %s" % key)
    if problems:
        return problems

    grid = scene["grid"]
    for key in ("rows", "cols", "tiles"):
        if key not in grid:
            problems.append("grid is missing: %s" % key)
    if problems:
        return problems

    if len(grid["tiles"]) != grid["rows"]:
        problems.append("grid has %d tile rows but says %d"
                        % (len(grid["tiles"]), grid["rows"]))
    for index, line in enumerate(grid["tiles"]):
        if len(line) != grid["cols"]:
            problems.append("tile row %d has %d entries but should have %d"
                            % (index, len(line), grid["cols"]))

    if scene["options"].get("camera") not in CAMERA_MODES:
        problems.append("unknown camera mode: %s" % scene["options"].get("camera"))
    if scene["options"].get("quality") not in QUALITY_LEVELS:
        problems.append("unknown quality level: %s" % scene["options"].get("quality"))

    for index, frame in enumerate(scene["frames"]):
        for key in ("row", "col", "from_row", "from_col", "step"):
            if key not in frame:
                problems.append("frame %d is missing: %s" % (index, key))
                break

    return problems
