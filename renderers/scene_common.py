"""The parts of a scene that every room shares.

Each room builds its own scene (`iso_scene` for Room 1, `room2_scene` for Room 2)
but the palette, the default view options and the two fingerprints are identical
everywhere, so they live here rather than being copied per room.
"""

import hashlib
import json

from core import actions
from ui.theme import COLORS

# Everything the JavaScript renderer needs to know about colour, taken straight
# from the one palette so no chamber can drift away from the interface.
PALETTE = {
    "background": COLORS["background"],
    "backgroundSoft": COLORS["background_soft"],
    "panel": COLORS["panel"],
    "metal": COLORS["metal"],
    "metalLight": COLORS["metal_light"],
    "gridLine": COLORS["grid_line"],
    "cyan": COLORS["cyan"],
    "cyanDim": COLORS["cyan_dim"],
    "purple": COLORS["purple"],
    "purpleDim": COLORS["purple_dim"],
    "danger": COLORS["danger"],
    "warning": COLORS["warning"],
    "success": COLORS["success"],
    "text": COLORS["text"],
    "textMuted": COLORS["text_muted"],
    "textDim": COLORS["text_dim"],
    "ice": "#BFE6F5",
    "iceDeep": "#7FC9E8",
    "oil": "#171226",
    # Room 2's industrial palette, still built from the same base colours.
    "bridge": "#4A5468",
    "bridgeLight": "#6A758C",
    "collapsing": "#8A6A2A",
    "shaft": "#04070C",
}

DEFAULT_OPTIONS = {
    "camera": "iso",
    "quality": "High",
    "speed": 1.0,
    "follow": False,
    "muted": True,
    "autoplay": True,
    "showCoordinates": True,
}

# The direction tables the renderer needs to animate a move.
DELTAS = {str(direction): list(delta) for direction, delta in actions.DELTAS.items()}
DIRECTION_NAMES = {str(key): value for key, value in actions.DIRECTION_NAMES.items()}


def merge_options(options):
    """The default view options with the caller's choices layered on top."""
    merged = dict(DEFAULT_OPTIONS)
    merged.update(options or {})
    return merged


def finalise(scene, room_number, map_hash):
    """Attach the two fingerprints a scene needs, and return it.

    `episodeId` identifies the recording — the map plus the frames, nothing else.
    The renderer stores its playback position under that key, so changing a view
    setting keeps the current step while a genuinely new episode starts over.

    `signature` identifies the whole scene including the view options, and is only
    used to decide whether the HTML has to be rebuilt on the Python side.
    """
    episode_material = json.dumps({
        "map": map_hash,
        "frames": [[frame["step"], frame["row"], frame["col"],
                    frame["cumulative_reward"]] for frame in scene["frames"]],
    }, sort_keys=True)
    scene["episodeId"] = "room%d-%s" % (
        room_number,
        hashlib.sha256(episode_material.encode("utf-8")).hexdigest()[:12])

    scene_material = json.dumps({
        "episode": scene["episodeId"],
        "options": scene["options"],
        "status": scene["meta"]["systemStatus"],
    }, sort_keys=True)
    scene["signature"] = "scene-" + hashlib.sha256(
        scene_material.encode("utf-8")).hexdigest()[:12]
    return scene
