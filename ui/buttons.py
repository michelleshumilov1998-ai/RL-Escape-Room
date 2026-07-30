"""Consistent buttons for every room.

Streamlit gives a keyed widget a CSS class of the form `st-key-<key>`, and the
stylesheet in `ui/theme.py` styles buttons by the *prefix* of that key.  So the
role of a button is decided by naming its key, and every room automatically gets
the same colours, sizes and spacing.

    PRIMARY   cyan filled     the main action of the page (Solve, Train)
    SUCCESS   green filled    running or confirming (Run Agent, Play Replay)
    DANGER    red outline     destructive or resetting (Reset)
    SECONDARY dark panel      everything else (Save, Load, Compare)

Use `action` for a button and `label_for` if you only need the standard wording.
"""

import streamlit as st

PRIMARY = "primary"
SUCCESS = "success"
DANGER = "danger"
SECONDARY = "secondary"

VALID_ROLES = (PRIMARY, SUCCESS, DANGER, SECONDARY)

# The standard wording and icon of every action in the game.  Rooms should take
# their labels from here so the same action is never called two different things.
ACTIONS = {
    "train": ("▸ Train", PRIMARY),
    "solve": ("▸ Solve Room", PRIMARY),
    "evaluate": ("◆ Evaluate", PRIMARY),
    "run": ("▶ Run Agent", SUCCESS),
    "replay": ("▶ Play Replay", SUCCESS),
    "pause": ("‖ Pause", SECONDARY),
    "step": ("▸❘ Step", SECONDARY),
    "reset": ("↻ Reset Animation", DANGER),
    "save": ("↓ Save Results", SECONDARY),
    "load": ("↑ Load Results", SECONDARY),
    "experiment": ("≣ Run Experiment", SECONDARY),
    "compare": ("⇄ Compare Algorithms", SECONDARY),
}


def action(name, key_suffix, role=None, label=None, **kwargs):
    """Draw one standard button and return True when it is clicked.

    `name` picks the wording and default role out of ACTIONS.  `key_suffix`
    keeps the widget key unique on the page, usually the room name.
    """
    default_label, default_role = ACTIONS.get(name, (name.title(), SECONDARY))
    role = role or default_role
    if role not in VALID_ROLES:
        raise ValueError("Unknown button role: %s" % role)

    # The key prefix is what the stylesheet matches on, so it comes first.
    key = "%s_%s_%s" % (role, name, key_suffix)
    return st.button(label or default_label, key=key, **kwargs)


def link(label, key_suffix, role=SECONDARY, **kwargs):
    """A button that is only used to move between screens."""
    key = "%s_link_%s" % (role, key_suffix)
    return st.button(label, key=key, **kwargs)
