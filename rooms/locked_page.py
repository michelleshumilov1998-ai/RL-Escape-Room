"""The placeholder page for a chamber that has not been built yet.

Rooms 2 to 5 all use this, so the facility feels complete from the first screen
even though only Room 1 is implemented.  It states plainly that the chamber is
not built rather than pretending to be a working room.
"""

import streamlit as st

from core import story
from ui import buttons, game_state, inventory, nav, theme


def render(room_number):
    """Draw the locked-chamber page for one room."""
    record = story.room(room_number)

    nav.room_header(room_number, "LOCKED", "not yet built",
                    algorithm=record["algorithm"])
    nav.rail(room_number)
    inventory.bar()

    theme.html(
        '<div class="r5-card" style="text-align:center;padding:34px 20px">'
        '  <div class="r5-label">Sealed bulkhead</div>'
        '  <div style="font-size:1.25rem;font-weight:700;margin:10px 0;'
        '       color:%(text)s">ROOM %(number)d — %(name)s</div>'
        '  <p style="color:%(muted)s;max-width:56ch;margin:0 auto">'
        '    This chamber is the next stage of the project. Its environment and '
        '    its learning method (<strong>%(algorithm)s</strong>) have not been '
        '    implemented yet, so there is nothing to run here.</p>'
        '  <p style="color:%(dim)s;max-width:56ch;margin:14px auto 0 auto">'
        '    Planned objective: %(mission)s</p>'
        '  <p style="color:%(dim)s;margin-top:10px">'
        '    Component held here: %(icon)s %(component)s</p>'
        '</div>' % {
            "text": theme.COLORS["text"],
            "muted": theme.COLORS["text_muted"],
            "dim": theme.COLORS["text_dim"],
            "number": room_number,
            "name": record["name"].upper(),
            "algorithm": record["algorithm"],
            "mission": record["mission"],
            "icon": record["component_icon"],
            "component": record["component"],
        })

    left, middle, right = st.columns([1, 1, 1])
    with middle:
        if buttons.link("↩ Back to Room 1", "locked%d" % room_number,
                        role="primary"):
            game_state.go_to_room(story.FIRST_ROOM)
            st.rerun()

    nav.sidebar_navigation(room_number)
