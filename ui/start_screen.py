"""The opening screen of PROJECT R-5.

Shows the title, the story, a picture of the facility with the five chambers, the
inventory bar and the Start / Continue controls.  It uses exactly the same
palette, typography and frame style as the rooms, so the game looks like one
piece from the first screen onwards.
"""

import streamlit as st

from core import story
from ui import buttons, game_state, inventory, theme


def _facility_diagram():
    """A small side-on drawing of the underground facility.

    Drawn with inline SVG so it needs no image files and no extra dependency,
    and so it can use the palette directly.
    """
    chambers = []
    for index, number in enumerate(story.ROOM_NUMBERS):
        record = story.room(number)
        status = game_state.room_status(number)
        x = 62 + index * 134
        if status == game_state.STATUS_SOLVED:
            colour, glow = theme.COLORS["success"], 0.85
        elif status == game_state.STATUS_ACTIVE:
            colour, glow = theme.COLORS["cyan"], 0.95
        else:
            colour, glow = theme.COLORS["metal"], 0.35

        chambers.append(
            '<g opacity="%(glow).2f">'
            '  <rect x="%(x)d" y="52" width="96" height="60" rx="8"'
            '        fill="%(panel)s" stroke="%(colour)s" stroke-width="1.6"/>'
            '  <circle cx="%(cx)d" cy="72" r="7" fill="%(colour)s"/>'
            '  <text x="%(cx)d" y="76" text-anchor="middle" font-size="9"'
            '        font-family="monospace" fill="%(bg)s" font-weight="700">%(num)d</text>'
            '  <text x="%(cx)d" y="98" text-anchor="middle" font-size="8.5"'
            '        font-family="monospace" fill="%(label)s"'
            '        letter-spacing="1">%(short)s</text>'
            '</g>' % {
                "glow": glow, "x": x, "cx": x + 48,
                "panel": theme.COLORS["background_soft"],
                "colour": colour, "bg": theme.COLORS["background"],
                "num": number, "label": theme.COLORS["text_muted"],
                "short": record["algorithm_short"],
            })

        if index < len(story.ROOM_NUMBERS) - 1:
            chambers.append(
                '<line x1="%d" y1="82" x2="%d" y2="82" stroke="%s"'
                ' stroke-width="1.4" stroke-dasharray="4 3" opacity="0.7"/>'
                % (x + 96, x + 134, theme.COLORS["grid_line"]))

    return (
        '<svg viewBox="0 0 760 176" width="100%%" height="176"'
        '     role="img" aria-label="Cross-section of the five test chambers"'
        '     style="display:block;max-width:100%%">'
        '  <defs>'
        '    <linearGradient id="surface" x1="0" y1="0" x2="0" y2="1">'
        '      <stop offset="0" stop-color="%(cyan)s" stop-opacity="0.22"/>'
        '      <stop offset="1" stop-color="%(cyan)s" stop-opacity="0"/>'
        '    </linearGradient>'
        '  </defs>'
        '  <rect x="0" y="0" width="760" height="26" fill="url(#surface)"/>'
        '  <text x="16" y="18" font-size="9" font-family="monospace"'
        '        fill="%(muted)s" letter-spacing="3">SURFACE — EXIT GATE</text>'
        '  <line x1="0" y1="26" x2="760" y2="26" stroke="%(metal)s" stroke-width="1.2"/>'
        '  <text x="16" y="46" font-size="9" font-family="monospace"'
        '        fill="%(dim)s" letter-spacing="3">SUB-LEVEL 4</text>'
        '  %(chambers)s'
        '  <text x="16" y="136" font-size="9" font-family="monospace"'
        '        fill="%(dim)s" letter-spacing="2">R-5 CHARGING PLATE</text>'
        '  <circle cx="30" cy="82" r="9" fill="%(metal_light)s"'
        '          stroke="%(cyan)s" stroke-width="1.4"/>'
        '  <circle cx="30" cy="82" r="3.4" fill="%(cyan)s"/>'
        '</svg>' % {
            "cyan": theme.COLORS["cyan"], "muted": theme.COLORS["text_muted"],
            "metal": theme.COLORS["metal"], "dim": theme.COLORS["text_dim"],
            "metal_light": theme.COLORS["metal_light"],
            "chambers": "".join(chambers),
        })


def render():
    """Draw the start screen."""
    theme.html(
        '<div class="r5-frame" style="text-align:center;padding:38px 22px 30px 22px">'
        '  <div class="r5-game-title">%(title)s</div>'
        '  <div class="r5-game-subtitle">%(subtitle)s</div>'
        '  <p style="color:%(muted)s;max-width:66ch;margin:22px auto 4px auto;'
        '     text-align:left">%(intro)s</p>'
        '</div>' % {
            "title": story.GAME_TITLE,
            "subtitle": story.GAME_SUBTITLE,
            "muted": theme.COLORS["text_muted"],
            "intro": story.INTRO_STORY,
        })

    theme.html('<div class="r5-card">'
               '<div class="r5-card-title"><span class="r5-card-accent"></span>'
               'Facility cross-section</div>%s</div>' % _facility_diagram())

    inventory.bar()

    left, middle, right = st.columns([1, 1, 1])
    with middle:
        if buttons.link("▸ Start Game", "start", role="primary"):
            game_state.go_to_room(story.FIRST_ROOM)
            st.rerun()
        if game_state.has_progress():
            current = st.session_state.get("current_room", story.FIRST_ROOM)
            if buttons.link("↻ Continue — Room %d" % current, "continue",
                            role="success"):
                game_state.go_to_room(current)
                st.rerun()

    with st.expander("Academic project information"):
        st.markdown(
            """
This is a teaching project about reinforcement learning. Each of the five
chambers is solved with a different method, so the same escape-room story is
used to compare how those methods behave.

| Room | Chamber | Method | Model of the world |
|------|---------|--------|--------------------|
| 1 | Laser Security Chamber | Value Iteration and Policy Iteration | known in advance |
| 2 | Broken Bridge Sector | SARSA | learned from experience |
| 3 | Reactor Control Chamber | Q-Learning | learned from experience |
| 4 | Drone Wind Tunnel | Policy Gradient | learned, continuous actions |
| 5 | Adaptive Storage Facility | Function approximation | learned, unseen layouts |

**Room 1 is implemented.** Rooms 2 to 5 are the next stages of the project and
are shown as locked.

No reinforcement learning library is used anywhere: every algorithm is written
out by hand so the update rules can be read directly in the source.
            """)
