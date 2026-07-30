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
    """A side-on drawing of the climb out of the facility.

    The chambers step upwards from the charging plate on sub-level 4 to the gate
    at the surface, because that ascent is the whole story of the game — drawn
    flat, the picture would say nothing about where R-5 is trying to get to.

    Inline SVG, so it needs no image files and no extra dependency, and so it can
    use the palette directly.
    """
    box_width = 134
    box_height = 58
    step_x = 158
    step_y = 34
    floor_y = 186

    levels = []
    chambers = []
    for index, number in enumerate(story.ROOM_NUMBERS):
        record = story.room(number)
        status = game_state.room_status(number)
        x = 118 + index * step_x
        y = floor_y - index * step_y

        if status == game_state.STATUS_SOLVED:
            colour, glow = theme.COLORS["success"], 1.0
        elif status == game_state.STATUS_ACTIVE:
            colour, glow = theme.COLORS["cyan"], 1.0
        else:
            colour, glow = theme.COLORS["metal"], 0.5

        # The depth each chamber sits at, ruled all the way across. This is what
        # fills the space above the staircase: the scale being climbed.
        levels.append(
            '<line x1="76" y1="%(mid)d" x2="880" y2="%(mid)d" stroke="%(line)s"'
            '      stroke-width="1" stroke-dasharray="1 7" opacity="0.5"/>'
            '<text x="4" y="%(ty)d" font-size="8.5" font-family="monospace"'
            '      fill="%(dim)s" letter-spacing="1.6">−%(depth)d m</text>'
            % {"mid": y + box_height / 2, "ty": y + box_height / 2 + 3,
               "line": theme.COLORS["grid_line"],
               "dim": theme.COLORS["text_dim"],
               "depth": (len(story.ROOM_NUMBERS) - index) * 6})

        # The shaft up to the next chamber, drawn before the boxes so they sit
        # on top of it.
        if index < len(story.ROOM_NUMBERS) - 1:
            chambers.append(
                '<path d="M %d %d H %d V %d" fill="none" stroke="%s"'
                ' stroke-width="1.3" stroke-dasharray="3 4" opacity="0.7"/>'
                % (x + box_width, y + box_height / 2, x + step_x + 16,
                   y + box_height / 2 - step_y, theme.COLORS["grid_line"]))

        chambers.append(
            '<g opacity="%(glow).2f">'
            '  <rect x="%(x)d" y="%(y)d" width="%(w)d" height="%(h)d" rx="4"'
            '        fill="%(panel)s" stroke="%(colour)s" stroke-width="1.4"/>'
            '  <rect x="%(x)d" y="%(y)d" width="3" height="%(h)d" fill="%(colour)s"/>'
            '  <text x="%(tx)d" y="%(ty)d" font-size="19" font-weight="700"'
            '        font-family="monospace" fill="%(colour)s">%(num)d</text>'
            '  <text x="%(lx)d" y="%(ty)d" font-size="10"'
            '        font-family="monospace" fill="%(label)s"'
            '        letter-spacing="1.5">%(short)s</text>'
            '</g>' % {
                "glow": glow, "x": x, "y": y, "w": box_width, "h": box_height,
                "tx": x + 15, "lx": x + 40, "ty": y + box_height / 2 + 6,
                "panel": theme.COLORS["background_soft"],
                "colour": colour, "num": number,
                "label": theme.COLORS["text_muted"],
                "short": record["algorithm_short"],
            })

    return (
        '<svg viewBox="0 0 900 268" width="100%%"'
        '     role="img" aria-label="Cross-section of the climb from sub-level 4'
        ' through the five test chambers to the surface"'
        '     style="display:block;max-width:100%%;height:auto">'
        '  <defs>'
        '    <linearGradient id="r5surface" x1="0" y1="0" x2="0" y2="1">'
        '      <stop offset="0" stop-color="%(cyan)s" stop-opacity="0.30"/>'
        '      <stop offset="1" stop-color="%(cyan)s" stop-opacity="0"/>'
        '    </linearGradient>'
        '  </defs>'
        '  %(levels)s'
        '  <rect x="0" y="0" width="900" height="30" fill="url(#r5surface)"/>'
        '  <line x1="0" y1="30" x2="900" y2="30" stroke="%(cyan)s"'
        '        stroke-width="1.1" opacity="0.6"/>'
        '  <text x="4" y="20" font-size="9.5" font-family="monospace"'
        '        fill="%(cyan)s" letter-spacing="3.4">SURFACE — EXIT GATE</text>'
        '  %(chambers)s'
        '  <line x1="0" y1="258" x2="900" y2="258" stroke="%(metal)s"'
        '        stroke-width="1.1"/>'
        '  <text x="4" y="252" font-size="9.5" font-family="monospace"'
        '        fill="%(dim)s" letter-spacing="3.4">'
        'SUB-LEVEL 4 — R-5 CHARGING PLATE</text>'
        '  <circle cx="70" cy="%(ry)d" r="10" fill="%(metal_light)s"'
        '          stroke="%(cyan)s" stroke-width="1.4"/>'
        '  <circle cx="70" cy="%(ry)d" r="3.6" fill="%(cyan)s"/>'
        '</svg>' % {
            "cyan": theme.COLORS["cyan"], "metal": theme.COLORS["metal"],
            "dim": theme.COLORS["text_dim"],
            "metal_light": theme.COLORS["metal_light"],
            "ry": floor_y + box_height / 2,
            "levels": "".join(levels),
            "chambers": "".join(chambers),
        })


def render():
    """Draw the start screen."""
    # The title and the route out of the facility are one object: the picture is
    # the most characteristic thing here, so it gets the hero rather than sitting
    # in a card underneath a mostly empty panel.
    theme.html(
        '<div class="r5-frame" style="padding:30px 26px 8px 26px">'
        '  <div style="display:flex;flex-wrap:wrap;align-items:flex-end;'
        '       justify-content:space-between;gap:22px">'
        '    <div>'
        '      <div class="r5-game-title">Project <em>R-5</em></div>'
        '      <div class="r5-game-subtitle">%(subtitle)s</div>'
        '    </div>'
        '    <p style="color:%(muted)s;max-width:52ch;margin:0">%(intro)s</p>'
        '  </div>'
        # Below about a tablet the diagram would shrink until its labels were
        # unreadable, so it keeps its size and scrolls inside its own box.
        '  <div style="margin-top:26px;overflow-x:auto">'
        '    <div style="min-width:640px">%(diagram)s</div>'
        '  </div>'
        '</div>' % {
            "subtitle": story.GAME_SUBTITLE,
            "muted": theme.COLORS["text_muted"],
            "intro": story.INTRO_STORY,
            "diagram": _facility_diagram(),
        })

    left, middle, right = st.columns([1.4, 1, 1.4])
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

    inventory.bar()

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
| 4 | Drone Wind Tunnel | Semi-Gradient SARSA over tile coding | learned, continuous state |
| 5 | Adaptive Storage Facility | Semi-Gradient Q-Learning over local features | learned, unseen layouts |

All five chambers are implemented. A chamber stays locked until the one before
it has been escaped.

No reinforcement learning library is used anywhere: every algorithm is written
out by hand so the update rules can be read directly in the source.
            """)
