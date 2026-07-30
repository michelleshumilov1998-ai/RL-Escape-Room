"""The room navigation rail and the laboratory frame header.

Both appear on every page, look identical everywhere, and read their contents
from `core/story.py` and `ui/game_state.py`.
"""

import streamlit as st

from core import story
from ui import game_state, theme

STATUS_CLASSES = {
    game_state.STATUS_SOLVED: "is-done",
    game_state.STATUS_ACTIVE: "is-current",
    game_state.STATUS_LOCKED: "is-locked",
}

STATUS_MARKS = {
    game_state.STATUS_SOLVED: "✔ CLEARED",
    game_state.STATUS_ACTIVE: "▶ ACTIVE",
    game_state.STATUS_LOCKED: "■ LOCKED",
}


def rail(current_room=None):
    """Room 1 -> Room 2 -> ... -> Room 5, with a status on each step.

    The current room is highlighted in cyan, cleared rooms in green and locked
    rooms are dimmed.  Every step also spells its status out in words, so the
    rail does not depend on colour alone.
    """
    pieces = ['<div class="r5-nav">']
    for number in story.ROOM_NUMBERS:
        status = game_state.room_status(number)
        css_class = STATUS_CLASSES[status]
        if number == current_room and status != game_state.STATUS_LOCKED:
            css_class = "is-current"
        record = story.room(number)
        pieces.append(
            '<div class="r5-nav-step %s">'
            '<div class="r5-nav-num">Room %d · %s</div>'
            '<div class="r5-nav-name">%s</div>'
            '</div>' % (css_class, number, STATUS_MARKS[status], record["name"]))
    pieces.append("</div>")
    theme.html("".join(pieces))


def warning_lights(count=3, accent=None, blink=True):
    """A little row of indicator lights for the laboratory frame."""
    accent = accent or theme.COLORS["cyan"]
    palette = [accent, theme.COLORS["warning"], theme.COLORS["danger"]]
    lamps = []
    for index in range(count):
        colour = palette[index % len(palette)]
        css_class = "r5-light r5-light-blink" if blink and index == 0 else "r5-light"
        lamps.append('<span class="%s" style="color:%s"></span>' % (css_class, colour))
    return '<span class="r5-lights">%s</span>' % "".join(lamps)


def room_header(room_number, status, status_note="", algorithm=None):
    """The metal nameplate that frames every room.

    Sector code, room title, subtitle, mission line, algorithm and status — the
    same six things in the same place in all five rooms.
    """
    record = story.room(room_number)
    accent = theme.ROOM_ACCENTS.get(room_number, theme.COLORS["cyan"])
    algorithm = algorithm or record["algorithm"]

    theme.html(
        '<div class="r5-frame">'
        '  <div class="r5-nameplate">'
        '    <div>'
        '      <div class="r5-sector">%(sector)s %(lights)s</div>'
        '      <div class="r5-room-title">Room %(number)d of %(total)d — %(name)s</div>'
        '      <div class="r5-room-subtitle">%(subtitle)s</div>'
        '    </div>'
        '    <div style="text-align:right">'
        '      <div class="r5-label">Learning method</div>'
        '      <div class="r5-mono" style="color:%(accent)s;font-size:1rem">%(algorithm)s</div>'
        '      <div style="margin-top:8px">%(badge)s</div>'
        '    </div>'
        '  </div>'
        '  <p class="r5-mission">%(mission)s</p>'
        '</div>' % {
            "sector": record["sector"],
            "lights": warning_lights(accent=accent),
            "number": room_number,
            "total": story.LAST_ROOM,
            "name": record["name"],
            "subtitle": record["subtitle"],
            "accent": accent,
            "algorithm": algorithm,
            "badge": theme.status_badge(status, status_note),
            "mission": record["mission"],
        })


def progress_line():
    """A one-line reminder of how far through the facility the player is."""
    done = game_state.solved_count()
    total = len(story.ROOM_NUMBERS)
    filled = "▰" * done + "▱" * (total - done)
    theme.html('<div class="r5-label" style="text-align:right">'
               'Facility progress &nbsp;%s&nbsp; %d / %d chambers cleared</div>'
               % (filled, done, total))


def sidebar_navigation(current_room):
    """Buttons in the sidebar for jumping between unlocked rooms."""
    st.sidebar.markdown('<div class="r5-label">Facility map</div>',
                        unsafe_allow_html=True)
    for number in story.ROOM_NUMBERS:
        record = story.room(number)
        status = game_state.room_status(number)
        locked = status == game_state.STATUS_LOCKED
        label = "%s  Room %d — %s" % (STATUS_MARKS[status].split()[0], number,
                                      record["name"])
        role = "primary" if number == current_room else "secondary"
        clicked = st.sidebar.button(label, key="%s_nav_room%d" % (role, number),
                                   disabled=locked,
                                   help=("Clear the previous chamber first"
                                         if locked else record["mission"]))
        if clicked:
            game_state.go_to_room(number)
            st.rerun()
