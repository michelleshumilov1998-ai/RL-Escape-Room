"""Cinematic room introductions and 'room cleared' screens.

Both are deliberately short.  The introduction is skippable and is only shown
once per room per session, remembered in `st.session_state`, so a Streamlit
rerun never makes the player sit through it again.
"""

import streamlit as st

from core import story
from ui import buttons, game_state, inventory, theme


def room_intro(room_number):
    """Show the chamber briefing before a room, if it has not been seen yet.

    Returns True when the introduction is on screen, which tells the page to
    stop drawing and wait.  Returns False once it has been dismissed.
    """
    if game_state.intro_seen(room_number):
        return False

    record = story.room(room_number)
    accent = theme.ROOM_ACCENTS.get(room_number, theme.COLORS["cyan"])

    theme.html(
        '<div class="r5-frame" style="text-align:center;padding:44px 22px">'
        '  <div class="r5-sector" style="letter-spacing:0.34em">%(sector)s</div>'
        '  <div style="font-size:2rem;font-weight:800;letter-spacing:0.16em;'
        '       margin:14px 0 6px 0;color:%(accent)s">%(title)s</div>'
        '  <div class="r5-room-subtitle">%(subtitle)s</div>'
        '  <div style="max-width:60ch;margin:26px auto 0 auto;text-align:left">'
        '    <div class="r5-row"><span class="r5-row-key">Environment model</span>'
        '      <span class="r5-row-value info">%(model)s</span></div>'
        '    <div class="r5-row"><span class="r5-row-key">Planning system</span>'
        '      <span class="r5-row-value info">%(algorithm)s</span></div>'
        '    <div class="r5-row"><span class="r5-row-key">Objective</span>'
        '      <span class="r5-row-value">%(mission)s</span></div>'
        '    <div class="r5-row"><span class="r5-row-key">Warning</span>'
        '      <span class="r5-row-value neg">%(warning)s</span></div>'
        '  </div>'
        '</div>' % {
            "sector": record["sector"],
            "accent": accent,
            "title": record["name"].upper(),
            "subtitle": record["subtitle"],
            "model": ("Known in advance" if record["model_known"] else
                      "Unknown — must be learned"),
            "algorithm": record["algorithm"],
            "mission": record["mission"],
            "warning": record["briefing_warning"],
        })

    left, middle, right = st.columns([1, 1, 1])
    with middle:
        if buttons.link("▸ Enter chamber", "intro%d" % room_number, role="primary"):
            game_state.mark_intro_seen(room_number)
            st.rerun()
    return True


def room_cleared(room_number):
    """The screen shown after a room has been escaped."""
    record = story.room(room_number)
    following = story.next_room(room_number)

    theme.html(
        '<div class="r5-transition">'
        '  <div class="r5-transition-title">%(headline)s</div>'
        '  <p style="color:%(muted)s;max-width:56ch;margin:16px auto 0 auto">%(story)s</p>'
        '</div>' % {
            "headline": record["cleared_headline"],
            "muted": theme.COLORS["text_muted"],
            "story": record["cleared_story"],
        })

    inventory.acquired_panel(room_number)
    inventory.bar()

    if following is None:
        theme.html('<p style="text-align:center;color:%s;margin-top:22px">'
                   'Every chamber is clear. The surface gate is next.</p>'
                   % theme.COLORS["text_muted"])
        return

    next_record = story.room(following)
    ready = next_record["implemented"]

    theme.html(
        '<div class="r5-card" style="margin-top:22px;text-align:center">'
        '  <div class="r5-label">Next sector</div>'
        '  <div style="font-size:1.2rem;font-weight:700;margin:6px 0">'
        '    ROOM %(number)d — %(name)s</div>'
        '  <div class="r5-label">Learning method: %(algorithm)s</div>'
        '  <p style="color:%(muted)s;margin-top:12px">%(note)s</p>'
        '</div>' % {
            "number": following,
            "name": next_record["name"].upper(),
            "algorithm": next_record["algorithm"],
            "muted": theme.COLORS["text_muted"],
            "note": (next_record["mission"] if ready else
                     "This chamber has not been built yet — it is the next stage "
                     "of the project."),
        })

    left, middle, right = st.columns([1, 1, 1])
    with middle:
        if ready:
            if buttons.link("▸ Continue to Room %d" % following,
                            "next%d" % following, role="primary"):
                game_state.go_to_room(following)
                st.rerun()
        else:
            if buttons.link("↩ Back to Room %d" % room_number,
                            "back%d" % room_number, role="secondary"):
                game_state.go_to_room(room_number)
                st.rerun()
