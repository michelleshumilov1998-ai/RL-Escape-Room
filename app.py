"""PROJECT R-5 — Escape from the Learning Lab.

Run it with:

    streamlit run app.py

This file is only the router.  It sets up the page, injects the shared
stylesheet and then hands over to whichever screen is current.  A single-page
router is used instead of Streamlit's automatic `pages/` folder so the start
screen, the room transitions and the navigation rail stay under our own control
and look the same everywhere.
"""

import streamlit as st

from core import story
from rooms import locked_page
from rooms.room1 import page as room1_page
from rooms.room2 import page as room2_page
from rooms.room3 import page as room3_page
from rooms.room4 import page as room4_page
from rooms.room5 import page as room5_page
from ui import completion, game_state, nav, start_screen, theme, transitions


def main():
    theme.configure_page()
    game_state.initialise()

    screen = st.session_state["screen"]
    room_number = st.session_state["current_room"]

    # The stylesheet carries the current room's accent colour, so injecting it
    # after the room is known keeps every frame and card the right colour.
    theme.inject(room_number if screen == game_state.SCREEN_ROOM else None)

    _sidebar_shell(room_number if screen == game_state.SCREEN_ROOM else None)

    if screen == game_state.SCREEN_START:
        start_screen.render()
        return

    if screen == game_state.SCREEN_COMPLETE:
        completion.render()
        return

    if screen == game_state.SCREEN_TRANSITION:
        transitions.room_cleared(st.session_state["transition_room"])
        return

    _render_room(room_number)


def _render_room(room_number):
    """Draw one room, after its briefing has been dismissed."""
    record = story.room(room_number)

    if not record["implemented"]:
        locked_page.render(room_number)
        return

    # The briefing is shown once per room per session and is skippable.
    if transitions.room_intro(room_number):
        return

    if room_number == 1:
        room1_page.render()
        return
    if room_number == 2:
        room2_page.render()
        return
    if room_number == 3:
        room3_page.render()
        return
    if room_number == 4:
        room4_page.render()
        return
    if room_number == 5:
        room5_page.render()
        return

    locked_page.render(room_number)


def _sidebar_shell(room_number=None):
    """The part of the sidebar that is the same on every screen.

    The ascent gauge is drawn here rather than by each room, so it sits above
    every room's own controls and reads the same on the start and closing
    screens too.
    """
    st.sidebar.markdown(
        '<div style="padding:2px 0 10px 0">'
        '  <div class="r5-label">Institute Archive · Sub-level 4</div>'
        '  <div style="font-size:1.15rem;font-weight:800;letter-spacing:0.14em;'
        '       color:%s">%s</div>'
        '  <div class="r5-label">%s</div>'
        '</div>' % (theme.COLORS["cyan"], story.GAME_TITLE, story.GAME_SUBTITLE),
        unsafe_allow_html=True)

    if st.sidebar.button("⌂ Start screen", key="secondary_home"):
        game_state.go_to_start()
        st.rerun()

    st.sidebar.divider()
    nav.ascent_gauge(room_number)
    st.sidebar.divider()


if __name__ == "__main__":
    main()
