"""The small amount of progress the game remembers between page renders.

Everything lives in `st.session_state`, so a Streamlit rerun never loses the
player's position, and solving a room never has to happen twice.

Each room keeps its own private area, reached with `room_store(number)`, so the
rooms can never overwrite each other's results.
"""

import streamlit as st

from core import story

SCREEN_START = "start"
SCREEN_ROOM = "room"
SCREEN_TRANSITION = "transition"
SCREEN_COMPLETE = "complete"

STATUS_LOCKED = "LOCKED"
STATUS_ACTIVE = "ACTIVE"
STATUS_SOLVED = "SOLVED"


def initialise():
    """Create anything that is missing.  Safe to call on every render."""
    st.session_state.setdefault("screen", SCREEN_START)
    st.session_state.setdefault("current_room", story.FIRST_ROOM)
    st.session_state.setdefault("solved_rooms", [])
    st.session_state.setdefault("inventory", [])
    st.session_state.setdefault("intro_seen", [])
    st.session_state.setdefault("transition_room", None)
    st.session_state.setdefault("rooms", {})
    st.session_state.setdefault("quality", "High")
    st.session_state.setdefault("muted", True)
    st.session_state.setdefault("view_mode", "Gameplay")


def room_store(number):
    """The private storage area for one room, created on first use."""
    rooms = st.session_state["rooms"]
    if number not in rooms:
        rooms[number] = {}
    return rooms[number]


# ----------------------------------------------------------------------
# Progress
# ----------------------------------------------------------------------

def has_progress():
    """True if the player has already started playing."""
    return bool(st.session_state.get("solved_rooms")) or \
        st.session_state.get("current_room", story.FIRST_ROOM) != story.FIRST_ROOM


def is_solved(number):
    return number in st.session_state.get("solved_rooms", [])


def is_unlocked(number):
    """A room is open once every room before it has been solved."""
    if number == story.FIRST_ROOM:
        return True
    return is_solved(number - 1)


def room_status(number):
    """One of LOCKED / ACTIVE / SOLVED, for the navigation rail and badges."""
    if is_solved(number):
        return STATUS_SOLVED
    if is_unlocked(number) and story.room(number)["implemented"]:
        return STATUS_ACTIVE
    return STATUS_LOCKED


def mark_solved(number):
    """Record that a room has been escaped and collect its component."""
    if number not in st.session_state["solved_rooms"]:
        st.session_state["solved_rooms"].append(number)
    component = story.room(number)["component"]
    if component not in st.session_state["inventory"]:
        st.session_state["inventory"].append(component)


def holds(component):
    return component in st.session_state.get("inventory", [])


def solved_count():
    return len(st.session_state.get("solved_rooms", []))


# ----------------------------------------------------------------------
# Navigation between screens
# ----------------------------------------------------------------------

def go_to_room(number):
    st.session_state["current_room"] = number
    st.session_state["screen"] = SCREEN_ROOM


def go_to_start():
    st.session_state["screen"] = SCREEN_START


def go_to_complete():
    """Show the closing scene."""
    st.session_state["screen"] = SCREEN_COMPLETE


def all_solved():
    """True once every implemented room has been cleared."""
    return all(is_solved(number) for number in story.ROOM_NUMBERS
               if story.room(number)["implemented"])


def go_to_transition(number):
    """Show the 'room cleared' screen for the room just completed."""
    st.session_state["transition_room"] = number
    st.session_state["screen"] = SCREEN_TRANSITION


def intro_seen(number):
    return number in st.session_state.get("intro_seen", [])


def mark_intro_seen(number):
    if number not in st.session_state["intro_seen"]:
        st.session_state["intro_seen"].append(number)
