"""Putting the isometric chamber onto a Streamlit page.

The HTML template is read from disk once and cached.  The scene is injected as
JSON, and the resulting page is handed to `streamlit.components.v1.html`, which
runs it in a sandboxed iframe.

The animation clock lives inside that iframe.  Nothing the player does with the
playback controls causes a Streamlit rerun, which is what keeps the animation
smooth and stops it restarting every time a widget changes.
"""

import json
import os

import streamlit as st

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "web", "room_iso.html")

PLACEHOLDER = "__SCENE__"

# How tall the component needs to be: the chamber plus nameplate, controls and
# legend.  The canvas sizes itself to the available width inside this box.
HEIGHT_DEFAULT = 760


@st.cache_data(show_spinner=False)
def _template():
    """The HTML template, read once per session."""
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


def build_html(scene):
    """The finished page for one scene."""
    payload = json.dumps(scene, separators=(",", ":"))
    # A literal "</script>" anywhere in the data would close the script tag
    # early, so the slash is escaped.  JSON treats "<\/" as "</".
    payload = payload.replace("</", "<\\/")
    return _template().replace(PLACEHOLDER, payload)


def render(scene, height=HEIGHT_DEFAULT):
    """Draw the chamber.

    The finished HTML is kept in `st.session_state` and only rebuilt when the
    scene's signature changes, so an unrelated widget change does not throw the
    string away and rebuild it.
    """
    cache = st.session_state.setdefault("_iso_html_cache", {})
    signature = scene.get("signature", "")

    if cache.get("signature") != signature:
        cache["signature"] = signature
        cache["html"] = build_html(scene)

    # `st.iframe` replaced `st.components.v1.html`; fall back to the older call
    # if this Streamlit is too old to have it.
    if hasattr(st, "iframe"):
        st.iframe(cache["html"], height=height)
    else:  # pragma: no cover - only on older Streamlit
        import streamlit.components.v1 as components
        components.html(cache["html"], height=height, scrolling=False)
