"""The closing scene, shown once every chamber has been cleared.

Deliberately restrained: the facility gate opens, the lighting turns from red to
green, and the numbers from all five rooms are laid out. It reuses the same frame,
cards and palette as everything else — the last screen should look like it belongs
to the same game as the first.
"""

import streamlit as st

from core import story
from ui import buttons, game_state, inventory, theme


def _gate_scene():
    """The gate opening onto daylight, as inline SVG.

    Drawn rather than animated so it loads instantly and needs no assets.
    """
    return (
        '<svg viewBox="0 0 760 240" width="100%%" height="240" role="img"'
        '     aria-label="The facility gate open onto a sunrise"'
        '     style="display:block;max-width:100%%">'
        '  <defs>'
        '    <linearGradient id="sky" x1="0" y1="1" x2="0" y2="0">'
        '      <stop offset="0" stop-color="%(warning)s" stop-opacity="0.85"/>'
        '      <stop offset="0.55" stop-color="%(cyan)s" stop-opacity="0.35"/>'
        '      <stop offset="1" stop-color="%(bg)s" stop-opacity="0"/>'
        '    </linearGradient>'
        '    <linearGradient id="beam" x1="0" y1="0" x2="1" y2="0">'
        '      <stop offset="0" stop-color="%(success)s" stop-opacity="0"/>'
        '      <stop offset="0.5" stop-color="%(success)s" stop-opacity="0.45"/>'
        '      <stop offset="1" stop-color="%(success)s" stop-opacity="0"/>'
        '    </linearGradient>'
        '  </defs>'
        '  <rect x="0" y="0" width="760" height="240" fill="%(bg)s"/>'
        '  <rect x="250" y="30" width="260" height="180" fill="url(#sky)"/>'
        '  <circle cx="380" cy="196" r="46" fill="%(warning)s" opacity="0.55"/>'
        '  <rect x="250" y="188" width="260" height="52" fill="%(panel)s"'
        '        opacity="0.5"/>'
        '  <rect x="0" y="26" width="760" height="10" fill="%(metal)s"/>'
        '  <rect x="120" y="36" width="130" height="204" fill="%(metal)s"/>'
        '  <rect x="510" y="36" width="130" height="204" fill="%(metal)s"/>'
        '  <rect x="240" y="36" width="12" height="204" fill="%(metalLight)s"/>'
        '  <rect x="508" y="36" width="12" height="204" fill="%(metalLight)s"/>'
        '  <rect x="250" y="150" width="260" height="14" fill="url(#beam)"/>'
        '  <circle cx="186" cy="70" r="7" fill="%(success)s"/>'
        '  <circle cx="574" cy="70" r="7" fill="%(success)s"/>'
        '  <text x="186" y="120" text-anchor="middle" font-size="9"'
        '        font-family="monospace" fill="%(muted)s" letter-spacing="2">GATE</text>'
        '  <text x="574" y="120" text-anchor="middle" font-size="9"'
        '        font-family="monospace" fill="%(muted)s" letter-spacing="2">OPEN</text>'
        '  <circle cx="330" cy="206" r="13" fill="%(metalLight)s"'
        '          stroke="%(cyan)s" stroke-width="1.6"/>'
        '  <circle cx="330" cy="206" r="5" fill="%(cyan)s"/>'
        '  <text x="330" y="182" text-anchor="middle" font-size="9"'
        '        font-family="monospace" fill="%(text)s">R-5</text>'
        '</svg>' % {
            "bg": theme.COLORS["background"], "panel": theme.COLORS["panel"],
            "metal": theme.COLORS["metal"],
            "metalLight": theme.COLORS["metal_light"],
            "cyan": theme.COLORS["cyan"], "success": theme.COLORS["success"],
            "warning": theme.COLORS["warning"], "text": theme.COLORS["text"],
            "muted": theme.COLORS["text_muted"],
        })


def render():
    """Draw the completion screen."""
    theme.html(
        '<div class="r5-transition">'
        '  <div class="r5-transition-title">PROJECT R-5 COMPLETE</div>'
        '  <p style="color:%(muted)s;max-width:56ch;margin:16px auto 0 auto">'
        '    All laboratory sectors cleared.<br/>'
        '    Adaptive navigation system verified.<br/>'
        '    Exit route unlocked.</p>'
        '</div>' % {"muted": theme.COLORS["text_muted"]})

    theme.html('<div class="r5-card" style="margin-top:14px">%s</div>'
               % _gate_scene())

    inventory.bar()

    st.markdown("### What R-5 learned on the way out")
    rows = []
    for number in story.ROOM_NUMBERS:
        record = story.room(number)
        store = st.session_state.get("rooms", {}).get(number, {})
        run = store.get("run")
        result = store.get("primary") or store.get("solution")
        rows.append({
            "Room": "%d — %s" % (number, record["name"]),
            "Method": record["algorithm"],
            "Cleared": "yes" if game_state.is_solved(number) else "no",
            "Component": record["component"],
            "Best run": ("%+.0f" % run["total_reward"]) if run else "—",
            "Steps": str(run["steps"]) if run else "—",
            "Training time": ("%.1f s" % result["runtime_seconds"]
                              if result and result.get("runtime_seconds") else "—"),
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.caption("Five chambers, five methods: a known model solved by planning, then "
               "four rooms learned from experience — on-policy, off-policy, "
               "continuous control, and finally a policy that transfers to "
               "warehouses it had never seen.")

    left, middle, right = st.columns(3)
    with left:
        if buttons.link("↺ Replay a room", "complete_replay", role="primary"):
            game_state.go_to_room(story.FIRST_ROOM)
            st.rerun()
    with middle:
        if buttons.link("≣ Room 5 analysis", "complete_analysis", role="success"):
            game_state.go_to_room(5)
            st.rerun()
    with right:
        if buttons.link("⌂ Main menu", "complete_menu", role="secondary"):
            game_state.go_to_start()
            st.rerun()
