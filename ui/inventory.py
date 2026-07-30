"""The persistent inventory bar.

Shows the five components R-5 needs to escape the facility.  Held components are
lit in amber with their icon; the rest are dimmed and greyed.  The bar looks the
same on every page, including the start screen.
"""

from core import story
from ui import game_state, theme


def bar(show_missing=True):
    """The inventory strip, drawn straight onto the page."""
    pieces = ['<div class="r5-inventory">',
              '<span class="r5-label" style="margin-right:6px">Components</span>']

    for number in story.ROOM_NUMBERS:
        record = story.room(number)
        held = game_state.holds(record["component"])
        if not held and not show_missing:
            continue
        css_class = "r5-item is-held" if held else "r5-item is-missing"
        # The word "locked" spells the state out, so the amber glow is never the
        # only thing telling the player whether a component is held.
        suffix = "" if held else " · locked"
        pieces.append(
            '<span class="%s" title="Recovered in Room %d — %s">'
            '<span class="r5-item-icon">%s</span>%s%s</span>'
            % (css_class, number, record["name"], record["component_icon"],
               record["component"], suffix))

    pieces.append("</div>")
    theme.html("".join(pieces))


def acquired_panel(room_number):
    """The 'new component acquired' block used by the transition screen."""
    record = story.room(room_number)
    theme.html(
        '<div style="display:flex;align-items:center;gap:14px;justify-content:center;'
        'margin:18px 0">'
        '  <span style="font-size:2.2rem;line-height:1">%s</span>'
        '  <div style="text-align:left">'
        '    <div class="r5-label">Component acquired</div>'
        '    <div style="font-size:1.15rem;font-weight:700;color:%s">%s</div>'
        '  </div>'
        '</div>' % (record["component_icon"], theme.COLORS["warning"],
                    record["component"]))
