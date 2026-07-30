"""The information cards used by every room.

There is one card builder, `card`, and a handful of small functions that fill it
in for a particular purpose.  Every room uses these, so the Mission card in
Room 1 looks exactly like the Mission card in Room 5.

A card is a title plus a list of rows.  Each row is a tuple:

    (label, value)              or
    (label, value, tone)

where `tone` is one of "", "pos", "neg", "warn", "info" and only changes the
colour of the value.  The label always carries the meaning in words as well, so
the cards stay readable without relying on colour.
"""

from core import actions
from ui import theme

TONE_NEUTRAL = ""
TONE_GOOD = "pos"
TONE_BAD = "neg"
TONE_WARN = "warn"
TONE_INFO = "info"


def _escape(value):
    """Very small HTML escape, enough for the values we display."""
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def card(title, rows, icon=""):
    """HTML for one information card."""
    pieces = ['<div class="r5-card">',
              '<div class="r5-card-title"><span class="r5-card-accent"></span>',
              _escape(title)]
    if icon:
        pieces.append('<span style="margin-left:auto">%s</span>' % icon)
    pieces.append("</div>")

    for row in rows:
        label, value = row[0], row[1]
        tone = row[2] if len(row) > 2 else TONE_NEUTRAL
        pieces.append(
            '<div class="r5-row"><span class="r5-row-key">%s</span>'
            '<span class="r5-row-value %s">%s</span></div>'
            % (_escape(label), tone, _escape(value)))

    pieces.append("</div>")
    return "".join(pieces)


def show(title, rows, icon=""):
    """Draw a card straight onto the page."""
    theme.html(card(title, rows, icon=icon))


def _tick(flag):
    """A yes/no value that reads clearly without depending on colour."""
    return "YES" if flag else "no"


# ----------------------------------------------------------------------
# The five standard cards
# ----------------------------------------------------------------------

def mission_card(objective, requirements=(), complete=False):
    """What the room is asking the robot to do, and how far along it is."""
    rows = [("Objective", objective)]
    for label, done in requirements:
        rows.append(("  " + label, _tick(done), TONE_GOOD if done else TONE_WARN))
    rows.append(("Status", "COMPLETE" if complete else "in progress",
                 TONE_GOOD if complete else TONE_WARN))
    return card("Mission", rows)


def agent_state_card(frame, extra_rows=()):
    """Where R-5 is and what it is carrying."""
    if frame is None:
        rows = [("Position", "—"), ("Battery", "—"), ("Previous direction", "—")]
    else:
        rows = [
            ("Position", "row %d, col %d" % (frame["row"], frame["col"]), TONE_INFO),
            ("Battery", _tick(frame["has_battery"]),
             TONE_GOOD if frame["has_battery"] else TONE_NEUTRAL),
            ("Previous direction", "%s %s" % (
                actions.direction_arrow(frame["previous_direction"]),
                frame["previous_direction_name"])),
            ("Chosen action", "%s %s" % (
                actions.direction_arrow(frame["action"]) if frame["action"] else "·",
                frame["action_name"])),
            ("Actual direction", "%s %s" % (
                actions.direction_arrow(frame["actual_direction"])
                if frame["actual_direction"] else "·",
                frame["actual_direction_name"]),
             TONE_WARN if frame["slipped"] else TONE_NEUTRAL),
        ]
    rows.extend(extra_rows)
    return card("Agent State", rows)


def reward_card(frame, laser_hits=0, wall_collisions=0, teleports=0):
    """The reward story of the current episode."""
    if frame is None:
        return card("Reward", [("Last reward", "—"), ("Cumulative", "—")])

    last = frame["reward"]
    total = frame["cumulative_reward"]
    rows = [
        ("Last reward", "%+.0f" % last,
         TONE_GOOD if last > 0 else (TONE_BAD if last < -1 else TONE_NEUTRAL)),
        ("Cumulative reward", "%+.0f" % total,
         TONE_GOOD if total > 0 else TONE_BAD),
        ("Laser hits", "%d  (%+d each)" % (laser_hits, -30) if laser_hits else "0",
         TONE_BAD if laser_hits else TONE_NEUTRAL),
        ("Wall collisions", str(wall_collisions),
         TONE_WARN if wall_collisions else TONE_NEUTRAL),
        ("Teleporter bonuses", "%d  (+5 each)" % teleports if teleports else "0",
         TONE_INFO if teleports else TONE_NEUTRAL),
        ("Battery bonus", "+10 collected" if frame["has_battery"] else "not collected",
         TONE_GOOD if frame["has_battery"] else TONE_NEUTRAL),
    ]
    return card("Reward", rows)


def algorithm_card(name, update_rule, model_known, exploration, hyperparameters):
    """What method is being used, and with which settings."""
    rows = [
        ("Algorithm", name, TONE_INFO),
        ("Update rule", update_rule),
        ("Environment model", "known in advance" if model_known else "unknown",
         TONE_GOOD if model_known else TONE_WARN),
        ("Exploration", exploration),
    ]
    for label, value in hyperparameters:
        rows.append(("  " + label, value))
    return card("Algorithm", rows)


def room_status_card(status, rows=()):
    """The room's own state: running, paused, escaped, and so on."""
    body = [("State", status.upper(),
             TONE_GOOD if status.upper() in ("SOLVED", "ESCAPED") else
             (TONE_BAD if status.upper() == "FAILED" else TONE_INFO))]
    body.extend(rows)
    return card("Room Status", body)


# ----------------------------------------------------------------------
# The event log
# ----------------------------------------------------------------------

def event_log(frames, limit=60):
    """A scrolling list of what happened, newest at the bottom."""
    if not frames:
        return ('<div class="r5-log"><div class="r5-log-line">'
                '<span class="r5-log-step">—</span>'
                '<span>No run recorded yet.</span></div></div>')

    # Every room writes its own event flags, so the log reads them all with
    # `.get` and colours whichever ones are present. Room 1 reports laser hits,
    # slips and teleports; Room 2 reports pit falls and collapsing bridges.
    lines = ['<div class="r5-log">']
    for frame in frames[-limit:]:
        css_class = ""
        if frame.get("laser_hit") or frame.get("pit_fall"):
            css_class = "is-laser"
        elif frame.get("used_teleport"):
            css_class = "is-teleport"
        elif frame.get("battery_collected") or frame.get("keycard_collected"):
            css_class = "is-battery"
        elif frame.get("reached_exit"):
            css_class = "is-exit"
        elif frame.get("bridge_collapsed"):
            css_class = "is-battery"
        elif frame.get("slipped"):
            css_class = "is-slip"

        lines.append(
            '<div class="r5-log-line %s"><span class="r5-log-step">%d</span>'
            '<span>%s</span><span style="margin-left:auto">%+.0f</span></div>'
            % (css_class, frame["step"], _escape(frame["event"]), frame["reward"]))
    lines.append("</div>")
    return "".join(lines)
