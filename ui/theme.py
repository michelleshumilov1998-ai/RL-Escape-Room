"""The one and only place where the look of PROJECT R-5 is defined.

Every colour, font size, spacing value and CSS rule used anywhere in the game
lives here.  No page and no room defines its own palette, so all five rooms
automatically look like parts of the same laboratory.

Rooms are allowed one accent colour of their own (`ROOM_ACCENTS`) for their
nameplate and warning lights, but it is always chosen from the global palette.
"""

import streamlit as st

# ----------------------------------------------------------------------
# The palette
# ----------------------------------------------------------------------

COLORS = {
    # `ink` is the floor of the whole facility: one step darker than
    # `background`, so a panel painted in `background` genuinely reads as
    # sitting *above* the page instead of merging into it.
    "ink": "#050A12",
    "background": "#08111F",
    "background_soft": "#101C2E",
    "panel": "#14233A",
    "panel_raised": "#1B2E4A",
    "metal": "#3B4658",
    "metal_light": "#55627A",
    "grid_line": "#263A55",
    "cyan": "#38D9FF",
    "cyan_dim": "#1E7F9B",
    "purple": "#8B5CF6",
    "purple_dim": "#5B3BA8",
    "danger": "#FF3B5C",
    "danger_dim": "#8F2136",
    "warning": "#FFC857",
    "warning_dim": "#8A6A2A",
    "success": "#32E875",
    "success_dim": "#1B7F41",
    "text": "#EAF2FF",
    "text_muted": "#9FB0C7",
    "text_dim": "#6C7C93",
}

# Each room gets one accent from the palette above, used for its nameplate.
ROOM_ACCENTS = {
    1: COLORS["cyan"],
    2: COLORS["warning"],
    3: COLORS["danger"],
    4: COLORS["purple"],
    5: COLORS["success"],
}

# ----------------------------------------------------------------------
# Typography, spacing and shape
# ----------------------------------------------------------------------

# The faces are fetched once from Google Fonts.  If that request fails — offline,
# or behind a proxy — every rule below still resolves through the fallback stacks,
# so the layout is identical and only the lettering changes.
FONT_IMPORT = ("@import url('https://fonts.googleapis.com/css2?"
               "family=Archivo:wght@400;500;600;700;800&"
               "family=Archivo+Expanded:wght@600;700;800&"
               "family=IBM+Plex+Mono:wght@400;500;600&display=swap');")

FONT_STACK = ('"Archivo", "Segoe UI", -apple-system, BlinkMacSystemFont, '
              '"Helvetica Neue", Arial, sans-serif')
# Wide, engineered capitals for nameplates — the lettering stencilled onto lab
# equipment rather than the lettering of a website.
DISPLAY_STACK = ('"Archivo Expanded", "Archivo", "Segoe UI", '
                 '-apple-system, BlinkMacSystemFont, Arial, sans-serif')
MONO_STACK = ('"IBM Plex Mono", "SF Mono", "Cascadia Mono", Consolas, '
              '"Liberation Mono", monospace')

# One scale, used everywhere.  Nothing outside this list should appear in CSS.
FONT_SIZES = {
    "game_title": "3.1rem",
    "room_title": "1.55rem",
    "section": "1.05rem",
    "body": "0.95rem",
    "metric_value": "1.5rem",
    "label": "0.72rem",
    "tiny": "0.66rem",
}

SPACING = {"xs": "4px", "sm": "8px", "md": "14px", "lg": "22px", "xl": "34px"}
RADIUS = {"sm": "5px", "md": "8px", "lg": "12px"}
SHADOW_CARD = "0 10px 30px rgba(0, 0, 0, 0.55)"
SHADOW_GLOW = "0 0 18px rgba(56, 217, 255, 0.28)"


def rgba(hex_color, alpha):
    """Turn '#38D9FF' plus an alpha into a CSS rgba() string."""
    hex_color = hex_color.lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return "rgba(%d, %d, %d, %s)" % (red, green, blue, alpha)


# ----------------------------------------------------------------------
# Status badges — every room reports its state with the same five words
# ----------------------------------------------------------------------

STATUS_STYLES = {
    "LOCKED": {"color": COLORS["text_dim"], "icon": "■"},
    # A chamber R-5 is standing in but has not solved yet. It is not LOCKED —
    # the door is open, the method simply has not been run.
    "STANDBY": {"color": COLORS["text_muted"], "icon": "◌"},
    "ACTIVE": {"color": COLORS["cyan"], "icon": "▶"},
    "SOLVED": {"color": COLORS["success"], "icon": "✔"},
    "TRAINING": {"color": COLORS["warning"], "icon": "◐"},
    "EVALUATING": {"color": COLORS["purple"], "icon": "◆"},
    "RUNNING": {"color": COLORS["cyan"], "icon": "▶"},
    "PAUSED": {"color": COLORS["warning"], "icon": "‖"},
    "FAILED": {"color": COLORS["danger"], "icon": "✖"},
    "ESCAPED": {"color": COLORS["success"], "icon": "✔"},
}


def status_badge(status, note=""):
    """HTML for one status badge.  Colour AND an icon AND the word itself, so it
    never relies on colour alone to carry the meaning."""
    style = STATUS_STYLES.get(status.upper(), STATUS_STYLES["ACTIVE"])
    extra = ('<span class="r5-badge-note">%s</span>' % note) if note else ""
    return ('<span class="r5-badge" style="color:%s;border-color:%s;background:%s">'
            '<span class="r5-badge-icon">%s</span>%s%s</span>'
            % (style["color"], rgba(style["color"], 0.45),
               rgba(style["color"], 0.10), style["icon"], status.upper(), extra))


# ----------------------------------------------------------------------
# The stylesheet
# ----------------------------------------------------------------------

def _stylesheet():
    """Build the whole stylesheet from the values above."""
    return """
/* ---------- page shell ---------------------------------------------- */
.stApp {
    background:
        radial-gradient(1200px 700px at 12%% -10%%, %(cyan_glow)s, transparent 60%%),
        radial-gradient(900px 600px at 105%% 8%%, %(purple_glow)s, transparent 62%%),
        linear-gradient(180deg, %(background_soft)s 0%%, %(background)s 45%%,
                        %(ink)s 100%%);
    background-attachment: fixed;
    color: %(text)s;
    font-family: %(font)s;
}
/* faint circuit grid over the whole facility */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        linear-gradient(%(grid_faint)s 1px, transparent 1px),
        linear-gradient(90deg, %(grid_faint)s 1px, transparent 1px);
    background-size: 46px 46px;
    mask-image: radial-gradient(ellipse at 50%% 0%%, black 10%%, transparent 78%%);
    -webkit-mask-image: radial-gradient(ellipse at 50%% 0%%, black 10%%, transparent 78%%);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { padding-top: 1.6rem; max-width: 1500px; }
.stApp a { color: %(cyan)s; }

/* ---------- typography ---------------------------------------------- */
.stApp, .stApp p, .stApp li, .stApp label { font-size: %(body)s; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: %(display)s;
    color: %(text)s;
    letter-spacing: 0.04em;
}
.stApp h1 { font-size: %(room_title)s; font-weight: 700; }
.stApp h2 { font-size: %(section)s; font-weight: 700; }
.stApp h3 { font-size: %(section)s; font-weight: 600; }
.r5-mono { font-family: %(mono)s; }

/* The title is stencilled, not glowing: flat text in the facility's own cyan,
   with the wide display face doing the work instead of a gradient fill. */
.r5-game-title {
    font-family: %(display)s;
    font-size: %(game_title)s;
    font-weight: 800;
    letter-spacing: 0.09em;
    line-height: 0.94;
    margin: 0;
    color: %(text)s;
    text-transform: uppercase;
}
.r5-game-title em {
    font-style: normal;
    color: %(cyan)s;
}
.r5-game-subtitle {
    font-family: %(mono)s;
    color: %(text_muted)s;
    letter-spacing: 0.32em;
    font-size: %(label)s;
    text-transform: uppercase;
    margin-top: 10px;
}
.r5-label {
    font-family: %(mono)s;
    font-size: %(label)s;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: %(text_dim)s;
}

/* ---------- the laboratory frame around every room ------------------ */
.r5-frame {
    position: relative;
    border: 1px solid %(metal)s;
    border-radius: %(radius_lg)s;
    background:
        linear-gradient(180deg, %(panel_raised)s 0%%, %(panel)s 100%%);
    box-shadow: %(shadow_card)s, inset 0 1px 0 %(metal_light_soft)s;
    padding: %(lg)s;
    margin-bottom: %(md)s;
    overflow: hidden;
}
/* The accent runs down the left edge rather than across the top: it reads as the
   lit strip on a rack unit, and it stacks legibly when frames sit one above
   another. */
.r5-frame::after {
    content: "";
    position: absolute;
    top: 0; bottom: 0; left: 0;
    width: 2px;
    background: linear-gradient(180deg,
        %(accent)s 0%%, %(accent_soft)s 46%%, transparent 100%%);
}
.r5-nameplate {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: %(md)s;
    justify-content: space-between;
}
.r5-sector {
    font-family: %(mono)s;
    font-size: %(label)s;
    letter-spacing: 0.24em;
    color: %(accent)s;
    text-transform: uppercase;
}
.r5-room-title {
    font-family: %(display)s;
    font-size: %(room_title)s;
    font-weight: 700;
    letter-spacing: 0.01em;
    margin: 3px 0 0 0;
    color: %(text)s;
}
.r5-room-subtitle {
    font-family: %(mono)s;
    font-size: %(label)s;
    letter-spacing: 0.16em;
    color: %(warning)s;
    text-transform: uppercase;
}
.r5-mission { color: %(text_muted)s; margin: 10px 0 0 0; max-width: 68ch; }

/* small warning lights on the frame */
.r5-lights { display: inline-flex; gap: 6px; align-items: center; }
.r5-light {
    width: 8px; height: 8px; border-radius: 50%%;
    box-shadow: 0 0 8px currentColor;
    background: currentColor;
}
.r5-light-blink { animation: r5-blink 2.4s ease-in-out infinite; }
@keyframes r5-blink { 0%%,72%%,100%% { opacity: 1; } 84%% { opacity: 0.18; } }

/* ---------- cards ---------------------------------------------------- */
.r5-card {
    border: 1px solid %(grid_line)s;
    border-radius: %(radius_md)s;
    background: linear-gradient(180deg, %(panel)s, %(background_soft)s);
    /* A lit top edge and a dark inner floor: the card reads as a plate screwed
       onto the console rather than a rectangle drawn on it. */
    box-shadow: %(shadow_card)s,
                inset 0 1px 0 %(metal_light_soft)s,
                inset 0 -18px 26px -22px %(ink)s;
    padding: %(md)s;
    margin-bottom: %(sm)s;
}
.r5-card-title {
    font-family: %(mono)s;
    font-size: %(label)s;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: %(text_dim)s;
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid %(grid_line)s;
}
.r5-card-title .r5-card-accent {
    width: 3px; height: 12px; border-radius: 2px; background: %(accent)s;
}
.r5-row {
    display: flex; justify-content: space-between; gap: %(md)s;
    padding: 4px 0;
    border-bottom: 1px dashed %(grid_faint_strong)s;
}
.r5-row:last-child { border-bottom: none; }
.r5-row-key { color: %(text_muted)s; font-size: %(body)s; }
.r5-row-value { font-family: %(mono)s; color: %(text)s; text-align: right; }
.r5-row-value.pos { color: %(success)s; }
.r5-row-value.neg { color: %(danger)s; }
.r5-row-value.warn { color: %(warning)s; }
.r5-row-value.info { color: %(cyan)s; }

.r5-metric-label {
    font-family: %(mono)s; font-size: %(label)s; letter-spacing: 0.16em;
    text-transform: uppercase; color: %(text_dim)s;
}
.r5-metric-value { font-size: %(metric_value)s; font-weight: 700; color: %(text)s; }

/* ---------- badges --------------------------------------------------- */
.r5-badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: %(mono)s; font-size: %(label)s;
    letter-spacing: 0.14em; text-transform: uppercase;
    border: 1px solid; border-radius: 999px;
    padding: 3px 11px; white-space: nowrap;
}
.r5-badge-icon { font-size: 0.8em; }
.r5-badge-note { opacity: 0.72; letter-spacing: 0.04em; }

/* ---------- room navigation rail ------------------------------------- */
/* One band, not three stacked strips: the chamber sequence, the components and
   the progress count all sit on the same line so the eye crosses them once. */
.r5-band {
    display: flex; align-items: center; flex-wrap: wrap;
    gap: %(md)s;
    border: 1px solid %(grid_line)s;
    border-radius: %(radius_md)s;
    background: linear-gradient(180deg, %(background_soft)s, %(ink)s);
    padding: 9px %(md)s;
    margin-bottom: %(sm)s;
}
.r5-band-progress { margin-left: auto; white-space: nowrap; }

.r5-nav {
    display: flex; align-items: stretch; gap: 0; flex-wrap: wrap;
    flex: 0 1 auto; min-width: 0;
}
.r5-nav-step {
    flex: 0 1 auto; min-width: 0;
    border: 1px solid transparent;
    border-radius: %(radius_sm)s;
    padding: 3px 9px;
    position: relative;
    display: flex; align-items: baseline; gap: 7px;
}
.r5-nav-step + .r5-nav-step { margin-left: 15px; }
.r5-nav-step + .r5-nav-step::before {
    content: "";
    position: absolute; left: -15px; top: 50%%;
    width: 15px; height: 1px; background: %(grid_line)s;
}
.r5-nav-num {
    font-family: %(mono)s; font-size: %(tiny)s; letter-spacing: 0.14em;
    color: %(text_dim)s; text-transform: uppercase;
}
.r5-nav-name { font-size: %(label)s; color: %(text_muted)s; }
.r5-nav-step.is-current {
    border-color: %(cyan_border)s;
    background: %(cyan_wash)s;
}
.r5-nav-step.is-current .r5-nav-name { color: %(text)s; }
.r5-nav-step.is-current .r5-nav-num { color: %(cyan)s; }
.r5-nav-step.is-done .r5-nav-num { color: %(success)s; }
.r5-nav-step.is-locked { opacity: 0.45; }

/* ---------- the ascent gauge ----------------------------------------- */
/* R-5 is climbing out of sub-level 4, so progress is vertical. The gauge is the
   shaft: the surface at the top, the charging plate at the bottom, and the five
   chambers as landings in between. */
.r5-gauge { padding: 2px 0 6px 0; }
.r5-gauge-cap {
    font-family: %(mono)s; font-size: %(tiny)s; letter-spacing: 0.2em;
    text-transform: uppercase; color: %(text_dim)s;
    padding-left: 26px;
}
.r5-gauge-cap.is-surface { color: %(cyan)s; }
.r5-gauge-shaft { position: relative; padding: 6px 0; }
/* the shaft wall the landings hang off */
.r5-gauge-shaft::before {
    content: "";
    position: absolute; left: 9px; top: 0; bottom: 0;
    width: 2px;
    background: linear-gradient(180deg,
        %(cyan_border)s 0%%, %(grid_line)s 30%%, %(grid_line)s 100%%);
}
.r5-landing {
    position: relative;
    display: flex; align-items: center; gap: 10px;
    padding: 5px 0 5px 26px;
    min-height: 30px;
}
.r5-landing::before {
    content: "";
    position: absolute; left: 4px; top: 50%%;
    width: 12px; height: 12px; margin-top: -6px;
    border-radius: 50%%;
    border: 2px solid %(grid_line)s;
    background: %(ink)s;
}
.r5-landing-name {
    font-size: %(tiny)s; color: %(text_dim)s;
    letter-spacing: 0.04em; line-height: 1.25;
}
.r5-landing-method {
    font-family: %(mono)s; font-size: %(tiny)s;
    color: %(text_dim)s; letter-spacing: 0.1em; text-transform: uppercase;
}
.r5-landing.is-done::before {
    border-color: %(success)s; background: %(success)s;
}
.r5-landing.is-done .r5-landing-name { color: %(text_muted)s; }
/* the car: where R-5 actually is */
.r5-landing.is-here::before {
    border-color: %(cyan)s; background: %(cyan)s;
    box-shadow: 0 0 0 4px %(cyan_wash)s, %(shadow_glow)s;
}
.r5-landing.is-here .r5-landing-name { color: %(text)s; font-weight: 600; }
.r5-landing.is-here .r5-landing-method { color: %(cyan)s; }
.r5-landing.is-locked { opacity: 0.4; }
.r5-gauge-readout {
    font-family: %(mono)s; font-size: %(tiny)s; letter-spacing: 0.12em;
    color: %(text_dim)s; text-transform: uppercase;
    padding-left: 26px; margin-top: 4px;
}
.r5-gauge-readout b { color: %(cyan)s; font-weight: 600; }

/* ---------- inventory bar -------------------------------------------- */
.r5-inventory { display: flex; flex-wrap: wrap; gap: %(sm)s; align-items: center; }
.r5-item {
    display: inline-flex; align-items: center; gap: 8px;
    border: 1px solid %(grid_line)s; border-radius: 999px;
    padding: 5px 13px 5px 9px;
    background: %(background_soft)s;
    font-size: %(label)s; letter-spacing: 0.08em;
    color: %(text_muted)s;
}
.r5-item.is-held {
    border-color: %(warning_border)s; color: %(text)s;
    background: linear-gradient(180deg, %(warning_wash)s, %(background_soft)s);
}
.r5-item-icon { font-size: 1rem; line-height: 1; }
.r5-item.is-missing .r5-item-icon { filter: grayscale(1); opacity: 0.42; }

/* ---------- buttons -------------------------------------------------- */
/* Role comes from the widget key prefix, so every room gets the same set. */
.stButton > button, .stDownloadButton > button {
    width: 100%%;
    font-family: %(mono)s;
    font-size: %(label)s;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 600;
    border-radius: %(radius_sm)s;
    border: 1px solid %(metal)s;
    background: linear-gradient(180deg, %(panel_raised)s, %(panel)s);
    color: %(text)s;
    padding: 0.5rem 0.7rem;
    transition: border-color 120ms ease, box-shadow 120ms ease, transform 60ms ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: %(cyan_border)s;
    box-shadow: %(shadow_glow)s;
}
.stButton > button:active { transform: translateY(1px); }
.stButton > button:disabled,
.stButton > button:disabled:hover {
    color: %(text_dim)s; border-color: %(grid_line)s;
    background: %(background_soft)s; box-shadow: none; opacity: 1;
}
/* The primary action is the only button carrying weight: a lit left edge, a
   brighter face and heavier lettering, so "Solve Room" never reads as the peer
   of "Reset Animation". */
div[class*="st-key-primary_"] .stButton > button,
div[class*="st-key-primary_"] button {
    border-color: %(cyan_border)s; color: %(cyan)s;
    background: linear-gradient(180deg, %(cyan_wash)s, %(panel)s);
    border-left: 3px solid %(cyan)s;
    font-weight: 700;
    box-shadow: inset 0 1px 0 %(cyan_wash)s;
}
div[class*="st-key-primary_"] button:hover {
    border-color: %(cyan)s;
    box-shadow: %(shadow_glow)s, inset 0 1px 0 %(cyan_wash)s;
}
div[class*="st-key-success_"] button {
    border-color: %(success_border)s; color: %(success)s;
    background: linear-gradient(180deg, %(success_wash)s, %(panel)s);
}
div[class*="st-key-success_"] button:hover {
    box-shadow: 0 0 18px %(success_wash)s; border-color: %(success)s;
}
div[class*="st-key-danger_"] button {
    border-color: %(danger_border)s; color: %(danger)s; background: transparent;
}
div[class*="st-key-danger_"] button:hover {
    box-shadow: 0 0 18px %(danger_wash)s; border-color: %(danger)s;
}

/* ---------- sidebar -------------------------------------------------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, %(panel)s 0%%, %(background)s 100%%);
    border-right: 1px solid %(metal)s;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { font-size: %(section)s; }
[data-testid="stSidebar"] label { color: %(text_muted)s !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-family: %(mono)s; font-size: %(label)s; letter-spacing: 0.08em;
}

/* ---------- inputs --------------------------------------------------- */
[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
    background: %(background_soft)s !important;
    border-color: %(grid_line)s !important;
    color: %(text)s !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: %(cyan)s; border-color: %(cyan)s;
    box-shadow: 0 0 10px %(cyan_glow_strong)s;
}
[data-testid="stMetricLabel"] p {
    font-family: %(mono)s; font-size: %(label)s;
    letter-spacing: 0.14em; text-transform: uppercase; color: %(text_dim)s;
}
[data-testid="stMetricValue"] { font-size: %(metric_value)s; color: %(text)s; }

/* ---------- containers, tabs, tables --------------------------------- */
[data-testid="stExpander"] {
    border: 1px solid %(grid_line)s; border-radius: %(radius_md)s;
    background: %(background_soft)s; overflow: hidden;
}
[data-testid="stExpander"] summary { font-family: %(mono)s; font-size: %(label)s;
    letter-spacing: 0.12em; text-transform: uppercase; color: %(text_muted)s; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid %(grid_line)s; }
.stTabs [data-baseweb="tab"] {
    font-family: %(mono)s; font-size: %(label)s; letter-spacing: 0.12em;
    text-transform: uppercase; color: %(text_dim)s;
}
.stTabs [aria-selected="true"] { color: %(cyan)s !important; }
[data-testid="stDataFrame"] { border: 1px solid %(grid_line)s; border-radius: %(radius_sm)s; }
[data-testid="stAlert"] { border-radius: %(radius_md)s; border: 1px solid %(grid_line)s; }
hr { border-color: %(grid_line)s; }
code { color: %(cyan)s; background: %(background_soft)s; border-radius: 4px; }

/* ---------- the event log ------------------------------------------- */
.r5-log {
    font-family: %(mono)s; font-size: %(label)s;
    max-height: 260px; overflow-y: auto;
    border: 1px solid %(grid_line)s; border-radius: %(radius_sm)s;
    background: %(background)s; padding: 8px 10px;
}
.r5-log-line { display: flex; gap: 10px; padding: 2px 0; color: %(text_muted)s; }
.r5-log-step { color: %(text_dim)s; min-width: 3.2em; text-align: right; }
.r5-log-line.is-laser { color: %(danger)s; }
.r5-log-line.is-teleport { color: %(purple)s; }
.r5-log-line.is-battery { color: %(warning)s; }
.r5-log-line.is-exit { color: %(success)s; }
.r5-log-line.is-slip { color: %(cyan)s; }

/* ---------- transition / completion panels --------------------------- */
.r5-transition {
    border: 1px solid %(success_border)s;
    border-radius: %(radius_lg)s;
    background:
        radial-gradient(600px 200px at 50%% 0%%, %(success_wash)s, transparent 70%%),
        linear-gradient(180deg, %(panel)s, %(background)s);
    padding: %(xl)s %(lg)s;
    text-align: center;
}
.r5-transition-title {
    font-family: %(mono)s; font-size: 1.4rem; font-weight: 700;
    letter-spacing: 0.28em; color: %(success)s; text-transform: uppercase;
}

/* ---------- accessibility & responsiveness --------------------------- */
/* Streamlit columns sit in a flex row that does not wrap by itself.  Below this
   width the rows are allowed to wrap and each column is given a floor, so the
   layout stacks instead of squeezing text down to one word per line. */
@media (max-width: 1200px) {
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 190px;
    }
    /* The environment and its cards get a full row each, so the map stays big
       and the cards stay readable. */
    .st-key-room1_main [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 100%%;
        flex: 1 1 100%%;
    }
}

@media (max-width: 900px) {
    .r5-game-title { font-size: 1.9rem; letter-spacing: 0.1em; }
    .r5-frame { padding: %(md)s; }
    .r5-nav-step + .r5-nav-step { margin-left: 0; }
    .r5-nav-step + .r5-nav-step::before { display: none; }
    .r5-nameplate { gap: %(sm)s; }
}
@media (prefers-reduced-motion: reduce) {
    .r5-light-blink { animation: none; }
    .stButton > button { transition: none; }
}
:focus-visible { outline: 2px solid %(cyan)s !important; outline-offset: 2px; }
""" % {
        "font": FONT_STACK,
        "display": DISPLAY_STACK,
        "mono": MONO_STACK,
        "body": FONT_SIZES["body"],
        "label": FONT_SIZES["label"],
        "tiny": FONT_SIZES["tiny"],
        "section": FONT_SIZES["section"],
        "room_title": FONT_SIZES["room_title"],
        "game_title": FONT_SIZES["game_title"],
        "metric_value": FONT_SIZES["metric_value"],
        "sm": SPACING["sm"],
        "md": SPACING["md"],
        "lg": SPACING["lg"],
        "xl": SPACING["xl"],
        "radius_sm": RADIUS["sm"],
        "radius_md": RADIUS["md"],
        "radius_lg": RADIUS["lg"],
        "shadow_card": SHADOW_CARD,
        "shadow_glow": SHADOW_GLOW,
        "ink": COLORS["ink"],
        "background": COLORS["background"],
        "background_soft": COLORS["background_soft"],
        "panel": COLORS["panel"],
        "panel_raised": COLORS["panel_raised"],
        "metal": COLORS["metal"],
        "grid_line": COLORS["grid_line"],
        "cyan": COLORS["cyan"],
        "purple": COLORS["purple"],
        "danger": COLORS["danger"],
        "warning": COLORS["warning"],
        "success": COLORS["success"],
        "text": COLORS["text"],
        "text_muted": COLORS["text_muted"],
        "text_dim": COLORS["text_dim"],
        "accent": "var(--r5-accent, %s)" % COLORS["cyan"],
        "accent_soft": "var(--r5-accent-soft, %s)" % rgba(COLORS["cyan"], 0.5),
        "metal_light_soft": rgba(COLORS["metal_light"], 0.30),
        "grid_faint": rgba(COLORS["grid_line"], 0.30),
        "grid_faint_strong": rgba(COLORS["grid_line"], 0.55),
        "cyan_glow": rgba(COLORS["cyan"], 0.09),
        "cyan_glow_strong": rgba(COLORS["cyan"], 0.45),
        "cyan_border": rgba(COLORS["cyan"], 0.55),
        "cyan_wash": rgba(COLORS["cyan"], 0.11),
        "purple_glow": rgba(COLORS["purple"], 0.10),
        "success_border": rgba(COLORS["success"], 0.50),
        "success_wash": rgba(COLORS["success"], 0.12),
        "danger_border": rgba(COLORS["danger"], 0.55),
        "danger_wash": rgba(COLORS["danger"], 0.16),
        "warning_border": rgba(COLORS["warning"], 0.50),
        "warning_wash": rgba(COLORS["warning"], 0.12),
    }


def configure_page(title="PROJECT R-5"):
    """Set up the Streamlit page.  Must run before anything is drawn."""
    st.set_page_config(page_title=title, page_icon="\U0001F916",
                       layout="wide", initial_sidebar_state="expanded")


def inject(room_number=None):
    """Put the stylesheet on the page.

    Called once per page render.  When a room number is given, that room's
    accent colour is exposed as the CSS variable `--r5-accent`, which the frame
    and card headers pick up automatically.
    """
    accent = ROOM_ACCENTS.get(room_number, COLORS["cyan"])
    variables = (":root{--r5-accent:%s;--r5-accent-soft:%s;}"
                 % (accent, rgba(accent, 0.5)))
    # `@import` is only honoured as the first rule in a stylesheet, so the font
    # request has to come before the variables and before everything else.
    st.markdown("<style>%s%s%s</style>"
                % (FONT_IMPORT, variables, _stylesheet()),
                unsafe_allow_html=True)


def html(markup):
    """Shorthand for writing one of our own HTML blocks to the page."""
    st.markdown(markup, unsafe_allow_html=True)
