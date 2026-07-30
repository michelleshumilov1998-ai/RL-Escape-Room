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

FONT_STACK = ('"Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, '
              '"Helvetica Neue", Arial, sans-serif')
MONO_STACK = ('"JetBrains Mono", "SF Mono", "Cascadia Mono", Consolas, '
              '"Liberation Mono", monospace')

# One scale, used everywhere.  Nothing outside this list should appear in CSS.
FONT_SIZES = {
    "game_title": "2.6rem",
    "room_title": "1.55rem",
    "section": "1.05rem",
    "body": "0.95rem",
    "metric_value": "1.5rem",
    "label": "0.72rem",
    "tiny": "0.66rem",
}

SPACING = {"xs": "4px", "sm": "8px", "md": "14px", "lg": "22px", "xl": "34px"}
RADIUS = {"sm": "6px", "md": "10px", "lg": "16px"}
SHADOW_CARD = "0 6px 22px rgba(0, 0, 0, 0.45)"
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
        linear-gradient(180deg, %(background_soft)s 0%%, %(background)s 55%%);
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
[data-testid="stMainBlockContainer"] { padding-top: 2.2rem; max-width: 1500px; }
.stApp a { color: %(cyan)s; }

/* ---------- typography ---------------------------------------------- */
.stApp, .stApp p, .stApp li, .stApp label { font-size: %(body)s; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: %(font)s;
    color: %(text)s;
    letter-spacing: 0.02em;
}
.stApp h1 { font-size: %(room_title)s; font-weight: 700; }
.stApp h2 { font-size: %(section)s; font-weight: 650; }
.stApp h3 { font-size: %(section)s; font-weight: 600; }
.r5-mono { font-family: %(mono)s; }

.r5-game-title {
    font-family: %(font)s;
    font-size: %(game_title)s;
    font-weight: 800;
    letter-spacing: 0.16em;
    line-height: 1.05;
    margin: 0;
    background: linear-gradient(96deg, %(text)s 8%%, %(cyan)s 52%%, %(purple)s 96%%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
.r5-game-subtitle {
    color: %(text_muted)s;
    letter-spacing: 0.34em;
    font-size: %(label)s;
    text-transform: uppercase;
    margin-top: 6px;
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
/* the rivet strip along the top of every frame */
.r5-frame::after {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg,
        transparent, %(accent_soft)s 18%%, %(accent)s 50%%, %(accent_soft)s 82%%, transparent);
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
    font-size: %(room_title)s;
    font-weight: 700;
    margin: 2px 0 0 0;
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
    box-shadow: %(shadow_card)s;
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
.r5-nav { display: flex; align-items: stretch; gap: 0; flex-wrap: wrap; }
.r5-nav-step {
    flex: 1 1 96px; min-width: 96px;
    border: 1px solid %(grid_line)s;
    border-radius: %(radius_sm)s;
    background: %(background_soft)s;
    padding: 8px 10px;
    position: relative;
}
.r5-nav-step + .r5-nav-step { margin-left: 14px; }
.r5-nav-step + .r5-nav-step::before {
    content: "";
    position: absolute; left: -14px; top: 50%%;
    width: 14px; height: 1px; background: %(grid_line)s;
}
.r5-nav-num {
    font-family: %(mono)s; font-size: %(tiny)s; letter-spacing: 0.18em;
    color: %(text_dim)s; text-transform: uppercase;
}
.r5-nav-name { font-size: %(label)s; color: %(text_muted)s; margin-top: 2px; }
.r5-nav-step.is-current {
    border-color: %(cyan_border)s;
    box-shadow: %(shadow_glow)s;
    background: linear-gradient(180deg, %(cyan_wash)s, %(background_soft)s);
}
.r5-nav-step.is-current .r5-nav-name { color: %(text)s; }
.r5-nav-step.is-done { border-color: %(success_border)s; }
.r5-nav-step.is-done .r5-nav-num { color: %(success)s; }
.r5-nav-step.is-locked { opacity: 0.5; }

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
div[class*="st-key-primary_"] .stButton > button,
div[class*="st-key-primary_"] button {
    border-color: %(cyan_border)s; color: %(cyan)s;
    background: linear-gradient(180deg, %(cyan_wash)s, %(panel)s);
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
    st.markdown("<style>%s%s</style>" % (variables, _stylesheet()),
                unsafe_allow_html=True)


def html(markup):
    """Shorthand for writing one of our own HTML blocks to the page."""
    st.markdown(markup, unsafe_allow_html=True)
