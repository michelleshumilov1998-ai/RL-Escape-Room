"""One dark Matplotlib theme for every graph in the game.

Matplotlib's defaults produce white plots, which would look like holes cut out of
the interface.  Everything drawn anywhere in PROJECT R-5 goes through `figure()`
so that can never happen.

The colour roles match the rest of the application:

    cyan    the primary metric
    purple  the comparison metric
    green   success
    red     failure and penalties
    amber   warnings and secondary events
"""

import matplotlib

matplotlib.use("Agg")  # No interactive window: the figures go straight to Streamlit.

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ui.theme import COLORS

# The order lines are drawn in when no colour is given.
CYCLE = [COLORS["cyan"], COLORS["purple"], COLORS["success"],
         COLORS["warning"], COLORS["danger"], COLORS["metal_light"]]

PRIMARY = COLORS["cyan"]
COMPARISON = COLORS["purple"]
SUCCESS = COLORS["success"]
FAILURE = COLORS["danger"]
WARNING = COLORS["warning"]

LINE_WIDTH = 1.9
GRID_ALPHA = 0.28


def apply():
    """Set the global Matplotlib style.  Safe to call more than once."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["background_soft"],
        "axes.facecolor": COLORS["background"],
        "savefig.facecolor": COLORS["background_soft"],
        "axes.edgecolor": COLORS["grid_line"],
        "axes.labelcolor": COLORS["text_muted"],
        "axes.titlecolor": COLORS["text"],
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLORS["grid_line"],
        "grid.alpha": GRID_ALPHA,
        "grid.linewidth": 0.7,
        "xtick.color": COLORS["text_dim"],
        "ytick.color": COLORS["text_dim"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": COLORS["text"],
        "legend.facecolor": COLORS["panel"],
        "legend.edgecolor": COLORS["grid_line"],
        "legend.fontsize": 8,
        "legend.labelcolor": COLORS["text_muted"],
        "lines.linewidth": LINE_WIDTH,
        "lines.solid_capstyle": "round",
        "figure.autolayout": True,
        "axes.prop_cycle": plt.cycler(color=CYCLE),
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def figure(width=6.4, height=3.2):
    """A new figure and axes, already themed.

    Built as a bare `Figure` rather than through `plt.subplots`, because pyplot
    keeps a global reference to every figure it creates and never lets go. This
    app draws a lot of graphs, so going through pyplot would pile them up in
    memory for the life of the session. A plain Figure is not registered
    anywhere, still picks up the rcParams set by `apply`, and is accepted by both
    `st.pyplot` and `savefig`.
    """
    apply()
    fig = Figure(figsize=(width, height))
    axes = fig.add_subplot(111)
    return fig, axes


def finish(axes, title=None, xlabel=None, ylabel=None, legend=False):
    """Apply the shared titles, labels and legend placement."""
    if title:
        axes.set_title(title, pad=9, loc="left")
    if xlabel:
        axes.set_xlabel(xlabel)
    if ylabel:
        axes.set_ylabel(ylabel)
    if legend:
        axes.legend(loc="best", framealpha=0.85)
    return axes


def empty_figure(message):
    """A themed placeholder used before a room has been solved."""
    fig, axes = figure(height=2.4)
    axes.text(0.5, 0.5, message, ha="center", va="center",
              color=COLORS["text_dim"], fontsize=9,
              transform=axes.transAxes)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.grid(False)
    return fig
