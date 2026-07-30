"""The graphs for Room 2.

Room 2 learns from experience, so unlike Room 1 its graphs are training curves:
what the reward did over the episodes, how exploration decayed, how often the
robot fell, and which route it settled on.

The ten graphs the assignment asks for:

  1. episode reward                6. Q-value convergence
  2. moving-average reward         7. pit falls
  3. episode length                8. bridge usage
  4. success rate                  9. keycard collection rate
  5. epsilon decay               10. training time

Every figure is built through `plots/style.py`, so they all match the interface
and the Room 1 graphs.
"""

from core import actions
from plots import style
from rooms.room2 import map_data, sarsa_agent
from ui.theme import COLORS

WINDOW_DEFAULT = 50


def _episodes(values):
    return range(1, len(values) + 1)


def _empty(message):
    return style.empty_figure(message)


# ----------------------------------------------------------------------
# 1 and 2: reward, raw and smoothed
# ----------------------------------------------------------------------

def episode_reward(results, window=WINDOW_DEFAULT):
    """Total reward per episode, with the moving average on top.

    The raw curve is drawn faintly because it is very noisy in a room where one
    wrong step costs -101; the moving average is the line to read.
    """
    if not results:
        return _empty("Train the agent to see the reward curve")

    fig, axes = style.figure()
    colours = [style.PRIMARY, style.COMPARISON]

    for index, result in enumerate(results):
        rewards = result["history"]["reward"]
        if not rewards:
            continue
        colour = colours[index % len(colours)]
        axes.plot(_episodes(rewards), rewards, color=colour, alpha=0.22,
                  linewidth=0.8)
        smoothed = sarsa_agent.moving_average(rewards, window)
        axes.plot(_episodes(smoothed), smoothed, color=colour, linewidth=2.0,
                  label="%s (mean of %d)" % (result["algorithm"], window))

    axes.axhline(map_data.perfect_return("walkway"), color=style.SUCCESS,
                 linestyle="--", linewidth=1.1,
                 label="perfect walkway (+%d)" % map_data.perfect_return("walkway"))
    axes.axhline(map_data.perfect_return("span"), color=style.WARNING,
                 linestyle=":", linewidth=1.1,
                 label="perfect bridge (+%d)" % map_data.perfect_return("span"))
    style.finish(axes, title="Episode reward", xlabel="Episode",
                 ylabel="Total reward", legend=True)
    return fig


def moving_average_reward(results, window=WINDOW_DEFAULT):
    """Just the smoothed reward, for when the raw curve is too busy."""
    if not results:
        return _empty("Train the agent to see the moving average")

    fig, axes = style.figure()
    colours = [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        smoothed = sarsa_agent.moving_average(result["history"]["reward"], window)
        axes.plot(_episodes(smoothed), smoothed,
                  color=colours[index % len(colours)], label=result["algorithm"])
    style.finish(axes, title="Moving-average reward (window %d)" % window,
                 xlabel="Episode", ylabel="Mean reward", legend=True)
    return fig


# ----------------------------------------------------------------------
# 3, 4, 5: length, success, epsilon
# ----------------------------------------------------------------------

def episode_length(results, window=WINDOW_DEFAULT):
    """How many steps each episode took.

    Short episodes early on are usually failures, not skill: falling into the
    shaft ends an episode in a handful of steps.
    """
    if not results:
        return _empty("Train the agent to see episode lengths")

    fig, axes = style.figure()
    colours = [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        smoothed = sarsa_agent.moving_average(result["history"]["length"], window)
        axes.plot(_episodes(smoothed), smoothed,
                  color=colours[index % len(colours)], label=result["algorithm"])

    for name, steps in map_data.ROUTE_STEPS_WITH_KEYCARD.items():
        axes.axhline(steps, color=COLORS["grid_line"], linestyle="--",
                     linewidth=1.0)
        axes.annotate("%s (%d)" % (name, steps), xy=(1, steps),
                      xytext=(4, 3), textcoords="offset points",
                      fontsize=7, color=COLORS["text_dim"])
    style.finish(axes, title="Episode length", xlabel="Episode",
                 ylabel="Steps (moving average)", legend=True)
    return fig


def success_rate(results, window=WINDOW_DEFAULT):
    """The share of recent episodes that reached the exit."""
    if not results:
        return _empty("Train the agent to see the success rate")

    fig, axes = style.figure()
    colours = [style.SUCCESS, style.COMPARISON]
    for index, result in enumerate(results):
        smoothed = sarsa_agent.moving_average(result["history"]["success"], window)
        axes.plot(_episodes(smoothed), [value * 100 for value in smoothed],
                  color=colours[index % len(colours)], label=result["algorithm"])
    axes.set_ylim(-3, 103)
    style.finish(axes, title="Success rate (running average)", xlabel="Episode",
                 ylabel="Episodes reaching the exit (%)", legend=True)
    return fig


def epsilon_decay(results):
    """How much exploration was left at each episode.

    Worth reading next to the reward curve: SARSA and Q-Learning only differ while
    this is above zero, because with no exploration the two updates coincide.
    """
    if not results:
        return _empty("Train the agent to see the exploration schedule")

    fig, axes = style.figure()
    colours = [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        epsilons = result["history"]["epsilon"]
        axes.plot(_episodes(epsilons), epsilons,
                  color=colours[index % len(colours)], label=result["algorithm"])
    style.finish(axes, title="Epsilon decay", xlabel="Episode",
                 ylabel="Epsilon", legend=True)
    return fig


# ----------------------------------------------------------------------
# 6: convergence
# ----------------------------------------------------------------------

def q_value_convergence(results):
    """The average size of the Q-table entries.

    While the values are still moving this climbs; once learning settles it goes
    flat, which is the signal that training has converged.
    """
    if not results:
        return _empty("Train the agent to see Q-value convergence")

    fig, axes = style.figure()
    colours = [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        values = result["history"]["mean_abs_q"]
        axes.plot(_episodes(values), values,
                  color=colours[index % len(colours)], label=result["algorithm"])
    style.finish(axes, title="Q-value convergence", xlabel="Episode",
                 ylabel="Mean |Q|", legend=True)
    return fig


# ----------------------------------------------------------------------
# 7, 8, 9: what the robot actually did
# ----------------------------------------------------------------------

def pit_falls(results, window=WINDOW_DEFAULT):
    """How often episodes ended in the shaft."""
    if not results:
        return _empty("Train the agent to see the pit falls")

    fig, axes = style.figure()
    colours = [style.FAILURE, style.COMPARISON]
    for index, result in enumerate(results):
        smoothed = sarsa_agent.moving_average(result["history"]["pit_fall"], window)
        axes.plot(_episodes(smoothed), [value * 100 for value in smoothed],
                  color=colours[index % len(colours)], label=result["algorithm"])
    axes.set_ylim(-3, 103)
    style.finish(axes, title="Pit falls", xlabel="Episode",
                 ylabel="Episodes ending in the shaft (%)", legend=True)
    return fig


def bridge_usage(results, window=WINDOW_DEFAULT):
    """How often the risky bridge was chosen over the safe walkway.

    This is the graph that shows the on-policy/off-policy difference, when the
    state representation is complete enough for it to appear.
    """
    if not results:
        return _empty("Train the agent to see the bridge usage")

    fig, axes = style.figure()
    colours = [style.WARNING, style.COMPARISON]
    for index, result in enumerate(results):
        smoothed = sarsa_agent.moving_average(result["history"]["used_span"], window)
        axes.plot(_episodes(smoothed), [value * 100 for value in smoothed],
                  color=colours[index % len(colours)], label=result["algorithm"])
    axes.set_ylim(-3, 103)
    style.finish(axes, title="Bridge usage — the risky short route",
                 xlabel="Episode", ylabel="Episodes crossing the bridge (%)",
                 legend=True)
    return fig


def keycard_rate(results, window=WINDOW_DEFAULT):
    """How often the keycard was collected.

    It should reach 100% and stay there: without the keycard the exit door never
    opens, so no episode can succeed without it.
    """
    if not results:
        return _empty("Train the agent to see the keycard rate")

    fig, axes = style.figure()
    colours = [style.WARNING, style.COMPARISON]
    for index, result in enumerate(results):
        smoothed = sarsa_agent.moving_average(result["history"]["keycard"], window)
        axes.plot(_episodes(smoothed), [value * 100 for value in smoothed],
                  color=colours[index % len(colours)], label=result["algorithm"])
    axes.set_ylim(-3, 103)
    style.finish(axes, title="Keycard collection rate", xlabel="Episode",
                 ylabel="Episodes collecting the keycard (%)", legend=True)
    return fig


# ----------------------------------------------------------------------
# 10: training time
# ----------------------------------------------------------------------

def training_time(results):
    """Wall-clock time as training progressed."""
    if not results:
        return _empty("Train the agent to see the training time")

    fig, axes = style.figure()
    colours = [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        elapsed = result["history"]["elapsed"]
        axes.plot(_episodes(elapsed), elapsed,
                  color=colours[index % len(colours)],
                  label="%s (%.1fs total)" % (result["algorithm"],
                                              result["runtime_seconds"]))
    style.finish(axes, title="Training time", xlabel="Episode",
                 ylabel="Elapsed seconds", legend=True)
    return fig


# ----------------------------------------------------------------------
# The learned policy, as a picture
# ----------------------------------------------------------------------

TILE_COLOURS = {
    map_data.WALL: COLORS["metal"],
    map_data.FLOOR: COLORS["background_soft"],
    map_data.BRIDGE: "#4A5468",
    map_data.COLLAPSING: "#8A6A2A",
    map_data.PIT: "#04070C",
    map_data.START: COLORS["cyan_dim"],
    map_data.EXIT: COLORS["success"],
    map_data.KEYCARD: COLORS["warning"],
    map_data.PLATFORM: "#6A758C",
}


def sector_map(highlight_path=None):
    """A top-down picture of the sector, used by the analysis view."""
    style.apply()
    fig, axes = style.figure(width=5.6, height=5.2)

    for row in range(map_data.GRID_ROWS):
        for col in range(map_data.GRID_COLS):
            tile = map_data.tile_at(row, col)
            axes.add_patch(_cell(col, row, TILE_COLOURS.get(tile, "#182B45")))
            label = {map_data.START: "S", map_data.EXIT: "E",
                     map_data.KEYCARD: "K", map_data.COLLAPSING: "C",
                     map_data.PLATFORM: "T"}.get(tile, "")
            if label:
                axes.text(col + 0.5, row + 0.5, label, ha="center", va="center",
                          fontsize=8, color=COLORS["text"], weight="bold")

    if highlight_path:
        xs = [col + 0.5 for _, col in highlight_path]
        ys = [row + 0.5 for row, _ in highlight_path]
        axes.plot(xs, ys, color=COLORS["cyan"], linewidth=1.6, alpha=0.9,
                  marker="o", markersize=3)

    _finish_grid(axes)
    style.finish(axes, title="Room 2 — top-down analysis view")
    return fig


def policy_arrows(policy, has_keycard=False, collapsed_mask=None):
    """The preferred action in every cell of the sector.

    The state has three or four parts, so a flat picture has to fix the keycard
    flag — and, when the bridge state is being tracked, the collapsed mask too.
    """
    if not policy:
        return _empty("Train the agent to see the policy")

    style.apply()
    fig, axes = style.figure(width=5.6, height=5.2)

    for row in range(map_data.GRID_ROWS):
        for col in range(map_data.GRID_COLS):
            tile = map_data.tile_at(row, col)
            axes.add_patch(_cell(col, row, TILE_COLOURS.get(tile, "#182B45")))

            if tile in (map_data.WALL, map_data.PIT):
                continue
            if tile == map_data.EXIT:
                axes.text(col + 0.5, row + 0.5, "EXIT", ha="center", va="center",
                          fontsize=6, color=COLORS["background"], weight="bold")
                continue

            key = ((row, col, has_keycard) if collapsed_mask is None
                   else (row, col, has_keycard, collapsed_mask))
            action = policy.get(key)
            if action is None:
                continue
            axes.text(col + 0.5, row + 0.5, actions.direction_arrow(action),
                      ha="center", va="center", fontsize=13, color=COLORS["text"])

    _finish_grid(axes)
    slice_label = "keycard=%s" % ("yes" if has_keycard else "no")
    if collapsed_mask is not None:
        slice_label += ", collapsed mask=%d" % collapsed_mask
    style.finish(axes, title="Preferred action   %s" % slice_label)
    return fig


def _cell(col, row, colour):
    from matplotlib.patches import Rectangle
    return Rectangle((col, row), 1, 1, facecolor=colour,
                     edgecolor=COLORS["grid_line"], linewidth=0.6)


def _finish_grid(axes):
    """Row 0 at the top, matching the way the map is written down."""
    axes.set_xlim(0, map_data.GRID_COLS)
    axes.set_ylim(map_data.GRID_ROWS, 0)
    axes.set_aspect("equal")
    axes.set_xticks([value + 0.5 for value in range(map_data.GRID_COLS)])
    axes.set_xticklabels(range(map_data.GRID_COLS))
    axes.set_yticks([value + 0.5 for value in range(map_data.GRID_ROWS)])
    axes.set_yticklabels(range(map_data.GRID_ROWS))
    axes.grid(False)
    axes.tick_params(length=0)


# ----------------------------------------------------------------------
# Experiment charts
# ----------------------------------------------------------------------

def experiment_chart(rows, metric="mean_reward", error="std_reward",
                     title=None, ylabel=None):
    """One parameter sweep as a bar chart with mean +- standard deviation."""
    if not rows:
        return _empty("Run an experiment to see the results")

    fig, axes = style.figure()
    labels = [str(row["value"]) for row in rows]
    heights = [row[metric] for row in rows]
    errors = [row.get(error, 0.0) for row in rows] if error else None

    axes.bar(labels, heights, color=style.PRIMARY, alpha=0.9, width=0.6,
             yerr=errors, capsize=4,
             error_kw={"ecolor": COLORS["warning"], "elinewidth": 1.2})
    axes.axhline(0, color=COLORS["grid_line"], linewidth=1)
    if len(labels) > 3:
        axes.tick_params(axis="x", labelrotation=20)
    style.finish(axes, title=title or "Parameter sweep",
                 xlabel=rows[0].get("parameter", "value"),
                 ylabel=ylabel or metric.replace("_", " "))
    return fig


def algorithm_comparison_chart(rows):
    """SARSA against Q-Learning on the measures that matter here."""
    if not rows:
        return _empty("Run the comparison to see the results")

    fig, axes = style.figure(width=6.6, height=3.4)
    labels = [row.get("value", row["algorithm"]) for row in rows]
    positions = range(len(rows))

    measures = [
        ("mean_reward", "Mean reward", style.PRIMARY),
        ("mean_pit_falls", "Pit falls (x100)", style.FAILURE),
        ("mean_span_use", "Bridge use (x100)", style.WARNING),
    ]
    width = 0.26
    for index, (key, label, colour) in enumerate(measures):
        scale = 100.0 if key != "mean_reward" else 1.0
        offset = (index - 1) * width
        axes.bar([position + offset for position in positions],
                 [row[key] * scale for row in rows], width=width,
                 color=colour, label=label)

    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels, fontsize=7)
    axes.tick_params(axis="x", labelrotation=15)
    style.finish(axes, title="SARSA vs Q-Learning", ylabel="value", legend=True)
    return fig
