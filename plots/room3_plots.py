"""The graphs for Room 3.

The eleven the brief asks for:

  1. episode reward              7. guard detection count
  2. moving-average reward       8. electrical hazard count
  3. episode length              9. Q-value magnitude
  4. success rate               10. training time
  5. generator completion rate  11. door unlock frequency
  6. average stage reached

Plus a sector map and a policy view. Everything is built through `plots/style.py`,
so the whole set matches the interface and the other rooms.
"""

from core import actions, tabular
from plots import style
from rooms.room3 import map_data
from ui.theme import COLORS

WINDOW_DEFAULT = 100


def _episodes(values):
    return range(1, len(values) + 1)


def _curves(results, key, window, percent=False, colours=None):
    """The shared shape of most of these graphs: one smoothed line per result."""
    fig, axes = style.figure()
    palette = colours or [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        values = result["history"].get(key) or []
        if not values:
            continue
        smoothed = tabular.moving_average(values, window)
        if percent:
            smoothed = [value * 100 for value in smoothed]
        axes.plot(_episodes(smoothed), smoothed,
                  color=palette[index % len(palette)], label=result["algorithm"])
    return fig, axes


def episode_reward(results, window=WINDOW_DEFAULT):
    """Total reward per episode, with the moving average over it."""
    if not results:
        return style.empty_figure("Train the agent to see the reward curve")

    fig, axes = style.figure()
    palette = [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        rewards = result["history"]["reward"]
        if not rewards:
            continue
        colour = palette[index % len(palette)]
        axes.plot(_episodes(rewards), rewards, color=colour, alpha=0.18,
                  linewidth=0.7)
        smoothed = tabular.moving_average(rewards, window)
        axes.plot(_episodes(smoothed), smoothed, color=colour, linewidth=2.0,
                  label="%s (mean of %d)" % (result["algorithm"], window))
    axes.axhline(0, color=COLORS["grid_line"], linewidth=1)
    style.finish(axes, title="Episode reward", xlabel="Episode",
                 ylabel="Total reward", legend=True)
    return fig


def moving_average_reward(results, window=WINDOW_DEFAULT):
    if not results:
        return style.empty_figure("Train the agent to see the moving average")
    fig, axes = _curves(results, "reward", window)
    style.finish(axes, title="Moving-average reward (window %d)" % window,
                 xlabel="Episode", ylabel="Mean reward", legend=True)
    return fig


def episode_length(results, window=WINDOW_DEFAULT):
    if not results:
        return style.empty_figure("Train the agent to see episode lengths")
    fig, axes = _curves(results, "length", window)
    shortest = map_data.mission_route_length()
    if shortest:
        axes.axhline(shortest, color=style.SUCCESS, linestyle="--", linewidth=1.1,
                     label="shortest legal mission (%d)" % shortest)
    style.finish(axes, title="Episode length", xlabel="Episode",
                 ylabel="Steps (moving average)", legend=True)
    return fig


def success_rate(results, window=WINDOW_DEFAULT):
    if not results:
        return style.empty_figure("Train the agent to see the success rate")
    fig, axes = _curves(results, "success", window, percent=True,
                        colours=[style.SUCCESS, style.COMPARISON])
    axes.set_ylim(-3, 103)
    style.finish(axes, title="Success rate (running average)", xlabel="Episode",
                 ylabel="Episodes reaching the exit (%)", legend=True)
    return fig


def generator_completion_rate(results, window=WINDOW_DEFAULT):
    """How often all three generators were brought up, in order."""
    if not results:
        return style.empty_figure("Train the agent to see the generator rate")
    fig, axes = _curves(results, "generators_done", window, percent=True,
                        colours=[style.WARNING, style.COMPARISON])
    axes.set_ylim(-3, 103)
    style.finish(axes, title="Generator completion rate — all three, in order",
                 xlabel="Episode", ylabel="Episodes completing the sequence (%)",
                 legend=True)
    return fig


def average_stage(results, window=WINDOW_DEFAULT):
    """How far through the sequence the agent typically got."""
    if not results:
        return style.empty_figure("Train the agent to see the stage reached")
    fig, axes = _curves(results, "stage", window,
                        colours=[style.WARNING, style.COMPARISON])
    axes.set_ylim(-0.1, 3.2)
    axes.set_yticks([0, 1, 2, 3])
    axes.set_yticklabels(["none", "A", "A+B", "A+B+C"])
    style.finish(axes, title="Average stage reached", xlabel="Episode",
                 ylabel="Generators running", legend=True)
    return fig


def guard_detections(results, window=WINDOW_DEFAULT):
    if not results:
        return style.empty_figure("Train the agent to see the guard detections")
    fig, axes = _curves(results, "guard_caught", window, percent=True,
                        colours=[style.FAILURE, style.COMPARISON])
    axes.set_ylim(-3, 103)
    style.finish(axes, title="Guard detections", xlabel="Episode",
                 ylabel="Episodes ending in detection (%)", legend=True)
    return fig


def hazard_hits(results, window=WINDOW_DEFAULT):
    if not results:
        return style.empty_figure("Train the agent to see the hazard count")
    fig, axes = _curves(results, "hazards", window,
                        colours=[style.WARNING, style.COMPARISON])
    style.finish(axes, title="Electrical hazards hit per episode",
                 xlabel="Episode", ylabel="Hazards (moving average)", legend=True)
    return fig


def q_value_magnitude(results):
    if not results:
        return style.empty_figure("Train the agent to see the Q-value magnitude")
    fig, axes = style.figure()
    palette = [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        values = result["history"]["mean_abs_q"]
        axes.plot(_episodes(values), values, color=palette[index % len(palette)],
                  label=result["algorithm"])
    style.finish(axes, title="Q-value magnitude", xlabel="Episode",
                 ylabel="Mean |Q|", legend=True)
    return fig


def door_unlock_frequency(results, window=WINDOW_DEFAULT):
    """How often R-5 actually walked through the blast door.

    Different from the generator rate: unlocking the door and then getting caught
    on the way to it counts for the generators but not here.
    """
    if not results:
        return style.empty_figure("Train the agent to see the door usage")
    fig, axes = _curves(results, "door_passed", window, percent=True,
                        colours=[style.SUCCESS, style.COMPARISON])
    axes.set_ylim(-3, 103)
    style.finish(axes, title="Door unlock frequency — blast door actually used",
                 xlabel="Episode", ylabel="Episodes entering the door (%)",
                 legend=True)
    return fig


def training_time(results):
    if not results:
        return style.empty_figure("Train the agent to see the training time")
    fig, axes = style.figure()
    palette = [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        elapsed = result["history"]["elapsed"]
        axes.plot(_episodes(elapsed), elapsed, color=palette[index % len(palette)],
                  label="%s (%.1fs total)" % (result["algorithm"],
                                              result["runtime_seconds"]))
    style.finish(axes, title="Training time", xlabel="Episode",
                 ylabel="Elapsed seconds", legend=True)
    return fig


def epsilon_decay(results):
    if not results:
        return style.empty_figure("Train the agent to see the exploration schedule")
    fig, axes = style.figure()
    palette = [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        epsilons = result["history"]["epsilon"]
        axes.plot(_episodes(epsilons), epsilons,
                  color=palette[index % len(palette)], label=result["algorithm"])
    style.finish(axes, title="Epsilon decay", xlabel="Episode", ylabel="Epsilon",
                 legend=True)
    return fig


# ----------------------------------------------------------------------
# The chamber, and the learned policy
# ----------------------------------------------------------------------

TILE_COLOURS = {
    map_data.WALL: COLORS["metal"],
    map_data.FLOOR: COLORS["background_soft"],
    map_data.START: COLORS["cyan_dim"],
    map_data.EXIT: COLORS["success"],
    map_data.GENERATOR_A: "#2E7FB8",
    map_data.GENERATOR_B: "#2E9FA8",
    map_data.GENERATOR_C: "#2EB88A",
    map_data.REACTOR_DOOR: "#8A6A2A",
    map_data.HAZARD: COLORS["danger"],
    map_data.SLIDING_DOOR: COLORS["metal_light"],
}

TILE_LABELS = {
    map_data.START: "S", map_data.EXIT: "E", map_data.GENERATOR_A: "A",
    map_data.GENERATOR_B: "B", map_data.GENERATOR_C: "C",
    map_data.REACTOR_DOOR: "R", map_data.HAZARD: "X",
    map_data.SLIDING_DOOR: "D",
}


def chamber_map(highlight_path=None, guard_cell=None, show_patrol=True):
    """A top-down picture of the chamber, used by the analysis view."""
    style.apply()
    fig, axes = style.figure(width=5.6, height=5.2)

    for row in range(map_data.GRID_ROWS):
        for col in range(map_data.GRID_COLS):
            tile = map_data.tile_at(row, col)
            axes.add_patch(_cell(col, row, TILE_COLOURS.get(tile, "#182B45")))
            label = TILE_LABELS.get(tile, "")
            if label:
                axes.text(col + 0.5, row + 0.5, label, ha="center", va="center",
                          fontsize=8, color=COLORS["text"], weight="bold")

    if show_patrol:
        xs = [col + 0.5 for _, col in map_data.PATROL] + [map_data.PATROL[0][1] + 0.5]
        ys = [row + 0.5 for row, _ in map_data.PATROL] + [map_data.PATROL[0][0] + 0.5]
        axes.plot(xs, ys, color=COLORS["danger"], linewidth=1.0, alpha=0.45,
                  linestyle="--", label="guard patrol")

    if highlight_path:
        xs = [col + 0.5 for _, col in highlight_path]
        ys = [row + 0.5 for row, _ in highlight_path]
        axes.plot(xs, ys, color=COLORS["cyan"], linewidth=1.6, alpha=0.9,
                  marker="o", markersize=3, label="R-5 path")

    if guard_cell:
        axes.plot([guard_cell[1] + 0.5], [guard_cell[0] + 0.5], marker="s",
                  markersize=9, color=COLORS["danger"], label="guard")

    _finish_grid(axes)
    axes.legend(loc="lower right", fontsize=6, framealpha=0.85)
    style.finish(axes, title="Room 3 — top-down analysis view")
    return fig


def policy_arrows(policy, stage=0, guard_index=0):
    """The preferred action in every cell, for one slice of the state.

    The state has four parts, so a flat picture has to fix the stage and where the
    guard is standing.
    """
    if not policy:
        return style.empty_figure("Train the agent to see the policy")

    style.apply()
    fig, axes = style.figure(width=5.6, height=5.2)
    guard = map_data.guard_cell(guard_index)

    for row in range(map_data.GRID_ROWS):
        for col in range(map_data.GRID_COLS):
            tile = map_data.tile_at(row, col)
            axes.add_patch(_cell(col, row, TILE_COLOURS.get(tile, "#182B45")))
            if tile == map_data.WALL:
                continue
            if (row, col) == guard:
                axes.text(col + 0.5, row + 0.5, "G", ha="center", va="center",
                          fontsize=9, color=COLORS["danger"], weight="bold")
                continue
            action = policy.get((row, col, stage, guard_index))
            if action is None:
                continue
            axes.text(col + 0.5, row + 0.5, actions.direction_arrow(action),
                      ha="center", va="center", fontsize=13, color=COLORS["text"])

    _finish_grid(axes)
    style.finish(axes, title="Preferred action   stage=%d, guard at %s, doors %s"
                 % (stage, guard,
                    "open" if map_data.door_is_open(guard_index) else "shut"))
    return fig


def _cell(col, row, colour):
    from matplotlib.patches import Rectangle
    return Rectangle((col, row), 1, 1, facecolor=colour,
                     edgecolor=COLORS["grid_line"], linewidth=0.6)


def _finish_grid(axes):
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

def experiment_chart(rows, metric="mean_reward", error="std_reward", title=None,
                     ylabel=None):
    if not rows:
        return style.empty_figure("Run an experiment to see the results")

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


def outcome_chart(rows):
    """Success, generator completion and detections side by side."""
    if not rows:
        return style.empty_figure("Run an experiment to see the results")

    fig, axes = style.figure(width=6.6, height=3.4)
    labels = [str(row["value"]) for row in rows]
    positions = range(len(rows))
    measures = [("mean_success", "Success", style.SUCCESS),
                ("mean_generators", "Generators done", style.WARNING),
                ("mean_caught", "Detected", style.FAILURE)]
    width = 0.26
    for index, (key, label, colour) in enumerate(measures):
        offset = (index - 1) * width
        axes.bar([position + offset for position in positions],
                 [row[key] * 100 for row in rows], width=width, color=colour,
                 label=label)
    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels, fontsize=7)
    axes.tick_params(axis="x", labelrotation=15)
    axes.set_ylim(0, 105)
    style.finish(axes, title="Outcomes", ylabel="percent of episodes", legend=True)
    return fig
