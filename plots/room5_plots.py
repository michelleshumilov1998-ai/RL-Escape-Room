"""The graphs for Room 5.

Training curves, and then the ones that matter most here: the generalisation
comparisons across training, validation and unseen test layouts.
"""

import math

from plots import style
from rooms.room5 import layout as L
from rooms.room5 import q_agent
from rooms.room5.environment import ACTION_ARROWS, ACTION_NAMES, RADAR_NAMES
from ui.theme import COLORS

WINDOW_DEFAULT = 100


def _episodes(values):
    return range(1, len(values) + 1)


def _empty(message):
    return style.empty_figure(message)


def _curves(results, key, window, percent=False, colours=None):
    fig, axes = style.figure()
    palette = colours or [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        values = result["history"].get(key) or []
        if not values:
            continue
        smoothed = q_agent.moving_average(values, window)
        if percent:
            smoothed = [value * 100 for value in smoothed]
        axes.plot(_episodes(smoothed), smoothed,
                  color=palette[index % len(palette)], label=result["algorithm"])
    return fig, axes


# ---------------- training curves ----------------

def episode_reward(results, window=WINDOW_DEFAULT):
    if not results:
        return _empty("Train the agent to see the reward curve")
    fig, axes = style.figure()
    palette = [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        rewards = result["history"]["reward"]
        colour = palette[index % len(palette)]
        axes.plot(_episodes(rewards), rewards, color=colour, alpha=0.15,
                  linewidth=0.6)
        smoothed = q_agent.moving_average(rewards, window)
        axes.plot(_episodes(smoothed), smoothed, color=colour, linewidth=2.0,
                  label="%s (mean of %d)" % (result["algorithm"], window))
    axes.axhline(0, color=COLORS["grid_line"], linewidth=1)
    style.finish(axes, title="Episode reward", xlabel="Episode",
                 ylabel="Total reward", legend=True)
    return fig


def moving_average_reward(results, window=WINDOW_DEFAULT):
    if not results:
        return _empty("Train the agent to see the moving average")
    fig, axes = _curves(results, "reward", window)
    style.finish(axes, title="Moving-average reward (window %d)" % window,
                 xlabel="Episode", ylabel="Mean reward", legend=True)
    return fig


def episode_length(results, window=WINDOW_DEFAULT):
    if not results:
        return _empty("Train the agent to see episode lengths")
    fig, axes = _curves(results, "length", window)
    style.finish(axes, title="Episode length", xlabel="Episode",
                 ylabel="Steps (moving average)", legend=True)
    return fig


def training_success(results, window=WINDOW_DEFAULT):
    if not results:
        return _empty("Train the agent to see the success rate")
    fig, axes = _curves(results, "success", window, percent=True,
                        colours=[style.SUCCESS, style.COMPARISON])
    axes.set_ylim(-3, 103)
    style.finish(axes, title="Training success rate", xlabel="Episode",
                 ylabel="Missions completed (%)", legend=True)
    return fig


def validation_success(results):
    """Measured on held-out layouts, periodically, without learning from them."""
    if not results:
        return _empty("Train the agent to see the validation curve")
    fig, axes = style.figure()
    palette = [style.WARNING, style.COMPARISON]
    drew = False
    for index, result in enumerate(results):
        episodes = result["history"].get("validation_episode") or []
        values = result["history"].get("validation_success") or []
        if not episodes:
            continue
        axes.plot(episodes, [value * 100 for value in values],
                  color=palette[index % len(palette)], marker="o", markersize=3.5,
                  label="%s (held-out layouts)" % result["algorithm"])
        drew = True
    if not drew:
        return _empty("No validation checks were recorded for this run")
    axes.set_ylim(-3, 103)
    style.finish(axes, title="Validation success rate — layouts never trained on",
                 xlabel="Episode", ylabel="Missions completed (%)", legend=True)
    return fig


def terminal_rate(results, window=WINDOW_DEFAULT):
    if not results:
        return _empty("Train the agent to see the terminal rate")
    fig, axes = _curves(results, "terminal", window, percent=True,
                        colours=[style.WARNING, style.COMPARISON])
    axes.set_ylim(-3, 103)
    style.finish(axes, title="Terminal activation rate", xlabel="Episode",
                 ylabel="Episodes activating the terminal (%)", legend=True)
    return fig


def robot_collisions(results, window=WINDOW_DEFAULT):
    if not results:
        return _empty("Train the agent to see the collision rate")
    fig, axes = _curves(results, "caught", window, percent=True,
                        colours=[style.FAILURE, style.COMPARISON])
    axes.set_ylim(-3, 103)
    style.finish(axes, title="Maintenance-robot collision rate", xlabel="Episode",
                 ylabel="Episodes ending in a collision (%)", legend=True)
    return fig


def static_collisions(results, window=WINDOW_DEFAULT):
    if not results:
        return _empty("Train the agent to see the racking collisions")
    fig, axes = _curves(results, "collisions", window,
                        colours=[style.WARNING, style.COMPARISON])
    style.finish(axes, title="Static collisions per episode", xlabel="Episode",
                 ylabel="Collisions (moving average)", legend=True)
    return fig


def timeout_rate(results, window=WINDOW_DEFAULT):
    if not results:
        return _empty("Train the agent to see the timeout rate")
    fig, axes = _curves(results, "timeout", window, percent=True,
                        colours=[style.COMPARISON, style.WARNING])
    axes.set_ylim(-3, 103)
    style.finish(axes, title="Timeout rate", xlabel="Episode",
                 ylabel="Episodes running out of time (%)", legend=True)
    return fig


def epsilon_decay(results):
    if not results:
        return _empty("Train the agent to see the exploration schedule")
    fig, axes = style.figure()
    palette = [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        values = result["history"]["epsilon"]
        axes.plot(_episodes(values), values, color=palette[index % len(palette)],
                  label=result["algorithm"])
    style.finish(axes, title="Epsilon decay", xlabel="Episode", ylabel="Epsilon",
                 legend=True)
    return fig


def td_error(results, window=WINDOW_DEFAULT):
    if not results:
        return _empty("Train the agent to see the TD error")
    fig, axes = _curves(results, "td_error", window)
    style.finish(axes, title="Mean absolute TD error", xlabel="Episode",
                 ylabel="mean |delta|", legend=True)
    return fig


def weight_norm(results):
    if not results:
        return _empty("Train the agent to see the weight norm")
    fig, axes = style.figure()
    palette = [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        values = result["history"]["weight_norm"]
        axes.plot(_episodes(values), values, color=palette[index % len(palette)],
                  label="%s (final %.1f)" % (result["algorithm"],
                                             result["weight_norm"]))
    style.finish(axes, title="Weight norm  ||w||", xlabel="Episode",
                 ylabel="||w||", legend=True)
    return fig


def training_time(results):
    if not results:
        return _empty("Train the agent to see the training time")
    fig, axes = style.figure()
    palette = [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        values = result["history"]["elapsed"]
        axes.plot(_episodes(values), values, color=palette[index % len(palette)],
                  label="%s (%.1fs total)" % (result["algorithm"],
                                              result["runtime_seconds"]))
    style.finish(axes, title="Training time", xlabel="Episode",
                 ylabel="Elapsed seconds", legend=True)
    return fig


def feature_activity(results):
    """How much of the weight vector is in use, and how sparse an update is."""
    if not results:
        return _empty("Train the agent to see the feature statistics")
    fig, axes = style.figure()
    palette = [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        values = result["history"]["active_features"]
        total = result["features"]["total_features"]
        axes.plot(_episodes(values), [value / total * 100 for value in values],
                  color=palette[index % len(palette)],
                  label="%s — %d of %d weights used"
                        % (result["algorithm"], result["active_features"], total))
    style.finish(axes, title="Feature activity", xlabel="Episode",
                 ylabel="percent of the weight vector", legend=True)
    return fig


# ---------------- generalisation ----------------

def generalisation_chart(rows):
    """Training against validation against unseen test layouts."""
    if not rows:
        return _empty("Run an experiment to see the generalisation results")
    fig, axes = style.figure(width=6.4, height=3.4)
    labels = [str(row["value"]) for row in rows]
    positions = range(len(rows))
    measures = [("mean_training_success", "Training layouts", style.PRIMARY),
                ("mean_validation_success", "Validation (unseen)", style.WARNING),
                ("mean_test_success", "Test (unseen)", style.SUCCESS)]
    width = 0.26
    for index, (key, label, colour) in enumerate(measures):
        offset = (index - 1) * width
        values = [(row.get(key) or 0.0) * 100 for row in rows]
        errors = [(row.get("std_" + key.replace("mean_", "")) or 0.0) * 100
                  for row in rows]
        axes.bar([position + offset for position in positions], values,
                 width=width, color=colour, label=label, yerr=errors, capsize=3,
                 error_kw={"ecolor": COLORS["text_dim"], "elinewidth": 0.9})
    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels, fontsize=7)
    axes.tick_params(axis="x", labelrotation=15)
    axes.set_ylim(0, 105)
    style.finish(axes, title="Success by layout split", ylabel="missions completed (%)",
                 legend=True)
    return fig


def experiment_chart(rows, metric="mean_test_success", error="std_test_success",
                     title=None, ylabel=None, percent=True):
    if not rows:
        return _empty("Run an experiment to see the results")
    fig, axes = style.figure()
    labels = [str(row["value"]) for row in rows]
    scale = 100.0 if percent else 1.0
    heights = [(row.get(metric) or 0.0) * scale for row in rows]
    errors = [(row.get(error) or 0.0) * scale for row in rows] if error else None
    axes.bar(labels, heights, color=style.PRIMARY, alpha=0.9, width=0.6,
             yerr=errors, capsize=4,
             error_kw={"ecolor": COLORS["warning"], "elinewidth": 1.2})
    if len(labels) > 2:
        axes.tick_params(axis="x", labelrotation=20)
    style.finish(axes, title=title or "Experiment",
                 xlabel=rows[0].get("parameter", "value"),
                 ylabel=ylabel or ("unseen success (%)" if percent else metric))
    return fig


# ---------------- radar and trajectory analysis ----------------

def radar_over_time(run):
    """The eight static rays over one episode."""
    if not run:
        return _empty("Run a mission to see the radar trace")
    fig, axes = style.figure()
    frames = run["frames"]
    steps = [frame["step"] for frame in frames]
    for direction in range(8):
        axes.plot(steps, [frame["radar_observation"][direction]
                          for frame in frames],
                  linewidth=1.0, alpha=0.8, label=RADAR_NAMES[direction])
    style.finish(axes, title="Static radar over the mission", xlabel="Step",
                 ylabel="normalised distance", legend=True)
    return fig


def nearest_robot_distance(run):
    """How close the nearest maintenance robot got, with the danger events on it."""
    if not run:
        return _empty("Run a mission to see the robot distances")
    fig, axes = style.figure()
    frames = run["frames"]
    steps = [frame["step"] for frame in frames]
    axes.plot(steps, [frame["nearest_robot"] for frame in frames],
              color=style.FAILURE, label="nearest robot")

    marks = (("terminal_activated", style.SUCCESS, "terminal"),
             ("collision", style.WARNING, "racking hit"),
             ("conveyor_event", style.COMPARISON, "conveyor"),
             ("caught", style.FAILURE, "collision"))
    for key, colour, label in marks:
        points = [(frame["step"], frame["nearest_robot"]) for frame in frames
                  if frame.get(key)]
        if points:
            axes.scatter([step for step, _ in points],
                         [value for _, value in points], s=26, color=colour,
                         zorder=5, label=label)
    style.finish(axes, title="Nearest maintenance robot, and danger events",
                 xlabel="Step", ylabel="normalised distance", legend=True)
    return fig


def action_distribution(runs):
    if not runs:
        return _empty("Run some missions to see the action distribution")
    counts = {action: 0 for action in ACTION_NAMES}
    for run in runs:
        for frame in run["frames"][1:]:
            if frame["action"] is not None:
                counts[frame["action"]] += 1
    fig, axes = style.figure()
    labels = [ACTION_NAMES[action] for action in sorted(counts)]
    values = [counts[action] for action in sorted(counts)]
    axes.bar(labels, values, color=style.PRIMARY, width=0.6)
    style.finish(axes, title="Action distribution over the evaluation episodes",
                 ylabel="times chosen")
    return fig


TILE_COLOURS = {
    L.SHELF: COLORS["metal"], L.CRATE: "#6A5433", L.FLOOR: COLORS["background_soft"],
    L.START: COLORS["cyan_dim"], L.EXIT: COLORS["success"],
    L.TERMINAL: COLORS["warning"], L.CHARGER: "#2E7FB8",
    L.CONVEYOR_UP: "#3B4658", L.CONVEYOR_DOWN: "#3B4658",
    L.CONVEYOR_LEFT: "#3B4658", L.CONVEYOR_RIGHT: "#3B4658",
}

TILE_LABELS = {L.START: "S", L.EXIT: "E", L.TERMINAL: "T", L.CHARGER: "F",
               L.CRATE: "C", L.CONVEYOR_UP: "^", L.CONVEYOR_DOWN: "v",
               L.CONVEYOR_LEFT: "<", L.CONVEYOR_RIGHT: ">"}


def layout_map(layout, run=None, show_routes=True):
    """The whole warehouse with the trajectory over it — analysis only.

    Labelled as such in the interface: the agent itself never receives this map.
    """
    from matplotlib.patches import Rectangle

    style.apply()
    fig, axes = style.figure(width=5.8, height=5.4)
    size = layout.size

    for row in range(size):
        for col in range(size):
            tile = layout.tile_at(row, col)
            axes.add_patch(Rectangle((col, row), 1, 1,
                                     facecolor=TILE_COLOURS.get(tile, "#182B45"),
                                     edgecolor=COLORS["grid_line"], linewidth=0.5))
            label = TILE_LABELS.get(tile, "")
            if label:
                axes.text(col + 0.5, row + 0.5, label, ha="center", va="center",
                          fontsize=7, color=COLORS["text"], weight="bold")

    if show_routes:
        for robot in layout.robot_templates:
            xs = [cell[1] + 0.5 for cell in robot.route]
            ys = [cell[0] + 0.5 for cell in robot.route]
            axes.plot(xs, ys, color=COLORS["danger"], linewidth=1.1, alpha=0.5,
                      linestyle="--")

    if run:
        frames = run["frames"]
        xs = [frame["col"] + 0.5 for frame in frames]
        ys = [frame["row"] + 0.5 for frame in frames]
        axes.plot(xs, ys, color=COLORS["cyan"], linewidth=1.5, alpha=0.9)
        for frame in frames:
            if frame.get("terminal_activated"):
                axes.plot([frame["col"] + 0.5], [frame["row"] + 0.5], marker="*",
                          markersize=12, color=COLORS["warning"])
            elif frame.get("caught"):
                axes.plot([frame["col"] + 0.5], [frame["row"] + 0.5], marker="X",
                          markersize=10, color=COLORS["danger"])
            elif frame.get("action") == 4:
                axes.plot([frame["col"] + 0.5], [frame["row"] + 0.5], marker="s",
                          markersize=3, color=COLORS["text_dim"])
        axes.plot([xs[0]], [ys[0]], marker="o", markersize=7,
                  color=COLORS["cyan"])

    axes.set_xlim(0, size)
    axes.set_ylim(size, 0)
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    axes.grid(False)
    style.finish(axes, title="Layout %s (%s split) — full analysis view"
                 % (layout.seed, layout.split))
    return fig
