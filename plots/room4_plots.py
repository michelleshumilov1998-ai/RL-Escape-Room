"""The graphs for Room 4.

The twelve the brief asks for, plus the analysis plots a continuous room needs:

   1. episode reward            7. collision count
   2. moving-average reward     8. epsilon decay
   3. episode length            9. weight norm
   4. success rate             10. mean absolute TD error
   5. final distance to goal   11. active tile statistics
   6. landing speed            12. training time

and then: trajectories, velocity over time, wind against thrust, the landing
approach, and a slice through the four-dimensional value function.
"""

import math

from plots import style
from rooms.room4 import chamber, sarsa_agent
from rooms.room4.environment import ACTION_ARROWS, ACTIONS, Room4Env
from ui.theme import COLORS

WINDOW_DEFAULT = 40


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
        smoothed = sarsa_agent.moving_average(values, window)
        if percent:
            smoothed = [value * 100 for value in smoothed]
        axes.plot(_episodes(smoothed), smoothed,
                  color=palette[index % len(palette)], label=result["algorithm"])
    return fig, axes


# ----------------------------------------------------------------------
# 1 to 12
# ----------------------------------------------------------------------

def episode_reward(results, window=WINDOW_DEFAULT):
    if not results:
        return _empty("Train the drone to see the reward curve")
    fig, axes = style.figure()
    palette = [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        rewards = result["history"]["reward"]
        colour = palette[index % len(palette)]
        axes.plot(_episodes(rewards), rewards, color=colour, alpha=0.18,
                  linewidth=0.7)
        smoothed = sarsa_agent.moving_average(rewards, window)
        axes.plot(_episodes(smoothed), smoothed, color=colour, linewidth=2.0,
                  label="%s (mean of %d)" % (result["algorithm"], window))
    axes.axhline(0, color=COLORS["grid_line"], linewidth=1)
    style.finish(axes, title="Episode reward", xlabel="Episode",
                 ylabel="Total reward", legend=True)
    return fig


def moving_average_reward(results, window=WINDOW_DEFAULT):
    if not results:
        return _empty("Train the drone to see the moving average")
    fig, axes = _curves(results, "reward", window)
    style.finish(axes, title="Moving-average reward (window %d)" % window,
                 xlabel="Episode", ylabel="Mean reward", legend=True)
    return fig


def episode_length(results, window=WINDOW_DEFAULT):
    if not results:
        return _empty("Train the drone to see episode lengths")
    fig, axes = _curves(results, "length", window)
    style.finish(axes, title="Episode length", xlabel="Episode",
                 ylabel="Steps (moving average)", legend=True)
    return fig


def success_rate(results, window=WINDOW_DEFAULT):
    """Safe landings only — reaching the pad too fast does not count."""
    if not results:
        return _empty("Train the drone to see the safe-landing rate")
    fig, axes = _curves(results, "success", window, percent=True,
                        colours=[style.SUCCESS, style.COMPARISON])
    axes.set_ylim(-3, 103)
    style.finish(axes, title="Safe-landing rate", xlabel="Episode",
                 ylabel="Episodes landing safely (%)", legend=True)
    return fig


def final_distance(results, window=WINDOW_DEFAULT):
    if not results:
        return _empty("Train the drone to see the final distances")
    fig, axes = _curves(results, "final_distance", window,
                        colours=[style.WARNING, style.COMPARISON])
    axes.axhline(chamber.GOAL_RADIUS, color=style.SUCCESS, linestyle="--",
                 linewidth=1.1, label="landing radius (%.1f m)" % chamber.GOAL_RADIUS)
    style.finish(axes, title="Final distance to the pad", xlabel="Episode",
                 ylabel="Metres (moving average)", legend=True)
    return fig


def landing_speed(results, threshold=0.7, crash_speed=2.4):
    """Speed on entering the pad, banded into safe, hard and crash."""
    if not results:
        return _empty("Train the drone to see the landing speeds")

    fig, axes = style.figure()
    for index, result in enumerate(results):
        speeds = result["history"]["landing_speed"]
        points = [(episode + 1, speed) for episode, speed in enumerate(speeds)
                  if speed == speed]                          # drop NaNs
        if not points:
            continue
        for label, low, high, colour in (
                ("safe", 0.0, threshold, style.SUCCESS),
                ("hard", threshold, crash_speed, style.WARNING),
                ("crash", crash_speed, float("inf"), style.FAILURE)):
            band = [(episode, speed) for episode, speed in points
                    if low <= speed < high]
            if band:
                axes.scatter([episode for episode, _ in band],
                             [speed for _, speed in band], s=7, color=colour,
                             alpha=0.7, label=label if index == 0 else None)
    axes.axhline(threshold, color=style.SUCCESS, linestyle="--", linewidth=1.1,
                 label="safe landing limit")
    axes.axhline(crash_speed, color=style.FAILURE, linestyle=":", linewidth=1.1,
                 label="crash threshold")
    style.finish(axes, title="Speed on entering the landing zone",
                 xlabel="Episode", ylabel="Speed (m/s)", legend=True)
    return fig


def collision_count(results, window=WINDOW_DEFAULT):
    if not results:
        return _empty("Train the drone to see the collision count")
    fig, axes = _curves(results, "collisions", window,
                        colours=[style.FAILURE, style.COMPARISON])
    style.finish(axes, title="Collisions per episode", xlabel="Episode",
                 ylabel="Collisions (moving average)", legend=True)
    return fig


def epsilon_decay(results):
    if not results:
        return _empty("Train the drone to see the exploration schedule")
    fig, axes = style.figure()
    palette = [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        epsilons = result["history"]["epsilon"]
        axes.plot(_episodes(epsilons), epsilons,
                  color=palette[index % len(palette)], label=result["algorithm"])
    style.finish(axes, title="Epsilon decay", xlabel="Episode", ylabel="Epsilon",
                 legend=True)
    return fig


def weight_norm(results):
    """||w|| over training — the size of the learned parameters."""
    if not results:
        return _empty("Train the drone to see the weight norm")
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


def td_error(results, window=WINDOW_DEFAULT):
    if not results:
        return _empty("Train the drone to see the TD error")
    fig, axes = _curves(results, "td_error", window)
    style.finish(axes, title="Mean absolute TD error", xlabel="Episode",
                 ylabel="mean |delta|", legend=True)
    return fig


def active_tiles(results):
    """How much of the weight vector has been touched, and how sparse an update is.

    The flat line is the point: however large the vector grows, one update only
    ever changes `num_tilings` weights.
    """
    if not results:
        return _empty("Train the drone to see the tile statistics")

    fig, axes = style.figure()
    palette = [style.PRIMARY, style.COMPARISON]
    for index, result in enumerate(results):
        values = result["history"]["active_features"]
        total = result["tile_coder"]["total_features"]
        share = [value / total * 100 for value in values]
        axes.plot(_episodes(share), share, color=palette[index % len(palette)],
                  label="%s — %d of %d weights used"
                        % (result["algorithm"], result["active_features"], total))
    first = results[0]["tile_coder"]
    axes.axhline(first["num_tilings"] / first["total_features"] * 100,
                 color=style.WARNING, linestyle="--", linewidth=1.1,
                 label="%d active per update" % first["num_tilings"])
    axes.set_yscale("log")
    style.finish(axes, title="Active tile statistics", xlabel="Episode",
                 ylabel="percent of the weight vector (log)", legend=True)
    return fig


def training_time(results):
    if not results:
        return _empty("Train the drone to see the training time")
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


# ----------------------------------------------------------------------
# Analysis plots
# ----------------------------------------------------------------------

def _draw_chamber(axes, show_wind=False):
    """The hall itself, as a backdrop for the trajectory plots."""
    from matplotlib.patches import Circle as CirclePatch, Rectangle as RectPatch

    axes.add_patch(RectPatch((0, 0), chamber.WIDTH, chamber.HEIGHT,
                             facecolor=COLORS["background_soft"],
                             edgecolor=COLORS["metal"], linewidth=1.2))
    for obstacle in chamber.OBSTACLES:
        if obstacle.kind == "circle":
            axes.add_patch(CirclePatch((obstacle.x, obstacle.y), obstacle.radius,
                                       facecolor=COLORS["metal"],
                                       edgecolor=COLORS["metal_light"],
                                       linewidth=0.9))
        else:
            axes.add_patch(RectPatch((obstacle.x0, obstacle.y0),
                                     obstacle.x1 - obstacle.x0,
                                     obstacle.y1 - obstacle.y0,
                                     facecolor=COLORS["metal"],
                                     edgecolor=COLORS["metal_light"],
                                     linewidth=0.9))

    axes.add_patch(CirclePatch(chamber.GOAL_POSITION, chamber.GOAL_RADIUS,
                               facecolor=COLORS["success"], alpha=0.35,
                               edgecolor=COLORS["success"], linewidth=1.4))
    axes.plot([chamber.START_POSITION[0]], [chamber.START_POSITION[1]],
              marker="o", markersize=6, color=COLORS["cyan"])

    for fan in chamber.FANS:
        axes.arrow(fan.x, fan.y, fan.direction_x * 0.8, fan.direction_y * 0.8,
                   head_width=0.18, color=COLORS["cyan"], alpha=0.75,
                   length_includes_head=True)

    if show_wind:
        step = 0.8
        y = step / 2
        while y < chamber.HEIGHT:
            x = step / 2
            while x < chamber.WIDTH:
                if chamber.blocking_obstacle(x, y) is None:
                    wind_x, wind_y = chamber.wind_at(x, y)
                    magnitude = math.hypot(wind_x, wind_y)
                    if magnitude > 0.03:
                        axes.arrow(x, y, wind_x * 0.32, wind_y * 0.32,
                                   head_width=0.09, color=COLORS["cyan"],
                                   alpha=min(0.7, 0.15 + magnitude * 0.2),
                                   length_includes_head=True)
                x += step
            y += step

    axes.set_xlim(-0.4, chamber.WIDTH + 0.4)
    axes.set_ylim(-0.4, chamber.HEIGHT + 0.4)
    axes.set_aspect("equal")
    axes.grid(False)


def trajectories(runs, title=None, show_wind=False):
    """Flight paths over the hall, green for a safe landing and red for a failure."""
    style.apply()
    fig, axes = style.figure(width=5.8, height=5.4)
    _draw_chamber(axes, show_wind=show_wind)

    for run in runs or []:
        xs = [frame["x"] for frame in run["frames"]]
        ys = [frame["y"] for frame in run["frames"]]
        colour = COLORS["success"] if run["success"] else COLORS["danger"]
        axes.plot(xs, ys, color=colour, linewidth=1.3, alpha=0.85)
        axes.plot([xs[-1]], [ys[-1]], marker="x", markersize=6, color=colour)

    style.finish(axes, title=title or "Flight paths", xlabel="x (m)",
                 ylabel="y (m)")
    return fig


def wind_field():
    """The airflow on its own, which is the room's central mechanic."""
    style.apply()
    fig, axes = style.figure(width=5.8, height=5.4)
    _draw_chamber(axes, show_wind=True)
    style.finish(axes, title="Wind field — the sum of all four fans",
                 xlabel="x (m)", ylabel="y (m)")
    return fig


def velocity_over_time(run):
    if not run:
        return _empty("Fly an episode to see its velocity trace")
    fig, axes = style.figure()
    frames = run["frames"]
    steps = [frame["step"] for frame in frames]
    axes.plot(steps, [frame["vx"] for frame in frames], color=style.PRIMARY,
              label="vx")
    axes.plot(steps, [frame["vy"] for frame in frames], color=style.COMPARISON,
              label="vy")
    axes.plot(steps, [frame["speed"] for frame in frames], color=style.SUCCESS,
              linewidth=2.0, label="speed")
    axes.axhline(0, color=COLORS["grid_line"], linewidth=1)
    style.finish(axes, title="Velocity over the episode", xlabel="Step",
                 ylabel="m/s", legend=True)
    return fig


def wind_and_thrust(run):
    if not run:
        return _empty("Fly an episode to see the forces")
    fig, axes = style.figure()
    frames = run["frames"]
    steps = [frame["step"] for frame in frames]
    axes.plot(steps, [math.hypot(frame["wind_x"], frame["wind_y"])
                      for frame in frames],
              color=style.COMPARISON, label="wind force")
    axes.plot(steps, [math.hypot(frame["thrust_x"], frame["thrust_y"])
                      for frame in frames],
              color=style.WARNING, label="thrust")
    style.finish(axes, title="Wind against thrust", xlabel="Step",
                 ylabel="force magnitude", legend=True)
    return fig


def landing_analysis(run, threshold=0.7):
    """Distance and speed over the last part of the flight."""
    if not run:
        return _empty("Fly an episode to see the landing approach")
    fig, axes = style.figure()
    frames = run["frames"][-40:]
    steps = [frame["step"] for frame in frames]
    axes.plot(steps, [frame["distance_to_goal"] for frame in frames],
              color=style.PRIMARY, label="distance to pad (m)")
    axes.plot(steps, [frame["speed"] for frame in frames], color=style.WARNING,
              label="speed (m/s)")
    axes.axhline(threshold, color=style.SUCCESS, linestyle="--", linewidth=1.1,
                 label="safe landing speed")
    axes.axhline(chamber.GOAL_RADIUS, color=style.SUCCESS, linestyle=":",
                 linewidth=1.1, label="landing radius")
    style.finish(axes, title="Landing approach", xlabel="Step", ylabel="value",
                 legend=True)
    return fig


def value_slice(agent, vx=0.0, vy=0.0, samples=26):
    """A slice through the value function at a fixed velocity.

    The state is four-dimensional so there is no complete picture to show; this
    fixes the velocity and sweeps position, plotting `max_a Q(s,a)` with the
    preferred action drawn on top.
    """
    if agent is None:
        return _empty("Train the drone to see a slice of the value function")

    style.apply()
    fig, axes = style.figure(width=5.8, height=5.4)

    grid = []
    for row in range(samples):
        line = []
        y = chamber.HEIGHT * (row + 0.5) / samples
        for col in range(samples):
            x = chamber.WIDTH * (col + 0.5) / samples
            if chamber.blocking_obstacle(x, y) is not None:
                line.append(float("nan"))
            else:
                line.append(max(agent.q_value((x, y, vx, vy), action)
                                for action in ACTIONS))
        grid.append(line)

    axes.imshow(grid, origin="lower", cmap="viridis",
                extent=(0, chamber.WIDTH, 0, chamber.HEIGHT), aspect="equal")

    # The preferred action, on a coarser grid so the arrows stay readable.
    coarse = 9
    for row in range(coarse):
        y = chamber.HEIGHT * (row + 0.5) / coarse
        for col in range(coarse):
            x = chamber.WIDTH * (col + 0.5) / coarse
            if chamber.blocking_obstacle(x, y) is not None:
                continue
            best = agent.best_action((x, y, vx, vy))
            axes.text(x, y, ACTION_ARROWS[best], ha="center", va="center",
                      fontsize=9, color="#FFFFFF")

    axes.plot([chamber.GOAL_POSITION[0]], [chamber.GOAL_POSITION[1]], marker="*",
              markersize=13, color=COLORS["success"])
    style.finish(axes, title="max Q(s,a) at vx=%.1f, vy=%.1f  (one slice of four "
                             "dimensions)" % (vx, vy),
                 xlabel="x (m)", ylabel="y (m)")
    return fig


# ----------------------------------------------------------------------
# Experiment charts
# ----------------------------------------------------------------------

def experiment_chart(rows, metric="mean_reward", error="std_reward", title=None,
                     ylabel=None):
    if not rows:
        return _empty("Run an experiment to see the results")
    fig, axes = style.figure()
    labels = [str(row["value"]) for row in rows]
    heights = [row.get(metric, 0.0) for row in rows]
    errors = [row.get(error, 0.0) for row in rows] if error else None
    axes.bar(labels, heights, color=style.PRIMARY, alpha=0.9, width=0.6,
             yerr=errors, capsize=4,
             error_kw={"ecolor": COLORS["warning"], "elinewidth": 1.2})
    axes.axhline(0, color=COLORS["grid_line"], linewidth=1)
    if len(labels) > 2:
        axes.tick_params(axis="x", labelrotation=20)
    style.finish(axes, title=title or "Parameter sweep",
                 xlabel=rows[0].get("parameter", "value"),
                 ylabel=ylabel or metric.replace("_", " "))
    return fig


def generalisation_chart(rows):
    """Training against validation against unseen test starts."""
    if not rows:
        return _empty("Run the generalisation test to see the results")
    fig, axes = style.figure(width=6.2, height=3.4)
    labels = [row["value"] for row in rows]
    positions = range(len(rows))
    successes = [row["mean_eval_success"] * 100 for row in rows]
    errors = [row["std_eval_success"] * 100 for row in rows]
    colours = [style.PRIMARY if row["seen_during_training"] else style.WARNING
               for row in rows]
    axes.bar(list(positions), successes, color=colours, width=0.55, yerr=errors,
             capsize=4, error_kw={"ecolor": COLORS["text_dim"], "elinewidth": 1.1})
    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels, fontsize=8)
    axes.set_ylim(0, 105)
    for position, row in zip(positions, rows):
        axes.annotate("seen" if row["seen_during_training"] else "unseen",
                      xy=(position, 3), ha="center", fontsize=7,
                      color=COLORS["background"])
    style.finish(axes, title="Safe landings by release point group",
                 ylabel="safe landings (%)")
    return fig
