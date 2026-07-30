"""The graphs for Room 1.

Room 1 is solved by Dynamic Programming, not by episode-based training, so the
graphs are about convergence rather than about learning curves:

  1. the largest value change per sweep (does it converge, and how fast),
  2. the value of the start state per sweep (what the plan is worth),
  3. how many states changed their preferred action per sweep,
  4. the final value table as a heatmap,
  5. comparisons between algorithms and between parameter settings.

Every figure is built through `plots/style.py`, so they all match the interface.
"""

from core import actions, tiles
from plots import style
from rooms.room1 import map_data
from ui.theme import COLORS


def convergence(results, theta=None):
    """Largest value change per sweep, on a logarithmic axis.

    `results` is a list of solver result dictionaries, so Value Iteration and
    Policy Iteration can be drawn on the same axes.
    """
    fig, axes = style.figure()
    colours = [style.PRIMARY, style.COMPARISON]

    plotted_any = False
    for index, result in enumerate(results):
        history = result["delta_history"]
        if not history:
            continue
        # A fully converged sweep can report exactly 0, which a log axis cannot
        # draw.  Only those zeros are lifted, to the smallest real value in the
        # history, so no genuine measurement is ever clamped.
        positive = [value for value in history if value > 0]
        floor = min(positive) if positive else 1e-12
        safe = [value if value > 0 else floor for value in history]
        axes.plot(range(1, len(safe) + 1), safe,
                  color=colours[index % len(colours)],
                  label=result["algorithm"], marker="o", markersize=2.6)
        plotted_any = True

    if not plotted_any:
        return style.empty_figure("Solve the room to see the convergence curve")

    if theta:
        axes.axhline(theta, color=style.WARNING, linestyle="--", linewidth=1.1,
                     label="theta = %g" % theta)
    axes.set_yscale("log")
    style.finish(axes, title="Convergence — largest value change per sweep",
                 xlabel="Sweep", ylabel="max |V_new - V|  (log scale)", legend=True)
    return fig


def start_state_value(results):
    """Value of the start state as the sweeps go by."""
    fig, axes = style.figure()
    colours = [style.PRIMARY, style.COMPARISON]

    plotted_any = False
    for index, result in enumerate(results):
        history = result["start_value_history"]
        if not history:
            continue
        axes.plot(range(1, len(history) + 1), history,
                  color=colours[index % len(colours)], label=result["algorithm"],
                  marker="o", markersize=2.6)
        plotted_any = True

    if not plotted_any:
        return style.empty_figure("Solve the room to see V(start)")

    axes.axhline(0, color=COLORS["grid_line"], linewidth=1)
    style.finish(axes, title="Value of the start state, V(start)",
                 xlabel="Sweep", ylabel="V(start)", legend=True)
    return fig


def policy_changes(results):
    """How many states changed their preferred action in each sweep."""
    fig, axes = style.figure()
    colours = [style.PRIMARY, style.COMPARISON]

    plotted_any = False
    for index, result in enumerate(results):
        history = result["policy_change_history"]
        if not history:
            continue
        sweeps = range(1, len(history) + 1)
        offset = -0.2 if index == 0 else 0.2
        axes.bar([sweep + offset for sweep in sweeps], history, width=0.4,
                 color=colours[index % len(colours)], label=result["algorithm"],
                 alpha=0.9)
        plotted_any = True

    if not plotted_any:
        return style.empty_figure("Solve the room to see the policy changes")

    style.finish(axes, title="States that changed their preferred action",
                 xlabel="Sweep", ylabel="States changed", legend=True)
    return fig


# ----------------------------------------------------------------------
# Value heatmap and policy arrows
# ----------------------------------------------------------------------

def _slice_grid(values, has_battery, previous_direction):
    """Pull one 10x10 slice out of the value table.

    The state has four parts, so a flat picture has to fix the battery flag and
    the previous direction.  Walls come out as None and are drawn blank.
    """
    grid = []
    for row in range(map_data.GRID_ROWS):
        line = []
        for col in range(map_data.GRID_COLS):
            tile = map_data.tile_at(row, col)
            if tiles.is_wall(tile):
                line.append(None)
            else:
                line.append(values.get((row, col, has_battery, previous_direction)))

        grid.append(line)
    return grid


def value_heatmap(values, has_battery=False, previous_direction=actions.NONE,
                  show_numbers=True):
    """The final value table as a heatmap, for one slice of the state."""
    if not values:
        return style.empty_figure("Solve the room to see the value heatmap")

    grid = _slice_grid(values, has_battery, previous_direction)
    numeric = [value for line in grid for value in line if value is not None]
    if not numeric:
        return style.empty_figure("No values available for this slice")

    lowest, highest = min(numeric), max(numeric)

    style.apply()
    fig, axes = style.figure(width=5.6, height=5.2)

    for row in range(map_data.GRID_ROWS):
        for col in range(map_data.GRID_COLS):
            value = grid[row][col]
            if value is None:
                # A wall: drawn as metal, with no number.
                axes.add_patch(_cell(col, row, COLORS["metal"]))
                continue
            share = 0.0 if highest == lowest else (value - lowest) / (highest - lowest)
            axes.add_patch(_cell(col, row, _heat_colour(share)))
            if show_numbers:
                axes.text(col + 0.5, row + 0.5, "%.0f" % value,
                          ha="center", va="center", fontsize=7,
                          color=COLORS["text"] if share > 0.45 else COLORS["text_muted"])

    _finish_grid(axes)
    style.finish(axes, title="V(s)   battery=%s, previous direction=%s"
                 % ("yes" if has_battery else "no",
                    actions.direction_name(previous_direction)))
    return fig


def policy_arrows(policy, values=None, has_battery=False,
                  previous_direction=actions.NONE):
    """The preferred action in every cell, as an arrow."""
    if not policy:
        return style.empty_figure("Solve the room to see the policy")

    style.apply()
    fig, axes = style.figure(width=5.6, height=5.2)

    for row in range(map_data.GRID_ROWS):
        for col in range(map_data.GRID_COLS):
            tile = map_data.tile_at(row, col)
            if tiles.is_wall(tile):
                axes.add_patch(_cell(col, row, COLORS["metal"]))
                continue

            axes.add_patch(_cell(col, row, _tile_colour(tile)))

            if tile == tiles.EXIT:
                axes.text(col + 0.5, row + 0.5, "EXIT", ha="center", va="center",
                          fontsize=6, color=COLORS["background"], weight="bold")
                continue

            action = policy.get((row, col, has_battery, previous_direction))
            if action is None:
                continue
            axes.text(col + 0.5, row + 0.5, actions.direction_arrow(action),
                      ha="center", va="center", fontsize=13,
                      color=COLORS["text"])

    _finish_grid(axes)
    style.finish(axes, title="Preferred action   battery=%s, previous direction=%s"
                 % ("yes" if has_battery else "no",
                    actions.direction_name(previous_direction)))
    return fig


def room_map(highlight_path=None):
    """A top-down picture of the chamber, used by the analysis view."""
    style.apply()
    fig, axes = style.figure(width=5.6, height=5.2)

    for row in range(map_data.GRID_ROWS):
        for col in range(map_data.GRID_COLS):
            tile = map_data.tile_at(row, col)
            axes.add_patch(_cell(col, row, _tile_colour(tile)))
            label = _tile_label(tile)
            if label:
                axes.text(col + 0.5, row + 0.5, label, ha="center", va="center",
                          fontsize=8, color=COLORS["text"], weight="bold")

    if highlight_path:
        xs = [col + 0.5 for _, col in highlight_path]
        ys = [row + 0.5 for row, _ in highlight_path]
        axes.plot(xs, ys, color=COLORS["cyan"], linewidth=1.6, alpha=0.85,
                  marker="o", markersize=3)

    _finish_grid(axes)
    style.finish(axes, title="Room 1 — top-down analysis view")
    return fig


def _heat_colour(share):
    """Blend from deep blue (low value) to cyan (high value)."""
    low = (0.05, 0.10, 0.20)
    high = (0.22, 0.85, 1.00)
    return tuple(low[index] + (high[index] - low[index]) * share for index in range(3))


def _tile_colour(tile):
    """The palette colour of each tile, matching the 3D renderer."""
    mapping = {
        tiles.WALL: COLORS["metal"],
        tiles.FLOOR: COLORS["background_soft"],
        tiles.START: COLORS["cyan_dim"],
        tiles.EXIT: COLORS["success"],
        tiles.LASER: COLORS["danger"],
        tiles.WEAK_ICE: "#7FC9E8",
        tiles.STRONG_ICE: "#CFEBF7",
        tiles.OIL: "#1A1424",
        tiles.BATTERY: COLORS["warning"],
        tiles.TELEPORT: COLORS["purple"],
    }
    if tiles.is_one_way_door(tile):
        return COLORS["metal_light"]
    return mapping.get(tile, COLORS["background_soft"])


def _tile_label(tile):
    """Short label drawn inside special tiles on the analysis map."""
    if tile == tiles.START:
        return "S"
    if tile == tiles.EXIT:
        return "E"
    if tile == tiles.BATTERY:
        return "B"
    if tile == tiles.TELEPORT:
        return "T"
    if tile == tiles.LASER:
        return "L"
    if tiles.is_one_way_door(tile):
        return actions.direction_arrow(tiles.door_direction(tile))
    return ""


def _cell(col, row, colour):
    """One square tile patch."""
    from matplotlib.patches import Rectangle
    return Rectangle((col, row), 1, 1, facecolor=colour,
                     edgecolor=COLORS["grid_line"], linewidth=0.6)


def _finish_grid(axes):
    """Shared axis setup so every grid picture lines up the same way.

    Row 0 is drawn at the top, matching the way the map is written down.
    """
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

def experiment_chart(rows, metric="mean_return", error="std_return",
                     title=None, ylabel=None):
    """One parameter sweep as a bar chart with mean +- standard deviation."""
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
    style.finish(axes, title=title or "Parameter sweep",
                 xlabel=rows[0].get("parameter", "value"),
                 ylabel=ylabel or metric.replace("_", " "))
    return fig


def comparison_chart(comparison):
    """Value Iteration against Policy Iteration on four measures at once."""
    value_result = comparison["value_iteration"]
    policy_result = comparison["policy_iteration"]
    value_batch = comparison["value_iteration_batch"]
    policy_batch = comparison["policy_iteration_batch"]

    fig, axes = style.figure(width=6.4, height=3.4)

    measures = ["Sweeps", "Runtime (ms)", "Mean return", "Mean steps"]
    value_numbers = [value_result["iterations"],
                     value_result["runtime_seconds"] * 1000.0,
                     value_batch["mean_return"], value_batch["mean_steps"]]
    policy_numbers = [policy_result["iterations"],
                      policy_result["runtime_seconds"] * 1000.0,
                      policy_batch["mean_return"], policy_batch["mean_steps"]]

    positions = range(len(measures))
    axes.bar([position - 0.2 for position in positions], value_numbers, width=0.4,
             color=style.PRIMARY, label="Value Iteration")
    axes.bar([position + 0.2 for position in positions], policy_numbers, width=0.4,
             color=style.COMPARISON, label="Policy Iteration")
    axes.set_xticks(list(positions))
    axes.set_xticklabels(measures)
    axes.set_yscale("symlog")
    style.finish(axes, title="Value Iteration vs Policy Iteration",
                 ylabel="value (symmetric log scale)", legend=True)
    return fig
