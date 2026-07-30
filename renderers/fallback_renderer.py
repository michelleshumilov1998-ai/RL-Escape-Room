"""A Matplotlib isometric renderer, used when the canvas component cannot run.

This draws a single still frame in the same isometric style and with the same
palette as the interactive version.  It has no animation, but it never depends on
JavaScript, a canvas or an iframe, so the game always has something to show.

It reads a **scene dictionary** — the contract in `renderers/base_renderer.py` —
rather than any particular room's map, so one renderer serves every room. It is
also what the tests use to prove every tile type can be drawn, since it runs in
plain Python.
"""

from matplotlib.patches import Polygon

from core import actions
from plots import style
from ui.theme import COLORS

# The same proportions the JavaScript renderer uses, in tile units.
TILE_WIDTH = 1.0
TILE_HEIGHT = 0.5
WALL_HEIGHT = 0.24
FLOOR_THICKNESS = 0.10
PIT_DEPTH = 0.26

# Tile colours by kind, matched to renderers/scene_common.PALETTE.
TILE_TOPS = {
    "floor": "#182B45",
    "start": "#1E7F9B",
    "exit": COLORS["success"],
    "exit_door": "#1B7F41",
    "weak_ice": "#7FC9E8",
    "strong_ice": "#BFE6F5",
    "oil": "#171226",
    "laser": "#2A1622",
    "teleport": "#3C2A6B",
    "battery": "#8A6A2A",
    "door": "#55627A",
    "wall": COLORS["metal"],
    # Room 2
    "bridge": "#4A5468",
    "collapsing_bridge": "#8A6A2A",
    "platform": "#6A758C",
    "keycard": "#8A6A2A",
    "pit": "#04070C",
    # Room 3
    "generator_a": "#2E7FB8",
    "generator_b": "#2E9FA8",
    "generator_c": "#2EB88A",
    "reactor_door": "#8A6A2A",
    "hazard": "#3A1420",
    "sliding_door": "#55627A",
    # Room 4
    "obstacle": COLORS["metal"],
    # Room 5
    "shelf": COLORS["metal"],
    "crate": "#6A5433",
    "terminal": "#8A6A2A",
    "conveyor": "#2F3846",
    "charger": "#1E5F7B",
}

# Every tile kind this renderer knows how to draw. The tests check that a scene
# never contains a kind that is missing from here.
KNOWN_KINDS = tuple(TILE_TOPS.keys())


def _project(row, col, height=0.0):
    """Cell coordinates to isometric plot coordinates."""
    x = (col - row) * TILE_WIDTH / 2.0
    y = -(col + row) * TILE_HEIGHT / 2.0 + height
    return x, y


def _quad(row, col, height=0.0, inset=0.0):
    pad = inset
    return [
        _project(row + pad, col + pad, height),
        _project(row + pad, col + 1 - pad, height),
        _project(row + 1 - pad, col + 1 - pad, height),
        _project(row + 1 - pad, col + pad, height),
    ]


def _shade(hex_colour, amount):
    """Darken (amount < 0) or lighten (amount > 0) a colour."""
    hex_colour = hex_colour.lstrip("#")
    channels = [int(hex_colour[index:index + 2], 16) for index in (0, 2, 4)]
    if amount < 0:
        channels = [int(channel * (1 + amount)) for channel in channels]
    else:
        channels = [int(channel + (255 - channel) * amount) for channel in channels]
    return "#%02X%02X%02X" % tuple(max(0, min(255, value)) for value in channels)


def _add(axes, points, colour, edge=None, width=0.6, zorder=1, alpha=1.0):
    axes.add_patch(Polygon(points, closed=True, facecolor=colour, edgecolor=edge,
                           linewidth=width, zorder=zorder, alpha=alpha))


def render_frame(scene, frame=None, title=None):
    """Draw one still isometric picture of a scene.

    `frame` is a replay frame; when it is None the scene's first frame is used.
    Returns a Matplotlib figure.
    """
    grid = scene["grid"]
    rows, cols = grid["rows"], grid["cols"]
    tiles_grid = grid["tiles"]
    if frame is None:
        frame = scene["frames"][0]

    item_taken = bool(frame.get("has_item") or frame.get("has_battery")
                      or frame.get("has_keycard"))
    escaped = bool(frame.get("reached_exit"))
    collapsed = {tuple(cell) for cell in frame.get("collapsed", [])}

    style.apply()
    fig, axes = style.figure(width=7.0, height=5.4)

    # Back to front, so nearer cells cover farther ones.
    cells = sorted(((row, col) for row in range(rows) for col in range(cols)),
                   key=lambda cell: cell[0] + cell[1])

    robot_depth = frame["row"] + frame["col"]
    robot_drawn = False

    for row, col in cells:
        depth = row + col
        if not robot_drawn and depth > robot_depth:
            _draw_robot(axes, scene, frame, zorder=depth * 10 + 5)
            robot_drawn = True

        kind = tiles_grid[row][col]
        base_z = depth * 10

        if kind == "wall":
            _draw_wall(axes, row, col, base_z)
            continue
        if kind == "pit" or (kind == "collapsing_bridge" and (row, col) in collapsed):
            _draw_pit(axes, row, col, base_z)
            continue

        _draw_floor(axes, row, col, kind, base_z)

        if kind == "laser":
            _draw_laser(axes, row, col, base_z)
        elif kind == "teleport":
            _draw_teleport(axes, row, col, base_z)
        elif kind in ("battery", "keycard") and not item_taken:
            _draw_collectable(axes, row, col, base_z)
        elif kind in ("exit", "exit_door"):
            _draw_exit(axes, row, col, base_z, unlocked=item_taken or escaped,
                       opened=escaped)
        elif kind in ("bridge", "collapsing_bridge", "platform"):
            _draw_girders(axes, row, col, base_z, kind)
        elif kind in ("generator_a", "generator_b", "generator_c"):
            _draw_generator(axes, row, col, base_z, kind[-1].upper(),
                            frame.get("stage", 0))
        elif kind == "hazard":
            _draw_hazard(axes, row, col, base_z)
        elif kind == "sliding_door":
            _draw_sliding_door(axes, row, col, base_z,
                               bool(frame.get("doors_open", False)))
        elif kind == "shelf":
            _draw_wall(axes, row, col, base_z)
        elif kind == "crate":
            _add(axes, _quad(row, col, WALL_HEIGHT * 0.5, inset=0.1), "#6A5433",
                 edge=COLORS["warning"], width=0.8, zorder=base_z + 2)
        elif kind == "terminal":
            _draw_exit(axes, row, col, base_z,
                       unlocked=frame.get("stage", 0) >= 1,
                       opened=frame.get("stage", 0) >= 1)
        elif kind == "charger":
            _add(axes, _quad(row, col, 0.02, inset=0.16), COLORS["cyan"],
                 zorder=base_z + 2, alpha=0.4)
        elif kind == "conveyor":
            _add(axes, _quad(row, col, 0.02, inset=0.2), COLORS["warning"],
                 zorder=base_z + 2, alpha=0.35)
        elif kind == "reactor_door":
            _draw_exit(axes, row, col, base_z,
                       unlocked=frame.get("stage", 0) >= 3,
                       opened=frame.get("stage", 0) >= 3)
        elif kind == "door":
            direction = scene["grid"]["doors"].get("%d,%d" % (row, col))
            if direction:
                _draw_one_way_door(axes, row, col, base_z, direction)

    if not robot_drawn:
        _draw_robot(axes, scene, frame, zorder=robot_depth * 10 + 5)

    if scene["meta"].get("hasGuard") and frame.get("guard_row") is not None:
        _draw_guard(axes, frame["guard_row"], frame["guard_col"],
                    zorder=(frame["guard_row"] + frame["guard_col"]) * 10 + 6)

    axes.set_aspect("equal")
    axes.axis("off")
    axes.set_xlim(-(rows * TILE_WIDTH / 2) - 0.6, (cols * TILE_WIDTH / 2) + 0.6)
    axes.set_ylim(-((rows + cols) * TILE_HEIGHT / 2) - 0.8, 1.0)
    axes.set_title(title or ("%s — isometric fallback view"
                             % scene["meta"]["title"].title()),
                   loc="left", pad=8)
    return fig


def _draw_floor(axes, row, col, kind, base_z):
    top_colour = TILE_TOPS.get(kind, TILE_TOPS["floor"])
    top = _quad(row, col, 0.0)
    lower = [(x, y - FLOOR_THICKNESS) for x, y in top]
    _add(axes, [top[3], top[2], lower[2], lower[3]],
         _shade(top_colour, -0.55), zorder=base_z)
    _add(axes, [top[2], top[1], lower[1], lower[2]],
         _shade(top_colour, -0.4), zorder=base_z)
    _add(axes, top, top_colour, edge=COLORS["grid_line"], zorder=base_z + 1)


def _draw_wall(axes, row, col, base_z):
    top = _quad(row, col, WALL_HEIGHT)
    base = _quad(row, col, 0.0)
    _add(axes, [top[3], top[2], base[2], base[3]],
         _shade(COLORS["metal"], -0.34), zorder=base_z)
    _add(axes, [top[2], top[1], base[1], base[2]],
         _shade(COLORS["metal"], -0.14), zorder=base_z)
    _add(axes, top, _shade(COLORS["metal"], 0.16),
         edge=COLORS["metal_light"], zorder=base_z + 1)


def _draw_pit(axes, row, col, base_z):
    """An open shaft, drawn with visible depth rather than as a dark tile."""
    rim = _quad(row, col, 0.0)
    bottom = _quad(row, col, -PIT_DEPTH)
    _add(axes, [rim[0], rim[1], bottom[1], bottom[0]],
         _shade(TILE_TOPS["pit"], 0.16), zorder=base_z)
    _add(axes, [rim[0], rim[3], bottom[3], bottom[0]],
         _shade(TILE_TOPS["pit"], 0.08), zorder=base_z)
    _add(axes, bottom, TILE_TOPS["pit"], zorder=base_z + 1)
    _add(axes, rim, "none", edge=COLORS["warning"], width=0.8, zorder=base_z + 2,
         alpha=0.45)


def _draw_girders(axes, row, col, base_z, kind):
    colour = (COLORS["warning"] if kind == "collapsing_bridge"
              else _shade("#6A758C", 0.1))
    for edge in ((0.08, 0.08, 0.08, 0.92), (0.92, 0.08, 0.92, 0.92)):
        start = _project(row + edge[0], col + edge[1], 0.05)
        end = _project(row + edge[2], col + edge[3], 0.05)
        axes.plot([start[0], end[0]], [start[1], end[1]], color=colour,
                  linewidth=1.1, alpha=0.8, zorder=base_z + 2)


def _draw_laser(axes, row, col, base_z):
    _add(axes, _quad(row, col, 0.01, inset=0.05), COLORS["danger"],
         zorder=base_z + 2, alpha=0.4)
    low = [_project(row + 0.5, col, 0.03), _project(row + 0.5, col + 1, 0.03)]
    axes.plot([low[0][0], low[1][0]],
              [low[0][1] + WALL_HEIGHT * 0.4, low[1][1] + WALL_HEIGHT * 0.4],
              color="#FF8FA3", linewidth=1.8, zorder=base_z + 4)


def _draw_teleport(axes, row, col, base_z):
    _add(axes, _quad(row, col, 0.02, inset=0.10), COLORS["purple"],
         zorder=base_z + 2, alpha=0.7)
    x, y = _project(row + 0.5, col + 0.5, 0.05)
    axes.plot([x, x], [y, y + WALL_HEIGHT * 1.4], color=COLORS["purple"],
              linewidth=4, alpha=0.45, zorder=base_z + 3)


def _draw_collectable(axes, row, col, base_z):
    x, y = _project(row + 0.5, col + 0.5, 0.16)
    axes.plot([x], [y], marker="s", markersize=9, color=COLORS["warning"],
              zorder=base_z + 4)
    axes.plot([x], [y], marker="s", markersize=4,
              color=_shade(COLORS["warning"], 0.45), zorder=base_z + 5)


def _draw_exit(axes, row, col, base_z, unlocked, opened):
    colour = COLORS["success"] if unlocked else COLORS["danger"]
    _add(axes, _quad(row, col, 0.05), _shade(colour, -0.5), edge=colour,
         width=1.1, zorder=base_z + 2)
    panel = [_project(row + 0.88, col + 0.12, 0.05),
             _project(row + 0.88, col + 0.88, 0.05),
             _project(row + 0.88, col + 0.88, WALL_HEIGHT * 0.95),
             _project(row + 0.88, col + 0.12, WALL_HEIGHT * 0.95)]
    _add(axes, panel, colour, edge=colour, width=1.2, zorder=base_z + 3,
         alpha=0.25 if opened else 0.55)


def _draw_one_way_door(axes, row, col, base_z, direction):
    _add(axes, _quad(row, col, 0.02), _shade(COLORS["metal_light"], -0.2),
         edge=COLORS["warning"], width=1.1, zorder=base_z + 2)
    delta_row, delta_col = actions.DELTAS[direction]
    start = _project(row + 0.5 - delta_row * 0.3, col + 0.5 - delta_col * 0.3, 0.09)
    end = _project(row + 0.5 + delta_row * 0.3, col + 0.5 + delta_col * 0.3, 0.09)
    axes.annotate("", xy=end, xytext=start,
                  arrowprops={"arrowstyle": "-|>", "color": COLORS["warning"],
                              "linewidth": 1.6},
                  zorder=base_z + 4)


def _draw_robot(axes, scene, frame, zorder):
    """R-5: the same metallic body and cyan core as the interactive renderer."""
    row, col = frame["row"], frame["col"]
    x, y = _project(row + 0.5, col + 0.5, 0.13)
    floor_x, floor_y = _project(row + 0.5, col + 0.5, 0.0)

    axes.plot([floor_x], [floor_y], marker="o", markersize=15, color="#000000",
              alpha=0.35, zorder=zorder - 1)
    axes.plot([x], [y], marker="o", markersize=15, color=COLORS["metal_light"],
              markeredgecolor="#FFFFFF", markeredgewidth=0.8, zorder=zorder)
    axes.plot([x], [y], marker="o", markersize=6, color=COLORS["cyan"],
              zorder=zorder + 1)

    direction = frame.get("previous_direction") or frame.get("action") or actions.NONE
    if direction and direction != actions.NONE:
        delta_row, delta_col = actions.DELTAS[direction]
        nose_x, nose_y = _project(row + 0.5 + delta_row * 0.32,
                                  col + 0.5 + delta_col * 0.32, 0.13)
        axes.plot([nose_x], [nose_y], marker="o", markersize=3.4, color="#FFFFFF",
                  zorder=zorder + 2)

    axes.text(x, y + 0.20, scene["meta"].get("robotLabel", "R-5"), ha="center",
              va="bottom", fontsize=7, color=COLORS["text"], zorder=zorder + 3)

def _draw_generator(axes, row, col, base_z, letter, stage):
    """A generator, dark until its turn and lit once it is running."""
    order = {"A": 1, "B": 2, "C": 3}
    needed = order.get(letter, 1)
    if stage >= needed:
        colour = COLORS["success"]
    elif stage == needed - 1:
        colour = COLORS["warning"]
    else:
        colour = _shade(COLORS["metal"], 0.15)

    _add(axes, _quad(row, col, WALL_HEIGHT * 0.6, inset=0.16),
         _shade(COLORS["metal"], 0.18), edge=COLORS["metal_light"], width=0.9,
         zorder=base_z + 2)
    x, y = _project(row + 0.5, col + 0.5, WALL_HEIGHT * 0.6)
    axes.plot([x], [y], marker="o", markersize=7, color=colour, zorder=base_z + 3)
    axes.text(x, y + 0.14, letter, ha="center", va="bottom", fontsize=7,
              color=COLORS["text"], weight="bold", zorder=base_z + 4)


def _draw_hazard(axes, row, col, base_z):
    """An electrical arc across the tile."""
    _add(axes, _quad(row, col, 0.02, inset=0.06), COLORS["danger"],
         zorder=base_z + 2, alpha=0.35)
    start = _project(row + 0.22, col + 0.5, 0.05)
    middle = _project(row + 0.5, col + 0.5, 0.13)
    end = _project(row + 0.78, col + 0.5, 0.05)
    axes.plot([start[0], middle[0], end[0]], [start[1], middle[1], end[1]],
              color="#9FD8FF", linewidth=1.4, zorder=base_z + 3)


def _draw_sliding_door(axes, row, col, base_z, open_now):
    """Two leaves that part when the door is open."""
    colour = COLORS["success"] if open_now else COLORS["warning"]
    _add(axes, _quad(row, col, 0.02), _shade(COLORS["metal_light"], -0.35),
         edge=colour, width=1.0, zorder=base_z + 2)
    gap = 0.36 if open_now else 0.02
    for side in (-1, 1):
        near, far = 0.5 + side * gap, 0.5 + side * 0.5
        leaf = [_project(row + min(near, far), col + 0.12, 0.03),
                _project(row + max(near, far), col + 0.12, 0.03),
                _project(row + max(near, far), col + 0.12, WALL_HEIGHT * 0.8),
                _project(row + min(near, far), col + 0.12, WALL_HEIGHT * 0.8)]
        _add(axes, leaf, _shade(COLORS["metal_light"], -0.1), edge=colour,
             width=0.9, zorder=base_z + 3)


def _draw_guard(axes, row, col, zorder):
    """The security robot: the same parts as R-5, squarer and red."""
    x, y = _project(row + 0.5, col + 0.5, 0.11)
    floor_x, floor_y = _project(row + 0.5, col + 0.5, 0.0)
    axes.plot([floor_x], [floor_y], marker="o", markersize=13, color="#000000",
              alpha=0.35, zorder=zorder - 1)
    axes.plot([x], [y], marker="s", markersize=12, color=COLORS["metal_light"],
              markeredgecolor=COLORS["danger"], markeredgewidth=1.0, zorder=zorder)
    axes.plot([x], [y], marker="o", markersize=5, color=COLORS["danger"],
              zorder=zorder + 1)
    axes.text(x, y + 0.19, "SEC-1", ha="center", va="bottom", fontsize=6,
              color=COLORS["danger"], zorder=zorder + 2)
