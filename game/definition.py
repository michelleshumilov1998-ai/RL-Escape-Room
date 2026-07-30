"""A room, in the shape the room screen's data contract asks for.

`web/assets/room/contract.js` is the authority on that shape; this is the
one place on the Python side that speaks it.  Everything else in `game/`
stays in its own terms — tiles, states, actions — and is translated here.

Two things are worth knowing about the translation.

POSITIONS ARE CENTRES, IN WORLD UNITS
The screen has a single coordinate system for every room, so a grid is not
a special case: cell (row, col) is an entity one unit square whose *centre*
is at (col + 0.5, row + 0.5).  A centre is the only thing that means the
same for a wall, a drone and an agent.

ONLY THE TYPES A ROOM ACTUALLY USES
The legend is generated from `entityTypes`, so handing over every entity in
`config.ENTITIES` would list room 1's lasers in the legend of a room that
has none.  The types are therefore filtered to what the layout contains —
plus the agent, which is never a tile but is always on screen.
"""

from game import config
from game.grid import TILE_KINDS


def _entity_type(definition):
    """One entry of config.ENTITIES, in the contract's spelling."""
    described = {
        "label": definition["label"],
        # The name of a CSS custom property. The colour itself stays in the
        # stylesheet; this side never writes a hex value down.
        "color": definition["colour"],
        "shape": definition["shape"],
        "inLegend": definition["in_legend"],
        "role": definition["role"],
    }
    # How this type looks when a step reports a state for one of its entities
    # — a collapsed plank, an opened door. Only the types that can change.
    if definition.get("states"):
        described["states"] = {
            name: {"shape": look["shape"], "color": look["colour"]}
            for name, look in definition["states"].items()
        }
    return described


def _parameter(name):
    specification = config.PARAMETERS[name]
    parameter = {
        "key": name,
        "label": specification["label"],
        "symbol": specification["symbol"],
        "type": "number",
        "min": specification["minimum"],
        "max": specification["maximum"],
        "step": specification["step"],
        "default": specification["default"],
        "description": specification["explanation"],
        # The contract asks the question the other way round from the config:
        # "does this apply live" rather than "what is its scope".
        "appliesLive": specification["scope"] == "live",
    }
    if specification.get("choices"):
        parameter["choices"] = list(specification["choices"])
    return parameter


def _info(room):
    """RoomInfo. Two fields are named differently from the room data."""
    info = room["info"]
    described = {
        "objective": info["objective"],
        "obstacles": list(info["obstacles"]),
        "actionSet": info["actions"],
        "rewards": [list(pair) for pair in info["rewards"]],
        "terminal": info["termination"],
    }
    if info.get("note"):
        described["note"] = info["note"]
    # Only for the rooms that are *about* a choice between routes. The screen
    # states it as a table rather than leaving it to be inferred from the map.
    if info.get("routes"):
        described["routes"] = [dict(route) for route in info["routes"]]
    return described


def build(room, env, parameters):
    """The static half of a room: layout, words, parameter schema.

    `parameters` supplies the defaults this run actually started with, which
    are not always the global ones — a room may override any of them.
    """
    rows, cols = env.rows, env.cols
    cell = 1.0

    kinds_used = set()
    entities = []
    for row in range(rows):
        for col in range(cols):
            kind = TILE_KINDS[env.tile_at(row, col)]
            kinds_used.add(kind)
            # Floor is the surface everything else sits on and the renderer
            # has already laid it down; drawing it again would rub out the
            # value wash on top of it.
            if kind == "floor":
                continue
            entities.append({
                "id": "r%dc%d" % (row, col),
                "type": kind,
                "position": {"x": col + 0.5, "y": row + 0.5},
                "size": {"width": cell, "height": cell},
            })

    # Things that are not tiles at all. A patrolling guard has no square of
    # the map to belong to: it is declared once, wherever its patrol starts,
    # and given a new position by every step of a recorded episode.
    for extra in env.loose_entities():
        kinds_used.add(extra["type"])
        entities.append(extra)

    # The agent is never a tile, and is always on screen.
    kinds_used.add("agent")
    entity_types = {kind: _entity_type(config.ENTITIES[kind])
                    for kind in sorted(kinds_used)}

    return {
        "id": room["key"],
        "name": room["name"],
        "sector": room["sector"],
        "worldSize": {"width": cols * cell, "height": rows * cell},
        "isGrid": True,
        "cellSize": cell,
        "entities": entities,
        "entityTypes": entity_types,
        "info": _info(room),
        "parameterSchema": [
            dict(_parameter(name), default=parameters.get(
                name, config.PARAMETERS[name]["default"]))
            for name in room["parameters"]
        ],
        # The two screens read their one metric from different places, so the
        # key cannot simply be passed through. The planner screen's strip
        # takes a *session* measurement straight off the algorithm snapshot
        # ("meanReward"); this screen averages a field of EpisodeMetrics over
        # the recent past, so the key has to name one of those. The label is
        # the room's own either way, and stays true because the averaging is
        # what makes it a mean.
        "metric": {
            "key": room.get("episode_metric", "reward"),
            "label": room["metric"]["label"],
            "format": room["metric"]["format"],
        },
    }
