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

from game import config, drone
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


def _schema(room, parameters):
    """The parameter controls, with this run's own defaults filled in."""
    return [
        dict(_parameter(name), default=parameters.get(
            name, config.PARAMETERS[name]["default"]))
        for name in room["parameters"]
    ]


def _metric(room):
    # The two screens read their one metric from different places, so the key
    # cannot simply be passed through. The planner screen's strip takes a
    # *session* measurement straight off the algorithm snapshot ("meanReward");
    # the room screen averages a field of EpisodeMetrics over the recent past,
    # so the key has to name one of those. The label is the room's own either
    # way, and stays true because the averaging is what makes it a mean.
    return {
        "key": room.get("episode_metric", "reward"),
        "label": room["metric"]["label"],
        "format": room["metric"]["format"],
    }


def build_continuous(room, env, parameters):
    """The static half of a room that has no grid under it.

    Room 4's entities are not cells and cannot be read off a layout, so the
    world states them itself and this only translates the surrounding fields.
    `isGrid` false is what tells the screen to draw no cell boundaries and to
    skip the value heatmap and the policy arrows, both of which are overlays
    over squares that do not exist here.
    """
    entities = env.entities()
    kinds_used = {entity["type"] for entity in entities}
    # The agent is never furniture, and is always on screen.
    kinds_used.add("agent")

    described = {
        "id": room["key"],
        "name": room["name"],
        "sector": room["sector"],
        "worldSize": {"width": env.width, "height": env.height},
        "isGrid": False,
        # HOW A CHAMBER WITH NO GRID IS STILL THE SAME BUILDING
        #
        # `isGrid` false says there are no *cells* — no state to lay a value
        # heatmap or a policy arrow over, because the state here is real
        # numbers. It was also, wrongly, taken to mean "no floor tiling and no
        # masonry", which is why rooms 4 and 5 looked like a different game
        # from rooms 1 to 3: a bare wash of floor colour inside a thin stroked
        # frame, against tiled stone rooms with brick walls.
        #
        # Those two things are architecture, not state. `tileSize` is the floor
        # tiling, at one metre — the same pitch as a cell of rooms 1 to 3, so
        # the two floors are continuous. `wallMargin` is how much room the
        # renderer has to leave around the world for the masonry ring the world
        # states in its own entities; see `chamber_wall_entities`.
        "tileSize": 1.0,
        "wallMargin": drone.CHAMBER_WALL,
        "entities": entities,
        "entityTypes": {kind: _entity_type(config.ENTITIES[kind])
                        for kind in sorted(kinds_used)},
        "info": _info(room),
        "parameterSchema": _schema(room, parameters),
        "metric": _metric(room),
        # Presentation hints and nothing else: both change how long you look
        # at a flight, never what happened in it.
        #
        # `stepsPerSecond` paces a *replay*, in recorded frames per second.
        #
        # WHY THIS IS THE ROOM'S NUMBER AND NOT A CONSTANT
        # It was 50 for every continuous room, which is right for room 4 and
        # ten times too fast for room 5 — because the two rooms do not mean the
        # same thing by "one recorded frame". Room 4 records every tick of
        # physics, so a frame is dt = 0.02 s and 50 a second replays a flight at
        # the speed it was flown. Room 5 holds each action for `action_repeat`
        # ticks and records one frame per *decision*, so a frame is 0.2 s and 50
        # a second played it back at ten times real speed. The drone crossed the
        # warehouse in a second and nothing in it could be followed.
        #
        # A room that records coarser frames therefore has to say so, and the
        # honest figure is 1 / (dt * action_repeat) — real time.
        #
        # `liveStepScale` paces *training as it is watched*, and it is the same
        # problem one layer up: the speed tiers count environment steps, and a
        # step here covers a five-hundredth of the chamber where a grid step
        # covers a tenth. See the note in `rooms.py`.
        "playback": {
            "stepsPerSecond": float(room.get("replay_steps_per_second", 50.0)),
            "liveStepScale": float(room.get("live_step_scale", 1)),
        },
        "speedLimit": env.speed_limit,
    }

    # Thresholds a drawing may want to mark, sent only by the rooms that have
    # them. Room 4 has a landing speed; room 5 has a sensor range and no
    # landing rule at all. Reading either unconditionally is what made room 5
    # answer 500 the first time it was opened.
    if hasattr(env, "landing_speed"):
        described["landingSpeed"] = env.landing_speed
    if hasattr(env, "sensor_range"):
        described["sensorRange"] = env.sensor_range
        # The half-angle of the forward cone, in radians. Sent because the
        # drawing has to be the sensor rather than a picture of one: the
        # renderer builds its fan from these two numbers and the environment
        # decides visibility from the same two, so they cannot disagree.
        described["sensorSpread"] = env.sensor_spread
    if room.get("charts"):
        described["charts"] = [dict(chart) for chart in room["charts"]]
    # What to call the objective at each mission stage. Only a room whose goal
    # changes part-way through has these; the screen hides the line without
    # them, so the four single-goal rooms are untouched.
    if room.get("objectives"):
        described["objectives"] = list(room["objectives"])
    # Whether escaping this chamber ends the story. Sent only by the room that
    # is last, so no screen has to know a room number to find out.
    if room.get("final"):
        described["isFinal"] = True
    if getattr(env, "pools", None):
        # Which layouts each pool holds. The page shows the counts and the
        # unseen-room control needs to know there is a test pool at all.
        described["layoutPools"] = {
            name: {"count": len(seeds),
                   "first": seeds[0], "last": seeds[-1]}
            for name, seeds in env.pools.items()
        }
    return described


def build(room, env, parameters):
    """The static half of a room: layout, words, parameter schema.

    `parameters` supplies the defaults this run actually started with, which
    are not always the global ones — a room may override any of them.
    """
    if not getattr(env, "is_grid", True):
        return build_continuous(room, env, parameters)

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
            entity = {
                "id": "r%dc%d" % (row, col),
                "type": kind,
                "position": {"x": col + 0.5, "y": row + 0.5},
                "size": {"width": cell, "height": cell},
            }

            # WHICH SIDES OF A HOLE ARE ACTUALLY ITS EDGES
            #
            # The abyss recipe draws the broken rim of the floor the hole was
            # made in, inset on all four sides — which is right for a single
            # cell and wrong for a block of them. Room 2's shaft is four cells
            # by five, and drawn tile by tile it came out as twenty separate
            # holes in a grid instead of one shaft with one rim.
            #
            # A tile cannot see its neighbours, so it is told: which of its four
            # sides face something that is *not* also a hole. Those are the ones
            # that get a rim; the rest open straight into the next cell and the
            # dark runs through unbroken. Only the pit needs it, so only the pit
            # is given it.
            if kind == "pit":
                def same(dr, dc):
                    r, c = row + dr, col + dc
                    if not (0 <= r < rows and 0 <= c < cols):
                        return False
                    return TILE_KINDS[env.tile_at(r, c)] == "pit"

                entity["appearance"] = {"rims": {
                    "top": not same(-1, 0), "bottom": not same(1, 0),
                    "left": not same(0, -1), "right": not same(0, 1),
                }}

            entities.append(entity)

    # SCENERY, DRAWN UNDER EVERYTHING ELSE.
    #
    # `room["decor"]` is a list of rectangles a room wants dressed. It says
    # nothing about the environment — these cells keep whatever tile they
    # already had, and `GridWorld` has never heard of this key — so it can add
    # nothing to the state space and change nothing about a step.
    #
    # Room 2 uses it for the sealed maintenance void in its middle. The rims
    # are worked out the same way the pit's are, so a block of cells reads as
    # one continuous space with one edge rather than as a grid of holes.
    for patch in room.get("decor", []):
        first_row, last_row = patch["rows"]
        first_col, last_col = patch["cols"]
        for row in range(first_row, last_row + 1):
            for col in range(first_col, last_col + 1):
                if not (0 <= row < rows and 0 <= col < cols):
                    continue
                kinds_used.add(patch["type"])
                entities.append({
                    "id": "decor-r%dc%d" % (row, col),
                    "type": patch["type"],
                    "position": {"x": col + 0.5, "y": row + 0.5},
                    "size": {"width": cell, "height": cell},
                    "appearance": {"rims": {
                        "top": row == first_row,
                        "bottom": row == last_row,
                        "left": col == first_col,
                        "right": col == last_col,
                    }},
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
        "parameterSchema": _schema(room, parameters),
        "metric": _metric(room),
    }
