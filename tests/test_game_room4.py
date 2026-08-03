"""Room 4's simulation core: continuous flight, tile coding, and the contract.

Everything here drives `game/` directly. Nothing imports a renderer, a server
or a browser, which is the property that matters: if the wind tunnel ever needs
drawing in order to advance, these tests stop passing.

The tests are grouped the way the room is built — physics, geometry, rewards,
the approximation, then the contract the screen reads — because a failure in
the first group explains failures in all the later ones.
"""

import math

import pytest

from game import algorithms, config, definition, rooms
from game.algorithms.base import Transition
from game.algorithms.tile_coding import TileCoder
from game.drone import ACTIONS, DOWN, HOLD, LEFT, RIGHT, UP, DroneWorld
from game.session import Session

ROOM = 4


def world(**overrides):
    """A fresh chamber, optionally with the room dict altered."""
    room = rooms.room(ROOM)
    if overrides:
        room = dict(room, **overrides)
    return DroneWorld(room)


@pytest.fixture
def env():
    return world()


# ----------------------------------------------------------------------
# The state, and one step of integration
# ----------------------------------------------------------------------

def test_reset_returns_the_configured_start_state(env):
    room = rooms.room(ROOM)
    assert env.reset() == (room["start"][0], room["start"][1], 0.0, 0.0)
    # And again, so reset is idempotent rather than merely first-time correct.
    env.step(RIGHT)
    assert env.reset() == (room["start"][0], room["start"][1], 0.0, 0.0)


def test_state_is_four_finite_numbers_for_every_action(env):
    for action in ACTIONS:
        env.reset()
        state, reward, done, info = env.step(action)
        assert len(state) == 4
        for value in state:
            assert isinstance(value, float)
            assert math.isfinite(value)
        assert math.isfinite(reward)
        assert isinstance(done, bool)
        assert set(("goal", "hazard", "speed", "slipped")) <= set(info)


def test_velocity_never_leaves_the_configured_range(env):
    """Held inside [-1, 1] on each axis however long the thrust is applied."""
    limit = env.speed_limit
    for action in (UP, DOWN, LEFT, RIGHT):
        env.reset()
        for _ in range(400):
            state, _, done, _ = env.step(action)
            assert -limit - 1e-9 <= state[2] <= limit + 1e-9
            assert -limit - 1e-9 <= state[3] <= limit + 1e-9
            if done:
                break


def test_hold_applies_no_thrust(env):
    env.reset()
    state, _, _, _ = env.step(HOLD)
    # Starting from rest, holding leaves it at rest and in place.
    assert state[2] == pytest.approx(0.0)
    assert state[3] == pytest.approx(0.0)
    assert state[0] == pytest.approx(env.start_position[0])
    assert state[1] == pytest.approx(env.start_position[1])


@pytest.mark.parametrize("action,axis,sign", [
    (RIGHT, 2, +1),
    (LEFT, 2, -1),
    # y grows downwards, so DOWN is the positive direction on that axis.
    (DOWN, 3, +1),
    (UP, 3, -1),
])
def test_each_thrust_acts_on_its_own_axis(env, action, axis, sign):
    env.reset()
    state, _, _, _ = env.step(action)
    other = 3 if axis == 2 else 2
    assert state[axis] * sign > 0
    assert state[other] == pytest.approx(0.0)
    # THE VELOCITY IS DISCRETE. One press moves the component by one whole
    # unit, so from rest a single thrust reaches the speed limit exactly.
    assert abs(state[axis]) == pytest.approx(env.speed_limit)
    assert state[axis] in (-1.0, 0.0, 1.0)


def test_coasting_keeps_the_velocity_because_there_is_no_global_drag(env):
    """A decay applied every tick cannot leave a velocity in {-1, 0, 1}.

    So there is no global drag: a coasting drone keeps its velocity until an
    action or a zone changes it. Only the slow zone bleeds speed, and it does
    so one whole unit at a time -- see the slow-zone test below.
    """
    env.reset()
    # Well away from any zone, so nothing but the action can act on it.
    env.state = (1.5, 8.2, 1.0, 0.0)
    state, _, _, _ = env.step(HOLD)
    assert state[2] == pytest.approx(1.0)


def test_position_integrates_with_the_updated_velocity(env):
    """Semi-implicit Euler: the new velocity carries the position, not the old.

    Worth pinning down, because the two differ by exactly one tick of thrust
    per step and the explicit form would leave the drone a step behind its own
    velocity for the whole flight.
    """
    env.reset()
    before = env.state
    state, _, _, _ = env.step(RIGHT)
    assert state[0] == pytest.approx(before[0] + state[2] * env.dt, rel=1e-9)
    assert state[0] > before[0] + before[2] * env.dt


# ----------------------------------------------------------------------
# The fields, which are functions of position and nothing else
# ----------------------------------------------------------------------

def test_wind_pushes_in_the_declared_direction(env):
    """A whole unit, and only in the direction the zone declares.

    With a discrete velocity the wind cannot add a fraction each tick, so it
    shoves the component a full unit with a probability per tick instead. Over
    many ticks the drone must end up moving the way the zone blows.
    """
    zone = next(z for z in env.zones if z.get("wind"))
    assert zone["wind"][1] > 0, "this room's zone blows downwards"
    pushed = 0
    for seed in range(60):
        env.reset(seed=seed)
        env.state = (zone["x"], zone["y"], 0.0, 0.0)
        for _ in range(120):
            state, _, done, _ = env.step(HOLD)
            if state[3] > 0:
                pushed += 1
                break
            if done:
                break
        # It is never pushed the wrong way.
        assert state[3] >= 0
    assert pushed > 0, "the wind never moved the drone in 60 attempts"


def test_wind_is_absent_outside_its_zone(env):
    env.reset()
    # Well clear of every zone, and coasting.
    env.state = (9.4, 9.4, 0.0, 0.0)
    assert env.zones_at(9.4, 9.4) == []
    state, _, _, _ = env.step(HOLD)
    assert state[3] == pytest.approx(0.0)


def test_wind_is_deterministic_for_the_same_state(env):
    """Same state, same action, same result — every time.

    This is the Markov property this room is built to keep. Wind that varied
    within an episode would be a force the four-number state cannot see.
    """
    zone = next(z for z in env.zones if z.get("wind"))
    start = (zone["x"], zone["y"], 0.2, -0.1)
    results = []
    for _ in range(5):
        env.reset()
        env.state = start
        results.append(env.step(DOWN))
    first = results[0]
    for other in results[1:]:
        assert other[0] == first[0]
        assert other[1] == pytest.approx(first[1])
        assert other[2] == first[2]


def test_wind_strength_scales_how_often_it_shoves():
    """Calm means never; stronger means sooner, on average."""
    def ticks_until_pushed(strength, seeds=40):
        chamber = world(wind=strength)
        zone = next(z for z in chamber.zones if z.get("wind"))
        total = 0
        for seed in range(seeds):
            chamber.reset(seed=seed)
            chamber.state = (zone["x"], zone["y"], 0.0, 0.0)
            waited = 300
            for tick in range(300):
                state, _, done, _ = chamber.step(HOLD)
                if state[3] > 0:
                    waited = tick + 1
                    break
                if done:
                    break
            total += waited
        return total / seeds

    calm = world(wind=0.0)
    zone = next(z for z in calm.zones if z.get("wind"))
    calm.reset()
    calm.state = (zone["x"], zone["y"], 0.0, 0.0)
    for _ in range(200):
        assert calm.step(HOLD)[0][3] == pytest.approx(0.0), (
            "a calm chamber must never shove")

    assert ticks_until_pushed(2.0) < ticks_until_pushed(0.5), (
        "a stronger wind must shove sooner on average")


def test_slow_zone_bleeds_speed_a_whole_unit_at_a_time(env):
    """Drag, discretised: inside the zone a coasting drone comes to rest.

    Outside it nothing slows the drone at all, because a per-tick decay cannot
    leave a velocity in {-1, 0, 1}.
    """
    zone = next(z for z in env.zones if z.get("extra_drag"))

    # Outside: it coasts for ever.
    env.reset(seed=0)
    env.state = (1.5, 8.2, 1.0, 0.0)
    for _ in range(50):
        assert env.step(HOLD)[0][2] == pytest.approx(1.0)

    # Inside: it is pulled to rest, in one whole step, within a second or so.
    stopped = 0
    for seed in range(40):
        env.reset(seed=seed)
        env.state = (zone["x"], zone["y"], 1.0, 0.0)
        for _ in range(150):
            state, _, done, _ = env.step(HOLD)
            assert state[2] in (0.0, 1.0), "it must never overshoot past rest"
            if state[2] == 0.0:
                stopped += 1
                break
            if done:
                break
    assert stopped > 20, "the slow zone stopped the drone only %d times" % stopped


def test_boost_zone_reaches_full_speed_in_one_press(env):
    """What "overcharge" means once a step is already the whole range.

    A thrust normally moves the velocity one unit. From rest that is already
    the limit, so the zone shows itself where it matters: reversing. Outside
    the zone, turning around from full speed takes two presses; inside, one.
    """
    zone = next(z for z in env.zones if z.get("thrust_scale"))
    assert zone["thrust_scale"] > 1.0

    # Outside: one press only cancels the motion.
    env.reset(seed=0)
    env.state = (1.5, 8.2, -1.0, 0.0)
    assert env.step(RIGHT)[0][2] == pytest.approx(0.0)

    # Inside: one press reverses it outright.
    env.reset(seed=0)
    env.state = (zone["x"], zone["y"], -1.0, 0.0)
    assert env.step(RIGHT)[0][2] == pytest.approx(env.speed_limit)


def test_boost_penalty_is_charged_once_per_visit(env):
    """Once on entry, not once per step.

    At a fiftieth of a second a step, a per-step charge would make one crossing
    cost hundreds — the field would be a wall wearing a warning stripe.
    """
    zone = next(z for z in env.zones if z.get("danger"))
    env.reset()
    env.state = (zone["x"], zone["y"], 0.0, 0.0)
    charges = []
    for _ in range(6):
        _, reward, _, _ = env.step(HOLD)
        charges.append(reward)
    # The first step inside pays the danger cost; the rest do not.
    assert charges[0] < env.rewards["danger"] / 2
    for later in charges[1:]:
        assert later > env.rewards["danger"] / 2


def test_leaving_and_returning_does_not_charge_again(env):
    """The charge is per run, which is what `_charged` records.

    Stated as a test because it is a decision rather than an accident: a cost
    that could be paid twice would make the field's price depend on how the
    route happened to wander, and only `reset` clears the record.
    """
    zone = next(z for z in env.zones if z.get("danger"))
    env.reset()
    env.state = (zone["x"], zone["y"], 0.0, 0.0)
    first = env.step(HOLD)[1]
    assert first < env.rewards["danger"] / 2

    env.state = (9.4, 9.4, 0.0, 0.0)          # out
    env.step(HOLD)
    env.state = (zone["x"], zone["y"], 0.0, 0.0)   # and back in
    again = env.step(HOLD)[1]
    assert again > env.rewards["danger"] / 2

    # A fresh run charges it again.
    env.reset()
    env.state = (zone["x"], zone["y"], 0.0, 0.0)
    assert env.step(HOLD)[1] < env.rewards["danger"] / 2


# ----------------------------------------------------------------------
# Collision, and the landing
# ----------------------------------------------------------------------

def test_pillar_collision_matches_the_drawn_geometry(env):
    """The disc tested is the disc drawn.

    `entities()` reports each pillar as the square that bounds its collision
    circle and the recipe fills that box, so a point just inside the declared
    radius must crash and a point just outside must not. Getting this wrong is
    the failure that reads as a bug in the physics: a drone visibly clear of
    the housing, destroyed anyway.
    """
    pillar = env.pillars[0]
    radius = pillar["radius"]
    for angle in (0.0, 1.1, 2.4, 3.7, 5.0):
        inside = (pillar["x"] + math.cos(angle) * radius * 0.92,
                  pillar["y"] + math.sin(angle) * radius * 0.92)
        outside = (pillar["x"] + math.cos(angle) * radius * 1.12,
                   pillar["y"] + math.sin(angle) * radius * 1.12)
        assert env._crashed((inside[0], inside[1], 0.0, 0.0))
        assert not env._crashed((outside[0], outside[1], 0.0, 0.0))

    # And the entity really does describe that circle.
    drawn = next(e for e in env.entities() if e["id"] == "pillar0")
    assert drawn["size"]["width"] == pytest.approx(radius * 2)
    assert drawn["size"]["height"] == pytest.approx(radius * 2)
    assert drawn["position"]["x"] == pytest.approx(pillar["x"])


def test_flying_into_a_pillar_ends_the_run(env):
    pillar = env.pillars[0]
    env.reset()
    # A clear half metre short of the housing, closing on it at the speed
    # limit. Starting exactly one step out would sit on the rounding.
    env.state = (pillar["x"] - pillar["radius"] - 0.5, pillar["y"], 1.0, 0.0)
    for _ in range(60):
        state, reward, done, info = env.step(RIGHT)
        if done:
            break
    assert done and info["hazard"] and not info["goal"]
    assert reward < env.rewards["wall"] / 2


@pytest.mark.parametrize("position", [
    (0.0, 5.0), (10.0, 5.0), (5.0, 0.0), (5.0, 10.0),
])
def test_the_chamber_boundary_is_fatal(env, position):
    assert env._crashed((position[0], position[1], 0.0, 0.0))


def test_leaving_the_chamber_terminates_the_episode(env):
    env.reset()
    env.state = (0.5, 5.0, -1.0, 0.0)
    for _ in range(60):
        state, reward, done, info = env.step(LEFT)
        if done:
            break
    assert done and info["hazard"]
    assert reward < env.rewards["wall"] / 2
    assert state[0] <= 0.0


def test_the_drawn_wall_sits_on_the_boundary_that_kills(env):
    """The masonry's inner face is exactly the boundary the run ends at.

    A wall with thickness *inside* the world would show the drone buried in
    masonry for several steps before the crash was reported. The chamber is
    therefore walled from the outside: the ring of tiles occupies the metre
    beyond each edge, its inner face lands on the boundary, and the drone's
    hull meets it at the instant of the collision.

    This used to be one `shell` entity spanning the whole chamber whose recipe
    drew only a frame. Same requirement, and now the same masonry tiles rooms
    1 to 3 are built from — see `chamber_wall_entities`.
    """
    walls = [e for e in env.entities() if e["type"] == "wall"]
    assert walls, "the chamber has no wall around it"

    for tile in walls:
        left = tile["position"]["x"] - tile["size"]["width"] / 2
        right = tile["position"]["x"] + tile["size"]["width"] / 2
        top = tile["position"]["y"] - tile["size"]["height"] / 2
        bottom = tile["position"]["y"] + tile["size"]["height"] / 2
        # Every tile is wholly outside the flyable world: no masonry is drawn
        # over any square metre the drone may occupy.
        outside = (right <= 1e-9 or left >= env.width - 1e-9
                   or bottom <= 1e-9 or top >= env.height - 1e-9)
        assert outside, "a wall tile is drawn inside the chamber at %r" % (
            tile["position"],)

    # And the ring is continuous: it reaches the full span of both edges.
    faces = {
        "left": max(t["position"]["x"] + t["size"]["width"] / 2
                    for t in walls if t["position"]["x"] < 0),
        "right": min(t["position"]["x"] - t["size"]["width"] / 2
                     for t in walls if t["position"]["x"] > env.width),
        "top": max(t["position"]["y"] + t["size"]["height"] / 2
                   for t in walls if t["position"]["y"] < 0),
        "bottom": min(t["position"]["y"] - t["size"]["height"] / 2
                      for t in walls if t["position"]["y"] > env.height),
    }
    assert faces["left"] == pytest.approx(0.0)
    assert faces["right"] == pytest.approx(env.width)
    assert faces["top"] == pytest.approx(0.0)
    assert faces["bottom"] == pytest.approx(env.height)


def test_the_chamber_is_walled_in_the_same_tiles_as_the_grid_rooms():
    """The whole point of the change: one masonry, one legend entry.

    The wall type rooms 4 and 5 use has to be the very type rooms 1 to 3 use,
    or the two halves of the laboratory are drawn by two recipes that can
    drift apart — which is exactly what happened with `tunnelWall`.
    """
    from game import config
    from game.grid import TILE_KINDS

    # `TILE_KINDS` maps a layout character to an entity type, so the types
    # are its values: '#' is a 'wall'.
    assert "wall" in TILE_KINDS.values(), \
        "the grid rooms' wall type is not called 'wall'"
    assert config.ENTITIES["wall"]["shape"] == "wall"
    # And nothing is left declaring a shell of its own.
    assert "tunnelWall" not in config.ENTITIES

    for room_number in (4, 5):
        session = Session(room_number)
        kinds = {entity["type"] for entity in session.env.entities()}
        assert "wall" in kinds, "room %d states no wall" % room_number
        described = session.describe()["definition"]
        assert described["entityTypes"]["wall"]["shape"] == "wall"
        assert described["entityTypes"]["wall"]["label"] \
            == config.ENTITIES["wall"]["label"]


def test_a_safe_landing_needs_both_components_below_the_threshold(env):
    """Both axes, not the combined speed. Sideways drift is a crash too."""
    pad = env.pad
    limit = env.landing_speed
    gentle = limit * 0.5
    fast = limit * 1.5

    cases = {
        (gentle, gentle): True,
        (fast, gentle): False,
        (gentle, fast): False,
        (fast, fast): False,
        (-gentle, -gentle): True,
        (-fast, gentle): False,
    }
    for (vx, vy), expected in cases.items():
        outcome, _ = env._landing((pad["x"], pad["y"], vx, vy))
        assert (outcome == "landed") is expected, (vx, vy)


def test_arriving_too_fast_is_a_hard_landing_not_a_success(env):
    pad = env.pad
    env.reset()
    # Diagonal at full speed: |v| = sqrt(2), above the 1.0 limit. Multiplying
    # the limit no longer works, because a component can never exceed 1.
    env.state = (pad["x"], pad["y"], 1.0, 1.0)
    state, reward, done, info = env.step(HOLD)
    assert done
    assert not info["goal"]
    assert info["hardLanding"]
    # The hard-landing cost, not the goal reward.
    assert reward < 0
    assert reward > env.rewards["wall"] / 2


def test_a_safe_landing_pays_the_goal_reward(env):
    pad = env.pad
    env.reset()
    env.state = (pad["x"], pad["y"], 0.0, 0.0)
    state, reward, done, info = env.step(HOLD)
    assert done and info["goal"] and info["landed"]
    assert reward > env.rewards["goal"] * 0.9


def test_the_platform_footprint_is_the_area_that_is_tested(env):
    drawn = next(e for e in env.entities() if e["id"] == "pad")
    pad = env.pad
    assert drawn["size"]["width"] == pytest.approx(pad["width"])
    assert drawn["size"]["height"] == pytest.approx(pad["height"])
    # Just inside each edge lands; just outside does not.
    for dx, dy, expected in (
            (pad["width"] / 2 * 0.95, 0.0, True),
            (pad["width"] / 2 * 1.05, 0.0, False),
            (0.0, pad["height"] / 2 * 0.95, True),
            (0.0, pad["height"] / 2 * 1.05, False)):
        outcome, _ = env._landing((pad["x"] + dx, pad["y"] + dy, 0.0, 0.0))
        assert (outcome == "landed") is expected


def test_the_platform_state_comes_from_the_environment(env):
    """The screen may only report a landing the model agreed to.

    `frame_extras` is the sole channel for that, and these are the four words
    it may say. A renderer deciding for itself would be able to announce a
    landing that never happened.
    """
    pad = env.pad
    def look(state):
        states, positions = env.frame_extras(state)
        assert positions is None
        return states[0]["state"]

    assert look((pad["x"], pad["y"], 0.0, 0.0)) == "landed"
    # One axis at full speed is |v| = 1, which is within the 1.0 limit; a
    # diagonal arrival is sqrt(2) and is not.
    assert look((pad["x"], pad["y"], 1.0, 0.0)) == "landed"
    assert look((pad["x"], pad["y"], 1.0, 1.0)) == "crashed"
    # Near and too fast: a warning, but nothing has happened yet. Too fast now
    # means |v| above the limit, and only a diagonal reaches that -- a single
    # axis at full speed is exactly 1.0 and is a legal approach.
    assert look((pad["x"], pad["y"] + 1.2, 1.0, -1.0)) == "fast"
    assert look((pad["x"], pad["y"] + 1.2, 0.0, -1.0)) == "clear"
    assert look((env.start_position[0], env.start_position[1], 0.0, 0.0)) == "clear"


def test_progress_shaping_is_symmetric(env):
    """Closing pays what drifting away costs, so circling earns nothing.

    A shaping term that rewarded approach without charging for retreat would
    make orbiting the platform more profitable than landing on it.
    """
    # Open air, and level with the pad on the y axis, so a step left and a
    # step right are exact mirror images. Off that line the two moves change
    # the straight-line distance by slightly different amounts -- a property
    # of the geometry, not of the shaping -- and the old fixture sat off it,
    # which showed up as a 0.0003 asymmetry once the velocity became exactly
    # one unit per tick.
    clear = (5.6, env.pad["y"], 0.0, 0.0)
    assert not env._crashed(clear)
    assert env.zones_at(clear[0], clear[1]) == []

    env.reset()
    env.state = clear
    towards = env.step(RIGHT)[1]
    env.reset()
    env.state = clear
    away = env.step(LEFT)[1]
    # Both carry the same step cost; the shaping is equal and opposite.
    assert towards > away
    assert (towards + away) == pytest.approx(2 * env.rewards["step"], abs=1e-6)


# ----------------------------------------------------------------------
# The approximation
# ----------------------------------------------------------------------

def test_tile_coder_returns_one_index_per_tiling():
    coder = TileCoder(dimensions=4, tilings=8, tiles_per_dimension=6)
    active = coder.active((0.3, 0.7, 0.5, 0.5))
    assert len(active) == coder.tilings
    assert len(set(active)) == coder.tilings          # no two tilings collide
    for index in active:
        assert 0 <= index < coder.features


def test_tile_coder_clamps_out_of_range_input():
    coder = TileCoder(dimensions=4, tilings=4, tiles_per_dimension=5)
    for scaled in ((-3.0, 0.5, 0.5, 0.5), (9.0, 0.5, 0.5, 0.5)):
        for index in coder.active(scaled):
            assert 0 <= index < coder.features


def test_tile_coder_generalises_between_near_states_and_not_far_ones():
    """The whole reason the room uses one.

    Neighbours share most of their active tiles, so learning about one teaches
    most of what there is to know about the other. Distant states share none.
    """
    coder = TileCoder(dimensions=4, tilings=8, tiles_per_dimension=6)
    here = set(coder.active((0.50, 0.50, 0.50, 0.50)))
    near = set(coder.active((0.51, 0.50, 0.50, 0.50)))
    far = set(coder.active((0.05, 0.95, 0.10, 0.90)))
    assert len(here & near) >= coder.tilings - 2
    assert here & far == set()


def test_semi_gradient_sarsa_update_is_the_documented_arithmetic():
    """One update, checked by hand against the rule in the docstring.

        w[i] += (alpha / tilings) * (r + gamma * Q(s',a') - Q(s,a))

    From zero weights Q is zero everywhere, so a single reward r moves each of
    the `tilings` active weights by exactly alpha*r/tilings — and Q(s,a), being
    the sum of them, moves by alpha*r.
    """
    env = world()
    learner = algorithms.build("semi_gradient_sarsa", env,
                               dict(alpha=0.5, gamma=0.9, epsilon=0.0,
                                    epsilon_min=0.0, epsilon_decay=1.0,
                                    tilings=8, episodes=10))
    state = env.start_state()
    assert learner.q(state, RIGHT) == pytest.approx(0.0)

    learner.update(Transition(state, RIGHT, 2.0, state, RIGHT, True))
    # done=True, so the target is the reward alone.
    expected_each = 0.5 * 2.0 / learner.coder.tilings
    active = learner.coder.active(learner.scale(state))
    for index in active:
        assert learner.weights[RIGHT][index] == pytest.approx(expected_each)
    assert learner.q(state, RIGHT) == pytest.approx(0.5 * 2.0)
    # And no other action was touched.
    assert learner.q(state, LEFT) == pytest.approx(0.0)


def test_terminal_updates_do_not_bootstrap():
    env = world()
    learner = algorithms.build("semi_gradient_sarsa", env,
                               dict(alpha=0.3, gamma=0.99, epsilon=0.0,
                                    epsilon_min=0.0, epsilon_decay=1.0,
                                    tilings=8, episodes=10))
    state = env.start_state()
    other = (5.0, 5.0, 0.0, 0.0)
    # Give the successor a large value, which a terminal target must ignore.
    for index in learner.coder.active(learner.scale(other)):
        learner.weights[UP][index] = 100.0

    ending = Transition(state, RIGHT, 1.0, other, UP, True)
    assert learner.target(ending) == pytest.approx(1.0)

    going = Transition(state, RIGHT, 1.0, other, UP, False)
    assert going.done is False
    assert learner.target(going) == pytest.approx(1.0 + 0.99 * learner.q(other, UP))


def test_alpha_is_divided_across_the_tilings():
    """Raising the tiling count must not raise the effective step size.

    Without the division, eight tilings would each move by the full amount and
    then be summed again on the next lookup, so Q would overshoot by 8x and the
    weights diverge within a few hundred steps.
    """
    moves = {}
    for tilings in (1, 4, 16):
        env = world()
        learner = algorithms.build("semi_gradient_sarsa", env,
                                   dict(alpha=0.4, gamma=0.9, epsilon=0.0,
                                        epsilon_min=0.0, epsilon_decay=1.0,
                                        tilings=tilings, episodes=10))
        state = env.start_state()
        learner.update(Transition(state, RIGHT, 1.0, state, RIGHT, True))
        moves[tilings] = learner.q(state, RIGHT)
    for tilings, value in moves.items():
        assert value == pytest.approx(0.4 * 1.0), tilings


def test_a_continuous_room_offers_no_policy_table():
    """`greedy_policy` is empty and `greedy_action` answers instead.

    The screens ask for a policy while painting, so this returns empty rather
    than raising — but the replay needs a real answer, and takes it one state
    at a time.
    """
    env = world()
    learner = algorithms.build("semi_gradient_sarsa", env,
                               dict(alpha=0.3, gamma=0.99, epsilon=0.0,
                                    epsilon_min=0.0, epsilon_decay=1.0,
                                    tilings=8, episodes=10))
    assert learner.greedy_policy() == {}
    assert learner.greedy_action(env.start_state()) in ACTIONS


def test_the_world_refuses_to_enumerate_its_states():
    env = world()
    with pytest.raises(NotImplementedError):
        env.all_states()
    with pytest.raises(NotImplementedError):
        env.transitions(env.start_state(), HOLD)


def test_a_planner_cannot_be_built_for_this_room():
    """Dynamic Programming needs the model, and this room does not have one."""
    env = world()
    with pytest.raises(NotImplementedError):
        algorithms.build("value_iteration", env,
                         dict(gamma=0.9, theta=1e-4))


# ----------------------------------------------------------------------
# The session, the recording and the contract
# ----------------------------------------------------------------------

def test_the_same_seed_reproduces_training():
    runs = []
    for _ in range(2):
        session = Session(ROOM, parameters={"episodes": 12})
        session.play()
        while session.state == "TRAINING":
            session.advance(steps=400)
        runs.append((session.history["reward"], session.history["length"]))
    assert runs[0][0] == runs[1][0]
    assert runs[0][1] == runs[1][1]


def test_room_four_uses_its_own_step_limit():
    session = Session(ROOM)
    assert session.step_limit == rooms.room(ROOM)["max_steps"]
    assert session.step_limit > config.SESSION["max_steps_per_episode"]


def test_recorded_frames_carry_position_and_velocity():
    session = Session(ROOM, parameters={"episodes": 4})
    session.play()
    while session.state == "TRAINING":
        session.advance(steps=400)

    batch = session.recorder.batch()
    assert batch["episodes"]
    for episode in batch["episodes"]:
        assert episode["steps"]
        for frame in episode["steps"]:
            assert set(("position", "velocity")) <= set(frame)
            assert frame["velocity"] is not None
            for key in ("x", "y"):
                assert math.isfinite(frame["position"][key])
                assert math.isfinite(frame["velocity"][key])


def test_the_greedy_replay_is_a_contract_episode():
    session = Session(ROOM, parameters={"episodes": 6})
    session.play()
    while session.state == "TRAINING":
        session.advance(steps=400)
    session.start_replay()

    replay = session.replay
    assert replay["greedy"] is True
    assert replay["epsilon"] == 0.0
    assert replay["outcome"] in ("success", "failure", "timeout")
    assert replay["steps"]
    assert replay["steps"][0]["velocity"] is not None
    # Actions are named with this room's own table, not the grid's four.
    named = [f["action"] for f in replay["steps"][1:] if f["action"]]
    assert named
    assert set(named) <= {"HOLD", "UP", "DOWN", "LEFT", "RIGHT"}


def test_rebuilding_a_session_restores_a_usable_model():
    """Changing a live parameter keeps the run; a reset-scope one rebuilds it."""
    session = Session(ROOM, parameters={"episodes": 8})
    session.play()
    while session.state == "TRAINING":
        session.advance(steps=400)
    trained = session.algorithm.episodes
    assert trained > 0

    session.set_parameters({"alpha": 0.2})
    assert session.algorithm.episodes == trained          # live: kept

    session.reset()
    assert session.algorithm.episodes == 0                # reset: rebuilt
    assert session.state == "IDLE"
    assert session.env.state == session.env.start_state()


def test_the_room_four_definition_satisfies_the_contract():
    """The same fields `web/assets/room/contract.js` checks at the boundary."""
    session = Session(ROOM)
    built = definition.build(rooms.room(ROOM), session.env, session.parameters)

    for field in ("id", "name", "worldSize", "entityTypes", "info"):
        assert built[field] is not None
    assert built["worldSize"]["width"] > 0
    assert built["worldSize"]["height"] > 0

    for entity in built["entities"]:
        assert entity["type"] in built["entityTypes"], entity["id"]
        assert isinstance(entity["position"]["x"], float)
        assert isinstance(entity["position"]["y"], float)
        assert entity["size"]["width"] > 0
        assert entity["size"]["height"] > 0

    for parameter in built["parameterSchema"]:
        for field in ("key", "label", "min", "max", "default"):
            assert parameter[field] is not None, parameter
        assert parameter["appliesLive"] in (True, False)


def test_room_four_is_not_a_grid():
    session = Session(ROOM)
    built = definition.build(rooms.room(ROOM), session.env, session.parameters)
    assert built["isGrid"] is False
    # A grid room needs a cellSize; this one must not pretend to have cells.
    assert "cellSize" not in built
    assert built["landingSpeed"] == pytest.approx(session.env.landing_speed)
    assert built["speedLimit"] == pytest.approx(session.env.speed_limit)


def test_every_entity_type_the_room_uses_is_declared_and_drawable():
    """No entity may name a type the config has not got.

    The legend is generated from these, so a type missing here is a thing that
    appears on the canvas and not in the key.
    """
    session = Session(ROOM)
    for entity in session.env.entities():
        assert entity["type"] in config.ENTITIES, entity["type"]
        assert config.ENTITIES[entity["type"]]["shape"]
        assert config.ENTITIES[entity["type"]]["colour"].startswith("--")


def test_the_fans_are_decoration_and_apply_no_force():
    """Deleting every fan must not change one number in `step`.

    The fans exist to explain the wind zones. If they were load-bearing, this
    room would have visual state the four-number state cannot see.

    Written by *removing the fans* rather than by comparing two chambers built
    the same way. An earlier version of this test did the latter, which
    compared a world against an identical copy of itself and would have passed
    however load-bearing the fans were.
    """
    with_fans = world()
    without = world()
    # There is no way to switch a fan off through the room dict — they are
    # generated, not declared — so the generator is what gets removed.
    without._fans_for = lambda zone, wind: []

    assert [e for e in with_fans.entities() if e["type"] == "fan"]
    assert not [e for e in without.entities() if e["type"] == "fan"]

    for chamber in (with_fans, without):
        chamber.reset()
    for action in (RIGHT,) * 120 + (UP,) * 200 + (HOLD,) * 80:
        one = with_fans.step(action)
        two = without.step(action)
        assert one[0] == two[0]                 # the state, exactly
        assert one[1] == pytest.approx(two[1])  # and the reward
        assert one[2] == two[2]
        if one[2]:
            break


def test_both_fan_banks_stand_at_the_mouth_of_a_wind_zone():
    """A fan that is not in front of the air it explains explains nothing.

    Each bank sits just outside its zone on the upwind edge, in a row across
    the mouth, and every fan points the way its zone actually blows — which is
    worked out from the zone's own wind vector, never written down twice.
    """
    env = world()
    banks = {}
    for entity in env.entities():
        if entity["type"] == "fan":
            banks.setdefault(entity["id"].split("-")[0], []).append(entity)

    winds = {zone["id"]: zone for zone in env.zones if zone.get("wind")}
    assert set(banks) == set(winds), (set(banks), set(winds))

    for zone_id, fans in banks.items():
        zone = winds[zone_id]
        assert len(fans) >= 1
        for fan in fans:
            assert fan["appearance"]["blows"] == env._blows(zone["wind"])
            # Inside the chamber, and clear of the zone it feeds.
            x, y = fan["position"]["x"], fan["position"]["y"]
            assert 0 < x < env.width and 0 < y < env.height
        # A bank blowing down stands in a row: same y, different x.
        if env._blows(zone["wind"]) in ("up", "down"):
            assert len({round(f["position"]["y"], 6) for f in fans}) == 1
            assert len({round(f["position"]["x"], 6) for f in fans}) == len(fans)


def test_the_second_bank_sits_on_the_route_the_agent_actually_flies():
    """The reason it was added: the first bank was not in the way.

    The learned route runs right along the floor and then climbs the east wall,
    which never comes near the top-left band — the chamber could be solved
    without meeting a fan at all. This pins the fix: the turn from the floor
    onto the climb is inside a wind zone.
    """
    env = world()
    corner = (8.2, 8.0)          # where the L turns upward
    assert env.zones_at(*corner), "the corner of the route is in open air"
    assert any(zone.get("wind") for zone in env.zones_at(*corner))

    # And it is a wide, shallow band rather than another square.
    band = next(zone for zone in env.zones_at(*corner) if zone.get("wind"))
    assert band["width"] > band["height"] * 2

    # Clear of the floor, so the downdraught is a hurdle and not a trap.
    assert env.height - (band["y"] + band["height"] / 2) >= 1.0


def test_the_scene_reports_velocity_for_the_live_view():
    session = Session(ROOM)
    scene = session.snapshot()["scene"]
    assert scene["velocity"] == {"x": 0.0, "y": 0.0}
    assert "steps" in scene
    # At rest there is no meaningful heading, and the screen is told so rather
    # than being handed an angle a rounding error picked.
    assert scene["facing"] is None


def test_the_room_declares_a_live_pace_and_the_grid_rooms_do_not():
    """The regression test for "the drone does not move".

    Nothing was broken when that was reported: the speed tiers count
    environment steps per second, and a step here covers a five-hundredth of
    the chamber where a grid step covers a tenth. At the Normal tier of 9 steps
    a second that is 11 pixels a second and 56 seconds to cross — motionless to
    look at. The room therefore says what one of its steps is worth, and the
    live loop scales the tier by it.

    Grid rooms must keep answering nothing, so their pacing is untouched.
    """
    session = Session(ROOM)
    built = definition.build(rooms.room(ROOM), session.env, session.parameters)
    scale = built["playback"]["liveStepScale"]
    assert scale > 1

    # At the Normal tier this has to cross the chamber in a few seconds, not a
    # minute. The arithmetic, so a change to either number is caught here.
    steps_per_second = 9 * scale
    metres_per_second = steps_per_second * session.env.dt * session.env.speed_limit
    seconds_to_cross = session.env.width / metres_per_second
    assert 2.0 < seconds_to_cross < 12.0, seconds_to_cross

    for number in (1, 2, 3):
        grid = Session(number)
        other = definition.build(rooms.room(number), grid.env, grid.parameters)
        assert "playback" not in other, "room %d gained a pacing hint" % number


def test_the_policy_does_not_collapse_onto_a_single_action():
    """A learner that always returns HOLD would sit still and look broken.

    Checked over states spread through the chamber rather than at the start
    alone, because an untrained approximator can easily be flat in one place
    and not in another — and after training the greedy policy must actually
    steer.
    """
    session = Session(ROOM, parameters={"episodes": 250})
    session.play()
    while session.state == "TRAINING":
        session.advance(budget_ms=200)

    learner = session.algorithm
    probes = [(x, y, vx, vy)
              for x in (1.5, 5.0, 8.0)
              for y in (2.0, 5.0, 8.0)
              for vx, vy in ((0.0, 0.0), (0.5, -0.5))]
    chosen = {learner.greedy_action(state) for state in probes}
    assert len(chosen) > 1, "the policy answers %r everywhere" % chosen

    # Every weight finite, and something actually learned.
    total = 0.0
    for vector in learner.weights.values():
        for weight in vector:
            assert math.isfinite(weight)
            total += abs(weight)
    assert total > 0.0, "no weight ever moved"

    # And the features are never empty, which would make every Q zero.
    for state in probes:
        assert len(learner.coder.active(learner.scale(state))) == learner.coder.tilings


def test_the_readout_describes_continuous_control():
    session = Session(ROOM)
    labels = [label for label, _ in session.readout()]
    for wanted in ("Position", "Velocity", "Speed", "Distance to pad",
                   "In field", "Landing limit", "Weight norm"):
        assert wanted in labels, wanted
