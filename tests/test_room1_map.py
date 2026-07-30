"""Tests for the Room 1 map — required checks 1 to 8, plus map sanity.

Numbered comments refer to the checklist in the project brief.
"""

from collections import Counter

from core import tiles
from rooms.room1 import map_data


def _counts():
    counter = Counter()
    for row in map_data.ROOM_MAP:
        counter.update(row)
    return counter


def test_map_is_exactly_ten_by_ten():
    """1. The map is exactly 10x10."""
    assert len(map_data.ROOM_MAP) == 10
    assert map_data.GRID_ROWS == 10
    assert map_data.GRID_COLS == 10
    for index, row in enumerate(map_data.ROOM_MAP):
        assert len(row) == 10, "row %d has %d cells" % (index, len(row))


def test_exactly_one_start():
    """2. There is exactly one start position."""
    assert _counts()[tiles.START] == 1
    assert map_data.start_cell() == (0, 0)


def test_exactly_one_exit():
    """3. There is exactly one exit."""
    assert _counts()[tiles.EXIT] == 1
    assert map_data.exit_cell() == (4, 9)


def test_exactly_two_teleporters():
    """4. There are exactly two teleporters, forming one pair."""
    assert _counts()[tiles.TELEPORT] == 2
    pads = map_data.teleport_cells()
    assert len(pads) == 2
    assert pads[0] != pads[1]


def test_at_least_one_laser():
    """5. There is at least one laser (the brief asks for at least three)."""
    assert _counts()[tiles.LASER] >= 3


def test_exactly_one_battery():
    """6. There is exactly one battery."""
    assert _counts()[tiles.BATTERY] == 1


def test_at_least_one_one_way_door():
    """7. There is at least one one-way door."""
    doors = sum(_counts()[door] for door in tiles.ONE_WAY_DOORS)
    assert doors >= 1


def test_every_character_is_valid():
    """8. All map characters are valid."""
    for row in map_data.ROOM_MAP:
        for character in row:
            assert character in tiles.VALID_TILES, "unknown tile %r" % character


def test_surface_variety_matches_the_brief():
    """The brief asks for at least three of each ice, and at least two oils."""
    counts = _counts()
    assert counts[tiles.WEAK_ICE] >= 3
    assert counts[tiles.STRONG_ICE] >= 3
    assert counts[tiles.OIL] >= 2
    assert counts[tiles.WALL] >= 4


def test_every_open_cell_is_reachable():
    """No part of the chamber may be sealed off by accident."""
    walkable = set(map_data.walkable_cells())
    reachable = map_data.reachable_cells()
    assert walkable - reachable == set(), \
        "unreachable cells: %s" % sorted(walkable - reachable)


def test_the_three_routes_exist_with_the_intended_lengths():
    """The room is built around a short risky route and a long safe one.

    If this fails the map has lost the trade-off the whole room demonstrates.
    """
    measured = map_data.route_lengths()
    assert measured == map_data.ROUTE_LENGTHS, \
        "route lengths changed: %s" % measured
    assert measured["shaft"] < measured["teleport"] < measured["ring"]


def test_the_one_way_door_cannot_make_the_room_unsolvable():
    """A route to the exit must exist even though the door only opens one way."""
    assert map_data.shortest_route_length() is not None


def test_map_hash_is_stable_and_short():
    """Saved results are checked against this fingerprint."""
    fingerprint = map_data.map_hash()
    assert isinstance(fingerprint, str)
    assert len(fingerprint) == 16
    assert map_data.map_hash() == fingerprint
