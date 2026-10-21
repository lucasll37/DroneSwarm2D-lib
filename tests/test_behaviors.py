import numpy as np
import pygame
import pytest

from DroneSwarm2D.behaviorsDefault import (
    BaseBehavior,
    BehaviorType,
    FriendAEWBehaviorDefault,
    FriendCommonBehaviorDefault,
    FriendRadarBehaviorDefault,
)
from DroneSwarm2D.core.utils import cell_center, intercept_direction, pos_to_cell

V = pygame.math.Vector2


def empty_matrices(settings):
    shape = (settings.GRID_WIDTH, settings.GRID_HEIGHT)
    return np.zeros(shape), np.zeros(shape), np.zeros(shape + (2,)), np.zeros(shape + (2,))


def state_from(pos, friend_intensity, enemy_intensity, friend_direction, enemy_direction):
    return {
        "drone_id": 0,
        "pos": np.array([[pos.x, pos.y]], dtype=np.float32),
        "friend_intensity": friend_intensity[None],
        "enemy_intensity": enemy_intensity[None],
        "friend_direction": friend_direction[None],
        "enemy_direction": enemy_direction[None],
    }


# ----------------------------------------------------------------------------- API

def test_behavior_type_compares_equal_to_strings():
    assert BehaviorType.RADAR == "RADAR" and str(BehaviorType.AEW) == "AEW"
    assert BehaviorType("COMMON") is BehaviorType.COMMON
    assert {BehaviorType.RADAR: 1}["RADAR"] == 1


def test_base_behavior_validates_type():
    class Dummy(BaseBehavior):
        def apply(self, state, joystick_controlled=False):
            return ("DUMMY", None, None, None), V(0, 0)

    assert Dummy(BehaviorType.AI).type == "AI"
    assert Dummy("RADAR").type == "RADAR"
    assert isinstance(Dummy(BehaviorType.RADAR).type, str)
    with pytest.raises(ValueError):
        Dummy("LASER")
    with pytest.raises(ValueError):
        Dummy(42)


def test_base_behavior_is_abstract():
    with pytest.raises(TypeError):
        BaseBehavior(BehaviorType.COMMON)


# ----------------------------------------------------------------------------- AEW / radar

def test_radar_never_moves(settings):
    info, vel = FriendRadarBehaviorDefault().apply(state_from(settings.CENTER + V(50, 0), *empty_matrices(settings)))
    assert info[0] == "RADAR" and vel == V(0, 0)


def test_aew_orbits_tangentially(settings):
    pos = settings.CENTER + V(120, -40)
    info, vel = FriendAEWBehaviorDefault().apply(state_from(pos, *empty_matrices(settings)))
    assert info[0] == "AEW"
    assert vel.length() == pytest.approx(settings.AEW_SPEED)
    assert vel.dot(pos - settings.CENTER) == pytest.approx(0, abs=1e-3)


def test_aew_at_the_centre_still_moves(settings):
    _, vel = FriendAEWBehaviorDefault().apply(state_from(V(settings.CENTER), *empty_matrices(settings)))
    assert vel.length() == pytest.approx(settings.AEW_SPEED)


# ----------------------------------------------------------------------------- common

def test_common_without_information_regroups(settings):
    pos = settings.CENTER + V(150, 0)
    info, vel = FriendCommonBehaviorDefault().apply(state_from(pos, *empty_matrices(settings)))
    assert info[0] == "HOLD - NO ENOUGH COMM"
    assert vel.normalize().dot((settings.CENTER - pos).normalize()) == pytest.approx(1)


def test_common_at_the_centre_waits(settings):
    info, vel = FriendCommonBehaviorDefault().apply(state_from(V(settings.CENTER), *empty_matrices(settings)))
    assert info[0] == "HOLD - WAIT" and vel == V(0, 0)


def test_common_pursues_with_the_real_enemy_velocity(settings):
    fi, ei, fd, ed = empty_matrices(settings)
    pos = settings.CENTER + V(120, 0)
    enemy_cell = pos_to_cell(settings.CENTER + V(300, 60))
    ei[enemy_cell] = 1.0
    ed[enemy_cell] = (-1.0, 0.0)

    info, vel = FriendCommonBehaviorDefault().apply(state_from(pos, fi, ei, fd, ed))

    assert info[0] == "PURSUING"
    expected = intercept_direction(pos, settings.FRIEND_SPEED, cell_center(enemy_cell), V(-1, 0) * settings.ENEMY_SPEED)
    assert vel == expected
    assert vel.length() == pytest.approx(settings.FRIEND_SPEED)


def test_common_leaves_the_enemy_to_a_closer_friend(settings):
    fi, ei, fd, ed = empty_matrices(settings)
    pos = settings.CENTER + V(-150, 0)
    enemy_cell = pos_to_cell(settings.CENTER + V(300, 0))
    ei[enemy_cell] = 1.0
    ed[enemy_cell] = (-1.0, 0.0)
    fi[pos_to_cell(settings.CENTER + V(200, 0))] = 1.0  # a friend much closer to the enemy

    info, _ = FriendCommonBehaviorDefault().apply(state_from(pos, fi, ei, fd, ed))
    assert info[0].startswith("HOLD")


def test_hold_waits_on_the_line_of_approach_of_the_most_urgent_enemy(settings):
    """Regression: feasibility must use the chosen enemy, not the last scanned cell."""
    fi, ei, fd, ed = empty_matrices(settings)
    center = settings.CENTER

    cell_a = pos_to_cell(center + V(-300, 0))            # heading to the centre (the threat)
    a = cell_center(cell_a)
    heading_a = (center - a).normalize()
    cell_b = pos_to_cell(center + V(100, 0))             # scanned last, flying away
    ei[cell_a], ed[cell_a] = 1.0, tuple(heading_a)
    ei[cell_b], ed[cell_b] = 1.0, (1.0, 0.0)

    pos = a + heading_a * 100                            # on A's line, between A and the centre
    for offset in (V(30, 0), V(0, 30), V(-30, 0)):       # enough friends in communication range
        fi[pos_to_cell(pos + offset)] = 1.0

    behavior = FriendCommonBehaviorDefault()
    info, vel = behavior._hold_position(pos, fi, ei, ed, friends_hold=[])
    assert info[0] == "HOLD - INTCPT"
    assert vel == V(0, 0)


def test_hold_spreads_tangentially(settings):
    fi, ei, fd, ed = empty_matrices(settings)
    pos = settings.CENTER + V(120, 0)
    friends = [pos + V(0, 25), pos + V(0, 50), pos + V(-25, 10)]
    for p in friends:
        fi[pos_to_cell(p)] = 1.0
    hold = [(pos_to_cell(p), cell_center(pos_to_cell(p))) for p in [pos, *friends]]

    info, vel = FriendCommonBehaviorDefault()._hold_position(pos, fi, ei, ed, friends_hold=hold)
    assert info[0] == "HOLD - SPREAD"
    radial = (pos - settings.CENTER).normalize()
    assert vel.dot(radial) == pytest.approx(0, abs=1e-6)
    assert vel.length() <= settings.FRIEND_SPEED + 1e-9
