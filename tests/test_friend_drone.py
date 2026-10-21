import random

import numpy as np
import pygame
import pytest
from conftest import fake_target

from DroneSwarm2D.behaviorsDefault import (
    BaseBehavior,
    BehaviorType,
    FriendAEWBehaviorDefault,
    FriendCommonBehaviorDefault,
    FriendRadarBehaviorDefault,
)
from DroneSwarm2D.core import FriendDrone as friend_module
from DroneSwarm2D.core.FriendDrone import FriendDrone
from DroneSwarm2D.core.utils import pos_to_cell

V = pygame.math.Vector2


class EnumRadar(BaseBehavior):
    """Custom radar that stores its type as the enum (as the old examples did)."""

    def __init__(self):
        super().__init__(BehaviorType.RADAR)
        self.type = BehaviorType.RADAR

    def apply(self, state, joystick_controlled=False):
        return ("RADAR", None, None, None), V(0, 0)


# ----------------------------------------------------------------------------- basics

def test_ids_and_default_behavior(settings):
    a = FriendDrone(settings.CENTER, (10, 10))
    b = FriendDrone(settings.CENTER, (20, 20))
    assert (a.drone_id, b.drone_id) == (0, 1)
    assert isinstance(a.behavior, FriendCommonBehaviorDefault)
    assert a.behavior_type == "COMMON"


def test_detection_range_by_role(settings, make_friend):
    assert make_friend((0, 0))._get_detection_range() == settings.FRIEND_DETECTION_RANGE
    assert make_friend((0, 0), FriendAEWBehaviorDefault())._get_detection_range() == settings.AEW_DETECTION_RANGE
    assert make_friend((0, 0), FriendRadarBehaviorDefault())._get_detection_range() == settings.RADAR_DETECTION_RANGE


def test_enum_typed_radar_is_recognised(settings, make_friend):
    """Regression: behaviours storing ``type`` as the enum were treated as interceptors."""
    radar = make_friend((0, 0), EnumRadar())
    common = make_friend((0, 0))
    assert radar._get_detection_range() == settings.RADAR_DETECTION_RANGE
    assert radar.image.get_width() > common.image.get_width()


def test_decay(make_friend):
    drone = make_friend((100, 100))
    drone.enemy_intensity[1, 1] = drone.friend_intensity[2, 2] = 1.0
    drone.decay_matrices()
    assert drone.enemy_intensity[1, 1] == pytest.approx(0.99)
    assert drone.friend_intensity[2, 2] == pytest.approx(0.99)


def test_get_state_shapes(settings, make_friend):
    state = make_friend((100, 50)).get_state()
    assert state["pos"].shape == (1, 2) and state["pos"].dtype == np.float32
    assert state["enemy_intensity"].shape == (1, settings.GRID_WIDTH, settings.GRID_HEIGHT)
    assert state["enemy_direction"].shape == (1, settings.GRID_WIDTH, settings.GRID_HEIGHT, 2)


# ----------------------------------------------------------------------------- detection

def test_direct_detection_needs_two_observations_for_a_heading(settings, make_friend):
    drone = make_friend((300, 300))
    drone.detection_mode = "direct"
    enemy = fake_target(380, 300)

    drone.update_local_enemy_detection([drone], [enemy])
    assert drone.enemy_intensity.max() == 0  # first sight: position only

    enemy.pos = V(390, 300)
    FriendDrone._logical_clock = 7
    drone.update_local_enemy_detection([drone], [enemy])
    cell = pos_to_cell(enemy.pos)
    assert drone.enemy_intensity[cell] == 1.0
    assert tuple(drone.enemy_direction[cell]) == (1.0, 0.0)
    assert drone.enemy_timestamp[cell] == 7


def test_direct_detection_clears_the_cell_left_behind(make_friend):
    drone = make_friend((300, 300))
    drone.detection_mode = "direct"
    enemy = fake_target(340, 300)
    drone.update_local_enemy_detection([drone], [enemy])
    enemy.pos = V(345, 300)
    drone.update_local_enemy_detection([drone], [enemy])
    old_cell = pos_to_cell(enemy.pos)
    enemy.pos = V(395, 300)
    drone.update_local_enemy_detection([drone], [enemy])
    assert drone.enemy_intensity[old_cell] == 0
    assert drone.enemy_intensity[pos_to_cell(enemy.pos)] == 1.0


def test_enemies_out_of_range_are_forgotten(settings, make_friend):
    drone = make_friend((100, 100))
    drone.detection_mode = "direct"
    enemy = fake_target(120, 100)
    drone.update_local_enemy_detection([drone], [enemy])
    assert id(enemy) in drone.current_enemy_pos_detection
    enemy.pos = V(100 + settings.FRIEND_DETECTION_RANGE + 1, 100)
    drone.update_local_enemy_detection([drone], [enemy])
    assert id(enemy) not in drone.current_enemy_pos_detection


def test_triangulation_requires_enough_sensors(settings, make_friend):
    n = settings.N_LINE_SIGHT_CROSSING
    friends = [make_friend((300 + 10 * k, 300)) for k in range(n)]
    for f in friends:
        f.detection_mode = "triangulation"
    enemy = fake_target(400, 320)

    # Only n - 1 sensors cover the enemy: never detected.
    for _ in range(2):
        friends[0].update_local_enemy_detection(friends[:-1], [enemy])
        enemy.pos += V(-1, 0)
    assert friends[0].enemy_intensity.max() == 0

    for _ in range(2):
        friends[0].update_local_enemy_detection(friends, [enemy])
        enemy.pos += V(-1, 0)
    assert friends[0].enemy_intensity.max() == 1.0


def test_clean_empty_cells_resets_stale_beliefs(settings, make_friend):
    drone = make_friend((300, 300))
    near, far = pos_to_cell(V(330, 300)), pos_to_cell(V(800, 700))
    drone.enemy_intensity[near] = drone.enemy_intensity[far] = 0.5
    drone._clean_empty_cells(settings.FRIEND_DETECTION_RANGE, current_time=42, is_enemy=True)
    assert drone.enemy_intensity[near] == 0 and drone.enemy_timestamp[near] == 42
    assert drone.enemy_intensity[far] == 0.5 and drone.enemy_timestamp[far] == 0


def test_friend_detection_ignores_aew_and_self(make_friend):
    drone = make_friend((300, 300))
    aew = make_friend((320, 300), FriendAEWBehaviorDefault())
    other = make_friend((340, 300))
    for _ in range(2):
        other.pos += V(1, 0)
        aew.pos += V(1, 0)
        drone.update_local_friend_detection([drone, aew, other])
    assert drone.friend_intensity[pos_to_cell(other.pos)] == 1.0
    assert drone.friend_intensity[pos_to_cell(aew.pos)] == 0 or pos_to_cell(aew.pos) == pos_to_cell(other.pos)
    assert id(drone) not in drone.current_friend_pos_detection


def test_passive_detection_traces_lines_of_sight(settings, make_friend):
    drone = make_friend((300, 300))
    near, far = fake_target(400, 300), fake_target(300 + settings.FRIEND_DETECTION_RANGE + 50, 300)
    drone.update_passive_detection([near, far])

    assert len(drone.direction_vectors) == 1
    assert next(iter(drone.direction_vectors.values())) == V(1, 0)
    matrix = drone.passive_detection_matrix
    g = settings.TRIANGULATION_GRANULARITY
    assert matrix.shape == (settings.GRID_WIDTH * g, settings.GRID_HEIGHT * g)
    marked = np.argwhere(matrix == 1)
    fine = settings.CELL_SIZE / g
    assert len(marked) == int(settings.FRIEND_DETECTION_RANGE / fine)
    assert (marked[:, 1] == int(300 // fine)).all()
    assert marked[:, 0].min() == int((300 + fine) // fine)

    drone.passive_detection_matrix = np.ones((2, 2))  # the attribute stays assignable
    assert drone.passive_detection_matrix.shape == (2, 2)


def test_passive_detection_stays_inside_the_grid(settings, make_friend):
    drone = make_friend((5, 5))
    drone.update_passive_detection([fake_target(0, 0.5), fake_target(-50, -50)])
    marked = np.argwhere(drone.passive_detection_matrix == 1)
    assert (marked >= 0).all()


# ----------------------------------------------------------------------------- communication

def test_merge_copies_only_newer_cells(make_friend):
    a, b = make_friend((100, 100)), make_friend((120, 100))
    a.enemy_intensity[1, 1], a.enemy_timestamp[1, 1] = 0.3, 5
    b.enemy_intensity[1, 1], b.enemy_timestamp[1, 1] = 0.9, 10
    b.enemy_direction[1, 1] = (0, 1)
    b.enemy_intensity[2, 2], b.enemy_timestamp[2, 2] = 0.9, 1
    a.enemy_intensity[2, 2], a.enemy_timestamp[2, 2] = 0.1, 3

    a.merge_enemy_matrix(b)
    assert a.enemy_intensity[1, 1] == 0.9 and tuple(a.enemy_direction[1, 1]) == (0, 1) and a.enemy_timestamp[1, 1] == 10
    assert a.enemy_intensity[2, 2] == 0.1 and a.enemy_timestamp[2, 2] == 3

    b.friend_intensity[3, 3], b.friend_timestamp[3, 3] = 1.0, 2
    a.merge_friend_matrix(b)
    assert a.friend_intensity[3, 3] == 1.0


def reference_neighbors(me, all_drones, comm_range, n_connections):
    """Original (O(n²) per drone) neighbour selection."""
    candidates = [o for o in all_drones if o is not me and me.pos.distance_to(o.pos) < comm_range]
    candidates.sort(key=lambda o: me.pos.distance_to(o.pos))
    reverse = []
    for other in all_drones:
        if other is me:
            continue
        others = [o for o in all_drones if o is not other and other.pos.distance_to(o.pos) < comm_range]
        others.sort(key=lambda o: other.pos.distance_to(o.pos))
        if me in others[:n_connections]:
            reverse.append(other)
    return list(dict.fromkeys(candidates[:n_connections] + reverse))


@pytest.mark.parametrize("seed", range(8))
def test_neighbors_match_reference(settings, make_friend, seed):
    rng = random.Random(seed)
    # Grid-aligned positions create many exact distance ties.
    drones = [make_friend((rng.randrange(0, 600, 50), rng.randrange(0, 500, 50))) for _ in range(15)]
    for drone in drones:
        drone.update_neighbors(drones)
        expected = reference_neighbors(drone, drones, settings.COMMUNICATION_RANGE, settings.N_CONNECTIONS)
        assert drone.neighbors == expected
    for drone in drones:  # links are symmetric
        assert all(drone in other.neighbors for other in drone.neighbors)


def test_neighbors_misspelled_alias_and_foreign_list(settings, make_friend):
    a, b, c = make_friend((100, 100)), make_friend((110, 100)), make_friend((900, 700))
    a.update_neghbors([a, b, c])
    assert a.neighbors == [b]
    a.update_neighbors([b, c])  # `a` is not part of the list
    assert a.neighbors == [b]


def test_communication_respects_message_loss(monkeypatch, make_friend):
    a, b = make_friend((100, 100)), make_friend((110, 100))
    b.enemy_intensity[4, 4], b.enemy_timestamp[4, 4] = 1.0, 1
    a.neighbors = [b]

    monkeypatch.setattr(friend_module, "MESSAGE_LOSS_PROBABILITY", 1.0)
    a.communication([a, b])
    assert a.enemy_intensity[4, 4] == 0
    assert a.messages_sent_this_cycle > 0 and a.active_connections > 0

    monkeypatch.setattr(friend_module, "MESSAGE_LOSS_PROBABILITY", 0.0)
    a.communication([a, b])
    assert a.enemy_intensity[4, 4] == 1.0


# ----------------------------------------------------------------------------- motion

def test_external_radius_and_bounds(settings, make_friend):
    drone = make_friend((0, 0))
    drone.pos = settings.CENTER + V(settings.EXTERNAL_RADIUS + 50, 0)
    drone._enforce_external_radius()
    assert drone.pos.distance_to(settings.CENTER) == pytest.approx(settings.EXTERNAL_RADIUS)
    assert drone.vel == V(0, 0)

    drone.pos, drone.vel = V(-5, settings.SIM_HEIGHT + 5), V(-1, 1)
    drone._enforce_simulation_bounds()
    assert drone.pos == V(0, settings.SIM_HEIGHT) and drone.vel == V(1, -1)


def test_return_to_base(settings, make_friend):
    drone = make_friend(settings.CENTER + V(100, 0))
    drone.update([], [drone], use_triangulation=False, return_to_base=True)
    assert drone.info[0] == "REGRESS"
    assert drone.vel == V(-settings.FRIEND_SPEED, 0)

    drone.pos = V(settings.CENTER)  # regression: used to raise "can't normalize zero vector"
    drone.update([], [drone], use_triangulation=False, return_to_base=True)
    assert drone.vel == V(0, 0)


def test_fixed_drone_never_moves(settings, make_friend):
    radar = make_friend(settings.CENTER + V(50, 0), FriendRadarBehaviorDefault(), fixed=True)
    start = V(radar.pos)
    for _ in range(3):
        radar.update([], [radar], use_triangulation=False)
    assert radar.pos == start and radar.info[0] == "FIXED"
    assert radar.get_state_percentages() == {"FIXED": 100.0}


def test_update_bookkeeping(settings, make_friend):
    drone = make_friend(settings.CENTER + V(150, 0))
    drone.update([], [drone], use_triangulation=True)
    assert drone.total_steps == 1 and len(drone.trajectory) == 1
    assert drone.distance_traveled == pytest.approx(settings.FRIEND_SPEED)
    assert drone.detection_mode == "triangulation"
    assert FriendDrone._logical_clock == 1


# ----------------------------------------------------------------------------- broken drones

def test_broken_drone_near_the_border_does_not_crash(settings, make_friend):
    """Regression: the fake pattern kept its first shape and crashed near the borders."""
    drone = make_friend((settings.SIM_WIDTH / 2, settings.SIM_HEIGHT / 2), broken=True)
    for pos in [(settings.SIM_WIDTH / 2, settings.SIM_HEIGHT / 2), (3, 3), (settings.SIM_WIDTH - 1, 5)]:
        drone.pos = V(pos)
        drone.update_local_enemy_detection([drone], [])
        drone.update_local_friend_detection([drone])
        xs, ys, disk = drone._sensor_window(settings.FRIEND_DETECTION_RANGE)
        assert drone.broken_enemy_intensity.shape == disk.shape


def test_broken_drone_writes_its_fake_pattern(settings, make_friend):
    drone = make_friend((300, 300), broken=True)
    drone.update_local_enemy_detection([drone], [])
    xs, ys, disk = drone._sensor_window(settings.FRIEND_DETECTION_RANGE)
    expected = np.where(disk, drone.broken_enemy_intensity, 0)
    assert np.array_equal(drone.enemy_intensity[xs, ys], expected)


def test_broken_patterns_change_over_time_and_between_drones(make_friend):
    a, b = make_friend((300, 300), broken=True), make_friend((300, 300), broken=True)
    patterns_a = {a._new_fake_pattern((15, 15))[0].tobytes() for _ in range(6)}
    patterns_b = {b._new_fake_pattern((15, 15))[0].tobytes() for _ in range(6)}
    assert len(patterns_a) > 1 and len(patterns_b) > 1
    assert patterns_a != patterns_b


# ----------------------------------------------------------------------------- drawing

def test_draw_does_not_crash(settings, make_friend):
    surface = pygame.Surface((settings.SIM_WIDTH, settings.SIM_HEIGHT)).convert_alpha()
    drone = make_friend(settings.CENTER + V(100, 0))
    drone.is_leader = drone.selected = True
    drone.detection_mode = "triangulation"
    drone.direction_vectors = {"0.00": V(1, 0)}
    drone.info = ("GO HOLD INTCPT", V(10, 10), V(20, 20), None)
    drone.trajectory = [V(100, 100), V(110, 105), V(120, 115)]
    drone.draw(surface, show_detection=True, show_comm_range=True, show_trajectory=True, show_debug=True)
