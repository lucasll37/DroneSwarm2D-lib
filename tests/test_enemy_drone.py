import math

import pygame
import pytest

from DroneSwarm2D.core.EnemyDrone import ENEMY_BEHAVIORS, RANDOM_ENEMY_BEHAVIORS, EnemyDrone

V = pygame.math.Vector2


class ConstantRng:
    def __init__(self, value):
        self.value = value

    def rand(self):
        return self.value


def run(enemy, steps, detector=None):
    for _ in range(steps):
        enemy.update(detector)


def test_random_border_position(settings):
    for _ in range(50):
        enemy = EnemyDrone(settings.CENTER)
        x, y = enemy.pos
        assert x in (0, settings.SIM_WIDTH) or y in (0, settings.SIM_HEIGHT)
        assert 0 <= x <= settings.SIM_WIDTH and 0 <= y <= settings.SIM_HEIGHT
        assert enemy.behavior_type in RANDOM_ENEMY_BEHAVIORS


def test_initial_velocity_points_to_the_target(settings):
    enemy = EnemyDrone(settings.CENTER, position=(0, settings.CENTER.y), behavior_type="direct")
    assert enemy.vel == V(settings.ENEMY_SPEED, 0)
    assert enemy.drone_id == 0 and EnemyDrone(settings.CENTER).drone_id == 1


@pytest.mark.parametrize("behavior", [b for b in ENEMY_BEHAVIORS if b != "joystick"])
def test_every_pattern_stays_in_bounds(settings, behavior):
    enemy = EnemyDrone(settings.CENTER, position=(10, 10), behavior_type=behavior)
    for _ in range(400):
        enemy.update()
        assert 0 <= enemy.pos.x <= settings.SIM_WIDTH and 0 <= enemy.pos.y <= settings.SIM_HEIGHT
        assert enemy.vel.length() <= settings.ENEMY_SPEED + 1e-9
        assert math.isfinite(enemy.pos.x) and math.isfinite(enemy.pos.y)


def test_direct_enemy_reaches_the_target_and_stops(settings):
    enemy = EnemyDrone(settings.CENTER, position=(settings.CENTER.x - 50, settings.CENTER.y), behavior_type="direct")
    run(enemy, 200)
    assert enemy.pos == settings.CENTER and enemy.vel == V(0, 0)


def test_start_delay(settings):
    enemy = EnemyDrone(settings.CENTER, position=(10, 10), behavior_type="zigzag")
    enemy.start_delay = 5
    run(enemy, 4)
    assert enemy.pos == V(10, 10)
    run(enemy, 2)
    assert enemy.pos != V(10, 10)


def test_aggressiveness_grows_towards_the_target(settings):
    far = EnemyDrone(settings.CENTER, position=settings.CENTER + V(settings.EXTERNAL_RADIUS, 0), behavior_type="direct")
    near_position = settings.CENTER + V(settings.INITIAL_DISTANCE - 1, 0)
    near = EnemyDrone(settings.CENTER, position=near_position, behavior_type="direct")
    for e in (far, near):
        e.start_delay = 0
        e.update()
    assert far.aggressiveness == pytest.approx(settings.INITIAL_AGGRESSIVENESS, abs=0.05)
    assert near.aggressiveness == 1.0


def test_detected_enemy_attacks_when_aggressive(settings, monkeypatch):
    monkeypatch.setattr(EnemyDrone, "class_np_rng", ConstantRng(0.0))
    enemy = EnemyDrone(settings.CENTER, position=(50, 50), behavior_type="zigzag")
    enemy.start_delay = 0
    enemy.update(detector=V(60, 60))
    assert enemy.desperate_attack
    enemy.update()
    assert enemy.info == "DESPERATE ATTACK"
    assert enemy.vel.normalize().dot((settings.CENTER - enemy.pos).normalize()) == pytest.approx(1, abs=1e-3)


def test_detected_enemy_escapes_when_not_aggressive(settings, monkeypatch):
    monkeypatch.setattr(EnemyDrone, "class_np_rng", ConstantRng(0.999))
    enemy = EnemyDrone(settings.CENTER, position=(200, 200), behavior_type="direct")
    enemy.start_delay = 0
    detector = V(250, 250)
    enemy.update(detector)
    assert enemy.escape_steps_count == 1
    enemy.update(detector)
    assert enemy.info.startswith("ESCAPE")
    assert enemy.vel.normalize() == (enemy.pos - enemy.vel - detector).normalize()
    run(enemy, settings.ESCAPE_STEPS)
    assert enemy.escape_steps_count == 0 and enemy.detector is None


def test_escape_from_a_detector_at_the_same_position(settings, monkeypatch):
    monkeypatch.setattr(EnemyDrone, "class_np_rng", ConstantRng(0.999))
    enemy = EnemyDrone(settings.CENTER, position=(200, 200), behavior_type="direct")
    enemy.start_delay = 0
    enemy.update(V(201, 201))
    enemy.update(V(enemy.pos))  # regression: zero vector used to crash
    assert enemy.vel.length() == pytest.approx(settings.ENEMY_SPEED)


def test_joystick_without_device(settings):
    enemy = EnemyDrone(settings.CENTER, position=(10, 10), behavior_type="joystick")
    assert enemy.start_delay == 0
    enemy.update(V(20, 20))
    assert enemy.info == "NO JOYSTICK" and not enemy.desperate_attack
    assert enemy.pos != V(10, 10)


def test_unknown_behavior_warns_and_flies_direct(settings):
    with pytest.warns(RuntimeWarning, match="Unknown enemy behaviour"):
        enemy = EnemyDrone(settings.CENTER, position=(0, settings.CENTER.y), behavior_type="teleport")
    enemy.start_delay = 0
    enemy.update()
    assert enemy.vel == V(settings.ENEMY_SPEED, 0)


def test_seeded_runs_are_identical(settings):
    def trajectory():
        EnemyDrone.set_class_seed(11)
        enemies = [EnemyDrone(settings.CENTER) for _ in range(5)]
        for _ in range(150):
            for e in enemies:
                e.update()
        return [(e.behavior_type, tuple(e.pos)) for e in enemies]

    assert trajectory() == trajectory()


def test_draw_does_not_crash(settings):
    surface = pygame.Surface((settings.SIM_WIDTH, settings.SIM_HEIGHT)).convert_alpha()
    enemy = EnemyDrone(settings.CENTER, position=(100, 100), behavior_type="direct")
    enemy.start_delay = 0
    run(enemy, 5)
    enemy.draw(surface, show_detection=True, show_trajectory=True, show_debug=True)
