import csv
from types import SimpleNamespace

import numpy as np
import pygame
import pytest
from conftest import FixedRandom

from DroneSwarm2D.core import AirTrafficEnv as env_module
from DroneSwarm2D.core.AirTrafficEnv import AirTrafficEnv, compute_global_matrices
from DroneSwarm2D.core.EnemyDrone import EnemyDrone

V = pygame.math.Vector2

LEGACY_INFO_KEYS = {
    "current_step", "accum_reward", "enemies_shotdown", "friends_shotdown", "sucessful_attacks",
    "interest_point_health", "state_percentages", "total_distance_traveled",
}


def frozen_enemy(env, position, behavior="direct"):
    enemy = EnemyDrone(env.interest_point.center, position=tuple(position), behavior_type=behavior)
    enemy.start_delay = 10 ** 9  # never moves
    return enemy


def common_drones(env):
    return [d for d in env.friend_drones if d.behavior_type == "COMMON"]


# ----------------------------------------------------------------------------- reset / step

def test_reset_builds_the_scenario(settings, make_env):
    env = make_env()
    obs, done = env.reset()
    assert not done
    assert obs.dtype == np.float32 and obs.shape == (4 * (settings.FRIEND_COUNT + settings.ENEMY_COUNT),)
    roles = [d.behavior_type for d in env.friend_drones]
    assert roles.count("RADAR") == settings.RADAR_COUNT and roles.count("AEW") == settings.AEW_COUNT
    assert sum(d.broken for d in env.friend_drones) == settings.BROKEN_COUNT
    assert all(d.fixed for d in env.friend_drones if d.behavior_type == "RADAR")
    assert env.leader is env.selected_drone and env.leader.is_leader and env.leader.selected
    assert len(env.enemy_drones) == settings.ENEMY_COUNT
    assert len(env.demilitarized_zones) == 1
    assert env.episode == 2 and env.current_step == 0


def test_step_returns_gym_like_tuple_with_all_info_keys(make_env):
    env = make_env()
    obs, reward, done, info = env.step(None)
    assert isinstance(reward, float) and isinstance(done, bool)
    assert LEGACY_INFO_KEYS | {"friends_returned", "enemies_remaining", "friends_remaining"} <= set(info)
    assert info["current_step"] == 1
    assert sum(info["state_percentages"].values()) == pytest.approx(100.0)


def run_steps(env, n):
    trace = []
    for _ in range(n):
        obs, reward, done, info = env.step(None)
        trace.append((obs.tobytes(), reward))
    return trace


def test_same_seed_same_episode(make_env):
    first = run_steps(make_env(seed=7), 120)
    second = run_steps(make_env(seed=7), 120)
    assert first == second
    assert run_steps(make_env(seed=8), 120) != first


def test_reset_replays_the_same_initial_state(make_env):
    env = make_env(seed=5)
    obs_a, _ = env.reset()
    run_steps(env, 10)
    obs_b, _ = env.reset()
    assert np.array_equal(obs_a, obs_b)


def test_episode_ends_without_enemies_or_at_max_steps(make_env):
    env = make_env(max_steps=3)
    for _ in range(2):
        assert not env.step(None)[2]
    assert env.step(None)[2]

    env = make_env()
    env.enemy_drones = []
    assert env.step(None)[2]


def test_pause_freezes_the_simulation(make_env):
    env = make_env()
    env.toggle_pause()
    obs_before = env._get_observation()
    obs, reward, done, info = env.step(None)
    assert info["current_step"] == 0 and reward == 0.0 and np.array_equal(obs, obs_before)


# ----------------------------------------------------------------------------- combat rules

@pytest.mark.parametrize("role, rand_val, expected", [
    ("COMMON", 0.10, "enemy"), ("COMMON", 0.60, "friend"), ("COMMON", 0.95, "both"),
    ("AEW", 0.10, "both"), ("AEW", 0.60, None), ("AEW", 0.95, None),
    ("RADAR", 0.10, "both"), ("RADAR", 0.60, None),
])
def test_engagement_outcomes(settings, role, rand_val, expected):
    # P(friend alive) = 0.5, P(enemy alive) = 0.2, P(both dead) = 0.3 in the test config.
    friend = SimpleNamespace(behavior_type=role)
    assert AirTrafficEnv._engagement_outcome(friend, rand_val) == expected


def test_kill_is_counted_as_enemy_shot_down(make_env):
    env = make_env()
    friend = common_drones(env)[0]
    env.enemy_drones = [frozen_enemy(env, friend.pos + V(5, 0))]
    env.rng = FixedRandom(0.1)
    _, _, done, info = env.step(None)
    assert info["enemies_shotdown"] == 1 and info["friends_shotdown"] == 0
    assert info["enemies_remaining"] == 0 and done


def test_lost_engagement_is_counted_as_friend_shot_down(make_env):
    env = make_env()
    friend = common_drones(env)[1]
    n_friends = len(env.friend_drones)
    env.enemy_drones = [frozen_enemy(env, friend.pos + V(5, 0))]
    env.rng = FixedRandom(0.6)
    _, _, _, info = env.step(None)
    assert info["friends_shotdown"] == 1 and info["enemies_shotdown"] == 0
    assert info["friends_remaining"] == n_friends - 1 and friend not in env.friend_drones


def test_successful_attack_is_not_a_kill(settings, make_env):
    """Regression: enemies that hit the target used to be counted as shot down."""
    env = make_env()
    env.enemy_drones = [frozen_enemy(env, env.interest_point.center)]
    radars = [d for d in env.friend_drones if d.behavior_type == "RADAR"]
    _, reward, _, info = env.step(None)
    assert info["sucessful_attacks"] == 1
    assert info["enemies_shotdown"] == 0
    assert info["interest_point_health"] == settings.INTEREST_POINT_INITIAL_HEALTH - settings.INTEREST_POINT_DAMAGE
    # Each hit destroys a ground radar.
    assert info["friends_shotdown"] == len(radars) == 1
    # Regression: the penalty is applied in the step of the attack (it used to be delayed).
    assert reward < -1000


def test_reward_bonus_counts_kills_only(make_env):
    env = make_env()
    base = env.compute_reward()
    env.enemy_drones = env.enemy_drones[1:]  # an enemy disappears without being killed
    assert env.compute_reward() < base
    env.enemies_neutralized = 1
    assert env.compute_reward() > base - 1


def test_dmz_protects_enemies_only_when_enabled(make_env):
    env = make_env()
    friend = common_drones(env)[0]
    friend.fixed = True
    zone = env.demilitarized_zones[0]
    zone.center = V(friend.pos)  # friends cannot leave EXTERNAL_RADIUS, so bring the zone to them
    env.rng = FixedRandom(0.1)

    env.enemy_drones = [frozen_enemy(env, friend.pos + V(3, 0))]
    env.step(None)
    assert len(env.enemy_drones) == 1

    env.toggle_dmz()
    env.step(None)
    assert len(env.enemy_drones) == 0


def test_return_to_base(make_env):
    env = make_env()
    env.enemy_drones = []
    n_friends = len(env.friend_drones)
    env.toggle_return()
    for _ in range(200):
        _, _, _, info = env.step(None)
    assert info["friends_returned"] == n_friends and info["friends_shotdown"] == 0
    assert env.friend_drones == [] and env.leader is None


# ----------------------------------------------------------------------------- regressions

def test_more_layers_than_drones(monkeypatch, make_env):
    monkeypatch.setattr(env_module, "INITIAL_N_LAYERS", 50)
    env = make_env()
    assert len(env.friend_drones) > 0


def test_joystick_enemy_is_reproducible(monkeypatch, make_env):
    monkeypatch.setattr(env_module, "JOYSTICK", "Enemy")
    EnemyDrone.class_seed = None
    first = make_env(seed=3)
    pos_first = [tuple(e.pos) for e in first.enemy_drones]
    EnemyDrone.class_seed = None
    second = make_env(seed=3)
    assert first.enemy_drones[0].behavior_type == "joystick"
    assert pos_first == [tuple(e.pos) for e in second.enemy_drones]


def test_click_selects_a_drone_even_without_selection(make_env):
    env = make_env()
    env.selected_drone.selected = False
    env.selected_drone = None
    target = env.friend_drones[-1]
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(target.pos.x), int(target.pos.y)), button=1))
    env.handle_events()
    assert env.selected_drone is target and target.selected


def test_toggles_and_legacy_aliases(make_env):
    env = make_env()
    pairs = [("toogle_save_frames", "save_frames"), ("toggle_trajetory", "show_trajectory"),
             ("toogle_return", "return_to_base"), ("toogle_triangulation", "use_triangulation"),
             ("toggle_graph", "show_graph"), ("toggle_debug", "show_debug"),
             ("toggle_comm_range", "show_friend_comm_range"),
             ("toggle_target_lines", "show_target_lines"), ("toggle_friend_range", "show_friend_detection_range"),
             ("toggle_enemy_range", "show_enemy_detection_range"), ("toggle_dashed", "show_dashed_lines"),
             ("toggle_tacview_export", "export_to_tacview")]
    for method, attribute in pairs:
        before = getattr(env, attribute)
        getattr(env, method)()
        assert getattr(env, attribute) is (not before)
    assert len(env.buttons) == 16


def test_compute_global_matrices_matches_reference(make_friend):
    drones = [make_friend((0, 0)) for _ in range(3)]
    rng = np.random.default_rng(2)
    for d in drones:
        shape = d.enemy_intensity.shape
        d.enemy_intensity[:] = np.where(rng.random(shape) > 0.7, rng.random(shape), 0)
        d.enemy_direction[:] = rng.normal(size=d.enemy_direction.shape)
    intensity, direction = compute_global_matrices(drones, "enemy_intensity", "enemy_direction")

    for i, j in np.ndindex(intensity.shape):
        best, best_dir = 0.0, np.zeros(2)
        for d in drones:
            if d.enemy_intensity[i, j] > best:
                best, best_dir = d.enemy_intensity[i, j], d.enemy_direction[i, j]
        assert intensity[i, j] == best and np.array_equal(direction[i, j], best_dir)


# ----------------------------------------------------------------------------- outputs

def test_human_mode_renders_every_layer(settings, make_env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = make_env(mode="human")
    env.render_fps = 0
    env.toggle_graph()
    env.toggle_target_lines()
    env.toggle_comm_range()
    env.toggle_save_frames()
    for _ in range(2):
        env.step(None)
    frames = list(tmp_path.glob("tmp/frames_*/frame_*.png"))
    assert len(frames) == 2
    graph = env.plot_individual_states()
    assert graph.get_size() == (settings.GRAPH_WIDTH, settings.GRAPH_HEIGHT)


def test_tacview_export(make_env, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = make_env()
    run_steps(env, 3)
    env.handle_tacview()
    files = list(tmp_path.glob("output/tacview_export_*/*.csv"))
    assert len(files) == len(env.tacview_logs) > 0
    with open(files[0], newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0][:3] == ["Time", "Longitude", "Latitude"] and len(rows) == 4
