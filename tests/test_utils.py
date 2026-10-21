import math
import random

import numpy as np
import pygame
import pytest

from DroneSwarm2D.core import rendering, utils
from DroneSwarm2D.core.utils import (
    active_cells,
    can_intercept,
    cell_center,
    generate_sparse_matrix,
    get_friends_hold,
    intercept_direction,
    pos_to_cell,
    sim_to_geo,
    smooth_matrix_with_kernel_10x10,
    symmetrical_flat_topped_gaussian_10x10,
)

V = pygame.math.Vector2


# ----------------------------------------------------------------------------- grid

def test_pos_to_cell_inside_and_clamped(settings):
    c = settings.CELL_SIZE
    assert pos_to_cell(V(0, 0)) == (0, 0)
    assert pos_to_cell(V(c * 3 + 1, c * 5 - 1)) == (3, 4)
    last_cell = (settings.GRID_WIDTH - 1, settings.GRID_HEIGHT - 1)
    assert pos_to_cell(V(settings.SIM_WIDTH, settings.SIM_HEIGHT)) == last_cell
    assert pos_to_cell(V(-50, -1)) == (0, 0)
    assert pos_to_cell(V(25, 25), cell_size=10, grid_width=100, grid_height=100) == (2, 2)


def test_cell_center(settings):
    c = settings.CELL_SIZE
    assert cell_center((2, 3)) == V(2.5 * c, 3.5 * c)
    assert pos_to_cell(cell_center((7, 9))) == (7, 9)


def test_active_cells_matches_ndenumerate():
    matrix = np.random.default_rng(0).random((13, 17))
    expected = [cell for cell, value in np.ndenumerate(matrix) if value >= 0.6]
    result = active_cells(matrix, 0.6)
    assert result == expected
    assert all(isinstance(i, int) and isinstance(j, int) for i, j in result)
    assert active_cells(np.zeros((3, 3)), 0.1) == []


def test_sim_to_geo_maps_corners(settings):
    assert sim_to_geo(0, 0) == pytest.approx(tuple(settings.GEO_TOP_LEFT))
    assert sim_to_geo(settings.SIM_WIDTH, settings.SIM_HEIGHT) == pytest.approx(tuple(settings.GEO_BOTTOM_RIGHT))


def test_load_svg_as_surface():
    surface = utils.load_svg_as_surface(str(utils.__file__).replace("core/utils.py", "assets/drone_0.svg"))
    assert surface.get_width() > 0 and surface.get_flags() & pygame.SRCALPHA


# ----------------------------------------------------------------------------- interception

def simulate_pursuit(chaser, speed, target, target_vel, steps=500):
    """Return the minimum distance reached when the chaser keeps its initial heading."""
    vel = intercept_direction(chaser, speed, target, target_vel)
    best = math.inf
    for _ in range(steps):
        chaser, target = chaser + vel, target + target_vel
        best = min(best, chaser.distance_to(target))
    return best


def test_intercept_stationary_target_points_straight_at_it():
    vel = intercept_direction(V(0, 0), 2.0, V(30, 40), V(0, 0))
    assert vel.length() == pytest.approx(2.0)
    assert vel.normalize() == V(30, 40).normalize()


def test_intercept_moving_target_is_reached():
    assert simulate_pursuit(V(0, 0), 3.0, V(200, 100), V(-1, 0.5)) < 3.0


def test_intercept_equal_speed_approaching_target():
    # Same speed (the "linear" branch): only reachable when the target comes closer.
    assert simulate_pursuit(V(0, 0), 2.0, V(300, 50), V(-2, 0)) < 2.0


def test_intercept_impossible_falls_back_to_direct_pursuit():
    vel = intercept_direction(V(0, 0), 1.0, V(100, 0), V(5, 0))
    assert vel == V(1, 0)


def test_intercept_same_position_returns_zero():
    assert intercept_direction(V(5, 5), 2.0, V(5, 5), V(0, 0)) == V(0, 0)


def test_can_intercept():
    poi = V(0, 0)
    # Enemy far away coming slowly towards the POI; fast chaser next to its path.
    assert can_intercept(V(400, 10), 3.0, V(500, 0), V(-1, 0), poi)
    # Enemy almost at the POI, chaser far behind.
    assert not can_intercept(V(1000, 0), 2.0, V(10, 0), V(-2, 0), poi)
    # Stationary targets are never "interceptable".
    assert not can_intercept(V(0, 10), 2.0, V(50, 0), V(0, 0), poi)
    # Degenerate case (same speed, perpendicular motion): only when already there.
    assert can_intercept(V(5, 5), 2.0, V(5, 5), V(2, 0), poi)
    assert not can_intercept(V(0, 0), 2.0, V(0, 10), V(2, 0), poi)


def test_can_intercept_defaults_to_interest_point(settings):
    target = settings.INTEREST_POINT_CENTER + V(300, 0)
    assert can_intercept(target + V(-50, 5), 3.0, target, V(-1, 0))


# ----------------------------------------------------------------------------- strategy

def make_state(pos, enemy_cells=(), friend_cells=(), enemy_dir=(-1.0, 0.0)):
    from conftest import SETTINGS as s
    enemy_intensity = np.zeros((s.GRID_WIDTH, s.GRID_HEIGHT))
    friend_intensity = np.zeros_like(enemy_intensity)
    enemy_direction = np.zeros((s.GRID_WIDTH, s.GRID_HEIGHT, 2))
    for cell in enemy_cells:
        enemy_intensity[cell] = 1.0
        enemy_direction[cell] = enemy_dir
    for cell in friend_cells:
        friend_intensity[cell] = 1.0
    return {
        "drone_id": 0,
        "pos": np.array([[pos.x, pos.y]], dtype=np.float32),
        "enemy_intensity": enemy_intensity[None],
        "enemy_direction": enemy_direction[None],
        "friend_intensity": friend_intensity[None],
        "friend_direction": np.zeros_like(enemy_direction)[None],
    }


def test_get_friends_hold_without_enemies_returns_every_candidate(settings):
    pos = settings.CENTER + V(100, 0)
    friends = [pos_to_cell(settings.CENTER + V(0, 100))]
    hold = get_friends_hold(make_state(pos, friend_cells=friends))
    assert {cell for cell, _ in hold} == {pos_to_cell(pos), friends[0]}


def test_get_friends_hold_returns_unassigned_friends(settings):
    pos = settings.CENTER + V(60, 0)
    enemy = pos_to_cell(settings.CENTER + V(300, 0))
    far_friend = pos_to_cell(settings.CENTER + V(-200, 0))
    hold = get_friends_hold(make_state(pos, enemy_cells=[enemy], friend_cells=[far_friend]))
    assert isinstance(hold, list)  # regression: used to return None
    assert [cell for cell, _ in hold] == [far_friend]


# ----------------------------------------------------------------------------- matrices

def test_generate_sparse_matrix_properties():
    matrix, direction = generate_sparse_matrix((20, 30), max_nonzero=15, seed=3)
    assert matrix.shape == (20, 30) and direction.shape == (20, 30, 2)
    nonzero = matrix > 0
    assert nonzero.sum() <= 15
    assert ((matrix >= 0) & (matrix <= 1)).all()
    assert (direction[matrix == 0] == 0).all() or (matrix == 0).sum() < matrix.size
    norms = np.linalg.norm(direction[nonzero], axis=-1)
    assert np.allclose(norms, 1.0)


def test_generate_sparse_matrix_is_seeded_and_side_effect_free():
    random.seed(99)
    np.random.seed(99)
    expected_py, expected_np = random.random(), np.random.random()

    random.seed(99)
    np.random.seed(99)
    a = generate_sparse_matrix((10, 10), seed=5)
    b = generate_sparse_matrix((10, 10), seed=5)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    # The global RNGs were not reseeded/advanced by the seeded calls.
    assert random.random() == expected_py and np.random.random() == expected_np


def test_flat_topped_gaussian():
    kernel = symmetrical_flat_topped_gaussian_10x10(0.5, sigma=2, flat_radius=1)
    assert kernel.shape == (10, 10)
    assert np.allclose(kernel, kernel.T) and np.allclose(kernel, kernel[::-1, ::-1])
    assert kernel.max() == pytest.approx(0.5)
    assert kernel.min() > 0
    kernel[0, 0] = 99  # returned arrays are writable copies
    assert symmetrical_flat_topped_gaussian_10x10(0.5, 2, 1)[0, 0] != 99


def reference_smooth(matrix, direction, sigma, flat_radius):
    """Original (loop-based) implementation of smooth_matrix_with_kernel_10x10."""
    result, result_direction = np.copy(matrix), np.copy(direction)
    for i, j in np.argwhere(matrix > 0):
        bump = symmetrical_flat_topped_gaussian_10x10(matrix[i, j], sigma, flat_radius)
        i0, j0 = max(0, i - 5), max(0, j - 5)
        i1, j1 = min(matrix.shape[0], i + 5), min(matrix.shape[1], j + 5)
        bi, bj = 5 - (i - i0), 5 - (j - j0)
        region = result[i0:i1, j0:j1]
        bump_region = bump[bi:bi + (i1 - i0), bj:bj + (j1 - j0)]
        mask = bump_region > region
        np.maximum(region, bump_region, out=region)
        for dim in range(direction.shape[-1]):
            for di in range(i1 - i0):
                for dj in range(j1 - j0):
                    if mask[di, dj]:
                        result_direction[i0 + di, j0 + dj, dim] = direction[i, j][dim]
    return result, result_direction


def test_smooth_matrix_matches_reference_and_keeps_inputs():
    rng = np.random.default_rng(1)
    matrix = np.where(rng.random((30, 25)) > 0.93, rng.random((30, 25)), 0.0)
    direction = rng.normal(size=(30, 25, 2))
    matrix_copy, direction_copy = matrix.copy(), direction.copy()

    result, result_direction = smooth_matrix_with_kernel_10x10(matrix, direction, sigma=2, flat_radius=1)
    expected, expected_direction = reference_smooth(matrix, direction, 2, 1)

    assert np.allclose(result, expected) and np.allclose(result_direction, expected_direction)
    assert np.array_equal(matrix, matrix_copy) and np.array_equal(direction, direction_copy)
    assert (result >= matrix).all()


# ----------------------------------------------------------------------------- drawing

def blank_surface(size=(300, 200)):
    surface = pygame.Surface(size).convert_alpha()
    surface.fill((0, 0, 0))
    return surface


def surfaces_equal(a, b):
    return (np.array_equal(pygame.surfarray.array3d(a), pygame.surfarray.array3d(b))
            and np.array_equal(pygame.surfarray.array_alpha(a), pygame.surfarray.array_alpha(b)))


def reference_dashed_circle(surface, color, center, radius, dash_length=5, space_length=5, width=1):
    num_dashes = int(2 * math.pi * radius / (dash_length + space_length))
    angle_between = 2 * math.pi / num_dashes
    for i in range(num_dashes):
        start, end = i * angle_between, i * angle_between + dash_length / radius
        pygame.draw.line(surface, color,
                         (center[0] + radius * math.cos(start), center[1] + radius * math.sin(start)),
                         (center[0] + radius * math.cos(end), center[1] + radius * math.sin(end)), width)


def reference_trajectory(surface, points, color, decay_rate):
    traj = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    n = len(points)
    for i in range(n - 1):
        alpha = max(int(255 * math.exp(-decay_rate * (n - 1 - i))), 30)
        pygame.draw.line(traj, color + (alpha,), (int(points[i].x), int(points[i].y)),
                         (int(points[i + 1].x), int(points[i + 1].y)), 2)
    surface.blit(traj, (0, 0))


def test_dashed_circle_matches_reference():
    a, b = blank_surface(), blank_surface()
    rendering.draw_dashed_circle(a, (255, 255, 255, 32), (150, 100), 70, 5, 5, 1)
    reference_dashed_circle(b, (255, 255, 255, 32), (150, 100), 70)
    assert surfaces_equal(a, b)


def test_fading_trajectory_matches_reference():
    rng = random.Random(4)
    points = [V(rng.uniform(-20, 320), rng.uniform(-20, 220)) for _ in range(60)]
    a, b = blank_surface(), blank_surface()
    rendering.draw_fading_trajectory(a, points, (255, 0, 0), decay_rate=0.05)
    reference_trajectory(b, points, (255, 0, 0), 0.05)
    assert surfaces_equal(a, b)


def test_drawing_helpers_handle_degenerate_input():
    surface = blank_surface()
    before = pygame.surfarray.array3d(surface).copy()
    rendering.draw_dashed_circle(surface, (255, 0, 0), (10, 10), 0)
    rendering.draw_dashed_line(surface, (255, 0, 0), V(5, 5), V(5, 5))
    rendering.draw_fading_trajectory(surface, [V(1, 1)], (255, 0, 0), 0.1)
    assert np.array_equal(before, pygame.surfarray.array3d(surface))
    rendering.draw_dashed_line(surface, (255, 0, 0), V(0, 0), V(100, 0), width=2)
    assert pygame.surfarray.array3d(surface)[2, 0, 0] == 255


def test_heatmap_and_direction_draw_only_active_cells(settings):
    surface = pygame.Surface((settings.SIM_WIDTH, settings.SIM_HEIGHT)).convert_alpha()
    surface.fill((0, 0, 0))
    intensity = np.zeros((settings.GRID_WIDTH, settings.GRID_HEIGHT))
    direction = np.zeros((settings.GRID_WIDTH, settings.GRID_HEIGHT, 2))
    intensity[3, 4] = 1.0
    direction[3, 4] = (1, 0)
    rendering.draw_heatmap(surface, intensity, "orange")
    rendering.draw_direction(surface, intensity, direction, 0.1)
    pixels = pygame.surfarray.array3d(surface)
    c = settings.CELL_SIZE
    assert tuple(pixels[3 * c + 1, 4 * c + 1]) == (255, 165, 0)
    assert pixels[: 3 * c].sum() == 0


def test_font_cache_reuses_fonts():
    assert rendering.get_font(12) is rendering.get_font(12)
    assert rendering.get_font(12) is not rendering.get_font(12, bold=True)


def test_button_toggles_and_calls_back():
    calls = []
    button = rendering.Button((0, 0, 50, 20), "B", lambda: calls.append(1), toggled=False)
    button.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(10, 10), button=1))
    button.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(100, 100), button=1))
    assert calls == [1] and button.toggled is True
    button.draw(blank_surface())
    assert rendering.Button.darken_color((10, 50, 100)) == (0, 20, 70)
