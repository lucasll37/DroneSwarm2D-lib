# type: ignore
"""
utils.py

This module provides various utility functions for simulation visualization,
matrix processing, coordinate conversion, and more.
"""

# Standard libraries
import math
import random
import io

# Third-party libraries
import numpy as np
import pygame
import cairosvg
import matplotlib.pyplot as plt

# Matplotlib utilities
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, used for 3D plotting
from matplotlib.backends.backend_agg import FigureCanvasAgg
# from numba import njit
from typing import Any, Tuple, Dict, Optional
from functools import lru_cache

# Project-specific imports
from .settings import (
    SIM_WIDTH, SIM_HEIGHT,
    GRID_WIDTH, GRID_HEIGHT,
    CELL_SIZE,
    INTEREST_POINT_CENTER,
    FRIEND_SPEED,
    ENEMY_SPEED
)


@lru_cache(maxsize=1)
def set_tensorflow():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    return tf

# -----------------------------------------------------------------------------
# Drawing Utilities
# -----------------------------------------------------------------------------
def draw_dashed_circle(surface: pygame.Surface,
                       color: Tuple[int, int, int],
                       center: Tuple[int, int],
                       radius: int,
                       dash_length: int = 5,
                       space_length: int = 5,
                       width: int = 1) -> None:
    """
    Draws a dashed circle on the provided Pygame surface.

    Args:
        surface (pygame.Surface): The surface to draw on.
        color (Tuple[int, int, int]): Color of the circle.
        center (Tuple[int, int]): (x, y) center of the circle.
        radius (int): Radius of the circle.
        dash_length (int): Length of each dash.
        space_length (int): Space between dashes.
        width (int): Line width.
    """
    if radius <= 0:
        return
    
    circumference = 2 * math.pi * radius
    num_dashes = int(circumference / (dash_length + space_length))
    angle_between = 2 * math.pi / num_dashes

    for i in range(num_dashes):
        start_angle = i * angle_between
        dash_angle = dash_length / radius  # Angle corresponding to dash length
        end_angle = start_angle + dash_angle

        start_pos = (center[0] + radius * math.cos(start_angle),
                     center[1] + radius * math.sin(start_angle))
        end_pos = (center[0] + radius * math.cos(end_angle),
                   center[1] + radius * math.sin(end_angle))
        pygame.draw.line(surface, color, start_pos, end_pos, width)
        
        
def draw_dashed_line(surface: pygame.Surface, color: Tuple[int, int, int],
                     start_pos: pygame.math.Vector2, end_pos: pygame.math.Vector2,
                     width: int = 1, dash_length: int = 5, space_length: int = 5) -> None:
    """
    Draws a dashed line from start_pos to end_pos on the given surface.
    """
    start = pygame.math.Vector2(start_pos)
    end = pygame.math.Vector2(end_pos)
    displacement = end - start
    length = displacement.length()
    if length == 0:
        return
    dash_vector = displacement.normalize() * dash_length
    num_dashes = int(length / (dash_length + space_length))
    for i in range(num_dashes):
        dash_start = start + (dash_length + space_length) * i * displacement.normalize()
        dash_end = dash_start + dash_vector
        pygame.draw.line(surface, color, dash_start, dash_end, width)

# -----------------------------------------------------------------------------
# Gaussian Bump Kernel Functions
# -----------------------------------------------------------------------------
# @njit
def symmetrical_flat_topped_gaussian_10x10(value: float, sigma: float, flat_radius: float) -> np.ndarray:
    """
    Creates a 10x10 Gaussian bump with a flat top.
    
    The continuous coordinates range from -4.5 to +4.5 to ensure symmetry.
    The bump has a flat top (value = 1) within the given flat_radius.

    Args:
        value (float): Peak value to scale at the center.
        sigma (float): Standard deviation for the Gaussian portion.
        flat_radius (float): Radius (in continuous coordinates) where the value is constant.

    Returns:
        np.ndarray: A 10x10 array representing the bump.
    """
    kernel_size = 10
    # Create a coordinate grid manually instead of using meshgrid
    x = np.linspace(-4.5, 4.5, kernel_size)
    
    # Initialize arrays for the bump
    bump = np.empty((kernel_size, kernel_size), dtype=np.float64)
    
    # Calculate the bump values directly
    for i in range(kernel_size):
        y = x[i]  # y-coordinate
        for j in range(kernel_size):
            xi = x[j]  # x-coordinate
            # Calculate distance from center
            r = np.sqrt(xi**2 + y**2)
            
            # Apply flat top or Gaussian falloff
            if r < flat_radius:
                bump[i, j] = 1.0
            else:
                bump[i, j] = np.exp(-0.5 * (r / sigma)**2)
    
    return value * bump

def smooth_matrix_with_kernel_10x10(matrix: np.ndarray,
                                    direction: np.ndarray,
                                    sigma: float = 1.0,
                                    flat_radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a 10x10 Gaussian bump (with a flat top) to each positive value in the matrix.
    The bump is centered at the pixel (i, j) without changing the original dimensions.
    
    For each peak, if the bump's value is greater than the current value in the region,
    both the intensity and the corresponding direction vector are updated.

    Args:
        matrix (np.ndarray): 2D array with values between 0 and 1.
        direction (np.ndarray): Array with same dimensions as matrix plus one extra dimension
                                for the direction vector (e.g., shape (n_rows, n_cols, 2)).
        sigma (float): Standard deviation for the Gaussian portion.
        flat_radius (float): Radius (in continuous coordinates) where the value is constant.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - result: Updated matrix with the applied bumps.
            - result_direction: Updated direction array.
    """
    result = np.copy(matrix)
    result_direction = np.copy(direction)
    peaks = np.argwhere(matrix > 0)
    kernel_size = 10
    anchor = kernel_size // 2
    
    # Otimização: verifique se temos muitos picos para pré-calcular
    if len(peaks) > 50:  # Limite arbitrário, ajuste conforme necessário
        # Calcule as dimensões da matriz de picos para pré-alocação
        peak_values = {}
        for (i, j) in peaks:
            value = matrix[i, j]
            if value not in peak_values:
                # Calcule o bump apenas para valores únicos
                peak_values[value] = symmetrical_flat_topped_gaussian_10x10(value, sigma, flat_radius)
    else:
        peak_values = None

    for (i, j) in peaks:
        value = matrix[i, j]
        
        # Use o bump pré-calculado, se disponível
        if peak_values is not None:
            bump = peak_values[value]
        else:
            bump = symmetrical_flat_topped_gaussian_10x10(value, sigma, flat_radius)
            
        bump_direction = direction[i, j]  # Vetor de direção no pico

        # Calcule índices mais eficientemente
        i_start = max(0, i - anchor)
        j_start = max(0, j - anchor)
        i_end = min(matrix.shape[0], i + anchor)
        j_end = min(matrix.shape[1], j + anchor)

        # Calcule índices do bump
        bump_i_start = max(0, anchor - (i - i_start))
        bump_j_start = max(0, anchor - (j - j_start))
        bump_i_end = bump_i_start + (i_end - i_start)
        bump_j_end = bump_j_start + (j_end - j_start)

        # Obtenha visualizações em vez de cópias quando possível
        region = result[i_start:i_end, j_start:j_end]
        bump_region = bump[bump_i_start:bump_i_end, bump_j_start:bump_j_end]

        # Crie uma máscara para pixels onde o valor do bump é maior que o valor atual
        mask = bump_region > region
        
        # Use operações in-place quando possível
        np.maximum(region, bump_region, out=region)

        # Atualize os vetores de direção onde o bump aumentou o valor
        for dim in range(direction.shape[-1]):
            for di in range(i_end - i_start):
                for dj in range(j_end - j_start):
                    if mask[di, dj]:
                        result_direction[i_start + di, j_start + dj, dim] = bump_direction[dim]

    return result, result_direction
# def symmetrical_flat_topped_gaussian_10x10(value: float, sigma: float, flat_radius: float) -> np.ndarray:
#     """
#     Creates a 10x10 Gaussian bump with a flat top.
    
#     The continuous coordinates range from -4.5 to +4.5 to ensure symmetry.
#     The bump has a flat top (value = 1) within the given flat_radius.

#     Args:
#         value (float): Peak value to scale at the center.
#         sigma (float): Standard deviation for the Gaussian portion.
#         flat_radius (float): Radius (in continuous coordinates) where the value is constant.

#     Returns:
#         np.ndarray: A 10x10 array representing the bump.
#     """
#     kernel_size = 10
#     x = np.linspace(-4.5, 4.5, kernel_size)
#     xx, yy = np.meshgrid(x, x)
#     r = np.sqrt(xx**2 + yy**2)

#     # Gaussian bump
#     bump = np.exp(-0.5 * (r / sigma)**2)
    
#     # Apply flat top where distance is less than flat_radius
#     bump[r < flat_radius] = 1.0
    
#     return value * bump

# def smooth_matrix_with_kernel_10x10(matrix: np.ndarray,
#                                     direction: np.ndarray,
#                                     sigma: float = 1.0,
#                                     flat_radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Applies a 10x10 Gaussian bump (with a flat top) to each positive value in the matrix.
#     The bump is centered at the pixel (i, j) without changing the original dimensions.
    
#     For each peak, if the bump's value is greater than the current value in the region,
#     both the intensity and the corresponding direction vector are updated.

#     Args:
#         matrix (np.ndarray): 2D array with values between 0 and 1.
#         direction (np.ndarray): Array with same dimensions as matrix plus one extra dimension
#                                 for the direction vector (e.g., shape (n_rows, n_cols, 2)).
#         sigma (float): Standard deviation for the Gaussian portion.
#         flat_radius (float): Flat top radius for the bump.

#     Returns:
#         Tuple[np.ndarray, np.ndarray]: 
#             - result: Updated matrix with the applied bumps.
#             - result_direction: Updated direction array.
#     """
#     result = np.copy(matrix)
#     result_direction = np.copy(direction)
#     peaks = np.argwhere(matrix > 0)
#     kernel_size = 10
#     anchor = kernel_size // 2

#     for (i, j) in peaks:
#         value = matrix[i, j]
#         bump = symmetrical_flat_topped_gaussian_10x10(value, sigma, flat_radius)
#         bump_direction = direction[i, j]  # Direction vector at the peak

#         i_start = i - anchor
#         i_end   = i_start + kernel_size
#         j_start = j - anchor
#         j_end   = j_start + kernel_size

#         bump_i_start = 0
#         bump_j_start = 0
#         bump_i_end = kernel_size
#         bump_j_end = kernel_size

#         # Adjust indices if the region goes beyond the matrix borders
#         if i_start < 0:
#             bump_i_start = -i_start
#             i_start = 0
#         if j_start < 0:
#             bump_j_start = -j_start
#             j_start = 0
#         if i_end > matrix.shape[0]:
#             bump_i_end -= (i_end - matrix.shape[0])
#             i_end = matrix.shape[0]
#         if j_end > matrix.shape[1]:
#             bump_j_end -= (j_end - matrix.shape[1])
#             j_end = matrix.shape[1]

#         region = result[i_start:i_end, j_start:j_end]
#         bump_region = bump[bump_i_start:bump_i_end, bump_j_start:bump_j_end]

#         # Create a mask for pixels where the bump value is greater than the current value
#         mask = bump_region > region
#         region = np.maximum(region, bump_region)
#         result[i_start:i_end, j_start:j_end] = region

#         # Update the direction vectors where the bump increased the value
#         region_direction = result_direction[i_start:i_end, j_start:j_end]
#         region_direction[mask] = bump_direction
#         result_direction[i_start:i_end, j_start:j_end] = region_direction

#     return result, result_direction

# -----------------------------------------------------------------------------
# Coordinate Conversion Functions
# -----------------------------------------------------------------------------
def sim_to_geo(pos_x: float, pos_y: float) -> Tuple[float, float]:
    """
    Converts simulation coordinates to geographic coordinates.

    Args:
        pos_x (float): X position in simulation.
        pos_y (float): Y position in simulation.
        
    Returns:
        Tuple[float, float]: (Longitude, Latitude) coordinates.
    """
    lon_left, lat_top = GEO_TOP_LEFT
    lon_right, lat_bottom = GEO_BOTTOM_RIGHT
    
    lon = lon_left + (pos_x / SIM_WIDTH) * (lon_right - lon_left)
    lat = lat_top + (pos_y / SIM_HEIGHT) * (lat_bottom - lat_top)
    
    return lon, lat

# -----------------------------------------------------------------------------
# SVG and Image Utilities
# -----------------------------------------------------------------------------
def load_svg_as_surface(svg_path: str) -> pygame.Surface:
    """
    Converts an SVG file to a Pygame Surface with alpha support.

    Args:
        svg_path (str): Path to the SVG file.

    Returns:
        pygame.Surface: Converted image as a Pygame surface.
    """
    # Convert SVG to PNG in memory
    png_data = cairosvg.svg2png(url=svg_path)
    image_data = io.BytesIO(png_data)
    surface = pygame.image.load(image_data).convert_alpha()
    return surface

# -----------------------------------------------------------------------------
# Grid and Positioning Utilities
# -----------------------------------------------------------------------------
def pos_to_cell(pos: pygame.math.Vector2, cell_size: int = CELL_SIZE,
                grid_width: int = GRID_WIDTH, grid_height:int = GRID_HEIGHT) -> Tuple[int, int]:
    """
    Converts a position (Vector2) to grid cell coordinates.

    Args:
        pos (pygame.math.Vector2): The position vector.

    Returns:
        Tuple[int, int]: Cell coordinates (x, y).
    """
    x: int = int(min(pos.x // cell_size, grid_width - 1))
    y: int = int(min(pos.y // cell_size, grid_height - 1))
    
    return (x, y)

# -----------------------------------------------------------------------------
# Interception and Pursuit Calculations
# -----------------------------------------------------------------------------
def intercept_direction(chaser_pos: pygame.math.Vector2,
                        chaser_speed: float,
                        target_pos: pygame.math.Vector2,
                        target_vel: pygame.math.Vector2) -> pygame.math.Vector2:
    """
    Calculates the optimal interception direction for a chaser to intercept a target.
    
    The method solves for the intercept time 't' from:
        |r + target_vel * t| = chaser_speed * t
    where r = target_pos - chaser_pos.
    
    If no valid solution exists (negative discriminant or non-positive t),
    the function returns the normalized vector from chaser to target.

    Args:
        chaser_pos (pygame.math.Vector2): Chaser's position.
        chaser_speed (float): Chaser's constant speed.
        target_pos (pygame.math.Vector2): Target's position.
        target_vel (pygame.math.Vector2): Target's velocity.

    Returns:
        pygame.math.Vector2: Unit vector in the direction of interception.
    """
    r = target_pos - chaser_pos
    a = target_vel.dot(target_vel) - chaser_speed ** 2
    b = 2 * r.dot(target_vel)
    c = r.dot(r)

    t = 0.0  # Fallback time

    if abs(a) < 1e-6:
        if abs(b) > 1e-6:
            t = -c / b
    else:
        discriminant = b ** 2 - 4 * a * c
        if discriminant >= 0:
            sqrt_disc = math.sqrt(discriminant)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)
            t_candidates = [t_val for t_val in (t1, t2) if t_val > 0]
            if t_candidates:
                t = min(t_candidates)

    if t <= 0:
        direction = target_pos - chaser_pos
    else:
        intercept_point = target_pos + target_vel * t
        direction = intercept_point - chaser_pos
        
    if direction.length() > 0:
        vel = direction.normalize() * chaser_speed
    else:
        vel = pygame.math.Vector2(0, 0)

    return vel
    
# -----------------------------------------------------------------------------
# Interception and Pursuit Calculations
# -----------------------------------------------------------------------------
def can_intercept(chaser_pos: pygame.math.Vector2,
                  chaser_speed: float,
                  target_pos: pygame.math.Vector2,
                  target_vel: pygame.math.Vector2,
                  point_of_interest = None) -> bool:
    """
    Determina se é possível que o perseguidor intercepte o alvo antes que o alvo atinja o ponto de interesse.
    
    Para isso, a função resolve a equação:
         ||r + target_vel * t|| = chaser_speed * t,
    onde r = target_pos - chaser_pos.
    
    Retorna True se existir um tempo t > 0 para a interceptação e, além disso, se o tempo de interceptação
    for menor que o tempo que o alvo levaria para atingir o ponto de interesse.
    Caso contrário, retorna False.
    """
    
    if point_of_interest is None:
        point_of_interest = INTEREST_POINT_CENTER
        
    # Calcula os coeficientes da equação quadrática
    r = target_pos - chaser_pos
    a = target_vel.dot(target_vel) - chaser_speed ** 2
    b = 2 * r.dot(target_vel)
    c = r.dot(r)
    
    # Caso linear
    if abs(a) < 1e-6:
        if abs(b) > 1e-6:
            t = -c / b
        else:
            # Se b também for zero, então o perseguidor já está no alvo.
            return r.length() == 0
    else:
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return False  # Não há solução real para t
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)
        t_candidates = [t_val for t_val in (t1, t2) if t_val > 0]
        if not t_candidates:
            return False
        t = min(t_candidates)

    # Se o tempo de interceptação for menor ou igual a zero, não é possível interceptar
    if t <= 0:
        return False
    
    # Calcula o tempo que o alvo levaria para atingir o ponto de interesse
    # Se target_vel for zero, não há movimento (não é possível interceptar em movimento)
    if target_vel.length() == 0:
        return False
    
    t_PI = (point_of_interest - target_pos).length() / target_vel.length()
    
    # O perseguidor consegue interceptar se o tempo de interceptação for menor que o tempo para o alvo atingir o PI    
    return t < t_PI

# -----------------------------------------------------------------------------
# Plotting Utilities
# -----------------------------------------------------------------------------
def plot_individual_states_matplotlib(state: Dict) -> None:
    """
    Generates a 3D plot of detection states for enemy and friend drones.
    
    Displays two subplots ("Enemy Detection" and "Friend Detection") with a color legend
    that maps colors to angles (in π radians). Also plots the drone's position as a red
    vertical line.

    Args:
        state (dict): Dictionary containing drone state information.
    """
    fig = plt.figure(figsize=(8, 6))
    cmap = plt.cm.hsv  # Color palette based on angles

    # Iterate over the two detection views
    for idx, plot_view in enumerate(["Enemy Detection"]):
    # for idx, plot_view in enumerate(["Enemy Detection", "Friend Detection"]):
        # ax = fig.add_subplot(2, 1, idx + 1, projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect((1, 1, 0.4))

        # Create meshgrid for grid dimensions
        x = np.linspace(CELL_SIZE / 2, SIM_WIDTH - CELL_SIZE / 2, GRID_WIDTH)
        y = np.linspace(CELL_SIZE / 2, SIM_HEIGHT - CELL_SIZE / 2, GRID_HEIGHT)
        X, Y = np.meshgrid(x, y)

        if plot_view == "Enemy Detection":
            to_plot = state['enemy_intensity']
            direction = state['enemy_direction']
        elif plot_view == "Friend Detection":
            to_plot = state['friend_intensity']
            direction = state['friend_direction']

        # Smooth the matrix and get resulting directions
        Z_smoothed, result_direction = smooth_matrix_with_kernel_10x10(
            to_plot, direction, sigma=2, flat_radius=1
        )

        # Adjust shape if necessary
        if Z_smoothed.shape != X.shape:
            Z_smoothed = Z_smoothed.T
            result_direction = result_direction.transpose(1, 0, 2)

        # Calculate angles from direction vectors
        dir_x = result_direction[..., 0]
        dir_y = result_direction[..., 1]
        angle = np.arctan2(dir_y, dir_x)
        norm_angle = (angle + math.pi) / (2 * math.pi)
        facecolors = cmap(norm_angle)
        facecolors[Z_smoothed < PLOT_THRESHOLD] = [1, 1, 1, 1]

        ax.plot_surface(X, Y, Z_smoothed, facecolors=facecolors,
                        linewidth=0, antialiased=True, shade=False)
        ax.contourf(X, Y, Z_smoothed, zdir='x', offset=ax.get_xlim()[0], cmap="Greys")
        ax.contourf(X, Y, Z_smoothed, zdir='y', offset=ax.get_ylim()[0], cmap="Greys")
        ax.plot_wireframe(X, Y, Z_smoothed, color='black', linewidth=0.2, rstride=1, cstride=1)

        # Plot drone's position as a red vertical line (in XY projection)
        # ax.plot([state['pos'][0], state['pos'][0]],
        #         [SIM_HEIGHT, 0],
        #         [0, 0], color='black', linewidth=4, label='Drone Position')
        ax.plot([state['pos'][0], state['pos'][0]],
                [state['pos'][1], state['pos'][1]],
                [0, 1], color='black', linewidth=4, zorder=10)

        x_offset = ax.get_xlim()[0]
        ax.plot([x_offset, x_offset],
                [state['pos'][1], state['pos'][1]],
                [0, 1], color='black', linewidth=4, zorder=10)

        y_offset = ax.get_ylim()[0]
        ax.plot([state['pos'][1], state['pos'][1]],
                [y_offset, y_offset],
                [0, 1], color='black', linewidth=4, zorder=10)

        # ax.set_title(plot_view, fontsize=12)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Recency', fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, SIM_WIDTH)
        ax.set_ylim(SIM_HEIGHT, 0)
        ax.set_zlim(0, 1.5)

    fig.suptitle("Detection and Position Plot", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.75, 1])

    norm = plt.Normalize(vmin=-math.pi, vmax=math.pi)
    mappable_for_colorbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable_for_colorbar.set_array([])
    cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.7])
    cbar = fig.colorbar(mappable_for_colorbar, cax=cbar_ax, label='Angle (rad)')
    cbar.set_ticks([-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi])
    cbar.set_ticklabels(['-π', '-π/2', '0', '+π/2', '+π'])
    plt.show()

# -----------------------------------------------------------------------------
# Sparse Matrix Generation
# -----------------------------------------------------------------------------
def generate_sparse_matrix(shape: Tuple[int, int] = (GRID_WIDTH, GRID_HEIGHT),
                           max_nonzero: int = 20, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a sparse matrix of the given shape with up to 'max_nonzero' nonzero elements.
    
    Nonzero values are sampled from a normal distribution (mean=1, std=1) and clipped to [0, 1].
    For each nonzero cell, a random normalized direction vector is generated.

    Args:
        shape (Tuple[int, int]): The shape of the intensity matrix.
        max_nonzero (int): Maximum number of nonzero elements.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - matrix: Array containing intensity values.
            - direction: Array of shape (rows, cols, 2) with corresponding direction vectors.
                         Cells with zero intensity have a (0, 0) vector.
    """
    matrix = np.zeros(shape)
    direction = np.zeros((shape[0], shape[1], 2), dtype=float)
    total_cells = shape[0] * shape[1]
    nonzero_count = min(random.randint(0, max_nonzero), total_cells)
    chosen_indices = random.sample(range(total_cells), nonzero_count)
    
    for idx in chosen_indices:
        i = idx // shape[1]
        j = idx % shape[1]
        value = np.random.normal(1, 1)
        matrix[i, j] = np.clip(value, 0, 1)
        
        # Generate a random normalized direction vector
        vec = pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        if vec.length() > 0:
            vec = vec.normalize()
        else:
            vec = pygame.math.Vector2(0, 0)
        direction[i, j] = (vec.x, vec.y)
    
    return matrix, direction


import os
import re
import sys
from typing import Tuple

def load_best_model(directory: str, pattern: str, custom_objects=None):
# def load_best_model(directory: str, pattern: str, custom_objects=None) -> Tuple[tf.keras.Model, Tuple[int, int]]:
    """
    Loads the best model from the given directory by selecting the file with the lowest validation loss,
    and extracts the size from the filename in a tuple (width, height).

    Args:
        directory (str): Directory containing saved model files.
        pattern (str): Regex pattern to extract the validation loss value from the filename.
        custom_objects (dict, optional): Custom objects to be passed to load_model.

    Returns:
        Tuple[tf.keras.Model, Tuple[int, int]]: The loaded model and a tuple with the size (width, height).

    Raises:
        FileNotFoundError: If no model file is found in the specified directory.
    """
    tf = set_tensorflow()
    best_file: str = ""
    min_val_metric_loss: float = float("inf")

    # Iterate over model files to find the one with the lowest val_metric_loss
    for filename in os.listdir(directory):
        if filename.endswith(".keras"):
            match = re.search(pattern, filename)
            if match:
                val_metric_loss: float = float(match.group(1))
                if val_metric_loss < min_val_metric_loss:
                    min_val_metric_loss = val_metric_loss
                    best_file = filename

    if not best_file:
        # raise FileNotFoundError(f"No model files found in the directory: {directory}")
        print(f"No model files found in the directory: {directory}")
        return None

    model_path: str = os.path.join(directory, best_file)
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"Error loading the model: {e}")
        # sys.exit(1)
        return None

    print(f"\n\nLoaded model: {best_file} with val_metric_loss={min_val_metric_loss:.4f}\n\n")

    return model



import numpy as np
import pygame
from .utils import pos_to_cell, can_intercept

 
def get_friends_hold(state, friend_activation_threshold_position: float = 0.7,
                    enemy_activation_threshold_position: float = 0.4):

    # Extração e preparação do estado
    pos = np.squeeze(state['pos'])
    pos = pygame.math.Vector2(pos[0], pos[1])
    friend_intensity = np.squeeze(state['friend_intensity'])
    enemy_intensity = np.squeeze(state['enemy_intensity'])
    enemy_direction = np.squeeze(state['enemy_direction'])
    
    enemy_targets = []
    # Identifica células da matriz de inimigos com intensidade acima do limiar
    for cell, intensity in np.ndenumerate(enemy_intensity):
        if intensity < enemy_activation_threshold_position:
            continue
        
        target_pos = pygame.math.Vector2((cell[0] + 0.5) * CELL_SIZE,
                                        (cell[1] + 0.5) * CELL_SIZE)
        distance_to_interest = target_pos.distance_to(INTEREST_POINT_CENTER)
        
        enemy_targets.append((cell, target_pos, distance_to_interest))
        
        
    # Obtém a célula correspondente à posição do drone self
    my_cell = pos_to_cell(pos)
    my_cell_center = pygame.math.Vector2((my_cell[0] + 0.5) * CELL_SIZE,
                                        (my_cell[1] + 0.5) * CELL_SIZE)
    
    # Obtém os candidatos amigos a partir da matriz friend_intensity.
    # Cada candidato é identificado pela célula em que há uma detecção ativa.
    friend_candidates = []
    for cell, intensity in np.ndenumerate(friend_intensity):
        if intensity >= friend_activation_threshold_position:
            candidate_pos = pygame.math.Vector2((cell[0] + 0.5) * CELL_SIZE,
                                                (cell[1] + 0.5) * CELL_SIZE)
            friend_candidates.append((cell, candidate_pos))
            
    # Inclui a própria célula self se não houver detecção ativa (para diferenciá-lo)
    if not any(cell == my_cell for cell, pos_candidate in friend_candidates):
        friend_candidates.append((my_cell, my_cell_center))
            
    if not enemy_targets:
        # Se nenhum alvo foi atribuído a self, mantém a posição.
        friends_hold = [(cell, candidate_pos) for (cell, candidate_pos) in friend_candidates]# if 1.1 * INITIAL_DISTANCE - candidate_pos.distance_to(INTEREST_POINT_CENTER) > 0]
        
        return friends_hold
    
    # Ordena os alvos pelo quão próximos estão do ponto de interesse
    enemy_targets.sort(key=lambda t: t[2])
    
    engagement_assignment = {}
    assigned_friend_cells = set()
    
    # Para cada alvo inimigo, em ordem, atribui o candidato amigo mais próximo que ainda não foi designado.
    for cell, target_pos, _ in enemy_targets:
        sorted_candidates = sorted(friend_candidates, key=lambda x: x[1].distance_to(target_pos))
        enemy_dir_vec = pygame.math.Vector2(enemy_direction[cell][0], enemy_direction[cell][1])  * ENEMY_SPEED
        
        for candidate in sorted_candidates:
            candidate_cell, candidate_pos = candidate
            
            if candidate_cell not in assigned_friend_cells and \
            can_intercept(candidate_pos, FRIEND_SPEED, target_pos, enemy_dir_vec, INTEREST_POINT_CENTER):
                
                engagement_assignment[tuple(target_pos)] = candidate_cell
                assigned_friend_cells.add(candidate_cell)
                break  # Avança para o próximo alvo
            
    
    # Se nenhum alvo foi atribuído a self, mantém a posição.
    friends_hold = [(cell, candidate_pos) for (cell, candidate_pos) in friend_candidates
        if cell not in assigned_friend_cells] # and 1.1 * INITIAL_DISTANCE - candidate_pos.distance_to(INTEREST_POINT_CENTER) > 0]
    
    return friends_hold