"""
multi_simulation_trajectories.py

This script simulates multiple enemy drones with different behavior types.
For each simulation instance, the trajectories of drones starting at random border 
points and moving toward an interest point are computed. The resulting trajectories 
are plotted in subplots and the figure is saved as an image.
"""

# -----------------------------------------------------------------------------
# Environment Setup and Imports
# -----------------------------------------------------------------------------
import os
os.environ["SDL_AUDIODRIVER"] = "dummy"  # Use dummy audio driver to avoid audio errors

import sys
import random
import math
import matplotlib.pyplot as plt
import pygame
pygame.init()

from typing import Tuple

# Get current working directory and add the config directory to sys.path
current_dir: str = os.getcwd()
config_dir: str = os.path.abspath(os.path.join(current_dir, "./src/environment"))
if config_dir not in sys.path:
    sys.path.append(config_dir)

# Project-specific imports
from EnemyDrone import EnemyDrone

EnemyDrone.set_class_seed(42)  # Set a fixed seed for reproducibility

os.makedirs('./images', exist_ok=True)

# -----------------------------------------------------------------------------
# Simulation Constants
# -----------------------------------------------------------------------------
SIM_WIDTH = 800
SIM_HEIGHT = 600
SPEED = 2
REACH_THRESHOLD = 5    # Minimum distance to consider that the drone has reached the interest point
MAX_STEPS = 5000       # Maximum iterations to avoid infinite loops

# -----------------------------------------------------------------------------
# Interest Point and Helper Functions
# -----------------------------------------------------------------------------
class InterestPoint:
    """
    A simple class representing an interest point in the simulation.
    """
    def __init__(self, x: float, y: float) -> None:
        self.center = pygame.math.Vector2(x, y)

def random_border_point(width: int, height: int) -> pygame.math.Vector2:
    """
    Returns a random point on the border of a rectangle of dimensions width x height.
    
    Args:
        width (int): The width of the rectangle.
        height (int): The height of the rectangle.
    
    Returns:
        pygame.math.Vector2: A random point along one of the rectangle's borders.
    """
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        return pygame.math.Vector2(random.uniform(0, width), 0)
    elif side == "bottom":
        return pygame.math.Vector2(random.uniform(0, width), height)
    elif side == "left":
        return pygame.math.Vector2(0, random.uniform(0, height))
    else:  # right
        return pygame.math.Vector2(width, random.uniform(0, height))

# -----------------------------------------------------------------------------
# Simulation Setup
# -----------------------------------------------------------------------------
# Define a list of available drone behavior types.
behaviors = [
    "direct", "zigzag", "zigzag_damped", "zigzag_unstable", "zigzag_variable_period",
    "spiral", "spiral_bounce", "spiral_oscillatory", "alternating", "bounce_approach",
    "circle_wait_advance"
]

num_drones = len(behaviors)      # Number of enemy drones per simulation instance
num_instances = 1                # Number of simulation instances to generate

# Define the interest point at the center of the simulation area.
interest_point = InterestPoint(SIM_WIDTH / 2, SIM_HEIGHT / 2)

# -----------------------------------------------------------------------------
# Plot Appearance Settings
# -----------------------------------------------------------------------------
# Generate distinct colors for each drone trajectory using the HSV colormap.
color_map = plt.get_cmap("hsv", num_drones)
# Define a list of different linestyles to explore various plot styles.
linestyles = ['-', '--', '-.', ':']

# -----------------------------------------------------------------------------
# Main Simulation and Plotting
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Runs the simulation instances, computes trajectories for drones with different behaviors,
    and plots the results in subplots. The final figure is saved as an image file.
    """
    # Create a figure with one subplot per simulation instance.
    fig, axes = plt.subplots(num_instances, 1, figsize=(16, 6 * num_instances))
    if num_instances == 1:
        axes = [axes]

    # For each simulation instance:
    for inst in range(num_instances):
        # Randomly select 'num_drones' distinct behaviors.
        selected_behaviors = behaviors # random.sample(behaviors, num_drones)
        
        drones = []            # List to store drone instances.
        trajectories = []      # List to store each drone's trajectory.
        start_positions = []   # List to record each drone's start position.
        
        # Create enemy drones with the selected behaviors.
        for behavior in selected_behaviors:
            start = random_border_point(SIM_WIDTH, SIM_HEIGHT)
            start_positions.append((start.x, start.y))
            # Instantiate an EnemyDrone with the given behavior and random start position.
            drone = EnemyDrone(interest_point.center, position=(start.x, start.y), behavior_type=behavior)
            drones.append(drone)
            trajectories.append([(drone.pos.x, drone.pos.y)])
        
        # Simulate until all drones reach the interest point or MAX_STEPS is reached.
        steps = 0
        active = [True] * num_drones  # Track which drones are still active.
        while any(active) and steps < MAX_STEPS:
            for i, drone in enumerate(drones):
                if active[i]:
                    # If the drone hasn't reached the interest point, update its state.
                    if (drone.pos - interest_point.center).length() > REACH_THRESHOLD:
                        drone.update()
                        trajectories[i].append((drone.pos.x, drone.pos.y))
                    else:
                        active[i] = False
            steps += 1
        
        # Plot trajectories for this simulation instance.
        ax = axes[inst]
        for i, traj in enumerate(trajectories):
            xs, ys = zip(*traj)
            # Rotate through the available linestyles.
            ls = linestyles[i % len(linestyles)]
            ax.plot(xs, ys, marker='o', markersize=2, linewidth=1, 
                    color=color_map(i), linestyle=ls, label=selected_behaviors[i])
            # Highlight the starting position with a red diamond with a black edge.
            ax.scatter(xs[0], ys[0], color='red', s=300, marker='8', edgecolors='black', zorder=5)
        
        # Highlight the interest point with a blue square with a black edge.
        ax.scatter(interest_point.center.x, interest_point.center.y, color='blue', s=300, 
                   marker='s', edgecolors='black', zorder=15, label='Interest Point')
        ax.set_title("Enemy Trajectories", fontsize=16, fontweight='bold')
        ax.set_xlim(0, SIM_WIDTH)
        ax.set_ylim(0, SIM_HEIGHT)
        ax.invert_yaxis()  # Invert y-axis to match pygame's coordinate system.
        
        # Configurar tamanho da fonte dos números dos eixos
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Configurar legenda fora do gráfico, embaixo, com 5 colunas (3 linhas)
        ax.legend(fontsize=16, bbox_to_anchor=(0.5, -0.15), loc='upper center', 
                  ncol=5, columnspacing=1.5, handletextpad=0.5)
    
    plt.tight_layout()
    # Ajustar espaçamento para acomodar a legenda externa (3 linhas)
    plt.subplots_adjust(bottom=0.3)
    output_path = "./images/multi_simulation_trajectories.png"
    plt.savefig(output_path, dpi=600)  # Adicionado DPI maior para melhor qualidade
    plt.close()
    print(f"Simulation trajectories saved as: {output_path}")

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()