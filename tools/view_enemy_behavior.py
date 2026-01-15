"""
simulate_trajectories.py

This script simulates the trajectories of an EnemyDrone using different behavior types.
It creates subplots for each behavior showing the drone's path from the start to the interest point.
The resulting figure is saved as an image.
"""

# -----------------------------------------------------------------------------
# Environment Setup and Imports
# -----------------------------------------------------------------------------
import os
import sys
import random
import math
import matplotlib.pyplot as plt
import pygame

# Use dummy audio driver to avoid audio errors
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Initialize pygame if not already initialized
if not pygame.get_init():
    pygame.init()

# Get current working directory and add configuration directory to the system path.
current_dir: str = os.getcwd()
config_dir: str = os.path.abspath(os.path.join(current_dir, "./src/environment"))
if config_dir not in sys.path:
    sys.path.append(config_dir)

# Project-specific imports
from EnemyDrone import EnemyDrone

os.makedirs('./images', exist_ok=True)

# -----------------------------------------------------------------------------
# Simulation Constants
# -----------------------------------------------------------------------------
SIM_WIDTH = 800
SIM_HEIGHT = 600
SPEED = 2
REACH_THRESHOLD = 5   # Minimum distance to consider that the drone has reached the interest point.
MAX_STEPS = 5000      # Maximum iterations to avoid infinite loops.

# -----------------------------------------------------------------------------
# InterestPoint Class
# -----------------------------------------------------------------------------
class InterestPoint:
    """
    A simple class representing an interest point in the simulation.
    """
    def __init__(self, x: float, y: float) -> None:
        self.center = pygame.math.Vector2(x, y)

# -----------------------------------------------------------------------------
# Main Simulation and Plotting Function
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Simulates the trajectories for different enemy drone behaviors and creates subplots.
    
    For each behavior, the EnemyDrone is updated until it reaches near the interest point,
    and its trajectory is recorded. A figure with subplots for all behaviors is saved as an image.
    """
    # List of behaviors to simulate.
    behaviors = [
        "direct", "zigzag", "zigzag_damped", "zigzag_unstable", "zigzag_variable_period",
        "spiral", "spiral_bounce", "spiral_oscillatory", "alternating", "bounce_approach",
        "circle_wait_advance"
    ]
    num_behaviors = len(behaviors)
    
    # Define the start position and interest point.
    start_pos = pygame.math.Vector2(0, SIM_HEIGHT / 2)
    interest_point = InterestPoint(SIM_WIDTH / 2, SIM_HEIGHT / 2)
    
    # Dictionary to store trajectories for each behavior.
    trajectories = {}
    
    # Simulate the trajectory for each behavior.
    for behavior in behaviors:
        # Create an enemy drone with the specified behavior.
        drone = EnemyDrone(interest_point.center, position=(0, SIM_HEIGHT // 2), behavior_type=behavior)
        traj = []
        steps = 0
        
        # Simulate until the drone reaches close to the interest point or maximum steps reached.
        while (drone.pos - interest_point.center).length() > REACH_THRESHOLD and steps < MAX_STEPS:
            traj.append((drone.pos.x, drone.pos.y))
            drone.update()
            steps += 1
        
        # Record the final position as well.
        traj.append((drone.pos.x, drone.pos.y))
        trajectories[behavior] = traj

    # -----------------------------------------------------------------------------
    # Plotting the Trajectories
    # -----------------------------------------------------------------------------
    cols = 4
    rows = math.ceil(num_behaviors / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, behavior in enumerate(behaviors):
        ax = axes[i]
        traj = trajectories[behavior]
        xs, ys = zip(*traj)
        ax.plot(xs, ys, marker='o', markersize=2, linewidth=1)
        ax.set_title(behavior)
        # Plot the interest point and starting point.
        ax.scatter([interest_point.center.x], [interest_point.center.y], color='red', label='Interest Point')
        ax.scatter([start_pos.x], [start_pos.y], color='green', label='Start')
        ax.set_xlim(0, SIM_WIDTH)
        ax.set_ylim(0, SIM_HEIGHT)
        ax.invert_yaxis()  # Adjust to pygame coordinate system.
        ax.legend(fontsize='small')
    
    # Remove empty subplots if any.
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    # Save the figure with all subplots.
    output_path = "./images/trajectories.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Trajectories figure saved as: {output_path}")

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()