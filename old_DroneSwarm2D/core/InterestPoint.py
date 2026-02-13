# type: ignore
"""
interest_point.py

This module defines the CircleInterestPoint class, which represents an interest
point in the game rendered as a circle. The circle's color changes dynamically 
based on its health, transitioning from green (full health) to red (low health).
"""

# Standard and third-party libraries
import pygame
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from typing import Tuple, List, Any, Optional

# Project-specific imports
# from Drone import Drone
from .EnemyDrone import EnemyDrone
from .settings import *

# -----------------------------------------------------------------------------
# CircleInterestPoint Class
# -----------------------------------------------------------------------------
class CircleInterestPoint:
    """
    Represents an interest point displayed as a circle in the game.
    
    The circle changes color based on its current health: full health is green,
    and as health decreases the color shifts toward red.
    """

    def __init__(self, center: pygame.math.Vector2, internal_radius: float, external_radius: float,
                 color: Tuple[int, int, int] = (0, 255, 0), seed: Optional[int] = None) -> None:
        """
        Initialize the interest point as a circle.
        
        Args:
            center (pygame.math.Vector2): The center of the circle.
            internal_radius (float): The radius for the inner circle.
            external_radius (float): The radius for the outer circle (outline).
            color (Tuple[int, int, int]): Base color of the circle (default is green).
        """
        self.center = center
        self.internal_radius = internal_radius
        self.external_radius = external_radius
        self.health = INTEREST_POINT_INITIAL_HEALTH  # Initial health from settings
        self.base_color = color

    def draw(self, surface: pygame.Surface) -> None:
        """
        Draws the interest point on the provided Pygame surface.
        
        The circle is drawn with a semi-transparent fill and an outline. The fill
        and outline colors change based on the current health, and the health value 
        is rendered as text at the center.
        
        Args:
            surface (pygame.Surface): The Pygame surface on which to draw the circle.
        """
        # Calculate health ratio and update color accordingly
        health_ratio = self.health / INTEREST_POINT_INITIAL_HEALTH
        health_ratio = max(0, min(1, health_ratio))
        # Blend from green (full health) to red (no health)
        current_color = (int(255 * (1 - health_ratio)), int(255 * health_ratio), 0)
        
        # Calculate the diameter for the inner circle
        diameter = int(self.internal_radius * 2)
        
        # Create an alpha-enabled surface for a transparent fill
        alpha_surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        fill_color = (current_color[0], current_color[1], current_color[2], 50)
        
        # Draw the filled circle on the alpha surface
        pygame.draw.circle(alpha_surface, fill_color, (int(self.internal_radius), int(self.internal_radius)), int(self.internal_radius))
        
        # Blit the alpha surface onto the main surface, centering on the circle's center
        surface.blit(alpha_surface, (int(self.center.x - self.internal_radius), int(self.center.y - self.internal_radius)))
        
        # Draw the inner circle outline on the main surface
        pygame.draw.circle(surface, current_color, (int(self.center.x), int(self.center.y)), int(self.internal_radius), 1)
        
        # Draw the outer circle outline with a lighter version of the current color
        current_color_2 = (int(255 * (1 - health_ratio) / 3), int(255 * health_ratio / 3), 0)
        pygame.draw.circle(surface, current_color_2, (int(self.center.x), int(self.center.y)), self.external_radius, 1)
        
        # Render and display the health text at the center of the circle
        font = pygame.font.SysFont(FONT_FAMILY, 15)
        health_text = font.render(f"Health: {self.health}", True, (255, 255, 255), (0, 10, 0))
        text_rect = health_text.get_rect(center=(self.center.x, self.center.y + 50))
        surface.blit(health_text, text_rect)