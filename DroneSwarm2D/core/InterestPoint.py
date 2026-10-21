"""The interest point: the asset the swarm defends.

PT-BR: O ponto de interesse: o alvo que o enxame defende.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pygame

from .rendering import get_font
from .settings import INTEREST_POINT_INITIAL_HEALTH


class CircleInterestPoint:
    """Circular interest point with health.

    An enemy that reaches ``INTEREST_POINT_ATTACK_RANGE`` of the centre deals
    ``INTEREST_POINT_DAMAGE`` to it. The colour goes from green (full health) to
    red (no health).

    PT-BR: Ponto de interesse circular com "vida". Cada inimigo que chega ao
    centro causa ``INTEREST_POINT_DAMAGE`` de dano; a cor vai de verde a vermelho.

    Args:
        center: Centre in pixels (copied, so the global ``CENTER`` is never mutated).
        internal_radius: Radius of the protected area (``INTERNAL_RADIUS``).
        external_radius: Maximum distance friendly drones may fly (``EXTERNAL_RADIUS``).
        color: Base colour (RGB).
        seed: Unused; kept for API compatibility.

    Attributes:
        health: Remaining health (starts at ``INTEREST_POINT_INITIAL_HEALTH``).
    """

    def __init__(
        self,
        center: pygame.math.Vector2,
        internal_radius: float,
        external_radius: float,
        color: Tuple[int, int, int] = (0, 255, 0),
        seed: Optional[int] = None,
    ) -> None:
        self.center: pygame.math.Vector2 = pygame.math.Vector2(center)
        self.internal_radius: float = internal_radius
        self.external_radius: float = external_radius
        self.health: int = INTEREST_POINT_INITIAL_HEALTH
        self.base_color: Tuple[int, int, int] = color

    @property
    def health_ratio(self) -> float:
        """Remaining health as a fraction in ``[0, 1]``."""
        return max(0.0, min(1.0, self.health / INTEREST_POINT_INITIAL_HEALTH))

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the protected area, the external radius and the health label.

        PT-BR: Desenha a área protegida, o raio externo e o rótulo de vida.
        """
        ratio = self.health_ratio
        current_color = (int(255 * (1 - ratio)), int(255 * ratio), 0)
        cx, cy = int(self.center.x), int(self.center.y)
        radius = int(self.internal_radius)

        diameter = int(self.internal_radius * 2)
        fill = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        pygame.draw.circle(fill, (*current_color, 50), (radius, radius), radius)
        surface.blit(fill, (int(self.center.x - self.internal_radius), int(self.center.y - self.internal_radius)))

        pygame.draw.circle(surface, current_color, (cx, cy), radius, 1)
        outer_color = (int(255 * (1 - ratio) / 3), int(255 * ratio / 3), 0)
        pygame.draw.circle(surface, outer_color, (cx, cy), int(self.external_radius), 1)

        health_text = get_font(15).render(f"Health: {self.health}", True, (255, 255, 255), (0, 10, 0))
        surface.blit(health_text, health_text.get_rect(center=(self.center.x, self.center.y + 50)))
