"""Demilitarised zones (DMZ): areas where no engagement may happen.

PT-BR: Zonas desmilitarizadas (DMZ): áreas onde nenhum engajamento pode ocorrer.

While the DMZ layer is enabled (``AirTrafficEnv.show_dmz``, the "Tog. DMZ"
button), an enemy inside a zone cannot be neutralised. Zones overlapping the
interest point are discarded by the environment.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pygame

from .rendering import get_font


class CircleDMZ:
    """Circular demilitarised zone.

    PT-BR: Zona desmilitarizada circular.

    Args:
        center: Centre in pixels.
        radius: Radius in pixels.
        color: RGB colour used to draw it.
        seed: Unused; kept for API compatibility.

    Attributes:
        dmz_id: Sequential id (class-wide counter).
    """

    dmz_id_counter: int = 0

    def __init__(
        self,
        center: pygame.math.Vector2,
        radius: float,
        color: Tuple[int, int, int] = (0, 0, 255),
        seed: Optional[int] = None,
    ) -> None:
        self.center: pygame.math.Vector2 = center
        self.radius: float = radius
        self.color: Tuple[int, int, int] = color
        self.dmz_id: int = self._assign_id()

    def _assign_id(self) -> int:
        current_id = self.__class__.dmz_id_counter
        self.__class__.dmz_id_counter += 1
        return current_id

    def contains(self, position: pygame.math.Vector2) -> bool:
        """Whether ``position`` lies inside (or on the border of) the zone."""
        return position.distance_to(self.center) <= self.radius

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the zone as a translucent disc with an "X" and its id.

        PT-BR: Desenha a zona como um disco translúcido com um "X" e seu id.
        """
        alpha = 75
        rgba = (*self.color, alpha)
        cx, cy = int(self.center.x), int(self.center.y)

        disc = pygame.Surface((int(self.radius * 2), int(self.radius * 2)), pygame.SRCALPHA)
        pygame.draw.circle(disc, rgba, (int(self.radius), int(self.radius)), int(self.radius))
        surface.blit(disc, (int(self.center.x - self.radius), int(self.center.y - self.radius)))
        pygame.draw.circle(surface, rgba, (cx, cy), int(self.radius), 2)

        arm = self.radius * 0.7
        pygame.draw.line(surface, rgba, (int(self.center.x - arm), int(self.center.y - arm)),
                         (int(self.center.x + arm), int(self.center.y + arm)), 2)
        pygame.draw.line(surface, rgba, (int(self.center.x + arm), int(self.center.y - arm)),
                         (int(self.center.x - arm), int(self.center.y + arm)), 2)

        label = get_font(10).render(f"ID: DMZ{self.dmz_id}", True, (255, 255, 255))
        label.set_alpha(128)
        surface.blit(label, (cx, cy))
