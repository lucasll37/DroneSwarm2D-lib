# type: ignore
# Standard and third-party libraries
import pygame

# Initialize pygame if not already initialized
if not pygame.get_init():
    pygame.init()
    
# from scipy.ndimage import gaussian_filter
# from matplotlib import pyplot as plt
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from typing import Tuple, List, Any, Optional

# # Project-specific imports
# # from Drone import Drone
# from EnemyDrone import EnemyDrone
from .settings import *

class CircleDMZ:
    """
    Representa uma zona desmilitarizada onde engajamentos são proibidos.
    
    Estas zonas são representadas como círculos azuis na simulação
    e não devem se sobrepor ao ponto de interesse.
    """
    
    dmz_id_counter: int = 0

    def __init__(self, center: pygame.math.Vector2, radius: float,
                 color: Tuple[int, int, int] = (0, 0, 255), seed: Optional[int] = None) -> None:
        """
        Inicializa uma zona desmilitarizada circular.
        
        Args:
            center (pygame.math.Vector2): O centro da zona.
            radius (float): O raio da zona.
            color (Tuple[int, int, int]): Cor da zona (padrão é azul).
        """
        self.center = center
        self.radius = radius
        self.color = color
        self.dmz_id: int = self.assign_id()
        
    def assign_id(self) -> int:
        current_id = self.__class__.dmz_id_counter
        self.__class__.dmz_id_counter += 1
        return current_id

    def draw(self, surface: pygame.Surface) -> None:
        """
        Desenha a zona desmilitarizada na superfície fornecida.
        
        Args:
            surface (pygame.Surface): A superfície para desenhar a zona.
        """
        # Desenhar círculo semi-transparente
        alpha_line = 75
        alpha_surface = pygame.Surface((int(self.radius * 2), int(self.radius * 2)), pygame.SRCALPHA)
        fill_color = (self.color[0], self.color[1], self.color[2], alpha_line)
        
        pygame.draw.circle(alpha_surface, fill_color, 
                         (int(self.radius), int(self.radius)), int(self.radius))
        
        surface.blit(alpha_surface, 
                   (int(self.center.x - self.radius), int(self.center.y - self.radius)))
        
        # Desenhar borda do círculo
        pygame.draw.circle(surface, (self.color[0], self.color[1], self.color[2], alpha_line), 
                         (int(self.center.x), int(self.center.y)), int(self.radius), 2)
        
        # Desenhar símbolo de "não engajamento" (um X)
        line_length = self.radius * 0.7
        pygame.draw.line(surface, (self.color[0], self.color[1], self.color[2], alpha_line),
                        (int(self.center.x - line_length), int(self.center.y - line_length)),
                        (int(self.center.x + line_length), int(self.center.y + line_length)), 2)
        pygame.draw.line(surface, (self.color[0], self.color[1], self.color[2], alpha_line),
                        (int(self.center.x + line_length), int(self.center.y - line_length)),
                        (int(self.center.x - line_length), int(self.center.y + line_length)), 2)
        
        # 
        font = pygame.font.SysFont(FONT_FAMILY, 10)
        label = font.render(f"ID: DMZ{self.dmz_id}", True, (255, 255, 255))
        label.set_alpha(128)
        surface.blit(label, (int(self.center.x), int(self.center.y)))