"""
DemilitarizedZone.py

Define a classe CircleDMZ que representa zonas desmilitarizadas circulares
onde engajamentos entre drones são proibidos.
"""
from typing import Optional, Tuple
import pygame

# Initialize pygame if not already initialized
if not pygame.get_init():
    pygame.init()

from .settings import FONT_FAMILY


class CircleDMZ:
    """
    Representa uma zona desmilitarizada circular onde engajamentos são proibidos.
    
    Estas zonas são representadas como círculos azuis semi-transparentes na simulação
    e não devem se sobrepor ao ponto de interesse.
    
    Attributes:
        center: Centro da zona (pygame.math.Vector2)
        radius: Raio da zona em pixels
        color: Cor RGB da zona (padrão: azul)
        dmz_id: Identificador único da zona
    """
    
    dmz_id_counter: int = 0

    def __init__(
        self, 
        center: pygame.math.Vector2, 
        radius: float,
        color: Tuple[int, int, int] = (0, 0, 255), 
        seed: Optional[int] = None
    ) -> None:
        """
        Inicializa uma zona desmilitarizada circular.
        
        Args:
            center: Centro da zona como Vector2
            radius: Raio da zona em pixels
            color: Cor RGB da zona (padrão: azul - (0, 0, 255))
            seed: Seed para geração de números aleatórios (não utilizada atualmente)
        """
        self.center: pygame.math.Vector2 = center
        self.radius: float = radius
        self.color: Tuple[int, int, int] = color
        self.dmz_id: int = self._assign_id()
        
    def _assign_id(self) -> int:
        """
        Atribui um ID único à zona.
        
        Returns:
            ID único da zona
        """
        current_id = self.__class__.dmz_id_counter
        self.__class__.dmz_id_counter += 1
        return current_id

    def draw(self, surface: pygame.Surface) -> None:
        """
        Desenha a zona desmilitarizada na superfície fornecida.
        
        A zona é renderizada como:
        - Círculo azul semi-transparente preenchido
        - Borda azul ao redor
        - Símbolo X no centro indicando "não engajamento"
        - Label com ID da zona
        
        Args:
            surface: Superfície pygame onde a zona será desenhada
        """
        alpha_line: int = 75
        
        # Desenhar círculo semi-transparente preenchido
        alpha_surface = pygame.Surface(
            (int(self.radius * 2), int(self.radius * 2)), 
            pygame.SRCALPHA
        )
        fill_color = (self.color[0], self.color[1], self.color[2], alpha_line)
        
        pygame.draw.circle(
            alpha_surface, 
            fill_color, 
            (int(self.radius), int(self.radius)), 
            int(self.radius)
        )
        
        surface.blit(
            alpha_surface, 
            (int(self.center.x - self.radius), int(self.center.y - self.radius))
        )
        
        # Desenhar borda do círculo
        pygame.draw.circle(
            surface, 
            (self.color[0], self.color[1], self.color[2], alpha_line), 
            (int(self.center.x), int(self.center.y)), 
            int(self.radius), 
            2
        )
        
        # Desenhar símbolo de "não engajamento" (um X)
        line_length: float = self.radius * 0.7
        line_color = (self.color[0], self.color[1], self.color[2], alpha_line)
        
        # Diagonal principal (\)
        pygame.draw.line(
            surface, 
            line_color,
            (int(self.center.x - line_length), int(self.center.y - line_length)),
            (int(self.center.x + line_length), int(self.center.y + line_length)), 
            2
        )
        
        # Diagonal secundária (/)
        pygame.draw.line(
            surface, 
            line_color,
            (int(self.center.x + line_length), int(self.center.y - line_length)),
            (int(self.center.x - line_length), int(self.center.y + line_length)), 
            2
        )
        
        # Desenhar label com ID
        font = pygame.font.SysFont(FONT_FAMILY, 10)
        label = font.render(f"ID: DMZ{self.dmz_id}", True, (255, 255, 255))
        label.set_alpha(128)
        surface.blit(label, (int(self.center.x), int(self.center.y)))