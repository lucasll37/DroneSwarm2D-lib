"""
InterestPoint.py

Define a classe CircleInterestPoint que representa pontos de interesse
a serem defendidos na simulação.
"""
from typing import Optional, Tuple
import pygame

from .settings import (
    INTEREST_POINT_INITIAL_HEALTH,
    FONT_FAMILY
)


class CircleInterestPoint:
    """
    Representa um ponto de interesse circular a ser defendido.
    
    O círculo muda de cor dinamicamente baseado em sua saúde:
    - Verde (0, 255, 0) quando com saúde completa
    - Amarelo quando com saúde média
    - Vermelho (255, 0, 0) quando com saúde baixa
    
    Attributes:
        center: Centro do círculo (pygame.math.Vector2)
        internal_radius: Raio interno do ponto de interesse
        external_radius: Raio externo (limite de defesa)
        health: Saúde atual do ponto de interesse
        base_color: Cor base do ponto (padrão: verde)
    """

    def __init__(
        self, 
        center: pygame.math.Vector2, 
        internal_radius: float, 
        external_radius: float,
        color: Tuple[int, int, int] = (0, 255, 0), 
        seed: Optional[int] = None
    ) -> None:
        """
        Inicializa o ponto de interesse como um círculo.
        
        Args:
            center: Centro do círculo como Vector2
            internal_radius: Raio interno em pixels
            external_radius: Raio externo em pixels (limite de defesa)
            color: Cor base RGB do círculo (padrão: verde)
            seed: Seed para geração de números aleatórios (não utilizada atualmente)
        """
        self.center: pygame.math.Vector2 = center
        self.internal_radius: float = internal_radius
        self.external_radius: float = external_radius
        self.health: int = INTEREST_POINT_INITIAL_HEALTH
        self.base_color: Tuple[int, int, int] = color

    def draw(self, surface: pygame.Surface) -> None:
        """
        Desenha o ponto de interesse na superfície pygame.
        
        O círculo é renderizado com:
        - Preenchimento semi-transparente colorido baseado na saúde
        - Borda interna colorida
        - Borda externa mais clara
        - Texto mostrando a saúde atual no centro
        
        A cor varia de verde (saúde completa) para vermelho (saúde baixa)
        usando interpolação linear: R = 255*(1-health_ratio), G = 255*health_ratio
        
        Args:
            surface: Superfície pygame onde o ponto será desenhado
        """
        # Calcular proporção de saúde (0.0 a 1.0)
        health_ratio: float = self.health / INTEREST_POINT_INITIAL_HEALTH
        health_ratio = max(0.0, min(1.0, health_ratio))  # Clamped entre 0 e 1
        
        # Interpolar cor: verde (saúde completa) -> vermelho (sem saúde)
        current_color: Tuple[int, int, int] = (
            int(255 * (1 - health_ratio)),  # Red aumenta quando saúde diminui
            int(255 * health_ratio),        # Green diminui quando saúde diminui
            0                               # Blue sempre 0
        )
        
        # Calcular diâmetro para o círculo interno
        diameter: int = int(self.internal_radius * 2)
        
        # Criar superfície com alpha para preenchimento semi-transparente
        alpha_surface = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        fill_color = (current_color[0], current_color[1], current_color[2], 50)
        
        # Desenhar círculo preenchido na superfície alpha
        pygame.draw.circle(
            alpha_surface, 
            fill_color, 
            (int(self.internal_radius), int(self.internal_radius)), 
            int(self.internal_radius)
        )
        
        # Blit na superfície principal, centralizado
        surface.blit(
            alpha_surface, 
            (
                int(self.center.x - self.internal_radius), 
                int(self.center.y - self.internal_radius)
            )
        )
        
        # Desenhar borda do círculo interno
        pygame.draw.circle(
            surface, 
            current_color, 
            (int(self.center.x), int(self.center.y)), 
            int(self.internal_radius), 
            1
        )
        
        # Desenhar borda do círculo externo (mais clara)
        current_color_outer: Tuple[int, int, int] = (
            int(255 * (1 - health_ratio) / 3),
            int(255 * health_ratio / 3),
            0
        )
        pygame.draw.circle(
            surface, 
            current_color_outer, 
            (int(self.center.x), int(self.center.y)), 
            int(self.external_radius), 
            1
        )
        
        # Renderizar texto com a saúde atual
        font = pygame.font.SysFont(FONT_FAMILY, 15)
        health_text = font.render(
            f"Health: {self.health}", 
            True, 
            (255, 255, 255),  # Texto branco
            (0, 10, 0)        # Fundo verde escuro
        )
        text_rect = health_text.get_rect(
            center=(self.center.x, self.center.y + 50)
        )
        surface.blit(health_text, text_rect)