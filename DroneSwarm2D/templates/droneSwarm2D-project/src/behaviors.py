"""
behaviors.py

Comportamentos customizados para drones amigos.

Define três tipos de comportamentos:
- FriendCommonBehavior: Comportamento padrão de perseguição/patrulha
- FriendAEWBehavior: Comportamento de órbita para drones AEW
- FriendRadarBehavior: Comportamento estacionário para radares
"""

import sys
from pathlib import Path
from typing import Tuple

import DroneSwarm2D
from DroneSwarm2D.core import settings

import numpy as np
import pygame
from DroneSwarm2D.core.utils import pos_to_cell, intercept_direction, can_intercept
from DroneSwarm2D.behaviorsDefault import BaseBehavior, BehaviorType


class FriendCommonBehavior(BaseBehavior):
    """
    Comportamento padrão para drones amigos.
    
    Persegue inimigos detectados ou patrulha em órbita circular
    ao redor do ponto de interesse quando não há ameaças.
    
    Attributes:
        friend_activation_threshold_position: Limiar de intensidade para detecção de amigos
        enemy_activation_threshold_position: Limiar de intensidade para detecção de inimigos
    """
    
    def __init__(
        self,
        friend_activation_threshold_position: float = 0.7,
        enemy_activation_threshold_position: float = 0.4
    ) -> None:
        """
        Inicializa comportamento comum.
        
        Args:
            friend_activation_threshold_position: Limiar para ativação de detecção de amigos
            enemy_activation_threshold_position: Limiar para ativação de detecção de inimigos
        """
        super().__init__(behavior_type=BehaviorType.COMMON)
        self.friend_activation_threshold_position: float = friend_activation_threshold_position
        self.enemy_activation_threshold_position: float = enemy_activation_threshold_position

    def apply(self, state: dict, joystick_controlled: bool = False) -> Tuple[tuple, pygame.math.Vector2]:
        """
        Aplica comportamento ao drone.
        
        Args:
            state: Dicionário com estado do drone (pos, intensities, directions)
            joystick_controlled: Se drone está sob controle manual
            
        Returns:
            Tupla (info, velocity) onde:
                - info: Tupla (estado, target, projection, friends_hold)
                - velocity: Vetor de velocidade pygame.math.Vector2
        """
        # Extração e preparação do estado
        pos = np.squeeze(state['pos'])
        pos = pygame.math.Vector2(pos[0], pos[1])
        friend_intensity = np.squeeze(state['friend_intensity'])
        enemy_intensity = np.squeeze(state['enemy_intensity'])
        friend_direction = np.squeeze(state['friend_direction'])
        enemy_direction = np.squeeze(state['enemy_direction'])
        
        patrol_radius: float = 150.0
        enemy_targets: list = []
        
        # Identifica células com inimigos detectados
        for cell, intensity in np.ndenumerate(enemy_intensity):
            if intensity < self.enemy_activation_threshold_position:
                continue
            
            target_pos = pygame.math.Vector2(
                (cell[0] + 0.5) * settings.CELL_SIZE,
                (cell[1] + 0.5) * settings.CELL_SIZE
            )
            distance_to_interest: float = target_pos.distance_to(settings.INTEREST_POINT_CENTER)
            
            enemy_targets.append((cell, target_pos, distance_to_interest))
        
        # Ordena alvos por distância ao ponto de interesse (mais próximo primeiro)
        enemy_targets.sort(key=lambda t: t[2])
        
        if len(enemy_targets) > 0:
            # Perseguir inimigo mais próximo do ponto de interesse
            enemy = enemy_targets[0]
            enemy_pos: pygame.math.Vector2 = enemy[1]
            enemy_dir = enemy_direction[enemy[0]]

            info: tuple = ("PURSUING", None, None, None)
            vel: pygame.math.Vector2 = intercept_direction(
                pos, 
                settings.FRIEND_SPEED, 
                enemy_pos, 
                enemy_dir
            )
            
        else:
            # Patrulhar: movimento circular ao redor do ponto de interesse
            info = ("PATROLLING", None, None, None)
            
            r_vec = pos - settings.INTEREST_POINT_CENTER
            current_distance: float = r_vec.length()
            
            if current_distance == 0:
                r_vec = pygame.math.Vector2(patrol_radius, 0)
                current_distance = patrol_radius
                
            # Calcular correção radial para manter órbita
            radial_error: float = patrol_radius - current_distance
            k_radial: float = 0.05
            radial_correction = k_radial * radial_error * r_vec.normalize()
            
            # Velocidade tangencial (perpendicular ao raio)
            tangent = pygame.math.Vector2(-r_vec.y, r_vec.x).normalize()
            vel = tangent * settings.FRIEND_SPEED + radial_correction

        return info, vel


class FriendAEWBehavior(BaseBehavior):
    """
    Comportamento para drones AEW (Airborne Early Warning).
    
    Mantém órbita constante ao redor do ponto de interesse
    no raio AEW_RANGE para detecção antecipada.
    
    Attributes:
        friend_activation_threshold_position: Limiar de intensidade para detecção de amigos
        enemy_activation_threshold_position: Limiar de intensidade para detecção de inimigos
    """
    
    def __init__(
        self,
        friend_activation_threshold_position: float = 0.7,
        enemy_activation_threshold_position: float = 0.4
    ) -> None:
        """
        Inicializa comportamento AEW.
        
        Args:
            friend_activation_threshold_position: Limiar para ativação de detecção de amigos
            enemy_activation_threshold_position: Limiar para ativação de detecção de inimigos
        """
        super().__init__(behavior_type=BehaviorType.AEW)
        self.friend_activation_threshold_position: float = friend_activation_threshold_position
        self.enemy_activation_threshold_position: float = enemy_activation_threshold_position

    def apply(self, state: dict, joystick_controlled: bool = False) -> Tuple[tuple, pygame.math.Vector2]:
        """
        Aplica comportamento de órbita AEW.
        
        Args:
            state: Dicionário com estado do drone
            joystick_controlled: Se drone está sob controle manual
            
        Returns:
            Tupla (info, velocity)
        """
        pos = np.squeeze(state['pos'])
        pos = pygame.math.Vector2(pos[0], pos[1])
        friend_intensity = np.squeeze(state['friend_intensity'])
        enemy_intensity = np.squeeze(state['enemy_intensity'])
        friend_direction = np.squeeze(state['friend_direction'])
        enemy_direction = np.squeeze(state['enemy_direction'])
        
        # Computar vetor radial a partir do ponto de interesse
        r_vec = pos - settings.INTEREST_POINT_CENTER
        current_distance: float = r_vec.length()
        
        if current_distance == 0:
            r_vec = pygame.math.Vector2(settings.AEW_RANGE, 0)
            current_distance = settings.AEW_RANGE
            
        # Calcular correção de órbita
        radial_error: float = settings.AEW_RANGE - current_distance
        k_radial: float = 0.05
        radial_correction = k_radial * radial_error * r_vec.normalize()
        
        # Velocidade tangencial (perpendicular ao raio)
        tangent = pygame.math.Vector2(-r_vec.y, r_vec.x).normalize()
        vel: pygame.math.Vector2 = tangent * settings.AEW_SPEED
        info: tuple = ("AEW", None, None, None)
        
        return info, vel


class FriendRadarBehavior(BaseBehavior):
    """
    Comportamento para drones RADAR estacionários.
    
    Permanece fixo em sua posição inicial para fornecer
    cobertura de detecção de longo alcance.
    
    Attributes:
        friend_activation_threshold_position: Limiar de intensidade para detecção de amigos
        enemy_activation_threshold_position: Limiar de intensidade para detecção de inimigos
        type: Tipo do comportamento (RADAR)
    """
    
    def __init__(
        self,
        friend_activation_threshold_position: float = 0.7,
        enemy_activation_threshold_position: float = 0.4
    ) -> None:
        """
        Inicializa comportamento RADAR.
        
        Args:
            friend_activation_threshold_position: Limiar para ativação de detecção de amigos
            enemy_activation_threshold_position: Limiar para ativação de detecção de inimigos
        """
        super().__init__(behavior_type=BehaviorType.RADAR)
        self.friend_activation_threshold_position: float = friend_activation_threshold_position
        self.enemy_activation_threshold_position: float = enemy_activation_threshold_position
        self.type = BehaviorType.RADAR
        
    def apply(self, state: dict, joystick_controlled: bool = False) -> Tuple[tuple, pygame.math.Vector2]:
        """
        Aplica comportamento RADAR (estacionário).
        
        Args:
            state: Dicionário com estado do drone (não utilizado)
            joystick_controlled: Se drone está sob controle manual (não utilizado)
            
        Returns:
            Tupla (info, velocity) com velocidade zero
        """
        info: tuple = ("RADAR", None, None, None)
        vel: pygame.math.Vector2 = pygame.math.Vector2(0, 0)
        
        return info, vel