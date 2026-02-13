import DroneSwarm2D
settings = DroneSwarm2D.init(
    config_path="./example/src/config.json",
    fullscreen=True
)

import numpy as np
import pygame
from DroneSwarm2D.core.utils import pos_to_cell, intercept_direction, can_intercept
from DroneSwarm2D.behaviorsDefault import BaseBehavior, BehaviorType

# -------------------------------------------------------------------------
# Planning Policy (Class Method)
# -------------------------------------------------------------------------   

class FriendCommonBehavior(BaseBehavior):
    def __init__(self, friend_activation_threshold_position: float = 0.7,
                 enemy_activation_threshold_position: float = 0.4):

        super().__init__(behavior_type=BehaviorType.COMMON)
        self.friend_activation_threshold_position = friend_activation_threshold_position
        self.enemy_activation_threshold_position = enemy_activation_threshold_position

    def apply(self, state, joystick_controlled: bool = False) -> tuple:         
        # Extração e preparação do estado
        pos = np.squeeze(state['pos'])
        pos = pygame.math.Vector2(pos[0], pos[1])
        friend_intensity = np.squeeze(state['friend_intensity'])
        enemy_intensity = np.squeeze(state['enemy_intensity'])
        friend_direction = np.squeeze(state['friend_direction'])
        enemy_direction = np.squeeze(state['enemy_direction'])
        
        patrol_radius = 150
        enemy_targets = []
        # Identifica células da matriz de inimigos com intensidade acima do limiar
        for cell, intensity in np.ndenumerate(enemy_intensity):
            if intensity < self.enemy_activation_threshold_position:
                continue
            
            target_pos = pygame.math.Vector2((cell[0] + 0.5) * settings.CELL_SIZE,
                                            (cell[1] + 0.5) * settings.CELL_SIZE)
            distance_to_interest = target_pos.distance_to(settings.INTEREST_POINT_CENTER)
            
            enemy_targets.append((cell, target_pos, distance_to_interest))
            
            
        # Obtém a célula correspondente à posição do drone self
        enemy_targets.sort(key=lambda t: t[2])
        
        if len(enemy_targets) > 0:
            enemy = enemy_targets[0]
            enemy_pos = enemy[1]
            enemy_direction = enemy_direction[enemy[0]]

            info = ("PURSUING", None, None, None)
            vel = intercept_direction(pos, settings.FRIEND_SPEED, enemy_pos, enemy_direction)
            
        else:
            info = ("PATROLLING", None, None, None)
            # Movimento circular em torno do ponto de interesse
            r_vec = pos - settings.INTEREST_POINT_CENTER
            current_distance = r_vec.length()
            
            if current_distance == 0:
                r_vec = pygame.math.Vector2(patrol_radius, 0)
                current_distance = patrol_radius
                
            # Calcular correção radial
            radial_error = patrol_radius - current_distance
            k_radial = 0.05  # Fator de correção radial
            radial_correction = k_radial * radial_error * r_vec.normalize()
            
            # Calcular velocidade tangencial (perpendicular ao radial)
            tangent = pygame.math.Vector2(-r_vec.y, r_vec.x).normalize()
            vel = tangent * settings.FRIEND_SPEED + radial_correction

        return info, vel
    
class FriendAEWBehavior(BaseBehavior):
    def __init__(self, friend_activation_threshold_position: float = 0.7,
                 enemy_activation_threshold_position: float = 0.4):

        super().__init__(behavior_type=BehaviorType.AEW)
        self.friend_activation_threshold_position = friend_activation_threshold_position
        self.enemy_activation_threshold_position = enemy_activation_threshold_position

        
    def apply(self, state, joystick_controlled: bool = False) -> tuple:
        
        pos = np.squeeze(state['pos'])
        pos = pygame.math.Vector2(pos[0], pos[1])
        friend_intensity = np.squeeze(state['friend_intensity'])
        enemy_intensity = np.squeeze(state['enemy_intensity'])
        friend_direction = np.squeeze(state['friend_direction'])
        enemy_direction = np.squeeze(state['enemy_direction'])
        

        # Compute radial vector from interest point
        r_vec = pos - settings.INTEREST_POINT_CENTER
        current_distance = r_vec.length()
        
        if current_distance == 0:
            r_vec = pygame.math.Vector2(settings.AEW_RANGE, 0)
            current_distance = settings.AEW_RANGE
            
        # Calculate orbit correction
        radial_error = settings.AEW_RANGE - current_distance
        k_radial = 0.05  # Radial correction factor
        radial_correction = k_radial * radial_error * r_vec.normalize()
        
        # Calculate tangential velocity (perpendicular to radial)
        tangent = pygame.math.Vector2(-r_vec.y, r_vec.x).normalize()
        vel = tangent * settings.AEW_SPEED
        info = ("AEW", None, None, None)
        
        return info, vel


class FriendRadarBehavior(BaseBehavior):
    def __init__(self, friend_activation_threshold_position: float = 0.7,
                 enemy_activation_threshold_position: float = 0.4):

        super().__init__(behavior_type=BehaviorType.RADAR)
        self.friend_activation_threshold_position = friend_activation_threshold_position
        self.enemy_activation_threshold_position = enemy_activation_threshold_position
        self.type = BehaviorType.RADAR
        
    def apply(self, state, joystick_controlled: bool = False) -> tuple:
        """
        Apply RADAR behavior - remain stationary.
        """
        info = ("RADAR", None, None, None)
        vel = pygame.math.Vector2(0, 0)
        
        return info, vel