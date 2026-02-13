# DroneSwarm2D/core/__init__.py
"""
Módulo core - Classes principais do simulador.

Este módulo contém as classes fundamentais do DroneSwarm2D:
- AirTrafficEnv: Ambiente de simulação principal
- FriendDrone: Drones defensivos autônomos
- EnemyDrone: Drones ofensivos
- InterestPoint: Pontos de interesse a serem defendidos
- DemilitarizedZone: Zonas onde engajamento é proibido
- settings: Sistema de configuração
- utils: Funções utilitárias

Nota: As classes não são importadas aqui para evitar imports circulares.
      Elas ficam disponíveis após chamar DroneSwarm2D.init().
"""

__all__: list[str] = []