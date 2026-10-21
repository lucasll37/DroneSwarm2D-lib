"""Behaviour API and default policies of the friendly drones.

PT-BR: API de comportamentos e políticas padrão dos drones amigos.
Import it only after ``DroneSwarm2D.init()``.
"""

from .behaviors import (
    BaseBehavior,
    BehaviorType,
    FriendAEWBehaviorDefault,
    FriendCommonBehaviorDefault,
    FriendRadarBehaviorDefault,
)

__all__ = [
    "BehaviorType",
    "BaseBehavior",
    "FriendCommonBehaviorDefault",
    "FriendRadarBehaviorDefault",
    "FriendAEWBehaviorDefault",
]
