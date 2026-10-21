"""Core of the simulator / Núcleo do simulador.

Modules:
    settings          Configuration loader (must be initialised first, via ``DroneSwarm2D.init``).
    AirTrafficEnv     The environment (``reset`` / ``step``), combat rules, reward and UI.
    FriendDrone       Defensive drones: perception, communication and motion.
    EnemyDrone        Attacking drones and their approach patterns.
    InterestPoint     The defended asset.
    DemilitarizedZone Areas where engagements are forbidden.
    rendering         Drawing helpers and UI widgets.
    utils             Grid geometry, interception maths and matrix helpers.

The classes are not imported here: they read the settings at import time, so
they become available only after ``DroneSwarm2D.init()``.
"""

__all__: list[str] = []
