"""DroneSwarm2D - a 2-D simulator of distributed defensive drone swarms.

PT-BR: DroneSwarm2D - simulador 2D de enxames defensivos de drones com
algoritmos distribuídos (trabalho de graduação no ITA).

The library reads its configuration at import time, so every class becomes
available only after :func:`init` is called::

    import DroneSwarm2D

    settings = DroneSwarm2D.init("config.json")          # opens the pygame window
    # settings = DroneSwarm2D.init("config.json", headless=True)  # no window
    env = DroneSwarm2D.AirTrafficEnv(mode="human", seed=42)

PT-BR: A configuração é lida na importação, por isso as classes só ficam
disponíveis depois de chamar :func:`init`.
"""

from typing import Any, Callable, Optional, Tuple

__version__ = "0.1.0"
__author__ = "Lucas"

# Filled in by init(); they are ``None`` until then.
AirTrafficEnv: Optional[type] = None  # type: ignore[assignment]
FriendDrone: Optional[type] = None  # type: ignore[assignment]
EnemyDrone: Optional[type] = None  # type: ignore[assignment]
InterestPoint: Optional[type] = None  # type: ignore[assignment]
CircleInterestPoint: Optional[type] = None  # type: ignore[assignment]
DemilitarizedZone: Optional[type] = None  # type: ignore[assignment]
CircleDMZ: Optional[type] = None  # type: ignore[assignment]
generate_sparse_matrix: Optional[Callable] = None  # type: ignore[assignment]
get_friends_hold: Optional[Callable] = None  # type: ignore[assignment]
create_video_from_frames: Optional[Callable] = None  # type: ignore[assignment]

_initialized: bool = False


def init(
    config_path: Optional[str] = None,
    fullscreen: bool = True,
    headless: bool = False,
    screen_size: Optional[Tuple[int, int]] = None,
) -> Any:
    """Load the configuration and expose the library classes.

    Must be called once, before using any other part of the library. Further
    calls return the already-loaded settings.

    PT-BR: Carrega a configuração e disponibiliza as classes. Deve ser chamada
    uma única vez, antes de usar o restante da biblioteca.

    Args:
        config_path: JSON scenario file. If ``None``, the first ``*.json`` found in
            a ``config/`` directory is used.
        fullscreen: Open the window in full-screen mode (ignored if ``headless``).
        headless: Run without a window (SDL dummy video driver) - for batch
            experiments, servers and CI. Combine with ``AirTrafficEnv(mode=None)``.
        screen_size: ``(width, height)`` of the virtual screen, overriding the
            display size. Defaults to the display size, or ``(1920, 1080)`` when
            ``headless``. Fixing it makes experiments independent of the monitor.

    Returns:
        The :mod:`DroneSwarm2D.core.settings` module (every config key is an attribute).

    Raises:
        FileNotFoundError: If the configuration file does not exist.

    Example:
        >>> import DroneSwarm2D
        >>> settings = DroneSwarm2D.init("config.json", headless=True)
        >>> settings.SIM_WIDTH, settings.SIM_HEIGHT
    """
    global AirTrafficEnv, FriendDrone, EnemyDrone, InterestPoint, CircleInterestPoint
    global DemilitarizedZone, CircleDMZ, generate_sparse_matrix, get_friends_hold
    global create_video_from_frames, _initialized

    from .core import settings

    if _initialized:
        settings.initialize(config_path, fullscreen, headless=headless, screen_size=screen_size)
        return settings

    settings.initialize(config_path, fullscreen, headless=headless, screen_size=screen_size)

    # The classes read the settings at import time, hence the late imports.
    from .core.AirTrafficEnv import AirTrafficEnv as _AirTrafficEnv
    from .core.DemilitarizedZone import CircleDMZ as _CircleDMZ
    from .core.EnemyDrone import EnemyDrone as _EnemyDrone
    from .core.FriendDrone import FriendDrone as _FriendDrone
    from .core.InterestPoint import CircleInterestPoint as _CircleInterestPoint
    from .core.utils import generate_sparse_matrix as _generate_sparse_matrix
    from .core.utils import get_friends_hold as _get_friends_hold
    from .tools.create_video import create_video_from_frames as _create_video_from_frames

    AirTrafficEnv = _AirTrafficEnv
    FriendDrone = _FriendDrone
    EnemyDrone = _EnemyDrone
    CircleInterestPoint = InterestPoint = _CircleInterestPoint
    CircleDMZ = DemilitarizedZone = _CircleDMZ
    generate_sparse_matrix = _generate_sparse_matrix
    get_friends_hold = _get_friends_hold
    create_video_from_frames = _create_video_from_frames

    _initialized = True
    return settings


def is_initialized() -> bool:
    """Whether :func:`init` has already been called."""
    return _initialized


__all__ = [
    "init",
    "is_initialized",
    "AirTrafficEnv",
    "FriendDrone",
    "EnemyDrone",
    "CircleInterestPoint",
    "CircleDMZ",
    "generate_sparse_matrix",
    "get_friends_hold",
    "create_video_from_frames",
]
