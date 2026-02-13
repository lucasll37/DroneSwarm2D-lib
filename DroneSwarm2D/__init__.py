# DroneSwarm2D/__init__.py
"""
DroneSwarm2D - Simulador de enxames de drones defensivos.
Biblioteca desenvolvida como parte do trabalho de graduação no ITA.

Este módulo utiliza inicialização lazy (preguiçosa) para evitar imports circulares.
Todas as classes principais ficam disponíveis apenas após chamar DroneSwarm2D.init().

Example:
    >>> import DroneSwarm2D
    >>> settings = DroneSwarm2D.init("config/scenario_1.json")
    >>> env = DroneSwarm2D.AirTrafficEnv(mode='human')
"""
from typing import Optional, Any, List, Callable

__version__ = "0.1.0"
__author__ = "Lucas"

# Variáveis que serão preenchidas após init()
# Usar type: ignore pois serão atribuídas dinamicamente
AirTrafficEnv: Optional[type] = None  # type: ignore
FriendDrone: Optional[type] = None  # type: ignore
EnemyDrone: Optional[type] = None  # type: ignore
InterestPoint: Optional[type] = None  # type: ignore
CircleInterestPoint: Optional[type] = None  # type: ignore
DemilitarizedZone: Optional[type] = None  # type: ignore
CircleDMZ: Optional[type] = None  # type: ignore
generate_sparse_matrix: Optional[Callable] = None  # type: ignore
get_friends_hold: Optional[Callable] = None  # type: ignore
load_best_model: Optional[Callable] = None  # type: ignore

_initialized: bool = False


def init(config_path: Optional[str] = None, fullscreen: bool = True) -> Any:
    """
    Inicializa a biblioteca DroneSwarm2D.
    
    DEVE ser chamada antes de usar qualquer classe da biblioteca.
    Carrega as configurações do arquivo JSON especificado e disponibiliza
    todas as classes e funções principais no namespace do módulo.
    
    Args:
        config_path: Caminho para arquivo de configuração JSON.
                    Se None, busca o primeiro .json no diretório config/.
        fullscreen: Se True, cria janela pygame em tela cheia.
                   Se False, cria janela em modo janela.
    
    Returns:
        Módulo settings com todas as configurações carregadas.
        Pode ser usado para acessar variáveis de configuração.
    
    Raises:
        FileNotFoundError: Se o arquivo de configuração não for encontrado.
        
    Example:
        >>> import DroneSwarm2D
        >>> settings = DroneSwarm2D.init("config/scenario_1.json")
        >>> print(settings.SIM_WIDTH, settings.SIM_HEIGHT)
        >>> env = DroneSwarm2D.AirTrafficEnv(mode='human')
    """
    global AirTrafficEnv, FriendDrone, EnemyDrone
    global CircleInterestPoint, CircleDMZ
    global generate_sparse_matrix, get_friends_hold, load_best_model
    global create_video_from_frames
    global _initialized
    
    # Se já foi inicializado, apenas retorna settings
    if _initialized:
        from .core import settings
        return settings
    
    # Inicializar configurações
    from .core import settings
    settings.initialize(config_path, fullscreen)
    
    # Agora pode importar as classes (settings já está disponível)
    from .core.AirTrafficEnv import AirTrafficEnv as _AirTrafficEnv
    from .core.FriendDrone import FriendDrone as _FriendDrone
    from .core.EnemyDrone import EnemyDrone as _EnemyDrone
    from .core.InterestPoint import CircleInterestPoint as _CircleInterestPoint
    from .core.DemilitarizedZone import CircleDMZ as _CircleDMZ
    from .core.utils import generate_sparse_matrix as _generate_sparse_matrix
    from .core.utils import get_friends_hold as _get_friends_hold
    from .core.utils import load_best_model as _load_best_model
    
    from .tools.create_video import create_video_from_frames as _create_video_from_frames
    
    # Atribuir às variáveis do módulo
    AirTrafficEnv = _AirTrafficEnv
    FriendDrone = _FriendDrone
    EnemyDrone = _EnemyDrone
    CircleInterestPoint = _CircleInterestPoint
    CircleDMZ = _CircleDMZ
    generate_sparse_matrix = _generate_sparse_matrix
    get_friends_hold = _get_friends_hold
    load_best_model = _load_best_model
    create_video_from_frames = _create_video_from_frames
    
    _initialized = True
    
    return settings


__all__ = [
    'init',
    'AirTrafficEnv',
    'FriendDrone',
    'EnemyDrone',
    'CircleInterestPoint',
    'CircleDMZ',
    'generate_sparse_matrix',
    'get_friends_hold',
    'load_best_model',
]