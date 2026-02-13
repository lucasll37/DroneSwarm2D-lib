# DroneSwarm2D/__init__.py
"""
DroneSwarm2D - Simulador de enxames de drones defensivos
Biblioteca desenvolvida como parte do trabalho de graduação no ITA
"""

__version__ = "0.1.0"
__author__ = "Lucas"

# Variáveis que serão preenchidas após init()
AirTrafficEnv = None
FriendDrone = None
EnemyDrone = None
InterestPoint = None
DemilitarizedZone = None

_initialized = False


def init(config_path=None, fullscreen=True):
    """
    Inicializa a biblioteca DroneSwarm2D.
    DEVE ser chamada antes de usar qualquer classe.
    
    Args:
        config_path: Caminho para arquivo de configuração JSON
        fullscreen: Se True, cria janela em tela cheia
    
    Returns:
        Módulo settings com todas as configurações carregadas
    
    Example:
        import DroneSwarm2D
        settings = DroneSwarm2D.init("config/scenario_1.json")
        env = DroneSwarm2D.AirTrafficEnv(mode='human')
    """
    global AirTrafficEnv, FriendDrone, EnemyDrone, CircleInterestPoint, CircleDMZ, _initialized, generate_sparse_matrix, get_friends_hold, load_best_model
    
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
    
    
    
    AirTrafficEnv = _AirTrafficEnv
    FriendDrone = _FriendDrone
    EnemyDrone = _EnemyDrone
    CircleInterestPoint = _CircleInterestPoint
    CircleDMZ = _CircleDMZ
    generate_sparse_matrix = _generate_sparse_matrix
    get_friends_hold = _get_friends_hold
    load_best_model = _load_best_model
    
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
]