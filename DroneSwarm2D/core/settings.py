# DroneSwarm2D/core/settings.py
"""
Gerenciador de configurações do DroneSwarm2D.

Carrega e processa arquivos de configuração JSON, fornecendo acesso global
às configurações através de um padrão Singleton.

A configuração deve ser inicializada uma vez no início da aplicação através
de `settings.initialize()` e depois fica disponível globalmente.

Example:
    >>> from DroneSwarm2D.core import settings
    >>> settings.initialize("config/scenario_1.json")
    >>> width = settings.SIM_WIDTH
    >>> center = settings.CENTER
"""

import os
import json
import glob
from typing import Dict, Any, List, Optional
import pygame


class ConfigManager:
    """
    Gerenciador de configurações do simulador (Singleton).
    
    A configuração é inicializada uma vez e depois acessada globalmente
    através de variáveis do módulo ou pela instância singleton.
    
    Attributes:
        config_path: Caminho para o arquivo de configuração JSON carregado
        config_name: Nome do arquivo de configuração (sem extensão)
        screen: Superfície pygame principal
        clock: Relógio pygame para controle de FPS
        
    Note:
        Esta classe implementa o padrão Singleton - apenas uma instância
        pode existir por vez.
    """
    
    _instance: Optional['ConfigManager'] = None
    _initialized: bool = False
    
    def __new__(cls, config_path: Optional[str] = None, fullscreen: bool = True) -> 'ConfigManager':
        """
        Garante que apenas uma instância exista (Singleton).
        
        Args:
            config_path: Caminho para arquivo .json de configuração
            fullscreen: Se True, cria janela em tela cheia
            
        Returns:
            Instância única do ConfigManager
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None, fullscreen: bool = True) -> None:
        """
        Inicializa o gerenciador de configurações.
        
        Args:
            config_path: Caminho para arquivo .json de configuração.
                        Se None, busca o primeiro .json em config/
            fullscreen: Se True, cria janela pygame em tela cheia
            
        Raises:
            FileNotFoundError: Se o arquivo de configuração não for encontrado
            
        Note:
            Evita reinicialização usando atributo de classe _initialized
        """
        # Evita reinicialização - USAR ATRIBUTO DE CLASSE
        if ConfigManager._initialized:
            return
        
        # Inicializar pygame se necessário
        if not pygame.get_init():
            pygame.init()
        
        # Resolver caminho do arquivo de configuração
        if config_path is None:
            config_path = self._find_default_config()
        else:
            config_path = os.path.abspath(config_path)
            
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
        
        self.config_path: str = config_path
        self.config_name: str = os.path.splitext(os.path.basename(config_path))[0]
        
        print(f"[ConfigManager] Carregando '{self.config_name}' de: {self.config_path}")
        
        # Carregar e processar configurações
        self._config: Dict[str, Any] = {}
        self._load_config()
        self._compute_derived_values()
        
        # Criar tela e relógio pygame
        self._setup_display(fullscreen)
        
        # Exportar para o namespace do módulo
        self._export_to_module()
        
        # Marcar como inicializado NA CLASSE
        ConfigManager._initialized = True
        print(f"[ConfigManager] Configuração inicializada com {len(self._config)} parâmetros")
    
    def _find_default_config(self) -> str:
        """
        Busca o primeiro arquivo .json no diretório config/.
        
        Returns:
            Caminho absoluto para o primeiro arquivo .json encontrado
            
        Raises:
            FileNotFoundError: Se nenhum arquivo .json for encontrado
            
        Note:
            Tenta múltiplos caminhos relativos comuns
        """
        # Tentar encontrar o diretório config relativo ao módulo
        module_dir: str = os.path.dirname(__file__)
        
        # Tentar diferentes caminhos possíveis
        possible_paths: List[str] = [
            os.path.join(module_dir, "../../config"),
            os.path.join(module_dir, "../config"),
            os.path.join(os.getcwd(), "config"),
        ]
        
        for config_dir in possible_paths:
            config_dir = os.path.abspath(config_dir)
            if os.path.exists(config_dir):
                json_paths: List[str] = sorted(glob.glob(os.path.join(config_dir, "*.json")))
                if json_paths:
                    return json_paths[0]
        
        raise FileNotFoundError(
            f"Nenhum arquivo .json encontrado. Procurado em: {possible_paths}"
        )
    
    def _load_config(self) -> None:
        """
        Carrega e processa o arquivo JSON de configuração.
        
        Este método:
        1. Lê o arquivo JSON
        2. Obtém informações do display pygame
        3. Processa strings com eval() para expressões Python
        4. Processa listas recursivamente
        
        Note:
            Usa eval() com namespace limitado para segurança
        """
        with open(self.config_path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = json.load(f)
        
        # Override dimensões da tela com informações do display
        display_info = pygame.display.Info()
        raw["FULL_WIDTH"] = display_info.current_w
        raw["FULL_HEIGHT"] = display_info.current_h
        
        # Processar cada chave do JSON
        for key, val in raw.items():
            if isinstance(val, str):
                # Tentar avaliar strings como expressões Python
                try:
                    self._config[key] = eval(val, {}, self._config)
                except Exception:
                    # Se falhar, manter como string
                    self._config[key] = val
            elif isinstance(val, list):
                # Processar listas recursivamente
                lst: List[Any] = []
                for item in val:
                    if isinstance(item, str):
                        try:
                            lst.append(eval(item, {}, self._config))
                        except Exception:
                            lst.append(item)
                    else:
                        lst.append(item)
                self._config[key] = lst
            else:
                self._config[key] = val
    
    def _compute_derived_values(self) -> None:
        """
        Calcula valores derivados das configurações base.
        
        Valores derivados incluem:
        - INTEREST_POINT_CENTER: Centro do ponto de interesse
        - CENTER: Alias para o centro
        - TYPE_OF_SCENARIO: Nome do cenário
        - ENEMY_BEHAVIOR: Processamento de comportamento None
        - DMZ: Processamento de zonas desmilitarizadas
        """
        # Centro do ponto de interesse
        self._config["INTEREST_POINT_CENTER"] = pygame.math.Vector2(
            self._config["SIM_WIDTH"] / 2,
            self._config["SIM_HEIGHT"] / 2
        )
        
        self._config["CENTER"] = self._config["INTEREST_POINT_CENTER"].copy()
        self._config["TYPE_OF_SCENARIO"] = self.config_name
        
        # Processar ENEMY_BEHAVIOR
        if "ENEMY_BEHAVIOR" in self._config:
            behavior = self._config["ENEMY_BEHAVIOR"]
            self._config["ENEMY_BEHAVIOR"] = None if behavior == "None" else behavior
        
        # Processar zonas desmilitarizadas (DMZ)
        if "DMZ" in self._config:
            dmz_list: List[tuple[float, float, int]] = []
            for expr_x, expr_y, radius in self._config["DMZ"]:
                x: float = float(eval(str(expr_x), {}, self._config))
                y: float = float(eval(str(expr_y), {}, self._config))
                r: int = int(radius)
                dmz_list.append((x, y, r))
            self._config["DMZ"] = dmz_list
    
    def _setup_display(self, fullscreen: bool) -> None:
        """
        Configura a janela pygame e o relógio.
        
        Args:
            fullscreen: Se True, cria janela em modo fullscreen
            
        Note:
            Define self.screen e self.clock como atributos da instância
        """
        width: int = self._config["FULL_WIDTH"]
        height: int = self._config["FULL_HEIGHT"]
        
        if fullscreen:
            self.screen: pygame.Surface = pygame.display.set_mode(
                (width, height),
                pygame.FULLSCREEN
            )
        else:
            self.screen = pygame.display.set_mode((width, height))
        
        self.clock: pygame.time.Clock = pygame.time.Clock()
    
    def _export_to_module(self) -> None:
        """
        Exporta todas as configurações como variáveis do módulo.
        
        Isso permite o uso de: `from settings import SIM_WIDTH`
        
        Note:
            Define __all__ no módulo para suportar `from settings import *`
        """
        import sys
        module = sys.modules[__name__]
        
        # Exportar todas as configurações como variáveis do módulo
        for key, value in self._config.items():
            setattr(module, key, value)
        
        # Exportar também screen e clock
        setattr(module, 'screen', self.screen)
        setattr(module, 'clock', self.clock)
        
        # IMPORTANTE: Definir __all__ para que "from settings import *" funcione
        all_exports: List[str] = list(self._config.keys()) + ['screen', 'clock']
        setattr(module, '__all__', all_exports)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtém valor de configuração por chave.
        
        Args:
            key: Nome da configuração
            default: Valor padrão se a chave não existir
            
        Returns:
            Valor da configuração ou default
            
        Example:
            >>> config = ConfigManager.get_instance()
            >>> width = config.get('SIM_WIDTH', 800)
        """
        return self._config.get(key, default)
    
    def __getattr__(self, name: str) -> Any:
        """
        Permite acesso direto aos atributos de configuração.
        
        Args:
            name: Nome do atributo
            
        Returns:
            Valor da configuração
            
        Raises:
            AttributeError: Se a configuração não existir
            
        Example:
            >>> config = ConfigManager.get_instance()
            >>> width = config.SIM_WIDTH
        """
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        if name in self._config:
            return self._config[name]
        
        raise AttributeError(
            f"Configuração '{name}' não encontrada em {self.config_path}"
        )
    
    def __getitem__(self, key: str) -> Any:
        """
        Permite acesso via subscript.
        
        Args:
            key: Nome da configuração
            
        Returns:
            Valor da configuração
            
        Example:
            >>> config = ConfigManager.get_instance()
            >>> width = config['SIM_WIDTH']
        """
        return self._config[key]
    
    def keys(self) -> List[str]:
        """
        Retorna lista de chaves disponíveis.
        
        Returns:
            Lista com nomes de todas as configurações
        """
        return list(self._config.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Retorna cópia do dicionário de configurações.
        
        Returns:
            Dicionário com todas as configurações
        """
        return self._config.copy()
    
    def __repr__(self) -> str:
        """
        Representação em string do ConfigManager.
        
        Returns:
            String descritiva do gerenciador
        """
        return f"ConfigManager(config='{self.config_name}', params={len(self._config)})"
    
    @classmethod
    def is_initialized(cls) -> bool:
        """
        Verifica se o ConfigManager já foi inicializado.
        
        Returns:
            True se já foi inicializado, False caso contrário
        """
        return cls._initialized
    
    @classmethod
    def get_instance(cls) -> Optional['ConfigManager']:
        """
        Retorna a instância singleton, se existir.
        
        Returns:
            Instância do ConfigManager ou None se não inicializado
        """
        return cls._instance


# -----------------------------------------------------------------------------
# Funções auxiliares para inicialização e acesso
# -----------------------------------------------------------------------------

def initialize(config_path: Optional[str] = None, fullscreen: bool = True) -> ConfigManager:
    """
    Inicializa a configuração do framework.
    
    Deve ser chamada uma vez no início da aplicação (main).
    
    Args:
        config_path: Caminho para o arquivo de configuração JSON
        fullscreen: Se True, cria janela em tela cheia
        
    Returns:
        Instância do ConfigManager
        
    Example:
        >>> # No main.py
        >>> from DroneSwarm2D.core import settings
        >>> settings.initialize("config/scenario_1.json")
    """
    return ConfigManager(config_path, fullscreen)


def get_config() -> ConfigManager:
    """
    Retorna a instância do ConfigManager.
    
    Returns:
        Instância do ConfigManager
        
    Raises:
        RuntimeError: Se initialize() não foi chamado ainda
        
    Example:
        >>> # Em qualquer módulo do framework
        >>> from DroneSwarm2D.core import settings
        >>> config = settings.get_config()
        >>> width = config.SIM_WIDTH
    """
    if not ConfigManager.is_initialized():
        raise RuntimeError(
            "ConfigManager não foi inicializado! "
            "Chame settings.initialize() no início da aplicação."
        )
    return ConfigManager.get_instance()


def is_initialized() -> bool:
    """
    Verifica se a configuração foi inicializada.
    
    Returns:
        True se inicializado, False caso contrário
    """
    return ConfigManager.is_initialized()


# -----------------------------------------------------------------------------
# __getattr__ para imports individuais
# -----------------------------------------------------------------------------

def __getattr__(name: str) -> Any:
    """
    Permite importação de variáveis individuais após inicialização.
    
    Suporta tanto "from settings import *" quanto "from settings import VAR"
    
    Args:
        name: Nome da variável a ser importada
        
    Returns:
        Valor da configuração
        
    Raises:
        AttributeError: Se o atributo não existir ou ConfigManager não foi inicializado
        
    Example:
        >>> from DroneSwarm2D.core.settings import SIM_WIDTH, CENTER
        >>> from DroneSwarm2D.core.settings import *
    """
    # Lista de atributos especiais do Python (ignorar sem erro)
    special_attrs: set = {
        '__path__', '__file__', '__cached__', '__loader__',
        '__spec__', '__package__', '__name__', '__doc__',
        '__builtins__', '__all__', '__dict__', '__class__',
        '__annotations__', '__wrapped__',
    }
    
    # Ignorar atributos especiais
    if name in special_attrs or (name.startswith('__') and name.endswith('__')):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
    # Verificar inicialização
    if not ConfigManager.is_initialized():
        raise RuntimeError(
            f"ConfigManager não foi inicializado! "
            f"Tentando acessar: '{name}'\n"
            f"Solução: Chame settings.initialize() no início do main.py"
        )
    
    config = ConfigManager.get_instance()
    
    # Tentar pegar do _config
    if name in config._config:
        return config._config[name]
    
    # Tentar pegar screen ou clock
    if name == 'screen':
        return config.screen
    if name == 'clock':
        return config.clock
    
    raise AttributeError(
        f"Configuração '{name}' não encontrada.\n"
        f"Disponíveis: {', '.join(list(config._config.keys())[:10])}..."
    )