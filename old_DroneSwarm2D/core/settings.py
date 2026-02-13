# DroneSwarm2D/core/settings.py
"""
Gerenciador de configurações do DroneSwarm2D.
Carrega e processa arquivos de configuração JSON.

A configuração deve ser inicializada uma vez no início da aplicação
e depois fica disponível globalmente para todo o framework.
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
    """
    
    _instance: Optional['ConfigManager'] = None
    _initialized: bool = False
    
    def __new__(cls, config_path: Optional[str] = None, fullscreen: bool = True):
        """Garante que apenas uma instância exista (Singleton)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None, fullscreen: bool = True):
        """
        Inicializa o gerenciador de configurações.
        
        Args:
            config_path: Caminho para arquivo .json de configuração.
                        Se None, busca o primeiro .json em ../../config/
            fullscreen: Se True, cria janela em tela cheia
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
        
        self.config_path = config_path
        self.config_name = os.path.splitext(os.path.basename(config_path))[0]
        
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
        """Busca o primeiro arquivo .json no diretório config/."""
        # Tenta encontrar o diretório config relativo ao módulo
        module_dir = os.path.dirname(__file__)
        
        # Tenta diferentes caminhos possíveis
        possible_paths = [
            os.path.join(module_dir, "../../config"),
            os.path.join(module_dir, "../config"),
            os.path.join(os.getcwd(), "config"),
        ]
        
        for config_dir in possible_paths:
            config_dir = os.path.abspath(config_dir)
            if os.path.exists(config_dir):
                json_paths = sorted(glob.glob(os.path.join(config_dir, "*.json")))
                if json_paths:
                    return json_paths[0]
        
        raise FileNotFoundError(
            f"Nenhum arquivo .json encontrado. Procurado em: {possible_paths}"
        )
    
    def _load_config(self) -> None:
        """Carrega e processa o arquivo JSON de configuração."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        
        # Override dimensões da tela com informações do display
        display_info = pygame.display.Info()
        raw["FULL_WIDTH"] = display_info.current_w
        raw["FULL_HEIGHT"] = display_info.current_h
        
        # Processar cada chave do JSON
        for key, val in raw.items():
            if isinstance(val, str):
                try:
                    self._config[key] = eval(val, {}, self._config)
                except Exception:
                    self._config[key] = val
            elif isinstance(val, list):
                lst = []
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
        """Calcula valores derivados das configurações base."""
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
            dmz_list = []
            for expr_x, expr_y, radius in self._config["DMZ"]:
                x = float(eval(str(expr_x), {}, self._config))
                y = float(eval(str(expr_y), {}, self._config))
                r = int(radius)
                dmz_list.append((x, y, r))
            self._config["DMZ"] = dmz_list
    
    def _setup_display(self, fullscreen: bool) -> None:
        """Configura a janela pygame e o relógio."""
        width = self._config["FULL_WIDTH"]
        height = self._config["FULL_HEIGHT"]
        
        if fullscreen:
            self.screen = pygame.display.set_mode(
                (width, height),
                pygame.FULLSCREEN
            )
        else:
            self.screen = pygame.display.set_mode((width, height))
        
        self.clock = pygame.time.Clock()
    
    def _export_to_module(self) -> None:
        """
        Exporta todas as configurações como variáveis do módulo.
        Permite: from settings import *
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
        all_exports = list(self._config.keys()) + ['screen', 'clock']
        setattr(module, '__all__', all_exports)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtém valor de configuração por chave."""
        return self._config.get(key, default)
    
    def __getattr__(self, name: str) -> Any:
        """Permite acesso direto aos atributos de configuração."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        if name in self._config:
            return self._config[name]
        
        raise AttributeError(
            f"Configuração '{name}' não encontrada em {self.config_path}"
        )
    
    def __getitem__(self, key: str) -> Any:
        """Permite acesso via subscript."""
        return self._config[key]
    
    def keys(self) -> List[str]:
        """Retorna lista de chaves disponíveis."""
        return list(self._config.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Retorna cópia do dicionário de configurações."""
        return self._config.copy()
    
    def __repr__(self) -> str:
        return f"ConfigManager(config='{self.config_name}', params={len(self._config)})"
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Verifica se o ConfigManager já foi inicializado."""
        return cls._initialized
    
    @classmethod
    def get_instance(cls) -> Optional['ConfigManager']:
        """Retorna a instância singleton, se existir."""
        return cls._instance


# Funções auxiliares para inicialização e acesso
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
        # No main.py
        from DroneSwarm2D.core import settings
        settings.initialize("config/scenario_1.json")
    """
    return ConfigManager(config_path, fullscreen)


def get_config() -> ConfigManager:
    """
    Retorna a instância do ConfigManager.
    Lança erro se não foi inicializada.
    
    Returns:
        Instância do ConfigManager
        
    Raises:
        RuntimeError: Se initialize() não foi chamado ainda
        
    Example:
        # Em qualquer módulo do framework
        from DroneSwarm2D.core import settings
        config = settings.get_config()
        width = config.SIM_WIDTH
    """
    if not ConfigManager.is_initialized():
        raise RuntimeError(
            "ConfigManager não foi inicializado! "
            "Chame settings.initialize() no início da aplicação."
        )
    return ConfigManager.get_instance()


def is_initialized() -> bool:
    """Verifica se a configuração foi inicializada."""
    return ConfigManager.is_initialized()


# __getattr__ para imports individuais e para evitar erro com atributos especiais
def __getattr__(name: str) -> Any:
    """
    Permite importação de variáveis individuais após inicialização.
    Suporta tanto "from settings import *" quanto "from settings import VAR"
    
    Example:
        from DroneSwarm2D.core.settings import SIM_WIDTH, CENTER
        from DroneSwarm2D.core.settings import *
    """
    # Lista de atributos especiais do Python (ignorar sem erro)
    special_attrs = {
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