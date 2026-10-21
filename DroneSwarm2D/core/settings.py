"""Configuration manager for DroneSwarm2D.

Loads a JSON scenario file, evaluates its expressions and exposes every key as a
module-level variable, so the rest of the library can simply do
``from DroneSwarm2D.core.settings import SIM_WIDTH``.

The configuration is a process-wide singleton: it must be initialised exactly once
(normally through :func:`DroneSwarm2D.init`) **before** any other module of the
library is imported, because those modules read the settings at import time.

PT-BR: Gerenciador de configuração. Carrega o JSON do cenário, avalia as
expressões e expõe cada chave como variável do módulo. É um singleton por
processo e deve ser inicializado uma única vez (via ``DroneSwarm2D.init``) antes
de importar os demais módulos da biblioteca.

Config values
-------------
* Numbers, booleans, lists and plain strings are used as-is.
* Strings are evaluated as Python expressions against the keys defined *above*
  them, e.g. ``"SIM_WIDTH": "int(FULL_WIDTH * 0.7)"``. Strings that fail to
  evaluate (``"Courier New"``) are kept as strings, and ``"None"`` becomes ``None``.
* ``FULL_WIDTH`` / ``FULL_HEIGHT`` always come from the display (or from the
  ``screen_size`` argument), whatever the file says.

.. warning::
   Expressions are evaluated with :func:`eval`. Only load configuration files you trust.

Example:
    >>> from DroneSwarm2D.core import settings
    >>> settings.initialize("config/scenario_1.json")
    >>> settings.SIM_WIDTH, settings.CENTER
"""

from __future__ import annotations

import glob
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Silence audio before pygame initialises: the simulator never plays sound and a
# missing audio device must not crash headless runs. Users can still override it.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402  (must come after the environment tweak above)

#: Screen size used for headless runs when ``screen_size`` is not given.
DEFAULT_HEADLESS_SIZE: Tuple[int, int] = (1920, 1080)


class ConfigManager:
    """Singleton that loads, evaluates and publishes the scenario configuration.

    PT-BR: Singleton que carrega, avalia e publica a configuração do cenário.

    Attributes:
        config_path: Absolute path of the loaded JSON file.
        config_name: File name without extension (used as the scenario name).
        screen: Main pygame display surface.
        clock: pygame clock used to cap the frame rate.
        headless: Whether the display is a dummy (off-screen) one.
    """

    _instance: Optional[ConfigManager] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> ConfigManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        config_path: Optional[str] = None,
        fullscreen: bool = True,
        headless: bool = False,
        screen_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Load the configuration (only on the first call).

        Args:
            config_path: Path to the JSON file. If ``None``, the first ``*.json``
                found in a ``config/`` directory is used.
            fullscreen: Open the pygame window in full-screen mode. Ignored when
                ``headless`` is true.
            headless: Use SDL's dummy video driver so no window is opened. Useful
                for batch experiments, CI and servers without a display.
            screen_size: ``(width, height)`` overriding ``FULL_WIDTH`` and
                ``FULL_HEIGHT``. Defaults to the display size, or to
                :data:`DEFAULT_HEADLESS_SIZE` in headless mode.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        if ConfigManager._initialized:
            requested = os.path.abspath(config_path) if config_path else None
            if requested and requested != self.config_path:
                warnings.warn(
                    f"DroneSwarm2D is already initialised with '{self.config_path}'; "
                    f"ignoring '{requested}'. The configuration can only be loaded once "
                    "per process.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return

        self.headless: bool = headless
        if headless:
            self._enable_dummy_video_driver()
            if screen_size is None:
                screen_size = DEFAULT_HEADLESS_SIZE

        if not pygame.get_init():
            pygame.init()

        if config_path is None:
            config_path = self._find_default_config()
        else:
            config_path = os.path.abspath(config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self.config_path: str = config_path
        self.config_name: str = os.path.splitext(os.path.basename(config_path))[0]

        print(f"[ConfigManager] Loading '{self.config_name}' from: {self.config_path}")

        self._config: Dict[str, Any] = {}
        self._load_config(screen_size)
        self._compute_derived_values()
        self._setup_display(fullscreen and not headless)
        self._export_to_module()

        ConfigManager._initialized = True
        print(f"[ConfigManager] Configuration ready with {len(self._config)} parameters")

    # ------------------------------------------------------------------ loading

    @staticmethod
    def _enable_dummy_video_driver() -> None:
        """Switch SDL to the off-screen ``dummy`` video driver."""
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            return
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        # If the display was already initialised with a real driver, restart it.
        if pygame.display.get_init():
            pygame.display.quit()
            pygame.display.init()

    @staticmethod
    def _find_default_config() -> str:
        """Return the first ``*.json`` file found in a ``config/`` directory.

        Raises:
            FileNotFoundError: If no candidate file exists.
        """
        module_dir = os.path.dirname(__file__)
        possible_paths: List[str] = [
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

        raise FileNotFoundError(f"No .json configuration found. Searched: {possible_paths}")

    def _evaluate(self, value: Any) -> Any:
        """Evaluate a string as an expression over the keys loaded so far."""
        if not isinstance(value, str):
            return value
        try:
            return eval(value, {}, self._config)  # noqa: S307 - trusted config files
        except Exception:
            return value

    def _load_config(self, screen_size: Optional[Tuple[int, int]] = None) -> None:
        """Read the JSON file and evaluate its values in declaration order."""
        with open(self.config_path, encoding="utf-8") as f:
            raw: Dict[str, Any] = json.load(f)

        if screen_size is not None:
            width, height = int(screen_size[0]), int(screen_size[1])
        else:
            display_info = pygame.display.Info()
            width, height = display_info.current_w, display_info.current_h
        raw["FULL_WIDTH"] = width
        raw["FULL_HEIGHT"] = height

        for key, val in raw.items():
            if isinstance(val, list):
                self._config[key] = [self._evaluate(item) for item in val]
            else:
                self._config[key] = self._evaluate(val)

    def _compute_derived_values(self) -> None:
        """Compute values that depend on others (centre, scenario name, DMZs)."""
        cfg = self._config
        cfg["INTEREST_POINT_CENTER"] = pygame.math.Vector2(cfg["SIM_WIDTH"] / 2, cfg["SIM_HEIGHT"] / 2)
        cfg["CENTER"] = cfg["INTEREST_POINT_CENTER"].copy()
        cfg["TYPE_OF_SCENARIO"] = self.config_name

        if "ENEMY_BEHAVIOR" in cfg:
            behavior = cfg["ENEMY_BEHAVIOR"]
            cfg["ENEMY_BEHAVIOR"] = None if behavior == "None" else behavior

        if "DMZ" in cfg:
            dmz_list: List[Tuple[float, float, int]] = []
            for expr_x, expr_y, radius in cfg["DMZ"]:
                x = float(eval(str(expr_x), {}, cfg))  # noqa: S307
                y = float(eval(str(expr_y), {}, cfg))  # noqa: S307
                dmz_list.append((x, y, int(radius)))
            cfg["DMZ"] = dmz_list

    def _setup_display(self, fullscreen: bool) -> None:
        """Create the pygame window (or off-screen surface) and the clock."""
        size = (self._config["FULL_WIDTH"], self._config["FULL_HEIGHT"])
        flags = pygame.FULLSCREEN if fullscreen else 0
        self.screen: pygame.Surface = pygame.display.set_mode(size, flags)
        self.clock: pygame.time.Clock = pygame.time.Clock()

    def _export_to_module(self) -> None:
        """Publish every key (plus ``screen`` and ``clock``) as module globals."""
        import sys

        module = sys.modules[__name__]
        for key, value in self._config.items():
            setattr(module, key, value)
        module.screen = self.screen
        module.clock = self.clock
        module.__all__ = list(self._config.keys()) + ["screen", "clock"]

    # ------------------------------------------------------------------- access

    def get(self, key: str, default: Any = None) -> Any:
        """Return ``key`` or ``default`` if it is not defined."""
        return self._config.get(key, default)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"Setting '{name}' not found in {self.config_path}")

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __contains__(self, key: object) -> bool:
        return key in self._config

    def keys(self) -> List[str]:
        """Return the names of all loaded settings."""
        return list(self._config.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of all loaded settings."""
        return self._config.copy()

    def __repr__(self) -> str:
        return f"ConfigManager(config='{self.config_name}', params={len(self._config)})"

    @classmethod
    def is_initialized(cls) -> bool:
        """Return whether the configuration was already loaded."""
        return cls._initialized

    @classmethod
    def get_instance(cls) -> Optional[ConfigManager]:
        """Return the singleton instance, or ``None`` before initialisation."""
        return cls._instance


# -----------------------------------------------------------------------------
# Module-level helpers
# -----------------------------------------------------------------------------

def initialize(
    config_path: Optional[str] = None,
    fullscreen: bool = True,
    headless: bool = False,
    screen_size: Optional[Tuple[int, int]] = None,
) -> ConfigManager:
    """Load the configuration. Call once, at the start of the application.

    PT-BR: Carrega a configuração. Chame uma única vez, no início da aplicação.

    See :class:`ConfigManager` for the meaning of the arguments.
    """
    return ConfigManager(config_path, fullscreen, headless=headless, screen_size=screen_size)


def get_config() -> ConfigManager:
    """Return the loaded :class:`ConfigManager`.

    Raises:
        RuntimeError: If :func:`initialize` was not called yet.
    """
    if not ConfigManager.is_initialized():
        raise RuntimeError(
            "ConfigManager is not initialised! Call DroneSwarm2D.init() "
            "(or settings.initialize()) at the start of the application."
        )
    return ConfigManager.get_instance()


def is_initialized() -> bool:
    """Return whether the configuration was already loaded."""
    return ConfigManager.is_initialized()


def __getattr__(name: str) -> Any:
    """Resolve ``from settings import NAME`` lazily, with a helpful error."""
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    if not ConfigManager.is_initialized():
        raise RuntimeError(
            f"ConfigManager is not initialised (while accessing '{name}').\n"
            "Fix: call DroneSwarm2D.init('path/to/config.json') before importing "
            "other DroneSwarm2D modules."
        )

    config = ConfigManager.get_instance()
    if name in config._config:
        return config._config[name]
    if name == "screen":
        return config.screen
    if name == "clock":
        return config.clock

    available = ", ".join(list(config._config.keys())[:10])
    raise AttributeError(f"Setting '{name}' not found.\nAvailable: {available}...")
