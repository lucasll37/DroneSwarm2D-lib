import os
import subprocess
import sys
import textwrap

import pygame
import pytest
from conftest import CONFIG_PATH, REPO_ROOT, SCREEN_SIZE

import DroneSwarm2D
from DroneSwarm2D.core import settings as settings_module
from DroneSwarm2D.core.settings import ConfigManager, get_config, is_initialized


def run_python(code):
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True,
                          cwd=REPO_ROOT, env=env, timeout=120)


def test_screen_size_override(settings):
    assert (settings.FULL_WIDTH, settings.FULL_HEIGHT) == SCREEN_SIZE
    assert settings.screen.get_size() == SCREEN_SIZE


def test_expressions_are_evaluated_in_order(settings):
    assert settings.SIM_WIDTH == int(SCREEN_SIZE[0] * 0.7)
    assert settings.SIM_HEIGHT == SCREEN_SIZE[1]
    assert settings.GRID_WIDTH == settings.SIM_WIDTH // settings.CELL_SIZE
    assert settings.GRAPH_WIDTH == settings.FULL_WIDTH - settings.SIM_WIDTH
    assert settings.EXTERNAL_RADIUS == pytest.approx(settings.INTERNAL_RADIUS * 4)
    assert settings.NEUTRALIZATION_PROB_BOTH_DEAD == pytest.approx(
        1 - settings.NEUTRALIZATION_PROB_FRIEND_ALIVE - settings.NEUTRALIZATION_PROB_ENEMY_ALIVE)


def test_derived_values(settings):
    assert isinstance(settings.CENTER, pygame.math.Vector2)
    assert settings.CENTER == pygame.math.Vector2(settings.SIM_WIDTH / 2, settings.SIM_HEIGHT / 2)
    assert settings.INTEREST_POINT_CENTER == settings.CENTER
    assert settings.CENTER is not settings.INTEREST_POINT_CENTER
    assert settings.TYPE_OF_SCENARIO == "test_config"


def test_none_strings_and_plain_strings(settings):
    assert settings.ENEMY_BEHAVIOR is None
    assert settings.JOYSTICK is None
    assert settings.FONT_FAMILY == "Courier New"


def test_dmz_is_evaluated(settings):
    assert settings.DMZ == [(settings.SIM_WIDTH * 0.15, settings.SIM_HEIGHT * 0.2, 40)]
    x, y, r = settings.DMZ[0]
    assert isinstance(x, float) and isinstance(y, float) and isinstance(r, int)


def test_config_manager_access():
    config = get_config()
    assert is_initialized() and ConfigManager.is_initialized()
    assert config["SIM_WIDTH"] == config.SIM_WIDTH == settings_module.SIM_WIDTH
    assert config.get("DOES_NOT_EXIST", 5) == 5
    assert "SIM_WIDTH" in config and "DOES_NOT_EXIST" not in config
    assert "CELL_SIZE" in config.keys()
    copy = config.to_dict()
    copy["SIM_WIDTH"] = -1
    assert config.SIM_WIDTH != -1
    assert "test_config" in repr(config)


def test_missing_setting_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = settings_module.DOES_NOT_EXIST
    with pytest.raises(AttributeError):
        _ = get_config().DOES_NOT_EXIST


def test_init_twice_returns_same_settings_and_warns_on_other_file(tmp_path):
    assert DroneSwarm2D.init(str(CONFIG_PATH)) is settings_module
    other = tmp_path / "other.json"
    other.write_text("{}")
    with pytest.warns(RuntimeWarning, match="already initialised"):
        DroneSwarm2D.init(str(other))
    assert settings_module.TYPE_OF_SCENARIO == "test_config"


def test_access_before_init_raises_helpful_error():
    result = run_python("""
        from DroneSwarm2D.core import settings
        try:
            settings.SIM_WIDTH
        except RuntimeError as exc:
            print("RuntimeError:", exc)
    """)
    assert "RuntimeError: ConfigManager is not initialised" in result.stdout, result.stderr


def test_missing_config_file_raises():
    result = run_python("""
        import DroneSwarm2D
        try:
            DroneSwarm2D.init("does-not-exist.json", headless=True)
        except FileNotFoundError as exc:
            print("FileNotFoundError:", exc)
    """)
    assert "FileNotFoundError: Configuration file not found" in result.stdout, result.stderr


def test_classes_are_none_before_init():
    result = run_python("""
        import DroneSwarm2D
        print(DroneSwarm2D.AirTrafficEnv is None, DroneSwarm2D.is_initialized())
    """)
    assert result.stdout.strip().endswith("True False"), result.stderr
