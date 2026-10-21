"""Shared fixtures.

The library configuration is a process-wide singleton, so it is loaded once for
the whole test session: headless (SDL dummy driver) with a fixed 1280x720
virtual screen, from ``tests/fixtures/test_config.json``.
"""

import os
import random
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402
import pytest  # noqa: E402

import DroneSwarm2D  # noqa: E402

TESTS_DIR = Path(__file__).parent
REPO_ROOT = TESTS_DIR.parent
CONFIG_PATH = TESTS_DIR / "fixtures" / "test_config.json"
SCREEN_SIZE = (1280, 720)

SETTINGS = DroneSwarm2D.init(str(CONFIG_PATH), fullscreen=False, headless=True, screen_size=SCREEN_SIZE)

from DroneSwarm2D.behaviorsDefault import FriendCommonBehaviorDefault  # noqa: E402
from DroneSwarm2D.core.EnemyDrone import EnemyDrone  # noqa: E402
from DroneSwarm2D.core.FriendDrone import FriendDrone  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    """The loaded ``DroneSwarm2D.core.settings`` module."""
    return SETTINGS


@pytest.fixture(autouse=True)
def isolated_drone_classes():
    """Reset the class-level counters, clocks and RNGs shared by all drones."""
    FriendDrone.friend_id_counter = 0
    FriendDrone.reset_clock()
    FriendDrone.set_class_seed(0)
    EnemyDrone.enemy_id_counter = 0
    EnemyDrone.set_class_seed(0)
    yield


@pytest.fixture
def make_env():
    """Factory of headless, already-reset environments."""

    def _make(**kwargs):
        kwargs.setdefault("mode", None)
        kwargs.setdefault("seed", 123)
        kwargs.setdefault("demilitarized_zones", SETTINGS.DMZ)
        env = DroneSwarm2D.AirTrafficEnv(**kwargs)
        env.reset()
        return env

    return _make


@pytest.fixture
def make_friend():
    """Factory of friendly drones around the interest point."""

    def _make(position, behavior=None, **kwargs):
        if behavior is None:
            behavior = FriendCommonBehaviorDefault()
        return FriendDrone(SETTINGS.CENTER, position, behavior=behavior, **kwargs)

    return _make


def fake_target(x, y):
    """Minimal object detectable by a FriendDrone (only ``.pos`` is used)."""
    return SimpleNamespace(pos=pygame.math.Vector2(x, y))


class FixedRandom(random.Random):
    """``random.Random`` whose ``random()`` always returns the same value."""

    def __init__(self, value):
        super().__init__(0)
        self.value = value

    def random(self):
        return self.value
