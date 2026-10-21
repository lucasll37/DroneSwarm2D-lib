import json
import os
import subprocess
import sys

import numpy as np
import pygame
import pytest
from conftest import REPO_ROOT

from DroneSwarm2D import __version__, cli
from DroneSwarm2D.tools.create_video import create_video_from_frames, get_frame_number

TEMPLATE_SRC = REPO_ROOT / "DroneSwarm2D" / "templates" / "droneSwarm2D-project" / "src"
EXAMPLE_SRC = REPO_ROOT / "example" / "src"


# ----------------------------------------------------------------------------- CLI

def test_init_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.init_project("demo")
    for name in ("src/main.py", "src/behaviors.py", "src/config.json", "README.md", "TUTORIAL.md"):
        assert (tmp_path / "demo" / name).is_file()
    assert not cli.init_project("demo")  # already exists
    assert not cli.init_project("other", template="nope")


def test_cli_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["init", "--name", "named"])
    assert exit_info.value.code == 0 and (tmp_path / "named").is_dir()

    cli.main([])  # no command: init with the defaults
    assert (tmp_path / "droneSwarm2D-project").is_dir()

    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert __version__ in capsys.readouterr().out

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["create_video", str(tmp_path / "missing")])
    assert exit_info.value.code == 1


def test_example_is_in_sync_with_the_template():
    for name in ("main.py", "behaviors.py"):
        assert (EXAMPLE_SRC / name).read_text() == (TEMPLATE_SRC / name).read_text(), name
    example = json.loads((EXAMPLE_SRC / "config.json").read_text())
    template = json.loads((TEMPLATE_SRC / "config.json").read_text())
    assert example.keys() == template.keys()
    assert example["JOYSTICK"] == template["JOYSTICK"] == "None"


def test_example_script_runs_headless():
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"}
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_SRC / "main.py"), "--headless", "--episodes", "2", "--max-steps", "40"],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "Episode 2/2" in result.stdout and "SUMMARY" in result.stdout


# ----------------------------------------------------------------------------- video

def test_get_frame_number():
    assert get_frame_number("frame_12.png") == 12
    assert get_frame_number("frame_7_final.png") == 7
    assert get_frame_number("shot-0042.png") == 42
    with pytest.raises(ValueError):
        get_frame_number("cover.png")


def write_frames(directory, count):
    directory.mkdir(parents=True)
    for n in range(count):
        surface = pygame.Surface((64, 48))
        surface.fill((n * 40 % 255, 0, 0))
        pygame.image.save(surface, str(directory / f"frame_{n}.png"))


def test_create_video(tmp_path):
    frames = tmp_path / "frames_20250101_000000-my-scenario"
    write_frames(frames, 4)
    pygame.image.save(pygame.Surface((64, 48)), str(frames / "cover.png"))  # skipped: no number

    video = create_video_from_frames(str(frames), output_dir=str(tmp_path / "video"), fps=10, remove_frames=True)
    assert video == str(tmp_path / "video" / "my-scenario.mp4")
    assert os.path.getsize(video) > 0
    assert not frames.exists()


def test_create_video_without_frames(tmp_path):
    assert create_video_from_frames(str(tmp_path / "missing")) is None
    (tmp_path / "empty").mkdir()
    assert create_video_from_frames(str(tmp_path / "empty")) is None


# ----------------------------------------------------------------------------- public API

def test_public_api_after_init():
    import DroneSwarm2D

    assert DroneSwarm2D.is_initialized()
    for name in DroneSwarm2D.__all__:
        assert getattr(DroneSwarm2D, name) is not None, name
    assert DroneSwarm2D.InterestPoint is DroneSwarm2D.CircleInterestPoint
    assert DroneSwarm2D.DemilitarizedZone is DroneSwarm2D.CircleDMZ


def test_backward_compatible_imports():
    from DroneSwarm2D.behaviorsDefault import (  # noqa: F401
        BaseBehavior,
        BehaviorType,
        FriendAEWBehaviorDefault,
        FriendCommonBehaviorDefault,
        FriendRadarBehaviorDefault,
    )
    from DroneSwarm2D.cli import create_video_from_frames as cli_video  # noqa: F401
    from DroneSwarm2D.core.AirTrafficEnv import (  # noqa: F401
        AirTrafficEnv,
        Button,
        compute_global_matrices,
        draw_direction,
        draw_friend_communication,
        draw_grid,
        draw_heatmap,
    )
    from DroneSwarm2D.core.FriendDrone import FriendDrone
    from DroneSwarm2D.core.utils import (  # noqa: F401
        can_intercept,
        draw_dashed_circle,
        draw_dashed_line,
        generate_sparse_matrix,
        get_friends_hold,
        intercept_direction,
        load_svg_as_surface,
        plot_individual_states_matplotlib,
        pos_to_cell,
        sim_to_geo,
        smooth_matrix_with_kernel_10x10,
        symmetrical_flat_topped_gaussian_10x10,
    )

    assert hasattr(FriendDrone, "update_neghbors") and hasattr(FriendDrone, "update_neighbors")
    assert isinstance(np.zeros(1), np.ndarray)
