"""Build an MP4 video from the PNG frames saved by the simulator.

PT-BR: Gera um vídeo MP4 a partir dos quadros PNG salvos pelo simulador.

Frames are saved by the "Tog. Save Frames" button as
``./tmp/frames_<timestamp>-<scenario>/frame_<n>.png``. The video is named after
the scenario: ``<output_dir>/<scenario>.mp4``.

Usage::

    python -m DroneSwarm2D create_video ./tmp/frames_20250101_120000-scenario
    python -m DroneSwarm2D.tools.create_video        # newest folder in ./tmp
"""

from __future__ import annotations

import os
import re
import shutil
from typing import List, Optional

import imageio.v2 as imageio


def get_frame_number(filename: str) -> int:
    """Extract the frame number from a file name such as ``frame_12.png``.

    Args:
        filename: File name (``<prefix>_<number>.png``).

    Returns:
        The frame number.

    Raises:
        ValueError: If the name contains no number.
    """
    try:
        return int(filename.split("_")[1].split(".")[0])
    except (IndexError, ValueError):
        digits = re.findall(r"\d+", filename)
        if not digits:
            raise ValueError(f"No frame number in {filename!r}") from None
        return int(digits[-1])


def _sorted_frames(frames_dir: str) -> List[str]:
    """PNG files of ``frames_dir`` sorted by frame number (unnumbered files are skipped)."""
    numbered = []
    for name in os.listdir(frames_dir):
        if not name.endswith(".png"):
            continue
        try:
            numbered.append((get_frame_number(name), name))
        except ValueError:
            print(f"Skipping '{name}': no frame number.")
    return [name for _, name in sorted(numbered)]


def create_video_from_frames(
    frames_dir: str,
    output_dir: str = "./video",
    fps: int = 30,
    codec: str = "libx264",
    remove_frames: bool = False,
) -> Optional[str]:
    """Encode the PNG frames of ``frames_dir`` into ``<output_dir>/<scenario>.mp4``.

    PT-BR: Codifica os quadros PNG de ``frames_dir`` em ``<output_dir>/<cenário>.mp4``.

    Args:
        frames_dir: Directory with ``frame_<n>.png`` files.
        output_dir: Where to write the video (created if needed).
        fps: Frames per second.
        codec: FFmpeg codec.
        remove_frames: Delete ``frames_dir`` after a successful encoding.

    Returns:
        Path of the video, or ``None`` if there was nothing to encode.
    """
    if not os.path.isdir(frames_dir):
        print(f"Error: Directory '{frames_dir}' does not exist.")
        return None

    filenames = _sorted_frames(frames_dir)
    if not filenames:
        print(f"Error: No PNG files found in '{frames_dir}'.")
        return None

    # "frames_<timestamp>-<scenario>" -> "<scenario>"
    folder_name = os.path.basename(os.path.normpath(frames_dir))
    scenario = folder_name.split("-", 1)[-1]

    os.makedirs(output_dir, exist_ok=True)
    video_filename = os.path.join(output_dir, f"{scenario}.mp4")

    with imageio.get_writer(video_filename, fps=fps, codec=codec) as writer:
        for name in filenames:
            try:
                writer.append_data(imageio.imread(os.path.join(frames_dir, name)))
            except Exception as exc:  # A corrupt frame must not abort the whole video.
                print(f"Error processing {name}: {exc}")

    print(f"Video saved as: {video_filename}")

    if remove_frames:
        try:
            shutil.rmtree(frames_dir)
            print(f"Frames directory '{frames_dir}' removed.")
        except OSError as exc:
            print(f"Error removing frames directory: {exc}")

    return video_filename


def main() -> None:
    """Encode the newest frames folder found in ``./tmp``."""
    tmp_dir = "./tmp"
    if not os.path.isdir(tmp_dir):
        print(f"Error: Directory '{tmp_dir}' does not exist.")
        return

    folders = sorted(f for f in os.listdir(tmp_dir) if os.path.isdir(os.path.join(tmp_dir, f)))
    if not folders:
        print(f"Error: No folders found in '{tmp_dir}'.")
        return

    create_video_from_frames(frames_dir=os.path.join(tmp_dir, folders[-1]), output_dir="./video")


if __name__ == "__main__":
    main()
