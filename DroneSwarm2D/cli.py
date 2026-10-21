"""Command-line interface: ``python -m DroneSwarm2D <command>``.

PT-BR: Interface de linha de comando: ``python -m DroneSwarm2D <comando>``.

Commands:
    init          Create a starter project (default when no command is given).
    create_video  Encode saved PNG frames into an MP4 video.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .tools.create_video import create_video_from_frames


def get_template_dir() -> Path:
    """Directory holding the project templates shipped with the package."""
    return Path(__file__).parent / "templates"


def init_project(
    project_name: str = "droneSwarm2D-project",
    template: str = "droneSwarm2D-project",
) -> bool:
    """Copy a project template into ``./<project_name>``.

    PT-BR: Copia um template de projeto para ``./<project_name>``.

    Args:
        project_name: Name of the directory to create (must not exist).
        template: Template name (a directory inside ``DroneSwarm2D/templates``).

    Returns:
        ``True`` on success, ``False`` otherwise (an error message is printed).
    """
    project_path = Path.cwd() / project_name
    if project_path.exists():
        print(f"\n❌ Error: directory '{project_name}' already exists!\n")
        return False

    template_dir = get_template_dir() / template
    if not template_dir.is_dir():
        print(f"\n❌ Error: template '{template}' not found in {get_template_dir()}!\n")
        return False

    print(f"\n📁 Creating project '{project_name}'...")
    try:
        shutil.copytree(template_dir, project_path, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    except OSError as exc:
        print(f"\n❌ Error while creating the project: {exc}\n")
        return False

    print(f"✅ Project created at: {project_path}")
    print("\n📝 Next steps:")
    print(f"\tcd {project_name}")
    print("\tpython src/main.py\n")
    return True


def create_video_command(args: argparse.Namespace) -> None:
    """Run the ``create_video`` command and exit with its status."""
    print(f"\n🎬 Creating video from the frames in: {args.path}")
    video_path = create_video_from_frames(
        frames_dir=args.path,
        output_dir=args.output_dir,
        fps=args.fps,
        codec=args.codec,
        remove_frames=args.remove_frames,
    )
    if video_path:
        print(f"✅ Video created: {video_path}\n")
        sys.exit(0)
    print("\n❌ Error: could not create the video.\n")
    sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of the CLI."""
    parser = argparse.ArgumentParser(
        prog="DroneSwarm2D",
        description="DroneSwarm2D - defensive drone swarm simulator",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    init_parser = subparsers.add_parser("init", help="Create a starter DroneSwarm2D project")
    init_parser.add_argument("--name", default="droneSwarm2D-project",
                             help="Project directory name (default: droneSwarm2D-project)")
    init_parser.add_argument("--template", default="droneSwarm2D-project",
                             help="Template to use (default: droneSwarm2D-project)")

    video_parser = subparsers.add_parser("create_video", help="Create an MP4 video from PNG frames")
    video_parser.add_argument("path", help="Directory containing the PNG frames")
    video_parser.add_argument("--output-dir", default="./video", help="Output directory (default: ./video)")
    video_parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    video_parser.add_argument("--codec", default="libx264", help="Video codec (default: libx264)")
    video_parser.add_argument("--remove-frames", action="store_true",
                              help="Delete the frames directory after creating the video")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point of ``python -m DroneSwarm2D``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        print("\n📦 No command given. Running 'init' with the default options...\n")
        init_project()
        return

    if args.command == "init":
        sys.exit(0 if init_project(project_name=args.name, template=args.template) else 1)
    elif args.command == "create_video":
        create_video_command(args)
    else:  # pragma: no cover - argparse rejects unknown commands
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
