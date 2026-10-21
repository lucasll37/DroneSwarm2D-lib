"""Measure the simulation speed of DroneSwarm2D.

PT-BR: Mede a velocidade da simulação do DroneSwarm2D.

The scenario is the example configuration with the drone counts given on the
command line. The configuration can only be loaded once per process, so each
run of this script measures one scenario.

Usage::

    python benchmarks/benchmark.py                       # 24 vs 24, headless
    python benchmarks/benchmark.py --friends 50 --enemies 50 --passive
    python benchmarks/benchmark.py --render              # include drawing (off-screen)
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "example" / "src" / "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--friends", type=int, default=24)
    parser.add_argument("--enemies", type=int, default=24)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--passive", action="store_true", help="use passive detection (triangulation)")
    parser.add_argument("--render", action="store_true", help="render every frame (off-screen, no FPS cap)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(BASE_CONFIG.read_text())
    config.update({
        "FRIEND_COUNT": args.friends, "ENEMY_COUNT": args.enemies,
        "RADAR_COUNT": 2, "AEW_COUNT": 2, "BROKEN_COUNT": 2, "RADAR_RANGE": 300,
        "PASSIVE_DETECTION": args.passive, "JOYSTICK": "None",
    })

    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "benchmark.json"
        config_path.write_text(json.dumps(config))

        import DroneSwarm2D

        settings = DroneSwarm2D.init(str(config_path), headless=True, screen_size=(1920, 1080))
        env = DroneSwarm2D.AirTrafficEnv(mode="human" if args.render else None, seed=args.seed,
                                         enemy_behavior=settings.ENEMY_BEHAVIOR, demilitarized_zones=settings.DMZ)
        env.render_fps = 0  # no frame cap

        times = []
        env.reset()
        for _ in range(args.steps):
            start = time.perf_counter()
            _, _, done, _ = env.step(None)
            times.append(time.perf_counter() - start)
            if done:
                env.reset()

    mode = "passive" if args.passive else "direct"
    print(f"{args.friends} friends vs {args.enemies} enemies, {mode} detection, render={args.render}, "
          f"{len(times)} steps")
    print(f"  mean   {1000 * statistics.mean(times):7.2f} ms/step")
    print(f"  median {1000 * statistics.median(times):7.2f} ms/step")
    print(f"  ~{1 / statistics.mean(times):6.1f} steps/s")


if __name__ == "__main__":
    sys.exit(main())
