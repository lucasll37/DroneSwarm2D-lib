"""Run DroneSwarm2D episodes with the behaviours defined in ``behaviors.py``.

PT-BR: Executa episódios do DroneSwarm2D com os comportamentos de ``behaviors.py``.

Usage::

    python src/main.py                         # full-screen window, 3 episodes
    python src/main.py --windowed --episodes 5
    python src/main.py --headless --episodes 20 --seed 7   # no window, fast
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

import DroneSwarm2D

CONFIG_PATH = Path(__file__).with_name("config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DroneSwarm2D episodes.")
    parser.add_argument("--episodes", type=int, default=3, help="number of episodes (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--max-steps", type=int, default=5000, help="step limit per episode (default: 5000)")
    parser.add_argument("--headless", action="store_true", help="run without a window, as fast as possible")
    parser.add_argument("--windowed", action="store_true", help="use a window instead of full screen")
    return parser.parse_args()


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print the averages over all episodes."""
    n = len(results)
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Episodes:                  {n}")
    print(f"  Mean reward:               {sum(r['accum_reward'] for r in results) / n:8.2f}")
    print(f"  Mean enemies shot down:    {sum(r['enemies_shotdown'] for r in results) / n:6.2f}")
    print(f"  Mean friends lost:         {sum(r['friends_shotdown'] for r in results) / n:6.2f}")
    print(f"  Mean steps:                {sum(r['current_step'] for r in results) / n:6.2f}")
    print(f"  Total successful attacks:  {sum(r['sucessful_attacks'] for r in results):4d}")
    print("=" * 70)


def main() -> None:
    args = parse_args()

    # 1. Load the configuration. This must happen before importing the behaviours.
    settings = DroneSwarm2D.init(str(CONFIG_PATH), fullscreen=not args.windowed, headless=args.headless)
    from behaviors import FriendAEWBehavior, FriendCommonBehavior, FriendRadarBehavior

    # 2. Create the environment.
    env = DroneSwarm2D.AirTrafficEnv(
        max_steps=args.max_steps,
        mode=None if args.headless else "human",
        friend_behavior=FriendCommonBehavior(),
        friend_aew_behavior=FriendAEWBehavior(),
        friend_radar_behavior=FriendRadarBehavior(),
        enemy_behavior=settings.ENEMY_BEHAVIOR,
        demilitarized_zones=settings.DMZ,
        seed=args.seed,
    )

    # 3. Run the episodes.
    results: List[Dict[str, Any]] = []
    for episode in range(1, args.episodes + 1):
        obs, done = env.reset()
        while not done:
            obs, reward, done, info = env.step(None)

        print(f"Episode {episode}/{args.episodes}: steps={info['current_step']:4d} "
              f"reward={info['accum_reward']:9.2f} enemies_shot_down={info['enemies_shotdown']:2d} "
              f"successful_attacks={info['sucessful_attacks']:2d} health={info['interest_point_health']:3d}")
        results.append(info)

    # 4. Report and clean up.
    print_summary(results)
    env.close()


if __name__ == "__main__":
    main()
