import os
import gc
import pandas as pd
import datetime
import sys

from pathlib import Path

import DroneSwarm2D
settings = DroneSwarm2D.init(config_path="./config/benchmark_formation_asym_m.json", fullscreen=True)

# Adiciona o diretório raiz ao Python path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.planning.behaviors import FriendCommonBehavior, FriendAEWBehavior, FriendRadarBehavior # <-- PLANNING MODULE
from src.imitation_learning.behaviors import FriendCommonBehaviorAI, FriendBenchmarkBehavior # <-- IMITATION LEARNING MODULE


def persist_episode_result(result: dict, csv_path: str) -> None:
    """
    Persists the result of an episode to a CSV file using pandas.
    
    Args:
        result (dict): A dictionary containing the episode results.
        csv_path (str): Path to the CSV file.
    """
    df = pd.DataFrame([result])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False, mode='w')
    else:
        df.to_csv(csv_path, index=False, mode='a', header=False)


def main() -> None:

    # Record start time and create a timestamped save folder.
    start_time = datetime.datetime.now()
    timestamp = start_time.strftime("%Y_%m_%d_%Hh%Mm%Ss")
    save_folder = os.path.join("./analysis/data", settings.TYPE_OF_SCENARIO)
    os.makedirs(save_folder, exist_ok=True)
    
    # Define the CSV file path for saving episode results.
    csv_path = os.path.join(save_folder, F"results_{timestamp}.csv")
        
    env = DroneSwarm2D.AirTrafficEnv(
        mode=None,
        friend_behavior=FriendBenchmarkBehavior(),
        # friend_behavior=FriendCommonBehavior(),
        friend_aew_behavior = FriendAEWBehavior(),
        friend_radar_behavior = FriendRadarBehavior(),
        enemy_behavior=settings.ENEMY_BEHAVIOR,
        demilitarized_zones=settings.DMZ,
        seed=42
    )
    
    print("✓ Ambiente criado com sucesso\n")
    print(f"Executando {settings.ANALYSIS_EPISODES} episódios...\n")
    
    
    # Run episodes and persist results.
    for episode in range(settings.ANALYSIS_EPISODES):        
        obs, done = env.reset()
        total_reward: float = 0.0

        while not done:
            action = None  # Implement your action logic here if needed.
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
        # Gather episode statistics.
        n_steps: int = info['current_step']
        accum_reward: float = info['accum_reward']
        enemies_shotdown: int = info['enemies_shotdown']
        friends_shotdown: int = info['friends_shotdown']
        sucessful_attacks: int = info['sucessful_attacks']
        interest_point_health: int = info['interest_point_health']
        state_percentages = info['state_percentages']
        total_distance = info['total_distance_traveled']

        result = {
            "episode": episode,
            "steps": n_steps,
            "accumulated_reward": accum_reward,
            "enemies_shotdown": enemies_shotdown,
            "friends_shotdown": friends_shotdown,
            "sucessful_attacks": sucessful_attacks,
            "interest_point_health": interest_point_health,
            "total_distance_traveled": total_distance
        }

        all_states = [
            "PURSUING",
            "HOLD - WAIT",
            "HOLD - RETURN",
            "HOLD - NO ENOUGH COMM",
            "HOLD - INTCPT",
            "GO HOLD INTCPT",
            "HOLD - SPREAD",
            "HOLD - JOYSTICK",
            "HOLD - NO JOYSTICK",
            "HOLD - BC",
            "HOLD - BENCHMARK"
        ]
        
        for state in all_states:
            percentage = state_percentages.get(state, 0)
            result[f"state {state}"] = percentage
        
        # Persist the episode result immediately.
        persist_episode_result(result, csv_path)
        
        print("-" * 50)
        print(f"FINAL: Air Traffic Env Episode {episode+1:3d}\n")
        print(f"\tSteps: {n_steps:4d}")
        print(f"\tAccumulated Reward: {accum_reward:7.3f}")
        print(f"\tEnemies Shotdown: {enemies_shotdown}")
        print(f"\tFriends Shotdown: {friends_shotdown}")
        print(f"\tSuccessful Attacks: {sucessful_attacks:3d}")
        print(f"\tInterest Point Health: {interest_point_health}")
        print(f"\tTotal Distance Traveled: {total_distance:.2f} px")
        print("\tState Percentages:")
        
        for state, percentage in state_percentages.items():
            print(f"\t\t{state}: {percentage:.2f}%")
            
        gc.collect()
        
    print("-" * 50)
        
    # Close the environment.
    env.close()

# -----------------------------------------------------------------------------
# Run the Simulation if Executed as Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()