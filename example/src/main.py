import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Agora pode importar normalmente
import DroneSwarm2D
settings = DroneSwarm2D.init(
    config_path="./example/src/config.json",
    fullscreen=True
)

from behaviors import FriendCommonBehavior, FriendRadarBehavior, FriendAEWBehavior

def main():
    """Executa simulação de enxame de drones."""
    print("=" * 70)
    print("Simulação DroneSwarm2D")
    print("=" * 70)
    
    env = DroneSwarm2D.AirTrafficEnv(
        mode='human',
        friend_behavior=FriendCommonBehavior(),
        friend_aew_behavior = FriendAEWBehavior(),
        friend_radar_behavior = FriendRadarBehavior(),
        enemy_behavior=settings.ENEMY_BEHAVIOR,
        demilitarized_zones=settings.DMZ,
        seed=42
    )
    
    print("✓ Ambiente criado com sucesso\n")
    
    # PASSO 3: Executar simulação
    NUM_EPISODES = 3
    
    print(f"Executando {NUM_EPISODES} episódios...\n")
    
    results = []
    for episode in range(NUM_EPISODES):
        obs, done = env.reset()
        episode_reward = 0.0
        
        print(f"→ Episódio {episode + 1}/{NUM_EPISODES} iniciado...")
        
        # Loop do episódio
        while not done:
            obs, reward, done, info = env.step(None)
            episode_reward += reward
        
        # Exibir estatísticas
        print(f"  ✓ Finalizado!")
        print(f"    - Steps: {info['current_step']:4d}")
        print(f"    - Reward: {info['accum_reward']:8.2f}")
        print(f"    - Inimigos abatidos: {info['enemies_shotdown']:2d}")
        print(f"    - Ataques bem-sucedidos: {info['sucessful_attacks']:2d}")
        print(f"    - IP Health: {info['interest_point_health']:3d}")
        print()
        
        results.append(info)
    
    # PASSO 4: Estatísticas finais
    print("=" * 70)
    print("ESTATÍSTICAS FINAIS")
    print("=" * 70)
    
    avg_reward = sum(r['accum_reward'] for r in results) / len(results)
    avg_enemies = sum(r['enemies_shotdown'] for r in results) / len(results)
    avg_steps = sum(r['current_step'] for r in results) / len(results)
    total_attacks = sum(r['sucessful_attacks'] for r in results)
    
    print(f"  Episódios executados:      {len(results)}")
    print(f"  Reward médio:              {avg_reward:8.2f}")
    print(f"  Inimigos abatidos (média): {avg_enemies:6.2f}")
    print(f"  Steps médio:               {avg_steps:6.2f}")
    print(f"  Total de ataques:          {total_attacks:4d}")
    print("=" * 70)
    
    # PASSO 5: Finalizar
    env.close()
    print("\n✓ Simulação concluída com sucesso!")


if __name__ == "__main__":
    main()