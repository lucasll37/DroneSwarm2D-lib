"""
main.py

Script principal para executar simulação DroneSwarm2D.

Este módulo executa múltiplos episódios de simulação com
comportamentos customizados e exibe estatísticas detalhadas.
"""

from typing import List, Dict, Any
import DroneSwarm2D

# Inicializar configurações
settings = DroneSwarm2D.init(
    config_path="./src/config.json",
    fullscreen=True
)

from behaviors import FriendCommonBehavior, FriendRadarBehavior, FriendAEWBehavior


def main() -> None:
    """
    Executa simulação de enxame de drones.
    
    Realiza múltiplos episódios de simulação, coleta estatísticas
    e exibe resultados agregados ao final.
    """
    print("=" * 70)
    print("Simulação DroneSwarm2D")
    print("=" * 70)
    
    # Criar ambiente de simulação
    env = DroneSwarm2D.AirTrafficEnv(
        mode='human',
        friend_behavior=FriendCommonBehavior(),
        friend_aew_behavior=FriendAEWBehavior(),
        friend_radar_behavior=FriendRadarBehavior(),
        enemy_behavior=settings.ENEMY_BEHAVIOR,
        demilitarized_zones=settings.DMZ,
        seed=42
    )
    
    print("✓ Ambiente criado com sucesso\n")
    
    # Configuração de episódios
    NUM_EPISODES: int = 3
    
    print(f"Executando {NUM_EPISODES} episódios...\n")
    
    results: List[Dict[str, Any]] = []
    
    for episode in range(NUM_EPISODES):
        # Resetar ambiente
        obs, done = env.reset()
        episode_reward: float = 0.0
        
        print(f"→ Episódio {episode + 1}/{NUM_EPISODES} iniciado...")
        
        # Loop principal do episódio
        while not done:
            obs, reward, done, info = env.step(None)
            episode_reward += reward
        
        # Exibir estatísticas do episódio
        print(f"  ✓ Finalizado!")
        print(f"    - Steps: {info['current_step']:4d}")
        print(f"    - Reward: {info['accum_reward']:8.2f}")
        print(f"    - Inimigos abatidos: {info['enemies_shotdown']:2d}")
        print(f"    - Ataques bem-sucedidos: {info['sucessful_attacks']:2d}")
        print(f"    - IP Health: {info['interest_point_health']:3d}")
        print()
        
        results.append(info)
    
    # Calcular e exibir estatísticas finais
    _print_final_statistics(results)
    
    # Finalizar ambiente
    env.close()
    print("\n✓ Simulação concluída com sucesso!")


def _print_final_statistics(results: List[Dict[str, Any]]) -> None:
    """
    Imprime estatísticas agregadas dos episódios.
    
    Args:
        results: Lista de dicionários info de cada episódio
    """
    print("=" * 70)
    print("ESTATÍSTICAS FINAIS")
    print("=" * 70)
    
    # Calcular médias
    avg_reward: float = sum(r['accum_reward'] for r in results) / len(results)
    avg_enemies: float = sum(r['enemies_shotdown'] for r in results) / len(results)
    avg_steps: float = sum(r['current_step'] for r in results) / len(results)
    total_attacks: int = sum(r['sucessful_attacks'] for r in results)
    
    # Exibir resultados
    print(f"  Episódios executados:      {len(results)}")
    print(f"  Reward médio:              {avg_reward:8.2f}")
    print(f"  Inimigos abatidos (média): {avg_enemies:6.2f}")
    print(f"  Steps médio:               {avg_steps:6.2f}")
    print(f"  Total de ataques:          {total_attacks:4d}")
    print("=" * 70)


if __name__ == "__main__":
    main()