# type: ignore
import os
import sys
import numpy as np
import pygame
import tensorflow as tf
import time
from typing import Dict

# Agora pode importar normalmente
import DroneSwarm2D
settings = DroneSwarm2D.init(
    config_path="./config/proposal_spread.json",
    fullscreen=True
)

# Ensure HOLD_SPREAD is set to True for compatibility 
settings.HOLD_SPREAD = True 

# Initialize pygame if not already initialized
if not pygame.get_init():
    pygame.init()

from DroneSwarm2D.core.utils import generate_sparse_matrix, get_friends_hold

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from planning.behaviors import FriendCommonBehavior

agent = FriendCommonBehavior()

def create_and_save_behavior_dataset(num_samples=10000, save_path="behavior_dataset", batch_size=16):
    """
    Cria um dataset de comportamento com tamanho fixo, feedback detalhado e salva em disco.
    
    Args:
        num_samples (int): Número de amostras a gerar
        save_path (str): Caminho onde salvar o dataset
        batch_size (int): Tamanho do batch
    
    Returns:
        tf.data.Dataset: Dataset criado
    """
    print("="*60)
    print(f"🎯 INICIANDO GERAÇÃO DE DATASET")
    print(f"📊 Meta: {num_samples:,} amostras válidas")
    print(f"📁 Destino: {save_path}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🎲 Filtro: 'HOLD - SPREAD' apenas")
    print("="*60)
    
    start_time = time.time()
    
    def generator():
        count = 0
        valid_samples = 0
        last_report_time = time.time()
        last_report_samples = 0
        
        # Configurações de feedback
        report_interval = max(100, num_samples // 50)  # Adaptativo baseado no tamanho
        time_report_interval = 15  # Reportar a cada 15 segundos
        
        while valid_samples < num_samples:
            count += 1
            
            # Generate a random position within the simulation boundaries
            pos = np.array([
                np.random.uniform(0, settings.SIM_WIDTH),
                np.random.uniform(0, settings.SIM_HEIGHT)
            ], dtype=np.float32)
            
            # Generate sparse matrices for intensities and directions
            friend_intensity, friend_direction = generate_sparse_matrix(max_nonzero = 40)
            enemy_intensity, enemy_direction = generate_sparse_matrix(max_nonzero = 40)
            
            # Organize the state into a dictionary
            state: Dict = {
                'pos': np.array(pos, dtype=np.float32),
                'friend_intensity': np.array(friend_intensity, dtype=np.float32),
                'enemy_intensity': np.array(enemy_intensity, dtype=np.float32),
                'friend_direction': np.array(friend_direction, dtype=np.float32),
                'enemy_direction': np.array(enemy_direction, dtype=np.float32)
            }
            
            # Compute the action using the class method of planning policy
            info, direction = agent.apply(state)
            str_state, _, _, _ = info
            
            # Skip invalid samples
            if str_state != 'HOLD - SPREAD':
                continue
            
            # Convert the action to a NumPy array with float32 type
            action = np.array(direction, dtype=np.float32)
            matrix_friends_hold = np.zeros((settings.GRID_WIDTH, settings.GRID_HEIGHT), dtype=np.float32)

            for cell, _ in get_friends_hold(state):
                matrix_friends_hold[cell] = 1.0
            
            # Filter state to keep only desired keys
            filtered_state = {'pos': pos, 'friends_hold': matrix_friends_hold}
            
            valid_samples += 1
            current_time = time.time()
            
            # Feedback detalhado por número de amostras
            if valid_samples % report_interval == 0:
                elapsed_time = current_time - start_time
                samples_per_sec = valid_samples / elapsed_time if elapsed_time > 0 else 0
                acceptance_rate = (valid_samples / count) * 100 if count > 0 else 0
                eta_seconds = (num_samples - valid_samples) / samples_per_sec if samples_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60
                
                print(f"\n📈 PROGRESSO: {valid_samples:,}/{num_samples:,} ({valid_samples/num_samples*100:.1f}%)")
                print(f"🎲 Tentativas: {count:,} (taxa de aceitação: {acceptance_rate:.2f}%)")
                print(f"⚡ Velocidade: {samples_per_sec:.1f} amostras/seg")
                print(f"⏱️  Decorrido: {elapsed_time/60:.1f} min | ETA: {eta_minutes:.1f} min")
                print("-" * 50)
            
            # Feedback por tempo (mais frequente, menos verboso)
            elif current_time - last_report_time >= time_report_interval:
                samples_since_last = valid_samples - last_report_samples
                time_since_last = current_time - last_report_time
                recent_speed = samples_since_last / time_since_last if time_since_last > 0 else 0
                acceptance_rate = (valid_samples / count) * 100 if count > 0 else 0
                
                print(f"⏰ {valid_samples:,}/{num_samples:,} amostras | "
                      f"Velocidade: {recent_speed:.1f}/s | "
                      f"Taxa: {acceptance_rate:.1f}% | "
                      f"Tentativas: {count:,}")
                
                last_report_time = current_time
                last_report_samples = valid_samples
            
            yield filtered_state, action
        
        # Estatísticas finais
        total_time = time.time() - start_time
        final_acceptance_rate = (valid_samples / count) * 100 if count > 0 else 0
        avg_speed = valid_samples / total_time if total_time > 0 else 0
        
        print("\n" + "="*60)
        print("🎉 GERAÇÃO CONCLUÍDA COM SUCESSO!")
        print(f"✅ Amostras válidas geradas: {valid_samples:,}")
        print(f"🎲 Tentativas totais: {count:,}")
        print(f"📊 Taxa de aceitação final: {final_acceptance_rate:.2f}%")
        print(f"⚡ Velocidade média: {avg_speed:.1f} amostras/seg")
        print(f"⏱️  Tempo total: {total_time/60:.1f} minutos")
        print(f"💾 Salvando em: {save_path}")
        print("="*60)
    
    # Create tf.data.Dataset from generator
    print("\n🔧 Criando TensorFlow Dataset...")
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            {
                'pos': tf.TensorSpec(shape=(2,), dtype=tf.float32),
                'friends_hold': tf.TensorSpec(shape=(settings.GRID_WIDTH, settings.GRID_HEIGHT), dtype=tf.float32)
            },
            tf.TensorSpec(shape=(2,), dtype=tf.float32)
        )
    )
    
    # Configure batching and prefetching
    print(f"📦 Configurando batches de tamanho {batch_size}...")
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Save dataset to disk
    print(f"\n💾 Salvando dataset em disco...")
    save_start_time = time.time()
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    tf.data.Dataset.save(dataset, save_path)
    
    save_time = time.time() - save_start_time
    print(f"✅ Dataset salvo com sucesso! (tempo de salvamento: {save_time:.1f}s)")
    
    return dataset


def load_behavior_dataset(save_path="./data/behaviorCloneDataset"):
    """
    Carrega um dataset salvo do disco com feedback.
    
    Args:
        save_path (str): Caminho do dataset salvo
    
    Returns:
        tf.data.Dataset: Dataset carregado
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"❌ Dataset não encontrado em: {save_path}")
    
    print(f"📂 Carregando dataset de: {save_path}")
    load_start_time = time.time()
    
    # Load the dataset
    dataset = tf.data.Dataset.load(
        save_path,
        element_spec=(
            {
                'pos': tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
                'friends_hold': tf.TensorSpec(shape=(None, settings.GRID_WIDTH, settings.GRID_HEIGHT), dtype=tf.float32)
            },
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
        )
    )
    
    load_time = time.time() - load_start_time
    print(f"✅ Dataset carregado com sucesso! (tempo: {load_time:.1f}s)")
    return dataset


def inspect_dataset(dataset, num_batches=3):
    """
    Inspeciona o dataset para verificar sua estrutura com feedback melhorado.
    
    Args:
        dataset: Dataset a ser inspecionado
        num_batches: Número de batches a examinar
    """
    print("\n" + "="*50)
    print("🔍 INSPEÇÃO DO DATASET")
    print("="*50)
    
    total_samples = 0
    
    for i, (states, actions) in enumerate(dataset.take(num_batches)):
        batch_size = states['pos'].shape[0]
        total_samples += batch_size
        
        print(f"\n📦 Batch {i+1}/{num_batches}:")
        print(f"  📊 Tamanho do batch: {batch_size}")
        print(f"  🗂️  Estrutura dos states:")
        for key, value in states.items():
            print(f"    • {key}: {value.shape} (dtype: {value.dtype})")
        print(f"  🎯 Actions shape: {actions.shape} (dtype: {actions.dtype})")
        
        # Mostrar estatísticas do primeiro batch
        if i == 0:
            print(f"\n📈 ESTATÍSTICAS (Batch 1):")
            print(f"  🎯 Pos exemplo: {states['pos'][0].numpy()}")
            print(f"  🎯 Action exemplo: {actions[0].numpy()}")
            
            enemy_min = tf.reduce_min(states['friends_hold']).numpy()
            enemy_max = tf.reduce_max(states['friends_hold']).numpy()
            enemy_mean = tf.reduce_mean(states['friends_hold']).numpy()
            
            print(f"  🔥 Enemy intensity - Min: {enemy_min:.3f}, Max: {enemy_max:.3f}, Mean: {enemy_mean:.3f}")
            
            # Verificar se há valores NaN ou Inf
            has_nan_pos = tf.reduce_any(tf.math.is_nan(states['pos']))
            has_nan_enemy = tf.reduce_any(tf.math.is_nan(states['friends_hold']))
            has_nan_actions = tf.reduce_any(tf.math.is_nan(actions))
            
            print(f"  🧪 Verificação de NaN:")
            print(f"    • Pos: {'❌ Contém NaN' if has_nan_pos else '✅ OK'}")
            print(f"    • Holding Friends: {'❌ Contém NaN' if has_nan_enemy else '✅ OK'}")
            print(f"    • Actions: {'❌ Contém NaN' if has_nan_actions else '✅ OK'}")
    
    print(f"\n📊 RESUMO:")
    print(f"  • Total de amostras inspecionadas: {total_samples}")
    print(f"  • Batches analisados: {num_batches}")
    print("="*50)


def setup_finite_dataset_training(dataset, validation_split=0.2, test_split=0.1):
    """
    Prepara dataset finito para treinamento com feedback melhorado.
    
    Args:
        dataset: Dataset finito do TensorFlow
        validation_split: Porcentagem para validação (0.2 = 20%)
        test_split: Porcentagem para teste (0.1 = 10%)
    
    Returns:
        train_ds, val_ds, test_ds: Datasets divididos
    """
    print("\n" + "="*50)
    print("📊 PREPARANDO DATASET PARA TREINAMENTO")
    print("="*50)
    
    # Calcular tamanho (única vez que percorre dataset completo)
    print("🔢 Calculando tamanho do dataset...")
    count_start = time.time()
    total_size = sum(1 for _ in dataset)
    count_time = time.time() - count_start
    
    # Calcular splits
    test_size = int(total_size * test_split)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size - test_size
    
    print(f"✅ Contagem concluída em {count_time:.1f}s")
    print(f"\n📊 DIVISÃO DO DATASET:")
    print(f"  🔵 Treino: {train_size:,} batches ({train_size/total_size*100:.1f}%)")
    print(f"  🟡 Validação: {val_size:,} batches ({val_size/total_size*100:.1f}%)")
    print(f"  🟢 Teste: {test_size:,} batches ({test_size/total_size*100:.1f}%)")
    print(f"  📦 Total: {total_size:,} batches")
    
    # Dividir datasets
    print(f"\n🔀 Dividindo datasets...")
    train_ds = dataset.take(train_size)
    remaining = dataset.skip(train_size)
    val_ds = remaining.take(val_size)
    test_ds = remaining.skip(val_size)
    
    # Otimizar para performance
    print(f"⚡ Otimizando performance...")
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    print(f"✅ Dataset preparado para treinamento!")
    print("="*50)
    
    return train_ds, val_ds, test_ds


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 INICIANDO SCRIPT DE GERAÇÃO DE DATASET")
    
    # Configurações
    # NUM_SAMPLES = 1024
    NUM_SAMPLES = 1024 *32 # 32 # 128 
    SAVE_PATH = f"./src/imitation_learning/data/behaviorCloneDataset_{NUM_SAMPLES}"
    BATCH_SIZE = 32
    
    print(f"⚙️  Configurações:")
    print(f"   • Amostras: {NUM_SAMPLES:,}")
    print(f"   • Caminho: {SAVE_PATH}")
    print(f"   • Batch size: {BATCH_SIZE}")
    
    # Criar dataset
    dataset = create_and_save_behavior_dataset(
        num_samples=NUM_SAMPLES,
        save_path=SAVE_PATH,
        batch_size=BATCH_SIZE
    )
    
    # Inspecionar o dataset criado
    # print(f"\n🔍 Inspecionando dataset criado...")
    # inspect_dataset(dataset, num_batches=2)
    
    print(f"\n✨ Script concluído com sucesso!")
    print(f"📁 Dataset disponível em: {SAVE_PATH}")