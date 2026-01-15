import os
import json
import numpy as np
import tensorflow as tf
import time

# Configurar DroneSwarm2D para pegar settings
import DroneSwarm2D
settings = DroneSwarm2D.init(
    config_path="./config/proposal_spread.json",
    fullscreen=True
)


def convert_joystick_jsonl_to_tf_dataset(jsonl_path, save_path, batch_size=32):
    """
    Converte arquivo JSONL de dados de joystick para TensorFlow Dataset.
    
    Args:
        jsonl_path (str): Caminho para arquivo .jsonl
        save_path (str): Caminho de saída do TF dataset
        batch_size (int): Tamanho do batch
    
    Returns:
        tf.data.Dataset: Dataset criado e salvo
    """
    print("="*60)
    print(f"🎯 CONVERTENDO DADOS DE JOYSTICK PARA TF DATASET")
    print(f"📂 Origem: {jsonl_path}")
    print(f"📁 Destino: {save_path}")
    print(f"📦 Batch size: {batch_size}")
    print("="*60)
    
    # Verificar se arquivo existe
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {jsonl_path}")
    
    # 1. CARREGAR DADOS DO JSONL
    print(f"\n📂 Carregando dados de: {jsonl_path}")
    load_start_time = time.time()
    
    samples = []
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  Erro ao parsear linha {line_num}: {e}")
                continue
    
    load_time = time.time() - load_start_time
    print(f"✅ Carregados {len(samples):,} samples em {load_time:.1f}s")
    
    if len(samples) == 0:
        raise ValueError("❌ Nenhum sample válido encontrado no arquivo!")
    
    # 2. CONVERTER PARA NUMPY ARRAYS
    print(f"\n🔄 Convertendo para arrays numpy...")
    positions = []
    friends_hold_list = []
    velocity_list = []
    
    for sample in samples:
        pos = np.array(sample['pos'], dtype=np.float32).flatten()
        vel = np.array(sample['velocity'], dtype=np.float32).flatten()
        
        positions.append(pos)
        friends_hold_list.append(sample['friends_hold'])
        velocity_list.append(vel)
    
    positions = np.array(positions, dtype=np.float32)
    friends_hold = np.array(friends_hold_list, dtype=np.float32)
    velocities = np.array(velocity_list, dtype=np.float32)
    
    print(f"✅ Shapes convertidas:")
    print(f"  • positions: {positions.shape}")
    print(f"  • friends_hold: {friends_hold.shape}")
    print(f"  • velocities: {velocities.shape}")
 
    print(f"\n🔧 Criando TensorFlow Dataset...")
    dataset = tf.data.Dataset.from_tensor_slices((
        {'pos': positions, 'friends_hold': friends_hold},
        velocities
    ))

    print(f"📦 Configurando batches de tamanho {batch_size}...")
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print(f"\n💾 Salvando dataset em disco...")
    save_start_time = time.time()
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    tf.data.Dataset.save(dataset, save_path)
    
    save_time = time.time() - save_start_time
    print(f"✅ Dataset salvo com sucesso! (tempo de salvamento: {save_time:.1f}s)")
    
    # Estatísticas finais
    print("\n" + "="*60)
    print("🎉 CONVERSÃO CONCLUÍDA COM SUCESSO!")
    print(f"✅ Total de samples: {len(samples):,}")
    print(f"✅ Shape positions: {positions.shape}")
    print(f"✅ Shape friends_hold: {friends_hold.shape}")
    print(f"✅ Shape velocities: {velocities.shape}")
    print(f"📁 Dataset salvo em: {save_path}")
    print("="*60)
    
    return dataset

def load_behavior_dataset(save_path="./src/imitation_learning/data/behaviorCloneDataset"):
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


# USO:
if __name__ == "__main__":
    # Converte o arquivo JSONL para TF Dataset
    dataset = convert_joystick_jsonl_to_tf_dataset(
        jsonl_path="./src/imitation_learning/data/joystick_data/joystick_session_20260111_142303.jsonl",
        save_path="./src/imitation_learning/data/behaviorCloneDataset",
        batch_size=32
    )