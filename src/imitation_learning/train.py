# type: ignore
import os
import sys
import numpy as np
import pandas as pd
import pygame
import tensorflow as tf
import itertools
import json
from glob import glob

from typing import Any, Dict, List, Optional, Tuple
from tensorflow.keras.layers import (
    Input, Dense, Flatten, Concatenate, BatchNormalization, 
    Dropout, Conv2D, MaxPooling2D, Reshape, Lambda
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import plot_model
from datetime import datetime
from pathlib import Path

import DroneSwarm2D
settings = DroneSwarm2D.init(
    config_path="./config/proposal_spread.json",
    fullscreen=True
)

from data import load_behavior_dataset, inspect_dataset, setup_finite_dataset_training

# Initialize pygame if not already initialized
if not pygame.get_init():
    pygame.init()


def cosine_similarity_loss(y_true, y_pred):
    """
    Loss baseada em 1 - semelhança de cosseno
    Quanto menor, melhor (vetores mais alinhados)
    """
    # Normalizar vetores
    y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
    
    # Calcular semelhança de cosseno
    cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
    
    # Retornar perda (1 - similaridade)
    return tf.reduce_mean(1.0 - cosine_sim)


def normalize_to_unit_vector(x):
    """
    Camada Lambda para normalizar saída para vetor unitário
    """
    return tf.nn.l2_normalize(x, axis=-1)


def get_optimizer(optimizer_name: str, learning_rate: float = 1e-3):
    """
    Retorna otimizador configurado
    
    Args:
        optimizer_name: 'adam', 'rmsprop' ou 'sgd'
        learning_rate: taxa de aprendizado inicial
    
    Returns:
        Otimizador Keras
    """
    if optimizer_name == 'adam':
        return Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        return RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        return SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Otimizador desconhecido: {optimizer_name}")


def build_model(
    architecture_type: str,
    num_hidden_layers: int,
    activation: str,
    optimizer_name: str,
    input_shapes: Dict[str, tuple],
    output_dim: int = 2
) -> Model:
    """
    Constrói modelo MLP ou CNN com configurações especificadas
    
    Args:
        architecture_type: 'mlp' ou 'cnn'
        num_hidden_layers: número de camadas ocultas (2, 3, ou 4)
        activation: 'relu', 'sigmoid' ou 'tanh'
        optimizer_name: 'adam', 'rmsprop' ou 'sgd'
        input_shapes: dicionário com shapes das entradas
        output_dim: dimensão do vetor de saída (padrão 2)
    
    Returns:
        Modelo Keras compilado
    """
    # Inputs
    pos_input = Input(shape=input_shapes['pos'], name='pos')
    friend_intensity_input = Input(
        shape=input_shapes['friends_hold'], 
        name='friends_hold'
    )
    
    if architecture_type == 'cnn':
        # Arquitetura CNN
        # Reshape para adicionar canal se necessário
        if len(input_shapes['friends_hold']) == 2:
            x_spatial = Reshape((*input_shapes['friends_hold'], 1))(friend_intensity_input)
        else:
            x_spatial = friend_intensity_input
        
        # Camadas convolucionais
        filters = 32
        for i in range(num_hidden_layers):
            x_spatial = Conv2D(
                filters, 
                (3, 3), 
                activation=activation, 
                padding='same',
                use_bias=False
            )(x_spatial)
            x_spatial = BatchNormalization()(x_spatial)
            x_spatial = MaxPooling2D((2, 2))(x_spatial)
            x_spatial = Dropout(0.1)(x_spatial)
            filters = min(filters * 2, 256)
        
        # Flatten
        x_spatial = Flatten()(x_spatial)
        
        # Concatenar com posição
        x = Concatenate()([pos_input, x_spatial])
        
        # Camadas densas finais
        x = Dense(128, activation=activation, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(64, activation=activation, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        
    else:  # MLP
        # Flatten spatial input
        flattenned_input = Flatten()(friend_intensity_input)
        
        # Concatenar inputs
        x = Concatenate()([pos_input, flattenned_input])
        
        # Definir unidades por camada baseado no número de camadas
        if num_hidden_layers == 2:
            units_list = [512, 256]
        elif num_hidden_layers == 3:
            units_list = [1024, 512, 256]
        elif num_hidden_layers == 4:
            units_list = [1024, 512, 256, 128]
        else:  # 5 camadas (original)
            units_list = [1024, 512, 256, 128, 64]
        
        dropout_rates = [0.2, 0.1, 0.1, 0.1, 0.05]
        
        for i, units in enumerate(units_list):
            x = Dense(units, activation=activation, use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rates[i])(x)
    
    # Camada de saída (linear, antes da normalização)
    action_output = Dense(output_dim, activation='linear', name='pre_normalized_output')(x)
    
    # Camada Lambda para normalizar para vetor unitário
    normalized_output = Lambda(
        normalize_to_unit_vector, 
        name='unit_vector_output'
    )(action_output)
    
    # Criar modelo
    model = Model(
        inputs=[pos_input, friend_intensity_input],
        outputs=normalized_output,
        name=f"{architecture_type}_{num_hidden_layers}layers_{activation}_{optimizer_name}"
    )
    
    # Obter otimizador
    optimizer = get_optimizer(optimizer_name)
    
    # Compilar com loss de cosseno
    model.compile(
        optimizer=optimizer,
        loss=cosine_similarity_loss,
        metrics=['mae']  # Métrica adicional
    )
    
    return model


def load_dataset_by_size(num_samples: int) -> Tuple[Any, Any, Any]:
    """
    Carrega dataset com número específico de amostras
    
    Args:
        num_samples: 512, 1024 ou 2048
    
    Returns:
        train_ds, val_ds, test_ds
    """
    dataset_path = f"./src/imitation_learning/data/behaviorCloneDataset_{num_samples}"
    dataset = load_behavior_dataset(dataset_path)
    train_ds, val_ds, test_ds = setup_finite_dataset_training(
        dataset, 
        validation_split=0.2, 
        test_split=0.1
    )
    return train_ds, val_ds, test_ds


class ExperimentTracker(Callback):
    """
    Callback para rastrear métricas de cada experimento
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.final_train_loss = None
        self.final_val_loss = None
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        val_loss = logs.get('val_loss', float('inf'))
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1
        
        self.final_train_loss = logs.get('loss')
        self.final_val_loss = val_loss
    
    def get_results(self) -> Dict[str, Any]:
        return {
            **self.config,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'final_train_loss': self.final_train_loss,
            'final_val_loss': self.final_val_loss
        }


def train_model_configuration(
    config: Dict[str, Any],
    train_ds: Any,
    val_ds: Any,
    experiment_id: int,
    base_output_dir: str
) -> Dict[str, Any]:
    """
    Treina um modelo com configuração específica
    
    Args:
        config: dicionário com configuração do modelo
        train_ds: dataset de treino
        val_ds: dataset de validação
        experiment_id: ID único do experimento
        base_output_dir: diretório base para salvar resultados
    
    Returns:
        Dicionário com resultados do experimento
    """
    print("\n" + "="*80)
    print(f"EXPERIMENTO #{experiment_id}")
    print(f"Arquitetura: {config['architecture']}")
    print(f"Camadas: {config['num_layers']}")
    print(f"Ativação: {config['activation']}")
    print(f"Otimizador: {config['optimizer']}")
    print(f"Dataset: {config['dataset_size']} amostras")
    print("="*80 + "\n")
    
    # Definir shapes de entrada
    input_shapes = {
        'pos': (2,),
        'friends_hold': (settings.GRID_WIDTH, settings.GRID_HEIGHT)
    }
    
    # Construir modelo
    model = build_model(
        architecture_type=config['architecture'],
        num_hidden_layers=config['num_layers'],
        activation=config['activation'],
        optimizer_name=config['optimizer'],
        input_shapes=input_shapes,
        output_dim=2
    )
    
    # Diretórios para este experimento
    exp_dir = os.path.join(base_output_dir, f"experiment_{experiment_id:03d}")
    model_dir = os.path.join(exp_dir, "models")
    log_dir = os.path.join(exp_dir, "logs")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Salvar configuração
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Salvar arquitetura do modelo
    model_architecture_path = os.path.join(exp_dir, "model_architecture.png")
    try:
        plot_model(
            model, 
            to_file=model_architecture_path, 
            show_shapes=True, 
            show_layer_names=True
        )
    except:
        print("Aviso: Não foi possível salvar diagrama da arquitetura")
    
    # Callbacks
    experiment_tracker = ExperimentTracker(config)
    
    checkpoint = CustomModelCheckpoint(
        base_path=model_dir,
        monitor='val_loss',
        experiment_id=experiment_id
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=32,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        verbose=1,
        mode='min',
        min_delta=1e-6,
        cooldown=0,
        min_lr=1e-8
    )
    
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        update_freq='epoch'
    )
    
    # Treinar modelo
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=200,
            callbacks=[
                experiment_tracker,
                checkpoint,
                early_stopping,
                reduce_lr,
                tensorboard
            ],
            verbose=1
        )
        
        # Obter resultados
        results = experiment_tracker.get_results()
        results['status'] = 'completed'
        results['total_epochs'] = len(history.history['loss'])
        
    except Exception as e:
        print(f"\nERRO no experimento {experiment_id}: {str(e)}\n")
        results = {
            **config,
            'status': 'failed',
            'error': str(e),
            'best_val_loss': None,
            'best_epoch': None,
            'final_train_loss': None,
            'final_val_loss': None,
            'total_epochs': 0
        }
    
    # Limpar memória
    del model
    tf.keras.backend.clear_session()
    
    return results


class CustomModelCheckpoint(Callback):
    """
    Callback customizado para salvar melhor modelo
    """
    def __init__(
        self, 
        base_path: str, 
        monitor: str = 'val_loss',
        experiment_id: int = 0
    ):
        super().__init__()
        self.base_path = base_path
        self.monitor = monitor
        self.best = float('inf')
        self.last_saved_model = None
        self.experiment_id = experiment_id
        
        os.makedirs(base_path, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}
        current_loss = logs.get(self.monitor)
        
        if current_loss is not None and current_loss < self.best:
            self.best = current_loss
            
            filename = (
                f"exp{self.experiment_id:03d}_"
                f"epoch{epoch+1:03d}_"
                f"{self.monitor}={current_loss:.6f}.keras"
            )
            filepath = os.path.join(self.base_path, filename)
            
            # Remover modelo anterior
            if self.last_saved_model and os.path.exists(self.last_saved_model):
                os.remove(self.last_saved_model)
            
            # Salvar novo modelo
            self.model.save(filepath)
            self.last_saved_model = filepath
            print(f"\n✓ Checkpoint: Novo melhor modelo salvo ({self.monitor}={self.best:.6f})\n")


def find_existing_grid_search() -> Optional[str]:
    """
    Procura por diretórios de grid search existentes
    
    Returns:
        Caminho do diretório mais recente ou None
    """
    base_dir = "./src/imitation_learning"
    pattern = os.path.join(base_dir, "grid_search")
    
    if os.path.exists(pattern):
        return pattern
    
    return None


def load_existing_results(base_dir: str) -> Tuple[pd.DataFrame, set]:
    """
    Carrega resultados existentes de um grid search anterior
    
    Args:
        base_dir: diretório do grid search
    
    Returns:
        DataFrame com resultados e set de experiment_ids completados
    """
    results_file = os.path.join(base_dir, "results_partial.csv")
    
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        completed_ids = set(df['experiment_id'].values)
        print(f"✓ Carregados {len(df)} resultados existentes")
        print(f"✓ Experimentos completados: {len(completed_ids)}")
        return df, completed_ids
    
    return pd.DataFrame(), set()


def get_config_hash(config: Dict[str, Any]) -> str:
    """
    Gera hash único para uma configuração
    
    Args:
        config: dicionário com configuração
    
    Returns:
        String hash da configuração
    """
    config_str = f"{config['architecture']}_{config['num_layers']}_{config['activation']}_{config['optimizer']}_{config['dataset_size']}"
    return config_str


def load_existing_config_map(base_dir: str) -> Dict[str, int]:
    """
    Carrega mapeamento de configurações para experiment_ids
    
    Args:
        base_dir: diretório do grid search
    
    Returns:
        Dicionário mapeando hash de config para experiment_id
    """
    config_map = {}
    
    # Procurar por todos os diretórios de experimento
    exp_dirs = glob(os.path.join(base_dir, "experiment_*"))
    
    for exp_dir in exp_dirs:
        config_file = os.path.join(exp_dir, "config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Extrair experiment_id do nome do diretório
            exp_id = int(os.path.basename(exp_dir).split('_')[1])
            
            # Gerar hash e mapear
            config_hash = get_config_hash(config)
            config_map[config_hash] = exp_id
    
    return config_map


def main(resume: bool = True) -> None:
    """
    Função principal para executar grid search de configurações
    
    Args:
        resume: Se True, tenta retomar grid search anterior
    """
    # Nome fixo do diretório
    base_output_dir = "./src/imitation_learning/grid_search"
    
    # Verificar se existe grid search anterior
    existing_results = pd.DataFrame()
    completed_experiments = set()
    config_to_exp_id = {}
    
    if resume and os.path.exists(base_output_dir):
        print("\n" + "="*80)
        print("GRID SEARCH EXISTENTE ENCONTRADO!")
        print("="*80)
        
        existing_results, completed_experiments = load_existing_results(base_output_dir)
        config_to_exp_id = load_existing_config_map(base_output_dir)
        
        if len(completed_experiments) > 0:
            print(f"\n✓ Retomando grid search anterior")
            print(f"✓ {len(completed_experiments)} experimentos já completados")
            
            user_input = input("\nDeseja continuar de onde parou? (s/n): ").strip().lower()
            if user_input != 's':
                print("\nCriando novo grid search...")
                # Renomear diretório antigo
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                old_dir = f"{base_output_dir}_backup_{timestamp}"
                os.rename(base_output_dir, old_dir)
                print(f"Backup salvo em: {old_dir}")
                
                existing_results = pd.DataFrame()
                completed_experiments = set()
                config_to_exp_id = {}
        print("="*80 + "\n")
    
    # Criar diretório se não existir
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Definir grid de hiperparâmetros
    grid = {
        'architecture': ['mlp', 'cnn'],
        'num_layers': [2, 3, 4],
        'activation': ['relu', 'sigmoid', 'tanh'],
        'optimizer': ['adam', 'rmsprop', 'sgd'],
        'dataset_size': [8192, 32768, 131072]
    }
    
    # Gerar todas as combinações
    keys = grid.keys()
    values = grid.values()
    configurations = [
        dict(zip(keys, combo)) 
        for combo in itertools.product(*values)
    ]
    
    total_experiments = len(configurations)
    
    # Salvar grid de configurações
    grid_info = {
        'total_experiments': total_experiments,
        'grid_parameters': grid,
        'created_at': datetime.now().isoformat(),
        'resume_count': len(completed_experiments)
    }
    
    with open(os.path.join(base_output_dir, 'grid_info.json'), 'w') as f:
        json.dump(grid_info, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"GRID SEARCH: {total_experiments} experimentos")
    print(f"{'='*80}")
    print(f"\nCombinações:")
    print(f"  - Arquiteturas: {grid['architecture']}")
    print(f"  - Camadas: {grid['num_layers']}")
    print(f"  - Ativações: {grid['activation']}")
    print(f"  - Otimizadores: {grid['optimizer']}")
    print(f"  - Datasets: {grid['dataset_size']}")
    print(f"\nDiretório de saída: {base_output_dir}")
    
    if len(completed_experiments) > 0:
        remaining = total_experiments - len(completed_experiments)
        print(f"\n✓ Experimentos já completados: {len(completed_experiments)}")
        print(f"✓ Experimentos restantes: {remaining}")
    
    print(f"{'='*80}\n")
    
    # Cache de datasets
    datasets_cache = {}
    
    # Lista para armazenar resultados
    all_results = existing_results.to_dict('records') if not existing_results.empty else []
    
    # Determinar próximo experiment_id
    if len(all_results) > 0:
        next_exp_id = max([r['experiment_id'] for r in all_results]) + 1
    else:
        next_exp_id = 1
    
    # Executar cada experimento
    experiments_run = 0
    for config_idx, config in enumerate(configurations):
        # Verificar se já foi executado
        config_hash = get_config_hash(config)
        
        if config_hash in config_to_exp_id:
            exp_id = config_to_exp_id[config_hash]
            if exp_id in completed_experiments:
                print(f"⊘ Pulando experimento #{exp_id} (já completado)")
                continue
        
        # Atribuir experiment_id
        if config_hash in config_to_exp_id:
            exp_id = config_to_exp_id[config_hash]
        else:
            exp_id = next_exp_id
            config_to_exp_id[config_hash] = exp_id
            next_exp_id += 1
        
        # Carregar dataset (com cache)
        dataset_size = config['dataset_size']
        if dataset_size not in datasets_cache:
            print(f"\nCarregando dataset com {dataset_size} amostras...")
            datasets_cache[dataset_size] = load_dataset_by_size(dataset_size)
        
        train_ds, val_ds, test_ds = datasets_cache[dataset_size]
        
        # Treinar modelo
        results = train_model_configuration(
            config=config,
            train_ds=train_ds,
            val_ds=val_ds,
            experiment_id=exp_id,
            base_output_dir=base_output_dir
        )
        
        results['experiment_id'] = exp_id
        
        # Adicionar ou atualizar resultado
        existing_idx = None
        for idx, r in enumerate(all_results):
            if r['experiment_id'] == exp_id:
                existing_idx = idx
                break
        
        if existing_idx is not None:
            all_results[existing_idx] = results
        else:
            all_results.append(results)
        
        completed_experiments.add(exp_id)
        experiments_run += 1
        
        # Salvar resultados parciais
        df = pd.DataFrame(all_results)
        df = df.sort_values('experiment_id')
        df.to_csv(
            os.path.join(base_output_dir, "results_partial.csv"),
            index=False
        )
        
        # Estatísticas atualizadas
        completed = len([r for r in all_results if r.get('status') == 'completed'])
        failed = len([r for r in all_results if r.get('status') == 'failed'])
        valid_losses = [r.get('best_val_loss') for r in all_results if r.get('best_val_loss') is not None]
        
        print(f"\n{'='*80}")
        print(f"Progresso: {completed + failed}/{total_experiments} experimentos")
        print(f"Completos: {completed} | Falhas: {failed}")
        print(f"Executados nesta sessão: {experiments_run}")
        if valid_losses:
            print(f"Melhor val_loss: {min(valid_losses):.6f}")
        print(f"{'='*80}\n")
    
    # Salvar resultados finais
    df_final = pd.DataFrame(all_results)
    df_final = df_final.sort_values('experiment_id')
    
    # Ordenar por melhor val_loss
    df_final_sorted = df_final[df_final['status'] == 'completed'].sort_values('best_val_loss')
    
    # Salvar CSVs
    csv_path = os.path.join(base_output_dir, "results_complete.csv")
    df_final.to_csv(csv_path, index=False)
    
    csv_sorted_path = os.path.join(base_output_dir, "results_sorted_by_performance.csv")
    df_final_sorted.to_csv(csv_sorted_path, index=False)
    
    # Análise estatística por parâmetro
    analysis = {}
    for param in ['architecture', 'num_layers', 'activation', 'optimizer', 'dataset_size']:
        if len(df_final_sorted) > 0:
            analysis[param] = df_final_sorted.groupby(param)['best_val_loss'].agg(['mean', 'std', 'min', 'count'])
    
    # Salvar resumo
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_path = os.path.join(base_output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GRID SEARCH - RESUMO DOS RESULTADOS\n")
        f.write("="*80 + "\n")
        f.write(f"Última atualização: {timestamp}\n\n")
        
        f.write(f"Total de experimentos: {total_experiments}\n")
        f.write(f"Experimentos bem-sucedidos: {len(df_final[df_final['status'] == 'completed'])}\n")
        f.write(f"Experimentos com falha: {len(df_final[df_final['status'] == 'failed'])}\n")
        f.write(f"Experimentos restantes: {total_experiments - len(completed_experiments)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("TOP 10 MELHORES CONFIGURAÇÕES\n")
        f.write("="*80 + "\n\n")
        
        for idx, row in df_final_sorted.head(10).iterrows():
            f.write(f"#{row['experiment_id']} - Val Loss: {row['best_val_loss']:.6f}\n")
            f.write(f"  Arquitetura: {row['architecture']}\n")
            f.write(f"  Camadas: {row['num_layers']}\n")
            f.write(f"  Ativação: {row['activation']}\n")
            f.write(f"  Otimizador: {row['optimizer']}\n")
            f.write(f"  Dataset: {row['dataset_size']} amostras\n")
            f.write(f"  Melhor época: {row['best_epoch']}\n\n")
        
        if len(analysis) > 0:
            f.write("="*80 + "\n")
            f.write("ANÁLISE POR PARÂMETRO\n")
            f.write("="*80 + "\n\n")
            
            for param, stats in analysis.items():
                f.write(f"\n{param.upper()}:\n")
                f.write(stats.to_string())
                f.write("\n\n")
    
    # Imprimir resumo
    print("\n" + "="*80)
    if experiments_run == 0:
        print("GRID SEARCH JÁ ESTAVA COMPLETO!")
    else:
        print("GRID SEARCH ATUALIZADO!")
    print("="*80)
    print(f"\nResultados salvos em: {base_output_dir}")
    print(f"\nEstatísticas:")
    print(f"  Total: {total_experiments}")
    print(f"  Completos: {len(df_final[df_final['status'] == 'completed'])}")
    print(f"  Falhas: {len(df_final[df_final['status'] == 'failed'])}")
    print(f"  Restantes: {total_experiments - len(completed_experiments)}")
    print(f"  Executados nesta sessão: {experiments_run}")
    
    if len(df_final_sorted) > 0:
        print(f"\nMelhor configuração:")
        best = df_final_sorted.iloc[0]
        print(f"  Experimento #{best['experiment_id']}")
        print(f"  Val Loss: {best['best_val_loss']:.6f}")
        print(f"  Arquitetura: {best['architecture']}")
        print(f"  Camadas: {best['num_layers']}")
        print(f"  Ativação: {best['activation']}")
        print(f"  Otimizador: {best['optimizer']}")
        print(f"  Dataset: {best['dataset_size']} amostras")
        print(f"  Melhor época: {best['best_epoch']}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Para forçar novo grid search, use: main(resume=False)
    main(resume=True)


# # type: ignore
# import os
# import sys
# import numpy as np
# import pandas as pd
# import pygame
# import tensorflow as tf
# import itertools
# import json

# from typing import Any, Dict, List, Optional, Tuple
# from tensorflow.keras.layers import (
#     Input, Dense, Flatten, Concatenate, BatchNormalization, 
#     Dropout, Conv2D, MaxPooling2D, Reshape, Lambda
# )

# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam, RMSprop, SGD
# from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, TensorBoard
# from tensorflow.keras.utils import plot_model
# from datetime import datetime
# from pathlib import Path

# import DroneSwarm2D
# settings = DroneSwarm2D.init(
#     config_path="./config/default.json",
#     fullscreen=True
# )

# from data import load_behavior_dataset, inspect_dataset, setup_finite_dataset_training

# # Initialize pygame if not already initialized
# if not pygame.get_init():
#     pygame.init()


# def cosine_similarity_loss(y_true, y_pred):
#     """
#     Loss baseada em 1 - semelhança de cosseno
#     Quanto menor, melhor (vetores mais alinhados)
#     """
#     # Normalizar vetores
#     y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
#     y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
    
#     # Calcular semelhança de cosseno
#     cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
    
#     # Retornar perda (1 - similaridade)
#     return tf.reduce_mean(1.0 - cosine_sim)


# def normalize_to_unit_vector(x):
#     """
#     Camada Lambda para normalizar saída para vetor unitário
#     """
#     return tf.nn.l2_normalize(x, axis=-1)


# def get_optimizer(optimizer_name: str, learning_rate: float = 1e-3):
#     """
#     Retorna otimizador configurado
    
#     Args:
#         optimizer_name: 'adam', 'rmsprop' ou 'sgd'
#         learning_rate: taxa de aprendizado inicial
    
#     Returns:
#         Otimizador Keras
#     """
#     if optimizer_name == 'adam':
#         return Adam(learning_rate=learning_rate)
#     elif optimizer_name == 'rmsprop':
#         return RMSprop(learning_rate=learning_rate)
#     elif optimizer_name == 'sgd':
#         return SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
#     else:
#         raise ValueError(f"Otimizador desconhecido: {optimizer_name}")


# def build_model(
#     architecture_type: str,
#     num_hidden_layers: int,
#     activation: str,
#     optimizer_name: str,
#     input_shapes: Dict[str, tuple],
#     output_dim: int = 2
# ) -> Model:
#     """
#     Constrói modelo MLP ou CNN com configurações especificadas
    
#     Args:
#         architecture_type: 'mlp' ou 'cnn'
#         num_hidden_layers: número de camadas ocultas (2, 3, ou 4)
#         activation: 'relu', 'sigmoid' ou 'tanh'
#         optimizer_name: 'adam', 'rmsprop' ou 'sgd'
#         input_shapes: dicionário com shapes das entradas
#         output_dim: dimensão do vetor de saída (padrão 2)
    
#     Returns:
#         Modelo Keras compilado
#     """
#     # Inputs
#     pos_input = Input(shape=input_shapes['pos'], name='pos')
#     friend_intensity_input = Input(
#         shape=input_shapes['friends_hold'], 
#         name='friends_hold'
#     )
    
#     if architecture_type == 'cnn':
#         # Arquitetura CNN
#         # Reshape para adicionar canal se necessário
#         if len(input_shapes['friends_hold']) == 2:
#             x_spatial = Reshape((*input_shapes['friends_hold'], 1))(friend_intensity_input)
#         else:
#             x_spatial = friend_intensity_input
        
#         # Camadas convolucionais
#         filters = 32
#         for i in range(num_hidden_layers):
#             x_spatial = Conv2D(
#                 filters, 
#                 (3, 3), 
#                 activation=activation, 
#                 padding='same',
#                 use_bias=False
#             )(x_spatial)
#             x_spatial = BatchNormalization()(x_spatial)
#             x_spatial = MaxPooling2D((2, 2))(x_spatial)
#             x_spatial = Dropout(0.1)(x_spatial)
#             filters = min(filters * 2, 256)
        
#         # Flatten
#         x_spatial = Flatten()(x_spatial)
        
#         # Concatenar com posição
#         x = Concatenate()([pos_input, x_spatial])
        
#         # Camadas densas finais
#         x = Dense(128, activation=activation, use_bias=False)(x)
#         x = BatchNormalization()(x)
#         x = Dropout(0.2)(x)
        
#         x = Dense(64, activation=activation, use_bias=False)(x)
#         x = BatchNormalization()(x)
#         x = Dropout(0.1)(x)
        
#     else:  # MLP
#         # Flatten spatial input
#         flattenned_input = Flatten()(friend_intensity_input)
        
#         # Concatenar inputs
#         x = Concatenate()([pos_input, flattenned_input])
        
#         # Definir unidades por camada baseado no número de camadas
#         if num_hidden_layers == 2:
#             units_list = [512, 256]
#         elif num_hidden_layers == 3:
#             units_list = [1024, 512, 256]
#         elif num_hidden_layers == 4:
#             units_list = [1024, 512, 256, 128]
#         else:  # 5 camadas (original)
#             units_list = [1024, 512, 256, 128, 64]
        
#         dropout_rates = [0.2, 0.1, 0.1, 0.1, 0.05]
        
#         for i, units in enumerate(units_list):
#             x = Dense(units, activation=activation, use_bias=False)(x)
#             x = BatchNormalization()(x)
#             x = Dropout(dropout_rates[i])(x)
    
#     # Camada de saída (linear, antes da normalização)
#     action_output = Dense(output_dim, activation='linear', name='pre_normalized_output')(x)
    
#     # Camada Lambda para normalizar para vetor unitário
#     normalized_output = Lambda(
#         normalize_to_unit_vector, 
#         name='unit_vector_output'
#     )(action_output)
    
#     # Criar modelo
#     model = Model(
#         inputs=[pos_input, friend_intensity_input],
#         outputs=normalized_output,
#         name=f"{architecture_type}_{num_hidden_layers}layers_{activation}_{optimizer_name}"
#     )
    
#     # Obter otimizador
#     optimizer = get_optimizer(optimizer_name)
    
#     # Compilar com loss de cosseno
#     model.compile(
#         optimizer=optimizer,
#         loss=cosine_similarity_loss,
#         metrics=['mae']  # Métrica adicional
#     )
    
#     return model


# def load_dataset_by_size(num_samples: int) -> Tuple[Any, Any, Any]:
#     """
#     Carrega dataset com número específico de amostras
    
#     Args:
#         num_samples: 512, 1024 ou 2048
    
#     Returns:
#         train_ds, val_ds, test_ds
#     """
#     dataset_path = f"./src/imitation_learning/data/behaviorCloneDataset_{num_samples}"
#     dataset = load_behavior_dataset(dataset_path)
#     train_ds, val_ds, test_ds = setup_finite_dataset_training(
#         dataset, 
#         validation_split=0.2, 
#         test_split=0.1
#     )
#     return train_ds, val_ds, test_ds


# class ExperimentTracker(Callback):
#     """
#     Callback para rastrear métricas de cada experimento
#     """
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__()
#         self.config = config
#         self.best_val_loss = float('inf')
#         self.best_epoch = 0
#         self.final_train_loss = None
#         self.final_val_loss = None
    
#     def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
#         logs = logs or {}
#         val_loss = logs.get('val_loss', float('inf'))
        
#         if val_loss < self.best_val_loss:
#             self.best_val_loss = val_loss
#             self.best_epoch = epoch + 1
        
#         self.final_train_loss = logs.get('loss')
#         self.final_val_loss = val_loss
    
#     def get_results(self) -> Dict[str, Any]:
#         return {
#             **self.config,
#             'best_val_loss': self.best_val_loss,
#             'best_epoch': self.best_epoch,
#             'final_train_loss': self.final_train_loss,
#             'final_val_loss': self.final_val_loss
#         }


# def train_model_configuration(
#     config: Dict[str, Any],
#     train_ds: Any,
#     val_ds: Any,
#     experiment_id: int,
#     base_output_dir: str
# ) -> Dict[str, Any]:
#     """
#     Treina um modelo com configuração específica
    
#     Args:
#         config: dicionário com configuração do modelo
#         train_ds: dataset de treino
#         val_ds: dataset de validação
#         experiment_id: ID único do experimento
#         base_output_dir: diretório base para salvar resultados
    
#     Returns:
#         Dicionário com resultados do experimento
#     """
#     print("\n" + "="*80)
#     print(f"EXPERIMENTO #{experiment_id}")
#     print(f"Arquitetura: {config['architecture']}")
#     print(f"Camadas: {config['num_layers']}")
#     print(f"Ativação: {config['activation']}")
#     print(f"Otimizador: {config['optimizer']}")
#     print(f"Dataset: {config['dataset_size']} amostras")
#     print("="*80 + "\n")
    
#     # Definir shapes de entrada
#     input_shapes = {
#         'pos': (2,),
#         'friends_hold': (settings.GRID_WIDTH, settings.GRID_HEIGHT)
#     }
    
#     # Construir modelo
#     model = build_model(
#         architecture_type=config['architecture'],
#         num_hidden_layers=config['num_layers'],
#         activation=config['activation'],
#         optimizer_name=config['optimizer'],
#         input_shapes=input_shapes,
#         output_dim=2
#     )
    
#     # Diretórios para este experimento
#     exp_dir = os.path.join(base_output_dir, f"experiment_{experiment_id:03d}")
#     model_dir = os.path.join(exp_dir, "models")
#     log_dir = os.path.join(exp_dir, "logs")
    
#     os.makedirs(model_dir, exist_ok=True)
#     os.makedirs(log_dir, exist_ok=True)
    
#     # Salvar configuração
#     config_path = os.path.join(exp_dir, "config.json")
#     with open(config_path, 'w') as f:
#         json.dump(config, f, indent=2)
    
#     # Salvar arquitetura do modelo
#     model_architecture_path = os.path.join(exp_dir, "model_architecture.png")
#     try:
#         plot_model(
#             model, 
#             to_file=model_architecture_path, 
#             show_shapes=True, 
#             show_layer_names=True
#         )
#     except:
#         print("Aviso: Não foi possível salvar diagrama da arquitetura")
    
#     # Callbacks
#     experiment_tracker = ExperimentTracker(config)
    
#     checkpoint = CustomModelCheckpoint(
#         base_path=model_dir,
#         monitor='val_loss',
#         experiment_id=experiment_id
#     )
    
#     early_stopping = EarlyStopping(
#         monitor='val_loss',
#         patience=32,
#         restore_best_weights=True,
#         verbose=1
#     )
    
#     reduce_lr = ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.5,
#         patience=8,
#         verbose=1,
#         mode='min',
#         min_delta=1e-6,
#         cooldown=0,
#         min_lr=1e-8
#     )
    
#     tensorboard = TensorBoard(
#         log_dir=log_dir,
#         histogram_freq=0,
#         write_graph=False,
#         update_freq='epoch'
#     )
    
#     # Treinar modelo
#     try:
#         history = model.fit(
#             train_ds,
#             validation_data=val_ds,
#             epochs=200,
#             callbacks=[
#                 experiment_tracker,
#                 checkpoint,
#                 early_stopping,
#                 reduce_lr,
#                 tensorboard
#             ],
#             verbose=1
#         )
        
#         # Obter resultados
#         results = experiment_tracker.get_results()
#         results['status'] = 'completed'
#         results['total_epochs'] = len(history.history['loss'])
        
#     except Exception as e:
#         print(f"\nERRO no experimento {experiment_id}: {str(e)}\n")
#         results = {
#             **config,
#             'status': 'failed',
#             'error': str(e),
#             'best_val_loss': None,
#             'best_epoch': None,
#             'final_train_loss': None,
#             'final_val_loss': None,
#             'total_epochs': 0
#         }
    
#     # Limpar memória
#     del model
#     tf.keras.backend.clear_session()
    
#     return results


# class CustomModelCheckpoint(Callback):
#     """
#     Callback customizado para salvar melhor modelo
#     """
#     def __init__(
#         self, 
#         base_path: str, 
#         monitor: str = 'val_loss',
#         experiment_id: int = 0
#     ):
#         super().__init__()
#         self.base_path = base_path
#         self.monitor = monitor
#         self.best = float('inf')
#         self.last_saved_model = None
#         self.experiment_id = experiment_id
        
#         os.makedirs(base_path, exist_ok=True)

#     def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
#         logs = logs or {}
#         current_loss = logs.get(self.monitor)
        
#         if current_loss is not None and current_loss < self.best:
#             self.best = current_loss
            
#             filename = (
#                 f"exp{self.experiment_id:03d}_"
#                 f"epoch{epoch+1:03d}_"
#                 f"{self.monitor}={current_loss:.6f}.keras"
#             )
#             filepath = os.path.join(self.base_path, filename)
            
#             # Remover modelo anterior
#             if self.last_saved_model and os.path.exists(self.last_saved_model):
#                 os.remove(self.last_saved_model)
            
#             # Salvar novo modelo
#             self.model.save(filepath)
#             self.last_saved_model = filepath
#             print(f"\n✓ Checkpoint: Novo melhor modelo salvo ({self.monitor}={self.best:.6f})\n")


# def main() -> None:
#     """
#     Função principal para executar grid search de configurações
#     """
#     # Configurar output
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     base_output_dir = f"./src/imitation_learning/grid_search_{timestamp}"
#     os.makedirs(base_output_dir, exist_ok=True)
    
#     # Definir grid de hiperparâmetros
#     grid = {
#         'architecture': ['mlp', 'cnn'],
#         'num_layers': [2, 3, 4],
#         'activation': ['relu', 'sigmoid', 'tanh'],
#         'optimizer': ['adam', 'rmsprop', 'sgd'],
#         'dataset_size': [8192, 32768, 131072]
#     }
    
#     # Gerar todas as combinações
#     keys = grid.keys()
#     values = grid.values()
#     configurations = [
#         dict(zip(keys, combo)) 
#         for combo in itertools.product(*values)
#     ]
    
#     total_experiments = len(configurations)
    
#     # Salvar grid de configurações
#     grid_info = {
#         'total_experiments': total_experiments,
#         'grid_parameters': grid,
#         'timestamp': timestamp
#     }
    
#     with open(os.path.join(base_output_dir, 'grid_info.json'), 'w') as f:
#         json.dump(grid_info, f, indent=2)
    
#     print(f"\n{'='*80}")
#     print(f"GRID SEARCH: {total_experiments} experimentos")
#     print(f"{'='*80}")
#     print(f"\nCombinações:")
#     print(f"  - Arquiteturas: {grid['architecture']}")
#     print(f"  - Camadas: {grid['num_layers']}")
#     print(f"  - Ativações: {grid['activation']}")
#     print(f"  - Otimizadores: {grid['optimizer']}")
#     print(f"  - Datasets: {grid['dataset_size']}")
#     print(f"\nDiretório de saída: {base_output_dir}")
#     print(f"{'='*80}\n")
    
#     # Cache de datasets
#     datasets_cache = {}
    
#     # Lista para armazenar resultados
#     all_results = []
    
#     # Executar cada experimento
#     for exp_id, config in enumerate(configurations, start=1):
#         # Carregar dataset (com cache)
#         dataset_size = config['dataset_size']
#         if dataset_size not in datasets_cache:
#             print(f"\nCarregando dataset com {dataset_size} amostras...")
#             datasets_cache[dataset_size] = load_dataset_by_size(dataset_size)
        
#         train_ds, val_ds, test_ds = datasets_cache[dataset_size]
        
#         # Treinar modelo
#         results = train_model_configuration(
#             config=config,
#             train_ds=train_ds,
#             val_ds=val_ds,
#             experiment_id=exp_id,
#             base_output_dir=base_output_dir
#         )
        
#         results['experiment_id'] = exp_id
#         all_results.append(results)
        
#         # Salvar resultados parciais
#         df = pd.DataFrame(all_results)
#         df.to_csv(
#             os.path.join(base_output_dir, "results_partial.csv"),
#             index=False
#         )
        
#         # Estatísticas atualizadas
#         completed = len([r for r in all_results if r.get('status') == 'completed'])
#         failed = len([r for r in all_results if r.get('status') == 'failed'])
#         valid_losses = [r.get('best_val_loss') for r in all_results if r.get('best_val_loss') is not None]
        
#         print(f"\n{'='*80}")
#         print(f"Progresso: {exp_id}/{total_experiments} experimentos")
#         print(f"Completos: {completed} | Falhas: {failed}")
#         if valid_losses:
#             print(f"Melhor val_loss: {min(valid_losses):.6f}")
#         print(f"{'='*80}\n")
    
#     # Salvar resultados finais
#     df_final = pd.DataFrame(all_results)
    
#     # Ordenar por melhor val_loss
#     df_final_sorted = df_final[df_final['status'] == 'completed'].sort_values('best_val_loss')
    
#     # Salvar CSVs
#     csv_path = os.path.join(base_output_dir, "results_complete.csv")
#     df_final.to_csv(csv_path, index=False)
    
#     csv_sorted_path = os.path.join(base_output_dir, "results_sorted_by_performance.csv")
#     df_final_sorted.to_csv(csv_sorted_path, index=False)
    
#     # Análise estatística por parâmetro
#     analysis = {}
#     for param in ['architecture', 'num_layers', 'activation', 'optimizer', 'dataset_size']:
#         analysis[param] = df_final_sorted.groupby(param)['best_val_loss'].agg(['mean', 'std', 'min', 'count'])
    
#     # Salvar resumo
#     summary_path = os.path.join(base_output_dir, "summary.txt")
#     with open(summary_path, 'w') as f:
#         f.write("="*80 + "\n")
#         f.write("GRID SEARCH - RESUMO DOS RESULTADOS\n")
#         f.write("="*80 + "\n\n")
        
#         f.write(f"Total de experimentos: {total_experiments}\n")
#         f.write(f"Experimentos bem-sucedidos: {len(df_final[df_final['status'] == 'completed'])}\n")
#         f.write(f"Experimentos com falha: {len(df_final[df_final['status'] == 'failed'])}\n\n")
        
#         f.write("="*80 + "\n")
#         f.write("TOP 10 MELHORES CONFIGURAÇÕES\n")
#         f.write("="*80 + "\n\n")
        
#         for idx, row in df_final_sorted.head(10).iterrows():
#             f.write(f"#{row['experiment_id']} - Val Loss: {row['best_val_loss']:.6f}\n")
#             f.write(f"  Arquitetura: {row['architecture']}\n")
#             f.write(f"  Camadas: {row['num_layers']}\n")
#             f.write(f"  Ativação: {row['activation']}\n")
#             f.write(f"  Otimizador: {row['optimizer']}\n")
#             f.write(f"  Dataset: {row['dataset_size']} amostras\n")
#             f.write(f"  Melhor época: {row['best_epoch']}\n\n")
        
#         f.write("="*80 + "\n")
#         f.write("ANÁLISE POR PARÂMETRO\n")
#         f.write("="*80 + "\n\n")
        
#         for param, stats in analysis.items():
#             f.write(f"\n{param.upper()}:\n")
#             f.write(stats.to_string())
#             f.write("\n\n")
    
#     # Imprimir resumo
#     print("\n" + "="*80)
#     print("GRID SEARCH CONCLUÍDO!")
#     print("="*80)
#     print(f"\nResultados salvos em: {base_output_dir}")
#     print(f"\nEstatísticas:")
#     print(f"  Total: {total_experiments}")
#     print(f"  Completos: {len(df_final[df_final['status'] == 'completed'])}")
#     print(f"  Falhas: {len(df_final[df_final['status'] == 'failed'])}")
    
#     if len(df_final_sorted) > 0:
#         print(f"\nMelhor configuração:")
#         best = df_final_sorted.iloc[0]
#         print(f"  Experimento #{best['experiment_id']}")
#         print(f"  Val Loss: {best['best_val_loss']:.6f}")
#         print(f"  Arquitetura: {best['architecture']}")
#         print(f"  Camadas: {best['num_layers']}")
#         print(f"  Ativação: {best['activation']}")
#         print(f"  Otimizador: {best['optimizer']}")
#         print(f"  Dataset: {best['dataset_size']} amostras")
#         print(f"  Melhor época: {best['best_epoch']}")
    
#     print("\n" + "="*80 + "\n")


# if __name__ == "__main__":
#     main()