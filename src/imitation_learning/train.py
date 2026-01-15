# type: ignore
import os
import sys
import numpy as np
import pygame
import tensorflow as tf

from typing import Any, Dict, List, Optional, Tuple
from tensorflow.keras.layers import (
    Input, Dense, Flatten, Concatenate, BatchNormalization, 
    Dropout, Conv2D, MaxPooling2D, Reshape
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import plot_model
from datetime import datetime

# Adicionar o diretório pai (DroneSwarm2D-bib/) ao sys.path
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Agora pode importar normalmente
import DroneSwarm2D
settings = DroneSwarm2D.init(
    config_path="./config/proposal_spread.json",
    fullscreen=True
)


from data import load_behavior_dataset, inspect_dataset, setup_finite_dataset_training

# Initialize pygame if not already initialized
if not pygame.get_init():
    pygame.init()


def build_model():
    pos_input = Input(shape=(2,), name='pos')

    friend_intensity_input = Input(shape=(settings.GRID_WIDTH, settings.GRID_HEIGHT), name='friends_hold')
    
    flattenned_input = Flatten()(friend_intensity_input)
          
    x = Dense(1024, activation='relu', use_bias=False)(Concatenate()([pos_input, flattenned_input]))#(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(512, activation='relu', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
        
    x = Dense(256, activation='relu', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    x = Dense(128, activation='relu', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    x = Dense(64, activation='relu', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.05)(x)
    
    
    action_output = Dense(2, activation='linear')(x)
    
    model = Model(
        inputs=[pos_input, friend_intensity_input],
        outputs=action_output
    )
    
    model.compile(optimizer=Adam(), loss='mse')
    
    return model


def test_model(model: tf.keras.Model, test_ds) -> None:
    """
    Tests the model and prints evaluation metrics.
    """
    print("=== Testing Model ===")
    
    # Extract true values
    y_true = []
    for batch in test_ds:
        _, targets = batch
        y_true.extend(targets.numpy())
    
    y_true = np.array(y_true)
        
    # Loss using model's compiled loss function
    test_loss = model.evaluate(test_ds, verbose=0)
    print(f"Test Loss: {test_loss:.6f}")


class CustomModelCheckpoint(Callback):
    """
    Custom callback to save the model when the monitored metric improves.
    It deletes the previous best model file to save disk space.
    """
    def __init__(self, base_path: str, monitor: str = 'val_loss') -> None:
        """
        Args:
            base_path (str): Base file path for saving the model.
            monitor (str): Metric name to monitor.
        """
        super(CustomModelCheckpoint, self).__init__()
        self.base_path: str = base_path
        self.monitor: str = monitor
        self.best: float = float('inf')
        self.last_saved_model: Optional[str] = None
        
        os.makedirs(base_path, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        current_loss: Optional[float] = logs.get(self.monitor)
        if current_loss is not None and current_loss < self.best:
            self.best = current_loss
            filepath: str = f"{self.base_path}/best_model_epoch={epoch+1:02d}_{self.monitor}={current_loss:.4f}.keras"
            
            # Remove the previous saved model if it exists
            if self.last_saved_model and os.path.exists(self.last_saved_model):
                os.remove(self.last_saved_model)
            
            # Save the current model
            self.model.save(filepath)
            self.last_saved_model = filepath
            print(f"\n\n\nCheckpoint: New best model saved | {self.monitor} = {self.best}\n\n")
            
    
checkpoint = CustomModelCheckpoint(base_path="./src/imitation_learning/models/", monitor='val_loss')

# Criar diretório de logs com timestamp único
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"./src/imitation_learning/logs/{timestamp}"

tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,           # Histogramas dos pesos a cada epoch
    write_graph=True,           # Salvar gráfico do modelo
    write_images=False,         # Salvar imagens dos pesos
    write_steps_per_second=True, # Velocidade de treinamento
    update_freq='epoch',        # Atualizar a cada epoch
    profile_batch=0,            # Profiling desabilitado
)

earlyStopping = EarlyStopping(
    monitor='val_loss',
    patience=64,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',          # Métrica a monitorar
    factor=0.5,                  # Fator de redução (lr = lr * factor)
    patience=8,                  # Épocas para esperar antes de reduzir
    verbose=1,                   # Imprimir mensagem quando reduzir
    mode='min',                  # 'min' para val_loss, 'max' para val_acc
    min_delta=1e-6,             # Mudança mínima para ser considerada melhoria
    cooldown=0,                 # Épocas para esperar após uma redução
    min_lr=1e-8                 # Learning rate mínimo
)
  
# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Main function to build and train the neural network model for behavior cloning.
    It creates a dataset from the behavior cloning data, builds the model,
    and trains the model using the dataset.
    """
    NUM_SAMPLES = 1024
    dataset = load_behavior_dataset(f"./src/imitation_learning/data/behaviorCloneDataset_{NUM_SAMPLES}")
    train_ds, val_ds, _ = setup_finite_dataset_training(dataset, validation_split=0.2, test_split=0.1)
    # inspect_dataset(dataset)

    model = build_model()

    os.makedirs('./src/imitation_learning/images', exist_ok=True)
    plot_model(model, to_file='./src/imitation_learning/images/model_architecture.png', show_shapes=True, show_layer_names=True)

    # Display the model summary
    model.summary()

    # Train the neural network
    # Adjust the number of epochs as needed
    history = model.fit(
        train_ds,                    # Dataset de treino
        validation_data=val_ds,      # Dataset de validação
        epochs=1000,                # Máximo de épocas
        # callbacks=[checkpoint, reduce_lr, tensorboard_callback],
        callbacks=[checkpoint, earlyStopping, reduce_lr, tensorboard_callback],
        verbose=1
    )
    
# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()