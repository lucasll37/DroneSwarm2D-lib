from functools import lru_cache
from typing import Dict, Optional, Any
import os
import re


@lru_cache(maxsize=1)
def set_tensorflow() -> Any:
    """
    Configura e retorna TensorFlow com CPU-only e logs mínimos.
    
    Usa lru_cache para garantir que a configuração ocorra apenas uma vez.
    Esta função é útil para ambientes sem GPU ou onde se deseja forçar CPU.
    
    Returns:
        Módulo tensorflow configurado.
        
    Note:
        - Define CUDA_VISIBLE_DEVICES=-1 para forçar CPU
        - Define TF_CPP_MIN_LOG_LEVEL=2 para reduzir logs
    """
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    return tf


def load_best_model(
    directory: str, 
    pattern: str, 
    custom_objects: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    """
    Carrega o melhor modelo do diretório especificado selecionando o arquivo
    com a menor validation loss, e extrai o tamanho do nome do arquivo.
    
    Args:
        directory: Diretório contendo arquivos de modelo salvos.
        pattern: Padrão regex para extrair o valor de validation loss do nome do arquivo.
        custom_objects: Objetos customizados para passar ao load_model.
        
    Returns:
        O modelo carregado (tf.keras.Model), ou None se nenhum modelo for encontrado
        ou ocorrer erro no carregamento.
        
    Raises:
        Não levanta exceções; imprime mensagens de erro e retorna None em caso de falha.
        
    Example:
        >>> model = load_best_model("./models", r"val_loss_(\\d+\\.\\d+)")
        >>> if model:
        ...     predictions = model.predict(data)
    """
    tf = set_tensorflow()
    best_file: str = ""
    min_val_metric_loss: float = float("inf")

    # Iterar sobre arquivos para encontrar o com menor val_metric_loss
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return None
        
    for filename in os.listdir(directory):
        if filename.endswith(".keras"):
            match = re.search(pattern, filename)
            if match:
                val_metric_loss: float = float(match.group(1))
                if val_metric_loss < min_val_metric_loss:
                    min_val_metric_loss = val_metric_loss
                    best_file = filename

    if not best_file:
        print(f"No model files found in the directory: {directory}")
        return None

    model_path: str = os.path.join(directory, best_file)
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

    print(f"\n\nLoaded model: {best_file} with val_metric_loss={min_val_metric_loss:.4f}\n\n")
    return model