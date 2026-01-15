import imageio
from PIL import Image
import numpy as np
import os

def video_to_gif_crop_simple(video_path, output_path, x1, y1, x2, y2, start_time=0, end_time=None, fps=10, use_percentage=False):
    """
    Versão simplificada que converte vídeo em GIF com recorte - usa apenas imageio e PIL.
    
    Args:
        video_path (str): Caminho para o arquivo de vídeo MP4
        output_path (str): Caminho para salvar o GIF
        x1, y1 (float): Coordenadas do ponto superior esquerdo do recorte
        x2, y2 (float): Coordenadas do ponto inferior direito do recorte
        start_time (float): Tempo de início em segundos (padrão: 0)
        end_time (float): Tempo de fim em segundos (padrão: final do vídeo)
        fps (int): Frames por segundo do GIF (padrão: 10)
        use_percentage (bool): Se True, coordenadas são percentuais (0.0 a 1.0)
    """
    
    # Verificar se o arquivo existe
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {video_path}")
    
    print("Carregando vídeo...")
    
    # Ler vídeo com imageio
    try:
        reader = imageio.get_reader(video_path)
        video_fps = reader.get_meta_data()['fps']
        total_frames = reader.count_frames()
        duration = total_frames / video_fps
        
        print(f"Informações do vídeo:")
        print(f"FPS original: {video_fps}")
        print(f"Duração: {duration:.2f} segundos")
        print(f"Total de frames: {total_frames}")
        
    except Exception as e:
        raise ValueError(f"Erro ao ler vídeo: {e}")
    
    # Obter primeiro frame para descobrir dimensões
    first_frame = reader.get_data(0)
    video_height, video_width = first_frame.shape[:2]
    
    print(f"Dimensões: {video_width}x{video_height}")
    
    # Converter coordenadas percentuais para pixels se necessário
    if use_percentage:
        if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 
                0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
            raise ValueError("Coordenadas percentuais devem estar entre 0.0 e 1.0")
        
        if x1 >= x2 or y1 >= y2:
            raise ValueError("x1 deve ser menor que x2 e y1 menor que y2")
        
        x1_px = int(x1 * video_width)
        y1_px = int(y1 * video_height)
        x2_px = int(x2 * video_width)
        y2_px = int(y2 * video_height)
        
        print(f"Coordenadas convertidas:")
        print(f"Superior esquerdo: ({x1:.1%}, {y1:.1%}) -> ({x1_px}, {y1_px})")
        print(f"Inferior direito: ({x2:.1%}, {y2:.1%}) -> ({x2_px}, {y2_px})")
    else:
        x1_px, y1_px, x2_px, y2_px = int(x1), int(y1), int(x2), int(y2)
        
        if (x1_px < 0 or y1_px < 0 or x2_px > video_width or y2_px > video_height or
            x1_px >= x2_px or y1_px >= y2_px):
            raise ValueError(f"Coordenadas inválidas. Vídeo: {video_width}x{video_height}")
    
    # Definir range de tempo
    if end_time is None:
        end_time = duration
    
    if start_time < 0 or start_time >= duration:
        raise ValueError(f"Tempo de início inválido: {start_time}")
    
    if end_time <= start_time or end_time > duration:
        raise ValueError(f"Tempo de fim inválido: {end_time}")
    
    # Calcular frames a extrair
    start_frame = int(start_time * video_fps)
    end_frame = int(end_time * video_fps)
    frame_step = max(1, int(video_fps / fps))
    
    print(f"Extraindo frames {start_frame} a {end_frame} (passo: {frame_step})")
    
    # Extrair frames
    frames = []
    current_frame = start_frame
    
    while current_frame < end_frame and current_frame < total_frames:
        try:
            frame = reader.get_data(current_frame)
            
            # Fazer recorte
            cropped = frame[y1_px:y2_px, x1_px:x2_px]
            
            if cropped.size == 0:
                raise ValueError("Recorte resultou em área vazia")
            
            frames.append(cropped)
            current_frame += frame_step
            
        except Exception as e:
            print(f"Erro no frame {current_frame}: {e}")
            break
    
    reader.close()
    
    if not frames:
        raise ValueError("Nenhum frame foi extraído")
    
    print(f"Extraídos {len(frames)} frames")
    print(f"Dimensões do recorte: {frames[0].shape[1]}x{frames[0].shape[0]}")
    
    # Converter para GIF
    print("Salvando GIF...")
    
    try:
        imageio.mimsave(
            output_path,
            frames,
            duration=1.0/fps,
            loop=0
        )
        
        print(f"✅ GIF salvo: {output_path}")
        
        # Mostrar tamanho do arquivo
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Tamanho: {file_size:.2f} MB")
        
    except Exception as e:
        raise ValueError(f"Erro ao salvar GIF: {e}")

def get_video_info_simple(video_path):
    """
    Retorna informações básicas do vídeo usando apenas imageio.
    """
    try:
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        first_frame = reader.get_data(0)
        
        info = {
            'width': first_frame.shape[1],
            'height': first_frame.shape[0],
            'fps': meta.get('fps', 30),
            'duration': reader.count_frames() / meta.get('fps', 30)
        }
        
        reader.close()
        return info
        
    except Exception as e:
        raise ValueError(f"Erro ao ler vídeo: {e}")

# Exemplo de uso simplificado
if __name__ == "__main__":
    video_file = "./video/proposal.mp4"  # Substitua pelo seu vídeo
    
    # Verificar informações do vídeo
    try:
        info = get_video_info_simple(video_file)
        print("=== INFORMAÇÕES DO VÍDEO ===")
        print(f"Dimensões: {info['width']}x{info['height']}")
        print(f"FPS: {info['fps']}")
        print(f"Duração: {info['duration']:.2f}s")
    except Exception as e:
        print(f"Erro ao obter informações: {e}")
        print("Continuando mesmo assim...")
    

    video_to_gif_crop_simple(
        video_path=video_file,
        output_path="./images/sim_0.gif",
        x1=0, y1=0,
        x2=0.7, y2=1,
        
        # Gráfico
        # x1=0.7, y1=0.04,
        # x2=0.95, y2=0.44,
        
        # Area de simulação
        # x1=0.15, y1=0.2,
        # x2=0.55, y2=0.8,
        start_time=0,
        end_time=237,
        fps=12,
        use_percentage=True
    )