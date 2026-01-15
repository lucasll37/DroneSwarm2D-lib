#!/usr/bin/env python3
"""
Script para unir dois GIFs lado a lado.
Os GIFs podem ter tamanhos diferentes mas devem ter a mesma quantidade de frames.
"""

from PIL import Image, ImageSequence
import sys


def merge_gifs_side_by_side(gif1_path, gif2_path, output_path, resize_mode='height'):
    """
    Une dois GIFs lado a lado.
    
    Args:
        gif1_path: Caminho para o primeiro GIF (esquerda)
        gif2_path: Caminho para o segundo GIF (direita)
        output_path: Caminho para salvar o GIF resultante
        resize_mode: 'height' - ajusta pela altura, 'width' - ajusta pela largura,
                     'max' - usa o maior tamanho, 'none' - mantém tamanhos originais
    """
    
    # Abre os dois GIFs
    gif1 = Image.open(gif1_path)
    gif2 = Image.open(gif2_path)
    
    # Obtém informações dos GIFs
    frames1 = [frame.copy() for frame in ImageSequence.Iterator(gif1)]
    frames2 = [frame.copy() for frame in ImageSequence.Iterator(gif2)]
    
    # Verifica se têm o mesmo número de frames
    if len(frames1) != len(frames2):
        raise ValueError(f"Os GIFs têm quantidade diferente de frames: {len(frames1)} vs {len(frames2)}")
    
    # Obtém dimensões originais
    width1, height1 = frames1[0].size
    width2, height2 = frames2[0].size
    
    print(f"GIF 1: {width1}x{height1}, {len(frames1)} frames")
    print(f"GIF 2: {width2}x{height2}, {len(frames2)} frames")
    
    # Calcula dimensões finais baseado no modo de redimensionamento
    if resize_mode == 'height':
        # Ajusta ambos para a mesma altura (usa a maior)
        target_height = max(height1, height2)
        new_width1 = int(width1 * target_height / height1)
        new_height1 = target_height
        new_width2 = int(width2 * target_height / height2)
        new_height2 = target_height
    elif resize_mode == 'width':
        # Ajusta ambos para a mesma largura (usa a maior)
        target_width = max(width1, width2)
        new_width1 = target_width
        new_height1 = int(height1 * target_width / width1)
        new_width2 = target_width
        new_height2 = int(height2 * target_width / width2)
    elif resize_mode == 'max':
        # Usa as dimensões máximas
        new_width1, new_height1 = width1, height1
        new_width2, new_height2 = width2, height2
    else:  # 'none'
        # Mantém dimensões originais
        new_width1, new_height1 = width1, height1
        new_width2, new_height2 = width2, height2
    
    # Dimensões do GIF final
    final_width = new_width1 + new_width2
    final_height = max(new_height1, new_height2)
    
    print(f"GIF final: {final_width}x{final_height}")
    
    # Processa cada frame
    merged_frames = []
    durations = []
    
    for i, (frame1, frame2) in enumerate(zip(frames1, frames2)):
        # Converte para RGB se necessário
        if frame1.mode == 'P':
            frame1 = frame1.convert('RGBA')
        if frame2.mode == 'P':
            frame2 = frame2.convert('RGBA')
        
        # Redimensiona frames se necessário
        if (frame1.size[0], frame1.size[1]) != (new_width1, new_height1):
            frame1 = frame1.resize((new_width1, new_height1), Image.Resampling.LANCZOS)
        if (frame2.size[0], frame2.size[1]) != (new_width2, new_height2):
            frame2 = frame2.resize((new_width2, new_height2), Image.Resampling.LANCZOS)
        
        # Cria frame combinado
        combined = Image.new('RGBA', (final_width, final_height), (255, 255, 255, 0))
        
        # Cola os frames lado a lado (centralizados verticalmente se alturas diferentes)
        y_offset1 = (final_height - new_height1) // 2
        y_offset2 = (final_height - new_height2) // 2
        
        combined.paste(frame1, (0, y_offset1))
        combined.paste(frame2, (new_width1, y_offset2))
        
        merged_frames.append(combined)
        
        # Obtém duração do frame (usa a do primeiro GIF como padrão)
        try:
            duration = frame1.info.get('duration', 100)
        except:
            duration = 100
        durations.append(duration)
    
    # Salva o GIF combinado
    merged_frames[0].save(
        output_path,
        save_all=True,
        append_images=merged_frames[1:],
        duration=durations,
        loop=0,  # Loop infinito
        optimize=False
    )
    
    print(f"GIF salvo com sucesso em: {output_path}")


if __name__ == "__main__":

    gif1_path = "images/matrix_recency_0.gif"
    gif2_path = "images/matrix_recency_1.gif"
    output_path = "images/matrix_recency.gif"
    resize_mode = 'height'
        
    merge_gifs_side_by_side(gif1_path, gif2_path, output_path, resize_mode)