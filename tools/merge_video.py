#!/usr/bin/env python3
"""
Script para unir dois vídeos lado a lado com títulos.
Os vídeos podem ter tamanhos diferentes mas devem ter a mesma quantidade de frames.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys


def add_text_pil(img, text, position, font_path=None, font_size=40, color=(255, 255, 255), 
                 bold=False, align='center', max_width=None):
    """
    Adiciona texto usando PIL para suportar caracteres especiais.
    
    Args:
        img: Imagem numpy (BGR)
        text: Texto a ser adicionado
        position: Tupla (x, y) para posição do texto
        font_path: Caminho para arquivo de fonte TTF (None usa fonte padrão)
        font_size: Tamanho da fonte
        color: Cor do texto em RGB
        bold: Se True, tenta usar fonte bold
        align: Alinhamento ('left', 'center', 'right')
        max_width: Largura máxima para centralizar
    """
    # Converte BGR para RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # Tenta carregar fonte
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Tenta fontes comuns do sistema
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "C:\\Windows\\Fonts\\arial.ttf",
                "C:\\Windows\\Fonts\\arialbd.ttf" if bold else "C:\\Windows\\Fonts\\arial.ttf",
            ]
            font = None
            for fp in font_paths:
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except:
                    continue
            if font is None:
                font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Calcula tamanho do texto
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Ajusta posição baseado no alinhamento
    x, y = position
    if align == 'center':
        if max_width:
            x = x + (max_width - text_width) // 2
        else:
            x = x - text_width // 2
    elif align == 'right':
        x = x - text_width
    
    # Desenha o texto
    draw.text((x, y), text, font=font, fill=color)
    
    # Converte de volta para BGR
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr


def merge_videos_side_by_side(video1_path, video2_path, output_path, resize_mode='height',
                               main_title=None, title1="Vídeo 1", title2="Vídeo 2",
                               main_title_height=80, title_height=60,
                               main_font_size=50, font_size=35, font_path=None):
    """
    Une dois vídeos lado a lado com títulos.
    
    Args:
        video1_path: Caminho para o primeiro vídeo (esquerda)
        video2_path: Caminho para o segundo vídeo (direita)
        output_path: Caminho para salvar o vídeo resultante
        resize_mode: 'height' - ajusta pela altura, 'width' - ajusta pela largura,
                     'max' - usa o maior tamanho, 'none' - mantém tamanhos originais
        main_title: Título principal superior (opcional)
        title1: Título do primeiro vídeo
        title2: Título do segundo vídeo
        main_title_height: Altura da área do título principal
        title_height: Altura da área dos títulos dos vídeos
        main_font_size: Tamanho da fonte do título principal
        font_size: Tamanho da fonte dos títulos dos vídeos
        font_path: Caminho para arquivo de fonte TTF personalizada
    """
    
    # Abre os dois vídeos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        raise ValueError("Erro ao abrir um ou ambos os vídeos")
    
    # Obtém informações dos vídeos
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Vídeo 1: {width1}x{height1}, {frame_count1} frames, {fps1} FPS")
    print(f"Vídeo 2: {width2}x{height2}, {frame_count2} frames, {fps2} FPS")
    
    # Verifica se têm número similar de frames
    if abs(frame_count1 - frame_count2) > 5:
        print(f"AVISO: Os vídeos têm quantidade diferente de frames: {frame_count1} vs {frame_count2}")
    
    # Usa o FPS do primeiro vídeo
    fps = fps1
    
    # Calcula dimensões finais baseado no modo de redimensionamento
    if resize_mode == 'height':
        target_height = max(height1, height2)
        new_width1 = int(width1 * target_height / height1)
        new_height1 = target_height
        new_width2 = int(width2 * target_height / height2)
        new_height2 = target_height
    elif resize_mode == 'width':
        target_width = max(width1, width2)
        new_width1 = target_width
        new_height1 = int(height1 * target_width / width1)
        new_width2 = target_width
        new_height2 = int(height2 * target_width / width2)
    elif resize_mode == 'max':
        new_width1, new_height1 = width1, height1
        new_width2, new_height2 = width2, height2
    else:  # 'none'
        new_width1, new_height1 = width1, height1
        new_width2, new_height2 = width2, height2
    
    # Dimensões do vídeo final
    final_width = new_width1 + new_width2
    top_margin = (main_title_height if main_title else 0) + title_height
    final_height = max(new_height1, new_height2) + top_margin
    
    print(f"Vídeo final: {final_width}x{final_height}, {fps} FPS")
    
    # Configura o codec e cria o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (final_width, final_height))
    
    if not out.isOpened():
        raise ValueError("Erro ao criar arquivo de saída")
    
    frame_number = 0
    
    # Processa cada frame
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # Redimensiona frames se necessário
        if (frame1.shape[1], frame1.shape[0]) != (new_width1, new_height1):
            frame1 = cv2.resize(frame1, (new_width1, new_height1), interpolation=cv2.INTER_LANCZOS4)
        if (frame2.shape[1], frame2.shape[0]) != (new_width2, new_height2):
            frame2 = cv2.resize(frame2, (new_width2, new_height2), interpolation=cv2.INTER_LANCZOS4)
        
        # Cria frame combinado com fundo preto
        combined = np.zeros((final_height, final_width, 3), dtype=np.uint8)
        
        # Cola os frames lado a lado
        video_start_y = top_margin
        video_height = final_height - top_margin
        y_offset1 = video_start_y + (video_height - new_height1) // 2
        y_offset2 = video_start_y + (video_height - new_height2) // 2
        
        combined[y_offset1:y_offset1+new_height1, 0:new_width1] = frame1
        combined[y_offset2:y_offset2+new_height2, new_width1:new_width1+new_width2] = frame2
        
        # Adiciona título principal (se existir)
        current_y = 0
        if main_title:
            combined = add_text_pil(combined, main_title, 
                                   (final_width // 2, main_title_height // 3),
                                   font_path=font_path, font_size=main_font_size,
                                   color=(255, 255, 255), bold=True, align='center')
            current_y = main_title_height
        
        # Adiciona os títulos dos vídeos
        # Título 1 (esquerda)
        combined = add_text_pil(combined, title1,
                               (new_width1 // 2, current_y + title_height // 3),
                               font_path=font_path, font_size=font_size,
                               color=(255, 255, 255), align='center')
        
        # Título 2 (direita)
        combined = add_text_pil(combined, title2,
                               (new_width1 + new_width2 // 2, current_y + title_height // 3),
                               font_path=font_path, font_size=font_size,
                               color=(255, 255, 255), align='center')
        
        # Escreve o frame no vídeo de saída
        out.write(combined)
        
        frame_number += 1
        if frame_number % 30 == 0:
            print(f"Processados {frame_number} frames...")
    
    # Libera recursos
    cap1.release()
    cap2.release()
    out.release()
    
    print(f"Vídeo salvo com sucesso em: {output_path}")
    print(f"Total de frames processados: {frame_number}")


if __name__ == "__main__":

    video1_path = "./video/_proposal.mp4"
    video2_path = "./video/_proposal_formation.mp4"
    output_path = "./tmp/proposal.mp4"
    resize_mode = 'height'
    
    # Personalize os títulos aqui
    main_title = "CENÁRIO 3 - TÁTICA DESCENTRALIZADA"
    title1 = "Atacantes isolados"
    title2 = "Atacantes em formação"
    
    merge_videos_side_by_side(
        video1_path, video2_path, output_path, resize_mode,
        main_title=main_title,
        title1=title1,
        title2=title2,
        main_title_height=80,  # Altura da barra do título principal
        title_height=60,        # Altura da barra dos títulos dos vídeos
        main_font_size=50,      # Tamanho da fonte do título principal
        font_size=35,           # Tamanho da fonte dos subtítulos
        font_path=None          # Use None para fonte padrão ou especifique caminho TTF
    )



