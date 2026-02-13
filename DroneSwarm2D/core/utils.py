"""
utils.py

Fornece funções utilitárias para visualização da simulação, processamento de matrizes,
conversão de coordenadas e cálculos diversos.

Este módulo contém:
- Funções de desenho (círculos tracejados, linhas tracejadas)
- Processamento de matrizes com kernels Gaussianos
- Conversão entre coordenadas de simulação e geográficas
- Carregamento de SVG
- Cálculos de interceptação e busca
- Geração de matrizes esparsas
"""

# Standard libraries
import math
import random
import io

# import re
# import sys
from typing import Tuple, Dict, Optional, Any, Callable
# from functools import lru_cache
# from pathlib import Path

# Third-party libraries
import numpy as np
import pygame
import cairosvg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, usado para plotting 3D
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Project-specific imports
from .settings import (
    SIM_WIDTH, SIM_HEIGHT,
    GRID_WIDTH, GRID_HEIGHT,
    CELL_SIZE,
    INTEREST_POINT_CENTER,
    FRIEND_SPEED,
    ENEMY_SPEED,
    PLOT_THRESHOLD
)


# @lru_cache(maxsize=1)
# def set_tensorflow() -> Any:
#     """
#     Configura e retorna TensorFlow com CPU-only e logs mínimos.
    
#     Usa lru_cache para garantir que a configuração ocorra apenas uma vez.
#     Esta função é útil para ambientes sem GPU ou onde se deseja forçar CPU.
    
#     Returns:
#         Módulo tensorflow configurado.
        
#     Note:
#         - Define CUDA_VISIBLE_DEVICES=-1 para forçar CPU
#         - Define TF_CPP_MIN_LOG_LEVEL=2 para reduzir logs
#     """
#     # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#     import tensorflow as tf
#     return tf


# -----------------------------------------------------------------------------
# Funções de Desenho
# -----------------------------------------------------------------------------

def draw_dashed_circle(
    surface: pygame.Surface,
    color: Tuple[int, int, int, int],
    center: Tuple[int, int],
    radius: int,
    dash_length: int = 5,
    space_length: int = 5,
    width: int = 1
) -> None:
    """
    Desenha um círculo tracejado na superfície pygame.
    
    O círculo é composto por múltiplos segmentos de linha (dashes) distribuídos
    uniformemente ao redor da circunferência.
    
    Args:
        surface: Superfície pygame onde o círculo será desenhado.
        color: Cor RGBA do círculo (R, G, B, Alpha).
        center: Centro do círculo como tupla (x, y).
        radius: Raio do círculo em pixels.
        dash_length: Comprimento de cada traço em pixels.
        space_length: Espaço entre traços em pixels.
        width: Espessura da linha em pixels.
        
    Note:
        Se radius <= 0, a função retorna sem desenhar nada.
    """
    if radius <= 0:
        return
    
    # Calcular circunferência e número de dashes
    circumference: float = 2 * math.pi * radius
    num_dashes: int = int(circumference / (dash_length + space_length))
    angle_between: float = 2 * math.pi / num_dashes

    for i in range(num_dashes):
        start_angle: float = i * angle_between
        dash_angle: float = dash_length / radius  # Ângulo correspondente ao dash
        end_angle: float = start_angle + dash_angle

        start_pos: Tuple[float, float] = (
            center[0] + radius * math.cos(start_angle),
            center[1] + radius * math.sin(start_angle)
        )
        end_pos: Tuple[float, float] = (
            center[0] + radius * math.cos(end_angle),
            center[1] + radius * math.sin(end_angle)
        )
        pygame.draw.line(surface, color, start_pos, end_pos, width)


def draw_dashed_line(
    surface: pygame.Surface, 
    color: Tuple[int, int, int, int],
    start_pos: pygame.math.Vector2, 
    end_pos: pygame.math.Vector2,
    width: int = 1, 
    dash_length: int = 5, 
    space_length: int = 5
) -> None:
    """
    Desenha uma linha tracejada entre dois pontos.
    
    Args:
        surface: Superfície pygame onde a linha será desenhada.
        color: Cor RGBA da linha (R, G, B, Alpha).
        start_pos: Posição inicial como Vector2.
        end_pos: Posição final como Vector2.
        width: Espessura da linha em pixels.
        dash_length: Comprimento de cada traço em pixels.
        space_length: Espaço entre traços em pixels.
        
    Note:
        Se start_pos == end_pos, a função retorna sem desenhar.
    """
    start = pygame.math.Vector2(start_pos)
    end = pygame.math.Vector2(end_pos)
    displacement = end - start
    length: float = displacement.length()
    
    if length == 0:
        return
    
    dash_vector = displacement.normalize() * dash_length
    num_dashes: int = int(length / (dash_length + space_length))
    
    for i in range(num_dashes):
        dash_start = start + (dash_length + space_length) * i * displacement.normalize()
        dash_end = dash_start + dash_vector
        pygame.draw.line(surface, color, dash_start, dash_end, width)


# -----------------------------------------------------------------------------
# Funções de Kernel Gaussiano
# -----------------------------------------------------------------------------

def symmetrical_flat_topped_gaussian_10x10(
    value: float, 
    sigma: float, 
    flat_radius: float
) -> np.ndarray:
    """
    Cria um kernel Gaussiano 10x10 com topo plano (flat-topped).
    
    As coordenadas contínuas variam de -4.5 a +4.5 para garantir simetria.
    O bump tem topo plano (valor = 1) dentro do raio especificado, e
    decai gaussianamente fora dele.
    
    Args:
        value: Valor de pico para escalar no centro.
        sigma: Desvio padrão para a porção Gaussiana.
        flat_radius: Raio (em coordenadas contínuas) onde o valor é constante (1.0).
        
    Returns:
        Array numpy 10x10 representando o kernel.
        
    Note:
        A função usa uma abordagem otimizada sem meshgrid para melhor performance.
    """
    kernel_size: int = 10
    
    # Criar grid de coordenadas manualmente (sem meshgrid para performance)
    x = np.linspace(-4.5, 4.5, kernel_size)
    
    # Inicializar array para o bump
    bump = np.empty((kernel_size, kernel_size), dtype=np.float64)
    
    # Calcular valores diretamente
    for i in range(kernel_size):
        y: float = x[i]  # Coordenada y
        for j in range(kernel_size):
            xi: float = x[j]  # Coordenada x
            
            # Calcular distância do centro
            r: float = np.sqrt(xi**2 + y**2)
            
            # Aplicar topo plano ou decaimento Gaussiano
            if r < flat_radius:
                bump[i, j] = 1.0
            else:
                bump[i, j] = np.exp(-0.5 * (r / sigma)**2)
    
    return value * bump


def smooth_matrix_with_kernel_10x10(
    matrix: np.ndarray,
    direction: np.ndarray,
    sigma: float = 1.0,
    flat_radius: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica um kernel Gaussiano 10x10 (com topo plano) a cada valor positivo na matriz.
    
    O bump é centralizado no pixel (i, j) sem mudar as dimensões originais.
    Para cada pico, se o valor do bump for maior que o valor atual na região,
    tanto a intensidade quanto o vetor de direção correspondente são atualizados.
    
    Args:
        matrix: Array 2D com valores entre 0 e 1.
        direction: Array com mesmas dimensões de matrix mais uma dimensão extra
                  para o vetor de direção (ex: shape (n_rows, n_cols, 2)).
        sigma: Desvio padrão para a porção Gaussiana.
        flat_radius: Raio do topo plano (em coordenadas contínuas).
        
    Returns:
        Tupla contendo:
        - result: Matriz atualizada com os bumps aplicados.
        - result_direction: Array de direção atualizado.
        
    Note:
        Otimizado para matrizes com muitos picos (>50) usando cache de kernels.
    """
    result = np.copy(matrix)
    result_direction = np.copy(direction)
    peaks = np.argwhere(matrix > 0)
    kernel_size: int = 10
    anchor: int = kernel_size // 2
    
    # Otimização: pré-calcular kernels para valores únicos se há muitos picos
    peak_values: Optional[Dict[float, np.ndarray]] = None
    if len(peaks) > 50:  # Threshold arbitrário
        peak_values = {}
        for (i, j) in peaks:
            value: float = matrix[i, j]
            if value not in peak_values:
                peak_values[value] = symmetrical_flat_topped_gaussian_10x10(
                    value, sigma, flat_radius
                )

    for (i, j) in peaks:
        value: float = matrix[i, j]
        
        # Usar bump pré-calculado se disponível
        if peak_values is not None:
            bump = peak_values[value]
        else:
            bump = symmetrical_flat_topped_gaussian_10x10(value, sigma, flat_radius)
            
        bump_direction = direction[i, j]  # Vetor de direção no pico

        # Calcular índices de forma eficiente
        i_start: int = max(0, i - anchor)
        j_start: int = max(0, j - anchor)
        i_end: int = min(matrix.shape[0], i + anchor)
        j_end: int = min(matrix.shape[1], j + anchor)

        # Calcular índices do bump
        bump_i_start: int = max(0, anchor - (i - i_start))
        bump_j_start: int = max(0, anchor - (j - j_start))
        bump_i_end: int = bump_i_start + (i_end - i_start)
        bump_j_end: int = bump_j_start + (j_end - j_start)

        # Obter views das regiões (sem cópia)
        region = result[i_start:i_end, j_start:j_end]
        bump_region = bump[bump_i_start:bump_i_end, bump_j_start:bump_j_end]

        # Criar máscara para pixels onde bump > valor atual
        mask = bump_region > region
        
        # Usar operações in-place
        np.maximum(region, bump_region, out=region)

        # Atualizar vetores de direção onde o bump aumentou o valor
        for dim in range(direction.shape[-1]):
            for di in range(i_end - i_start):
                for dj in range(j_end - j_start):
                    if mask[di, dj]:
                        result_direction[i_start + di, j_start + dj, dim] = bump_direction[dim]

    return result, result_direction


# -----------------------------------------------------------------------------
# Conversão de Coordenadas
# -----------------------------------------------------------------------------

def sim_to_geo(pos_x: float, pos_y: float) -> Tuple[float, float]:
    """
    Converte coordenadas de simulação para coordenadas geográficas.
    
    Args:
        pos_x: Posição X na simulação.
        pos_y: Posição Y na simulação.
        
    Returns:
        Tupla (Longitude, Latitude) em coordenadas geográficas.
        
    Note:
        Requer que GEO_TOP_LEFT e GEO_BOTTOM_RIGHT estejam definidos em settings.
    """
    from .settings import GEO_TOP_LEFT, GEO_BOTTOM_RIGHT
    
    lon_left, lat_top = GEO_TOP_LEFT
    lon_right, lat_bottom = GEO_BOTTOM_RIGHT
    
    lon: float = lon_left + (pos_x / SIM_WIDTH) * (lon_right - lon_left)
    lat: float = lat_top + (pos_y / SIM_HEIGHT) * (lat_bottom - lat_top)
    
    return lon, lat


# -----------------------------------------------------------------------------
# Utilitários de SVG e Imagem
# -----------------------------------------------------------------------------

def load_svg_as_surface(svg_path: str) -> pygame.Surface:
    """
    Converte um arquivo SVG para uma superfície pygame com suporte a alpha.
    
    Args:
        svg_path: Caminho para o arquivo SVG.
        
    Returns:
        Superfície pygame com a imagem convertida.
        
    Raises:
        FileNotFoundError: Se o arquivo SVG não for encontrado.
    """
    # Converter SVG para PNG em memória
    png_data: bytes = cairosvg.svg2png(url=svg_path)
    image_data = io.BytesIO(png_data)
    surface: pygame.Surface = pygame.image.load(image_data).convert_alpha()
    return surface


# -----------------------------------------------------------------------------
# Utilitários de Grid e Posicionamento
# -----------------------------------------------------------------------------

def pos_to_cell(
    pos: pygame.math.Vector2, 
    cell_size: int = CELL_SIZE,
    grid_width: int = GRID_WIDTH, 
    grid_height: int = GRID_HEIGHT
) -> Tuple[int, int]:
    """
    Converte uma posição (Vector2) para coordenadas de célula na grid.
    
    Args:
        pos: Vetor de posição.
        cell_size: Tamanho de cada célula em pixels.
        grid_width: Largura da grid em células.
        grid_height: Altura da grid em células.
        
    Returns:
        Tupla (x, y) com coordenadas da célula.
        
    Note:
        As coordenadas são clampadas para evitar índices fora dos limites.
    """
    x: int = int(min(pos.x // cell_size, grid_width - 1))
    y: int = int(min(pos.y // cell_size, grid_height - 1))
    
    return (x, y)


# -----------------------------------------------------------------------------
# Cálculos de Interceptação e Busca
# -----------------------------------------------------------------------------

def intercept_direction(
    chaser_pos: pygame.math.Vector2,
    chaser_speed: float,
    target_pos: pygame.math.Vector2,
    target_vel: pygame.math.Vector2
) -> pygame.math.Vector2:
    """
    Calcula a direção ótima de interceptação para um perseguidor alcançar um alvo.
    
    O método resolve para o tempo 't' de interceptação usando:
        |r + target_vel * t| = chaser_speed * t
    onde r = target_pos - chaser_pos.
    
    Se não houver solução válida (discriminante negativo ou t não-positivo),
    retorna o vetor normalizado do perseguidor para o alvo.
    
    Args:
        chaser_pos: Posição do perseguidor.
        chaser_speed: Velocidade constante do perseguidor.
        target_pos: Posição do alvo.
        target_vel: Velocidade do alvo.
        
    Returns:
        Velocidade como Vector2 na direção de interceptação.
        
    Note:
        Se não houver interceptação possível, retorna direção direta ao alvo.
    """
    r = target_pos - chaser_pos
    a: float = target_vel.dot(target_vel) - chaser_speed ** 2
    b: float = 2 * r.dot(target_vel)
    c: float = r.dot(r)

    t: float = 0.0  # Tempo de fallback

    # Resolver equação quadrática
    if abs(a) < 1e-6:  # Caso quase-linear
        if abs(b) > 1e-6:
            t = -c / b
    else:
        discriminant: float = b ** 2 - 4 * a * c
        if discriminant >= 0:
            sqrt_disc: float = math.sqrt(discriminant)
            t1: float = (-b + sqrt_disc) / (2 * a)
            t2: float = (-b - sqrt_disc) / (2 * a)
            t_candidates = [t_val for t_val in (t1, t2) if t_val > 0]
            if t_candidates:
                t = min(t_candidates)

    # Calcular ponto de interceptação e direção
    if t <= 0:
        direction = target_pos - chaser_pos
    else:
        intercept_point = target_pos + target_vel * t
        direction = intercept_point - chaser_pos
        
    # Normalizar e aplicar velocidade
    if direction.length() > 0:
        vel = direction.normalize() * chaser_speed
    else:
        vel = pygame.math.Vector2(0, 0)

    return vel


def can_intercept(
    chaser_pos: pygame.math.Vector2,
    chaser_speed: float,
    target_pos: pygame.math.Vector2,
    target_vel: pygame.math.Vector2,
    point_of_interest: Optional[pygame.math.Vector2] = None
) -> bool:
    """
    Determina se o perseguidor pode interceptar o alvo antes que ele atinja o ponto de interesse.
    
    Resolve a equação:
        ||r + target_vel * t|| = chaser_speed * t
    onde r = target_pos - chaser_pos.
    
    Retorna True se existir t > 0 para interceptação E se o tempo de interceptação
    for menor que o tempo para o alvo atingir o ponto de interesse.
    
    Args:
        chaser_pos: Posição do perseguidor.
        chaser_speed: Velocidade constante do perseguidor.
        target_pos: Posição do alvo.
        target_vel: Velocidade do alvo.
        point_of_interest: Ponto a ser defendido. Se None, usa INTEREST_POINT_CENTER.
        
    Returns:
        True se a interceptação for possível antes do alvo atingir o ponto de interesse.
        
    Example:
        >>> can_intercept(pos_friend, FRIEND_SPEED, pos_enemy, vel_enemy)
        True
    """
    if point_of_interest is None:
        point_of_interest = INTEREST_POINT_CENTER
        
    # Calcular coeficientes da equação quadrática
    r = target_pos - chaser_pos
    a: float = target_vel.dot(target_vel) - chaser_speed ** 2
    b: float = 2 * r.dot(target_vel)
    c: float = r.dot(r)
    
    # Resolver para o tempo de interceptação
    if abs(a) < 1e-6:  # Caso quase-linear
        if abs(b) > 1e-6:
            t: float = -c / b
        else:
            # Perseguidor já está no alvo
            return r.length() == 0
    else:
        discriminant: float = b ** 2 - 4 * a * c
        if discriminant < 0:
            return False  # Sem solução real
        sqrt_disc: float = math.sqrt(discriminant)
        t1: float = (-b + sqrt_disc) / (2 * a)
        t2: float = (-b - sqrt_disc) / (2 * a)
        t_candidates = [t_val for t_val in (t1, t2) if t_val > 0]
        if not t_candidates:
            return False
        t = min(t_candidates)

    # Verificar se tempo de interceptação é positivo
    if t <= 0:
        return False
    
    # Calcular tempo para alvo atingir ponto de interesse
    if target_vel.length() == 0:
        return False  # Alvo estacionário
    
    t_pi: float = (point_of_interest - target_pos).length() / target_vel.length()
    
    # Perseguidor intercepta se t < t_pi
    return t < t_pi


# -----------------------------------------------------------------------------
# Utilitários de Plotting
# -----------------------------------------------------------------------------

def plot_individual_states_matplotlib(state: Dict[str, Any]) -> None:
    """
    Gera um plot 3D dos estados de detecção de drones inimigos e amigos.
    
    Exibe dois subplots ("Enemy Detection" e "Friend Detection") com legenda
    de cores que mapeia cores para ângulos (em radianos π). Também plota a
    posição do drone como uma linha vermelha vertical.
    
    Args:
        state: Dicionário contendo informações de estado do drone.
               Deve conter as chaves:
               - 'pos': array com posição [x, y]
               - 'enemy_intensity': matriz de intensidade de inimigos
               - 'enemy_direction': matriz de direção de inimigos
               - 'friend_intensity': matriz de intensidade de amigos
               - 'friend_direction': matriz de direção de amigos
               
    Note:
        Usa matplotlib para renderização. A janela será exibida com plt.show().
    """
    fig = plt.figure(figsize=(8, 6))
    cmap = plt.cm.hsv  # Paleta de cores baseada em ângulos

    # Iterar sobre as duas views de detecção
    for idx, plot_view in enumerate(["Enemy Detection"]):
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect((1, 1, 0.4))

        # Criar meshgrid para dimensões da grid
        x = np.linspace(CELL_SIZE / 2, SIM_WIDTH - CELL_SIZE / 2, GRID_WIDTH)
        y = np.linspace(CELL_SIZE / 2, SIM_HEIGHT - CELL_SIZE / 2, GRID_HEIGHT)
        X, Y = np.meshgrid(x, y)

        if plot_view == "Enemy Detection":
            to_plot = state['enemy_intensity']
            direction = state['enemy_direction']
        elif plot_view == "Friend Detection":
            to_plot = state['friend_intensity']
            direction = state['friend_direction']

        # Suavizar matriz e obter direções resultantes
        Z_smoothed, result_direction = smooth_matrix_with_kernel_10x10(
            to_plot, direction, sigma=2, flat_radius=1
        )

        # Ajustar shape se necessário
        if Z_smoothed.shape != X.shape:
            Z_smoothed = Z_smoothed.T
            result_direction = result_direction.transpose(1, 0, 2)

        # Calcular ângulos dos vetores de direção
        dir_x = result_direction[..., 0]
        dir_y = result_direction[..., 1]
        angle = np.arctan2(dir_y, dir_x)
        norm_angle = (angle + math.pi) / (2 * math.pi)
        facecolors = cmap(norm_angle)
        facecolors[Z_smoothed < PLOT_THRESHOLD] = [1, 1, 1, 1]

        # Plotar superfície 3D
        ax.plot_surface(X, Y, Z_smoothed, facecolors=facecolors,
                        linewidth=0, antialiased=True, shade=False)
        ax.contourf(X, Y, Z_smoothed, zdir='x', offset=ax.get_xlim()[0], cmap="Greys")
        ax.contourf(X, Y, Z_smoothed, zdir='y', offset=ax.get_ylim()[0], cmap="Greys")
        ax.plot_wireframe(X, Y, Z_smoothed, color='black', linewidth=0.2, rstride=1, cstride=1)

        # Plotar posição do drone como linha vertical preta
        ax.plot([state['pos'][0], state['pos'][0]],
                [state['pos'][1], state['pos'][1]],
                [0, 1], color='black', linewidth=4, zorder=10)

        x_offset = ax.get_xlim()[0]
        ax.plot([x_offset, x_offset],
                [state['pos'][1], state['pos'][1]],
                [0, 1], color='black', linewidth=4, zorder=10)

        y_offset = ax.get_ylim()[0]
        ax.plot([state['pos'][0], state['pos'][0]],
                [y_offset, y_offset],
                [0, 1], color='black', linewidth=4, zorder=10)

        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Recency', fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, SIM_WIDTH)
        ax.set_ylim(SIM_HEIGHT, 0)
        ax.set_zlim(0, 1.5)

    fig.suptitle("Detection and Position Plot", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.75, 1])

    # Adicionar colorbar
    norm = plt.Normalize(vmin=-math.pi, vmax=math.pi)
    mappable_for_colorbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable_for_colorbar.set_array([])
    cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.7])
    cbar = fig.colorbar(mappable_for_colorbar, cax=cbar_ax, label='Angle (rad)')
    cbar.set_ticks([-math.pi, -math.pi / 2, 0, math.pi / 2, math.pi])
    cbar.set_ticklabels(['-π', '-π/2', '0', '+π/2', '+π'])
    plt.show()


# -----------------------------------------------------------------------------
# Geração de Matrizes Esparsas
# -----------------------------------------------------------------------------

def generate_sparse_matrix(
    shape: Tuple[int, int] = (GRID_WIDTH, GRID_HEIGHT),
    max_nonzero: int = 20, 
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera uma matriz esparsa com até 'max_nonzero' elementos não-zero.
    
    Valores não-zero são amostrados de uma distribuição normal (média=1, std=1)
    e clampados para [0, 1]. Para cada célula não-zero, um vetor de direção
    normalizado aleatório é gerado.
    
    Args:
        shape: Forma da matriz de intensidade (linhas, colunas).
        max_nonzero: Número máximo de elementos não-zero.
        seed: Seed para geração de números aleatórios. Se None, usa seed aleatória.
        
    Returns:
        Tupla contendo:
        - matrix: Array com valores de intensidade.
        - direction: Array de forma (rows, cols, 2) com vetores de direção.
                    Células com intensidade zero têm vetor (0, 0).
                    
    Example:
        >>> matrix, direction = generate_sparse_matrix((50, 50), max_nonzero=10)
        >>> assert matrix.shape == (50, 50)
        >>> assert direction.shape == (50, 50, 2)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    matrix = np.zeros(shape)
    direction = np.zeros((shape[0], shape[1], 2), dtype=float)
    total_cells: int = shape[0] * shape[1]
    nonzero_count: int = min(random.randint(0, max_nonzero), total_cells)
    chosen_indices = random.sample(range(total_cells), nonzero_count)
    
    for idx in chosen_indices:
        i: int = idx // shape[1]
        j: int = idx % shape[1]
        value: float = np.random.normal(1, 1)
        matrix[i, j] = np.clip(value, 0, 1)
        
        # Gerar vetor de direção normalizado aleatório
        vec = pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        if vec.length() > 0:
            vec = vec.normalize()
        else:
            vec = pygame.math.Vector2(0, 0)
        direction[i, j] = (vec.x, vec.y)
    
    return matrix, direction


# -----------------------------------------------------------------------------
# Utilitários TensorFlow
# -----------------------------------------------------------------------------

# def load_best_model(
#     directory: str, 
#     pattern: str, 
#     custom_objects: Optional[Dict[str, Any]] = None
# ) -> Optional[Any]:
#     """
#     Carrega o melhor modelo do diretório especificado selecionando o arquivo
#     com a menor validation loss, e extrai o tamanho do nome do arquivo.
    
#     Args:
#         directory: Diretório contendo arquivos de modelo salvos.
#         pattern: Padrão regex para extrair o valor de validation loss do nome do arquivo.
#         custom_objects: Objetos customizados para passar ao load_model.
        
#     Returns:
#         O modelo carregado (tf.keras.Model), ou None se nenhum modelo for encontrado
#         ou ocorrer erro no carregamento.
        
#     Raises:
#         Não levanta exceções; imprime mensagens de erro e retorna None em caso de falha.
        
#     Example:
#         >>> model = load_best_model("./models", r"val_loss_(\\d+\\.\\d+)")
#         >>> if model:
#         ...     predictions = model.predict(data)
#     """
#     tf = set_tensorflow()
#     best_file: str = ""
#     min_val_metric_loss: float = float("inf")

#     # Iterar sobre arquivos para encontrar o com menor val_metric_loss
#     if not os.path.exists(directory):
#         print(f"Directory not found: {directory}")
#         return None
        
#     for filename in os.listdir(directory):
#         if filename.endswith(".keras"):
#             match = re.search(pattern, filename)
#             if match:
#                 val_metric_loss: float = float(match.group(1))
#                 if val_metric_loss < min_val_metric_loss:
#                     min_val_metric_loss = val_metric_loss
#                     best_file = filename

#     if not best_file:
#         print(f"No model files found in the directory: {directory}")
#         return None

#     model_path: str = os.path.join(directory, best_file)
#     try:
#         model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
#     except Exception as e:
#         print(f"Error loading the model: {e}")
#         return None

#     print(f"\n\nLoaded model: {best_file} with val_metric_loss={min_val_metric_loss:.4f}\n\n")
#     return model


# -----------------------------------------------------------------------------
# Utilitários de Estratégia
# -----------------------------------------------------------------------------

def get_friends_hold(
    state: Dict[str, Any], 
    friend_activation_threshold_position: float = 0.7,
    enemy_activation_threshold_position: float = 0.4
) -> list:
    """
    Determina quais drones amigos devem manter posição defensiva.
    
    Analisa o estado atual e atribui drones amigos para interceptar inimigos
    detectados. Drones que não conseguem interceptar ou não foram atribuídos
    devem manter posição ("hold").
    
    Args:
        state: Dicionário de estado contendo matrizes de detecção e posição.
        friend_activation_threshold_position: Limiar de intensidade para considerar amigos.
        enemy_activation_threshold_position: Limiar de intensidade para considerar inimigos.
        
    Returns:
        Lista de tuplas (cell, position) indicando posições de hold para drones amigos.
        
    Note:
        Usa can_intercept() para determinar se um drone pode interceptar um alvo.
    """
    # Extração e preparação do estado
    pos = np.squeeze(state['pos'])
    pos = pygame.math.Vector2(pos[0], pos[1])
    friend_intensity = np.squeeze(state['friend_intensity'])
    enemy_intensity = np.squeeze(state['enemy_intensity'])
    enemy_direction = np.squeeze(state['enemy_direction'])
    
    enemy_targets = []
    # Identificar células com inimigos detectados
    for cell, intensity in np.ndenumerate(enemy_intensity):
        if intensity < enemy_activation_threshold_position:
            continue
        
        target_pos = pygame.math.Vector2(
            (cell[0] + 0.5) * CELL_SIZE,
            (cell[1] + 0.5) * CELL_SIZE
        )
        distance_to_interest: float = target_pos.distance_to(INTEREST_POINT_CENTER)
        
        enemy_targets.append((cell, target_pos, distance_to_interest))
        
    # Obter célula do drone atual
    my_cell = pos_to_cell(pos)
    my_cell_center = pygame.math.Vector2(
        (my_cell[0] + 0.5) * CELL_SIZE,
        (my_cell[1] + 0.5) * CELL_SIZE
    )
    
    # Obter candidatos amigos
    friend_candidates = []
    for cell, intensity in np.ndenumerate(friend_intensity):
        if intensity >= friend_activation_threshold_position:
            candidate_pos = pygame.math.Vector2(
                (cell[0] + 0.5) * CELL_SIZE,
                (cell[1] + 0.5) * CELL_SIZE
            )
            friend_candidates.append((cell, candidate_pos))
            
    # Incluir própria célula se não houver detecção ativa
    if not any(cell == my_cell for cell, _ in friend_candidates):
        friend_candidates.append((my_cell, my_cell_center))
            
    if not enemy_targets:
        # Nenhum alvo detectado - todos mantêm posição
        friends_hold = [(cell, candidate_pos) for (cell, candidate_pos) in friend_candidates]
        return friends_hold
    
    # Ordenar alvos por proximidade ao ponto de interesse
    enemy_targets.sort(key=lambda t: t[2])
    
    engagement_assignment = {}
    assigned_friend_cells = set()
    
    # Atribuir candidatos amigos aos alvos inimigos
    for cell, target_pos, _ in enemy_targets:
        sorted_candidates = sorted(
            friend_candidates, 
            key=lambda x: x[1].distance_to(target_pos)
        )
        enemy_dir_vec = pygame.math.Vector2(
            enemy_direction[cell][0], 
            enemy_direction[cell][1]
        ) * ENEMY_SPEED
        
        for candidate in sorted_candidates:
            candidate_cell, candidate_pos = candidate
            
            if (candidate_cell not in assigned_friend_cells and 
                can_intercept(candidate_pos, FRIEND_SPEED, target_pos, 
                            enemy_dir_vec, INTEREST_POINT_CENTER)):
                
                engagement_assignment[tuple(target_pos)] = candidate_cell
                assigned_friend_cells.add(candidate_cell)
                break  # Próximo alvo
    
    # Drones não atribuídos devem manter posição
    friends_hold = [
        (cell, candidate_pos) 
        for (cell, candidate_pos) in friend_candidates
        if cell not in assigned_friend_cells
    ]
    
    return friends_hold