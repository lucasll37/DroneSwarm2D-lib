import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle
import os

# Criar diretório para salvar imagens
os.makedirs('./images', exist_ok=True)

def create_fsm_diagram():
    """
    Cria um diagrama de máquina de estados finitos baseado na API mostrada na imagem.
    Estados: RETURN, HOLD_WAIT, PURSUING (central/vermelho), HOLD_NO_ENOUGH_COMM, 
             HOLD_GO_INTCPT, HOLD_INTCPT
    """
    
    # Configuração da figura com fundo transparente - ajustada para área menor
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(2, 9.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Fundo transparente
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0)  # Transparente
    
    # Definir posições dos estados (layout circular com PURSUING no centro)
    states = {
        'PURSUING': {'pos': (5, 5), 'label': 'PURSUING', 'color': '#FF0000', 'size': 1.2},  # Estado central vermelho
        'RETURN': {'pos': (5, 8.5), 'label': 'RETURN', 'color': '#404040', 'size': 0.8},
        'HOLD_WAIT': {'pos': (1.5, 6.5), 'label': 'HOLD - WAIT', 'color': '#404040', 'size': 1.2},
        'HOLD_NO_ENOUGH_COMM': {'pos': (8.5, 6.5), 'label': 'HOLD\nNO_ENOUGH_COMM', 'color': '#404040', 'size': 1.2},
        'HOLD_GO_INTCPT': {'pos': (1.5, 3.5), 'label': 'HOLD\nGO INTCPT', 'color': '#404040', 'size': 1.2},
        'HOLD_INTCPT': {'pos': (8.5, 3.5), 'label': 'HOLD - INTCPT', 'color': '#404040', 'size': 1.2}
    }
    
    # Definir transições (baseado nas setas visíveis na imagem)
    transitions = [
        # Transições de PURSUING para outros estados
        ('PURSUING', 'RETURN', '', '#000000', 0.0),
        ('PURSUING', 'HOLD_WAIT', '', '#000000', 0.0),
        ('PURSUING', 'HOLD_NO_ENOUGH_COMM', '', '#000000', 0.0),
        ('PURSUING', 'HOLD_GO_INTCPT', '', '#000000', 0.0),
        ('PURSUING', 'HOLD_INTCPT', '', '#000000', 0.0),
        
        # Transições de volta para PURSUING
        ('RETURN', 'PURSUING', '', '#000000', 0.0),
        ('HOLD_WAIT', 'PURSUING', '', '#000000', 0.0),
        ('HOLD_NO_ENOUGH_COMM', 'PURSUING', '', '#000000', 0.0),
        ('HOLD_GO_INTCPT', 'PURSUING', '', '#000000', 0.0),
        ('HOLD_INTCPT', 'PURSUING', '', '#000000', 0.0),
        
        # Transições entre estados HOLD
        ('HOLD_WAIT', 'RETURN', '', '#000000', 0.3),
        ('RETURN', 'HOLD_NO_ENOUGH_COMM', '', '#000000', 0.3),
        ('HOLD_NO_ENOUGH_COMM', 'HOLD_INTCPT', '', '#000000', 0.0),
        ('HOLD_INTCPT', 'HOLD_GO_INTCPT', '', '#000000', 0.0),
        ('HOLD_GO_INTCPT', 'HOLD_WAIT', '', '#000000', 0.3),
        
        # Transições circulares entre estados periféricos
        ('HOLD_WAIT', 'HOLD_GO_INTCPT', '', '#000000', 0.4),
        ('HOLD_GO_INTCPT', 'HOLD_INTCPT', '', '#000000', 0.4),
        ('HOLD_INTCPT', 'HOLD_NO_ENOUGH_COMM', '', '#000000', 0.4),
        ('HOLD_NO_ENOUGH_COMM', 'RETURN', '', '#000000', 0.4),
        ('RETURN', 'HOLD_WAIT', '', '#000000', 0.4),
    ]
    
    # Função para desenhar seta reta ou curvada
    def draw_arrow(start, end, color, curvature=0.0):
        x1, y1 = states[start]['pos']
        x2, y2 = states[end]['pos']
        
        start_radius = states[start]['size'] * 0.6
        end_radius = states[end]['size'] * 0.6
        
        if curvature == 0.0:
            # Seta reta
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            
            # Normalizar
            dx_norm = dx / length
            dy_norm = dy / length
            
            # Ajustar pontos para não sobrepor os círculos
            start_x = x1 + start_radius * dx_norm
            start_y = y1 + start_radius * dy_norm
            end_x = x2 - end_radius * dx_norm
            end_y = y2 - end_radius * dy_norm
            
            ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        else:
            # Seta curvada
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            
            # Vetor perpendicular
            perp_x = -dy / length
            perp_y = dx / length
            
            # Ponto de controle
            ctrl_x = mid_x + curvature * perp_x * length
            ctrl_y = mid_y + curvature * perp_y * length
            
            # Ajustar pontos de início e fim
            angle_start = np.arctan2(ctrl_y - y1, ctrl_x - x1)
            angle_end = np.arctan2(ctrl_y - y2, ctrl_x - x2)
            
            start_x = x1 + start_radius * np.cos(angle_start)
            start_y = y1 + start_radius * np.sin(angle_start)
            end_x = x2 + end_radius * np.cos(angle_end + np.pi)
            end_y = y2 + end_radius * np.sin(angle_end + np.pi)
            
            # Curva bezier
            t = np.linspace(0, 1, 100)
            curve_x = (1-t)**2 * start_x + 2*(1-t)*t * ctrl_x + t**2 * end_x
            curve_y = (1-t)**2 * start_y + 2*(1-t)*t * ctrl_y + t**2 * end_y
            
            ax.plot(curve_x, curve_y, color=color, linewidth=1.5)
            
            # Seta no final
            ax.annotate('', xy=(end_x, end_y), 
                       xytext=(curve_x[-5], curve_y[-5]),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    
    # Desenhar estados
    for state_id, state_info in states.items():
        x, y = state_info['pos']
        radius = state_info['size'] * 0.6
        
        # Círculo do estado
        circle = Circle((x, y), radius, 
                       facecolor=state_info['color'], 
                       edgecolor='#000000',  # Borda preta
                       linewidth=2,
                       zorder=10)
        ax.add_patch(circle)
        
        # Texto do estado
        ax.text(x, y, state_info['label'], ha='center', va='center', 
                fontsize=10 if state_id == 'PURSUING' else 8, 
                fontweight='bold', 
                color='white' if state_id == 'PURSUING' else 'white',
                zorder=15)
    
    # Desenhar todas as transições
    for transition in transitions:
        start, end, label, color, curvature = transition
        draw_arrow(start, end, color, curvature)
    
    plt.tight_layout()
    
    # Salvar a imagem
    output_path = "./images/fsm_api_based.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', transparent=True)
    plt.close()
    
    print(f"Diagrama FSM baseado na API salvo como: {output_path}")
    
    return output_path

def create_customizable_fsm(custom_states=None, custom_transitions=None):
    """
    Versão customizável do diagrama FSM com fundo transparente.
    
    Parameters:
    custom_states: dict - Dicionário com estados personalizados
    custom_transitions: list - Lista com transições personalizadas
    """
    
    if custom_states is None:
        # Estados padrão
        custom_states = {
            'S0': {'pos': (5, 7), 'label': 'State 0', 'color': '#E8F4FD'},
            'S1': {'pos': (2, 5), 'label': 'State 1', 'color': '#D4E6F1'},
            'S2': {'pos': (8, 5), 'label': 'State 2', 'color': '#FCF3CF'},
            'S3': {'pos': (1, 2), 'label': 'State 3', 'color': '#FADBD8'},
            'S4': {'pos': (9, 2), 'label': 'State 4', 'color': '#D5F4E6'},
            'S5': {'pos': (5, 0.5), 'label': 'State 5', 'color': '#EBDEF0'}
        }
    
    if custom_transitions is None:
        # Transições padrão
        custom_transitions = [
            ('S0', 'S1', 'transition_1', '#2E86AB', 0.2),
            ('S1', 'S2', 'transition_2', '#A23B72', 0.3),
            ('S2', 'S3', 'transition_3', '#F18F01', 0.2),
            ('S3', 'S4', 'transition_4', '#C73E1D', 0.4),
            ('S4', 'S5', 'transition_5', '#7B68EE', 0.2),
            ('S5', 'S0', 'transition_6', '#228B22', 0.3),
        ]
    
    # Configuração da figura com fundo transparente
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Fundo transparente
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(0)
    
    # Desenhar estados customizados
    state_radius = 0.6
    for state_id, state_info in custom_states.items():
        x, y = state_info['pos']
        
        circle = Circle((x, y), state_radius, 
                       facecolor=state_info['color'], 
                       edgecolor='black', 
                       linewidth=2.5,
                       zorder=10)
        ax.add_patch(circle)
        
        ax.text(x, y + 0.1, state_id, ha='center', va='center', 
                fontsize=12, fontweight='bold', zorder=15)
        ax.text(x, y - 0.2, state_info['label'], ha='center', va='center', 
                fontsize=9, style='italic', zorder=15)
    
    plt.tight_layout()
    output_path = "./images/custom_fsm_transparent.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', transparent=True)
    plt.close()
    
    return output_path

# Executar a função principal
if __name__ == "__main__":
    create_fsm_diagram()
    print("Diagrama de máquina de estados finitos baseado na API criado com sucesso!")
    
    # Exemplo de como usar a versão customizável:
    # custom_states = {
    #     'INIT': {'pos': (5, 7), 'label': 'Initialize', 'color': '#FFE4E1'},
    #     'WORK': {'pos': (3, 4), 'label': 'Working', 'color': '#E0FFE0'},
    #     # ... etc
    # }
    # 
    # custom_transitions = [
    #     ('INIT', 'WORK', 'start', '#FF0000', 0.2),
    #     # ... etc
    # ]
    # 
    # create_customizable_fsm(custom_states, custom_transitions)