import pygame
import sys

# Inicializar pygame
pygame.init()
pygame.joystick.init()

# Configurações da tela
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Controle PS5 com Pygame")

# Cores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Verificar controles conectados
if pygame.joystick.get_count() == 0:
    print("Nenhum controle conectado!")
    print("Conecte seu controle PS5 via USB ou Bluetooth")
    sys.exit()

# Conectar ao controle PS5
joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"Controle conectado: {joystick.get_name()}")

# Mapeamento dos botões do PS5
PS5_BUTTONS = {
    0: "X", 1: "Círculo", 2: "Quadrado", 3: "Triângulo",
    4: "Share", 5: "PS", 6: "Options", 7: "L3", 8: "R3",
    9: "L1", 10: "R1", 11: "D-Pad ↑", 12: "D-Pad ↓",
    13: "D-Pad ←", 14: "D-Pad →", 15: "Touchpad"
}

# Jogador
player_x = WIDTH // 2
player_y = HEIGHT // 2
player_speed = 5
player_size = 30

# Configurações do jogo
clock = pygame.time.Clock()
running = True
font = pygame.font.Font(None, 36)

# Variáveis para demonstrar funcionalidades
button_states = {}
vibration_time = 0

def handle_joystick_input():
    global player_x, player_y, vibration_time
    
    # Movimento com analógico esquerdo
    left_x = joystick.get_axis(0)  # -1 a 1
    left_y = joystick.get_axis(1)  # -1 a 1
    
    # Dead zone para evitar drift
    if abs(left_x) > 0.1:
        player_x += left_x * player_speed
    if abs(left_y) > 0.1:
        player_y += left_y * player_speed
    
    # Limites da tela
    player_x = max(player_size, min(WIDTH - player_size, player_x))
    player_y = max(player_size, min(HEIGHT - player_size, player_y))
    
    # Movimento com D-Pad
    if joystick.get_button(11):  # D-Pad Cima
        player_y -= player_speed
    if joystick.get_button(12):  # D-Pad Baixo
        player_y += player_speed
    if joystick.get_button(13):  # D-Pad Esquerda
        player_x -= player_speed
    if joystick.get_button(14):  # D-Pad Direita
        player_x += player_speed
    
    # Triggers para vibração (apenas no Windows/Linux com drivers adequados)
    l2_trigger = joystick.get_axis(4)  # -1 a 1
    r2_trigger = joystick.get_axis(5)  # -1 a 1
    
    # Converter triggers de -1,1 para 0,1
    l2_value = (l2_trigger + 1) / 2
    r2_value = (r2_trigger + 1) / 2
    
    # Vibração baseada nos triggers
    if l2_value > 0.5 or r2_value > 0.5:
        # Nota: vibração pode não funcionar em todos os sistemas
        try:
            joystick.rumble(l2_value, r2_value, 100)
            vibration_time = 10
        except:
            pass  # Vibração não suportada
    
    # Atualizar estados dos botões
    for i in range(joystick.get_numbuttons()):
        button_states[i] = joystick.get_button(i)

def draw_info():
    y_offset = 10
    
    # Informações do controle
    info_text = font.render(f"Controle: {joystick.get_name()}", True, WHITE)
    screen.blit(info_text, (10, y_offset))
    y_offset += 30
    
    # Posição do jogador
    pos_text = font.render(f"Posição: ({int(player_x)}, {int(player_y)})", True, WHITE)
    screen.blit(pos_text, (10, y_offset))
    y_offset += 30
    
    # Estados dos analógicos
    left_stick = font.render(f"Analógico L: ({joystick.get_axis(0):.2f}, {joystick.get_axis(1):.2f})", True, WHITE)
    screen.blit(left_stick, (10, y_offset))
    y_offset += 25
    
    right_stick = font.render(f"Analógico R: ({joystick.get_axis(2):.2f}, {joystick.get_axis(3):.2f})", True, WHITE)
    screen.blit(right_stick, (10, y_offset))
    y_offset += 25
    
    # Triggers
    l2_val = (joystick.get_axis(4) + 1) / 2
    r2_val = (joystick.get_axis(5) + 1) / 2
    triggers = font.render(f"L2: {l2_val:.2f} | R2: {r2_val:.2f}", True, WHITE)
    screen.blit(triggers, (10, y_offset))
    y_offset += 30
    
    # Botões pressionados
    pressed_buttons = [PS5_BUTTONS.get(i, f"Btn{i}") for i, pressed in button_states.items() if pressed]
    if pressed_buttons:
        buttons_text = font.render(f"Botões: {', '.join(pressed_buttons)}", True, GREEN)
        screen.blit(buttons_text, (10, y_offset))

def draw_player():
    # Cor do jogador baseada nos botões
    color = WHITE
    if button_states.get(0):  # X
        color = BLUE
    elif button_states.get(1):  # Círculo
        color = RED
    elif button_states.get(2):  # Quadrado
        color = (255, 0, 255)  # Magenta
    elif button_states.get(3):  # Triângulo
        color = GREEN
    
    # Desenhar jogador
    pygame.draw.circle(screen, color, (int(player_x), int(player_y)), player_size)
    
    # Indicador de direção baseado no analógico direito
    right_x = joystick.get_axis(2)
    right_y = joystick.get_axis(3)
    
    if abs(right_x) > 0.3 or abs(right_y) > 0.3:
        end_x = player_x + right_x * 50
        end_y = player_y + right_y * 50
        pygame.draw.line(screen, RED, (player_x, player_y), (end_x, end_y), 3)

# Loop principal
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # Eventos específicos do joystick
        elif event.type == pygame.JOYBUTTONDOWN:
            print(f"Botão pressionado: {PS5_BUTTONS.get(event.button, f'Botão {event.button}')}")
            
            # Sair com botão Options
            if event.button == 6:  # Options
                running = False
        
        elif event.type == pygame.JOYBUTTONUP:
            print(f"Botão solto: {PS5_BUTTONS.get(event.button, f'Botão {event.button}')}")
        
        elif event.type == pygame.JOYAXISMOTION:
            # Só mostra movimento significativo dos eixos
            if abs(event.value) > 0.3:
                axis_name = ["L-X", "L-Y", "R-X", "R-Y", "L2", "R2"][event.axis] if event.axis < 6 else f"Eixo{event.axis}"
                print(f"{axis_name}: {event.value:.2f}")
    
    # Atualizar lógica do jogo
    handle_joystick_input()
    
    # Desenhar tudo
    screen.fill(BLACK)
    draw_player()
    draw_info()
    
    # Atualizar tela
    pygame.display.flip()
    clock.tick(60)

# Finalizar
pygame.quit()
sys.exit()