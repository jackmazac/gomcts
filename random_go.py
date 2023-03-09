import pygame
import random

# Define board size and screen dimensions
B = 9
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
GRID_SIZE = SCREEN_WIDTH // B
RADIUS = GRID_SIZE // 2 - 5

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Go")
clock = pygame.time.Clock()

# Create board and draw grid
board = [['.' for i in range(B)] for j in range(B)]

def draw_grid(surface):
    for i in range(B):
        pygame.draw.line(surface, BLACK, (GRID_SIZE//2, GRID_SIZE//2 + i*GRID_SIZE), (SCREEN_WIDTH - GRID_SIZE//2, GRID_SIZE//2 + i*GRID_SIZE), 2)
        pygame.draw.line(surface, BLACK, (GRID_SIZE//2 + i*GRID_SIZE, GRID_SIZE//2), (GRID_SIZE//2 + i*GRID_SIZE, SCREEN_HEIGHT - GRID_SIZE//2), 2)

def draw_stones(surface):
    for i in range(B):
        for j in range(B):
            if board[i][j] != '.':
                color = BLACK if board[i][j] == 'B' else WHITE
                pygame.draw.circle(surface, color, (j*GRID_SIZE+GRID_SIZE//2, i*GRID_SIZE+GRID_SIZE//2), RADIUS)

def is_valid_move(x, y):
    if x < 0 or x >= len(board) or y < 0 or y >= len(board):
        return False
    if board[x][y] != '.':
        return False
    return True

def place_stone(x, y, color):
    board[x][y] = color

def get_random_move():
    while True:
        x = random.randint(0, len(board) - 1)
        y = random.randint(0, len(board) - 1)
        if is_valid_move(x, y):
            return x, y

# Main game loop
turn = 0
game_over = False
while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True

    if turn % 2 == 0:
        color = 'B'
        x, y = get_random_move()
        place_stone(x, y, color)
        turn += 1
    else:
        color = 'W'
        x, y = get_random_move()
        place_stone(x, y, color)
        turn += 1

    # Check if all intersections on the board are filled
    if turn == B * B:
        game_over = True

    # Draw grid and stones
    surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    surface.fill(GRAY)
    draw_grid(surface)
    draw_stones(surface)
    screen.blit(surface, (0, 0))

    # Update display
    pygame.display.update()

    # Limit frame rate
    clock.tick(30)

# Determine the winner
black_stones = sum([row.count('B') for row in board])
white_stones = sum([row.count('W') for row in board])
if black_stones > white_stones:
    print("Black wins with {} stones!".format(black_stones))
elif white_stones > black_stones:
    print("White wins with {} stones!".format(white_stones))
else:
    print("The game is a tie!")

# Quit Pygame
pygame.quit()

