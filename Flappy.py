import asyncio
import platform
import pygame
import random
import math
from NeuralNetwork import FlappyBirdNN
import torch
import torch.nn as nn

# Initialize Pygame
pygame.init()

# Game constants
WIDTH, HEIGHT = 400, 600
BIRD_X, BIRD_Y = 100, 300
GRAVITY = 0.5
FLAP = -8
PIPE_WIDTH = 50
PIPE_GAP = 150
PIPE_SPEED = 3
FPS = 60

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Game variables
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

NUM_BIRDS = 300

class Bird:
    def __init__(self):
        self.x = BIRD_X
        self.y = BIRD_Y
        self.velocity = 0
    
    def flap(self):
        self.velocity = FLAP
    
    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity
    
    def draw(self):
        pygame.draw.circle(screen, BLUE, (int(self.x), int(self.y)), 20)

class Pipe:
    def __init__(self):
        self.x = WIDTH
        self.height = random.randint(150, 400)
    
    def update(self):
        self.x -= PIPE_SPEED
    
    def draw(self):
        pygame.draw.rect(screen, GREEN, (self.x, 0, PIPE_WIDTH, self.height))
        pygame.draw.rect(screen, GREEN, (self.x, self.height + PIPE_GAP, PIPE_WIDTH, HEIGHT))

    def offscreen(self):
        return self.x < -PIPE_WIDTH
    
    def collides(self, bird):
        bird_rect = pygame.Rect(bird.x - 20, bird.y - 20, 40, 40)
        top_pipe = pygame.Rect(self.x, 0, PIPE_WIDTH, self.height)
        bottom_pipe = pygame.Rect(self.x, self.height + PIPE_GAP, PIPE_WIDTH, HEIGHT)
        return bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe)
    
def fitness_function(bird):
    # Fitness function based on total x distance traveled
    return bird.x

def next_generation():
    global birds, models, fitness, generation, alive, pipes, score, game_over
    chart_data.append((generation, score))
    # Select top 10% as parents (at least 2)
    num_parents = max(2, int(NUM_BIRDS * 0.01))
    sorted_indices = sorted(range(NUM_BIRDS), key=lambda i: fitness[i], reverse=True)
    parents = [models[i] for i in sorted_indices[:num_parents]]
    
    new_models = []
    for _ in range(NUM_BIRDS):
        parent1, parent2 = random.sample(parents, 2)
        # Use your neural network's crossover method
        child = parent1.crossover(parent2)
        # Mutate child weights (adjust mutation rate as needed)
        child.mutate(rate=0.1)
        new_models.append(child)
    models = new_models

    # Reset game values for a new generation
    fitness = [0 for _ in range(NUM_BIRDS)]
    birds = [Bird() for _ in range(NUM_BIRDS)]
    alive = [True for _ in range(NUM_BIRDS)]
    pipes = [Pipe()]
    score = 0
    generation += 1
    game_over = False

def setup():
    global birds, pipes, score, game_over, models, alive, fitness, generation
    fitness = [0 for _ in range(NUM_BIRDS)]
    models = [FlappyBirdNN() for _ in range(NUM_BIRDS)]
    birds = [Bird() for _ in range(NUM_BIRDS)]
    pipes = [Pipe()]
    score = 0
    generation = 0
    game_over = False
    alive = [True for _ in range(NUM_BIRDS)]

chart_data = [] 
def draw_chart():
    global chart_data, screen
    font = pygame.font.Font(None, 24)
    # Title for the table
    header = font.render("Gen    Score", True, (0, 0, 0))
    screen.blit(header, (WIDTH - 150, 20))
    # Draw the last few generations (e.g., last 5)
    start_y = 50
    for gen, scr in chart_data[-5:]:
        line = font.render(f"{gen:<6} {scr}", True, (0, 0, 0))
        screen.blit(line, (WIDTH - 150, start_y))
        start_y += 20

def update_loop():
    global birds, pipes, score, game_over, models, alive, fitness, generation
    if not game_over:
        for i, bird in enumerate(birds):
            if not alive[i]:
                continue
            if pipes:
                pipe = next((p for p in pipes if p.x + PIPE_WIDTH > bird.x), None)
                if pipe:
                    x_dist = pipe.x - bird.x
                    y_top = bird.y - pipe.height
                    y_bottom = bird.y - (pipe.height + PIPE_GAP)
                    input_tensor = torch.tensor([x_dist, y_top, y_bottom], dtype=torch.float32)
                    output = models[i](input_tensor)
                    if output.item() > 0.5:
                        bird.flap()
            bird.update()

            # Check collisions
            for pipe in pipes[:]:
                if pipe.collides(bird):
                    alive[i] = False
            if bird.y > HEIGHT or bird.y < 0:
                alive[i] = False
            # Update fitness each bird that is alive update its fitness 
            # based on flight time
            if alive[i]:
                fitness[i] += 1 / FPS

        # If all birds are dead, end game
        if pipes[-1].x < WIDTH - 200:
            pipes.append(Pipe())
        for pipe in pipes[:]:
            pipe.update()
            if pipe.offscreen():
                pipes.remove(pipe)
                score += 1
        if not any(alive):
            game_over = True
        
        # # Update pipes
        # if pipes[-1].x < WIDTH - 200:
        #     pipes.append(Pipe())
        # for pipe in pipes[:]:
        #     pipe.update()
        #     if pipe.offscreen():
        #         pipes.remove(pipe)
        #         score += 1
        #     if pipe.collides(bird):
        #         game_over = True
        
        # # Check boundaries
        # if bird.y > HEIGHT or bird.y < 0:
        #     game_over = True
    
    # Draw everything
    screen.fill(WHITE)
    for pipe in pipes:
        pipe.draw()
    for i, bird in enumerate(birds):
        if alive[i]:
            bird.draw()
    font = pygame.font.Font(None, 36)
    score_text = font.render(str(score), True, (0, 0, 0))
    generation_text = font.render(str(generation), True, (0, 0, 0))
    screen.blit(score_text, (WIDTH // 2, 50))
    screen.blit(generation_text, (WIDTH // 2, 100)) 
    if game_over:
        game_over_text = font.render("Game Over", True, (255, 0, 0))
        screen.blit(game_over_text, (WIDTH // 2 - 80, HEIGHT // 2))

    draw_chart()
    pygame.display.flip()

async def main():
    setup()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return  # or use exit() to force quit
            # if event.type == pygame.KEYDOWN:
            #     # Option to manually trigger a reset or next generation
        if game_over:
            next_generation()
        update_loop()
        await asyncio.sleep(0)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())