import pygame
import random

pygame.init()

# Set up the window
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Random Walk Simulation")

# Set up the walker
walker_x, walker_y = screen_width // 2, screen_height // 2
walker_color = pygame.Color("white")
walker_radius = 1

# Set the desired simulation speed
simulation_speed = 1  # in milliseconds

# Game loop
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Randomly move the walker
    direction = random.choice(["up", "down", "left", "right"])
    if direction == "up":
        walker_y -= 1
    elif direction == "down":
        walker_y += 1
    elif direction == "left":
        walker_x -= 1
    elif direction == "right":
        walker_x += 1

    # Wrap the walker around the screen
    if walker_x < 0:
        walker_x = screen_width - 1
    elif walker_x >= screen_width:
        walker_x = 0
    if walker_y < 0:
        walker_y = screen_height - 1
    elif walker_y >= screen_height:
        walker_y = 0

    # Clear the screen
    # screen.fill((0, 0, 0))

    # Draw the walker
    pygame.draw.circle(screen, walker_color, (walker_x, walker_y), walker_radius)

    # Update the display
    pygame.display.flip()

    # Delay for the desired simulation speed
    pygame.time.delay(simulation_speed)

# Quit the game
pygame.quit()