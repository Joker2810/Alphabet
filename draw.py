import os
import pygame
from PIL import Image

# Initialize Pygame
pygame.init()
font = pygame.font.SysFont('Arial', 15)

def main():

    # Set the window size
    WIDTH, HEIGHT = 280, 280

    # Create the window
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # Set the title
    pygame.display.set_caption('Drawing App')

    # Initialize some variables
    drawing = False  # Keep track of whether we are drawing or not
    last_pos = (0, 0)  # Keep track of the previous mouse position
    drawn_area_rect = None  # Bounding box of the drawn area

    # Run the game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                os.remove("Playerdrawing.jpg")
                pygame.image.save(screen, "Playerdrawing.jpg")
                with Image.open("Playerdrawing.jpg") as im:
                    # Resize the image
                    im_resized = im.resize((28, 28))
                    # Save the resized image
                    im_resized.save('Playerdrawing.jpg')
                im = Image.open('Playerdrawing.jpg')
                im_gray = im.convert('L')
                im_gray.save('Playerdrawinggray.jpg')
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Start drawing
                drawing = True
                last_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                # Stop drawing
                if drawing:
                    '''os.remove("Playerdrawing.jpg")
                    pygame.image.save(screen, "Playerdrawing.jpg")
                    with Image.open("Playerdrawing.jpg") as im:
                        # Resize the image
                        im_resized = im.resize((28, 28))
                        # Save the resized image
                        im_resized.save('Playerdrawing.jpg')
                    im = Image.open('Playerdrawing.jpg')
                    im_gray = im.convert('L')
                    im_gray.save('Playerdrawinggray.jpg')
                    running = False'''
                    pass
                drawing = False
            elif event.type == pygame.MOUSEMOTION:
                # If we are drawing, draw a line from the previous mouse position to the current mouse position
                if drawing:
                    pygame.draw.line(screen, (255, 255, 255), last_pos, event.pos, 40)  # Adjusted line thickness
                    last_pos = event.pos

        pygame.display.update()

    # Find the bounding box of the drawn area
    drawn_area_rect = get_drawn_area_rect(screen)

    # Copy the drawn area to a new surface
    drawn_surface = pygame.Surface((drawn_area_rect.width, drawn_area_rect.height), pygame.SRCALPHA)
    drawn_surface.blit(screen, (0, 0), area=drawn_area_rect)

    # Create a new black window
    new_screen = pygame.display.set_mode((WIDTH, HEIGHT))
    new_screen.fill((0, 0, 0))  # Fill with black color

    # Calculate the position to paste the drawn content at the center
    center_x = (WIDTH - drawn_area_rect.width) // 2
    center_y = (HEIGHT - drawn_area_rect.height) // 2

    # Paste the drawn content onto the new screen at the center position
    new_screen.blit(drawn_surface, (center_x, center_y))

    pygame.display.flip()

    # Save the centered image
    pygame.image.save(new_screen, "CenteredDrawing.jpg")
    im = Image.open('CenteredDrawing.jpg')
    im_gray = im.convert('L')
    im_resized = im_gray.resize((28, 28))
    im_rotated = im_resized.rotate(270)
    im_rotated.save('CenteredDrawinggray.jpg')

    # Wait for a key press before quitting
    wait_for_key()

    # Quit Pygame
    pygame.quit()

def get_drawn_area_rect(screen):
    # Get all non-black pixels in the screen
    non_black_pixels = [(x, y) for x in range(screen.get_width()) for y in range(screen.get_height())
                        if screen.get_at((x, y)) != (0, 0, 0, 255)]

    if non_black_pixels:
        # Calculate the bounding box of the drawn area
        min_x = min(x for x, y in non_black_pixels)
        max_x = max(x for x, y in non_black_pixels)
        min_y = min(y for x, y in non_black_pixels)
        max_y = max(y for x, y in non_black_pixels)

        return pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
    else:
        # If there are no non-black pixels, return an empty rectangle
        return pygame.Rect(0, 0, 0, 0)

def wait_for_key():
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                waiting = False
            elif event.type == pygame.KEYDOWN:
                waiting = False

