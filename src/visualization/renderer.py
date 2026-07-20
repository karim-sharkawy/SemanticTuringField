import pygame

from .colors import cluster_color


class Renderer:

    def __init__(self, width=1200, height=800):

        pygame.init()

        self.screen = pygame.display.set_mode((width, height))

        pygame.display.set_caption("Semantic Turing Field")

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Arial", 14)

        self.width = width
        self.height = height

    def draw(
        self,
        camera,
        positions,
        clusters,
    ):

        self.screen.fill((18, 18, 18))

        for i, pos in enumerate(positions):

            x, y = camera.world_to_screen(pos)

            pygame.draw.circle(
                self.screen,
                cluster_color(clusters[i]),
                (int(x), int(y)),
                4,
            )

        pygame.display.flip()

        self.clock.tick(60)