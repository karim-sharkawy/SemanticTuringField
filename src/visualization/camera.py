import pygame


class Camera:

    def __init__(self):

        self.zoom = 80

        self.offset = pygame.Vector2(600, 400)

    def world_to_screen(self, pos):

        return (
            pos[0] * self.zoom + self.offset.x,
            pos[1] * self.zoom + self.offset.y,
        )

    def screen_to_world(self, pos):

        return (
            (pos[0] - self.offset.x) / self.zoom,
            (pos[1] - self.offset.y) / self.zoom,
        )