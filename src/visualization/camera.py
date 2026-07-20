import pygame


class Camera:

    def __init__(self):

        self.zoom = 80

        self.offset = pygame.Vector2(600, 400)

        self.dragging = False

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
    
    def handle_event(self, event):

        if event.type == pygame.MOUSEBUTTONDOWN:

            if event.button == 1:

                self.dragging = True

        elif event.type == pygame.MOUSEBUTTONUP:

            if event.button == 1:

                self.dragging = False

        elif event.type == pygame.MOUSEMOTION:

            if self.dragging:

                self.offset += pygame.Vector2(event.rel)

        elif event.type == pygame.MOUSEWHEEL:

            if event.y > 0:

                self.zoom *= 1.1

            else:

                self.zoom /= 1.1

            self.zoom = max(10, min(self.zoom, 400))