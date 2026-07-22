"""
Renderer

Responsible for drawing the Semantic Turing Field.

Responsibilities
----------------
- Draw particles
- Draw hover labels
- Draw UI text
- Maintain 60 FPS
"""

import numpy as np
import pygame

from src.visualization.colors import cluster_color


class Renderer:

    def __init__(self, width=1200, height=800):

        pygame.init()

        self.width = width
        self.height = height

        self.screen = pygame.display.set_mode((width, height))

        pygame.display.set_caption("Semantic Turing Field")

        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Arial", 16)

        self.background = (18, 18, 18)

        self.particle_radius = 4

    def draw(
        self,
        camera,
        positions,
        clusters,
        labels,
        words,
        simulation,
        paused,
    ):

        self.screen.fill(self.background)

        self.draw_particles(
            camera,
            positions,
            clusters,
        )

        self.draw_hover_label(
            camera,
            positions,
            words,
        )

        self.draw_ui(
            simulation,
            paused,
        )

        pygame.display.flip()

        self.clock.tick(60)

    ### Particle Drawing
    def draw_particles(
        self,
        camera,
        positions,
        clusters,
    ):

        for i, pos in enumerate(positions):

            x, y = camera.world_to_screen(pos)

            if not np.isfinite(x) or not np.isfinite(y):
                print(f"Invalid particle {i}: ({x}, {y})")
                print(pos)
                raise RuntimeError("Particle position became invalid.")

            pygame.draw.circle(
                self.screen,
                cluster_color(clusters[i]),
                #(int(x), int(y)),
                (x, y),
                #(int(round(x)), int(round(y))),
                self.particle_radius,
            )

    ### Hover Labels
    def draw_hover_label(
        self,
        camera,
        positions,
        words,
    ):

        mouse = pygame.mouse.get_pos()

        hovered = None

        for i, pos in enumerate(positions):

            sx, sy = camera.world_to_screen(pos)

            dx = mouse[0] - sx
            dy = mouse[1] - sy

            if dx * dx + dy * dy < 100:

                hovered = i
                break

        if hovered is None:
            return

        label = self.font.render(
            words[hovered],
            True,
            (255, 255, 255),
        )

        self.screen.blit(
            label,
            (
                mouse[0] + 12,
                mouse[1] + 12,
            ),
        )

    ### UI
    def draw_ui(
        self,
        simulation,
        paused,
    ):

        status = "Paused" if paused else "Running"

        ui = [

            f"Status: {status}",

            f"Alpha: {simulation.alpha:.2f}",

            f"Beta: {simulation.beta:.2f}",

            f"Damping: {simulation.damping:.3f}",

            f"dt: {simulation.dt:.3f}",

            "",

            "Q/A : Alpha",

            "W/S : Beta",

            "E/D : Damping",

            "T/G : dt",

            "R : Reset",

            "SPACE : Pause",

            "RIGHT : Step",

            "Mouse Wheel : Zoom",

            "Drag : Pan",

        ]

        y = 10

        for line in ui:

            surface = self.font.render(
                line,
                True,
                (230, 230, 230),
            )

            self.screen.blit(
                surface,
                (
                    10,
                    y,
                ),
            )

            y += 22