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

from __future__ import annotations

from typing import Any

import numpy as np
import pygame

from src.visualization.colors import cluster_color


class Renderer:
    def __init__(self, width: int = 1200, height: int = 800) -> None:

        pygame.init()

        self.width: int = width
        self.height: int = height

        self.screen: pygame.Surface = pygame.display.set_mode((width, height))

        pygame.display.set_caption("Semantic Turing Field")

        self.clock: pygame.time.Clock = pygame.time.Clock()

        self.font: pygame.font.Font = pygame.font.SysFont("Arial", 16)

        self.background: tuple[int, int, int] = (18, 18, 18)

        self.particle_radius: int = 4

    def draw(
        self,
        camera: Any,
        positions: np.ndarray,
        clusters: np.ndarray,
        labels: list[str],
        words: list[str],
        simulation: Any,
        paused: bool,
        input_handler: Any,
    ) -> None:

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

        self.draw_sentence_box(input_handler)
        
        pygame.display.flip()

        self.clock.tick(60)

    ### Particle Drawing
    def draw_particles(
        self,
        camera: Any,
        positions: np.ndarray,
        clusters: np.ndarray,
    ) -> None:

        for i, pos in enumerate(positions):
            x, y = camera.world_to_screen(pos)

            if not np.isfinite(x) or not np.isfinite(y):
                print(f"Invalid particle {i}: ({x}, {y})")
                print(pos)
                raise RuntimeError("Particle position became invalid.")

            pygame.draw.circle(
                self.screen,
                cluster_color(clusters[i]),
                (x, y),
                self.particle_radius,
            )

    ### Hover Labels
    def draw_hover_label(
        self,
        camera: Any,
        positions: np.ndarray,
        words: list[str],
    ) -> None:

        mouse: tuple[int, int] = pygame.mouse.get_pos()

        hovered: int | None = None

        for i, pos in enumerate(positions):
            sx, sy = camera.world_to_screen(pos)

            dx = mouse[0] - sx
            dy = mouse[1] - sy

            if dx * dx + dy * dy < 100:
                hovered = i
                break

        if hovered is None:
            return

        label: pygame.Surface = self.font.render(
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
        simulation: Any,
        paused: bool,
    ) -> None:

        status: str = "Paused" if paused else "Running"

        ui: list[str] = [
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
            "F5 : Save State",
            "F9 : Load State",
        ]

        y: int = 10

        for line in ui:
            surface: pygame.Surface = self.font.render(
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

    def draw_sentence_box(self, input_handler):
        y = self.height - 45

        pygame.draw.rect(
            self.screen,
            (40,40,40),
            (0, y, self.width, 45),
        )

        if input_handler.typing:
            text = "> " + input_handler.text + "_"

        else:
            text = "Press ENTER to type a sentence"

        surface = self.font.render(
            text,
            True,
            (255,255,255),
        )

        self.screen.blit(
            surface,
            (10, y + 12),
        )
