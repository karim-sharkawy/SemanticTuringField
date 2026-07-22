"""
Input Handler

Processes all keyboard and mouse input for the Semantic
Turing Field application.

Responsibilities
----------------
- Camera controls
- Keyboard shortcuts
- Pause / Resume
- Single-step mode
- Reset simulation
"""

import pygame


class InputHandler:

    def __init__(self, camera):

        self.camera = camera

        # Simulation state
        self.paused = False
        self.single_step = False

        # Gravity-wave sentence
        self.current_sentence = None

        # Future feature
        self.gravity_steps_remaining = 0

    def handle_event(
        self,
        event,
        simulation,
    ):
        """
        Handle one pygame event.

        Returns
        -------
        bool
            False if application should quit.
        """

        # Quit
        if event.type == pygame.QUIT:
            return False

        # Camera always gets events first.
        self.camera.handle_event(event)

        # Keyboard
        if event.type == pygame.KEYDOWN:

            # Quit
            if event.key == pygame.K_ESCAPE:
                return False

            # Pause
            elif event.key == pygame.K_SPACE:
                self.paused = not self.paused

            # Single step
            elif event.key == pygame.K_RIGHT:
                self.single_step = True

            # Reset
            elif event.key == pygame.K_r:
                simulation.reset()

            # Alpha
            elif event.key == pygame.K_q:
                simulation.alpha += 0.05

            elif event.key == pygame.K_a:
                simulation.alpha = max(
                    0.0,
                    simulation.alpha - 0.05,
                )

            # Beta
            elif event.key == pygame.K_w:
                simulation.beta += 0.05

            elif event.key == pygame.K_s:
                simulation.beta -= 0.05

            # Damping
            elif event.key == pygame.K_e:
                simulation.damping = min(
                    0.999,
                    simulation.damping + 0.005,
                )

            elif event.key == pygame.K_d:
                simulation.damping = max(
                    0.0,
                    simulation.damping - 0.005,
                )

            # Time step
            elif event.key == pygame.K_t:
                simulation.dt += 0.005

            elif event.key == pygame.K_g:
                simulation.dt = max(
                    0.001,
                    simulation.dt - 0.005,
                )

        return True

    def should_step(self):
        """
        Determine whether the simulation
        should advance this frame.
        """

        # Normal running
        if not self.paused:
            return True

        # Single-step mode
        if self.single_step:

            self.single_step = False
            return True

        return False