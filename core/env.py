import math

import pygame

from core.config import Action, Colors, Physics, Screen
from core.drone import Drone
from core.expert import Expert


class Environment:
    def __init__(self, useExpert=False, render=True):
        # Render flag to open/close PyGame rendering
        self.render = render

        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((Screen.WIDTH, Screen.HEIGHT))
            pygame.display.set_caption("2D Drone Positioning Simulator")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("JetBrains Mono", 12)

        # Start drone in the middle
        self.drone = Drone(Screen.WIDTH / 2, Screen.HEIGHT / 2)
        # Expert for auto-piloting & data collection
        self.expert = Expert() if useExpert else None

    def _enforceBoundaries(self):
        """Keeps drone on screen and stops velocity on impact"""
        x, y, vx, vy, theta, omega = self.drone.state
        margin = Screen.MARGIN

        if y > Screen.HEIGHT - margin:
            self.drone.state[1] = Screen.HEIGHT - margin
            self.drone.state[3] = min(0, vy)
            # Dampen the rotation on hard ground impact
            self.drone.state[5] *= 0.5

        if y < margin:
            self.drone.state[1] = margin
            self.drone.state[3] = max(0, vy)

        if x < margin:
            self.drone.state[0] = margin
            self.drone.state[2] = max(0, vx)

        if x > Screen.WIDTH - margin:
            self.drone.state[0] = Screen.WIDTH - margin
            self.drone.state[2] = min(0, vx)

    def runUserControl(self):
        isRunning: bool = True

        while isRunning:
            # At start, ~82% power on both (hover)
            action = Action(0.82, 0.82)

            if self.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        isRunning = False

            # For expert control
            if self.expert:
                action = self.expert.getAction(self.drone.state, Physics.DT)

            else:
                keys = pygame.key.get_pressed()

                # Map keys directly to named attributes
                if keys[pygame.K_w]:
                    action.LEFT = 1.0
                if keys[pygame.K_s]:
                    action.LEFT = 0.0
                if keys[pygame.K_UP]:
                    action.RIGHT = 1.0
                if keys[pygame.K_DOWN]:
                    action.RIGHT = 0.0

            # Physics Step
            self.drone.step(action)
            self._enforceBoundaries()

            # Render
            if self.render:
                self.screen.fill(Colors.BLACK)
                pygame.draw.rect(
                    self.screen, Colors.GRAY, (0, 0, Screen.WIDTH, Screen.HEIGHT), 2
                )
                self._drawDrone()
                self._drawHUD()

                pygame.display.flip()
                self.clock.tick(Physics.FPS)

        if self.render:
            pygame.quit()

    def _calculateRotorColors(self):
        # Check for lateral input type
        if self.drone.action.RIGHT > 0.1:
            return Colors.BLUE, Colors.BLUE
        elif self.drone.action.LEFT > 0.1:
            return Colors.YELLOW, Colors.YELLOW

        else:
            return Colors.GRAY, Colors.GRAY

    # -------------------------------- RENDERING --------------------------------
    def _drawDroneBody(self, cx, cy, theta):
        """Draws the physical frame, rotors, and center of mass."""
        # Calculate rotor endpoints
        leftX = cx - int(Physics.ARM_LENGTH * math.cos(theta))
        leftY = cy - int(Physics.ARM_LENGTH * math.sin(theta))
        rightX = cx + int(Physics.ARM_LENGTH * math.cos(theta))
        rightY = cy + int(Physics.ARM_LENGTH * math.sin(theta))

        # Draw Frame
        pygame.draw.line(self.screen, Colors.WHITE, (leftX, leftY), (rightX, rightY), 4)

        # Fetch calculated colors and draw rotors
        colorLeft, colorRight = self._calculateRotorColors()
        pygame.draw.circle(self.screen, colorLeft, (leftX, leftY), Physics.ROTOR_RADIUS)
        pygame.draw.circle(
            self.screen, colorRight, (rightX, rightY), Physics.ROTOR_RADIUS
        )

        # Draw Center of Mass
        pygame.draw.circle(self.screen, Colors.WHITE, (cx, cy), 4)

    def _drawDrone(self):
        """Main orchestrator for drone rendering."""
        x, y, vx, vy, theta, omega = self.drone.state
        cx, cy = int(x), int(y)

        # Draw layers in correct order (Body first, Vector on top)
        self._drawDroneBody(cx, cy, theta)

        if self.expert:
            targetX, targetY = self.expert.waypoints[self.expert.currWaypointIdx]
            pygame.draw.circle(
                self.screen, Colors.GREEN, (int(targetX), int(targetY)), 6
            )
            pygame.draw.line(
                self.screen, Colors.GREEN, (cx, cy), (int(targetX), int(targetY)), 1
            )

    def _drawHUD(self):
        x, y, vx, vy, theta, omega = self.drone.state
        texts = [
            f"Position ::: ({x:.1f}, {y:.1f})",
            f"Velocity ::: ({vx:.1f}, {vy:.1f})",
            f"Tilt ::: {math.degrees(theta):.1f} deg",
            f"Action ::: Left: {self.drone.action.LEFT:.2f} | Right: {self.drone.action.RIGHT:.2f}",
            "",
            "Controls: W/S (Up/Down), A/D (Left/Right)",
        ]
        for i, text in enumerate(texts):
            surface = self.font.render(text, True, Colors.WHITE)
            self.screen.blit(surface, (10, 10 + i * 20))


if __name__ == "__main__":
    env = Environment(useExpert=False)
    env.runUserControl()
