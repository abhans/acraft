import math

import pygame

from core.config import Action, Colors, Physics, Screen
from core.drone import Drone2D


class DroneEnvironment:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Screen.WIDTH, Screen.HEIGHT))
        pygame.display.set_caption(
            "2D Drone Positioning Simulator - High Level Control"
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("JetBrains Mono", 12)

        # Start drone in the middle
        self.drone = Drone2D(Screen.WIDTH / 2, Screen.HEIGHT / 2)

    def _calculateRotorColors(self):
        """Determines rotor color. Lateral movement always takes priority."""
        # Check for Lateral Input
        if self.drone.action.RIGHT > 0.1:
            return Colors.BLUE, Colors.BLUE
        elif self.drone.action.LEFT > 0.1:
            return Colors.YELLOW, Colors.YELLOW

        # Fallback to Vertical Input
        elif self.drone.action.UP > 0.1:
            return Colors.GREEN, Colors.GREEN
        elif self.drone.action.DOWN > 0.1:
            return Colors.RED, Colors.RED

        else:
            return Colors.GRAY, Colors.GRAY

    def _drawLateralVector(self, cx, cy):
        """Draws a horizontal arrow indicating net lateral intent."""
        lateralIntent = self.drone.action.RIGHT - self.drone.action.LEFT

        # Only draw if there is actual lateral intent
        if abs(lateralIntent) < 0.1:
            return

        vectorLength = int(lateralIntent * 60)
        vecColor = Colors.BLUE if lateralIntent > 0 else Colors.YELLOW
        endX = cx + vectorLength

        # Draw main thrust line
        pygame.draw.line(self.screen, vecColor, (cx, cy), (endX, cy), 3)

        # Draw directoin vector
        arrowSize = 6
        if vectorLength > 0:
            pygame.draw.polygon(
                self.screen,
                vecColor,
                [
                    (endX, cy),
                    (endX - arrowSize, cy - arrowSize),
                    (endX - arrowSize, cy + arrowSize),
                ],
            )
        else:
            pygame.draw.polygon(
                self.screen,
                vecColor,
                [
                    (endX, cy),
                    (endX + arrowSize, cy - arrowSize),
                    (endX + arrowSize, cy + arrowSize),
                ],
            )

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
        self._drawLateralVector(cx, cy)

    def _drawHUD(self):
        x, y, vx, vy, theta, omega = self.drone.state
        texts = [
            f"Pos: ({x:.1f}, {y:.1f})",
            f"Vel: ({vx:.1f}, {vy:.1f})",
            f"Tilt: {math.degrees(theta):.1f} deg",
            f"Action (V, Lat): ({self.drone.action.UP - self.drone.action.DOWN:.1f}, {self.drone.action.RIGHT - self.drone.action.LEFT:.1f})",
            "",
            "Controls: W/S (Up/Down), A/D (Left/Right)",
        ]
        for i, text in enumerate(texts):
            surface = self.font.render(text, True, Colors.WHITE)
            self.screen.blit(surface, (10, 10 + i * 20))

    def _enforceBoundaries(self):
        """Keeps drone on screen and stops velocity on impact"""
        x, y, vx, vy, theta, omega = self.drone.state
        margin = Screen.MARGIN

        if y > Screen.HEIGHT - margin:
            self.drone.state[1] = Screen.HEIGHT - margin
            self.drone.state[3] = min(0, vy)
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
        running = True
        while running:
            action = Action()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()

            # Map keys directly to named attributes
            if keys[pygame.K_w]:
                action.UP = 1.0
            if keys[pygame.K_s]:
                action.DOWN = 1.0
            if keys[pygame.K_a]:
                action.LEFT = 1.0
            if keys[pygame.K_d]:
                action.RIGHT = 1.0

            # Physics Step
            self.drone.step(action)
            self._enforceBoundaries()

            # Render
            self.screen.fill(Colors.BLACK)
            pygame.draw.rect(
                self.screen, Colors.GRAY, (0, 0, Screen.WIDTH, Screen.HEIGHT), 2
            )
            self._drawDrone()
            self._drawHUD()

            pygame.display.flip()
            self.clock.tick(Physics.FPS)

        pygame.quit()


if __name__ == "__main__":
    env = DroneEnvironment()
    env.runUserControl()
