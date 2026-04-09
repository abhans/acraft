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
        self.font = pygame.font.SysFont("JetBrains Mono", 16)

        # Start drone in the middle
        self.drone = Drone2D(Screen.WIDTH / 2, Screen.HEIGHT / 2)

    def _drawDrone(self):
        x, y, vx, vy, theta, omega = self.drone.state
        cx, cy = int(x), int(y)

        # Calculate rotor endpoints
        leftX = cx - int(Physics.ARM_LENGTH * math.cos(theta))
        leftY = cy - int(Physics.ARM_LENGTH * math.sin(theta))
        rightX = cx + int(Physics.ARM_LENGTH * math.cos(theta))
        rightY = cy + int(Physics.ARM_LENGTH * math.sin(theta))

        # Draw Body
        pygame.draw.line(self.screen, Colors.WHITE, (leftX, leftY), (rightX, rightY), 4)

        # Determine rotor color based on thrust (Green for UP, Red for DOWN)
        lThrust = self.drone.action.UP - self.drone.action.DOWN
        rThrust = self.drone.action.UP - self.drone.action.DOWN
        colorLeft = (
            (0, int(255 * lThrust), 0) if lThrust > 0 else (int(-255 * lThrust), 0, 0)
        )
        colorRight = (
            (0, int(255 * rThrust), 0) if rThrust > 0 else (int(-255 * rThrust), 0, 0)
        )

        # Draw Rotors
        pygame.draw.circle(self.screen, colorLeft, (leftX, leftY), Physics.ROTOR_RADIUS)
        pygame.draw.circle(
            self.screen, colorRight, (rightX, rightY), Physics.ROTOR_RADIUS
        )

        # Draw Center of Mass
        pygame.draw.circle(self.screen, Colors.GRAY, (cx, cy), 4)

    def _drawHUD(self):
        x, y, vx, vy, theta, omega = self.drone.state
        texts = [
            f"Pos: ({x:.1f}, {y:.1f})",
            f"Vel: ({vx:.1f}, {vy:.1f})",
            f"Tilt: {math.degrees(theta):.1f} deg",
            f"Action (V, Lat): ({self.drone.action.UP - self.drone.action.DOWN:.1f}, {self.drone.action.RIGHT - self.drone.action.LEFT:.1f})",  # Updated
            "",
            "Controls: W/S (Up/Down), A/D (Left/Right)",
        ]
        for i, text in enumerate(texts):
            surface = self.font.render(text, True, Colors.WHITE)
            self.screen.blit(surface, (10, 10 + i * 20))

    def _enforceBoundaries(self):
        """Keeps drone on screen and stops velocity on impact"""
        x, y, vx, vy, theta, omega = self.drone.state
        Screen.MARGIN = 20

        if y > Screen.HEIGHT - Screen.MARGIN:
            self.drone.state[1] = Screen.HEIGHT - Screen.MARGIN
            self.drone.state[3] = min(0, vy)  # Stop downward momentum
        if y < Screen.MARGIN:
            self.drone.state[1] = Screen.MARGIN
            self.drone.state[3] = max(0, vy)  # Stop upward momentum
        if x < Screen.MARGIN:
            self.drone.state[0] = Screen.MARGIN
            self.drone.state[2] = max(0, vx)
        if x > Screen.WIDTH - Screen.MARGIN:
            self.drone.state[0] = Screen.WIDTH - Screen.MARGIN
            self.drone.state[2] = min(0, vx)

    def runUserControl(self):
        running = True
        while running:
            # Action format: [vertical, lateral] -> defaults to [0,0] which is a stable hover
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
