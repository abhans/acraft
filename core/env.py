import math
import os
import random

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from core.config import WAYPOINTS, Action, Colors, Physics, Screen, Wind
from core.drone import Drone
from core.expert import Expert
from core.logger import flogger


class Environment(gym.Env):
    def __init__(self, useExpert=False, render=True, stepsMax=500):
        super(Environment, self).__init__()
        # Render flag to open/close PyGame rendering
        self.render = render
        self.stepsMax = stepsMax
        self.currStep = 0

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

        # Waypoint tracking for reward computation
        self.waypoints: list[tuple[int, ...]] = WAYPOINTS
        self.currWaypointIdx: int = 0
        self.waypointThresh: float = 10.0  # In Pixels
        self.waypointWinds: dict = {}  # Store wind conditions per waypoint

        # The Wind Effect
        self.windDirection: float = 0.0
        self.targetWindDirection: float = 0.0

        self.windSpeed: float = Wind.SPEED
        self.targetWindSpeed: float = Wind.SPEED

        self.time: float = 0.0
        self.hitBoundary: bool = False

        # Actions and Spaces
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

    def _normalizeState(self, state, target) -> np.ndarray:
        x, y, vx, vy, theta, omega, windFx, windFy = state
        targetX, targetY = target

        # Calculate relative position to the target
        dx = targetX - x
        dy = targetY - y

        normDx = np.clip(dx / Screen.WIDTH, -1.0, 1.0)
        normDy = np.clip(dy / Screen.HEIGHT, -1.0, 1.0)

        normVx = np.clip(vx / 1000.0, -1.0, 1.0)
        normVy = np.clip(vy / 1000.0, -1.0, 1.0)
        normTheta = np.clip(theta / math.pi, -1.0, 1.0)
        # Omega is normalized by an estimated max angular velocity (5 rad/s)
        normOmega = np.clip(omega / 5.0, -1.0, 1.0)

        # Normalize wind forces (assuming max ~200 pixels/s^2)
        normWindFx = np.clip(windFx / 300.0, -1.0, 1.0)
        normWindFy = np.clip(windFy / 300.0, -1.0, 1.0)

        return np.array(
            [
                normDx,
                normDy,
                normVx,
                normVy,
                normTheta,
                normOmega,
                normWindFx,
                normWindFy,
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        # Seed for Gymnasium environment
        super().reset(seed=seed)

        # Generate random wind conditions for each waypoint
        self.waypointWinds = {}
        for i in range(len(self.waypoints)):
            direction = Wind.DIRECTION + random.uniform(-math.pi / 4, math.pi / 4)
            speed = Wind.SPEED + random.uniform(-50, 50)
            self.waypointWinds[i] = (direction, speed)

        # Set initial wind from first waypoint
        self.currWaypointIdx = 0
        self.targetWindDirection, self.targetWindSpeed = self.waypointWinds[0]
        
        self.windDirection = self.targetWindDirection
        self.windSpeed = self.targetWindSpeed

        isRandomStart: bool = False

        if options is not None:
            isRandomStart = options.get("randomStart", False)
        elif self.expert:
            isRandomStart = True

        if not isRandomStart:
            self.drone.state = np.array(
                [Screen.WIDTH / 2, Screen.HEIGHT / 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )

        else:
            # ---- RANODM INITIALIZATION ----
            #   (Same with the DataCollector)
            initX = random.uniform(
                Screen.MARGIN + 50, Screen.WIDTH - Screen.MARGIN - 50
            )
            initY = random.uniform(
                Screen.MARGIN + 50, Screen.HEIGHT - Screen.MARGIN - 50
            )
            # Pick a random initial tilt and velocity
            initVx = random.uniform(-200, 200)
            initVy = random.uniform(-200, 200)
            initTheta = random.uniform(-0.3, 0.3)  # Up to ~17 degrees off axis

            self.drone.state = np.array(
                [initX, initY, initVx, initVy, initTheta, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )

        if self.expert:
            self.expert.currWaypointIdx = 0
        else:
            # Initialize waypoint tracking for RL agent
            self.currWaypointIdx = 0

        self.currStep = 0
        target = self.waypoints[self.currWaypointIdx]
        normState = self._normalizeState(self.drone.state, target)

        # Reset turbulence timer
        self.time = 0.0
        # Reset boundary hit flag
        self.hitBoundary = False
        
        return normState, {}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.currStep += 1
        self.time += Physics.DT
        flogger.info(f"[STEP] Step {self.currStep} | Action: {action}")

        # ----------------------- SMOOTH WIND TRANSITION -----------------------
        # Slowly blend current wind towards the target wind (5% per frame)
        self.windDirection += (self.targetWindDirection - self.windDirection) * 0.05
        self.windSpeed += (self.targetWindSpeed - self.windSpeed) * 0.05

        if isinstance(action, np.ndarray):
            action = Action(*action)

        # ------------------------- WIND CALCULATION ------------------------
        # Drift the global wind direction slowly over time
        self.windDirection += (
            Wind.DRIFT_SPEED * math.cos(self.windDirection)
        ) * Physics.DT
        self.windDirection += (
            Wind.DRIFT_DIR_SPEED * math.sin(self.windDirection * 0.7)
        ) * Physics.DT

        # Get localized turbulance
        X, Y = self.drone.state[:2]

        gustX = (
            math.sin(X * 0.01 + self.time * 0.5) * Wind.GUST_STRENGTH
            + math.cos(Y * 0.015 + self.time * 0.3) * Wind.GUST_STRENGTH
        )
        gustY = (
            math.cos(Y * 0.02 + self.time * 0.4) * Wind.GUST_STRENGTH
            - math.sin(X * 0.008 - self.time * 0.2) * Wind.GUST_STRENGTH
        )

        # Combine base wind with gusts to get total wind force
        effWindX = self.windSpeed * math.cos(self.windDirection) + gustX
        effWindY = self.windSpeed * math.sin(self.windDirection) + gustY
        effectiveWind = (effWindX, effWindY)
        flogger.info(f"[WIND] Effective wind: {effectiveWind}")

        # Calculate distance to current waypoint
        prevX, prevY = self.drone.state[:2]
        targetX, targetY = self.waypoints[self.currWaypointIdx]
        prevDistance = np.hypot(prevX - targetX, prevY - targetY)

        # -------------------------- DRONE PHYSICS STEP ------------------------
        self.drone.step(action, wind=effectiveWind)

        # ---------------------- WAYPOINT & REWARD LOGIC ----------------------
        x, y, vx, vy, theta, omega = self.drone.state[:6]
        outOfBounds = (
            x < 0 or x > Screen.WIDTH or
            y < 0 or y > Screen.HEIGHT
        )
        distance = np.hypot(x - targetX, y - targetY)
        
        # ------------------------ REWARD STRUCTURE ----------------------
        # Distance reward: Closer to the target is better
        reward = (prevDistance - distance) * 0.1

        # Orientation reward: Encourage level flight (theta near 0)
        reward -= abs(theta) * 0.02
        reward -= abs(omega) * 0.05

        # Boundary penalty: Discourage hitting the walls
        if self.hitBoundary:
            reward -= 5.0
            self.hitBoundary = False  # Reset for next step

        # Check if waypoint reached
        waypointReached = False
        if distance < self.waypointThresh:
            # Bonus for reaching waypoint
            reward += 20.0  
            waypointReached = True
            # Move to next waypoint (cyclic)
            self.currWaypointIdx = (self.currWaypointIdx + 1) % len(self.waypoints)
            # Switch wind condition to new waypoint
            self.targetWindDirection, self.targetWindSpeed = self.waypointWinds[self.currWaypointIdx]

        target = self.waypoints[self.currWaypointIdx]
        flogger.info(f"[STATE] Position: ({x:.1f}, {y:.1f}) | Velocity: ({vx:.1f}, {vy:.1f}) | Distance to Target: {distance:.1f} | Reward: {reward:.2f} | Waypoint Reached: {waypointReached}")
        normState = self._normalizeState(self.drone.state, target)
        
        speed = np.hypot(vx, vy)
        terminated = (
            abs(theta) > np.radians(75)
            or speed > 900
            or outOfBounds
        )

        # Heavy penalty for crashing
        if terminated:
            reward -= 15.0

        isDone = self.currStep >= self.stepsMax

        flogger.info(f"[REWARD] Reward: {reward:.2f} | Terminated: {terminated} | Done: {isDone}\n")

        return (
            normState,
            reward,
            terminated,
            isDone,
            {"rawState": self.drone.state, "waypointReached": waypointReached},
        )

    def getExpertAction(self) -> np.ndarray:
        if self.expert:
            return self.expert.getAction(self.drone.state, Physics.DT).toNumpy()
        else:
            raise ValueError("Environment not initialized with an expert controller.")

    def _enforceBoundaries(self):
        """Keeps drone on screen and stops velocity on impact"""
        x, y, vx, vy, theta, omega, _, _ = self.drone.state
        margin = Screen.MARGIN

        if y > Screen.HEIGHT - margin:
            self.drone.state[1] = Screen.HEIGHT - margin
            self.drone.state[3] = min(0, vy)
            # Dampen the rotation on hard ground impact
            self.drone.state[5] *= 0.5
            self.hitBoundary = True

        if y < margin:
            self.drone.state[1] = margin
            self.drone.state[3] = max(0, vy)
            self.hitBoundary = True

        if x < margin:
            self.drone.state[0] = margin
            self.drone.state[2] = max(0, vx)
            self.hitBoundary = True

        if x > Screen.WIDTH - margin:
            self.drone.state[0] = Screen.WIDTH - margin
            self.drone.state[2] = min(0, vx)
            self.hitBoundary = True

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
                action = self.getExpertAction()

            else:
                keys = pygame.key.get_pressed()

                # Map keys directly to named attributes
                if keys[pygame.K_w]:
                    action.LEFT = min(1.0, action.LEFT + 0.02)
                if keys[pygame.K_s]:
                    action.LEFT = max(0.0, action.LEFT - 0.02)
                if keys[pygame.K_UP]:
                    action.RIGHT = max(0.0, action.RIGHT + 0.02)
                if keys[pygame.K_DOWN]:
                    action.RIGHT = max(0.0, action.RIGHT - 0.02)

            # Physics step
            _, _, terminated, truncated, _ = self.step(action)
            done = terminated or truncated

            if done:
                self.reset(options={"randomStart": True})

            # Render
            if self.render:
                self.screen.fill((Colors.BLACK.R, Colors.BLACK.G, Colors.BLACK.B))
                pygame.draw.rect(
                    self.screen,
                    (Colors.GRAY.R, Colors.GRAY.G, Colors.GRAY.B),
                    (0, 0, Screen.WIDTH, Screen.HEIGHT),
                    2,
                )
                self._drawDrone()
                self._drawHUD()

                pygame.display.flip()
                self.clock.tick(Physics.FPS)

        if self.render:
            pygame.quit()
            os.system("clear")

    def _thrustToColor(self, thrust):
        """
        Calculate color based on thrust gradient.
        """

        def lerp(a, b, t):
            return a + (b - a) * t

        if thrust <= 0.4:
            factor = thrust / 0.4
            return (
                int(lerp(Colors.BLUE.R, Colors.YELLOW.R, factor)),
                int(lerp(Colors.BLUE.G, Colors.YELLOW.G, factor)),
                int(lerp(Colors.BLUE.B, Colors.YELLOW.B, factor)),
            )
        elif thrust <= 0.8:
            factor = (thrust - 0.4) / 0.4
            return (
                int(lerp(Colors.YELLOW.R, Colors.ORANGE.R, factor)),
                int(lerp(Colors.YELLOW.G, Colors.ORANGE.G, factor)),
                int(lerp(Colors.YELLOW.B, Colors.ORANGE.B, factor)),
            )
        else:
            factor = (thrust - 0.8) / 0.2
            return (
                int(lerp(Colors.ORANGE.R, Colors.RED.R, factor)),
                int(lerp(Colors.ORANGE.G, Colors.RED.G, factor)),
                int(lerp(Colors.ORANGE.B, Colors.RED.B, factor)),
            )

    def _calculateRotorColors(self):
        """
        Calculate rotor colors based on thrust gradient.
        """
        lThrust = self.drone.action.LEFT
        rThrust = self.drone.action.RIGHT
        return self._thrustToColor(lThrust), self._thrustToColor(rThrust)

    # -------------------------------- RENDERING --------------------------------
    def _drawDroneBody(self, cx, cy, theta):
        """
        Draws the physical frame, rotors, and center of mass.
        """
        # Calculate rotor endpoints
        leftX = cx - int(Physics.ARM_LENGTH * math.cos(theta))
        leftY = cy - int(Physics.ARM_LENGTH * math.sin(theta))
        rightX = cx + int(Physics.ARM_LENGTH * math.cos(theta))
        rightY = cy + int(Physics.ARM_LENGTH * math.sin(theta))

        # Draw Frame
        pygame.draw.line(
            self.screen,
            (Colors.WHITE.R, Colors.WHITE.G, Colors.WHITE.B),
            (leftX, leftY),
            (rightX, rightY),
            4,
        )

        # Fetch calculated colors and draw rotors
        colorLeft, colorRight = self._calculateRotorColors()
        pygame.draw.circle(self.screen, colorLeft, (leftX, leftY), Physics.ROTOR_RADIUS)
        pygame.draw.circle(
            self.screen, colorRight, (rightX, rightY), Physics.ROTOR_RADIUS
        )

        # Draw Center of Mass
        pygame.draw.circle(
            self.screen, (Colors.WHITE.R, Colors.WHITE.G, Colors.WHITE.B), (cx, cy), 4
        )

    def _drawDrone(self):
        """
        Main orchestrator for drone rendering.
        """
        x, y, vx, vy, theta, omega = self.drone.state[:6]
        cx, cy = int(x), int(y)

        # Draw layers in correct order (Body first, Vector on top)
        self._drawDroneBody(cx, cy, theta)

        if self.expert:
            targetX, targetY = self.expert.waypoints[self.expert.currWaypointIdx]
            pygame.draw.circle(
                self.screen,
                (Colors.GREEN.R, Colors.GREEN.G, Colors.GREEN.B),
                (int(targetX), int(targetY)),
                6,
            )
            pygame.draw.line(
                self.screen,
                (Colors.GREEN.R, Colors.GREEN.G, Colors.GREEN.B),
                (cx, cy),
                (int(targetX), int(targetY)),
                1,
            )

    def _drawHUD(self):
        x, y, vx, vy, theta, omega, windX, windY = self.drone.state
        texts = [
            f"Position ::: ({x:.1f}, {y:.1f})",
            f"Velocity ::: ({vx:.1f}, {vy:.1f})",
            f"Tilt ::: {math.degrees(theta):.1f} deg",
            f"Action ::: Left: {self.drone.action.LEFT:.2f} | Right: {self.drone.action.RIGHT:.2f}",
            "",
        ]
        for i, text in enumerate(texts):
            surface = self.font.render(
                text, True, (Colors.WHITE.R, Colors.WHITE.G, Colors.WHITE.B)
            )
            self.screen.blit(surface, (10, 10 + i * 20))

        # -------------------------------- WIND VISUALIZATION --------------------------------
        # Calculate arrow properties
        lenArrows = 50
        angle = math.atan2(windY, windX)

        # Arrow base (center of screen)
        baseX, baseY = Screen.WIDTH // 2, Screen.HEIGHT // 2

        # Arrow tip (displaced from center)
        tipX = baseX + lenArrows * math.cos(angle)
        tipY = baseY - lenArrows * math.sin(angle)

        # Draw the arrow shaft
        pygame.draw.line(
            self.screen,
            (Colors.GRAY.R, Colors.GRAY.G, Colors.GRAY.B),
            (baseX, baseY),
            (tipX, tipY),
            3,
        )

        # Draw arrowhead
        lAngle = angle + math.pi * 0.85
        rAngle = angle - math.pi * 0.85
        lenHead = 10

        leftX = tipX + lenHead * math.cos(lAngle)
        leftY = tipY - lenHead * math.sin(lAngle)
        rightX = tipX + lenHead * math.cos(rAngle)
        rightY = tipY - lenHead * math.sin(rAngle)

        pygame.draw.polygon(
            self.screen,
            (Colors.GRAY.R, Colors.GRAY.G, Colors.GRAY.B),
            [(tipX, tipY), (leftX, leftY), (rightX, rightY)],
        )

        self._visualizeThrust()

        # Render controls after thrust bars
        textControls = "Controls: W/S (Up/Down), A/D (Left/Right)"
        sfControls = self.font.render(
            textControls, True, (Colors.GRAY.R, Colors.GRAY.G, Colors.GRAY.B)
        )
        self.screen.blit(sfControls, (10, 10 + len(texts) * 20 + 60))

    def _visualizeThrust(self):
        """
        Visualize the thrust levels of both rotors with colored bars.
        """
        barLength = 20
        lThrust = self.drone.action.LEFT
        rThrust = self.drone.action.RIGHT

        lColor = self._thrustToColor(lThrust)
        rColor = self._thrustToColor(rThrust)

        # Calculate label widths
        lLabelText = "Left Rotor Thrust: "
        rLabelText = "Right Rotor Thrust: "
        lLabel = self.font.render(
            lLabelText, True, (Colors.WHITE.R, Colors.WHITE.G, Colors.WHITE.B)
        )
        rLabel = self.font.render(
            rLabelText, True, (Colors.WHITE.R, Colors.WHITE.G, Colors.WHITE.B)
        )

        # Position bars
        barStartX = 10 + max(lLabel.get_width(), rLabel.get_width()) + 10
        barY1 = 10 + 5 * 20
        barY2 = 10 + 6 * 20

        # Render labels
        self.screen.blit(lLabel, (10, barY1))
        self.screen.blit(rLabel, (10, barY2))

        # ---------------- RENDER: Left Rotor Bar ----------------
        filledCount = int(lThrust * barLength)
        emptyCount = barLength - filledCount

        bracketLeft = self.font.render(
            "[", True, (Colors.WHITE.R, Colors.WHITE.G, Colors.WHITE.B)
        )
        filledBar = self.font.render("|" * filledCount, True, lColor)
        emptyBar = self.font.render(
            "|" * emptyCount, True, (Colors.WHITE.R, Colors.WHITE.G, Colors.WHITE.B)
        )
        bracketRight = self.font.render(
            "]", True, (Colors.WHITE.R, Colors.WHITE.G, Colors.WHITE.B)
        )

        currentX = barStartX
        self.screen.blit(bracketLeft, (currentX, barY1))
        currentX += bracketLeft.get_width()
        self.screen.blit(filledBar, (currentX, barY1))
        currentX += filledBar.get_width()
        self.screen.blit(emptyBar, (currentX, barY1))
        currentX += emptyBar.get_width()
        self.screen.blit(bracketRight, (currentX, barY1))

        # ---------------- RENDER: Right Rotor Bar ----------------
        filledCount = int(rThrust * barLength)
        emptyCount = barLength - filledCount

        bracketLeft = self.font.render(
            "[", True, (Colors.WHITE.R, Colors.WHITE.G, Colors.WHITE.B)
        )
        filledBar = self.font.render("|" * filledCount, True, rColor)
        emptyBar = self.font.render(
            "|" * emptyCount, True, (Colors.WHITE.R, Colors.WHITE.G, Colors.WHITE.B)
        )
        bracketRight = self.font.render(
            "]", True, (Colors.WHITE.R, Colors.WHITE.G, Colors.WHITE.B)
        )

        currentX = barStartX
        self.screen.blit(bracketLeft, (currentX, barY2))
        currentX += bracketLeft.get_width()
        self.screen.blit(filledBar, (currentX, barY2))
        currentX += filledBar.get_width()
        self.screen.blit(emptyBar, (currentX, barY2))
        currentX += emptyBar.get_width()
        self.screen.blit(bracketRight, (currentX, barY2))


if __name__ == "__main__":
    env = Environment(useExpert=False)
    env.runUserControl()
