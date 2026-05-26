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
        self.render: bool = render
        self.stepsMax: int = stepsMax
        self.currStep: int = 0

        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((Screen.WIDTH, Screen.HEIGHT))
            pygame.display.set_caption("2D Drone Positioning Simulator")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("JetBrains Mono", 12)

        # Start drone in the middle
        self.drone: Drone = Drone(
            Screen.WIDTH / 2,
            Screen.HEIGHT / 2
        )
        # Expert for auto-piloting & data collection
        self.expert: Expert | None = Expert() if useExpert else None

        # Waypoint tracking for reward computation
        self.waypoints: list[tuple[int, ...]] = WAYPOINTS
        self.currWaypointIdx: int = 0
        self.waypointThresh: float = 10.0  # In Pixels
        self.waypointWinds: dict = {}  # Store wind conditions per waypoint
        self._waypointBonusGiven: bool = False
        self.cycleWaypoints: bool = True

        # -------- The Wind Effects --------
        self.windDirection: float = 0.0
        self.targetWindDirection: float = 0.0

        self.windSpeed: float = Wind.SPEED
        self.targetWindSpeed: float = Wind.SPEED

        self.time: float = 0.0

        # Actions and Spaces
        # TODO: These variables are not used anywhere! Likely that something is missing
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

    def _normalizeState(self, state) -> np.ndarray:
        x, y, vx, vy, theta, omega, windFx, windFy = state
        
        absX = (x - Screen.WIDTH / 2) / (Screen.WIDTH / 2)
        absY = (y - Screen.HEIGHT / 2) / (Screen.HEIGHT / 2) 

        normX = np.clip(absX, -1.0, 1.0)
        normY = np.clip(absY, -1.0, 1.0)

        normVx = np.clip(
            (vx / 1500.0),
            -1.0, 1.0
        )
        normVy = np.clip(
            (vy / 1500.0),
            -1.0, 1.0
        )

        normTheta = np.clip(
            (theta / math.pi),
            -1.0, 1.0
        )
        normOmega = np.clip(
            (omega / 10.0),
            -1.0, 1.0
        )

        normWindFx = np.clip(
            (windFx / 25.0),
            -1.0, 1.0
        )
        normWindFy = np.clip(
            (windFy / 25.0),
            -1.0, 1.0
        )

        return np.array(
            [
                normX,
                normY,
                normVx,
                normVy,
                normTheta,
                normOmega,
                normWindFx,
                normWindFy,
            ],
            dtype=np.float32,
        )

    def reset(self, seed: int = 42, options: dict | None = None) -> tuple[np.ndarray, dict]:
        # Seed for Gymnasium environment
        super().reset(seed=seed)

        # Generate random wind conditions for each waypoint
        self.waypointWinds = {}
        for i in range(len(self.waypoints)):
            direction = Wind.DIRECTION + random.uniform(-math.pi, math.pi)
            speed = Wind.SPEED + random.uniform(-100, 100)
            self.waypointWinds[i] = (direction, speed)

        flogger.info(f"[WIND] Waypoint Winds: {self.waypointWinds}")

        # Set initial wind from first waypoint
        self.targetWindDirection, self.targetWindSpeed = self.waypointWinds[0]
        
        self.windDirection = self.targetWindDirection
        self.windSpeed = self.targetWindSpeed

        isRandomStart: bool = False

        if options is not None:
            isRandomStart = options.get("randomStart", False)
        # If expert, start from a random coordinate
        elif self.expert:
            isRandomStart = True

        # -------- STARTING CONDITIONS --------
        if not isRandomStart:
            self.drone.state = np.array(
                [Screen.WIDTH / 2, Screen.HEIGHT / 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )

        else:
            initX = random.uniform(
                Screen.MARGIN + 100, Screen.WIDTH - Screen.MARGIN - 100
            )
            initY = random.uniform(
                Screen.MARGIN + 100, Screen.HEIGHT - Screen.MARGIN - 100
            )
            # Pick a random initial tilt and velocity
            initVx = random.uniform(0, 200)
            initVy = random.uniform(0, 200)
            initTheta = random.uniform(-0.2, 0.2)

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
        self._waypointBonusGiven = False
        normState = self._normalizeState(self.drone.state)

        # Reset turbulence timer
        self.time = 0.0
        
        return normState, {}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.currStep += 1
        self.time += Physics.DT
        flogger.info(f"[STEP] Step {self.currStep} | Action: {action}")

        # ----------------------- SMOOTH WIND TRANSITION -----------------------
        # Slowly blend current wind towards the target wind (5% per frame)
        self.windDirection += (self.targetWindDirection - self.windDirection) * 0.05
        self.windSpeed += (self.targetWindSpeed - self.windSpeed) * 0.05

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

        # Store effective wind inside drone state
        self.drone.state[6] = effWindX
        self.drone.state[7] = effWindY

        # Calculate distance to current waypoint
        targetX, targetY = self.waypoints[self.currWaypointIdx]
        prevDistance = np.hypot(X - targetX, Y - targetY)

        # -------------------------- DRONE PHYSICS STEP ------------------------
        flogger.info(f"[STATE] Drone State (t: {self.currStep}): {self.drone.state}")
        flogger.info(f"[STATE] Drone Action (t: {self.currStep}): {self.drone.action}")

        self.drone.step(action)

        # ---------------------- WAYPOINT & REWARD LOGIC ----------------------
        x, y, vx, vy, theta, omega = self.drone.state[:6]

        outOfBounds = (
            (x < 0 or x > Screen.WIDTH) or
            (y < 0 or y > Screen.HEIGHT)
        )
        distance = np.hypot(x - targetX, y - targetY)
        speed = np.hypot(vx, vy)

        terminated = (
            abs(theta) > np.radians(75)
            or speed > 1500.0
            or outOfBounds
        )

        # ------------------------ REWARD STRUCTURE ----------------------
        # Distance reward: Closer to the target is better
        if prevDistance > distance:
            reward = (prevDistance - distance) * 0.8
        elif prevDistance <= distance:
            reward = (prevDistance - distance) * 1.5

        # When close to the target, heavily penalize remaining speed.
        if distance < 150.0:
            fProximity = 1.0 - (distance / 150.0)
            reward -= fProximity * speed * 0.05

        # Orientation reward: Encourage level flight (theta near 0)
        reward -= abs(theta) * 0.08
        reward -= abs(omega) * 0.2

        # Boundary penalty: Discourage hitting the walls
        if outOfBounds:
            reward -= 50.0

        # Check if waypoint reached
        waypointReached = False
        if distance < self.waypointThresh:
            waypointReached = True
            # Provide bonus reward for the first waypoint hit
            if not self._waypointBonusGiven:
                reward += 2.0
                self._waypointBonusGiven = True
            
            # Bonus reward for staying in the proximity
            reward += 10.0

            if self.cycleWaypoints:
                # Move to next waypoint (cyclic)
                self.currWaypointIdx = (self.currWaypointIdx + 1) % len(self.waypoints)
                # Switch wind condition to new waypoint
                self.targetWindDirection, self.targetWindSpeed = self.waypointWinds[self.currWaypointIdx]

        flogger.info(f"[STATE] Position: ({x:.1f}, {y:.1f}) | Velocity: ({vx:.1f}, {vy:.1f}) | Distance: {distance:.1f} | Wind: ({effWindX}, {effWindY})")

        # Heavy penalty for crashing
        if terminated:
            reward -= 15.0

        flogger.info(f"[REWARD] Reward: {reward:.2f} | Terminated: {terminated}")

        isDone = self.currStep >= self.stepsMax
        # If the episode is done, log the "status" of the episode
        if isDone:
            flogger.info(f" Step {self.currStep} ::: Successful? -> {isDone} ".center(100, chr(45)))

        # Normalize state before taking an action step
        normState = self._normalizeState(self.drone.state)

        return (
            normState,
            reward,
            terminated,
            isDone,
            {"rawState": self.drone.state, "waypointReached": waypointReached},
        )

    def getExpertAction(self) -> np.ndarray:
        if self.expert:
            return self.expert.getAction(
                self.drone.state,
                self.waypoints[self.currWaypointIdx],
                Physics.DT
            ).toNumpy()
        else:
            raise ValueError("Environment not initialized with an expert controller.")

    def runUserControl(self):
        isRunning: bool = True

        while isRunning:
            # At start, ~82% power on both (hover)
            action = Action(0.75, 0.75)

            if self.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        isRunning = False

            # For "expert" control
            if self.expert:
                action = self.getExpertAction()

            # For "user" control
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
    # -------------------------------- RENDERING --------------------------------
    def runExpertCollectorPipeline(self, numTrajec=2000):
        """
        Visualizes the exact process of the ExpertCollector to debug trajectory health.
        """
        from core.collector import ExpertCollector

        collector = ExpertCollector(
            numTrajec=numTrajec, 
            stepsPerTrajec=1000,
            waypointIdx=0
        )
        # Collect the data and render the process
        collector.collect(environment=self)

        if self.render:
            pygame.quit()
            os.system("clear")

    def _drawCollectorHUD(self, accepted, rejected, attempts, target, reward, wpHit, step, maxStep, crashed):
        """Draws real-time stats at the bottom of the screen during collection."""
        texts = [
            f"[COLLECTOR] Waypoint: {target} | Accepted: {accepted} / {target} | Rejected: {rejected} | Attempts: {attempts}",
            f"Step: {step} / {maxStep} | Crashed?: {crashed} | Waypoint Hits: {wpHit} | Reward: {reward:.1f}",
        ]
        yStart = Screen.HEIGHT - 50
        for i, text in enumerate(texts):
            color = (Colors.WHITE.R, Colors.WHITE.G, Colors.WHITE.B)
            surface = self.font.render(text, True, color)
            self.screen.blit(surface, (10, yStart + i * 20))

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

        # ---------------- DRAW DRONE ----------------
        self._drawDroneBody(cx, cy, theta)

        # ---------------- DRAW CURRENT WAYPOINT ----------------
        targetX, targetY = self.waypoints[self.currWaypointIdx]

        # Current target waypoint
        pygame.draw.circle(
            self.screen,
            (Colors.GREEN.R, Colors.GREEN.G, Colors.GREEN.B),
            (int(targetX), int(targetY)),
            8,
        )

        # Guidance line
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
            f"Wind ::: X: {windX:.3f} | Y: {windY:.3f}"
        ]
        for i, text in enumerate(texts):
            surface = self.font.render(
                text, True, (Colors.WHITE.R, Colors.WHITE.G, Colors.WHITE.B)
            )
            self.screen.blit(surface, (10, 10 + i * 20))

        # -------------------------------- WIND VISUALIZATION --------------------------------
        # Calculate arrow properties
        windMagnitude = math.hypot(windX, windY)
        lenArrows = max(windMagnitude / 2.0, 5.0)
        angle = math.atan2(windY, windX)

        # Arrow base (center of screen)
        baseX, baseY = Screen.WIDTH - 100, Screen.HEIGHT - 100
        
        # Draw circle to contain wind magnitiude visualization
        circleRadius = max( int(lenArrows), 25.0)
        if circleRadius > 0:
            pygame.draw.circle(
                self.screen,
                (Colors.RED.R, Colors.RED.G, Colors.RED.B),
                (baseX, baseY),
                circleRadius,
                1
            )

        # Arrow tip (displaced from center)
        tipX = baseX + lenArrows * math.cos(angle)
        tipY = baseY + lenArrows * math.sin(angle)

        # Draw the arrow shaft
        pygame.draw.line(
            self.screen,
            (Colors.RED.R, Colors.RED.G, Colors.RED.B),
            (baseX, baseY),
            (tipX, tipY),
            3,
        )

        # Draw arrowhead
        lAngle = angle + math.pi * 0.85
        rAngle = angle - math.pi * 0.85
        lenHead = 5

        leftX = tipX + lenHead * math.cos(lAngle)
        leftY = tipY + lenHead * math.sin(lAngle)
        rightX = tipX + lenHead * math.cos(rAngle)
        rightY = tipY + lenHead * math.sin(rAngle)

        pygame.draw.polygon(
            self.screen,
            (Colors.RED.R, Colors.RED.G, Colors.RED.B),
            [(tipX, tipY), (leftX, leftY), (rightX, rightY)],
        )

        lblWind = self.font.render("Active Wind", True, (Colors.RED.R, Colors.RED.G, Colors.RED.B))
        labelRect = lblWind.get_rect(center=(baseX, baseY - circleRadius - 15))
        self.screen.blit(lblWind, labelRect)


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
    env = Environment(useExpert=True, render=False)
    env.runExpertCollectorPipeline()
    # env.runUserControl()
