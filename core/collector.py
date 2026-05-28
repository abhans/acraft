import logging
import os

import numpy as np
import pygame
from tqdm import tqdm

from core.config import Colors, Physics, Screen
from core.env import Environment
from core.logger import FileLogger

flogger = FileLogger(
    "Acraft",
    level=logging.DEBUG,
    filename="collect.log", path="logs"
)

class ExpertCollector:
    def __init__(
            self,
            numTrajec,
            stepsPerTrajec: int = 500,
            waypointIdx: int = 0
    ):
        self.numTrajec: int = numTrajec
        self.stepsPerTrajec: int = stepsPerTrajec
        self.targetWaypointIdx: int = waypointIdx
        self.allTrajec: list[np.ndarray] = []
        self.maxAttempts: int | None = None

        # Quality Thresholds for filtering out bad trajectories
        self.qualityThresholds = {
            "maxTheta": np.radians(30),  # Max tilt angle in radians
            "maxOmega": 15.0,            # Max angular velocity in radians/s
            "minReward": 50.0,            # Minimum cumulative reward for a trajectory to be considered valid
            "minWaypointHit": 1,         # Minimum number of waypoints that must be reached in a trajectory
        }

    def _isTrajeectoryHealthy(
            self,
            reward: float,
            theta: float,
            omega: float,
            waypointsHit: int,
            crashed: bool,
            nSteps: int
    ):
        return (
            (not crashed) 
            and nSteps == self.stepsPerTrajec
            and waypointsHit >= self.qualityThresholds["minWaypointHit"]
            and reward >= self.qualityThresholds["minReward"]
            and abs(theta) <= self.qualityThresholds["maxTheta"]
            and abs(omega) <= self.qualityThresholds["maxOmega"]
        )

    def collect(self, environment):
        if environment is None:
            environment = Environment(
                useExpert=True,
                render=False,
                stepsMax=self.stepsPerTrajec
            )

        # * PPO training for a single target
        environment.cycleWaypoints = False
        targetCoords: tuple[int, int] = environment.waypoints[self.targetWaypointIdx]
        
        # Create counters for accepted/rejected trajectories
        accepted: int = 0
        rejected: int = 0
        totalAttempts: int = 0

        self.maxAttempts = self.numTrajec * 20

        # Wrap the range() in tqdm for a beautiful progress bar
        with tqdm(range(self.numTrajec), desc=f"[DATA] Collecting Trajectories for X:{targetCoords[0]} | Y:{targetCoords[1]}  ", unit=" traj", ncols=150) as pbar: 
            # Current Target
            while accepted < self.numTrajec and totalAttempts < self.maxAttempts:
                # Update the number of totalAttempts for trajectory
                totalAttempts += 1

                environment.reset(options={"randomStart": True})
                # Randomize expert behavior for this trajectory
                environment.expert.randomize()

                environment.currWaypointIdx = self.targetWaypointIdx
                environment._waypointBonusGiven = False

                environment.targetWindDirection, environment.targetWindSpeed = environment.waypointWinds[self.targetWaypointIdx]
                environment.windDirection = environment.targetWindDirection
                environment.windSpeed = environment.targetWindSpeed

                dataTrajec: list[np.ndarray] = []
                
                tjReward: float = 0.0
                tjThetaSum: float = 0.0
                tjOmegaSum: float = 0.0
                tjWaypointsHit: int = 0
                tjCrashed: bool = False

                for step in range(self.stepsPerTrajec):
                    # ------------------- UI EVENT HANDLING -------------------
                    if environment.render:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                # Force quit the loop gracefully
                                totalAttempts = self.maxAttempts
                                break
                
                    currState = environment.drone.state.copy()

                    # Get expert action
                    action = environment.expert.getAction(currState, targetCoords, Physics.DT)

                    _, reward, terminated, truncated, info = environment.step(action)
                    # Calculate the metrics for quality assessment
                    tjReward += reward
                    tjThetaSum += abs( currState[4] )  # theta
                    tjOmegaSum += abs( currState[5] )  # omega
                    tjWaypointsHit += int(info.get("waypointReached", False))

                    if terminated:
                        tjCrashed = True
                        break

                    # Process data for PPO training
                    actionRaw = action.toNumpy().astype(np.float32)

                    # Transition : [state (8), action (2)]
                    transition = np.concatenate([currState.astype(np.float32), actionRaw])
                    dataTrajec.append(transition)
                    
                    # ------------------- UI RENDERING -------------------
                    if environment.render:
                        environment.screen.fill((Colors.BLACK.R, Colors.BLACK.G, Colors.BLACK.B))
                        pygame.draw.rect(
                            environment.screen,
                            (Colors.GRAY.R, Colors.GRAY.G, Colors.GRAY.B),
                            (0, 0, Screen.WIDTH, Screen.HEIGHT), 2
                        )
                        
                        environment._drawDrone()
                        environment._drawHUD()
                        environment._drawCollectorHUD(
                            accepted, rejected,
                            totalAttempts,
                            targetCoords, tjReward, tjWaypointsHit,
                            step, self.stepsPerTrajec,
                            tjCrashed
                        )
                        
                        pygame.display.flip()
                        environment.clock.tick(Physics.FPS)
                
                # ------------------- EVALUATE TRAJECTORY -------------------
                # After collecting the trajectory, check if it's healthy
                numSteps = len(dataTrajec)
                thetaAverage = tjThetaSum / max(numSteps, 1)  # Average theta over the trajectory
                omegaAverage = tjOmegaSum / max(numSteps, 1)  # Average omega over the trajectory

                if self._isTrajeectoryHealthy(
                    reward=tjReward,
                    theta=thetaAverage,
                    omega=omegaAverage,
                    waypointsHit=tjWaypointsHit,
                    crashed=tjCrashed,
                    nSteps=numSteps
                ):
                    self.allTrajec.append(np.array(dataTrajec))
                    accepted += 1
                    pbar.update(1)
                    textStatus = "Trajectory :: ACCEPTED"
                    colorStatus = Colors.GREEN
                # If the trajectory is not healthy, discard it and continue collecting
                else:
                    rejected += 1
                    textStatus = "Trajectory :: REJECTED"
                    colorStatus = Colors.RED

                # ------------------- UI RESULT SCREEN -------------------
                if environment.render:
                    environment.screen.fill((Colors.BLACK.R, Colors.BLACK.G, Colors.BLACK.B))
                    environment._drawDrone()
                    
                    statusSurface = environment.font.render(
                        textStatus, True,
                        (colorStatus.R, colorStatus.G, colorStatus.B)
                    )
                    environment.screen.blit(statusSurface, (Screen.WIDTH // 2 - 80, 50))
                    
                    reasons = []
                    if tjCrashed:
                        reasons.append(f"Crashed at step {numSteps}")

                    elif numSteps != self.stepsPerTrajec: 
                        reasons.append(f"Short duration ({numSteps} / {self.stepsPerTrajec})")

                    elif tjWaypointsHit < self.qualityThresholds["minWaypointHit"]:
                        reasons.append("No Waypoints Hit")

                    elif tjReward < self.qualityThresholds["minReward"]:
                        reasons.append(f"Low Reward ({tjReward:.1f})")

                    elif abs(thetaAverage) > self.qualityThresholds["maxTheta"]:
                        reasons.append(f"High Average Theta ({np.degrees(thetaAverage):.1f}°)")
                    
                    for i, reason in enumerate(reasons):
                        respSurface = environment.font.render(f"-> {reason}", True, (Colors.ORANGE.R, Colors.ORANGE.G, Colors.ORANGE.B))
                        environment.screen.blit(respSurface, (Screen.WIDTH // 2 - 100, 80 + i * 20))

                    pygame.display.flip()
                    pygame.time.wait(1500) # Pause briefly to show the result

            # Log the metrics of the trajectory
            flogger.debug(f" Trajectory Metrics [{textStatus}] ".center(50, chr(45)))
            flogger.debug(f":: Crashed?: {tjCrashed}")
            flogger.debug(f":: Steps: {numSteps} / {self.stepsPerTrajec}")
            flogger.debug(f":: Waypoints Hit: {tjWaypointsHit}")
            flogger.debug(f":: Total Reward: {tjReward:.2f} (Need >= {self.qualityThresholds['minReward']})")
            flogger.debug(f":: Average Theta: {np.degrees(thetaAverage):.2f}° (Need <= {np.degrees(self.qualityThresholds['maxTheta']):.2f}°)")
            flogger.debug(f":: Average Omega: {omegaAverage:.2f} (Need <= {self.qualityThresholds['maxOmega']})\n")

        print("\n[DATA] Collection complete!")
        print(f"\t>>> Accepted: {accepted} | Rejected: {rejected} | Total Attempts: {totalAttempts}")
        # If we exit the loop without collecting enough trajectories, raise an error
        if accepted < self.numTrajec:
            raise RuntimeError(
                f"Only collected {accepted} / {self.numTrajec} healthy trajectories. "
                f"Relax the acceptance thresholds or improve the expert."
            )
        
        # Save the collected expert data
        self.save()


    def save(self, dir="data", filename="expert.npz"):
        if not os.path.exists(dir):
            os.makedirs(dir)

        filepath = os.path.join(dir, filename)

        if not self.allTrajec:
            raise RuntimeError("No trajectories collected; refusing to save empty dataset.")

        # Save as a compressed numpy ZIP file
        np.savez_compressed(filepath, trajectories=self.allTrajec)

        transitionTotal = sum(trajec.shape[0] for trajec in self.allTrajec)
        print(
            f"\n[SAVE] Saved {len(self.allTrajec)} trajectories",
            f"({transitionTotal} transitions) to '{filepath}'",
        )
        print(f"[SAVE] Shape per trajectory: {self.allTrajec[0].shape}")
