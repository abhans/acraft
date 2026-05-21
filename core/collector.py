import os

import numpy as np
from tqdm import tqdm

from core.config import Physics
from core.env import Environment


class DataCollector:
    def __init__(
            self,
            numTrajec: int = 100,
            stepsPerTrajec: int = 500,
            maxAttemptsMultiplier: int = 30
    ):
        self.numTrajec: int = numTrajec
        self.stepsPerTrajec: int = stepsPerTrajec
        self.allTrajec: list[np.ndarray] = []
        self.maxAttempts: int = numTrajec * maxAttemptsMultiplier

        # Quality Thresholds for filtering out bad trajectories
        self.qualityThresholds = {
            "maxTheta": np.radians(30),  # Max tilt angle in radians
            "maxOmega": 8.0,             # Max angular velocity in radians/s
            "minReward": -50.0,          # Minimum cumulative reward for a trajectory to be considered valid
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

    def collect(self):
        env = Environment(useExpert=True, render=False, stepsMax=self.stepsPerTrajec)
        
        # Create counters for accepted/rejected trajectories
        accepted: int = 0
        rejected: int = 0
        attempts: int = 0

        # Wrap the range() in tqdm for a beautiful progress bar
        with tqdm(range(self.numTrajec), desc="[DATA] Collecting Trajectories", unit=" traj", ncols=100) as pbar: 
            while accepted < self.numTrajec and attempts < self.maxAttempts:
                attempts += 1

                env.reset(options={"randomStart": True})
                # Randomize expert behavior for this trajectory
                env.expert.randomize()

                dataTrajec: list[np.ndarray] = []
                tjReward: float = 0.0
                tjThetaSum: float = 0.0
                tjOmegaSum: float = 0.0
                tjWaypointsHit: int = 0
                tjCrashed: bool = False

                for step in range(self.stepsPerTrajec):
                    currState = env.drone.state.copy()

                    # Get expert action
                    action = env.expert.getAction(currState, Physics.DT)

                    _, reward, terminated, truncated, info = env.step(action)
                    # Calculate the metrics for quality assessment
                    tjReward += reward
                    tjThetaSum += abs(currState[4])  # theta
                    tjOmegaSum += abs(currState[5])  # omega
                    tjWaypointsHit += int(info.get("waypointReached", False))
                    tjCrashed = tjCrashed or terminated


                    # Process data for PPO training
                    actionRaw = action.toNumpy().astype(np.float32)

                    # Transition : [state (8), action (2)]
                    transition = np.concatenate([currState.astype(np.float32), actionRaw])
                    dataTrajec.append(transition)

                    if terminated:
                        tjCrashed = True
                        break
                    
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
                # If the trajectory is not healthy, discard it and continue collecting
                else:
                    rejected += 1

        print("\n[DATA] Collection complete!")
        print(f"\t>>> Accepted: {accepted} | Rejected: {rejected} | Attempts: {attempts}")

        # If we exit the loop without collecting enough trajectories, raise an error
        if accepted < self.numTrajec:
            raise RuntimeError(
                f"Only collected {accepted}/{self.numTrajec} healthy trajectories. "
                f"Relax the acceptance thresholds or improve the expert."
            )


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


if __name__ == "__main__":
    collector = DataCollector(numTrajec=20, stepsPerTrajec=500)
    collector.collect()
    collector.save()
