import math
import os
import random

import numpy as np
from tqdm import tqdm

from core.config import Physics, Screen
from core.env import Environment


class DataCollector:
    def __init__(self, numTrajec=100, stepsPerTrajec=500):
        self.numTrajec: int = numTrajec
        self.stepsPerTrajec: int = stepsPerTrajec
        self.allTrajec: list[np.ndarray] = []

    def normalizeState(self, state, range=(-1, 1)):
        """Scales raw physics state into a range for Neural Network stability."""
        x, y, vx, vy, theta, omega = state

        # Normalize the state variables
        normX = (x - (Screen.WIDTH / 2)) / (Screen.WIDTH / 2)
        normY = (y - (Screen.HEIGHT / 2)) / (Screen.HEIGHT / 2)

        normVX = np.clip(vx / 1000.0, range[0], range[1])
        normVY = np.clip(vy / 1000.0, range[0], range[1])

        normTheta = theta / math.pi
        normOmega = np.clip(omega / 5.0, range[0], range[1])

        return np.array(
            [normX, normY, normVX, normVY, normTheta, normOmega], dtype=np.float32
        )

    def initiate(self, environment):
        # Start from a random position and velocity
        initX = random.uniform(Screen.MARGIN + 50, Screen.WIDTH - Screen.MARGIN - 50)
        initY = random.uniform(Screen.MARGIN + 50, Screen.HEIGHT - Screen.MARGIN - 50)
        # Pick a random initial tilt and velocity
        initVx = random.uniform(-200, 200)
        initVy = random.uniform(-200, 200)
        initTheta = random.uniform(-0.3, 0.3)  # Up to ~17 degrees off axis

        environment.drone.state = np.array(
            [initX, initY, initVx, initVy, initTheta, 0.0], dtype=np.float32
        )

        environment.expert.currWaypointIdx = 0
        return list()

    def collect(self):
        env = Environment(useExpert=True, render=False)

        # Wrap the range() in tqdm for a beautiful progress bar
        for idxTrajec in tqdm(
            range(self.numTrajec),
            desc="[DATA] Collecting Trajectories",
            unit=" traj",
            ncols=100,
        ):
            dataTrajec = self.initiate(env)

            for step in range(self.stepsPerTrajec):
                currState = env.drone.state.copy()

                # Get expert action
                action = env.expert.getAction(currState, Physics.DT)

                env.step(action)

                # Process data for GAIL training
                normState = self.normalizeState(currState)
                actionRaw = action.to_numpy().astype(np.float64)

                # Transition : [state (6), action (2)]
                transition = np.concatenate([normState, actionRaw])
                dataTrajec.append(transition)

            # Trajectory : (steps, transition)
            self.allTrajec.append(np.array(dataTrajec))

        print("\n[DATA] Collection complete!")

    def save(self, dir="data", filename="expert.npz"):
        if not os.path.exists(dir):
            os.makedirs(dir)

        filepath = os.path.join(dir, filename)

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
