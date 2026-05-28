import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Assuming you can import screen config to match env.py exactly
from core.config import WAYPOINTS, Screen


class ExpertDataset(Dataset):
    def __init__(self, path: str, targetIdx: int = 0):
        data = np.load(path)
        self.transitions = np.concatenate(data["trajectories"], axis=0).astype(np.float32)
        self.targetIdx: int = targetIdx

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        # * Dx and Dy is to be calculated.
        stateRaw = self.transitions[idx, :8]  # [X, Y, Vx, Vy, Theta, Omega, windFx, windFy]
        action = self.transitions[idx, 8:]    # [Tleft, Tright]
        x, y, vx, vy, theta, omega, windFx, windFy = stateRaw
        
        normX = np.clip(
            (x - Screen.WIDTH / 2) / (Screen.WIDTH / 2),
            -1.0, 1.0
        )
        normY = np.clip(
            (y - Screen.HEIGHT / 2) / (Screen.HEIGHT / 2),
            -1.0, 1.0
        )

        # Relative Distance To Target 
        targetX, targetY = WAYPOINTS[self.targetIdx]
        dx = targetX - x
        dy = targetY - y

        maxDistance = math.hypot(
            Screen.WIDTH / 2,
            Screen.HEIGHT / 2
        )

        normDx = np.clip(
            dx / maxDistance,
            -1.0, 1.0
        )

        normDy = np.clip(
            dy / maxDistance,
            -1.0, 1.0
        )
        # Active Speed
        normVx = np.clip(
            (vx / 1500.0),
            -1.0, 1.0
        )
        normVy = np.clip(
            (vy / 1500.0),
            -1.0, 1.0
        )

        # Active Orientation in Angles 
        normTheta = np.clip(
            (theta / math.pi),
            -1.0, 1.0
        )

        # Active Amgular Speed
        normOmega = np.clip(
            (omega / 10.0),
            -1.0, 1.0
        )

        # Active Wind Forces
        normWindFx = np.clip(
            (windFx / 25.0),
            -1.0, 1.0
        )
        normWindFy = np.clip(
            (windFy / 25.0),
            -1.0, 1.0
        )

        normState = np.array([
            normX, normY, normVx, normVy, 
            normTheta, normOmega, 
            normWindFx, normWindFy,
            normDx, normDy
        ], dtype=np.float32)

        tState = torch.as_tensor(normState, dtype=torch.float32)
        tAction = torch.as_tensor(action, dtype=torch.float32)

        return tState, tAction

def getExpertDataloader(path: str, sBatch: int) -> torch.utils.data.DataLoader:
    dataset = ExpertDataset(path)
    return DataLoader(
        dataset,
        batch_size=sBatch,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )