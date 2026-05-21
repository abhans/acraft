import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Assuming you can import screen config to match env.py exactly
from core.config import Screen


class ExpertDataset(Dataset):
    def __init__(self, path: str):
        data = np.load(path)
        self.transitions = np.concatenate(data["trajectories"], axis=0).astype(np.float32)

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        stateRaw = self.transitions[idx, :8]  # [X, Y, Vx, Vy, Theta, Omega, windFx, windFy]
        action = self.transitions[idx, 8:10]   # [Tleft, Tright]
        
        x, y, vx, vy, theta, omega, windFx, windFy = stateRaw
        
        normX = (x - Screen.WIDTH / 2) / (Screen.WIDTH / 2)
        normY = (y - Screen.HEIGHT / 2) / (Screen.HEIGHT / 2)
        normVx = np.clip(vx / 1000.0, -1.0, 1.0)
        normVy = np.clip(vy / 1000.0, -1.0, 1.0)
        normTheta = np.clip(theta / math.pi, -1.0, 1.0)
        normOmega = np.clip(omega / 5.0, -1.0, 1.0)
        normWindFx = np.clip(windFx / 300.0, -1.0, 1.0)
        normWindFy = np.clip(windFy / 300.0, -1.0, 1.0)

        normState = np.array([
            normX, normY, normVx, normVy, 
            normTheta, normOmega, normWindFx, normWindFy
        ], dtype=np.float32)

        tState = torch.tensor(normState, dtype=torch.float32)
        tAction = torch.tensor(action, dtype=torch.float32)

        return tState, tAction

def getExpertDataloader(path: str, sBatch: int) -> torch.utils.data.DataLoader:
    dataset = ExpertDataset(path)
    return DataLoader(
        dataset,
        batch_size=sBatch,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )