import numpy as np
import params
import torch
from torch.utils.data import DataLoader, Dataset


class ExpertDataset(Dataset):
    def __init__(self, path: str):
        data = np.load(path)
        self.transitions = np.concatenate(data["trajectories"], axis=0).astype(
            np.float32
        )

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        state = self.transitions[idx, :6]  # [X, Y, Vx, Vy, Theta, Omega]
        action = self.transitions[idx, 6:]  # [Tleft, Tright]
        # Transform the data to tensors
        tState = torch.tensor(state, dtype=torch.float32)
        tAction = torch.tensor(action, dtype=torch.float32)

        return tState, tAction


def getExpertDataloader(path: str, params: params.Params) -> DataLoader:
    dataset = ExpertDataset(path)
    return DataLoader(
        dataset,
        batch_size=params.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        # Other params if needed
    )
