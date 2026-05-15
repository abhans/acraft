import os
from dataclasses import dataclass, field
from pathlib import Path


class Paths:
    """
    Defines all file paths for model training.
    Automatically resolves paths relative to the project root.
    """

    def __init__(self) -> None:
        # __file__ is core/model/params.py. .parent.parent.parent gets us to the project root.
        self._ROOT = Path(__file__).resolve().parent.parent.parent
        self._CORE = self._ROOT / "core"
        self._DATA = self._ROOT / "data"
        self.EXPERT = self._DATA / "expert.npz"
        self._TESTS = self._ROOT / "tests"
        # ------------ Paths for the Model ------------
        self.TEST = self._TESTS / self.getCurrentTest()
        self.WEIGHTS = self.TEST / "policy.pth"
        self.METRICS = self.TEST / "metrics.json"

    def ensureDirs(self):
        """Creates parent directories if they don't exist."""
        self.EXPERT.parent.mkdir(parents=True, exist_ok=True)
        self.WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        self.METRICS.parent.mkdir(parents=True, exist_ok=True)

    def getCurrentTest(self) -> str:
        currTestID: int = 1
        testIDs: list[int] = [int(test) for test in os.listdir(self._TESTS)]

        for ID in testIDs:
            if currTestID in testIDs:
                currTestID += 1

        return f"0{currTestID}" if currTestID < 10 else f"{currTestID}"


@dataclass
class Update:
    """
    Controls the ratio of network updates per training iteration.
    Doing multiple Discriminator updates per Generator update is a
    common trick to prevent the Discriminator from overpowering the Policy.
    """

    GENERATOR: int = 3
    DISCRIMINATOR: int = 1


@dataclass
class Params:
    """
    Hyperparameters for GAIL training based on PPO.
    """

    PATHS: Paths = Paths()
    # --- Optimizers ---
    LR: float = 4.0e-5
    LR_DISCRIMINATOR: float = 5e-6
    # --- Environment & Buffer ---
    BUFFER_SIZE: int = 1024  # Steps collected per iteration
    GAMMA: float = 0.99  # Discount factor for future rewards
    LAMBDA_GAE: float = 0.95  # GAE smoothing parameter (replaces WGAN-GP lambda)
    # --- PPO Specifics ---
    PPO_EPOCHS: int = 6  # Times to loop over the buffer per update
    BATCH_SIZE: int = 64  # Minibatch size for PPO
    CLIP_EPSILON: float = 0.20  # PPO clipping parameter (epsilon)
    COEFF_VALUE: float = 1.0  # Weight for the Critic's MSE loss
    COEFF_ENTROPY: float = 0.2  # Weight for the Entropy bonus (lambda in GAIL paper)
    # --- Training Loop ---
    UPDATES: Update = field(default_factory=Update)  # Update ratios
    ITERATIONS: int = 500  # Total number of times to collect a buffer and update
