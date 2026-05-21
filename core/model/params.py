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
        self.METRICS = self.TEST / "metrics.csv"

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
    For every 1 buffer rollout, we perform <GENERATOR> PPO updates.
    """
    GENERATOR: int = 3


@dataclass
class Params:
    """
    Hyperparameters for PPO training (without GAIL).
    """

    PATHS: Paths = Paths()
    # --- Optimizers ---
    LR: float = 3e-4                # Learning rate for both Policy and Critic
    # --- Environment & Buffer ---
    BUFFER_SIZE: int = 4096         # Steps collected per iteration
    GAMMA: float = 0.99             # Discount factor for future rewards
    LAMBDA_GAE: float = 0.95        # GAE smoothing parameter
    # --- PPO Specifics ---
    PPO_EPOCHS: int = 4             # Times to loop over the buffer per update
    BATCH_SIZE: int = 256           # Minibatch size for PPO
    CLIP_EPSILON: float = 0.2       # PPO clipping parameter (epsilon)
    COEFF_VALUE: float = 1.0        # Weight for the Critic's MSE loss
    COEFF_ENTROPY: float = 0.015    # Weight for the Entropy bonus
    # --- Training Loop ---
    # Update ratios
    UPDATES: Update = field(default_factory=Update)
    ITERATIONS: int = 1000          # Total number of times to collect a buffer and update
