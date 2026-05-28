import os
from dataclasses import dataclass
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
        self.TEST: Path | None = None


    def setTest(self, testID: str):
        self.TEST = self._TESTS / testID

    def createTest(self):
        self.TEST = self._TESTS / self.initiateTest()

    def ensure(self, weights, metrics):
        """
        Creates parent directories if they don't exist.
        """
        self.EXPERT.parent.mkdir(parents=True, exist_ok=True)
        weights.parent.mkdir(parents=True, exist_ok=True)
        metrics.parent.mkdir(parents=True, exist_ok=True)

    def getCurrentTest(self) -> str:
        """
        Returns the latest available test ID.
        """
        self._TESTS.mkdir(parents=True, exist_ok=True)

        testIDs = []

        for item in os.listdir(self._TESTS):
            if item.isdigit():
                testIDs.append(int(item))
        
        currTestID = max(testIDs)

        return f"{currTestID:02d}"
    
    def initiateTest(self):
        """
        Returns the "next" available test ID.
        """
        self._TESTS.mkdir(parents=True, exist_ok=True)

        testIDs = []
        initTest: int = 1

        for item in os.listdir(self._TESTS):
            if item.isdigit():
                testIDs.append(int(item))
                initTest += 1
        
        return f"{initTest:02d}"


@dataclass
class Update:
    """
    Controls the ratio of network updates per training iteration.
    For every 1 buffer rollout, we perform <GENERATOR> PPO updates.
    """
    GENERATOR: int = 3        # Number of PPO updates per buffer rollout

@dataclass
class Splits:
    """
    Hyperparameters for dataset splitting.
    """
    TRAIN: float = 0.8        # Fraction of data to use for training
    VALIDATION: float = 0.2   # Fraction of data to use for validation


class BClone:
    """
    Hyperparameters for 'Behavior Clone' training.
    """
    def __init__(self, paths: Paths):
        self.PATHS = paths                          # All file paths
        
        if paths.TEST is None:
            raise ValueError(
                "[BClone] Paths.TEST is not set. "
                "Call paths.setTest() or paths.createTest() first."
            )

        self.MODEL = self.PATHS.TEST / "bclone"     # Path to save the trained policy weights/metrics
        self.METRICS = self.MODEL / "metrics.csv"   # Path to save training metrics
        self.WEIGHTS = self.MODEL / "bclone.pth"    # Path to save trained policy weights
        # --- Training ---
        self.BATCH_SIZE: int = 512                  # Batch size for training
        self.EPOCHS: int = 50                       # Number of epochs to train
        self.LR: float = 1e-5                       # Learning rate for the optimizer
        # Dataset splits
        self.SPLITS: Splits = Splits()

        # Ensure that the directories exists
        self.PATHS.ensure(
            self.WEIGHTS,
            self.METRICS,
        )


class PPO:
    """
    Hyperparameters for PPO training.
    """
    def __init__(self, paths: Paths):
        if paths.TEST is None:
            raise ValueError(
                "[PPO] Paths.TEST is not set. "
                "Call paths.setTest() or paths.createTest() first."
            )

        self.PATHS = paths                          # All file paths

        self.MODEL = self.PATHS.TEST / "ppo"        # Path to save the trained policy weights/metrics
        self.METRICS = self.MODEL / "metrics.csv"   # Path to save training metrics
        self.WEIGHTS = self.MODEL / "ppo.pth"       # Path to save trained policy weights
        # --- Optimizers ---
        self.LR: float = 2e-4                       # Learning rate for both Policy and Critic
        # --- Environment & Buffer ---
        self.BUFFER_SIZE: int = 4096                # Steps collected per iteration
        self.GAMMA: float = 0.99                    # Discount factor for future rewards
        self.LAMBDA_GAE: float = 0.95               # GAE smoothing parameter
        # --- PPO Specifics ---
        self.EPOCHS: int = 4                        # Times to loop over the buffer per update
        self.BATCH_SIZE: int = 512                  # Minibatch size for PPO
        self.CLIP_EPSILON: float = 0.1              # PPO clipping parameter (epsilon)
        self.COEFF_VALUE: float = 1.0               # Weight for the Critic's MSE loss
        self.COEFF_ENTROPY: float = 0.001           # Weight for the Entropy bonus
        self.COEFF_BCLONE: float = 0.1              # Weight for the Behavior Cloning loss
        # --- Training Loop ---
        # Update ratios
        self.UPDATES: Update = Update()
        self.ITERATIONS: int = 1000                 # Total number of times to collect a buffer and update

        # Ensure that the directories exists
        self.PATHS.ensure(
            self.WEIGHTS,
            self.METRICS,
        )
