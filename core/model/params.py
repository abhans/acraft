from dataclasses import dataclass


@dataclass
class Update:
    """
    Represents a single training update for the
    components generator or discriminator
    """

    GENERATOR: int = 1
    DISCRIMINATOR: int = 5


@dataclass
class Params:
    """
    Hyperparameters for GAIL training
    and expert behavior.
    """

    LR: float = 3e-4
    BATCH_SIZE: int = 64
    EPOCHS: int = 100
    UPDATES: Update = Update(1, 5)
    # Gradient penalty for WGAN-GP
    LAMBDA: float = 10.0
