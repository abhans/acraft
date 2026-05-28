import json
import logging
import os
from logging import handlers
from pathlib import Path

fmtDEBUG: str = ">>> %(asctime)s | %(levelname)s: %(msg)s -> %(name)s @ %(filename)s: Ln %(lineno)d"

class BaseLogger(logging.Logger):
    def __init__(self, name: str, level: int = logging.NOTSET):
        """
        Base Logger class for logging, handling exceptions and configuration.

        :param name: Name of the logger.
        :type name: str
        :param level: Logging level. Default is ``logging.NOTSET``.
        :type level: int
        """
        super().__init__(name, level)
        self._formatter = logging.Formatter(
            "%(msg)s",
            datefmt="%H:%M:%S",
        )
        self._handler()

    def _handler(self) -> None:
        """
        Set the handler for the logger. This is a "stream" handler that outputs to the console.
        """
        _handler = logging.StreamHandler()
        _handler.setFormatter(self._formatter)

        self.addHandler(_handler)


class FileLogger(BaseLogger):
    def __init__(
            self, name: str,
            filename: str,
            path: str | Path,
            level: int = logging.INFO,
            n_backup: int = 0
    ):
        """
        Logger that saves the logs to a set of files.

        :param name: Name of the logger.
        :type name: str
        :param filename: Name of the file to save the logs.
        :type filename: str
        :param path: Path to the directory where the log file will be saved. Default is ``'logs'`` directory.
        :type path: str
        :param level: Logging level. Default is ``logging.INFO``.
        :type level: int
        :param n_backup: Number of backup files to keep. Default is ``0`` (No rollover).
        :type n_backup: int
        """
        self._filename = filename
        self._path = path
        self._n_backup = n_backup
        # Rest of the parameters are passed to the parent class
        super().__init__(name, level)
        
    def _handler(self) -> None:
        """
        Set the handler for the logger.
        """
        if not os.path.exists(self._path):
            os.makedirs(self._path)
        # Set the file handler with rotation
        _file_handler = handlers.RotatingFileHandler(
            filename=Path(self._path) / self._filename, 
            maxBytes=1024,
            backupCount=self._n_backup
        )
        _file_handler.setFormatter(self._formatter)
        
        self.addHandler(_file_handler)

class TrainLogger:
    def __init__(self):
        self.history = {
            "iteration": [],
            "policyLoss": [],
            "valueLoss": [],
            "entropy": [],
            "reward": [],
        }

    def log(self, iteration: int, ppoStats: dict, avgReward: float = 0.0):
        self.history["iteration"].append(iteration)
        self.history["policyLoss"].append(float(ppoStats["policyLoss"]))
        self.history["valueLoss"].append(float(ppoStats["valueLoss"]))
        self.history["entropy"].append(float(ppoStats["entropy"]))
        self.history["reward"].append(float(avgReward))

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=4)

        print(f"\n[SYSTEM] Training metrics saved to '{filepath}'")

logger = BaseLogger("Acraft")
flogger = FileLogger(
    "Acraft",
    filename="acraft.log", path="logs"
)