import json


class GAILLogger:
    def __init__(self):
        self.history = {
            "iteration": [],
            "discLoss": [],
            "discAcc": [],
            "policyLoss": [],
            "valueLoss": [],
            "entropy": [],
        }

    def log(self, iteration: int, discLoss: float, discAcc: float, ppoStats: dict):
        self.history["iteration"].append(iteration)
        self.history["discLoss"].append(discLoss)
        self.history["discAcc"].append(discAcc)
        self.history["policyLoss"].append(ppoStats["policyLoss"])
        self.history["valueLoss"].append(ppoStats["valueLoss"])
        self.history["entropy"].append(ppoStats["entropy"])

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=4)

        print(f"\n[SYSTEM] Training metrics saved to '{filepath}'")
