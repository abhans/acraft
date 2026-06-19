import copy

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm

from core.env import Environment
from core.model.dataset import ExpertDataset
from core.model.modules import Actor
from core.model.params import PPO, BClone, Paths, Splits


class BehaviorClone:
    def __init__(
        self,
        test: str,
        splits: Splits,
        pathExpert: str,
        batchSize: int = 512,
        epochs: int = 50,
        lr: float = 3e-4,
        targetIdx: int = 0,
        hiddenDims: int | None = None,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.batchSize = batchSize
        self.epochs = epochs
        # Metrics for tracking training progress
        paths: Paths = Paths()
        paths.setTest(test)
        self.params: dict[str, PPO | BClone] = {
            "ppo": PPO(paths),
            "bclone": BClone(paths)
        }
        self.metrics: list[dict] = []
        # -------------------- Dataset Configuration & Loaders --------------------
        # Load the expert dataset
        dataset = ExpertDataset(pathExpert, targetIdx=targetIdx)

        sValidation = int(len(dataset) * splits.VALIDATION)
        sTrain = len(dataset) - sValidation

        trainDataset, valDataset = random_split(
            dataset,
            [sTrain, sValidation]
        )

        # Data Loaders
        self.trainLoader = torch.utils.data.DataLoader(
            trainDataset,
            batch_size=batchSize,
            shuffle=True,
            drop_last=True,
        )

        self.valLoader = torch.utils.data.DataLoader(
            valDataset,
            batch_size=batchSize,
            shuffle=False,
            drop_last=False,
        )

        # --------------------------- Addition of Actor ---------------------------
        _hiddenDims = hiddenDims if hiddenDims is not None else self.params["bclone"].HIDDEN_DIMS
        self.policy = Actor(
            dimState=10,
            dimAction=2,
            dimHidden=_hiddenDims
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=lr
        )

        # Initiate best validation loss and corresponding weights
        self.lossBest = float("inf")
        self.weightsBest = None

    def _runEpoch(self, training: bool = True, env: Environment | None = None):
        """
        Run a single epoch of training or validation.

        :param training: 
        
        - If True, runs in ``training`` mode (with backpropagation).
        - If False, runs in ``evaluation`` mode (no backpropagation).
        
        :return: Average loss for the epoch.
        """
        loader = self.trainLoader if training else self.valLoader

        totalLoss: float = 0.0

        self.policy.train() if training else self.policy.eval()
        
        # Iterate over batches
        for states, expertActions in loader:
            states = states.to(self.device)
            expertActions = expertActions.to(self.device)

            with torch.set_grad_enabled(training):
                predActions, _ = self.policy.forward(states)

                # Compute MSE loss between predicted and expert actions
                loss = F.huber_loss(
                    predActions,
                    expertActions
                )

                # Backpropagation and optimization step (only during training)
                if training:
                    self.optimizer.zero_grad()

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        max_norm=1.0
                    )

                    self.optimizer.step()

            totalLoss += loss.item()

            # -------------------- VISUALIZATION (Episodes) --------------------
            if env is not None and env.render:
                import pygame

                from core.config import Colors, Physics, Screen

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return totalLoss / max( len(loader), 1 )
                    
                visState = env._normalizeState(env.drone.state)
                tState = torch.as_tensor(visState, dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    tAction = self.policy.actDeterministic(tState).squeeze(0).cpu().numpy()

                # Update the visual environment
                _, _, terminated, truncated, _ = env.step(tAction)

                env.screen.fill((Colors.BLACK.R, Colors.BLACK.G, Colors.BLACK.B))
                pygame.draw.rect(
                    env.screen,
                    (Colors.GRAY.R, Colors.GRAY.G, Colors.GRAY.B),
                    (0, 0, Screen.WIDTH, Screen.HEIGHT), 2
                )

                # Draw the HUD to the screen
                env._drawDrone()
                env._drawHUD()
                
                textMode: str = "BClone Training" if training else "BClone Validation"
                textColor: Colors = (Colors.GREEN.R, Colors.GREEN.G, Colors.GREEN.B) \
                    if training else (Colors.YELLOW.R, Colors.YELLOW.G, Colors.YELLOW.B)
                
                surfaceText = env.font.render(
                    textMode, True, 
                    textColor
                )
                env.screen.blit(surfaceText, (Screen.WIDTH // 2 - 60, 10))

                pygame.display.flip()
                env.clock.tick(Physics.FPS)

                # Reset the environment to keep it in sync
                if terminated or truncated:
                    env.reset(options={"randomStart": False})

        return totalLoss / len(loader)

    def train(self, env: Environment | None = None):
        """
        Train the policy using behavior cloning on the expert dataset.
        """
        pbar = tqdm(
            range(self.epochs),
            desc="[BCLONE] Training",
            unit="epoch",
            ascii=False,
            ncols=150
        )

        # Iterate over epochs
        for epoch in pbar:
            lossTrain = self._runEpoch(training=True, env=env)
            lossVal = self._runEpoch(training=False, env=env)

            # Update best validation loss and corresponding weights
            if lossVal < self.lossBest:
                self.lossBest = lossVal
                self.weightsBest = copy.deepcopy(
                    self.policy.state_dict()
                )
        
            pbar.set_postfix(
                Train=f"{lossTrain:.5f}",
                Val=f"{lossVal:.5f}",
            )
            # Add metrics to the list for saving later
            self.metrics.append({   
                "Epoch": epoch + 1,
                "Loss:Train": lossTrain,
                "Loss:Validation": lossVal,
                "Loss:Best:Validation": self.lossBest,
            })

        print(
            f"\n[BCLONE] Best Validation Loss: "
            f"{self.lossBest:.6f}"
        )

    def saveMetrics(self, path):
        df = pd.DataFrame(self.metrics)
        df.to_csv(path, index=False)

        print(f"[BCLONE] Saved metrics to '{path}'")

    def save(self, path):
        torch.save(
            {
                "policy": self.weightsBest,
                "lossVal": self.lossBest,
            },
            path
        )

        print(f"[BCLONE] Saved pretrained policy to '{path}'")