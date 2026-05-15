import torch
import torch.nn.functional as F
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, dimState=6, dimAction=2, hidden_dim=128):
        super(Discriminator, self).__init__()

        # Input: 8D (6D state + 2D action)
        self.net = nn.Sequential(
            nn.Linear(dimState + dimAction, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),  # Single logit output
        )

    def forward(self, state, action):
        """
        Returns probability between 0 and 1.
        """
        z = torch.cat([state, action], dim=-1)
        return self.net(z)

    def step(self, expStates, expActions, states, actions):
        """
        The Discriminator Gradient Step
        Maximizing Expert[log(D)] + Learner[log(1 - D)]
        """
        # Expert data (Label = 1 - "This is expert behavior")
        expPred = self.forward(expStates, expActions)

        # Learner data (Label = 0 - "This is amateur behavior")
        pred = self.forward(states, actions)

        # Binary Cross Entropy Loss
        loss = F.binary_cross_entropy_with_logits(
            expPred, torch.ones_like(expPred)
        ) + F.binary_cross_entropy_with_logits(pred, torch.zeros_like(pred))

        # Calculate accuracies for logging
        expAcc = (torch.sigmoid(expPred) > 0.5).float().mean()
        learnerAcc = (torch.sigmoid(pred) < 0.5).float().mean()
        # Total accuracy (50% is better)
        totalAcc = (expAcc + learnerAcc) / 2.0

        return loss, totalAcc


def reward(
    state: torch.Tensor,
    action: torch.Tensor,
    discriminator: Discriminator,
    eps: float = 1e-8,
):
    """
    The Cost Function of the Discriminator.
    """
    with torch.no_grad():
        logits = discriminator(state, action)
        output = torch.sigmoid(logits)
        reward = torch.log(output + eps)

    return reward
