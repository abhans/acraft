import numpy as np
import torch
import torch.nn as nn
from torch import distributions


class Critic(nn.Module):
    def __init__(self, dimState=8, dimHidden=128):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dimState, dimHidden),
            nn.Tanh(),
            nn.Linear(dimHidden, dimHidden),
            nn.Tanh(),
            nn.Linear(dimHidden, 1),
        )

        self._initWeights()

    def _initWeights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # Final layer for value output usually has a gain of 1
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)

    def forward(self, state):
        # To ensure the shape is (batch_size,)
        return self.net(state).squeeze(-1)


class Policy(nn.Module):
    def __init__(self, dimState=8, dimAction=2, dimHidden=128):
        super(Policy, self).__init__()

        # Shared feature extractor
        self.nnFeature = nn.Sequential(
            nn.Linear(dimState, dimHidden),
            nn.Tanh(),
            nn.Linear(dimHidden, dimHidden),
            nn.Tanh(),
        )

        # Mean network (outputs unbounded real numbers)
        self.nnMean = nn.Linear(dimHidden, dimAction)
        # Log standard deviation network (unbounded real number for numerical stability)
        self.logStd = nn.Parameter(torch.ones(dimAction) * -0.5)

        self._initWeights()

    def _initWeights(self):
        for module in self.nnFeature.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # Initialize "Mean" layer with smaller gain to 
        #  prevent large initial actions
        nn.init.orthogonal_(self.nnMean.weight, gain=0.01)
        nn.init.constant_(self.nnMean.bias, 0.0)

    def forward(self, state, eps: float = 1e-5):
        """
        Given state St, returns action At.
        """
        features = self.nnFeature(state)
        mean = self.nnMean(features)
        logStd = self.logStd.clamp(-5, 2)
        # Standard deviation must be strictly positive
        std = torch.exp(logStd).expand_as(mean)

        dist = distributions.Normal(mean, std)
        action = dist.rsample()
        # Squash the Gaussian distribution through a Sigmoid to bound actions strictly to [0, 1]
        # This mathematically maps the real line (-inf, inf) to (0, 1)
        fAction = torch.sigmoid(action).clamp(eps, 1.0 - eps)
        logProb = dist.log_prob(action)

        # Use the Jacobian of the Sigmoid
        #   log |det(J)| = log(action * (1 - action))
        logProb -= torch.log(
            fAction.clamp(eps, 1.0 - eps) * (1.0 - fAction.clamp(eps, 1.0 - eps))
        )
        # Get the sum over dimensions for joint log probability
        logProb = logProb.sum(dim=-1)

        return (fAction, logProb)

    def evaluate(self, states, actions, eps: float = 1e-5):
        """
        PPO requires log probabilities of taken actions and the policy's entropy.
        """
        features = self.nnFeature(states)
        mean = self.nnMean(features)
        logStd = self.logStd.clamp(-5, 2)
        std = torch.exp(logStd).expand_as(mean)

        dist = distributions.Normal(mean, std)
        pActions = torch.logit(actions.clamp(eps, 1.0 - eps))

        # Calculate log probability of the action taken, according to our squashed Gaussian
        logProb = dist.log_prob(pActions)

        # Sigmoid Jacobian correction
        logProb -= torch.log(
            actions.clamp(eps, 1.0 - eps) 
            * (1.0 - actions.clamp(eps, 1.0 - eps))
        )

        logProb = logProb.sum(dim=-1)

        # Calculate the entropy
        entropy = dist.entropy().sum(dim=-1)

        return logProb, entropy
