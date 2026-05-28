import numpy as np
import torch
import torch.nn as nn
from torch import distributions


class Critic(nn.Module):
    def __init__(self, dimState=10, dimHidden=256):
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
        """
        Initialize weights of the critic network using orthogonal initialization.
        
        The gain is set to ``sqrt(2)`` for hidden layers to maintain variance,
        and a gain of 1 for the final layer to ensure value outputs are on a reasonable scale at the start of training.
        This helps stabilize learning in the early stages.
        
        Biases are initialized to zero for all layers.

        Orthogonal initialization is chosen for its ability to maintain the 
        variance of activations across layers, which can lead to better convergence properties.
        
        The final layer's gain is set to ``1`` to prevent excessively large initial value estimates, which can destabilize training.
        """
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
    def __init__(self, dimState=10, dimAction=2, dimHidden=256):
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
        self.logStd = nn.Parameter(torch.ones(dimAction) * -2.0)

        self._initWeights()

    def _initWeights(self):
        """
        Initialize weights of the policy network using orthogonal initialization.
        
        The gain is set to ``sqrt(2)`` for hidden layers to maintain variance, 
         and a smaller gain for the mean layer to prevent large initial actions.

        The ``logStd`` is initialized to -0.5, which corresponds to 
         a standard deviation of approximately 0.6, providing a reasonable starting exploration level.
        """
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

        :param state: Input state tensor of shape ``(sBatch, dimState)``
        :param eps: Small constant for numerical stability in log calculations
        :return: Tuple of (action, log probability) where:

            - action: Tensor of shape ``(sBatch, dimAction)`` with values in ``[0, 1]``
            - log probability: Tensor of shape ``(sBatch,)`` representing the log probability of the sampled action under the policy
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
        fAction = torch.sigmoid(action)
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

        :param states: Tensor of shape ``(sBatch, dimState)``
        :param actions: Tensor of shape ``(sBatch, dimAction)`` with values in ``[0, 1]``
        :param eps: Small constant for numerical stability in log calculations
        :return: Tuple of (log probabilities, entropies) where:

            - log probabilities: Tensor of shape ``(sBatch,)`` representing the log probability of the given actions under the current policy
            - entropies: Tensor of shape ``(sBatch,)`` representing the entropy of the policy's action distribution for each state
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

        # Sigmoid Jacobian correction
        entropy += torch.log(
            actions.clamp(eps, 1.0 - eps) * (1.0 - actions.clamp(eps, 1.0 - eps))
        ).sum(dim=-1)

        return logProb, entropy
