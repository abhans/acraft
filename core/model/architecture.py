import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=128):
        super(Discriminator, self).__init__()
        
        # Input: 8D (6D state + 2D action)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1), # Single logit output
            nn.Sigmoid() # Sigmoid to bound output strictly (0, 1)
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
        expZ = torch.cat([expStates, expActions], dim=-1)
        expPred = self.forward(expZ)
        
        # Learner data (Label = 0 - "This is amateur behavior")
        Z = torch.cat([states, actions], dim=-1)
        pred = self.forward(Z)
        
        # Binary Cross Entropy Loss
        loss = F.binary_cross_entropy(expPred, torch.ones_like(expPred)) + \
                F.binary_cross_entropy(pred, torch.zeros_like(pred))
        
        # Calculate accuracy for logging
        expAccuracy = (expPred > 0.5).float().mean()
        
        return loss, expAccuracy



class Policy(nn.Module):
    def __init__(self, dimState=6, dimAction=2, dimHidden=128):
        super(Policy, self).__init__()
        
        # Shared feature extractor
        self.nnFeature = nn.Sequential(
            nn.Linear(dimState, dimHidden),
            nn.Tanh(),
            nn.Linear(dimHidden, dimHidden),
            nn.Tanh()
        )
        
        # Mean network (outputs unbounded real numbers)
        self.nnMean = nn.Linear(dimHidden, dimAction)
        # Log standard deviation network (unbounded real number for numerical stability)
        self.nnLogStd = nn.Linear(dimHidden, dimAction)

    def forward(self, state):
        """
        Given state St, returns action At.
        """
        features = self.nnFeature(state)
        mean = self.nnMean(features)
        logStd = self.nnLogStd(features)
        
        # Standard deviation must be strictly positive
        std = torch.exp(logStd.clamp(-20, 2))
        
        # Squash the Gaussian distribution through a Sigmoid to bound actions strictly to [0, 1]
        # This mathematically maps the real line (-inf, inf) to (0, 1)
        fMean = torch.sigmoid(mean)
        
        # Create a Tanh-normal distribution to easily calculate log probs and entropy
        # PyTorch handles the variance scaling internally based on our 'std'
        dist = torch.distributions.Normal(fMean, std)
        action = dist.sample()
        
        return action

    def evaluate(self, states, actions):
        """
        PPO requires log probabilities of taken actions and the policy's entropy.
        """
        features = self.nnFeature(states)
        mean = self.nnMean(features)
        logStd = self.nnLogStd(features)
        std = torch.exp(logStd.clamp(-20, 2))
        
        fMean = torch.sigmoid(mean)
        
        # Calculate log probability of the action taken, according to our squashed Gaussian
        logProb = torch.distributions.Normal(fMean, std).log_prob(actions)
        
        # Calculate entropy of the policy: H(pi) = -E[log pi(a|s)]
        entropy = torch.distributions.Normal(fMean, std).entropy()
        
        return logProb, entropy
    
def reward(state, action, discriminator, eps=1e-8):
    """
    The Cost Function is the Discriminator.
    """
    with torch.no_grad():
        Z = torch.cat([state, action], dim=-1)
        output = discriminator(Z)
        reward = -torch.log(output + 1e-8)

        return reward