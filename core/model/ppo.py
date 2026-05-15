import torch
import torch.nn.functional as F
from torch import nn


class RolloutBuffer:
    def __init__(
        self,
        sBuffer: int,
        dimState: int = 6,
        dimAction: int = 2,
        gamma: float = 0.99,
        lambdaGAE: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        self.sBuffer = sBuffer
        self.gamma = gamma
        self.lambdaGAE = lambdaGAE
        self.ptr = 0

        self.states = torch.zeros(
            (sBuffer, dimState), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros(
            (sBuffer, dimAction), dtype=torch.float32, device=device
        )
        self.rewards = torch.zeros(sBuffer, dtype=torch.float32, device=device)
        self.dones = torch.zeros(sBuffer, dtype=torch.float32, device=device)

        self.advantages = torch.zeros(sBuffer, dtype=torch.float32, device=device)
        self.returns = torch.zeros(sBuffer, dtype=torch.float32, device=device)

        self.oldProbs = torch.zeros(sBuffer, dtype=torch.float32, device=device)

    def store(self, state, action, reward, done, probs):
        if self.ptr >= self.sBuffer:
            raise RuntimeError("Rollout buffer is full.")
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.oldProbs[self.ptr] = probs
        # Increment the pointer
        self.ptr += 1

    def computeAdvantages(self, values: torch.Tensor, lastVal: float, doneLast: bool):
        """
        Computes GAE.
        'values' should be the Critic's predictions for self.states[:self.ptr]
        """
        lastGAE = 0.0

        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                nextNonterminal = 1.0 - doneLast
                nextVal = lastVal
            else:
                nextNonterminal = 1.0 - self.dones[step + 1]
                nextVal = values[step + 1]

            # Subtract the value of the current state
            delta = (
                self.rewards[step]
                + self.gamma * nextVal * nextNonterminal
                - values[step]
            )

            lastGAE = delta + self.gamma * self.lambdaGAE * nextNonterminal * lastGAE
            self.advantages[step] = lastGAE

            # Return Advantage + Value
            self.returns[step] = lastGAE + values[step]

    def getTensors(self):
        return (
            self.states[: self.ptr],
            self.actions[: self.ptr],
            self.advantages[: self.ptr],
            self.returns[: self.ptr],
            self.oldProbs[: self.ptr],
        )

    def clear(self):
        self.ptr = 0


def updatePPO(
    policy: nn.Module,
    critic: nn.Module,
    optimizerPolicy: torch.optim.Optimizer,
    optimizerCritic: torch.optim.Optimizer,
    bufferData: tuple,
    epochs: int = 4,
    sMinibatch: int = 64,
    clipEpsilon: float = 0.2,
    coeffValue: float = 0.5,
    coeffEntropy: float = 0.01,
):
    """
    Performs the PPO update step.
    """
    states, actions, advantages, returns, oldProbs = bufferData

    # Normalize Advantages to keep training stable
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    totalPolicyLoss = 0.0
    totalValueLoss = 0.0
    totalEntropy = 0.0

    # PPO Epochs
    for _ in range(epochs):
        # Generate random indices for shuffling
        indices = torch.randperm(len(states), device=states.device)

        # Mini-batches
        for start in range(0, len(states), sMinibatch):
            end = start + sMinibatch

            # Skip incomplete mini-batches
            if end > len(states):
                break

            # Data to be processed in minibatches
            mbIndices = indices[start:end]
            mbStates = states[mbIndices]
            mbActions = actions[mbIndices]
            mbAdvantages = advantages[mbIndices]
            mbReturns = returns[mbIndices]
            mbOldProbs = oldProbs[mbIndices]

            # ------------------- POLICY GRADIENT -------------------
            # Evaluate new log probs and entropy based on current policy weights
            newProbs, entropy = policy.evaluate(mbStates, mbActions)

            # Calculate the ratio: exp(log_pi_new - log_pi_old)
            ratio = torch.exp(newProbs - mbOldProbs)

            # PPO Clipped Surrogate Objective (Eq. 7)
            surrogate1 = ratio * mbAdvantages
            surrogate2 = (
                torch.clamp(ratio, 1.0 - clipEpsilon, 1.0 + clipEpsilon) * mbAdvantages
            )

            # We want to MAXIMIZE this, so we take the min and add a negative sign for the optimizer
            policyLoss = -torch.min(surrogate1, surrogate2).mean()

            # ------------------- VALUE GRADIENT -------------------
            # Critic predicts the value of the state
            predVals = critic(mbStates)

            # Standard Mean Squared Error between predicted value and actual returns
            valueLoss = F.mse_loss(predVals, mbReturns)

            # ------------------- ENTROPY BONUS -------------------
            # This directly corresponds to the -λ * H(π)
            # We want to MAXIMIZE entropy, so we subtract it from the loss
            entropyMean = entropy.mean()

            # ------------------- BACKPROPAGATION -------------------
            # Total PPO Loss
            loss = policyLoss + coeffValue * valueLoss - coeffEntropy * entropyMean

            optimizerPolicy.zero_grad()
            optimizerCritic.zero_grad()

            loss.backward()

            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)

            optimizerPolicy.step()
            optimizerCritic.step()

            # Save the total losses and entropy
            totalPolicyLoss += policyLoss.item()
            totalValueLoss += valueLoss.item()
            totalEntropy += entropyMean.item()

    # Return averages over the entire update
    numUpdates = epochs * (len(states) // sMinibatch)
    return {
        "policyLoss": totalPolicyLoss / numUpdates,
        "valueLoss": totalValueLoss / numUpdates,
        "entropy": totalEntropy / numUpdates,
    }
