"""To run the training script, execute `python -W ignore -m src.train`"""

import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from core.env import Environment
from core.logger import GAILLogger

# ---- User-defined Models ----
from core.model.critic import Critic, Policy
from core.model.gail import Discriminator, reward
from core.model.params import Params
from core.model.ppo import RolloutBuffer, updatePPO


def loadExpertData(filepath: str, device: torch.device):
    """Loads and flattens expert .npz file into state/action tensors."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Expert data not found at '{filepath}'. Run collector.py first!"
        )

    data = np.load(filepath, allow_pickle=True)
    trajectories = data["trajectories"]
    allTransitions = np.concatenate(trajectories, axis=0)

    states = torch.tensor(allTransitions[:, :6], dtype=torch.float32, device=device)
    actions = torch.tensor(allTransitions[:, 6:], dtype=torch.float32, device=device)

    return states, actions


"""To watch the GPU load, execute `watch -n 0.1 nvidia-smi`"""


def main():
    # -------------------- Load Params --------------------
    params = Params()
    params.PATHS.ensureDirs()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Using device: {DEVICE}")

    # -------------------- Load Expert Data --------------------
    expStates, expActions = loadExpertData(params.PATHS.EXPERT, DEVICE)
    print(f"[DATA] Loaded {expStates.shape[0]} expert transitions.")

    # -------------------- Initialize Environment --------------------
    env = Environment(useExpert=False, render=False)

    # -------------------- Initialize Networks --------------------
    policy = Policy(dimState=6, dimAction=2).to(DEVICE)
    critic = Critic(dimState=6).to(DEVICE)
    discriminator = Discriminator(dimState=6, dimAction=2).to(DEVICE)

    # -------------------- Initialize Optimizers --------------------
    optimizerPolicy = optim.Adam(policy.parameters(), lr=params.LR)
    optimizerCritic = optim.Adam(critic.parameters(), lr=params.LR)
    optimizerDisc = optim.Adam(discriminator.parameters(), lr=params.LR_DISCRIMINATOR)

    # -------------------- Initialize Rollout Buffer --------------------
    buffer = RolloutBuffer(
        sBuffer=params.BUFFER_SIZE,
        dimState=6,
        dimAction=2,
        gamma=params.GAMMA,
        lambdaGAE=params.LAMBDA_GAE,
        device=DEVICE,
    )

    # -------------------- Initialize Logger --------------------
    logger = GAILLogger()

    # The progress bar
    pbar = tqdm(
        range(params.ITERATIONS),
        desc="[GAIL] Training Progress",
        ascii=False,
    )
    # ---------------------------------------- T R A I N I N G  L O O P ----------------------------------------
    for iteration in pbar:
        buffer.clear()

        # --- 01: ROLLOUT ---
        state, _ = env.reset(options={"randomStart": True})
        stateTensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(
            0
        )

        for step in range(params.BUFFER_SIZE):
            with torch.no_grad():
                actionTensor, logProb = policy(stateTensor)

            npAction = actionTensor.squeeze(0).cpu().numpy()
            nextState, _, terminated, truncated, _ = env.step(npAction)
            done = terminated or truncated

            rRaw = reward(stateTensor, actionTensor, discriminator)
            rValue = rRaw.squeeze(0).item()

            buffer.store(
                state=stateTensor.squeeze(0),
                action=actionTensor.squeeze(0),
                reward=rValue,
                done=float(done),
                probs=logProb.squeeze(0),
            )

            state = nextState
            stateTensor = torch.tensor(
                state, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)

            if done:
                state, _ = env.reset()
                stateTensor = torch.tensor(
                    state, dtype=torch.float32, device=DEVICE
                ).unsqueeze(0)

        # --- 02: COMPUTE GAE ---
        lastVal = 0.0 if done else critic(stateTensor).item()

        # In-place batch normalization
        batchRewards = buffer.rewards[: buffer.ptr]
        buffer.rewards[: buffer.ptr] = (batchRewards - batchRewards.mean()) / (
            batchRewards.std() + 1e-8
        )

        with torch.no_grad():
            values = critic(buffer.states[: buffer.ptr])

        buffer.computeAdvantages(values, lastVal, done)

        # --- 03: UPDATE DISCRIMINATOR ---
        bufferData = buffer.getTensors()
        learnerStates, learnerActions = bufferData[0], bufferData[1]

        # Update Discriminator multiple times per Generator update
        #   to prevent it from becoming too accurate, too fast.
        for _ in range(params.UPDATES.DISCRIMINATOR):
            expIndices = torch.randperm(expStates.shape[0], device=DEVICE)[: buffer.ptr]
            expStatesBatch = expStates[expIndices]
            expActionsBatch = expActions[expIndices]

            discLoss, discAcc = discriminator.step(
                expStatesBatch, expActionsBatch, learnerStates, learnerActions
            )
            optimizerDisc.zero_grad()
            discLoss.backward()
            optimizerDisc.step()

        # --- 04: UPDATE POLICY & CRITIC w/PPO ---
        for _ in range(params.UPDATES.GENERATOR):
            ppoStats = updatePPO(
                policy=policy,
                critic=critic,
                optimizerPolicy=optimizerPolicy,
                optimizerCritic=optimizerCritic,
                bufferData=bufferData,
                epochs=params.PPO_EPOCHS,
                sMinibatch=params.BATCH_SIZE,
                clipEpsilon=params.CLIP_EPSILON,
                coeffValue=params.COEFF_VALUE,
                coeffEntropy=params.COEFF_ENTROPY,
            )

        # --- 05: LOGGING THE METRICS ---
        logger.log(
            iteration=iteration,
            discLoss=discLoss.item(),
            discAcc=discAcc.item(),
            ppoStats=ppoStats,
        )

        pbar.set_postfix(
            ADisc=f"{discAcc.item():.2f}",
            LDisc=f"{discLoss.item():.3f}",
            LPol=f"{ppoStats['policyLoss']:.3f}",
            LVal=f"{ppoStats['valueLoss']:.3f}",
            Ent=f"{ppoStats['entropy']:.3f}",
        )

    # -------------------- Save the Model --------------------
    torch.save(policy.state_dict(), params.PATHS.WEIGHTS)
    logger.save(params.PATHS.METRICS)
    print(f"\n[SYSTEM] Training complete. Policy saved to '{params.PATHS.WEIGHTS}'.")


if __name__ == "__main__":
    main()
