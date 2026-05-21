"""To run the training script, execute `python -W ignore -m src.train`"""

import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from core.env import Environment

# ---- User-defined Models ----
from core.model.critic import Critic, Policy
from core.model.params import Params
from core.model.ppo import RolloutBuffer, updatePPO

"""To watch the GPU load, execute `watch -n 0.1 nvidia-smi`"""


def main():
    # -------------------- Load Params --------------------
    params = Params()
    params.PATHS.ensureDirs()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Using device: {DEVICE}")

    # -------------------- Initialize Environment --------------------
    env = Environment(useExpert=False, render=False)

    # -------------------- Initialize Networks --------------------
    policy = Policy(dimState=8, dimAction=2).to(DEVICE)
    critic = Critic(dimState=8).to(DEVICE)

    # -------------------- Initialize Optimizers --------------------
    optimizerPolicy = optim.Adam(policy.parameters(), lr=params.LR)
    optimizerCritic = optim.Adam(critic.parameters(), lr=params.LR)

    # -------------------- Initialize Rollout Buffer --------------------
    buffer = RolloutBuffer(
        sBuffer=params.BUFFER_SIZE,
        dimState=8,
        dimAction=2,
        gamma=params.GAMMA,
        lambdaGAE=params.LAMBDA_GAE,
        device=DEVICE,
    )

    # The progress bar
    pbar = tqdm(
        range(params.ITERATIONS),
        desc="[PPO] Training Progress",
        ascii=False,
    )

    metrics: list[str, ...] = []

    # ---------------------------------------- T R A I N I N G  L O O P ----------------------------------------
    for iteration in pbar:
        buffer.clear()

        # --- 01: ROLLOUT ---
        state, _ = env.reset(options={"randomStart": True})
        stateTensor = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
        
        winReward: int = 20
        currEpisodeReward: float = 0.0
        episodeRewards: list[float] = []
        done: bool = False

        for step in range(params.BUFFER_SIZE):
            with torch.no_grad():
                actionTensor, logProb = policy(stateTensor)

            npAction = actionTensor.squeeze(0).cpu().numpy()
            nextState, reward, terminated, truncated, info = env.step(npAction)
            done = terminated or truncated

            buffer.store(
                state=stateTensor.squeeze(0),
                action=actionTensor.squeeze(0),
                reward=reward,
                done=float(done),
                probs=logProb.squeeze(0),
            )

            currEpisodeReward += reward

            state = nextState
            stateTensor = torch.tensor(
                state, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)

            if done:
                episodeRewards.append(currEpisodeReward)
                # Reset for the next episode
                currEpisodeReward = 0.0

                state, _ = env.reset(options={"randomStart": True})
                stateTensor = torch.tensor(
                    state, dtype=torch.float32, device=DEVICE
                ).unsqueeze(0)

        # --- 02: COMPUTE GAE ---
        with torch.no_grad():
            lastVal = (
                torch.tensor(0.0, device=DEVICE)
                if done
                else critic(stateTensor).squeeze(0)
            )

        with torch.no_grad():
            values = critic(buffer.states[: buffer.ptr])

        buffer.computeAdvantages(values, lastVal, done)

        # --- 03: UPDATE POLICY & CRITIC w/PPO ---
        bufferData = buffer.getTensors()
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

        # Calculate explained variance for the value function
        #  Helps understanding how well the Critic is learning
        with torch.no_grad():
            predValues = critic(buffer.states[: buffer.ptr])

        returns = buffer.returns[: buffer.ptr]
        varExplained = 1 - torch.var(returns - predValues, unbiased=False) / (torch.var(returns, unbiased=False) + 1e-8)


        # --- 04: LOGGING THE METRICS ---
        avgReward = (
            sum(episodeRewards) / len(episodeRewards)
            if episodeRewards
            else currEpisodeReward
        )

        recentRewards = episodeRewards[-winReward:]
        movingAvgReward = (
            sum(recentRewards) / len(recentRewards)
            if recentRewards
            else avgReward
        )

        pbar.set_postfix(
            LPol=f"{ppoStats['policyLoss']:.3f}",
            LVal=f"{ppoStats['valueLoss']:.3f}",
            Ent=f"{ppoStats['entropy']:.3f}",
            Reward=f"{avgReward:.2f}",
        )

        # Add metrics to the list for saving later
        metrics.append({
            "Iter": iteration,
            "Average Reward": avgReward,
            "Loss:Policy": ppoStats["policyLoss"],
            "Loss:Value": ppoStats["valueLoss"],
            "Entropy": ppoStats["entropy"],
            "Explained Variance": varExplained.item(),
            "Moving Average Reward": movingAvgReward,
            "Buffer Size": buffer.ptr,
        })


    # -------------------- Save the Model --------------------
    checkpoint = {
        "policy": policy.state_dict(),
        "critic": critic.state_dict(),
        "optimizerPolicy": optimizerPolicy.state_dict(),
        "optimizerCritic": optimizerCritic.state_dict(),
        "params": vars(params),
    }
    torch.save(checkpoint, params.PATHS.WEIGHTS)
    # Save the training metrics to a CSV file
    dfMetrics = pd.DataFrame(metrics)
    dfMetrics.to_csv(params.PATHS.METRICS, index=False)

    print(f"\n[SYSTEM] Training complete. Policy saved to '{params.PATHS.WEIGHTS}'.")


if __name__ == "__main__":
    main()
