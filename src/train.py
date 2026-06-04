"""To run the training script, execute `python -W ignore -m src.train`"""

import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from core.env import Environment

# ---- User-defined Models ----
from core.model.dataset import getExpertDataloader
from core.model.modules import Actor, Critic
from core.model.params import PPO, BClone, Paths
from core.model.ppo import RolloutBuffer, updatePPO

"""To watch the GPU load, execute `watch -n 0.1 nvidia-smi`"""


def main():
    # -------------------- Load Params --------------------
    paths = Paths()
    # * For a single PPO run
    testID: str = paths.getCurrentTest()
    paths.setTest(testID)
    print(f"[SYSTEM::PPO] Training for Test {testID}")
    
    params: dict[str, PPO | BClone] = {
        "ppo": PPO(paths),
        "bclone": BClone(paths)
    }
    
    # Check the test directories exists, if not create them
    paths.ensure(params["ppo"].WEIGHTS, params["ppo"].METRICS)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM::PPO] Using device: {str(DEVICE).upper()}")

    # -------------------- Initialize Environment --------------------
    # * Change the "render" to speed up training (disables visualization)
    env = Environment(
        useExpert=False,    
        render=False,
        stepsMax=1000
    )
    # * PPO training for a single target waypoint
    env.cycleWaypoints = False

    # -------------------- Initialize Networks --------------------
    # TODO: Experiment on different values for hidden dimensions
    policy = Actor(dimState=10, dimAction=2, dimHidden=params["ppo"].HIDDEN_DIMS).to(DEVICE)
    critic = Critic(dimState=10, dimHidden=params["ppo"].HIDDEN_DIMS).to(DEVICE)

    # ----------- Load Pre-Trained Behavior Clone Weights -----------
    if params["bclone"].WEIGHTS.exists():
        print(f"[SYSTEM] Loading  Behavior Clone weights: {params["bclone"].WEIGHTS}")
        ckpt = torch.load(params["bclone"].WEIGHTS, map_location=DEVICE)
        policy.load_state_dict(ckpt["policy"])
        print(f"[SYSTEM] Weights loaded successfully (Val Loss: {ckpt['lossVal']:.5f})")

    # -------------------- Initialize Optimizers --------------------
    optimizerPolicy = optim.Adam(policy.parameters(), lr=params["ppo"].LR)
    optimizerCritic = optim.Adam(critic.parameters(), lr=params["ppo"].LR)

    # -------------------- Initialize Rollout Buffer --------------------
    buffer = RolloutBuffer(
        sBuffer=params["ppo"].BUFFER_SIZE,
        dimState=10,
        dimAction=2,
        gamma=params["ppo"].GAMMA,
        lambdaGAE=params["ppo"].LAMBDA_GAE,
        device=DEVICE,
    )
    
    # -------------------- Load Expert Data --------------------
    eLoader = getExpertDataloader(paths.EXPERT, params["ppo"].BATCH_SIZE)

    # The progress bar
    pbar = tqdm(
        range(params["ppo"].ITERATIONS),
        desc="[PPO] Training Progress",
        ascii=False,
        ncols=150
    )

    metrics: list[str, ...] = []

    # ---------------------------------------- T R A I N I N G  L O O P ----------------------------------------
    for iteration in pbar:
        buffer.clear()

        # --- 01: ROLLOUT ---
        state, _ = env.reset(options={"randomStart": True}, seed=None)
        stateTensor: torch.Tensor = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
        
        currEpisodeReward: float = 0.0
        episodeRewards: list[float] = []
        done: bool = False

        for step in range(params["ppo"].BUFFER_SIZE):
            with torch.no_grad():
                actionTensor, logProb = policy.forward(stateTensor)

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
            stateTensor = torch.as_tensor(
                state, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)

            if env.render:
                import pygame

                from core.config import Colors, Screen
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        break

                env.screen.fill((Colors.BLACK.R, Colors.BLACK.G, Colors.BLACK.B))
                pygame.draw.rect(
                    env.screen,
                    (Colors.GRAY.R, Colors.GRAY.G, Colors.GRAY.B),
                    (0, 0, Screen.WIDTH, Screen.HEIGHT), 2
                )
                env._drawDrone()
                env._drawHUD()
                
                # Show live PPO stats on screen
                statusText = f"PPO Rollout | Step: {step} | Ep Reward: {currEpisodeReward:.1f}"
                surfaceText = env.font.render(
                    statusText, True,
                    (Colors.YELLOW.R, Colors.YELLOW.G, Colors.YELLOW.B)
                )
                env.screen.blit(surfaceText, (Screen.WIDTH // 2 - 120, 10))
                
                pygame.display.flip()
                env.clock.tick(60)

            if done:
                episodeRewards.append(currEpisodeReward)
                # Reset for the next episode
                currEpisodeReward = 0.0

                state, _ = env.reset(options={"randomStart": True}, seed=None)
                stateTensor = torch.as_tensor(
                    state, dtype=torch.float32, device=DEVICE
                ).unsqueeze(0)

        # --- 02: COMPUTE GAE ---
        with torch.no_grad():
            lastVal = (
                torch.as_tensor(0.0, device=DEVICE)
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
            optPolicy=optimizerPolicy,
            optCritic=optimizerCritic,
            bufferData=bufferData,
            epochs=params["ppo"].EPOCHS,
            sMinibatch=params["ppo"].BATCH_SIZE,
            clipEps=params["ppo"].CLIP_EPSILON,
            coeffValue=params["ppo"].COEFF_VALUE,
            coeffEntropy=params["ppo"].COEFF_ENTROPY,
            loader=eLoader,
            coeffBClone=params["ppo"].COEFF_BCLONE
        )

        # Calculate explained variance for the value function
        #  Helps understanding how well the Critic is learning
        with torch.no_grad():
            predValues = critic(buffer.states[: buffer.ptr])

        returns = buffer.returns[: buffer.ptr]

        # --- 04: LOGGING THE METRICS ---
        varExplained = 1 - torch.var(returns - predValues, unbiased=False) / (torch.var(returns, unbiased=False) + 1e-8)

        avgReward = (
            sum(episodeRewards) / len(episodeRewards)
            if episodeRewards
            else currEpisodeReward
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
            "Loss:Actor": ppoStats["policyLoss"],
            "Loss:Value": ppoStats["valueLoss"],
            "Entropy": ppoStats["entropy"],
            "Explained Variance": varExplained.item(),
            "Buffer Size": buffer.ptr,
        })


    # -------------------- Save the Model --------------------
    checkpoint = {
        "policy": policy.state_dict(),
        "critic": critic.state_dict(),
        "optimizerPolicy": optimizerPolicy.state_dict(),
        "optimizerCritic": optimizerCritic.state_dict(),
        "params": vars(params["ppo"]),
    }
    torch.save(checkpoint, params["ppo"].WEIGHTS)
    # Save the training metrics to a CSV file
    dfMetrics = pd.DataFrame(metrics)
    dfMetrics.to_csv(params["ppo"].METRICS, index=False)

    print(f"\n[SYSTEM:::PPO] Training complete. Actor saved to '{params["ppo"].WEIGHTS}'.")


if __name__ == "__main__":
    main()
