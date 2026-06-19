"""Batch experiment runner. Run: python -W ignore -m src.experiments"""

import datetime
import json
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

from core.config import WAYPOINTS
from core.env import Environment
from core.model.bc import BehaviorClone
from core.model.dataset import getExpertDataloader
from core.model.modules import Actor, Critic
from core.model.params import PPO, Paths, Splits
from core.model.ppo import RolloutBuffer, updatePPO

# Initialize Rich Console
console = Console()

EXPERIMENTS: list[dict] = [
    # BC auxiliary coefficient ablation (dimsHidden=128, wp=0)
    {"name": "Baseline",           "dimsHidden": 128, "coeffBC": 0.1, "wpIdx": 0, "sizeBuffer": 4096},
    {"name": "NO BClone Auxiliary","dimsHidden": 128, "coeffBC": 0.0, "wpIdx": 0, "sizeBuffer": 4096},
    {"name": "Strong BClone",      "dimsHidden": 128, "coeffBC": 0.5, "wpIdx": 0, "sizeBuffer": 4096},
    # Network width ablation (coeffBC=0.1, wp=0)
    {"name": "Small Net",          "dimsHidden": 64,  "coeffBC": 0.1, "wpIdx": 0, "sizeBuffer": 4096},
    {"name": "Large Net",          "dimsHidden": 256, "coeffBC": 0.1, "wpIdx": 0, "sizeBuffer": 4096},
    # Rollout buffer size ablation (dimsHidden=128, coeffBC=0.1, wp=0)
    {"name": "Small Buffer",       "dimsHidden": 128, "coeffBC": 0.1, "wpIdx": 0, "sizeBuffer": 2048},
    {"name": "Large Buffer",       "dimsHidden": 128, "coeffBC": 0.1, "wpIdx": 0, "sizeBuffer": 8192},
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resetToWaypoint(env: Environment, wpIdx: int) -> object:
    state, _ = env.reset(options={"randomStart": True}, seed=None)
    if wpIdx != 0:
        env.currWaypointIdx = wpIdx
        env._waypointBonusGiven = False
        env.targetWindDirection, env.targetWindSpeed = env.waypointWinds[wpIdx]
        env.windDirection = env.targetWindDirection
        env.windSpeed = env.targetWindSpeed
        state = env._normalizeState(env.drone.state)
    return state


def _saveConfig(testDir: Path, exp: dict, expertPath: Path) -> None:
    config = {
        "experiment": {
            "name": exp["name"],
            "test": testDir.name,
            "timestamp": datetime.datetime.now().isoformat(),
            "waypoints": {
                "index": exp["wpIdx"],
                "coordinate": list(WAYPOINTS[exp["wpIdx"]])
            }
        },
        "bclone": {
            "dimsHidden": exp["dimsHidden"],
            "sizeBatch": 512,
            "epochs": 25,
            "lr": 2.5e-6,
            "expertPath": str(expertPath),
        },
        "ppo": {
            "dimsHidden": exp["dimsHidden"],
            "sizeBuffer": exp["sizeBuffer"],
            "gamma": 0.995,
            "lambdaGAE": 0.95,
            "epochs": 4,
            "sizeBatch": 512,
            "clipEpsilon": 0.2,
            "coefficients": {
                "value": 1.0,
                "entropy": 0.005,
                "bclone": exp["coeffBC"]
            },
            "lr": 1.25e-4,
            "iterations": 1000,
        },
    }
    configPath = testDir / "config.json"
    configPath.write_text(json.dumps(config, indent=2))
    console.print(f"[bold yellow] [CONFIG][/bold yellow] Saved to [dim]'{configPath}'[/dim]")


def _runBC(exp: dict, testID: str, expertPath: Path) -> Path:
    trainer = BehaviorClone(
        test=testID,
        pathExpert=str(expertPath),
        batchSize=512,
        epochs=25,
        lr=2.5e-6,
        splits=Splits(),
        targetIdx=exp["wpIdx"],
        hiddenDims=exp["dimsHidden"],
    )
    trainer.train()
    bcWeights: Path = trainer.params["bclone"].WEIGHTS
    trainer.save(str(bcWeights))
    trainer.saveMetrics(str(trainer.params["bclone"].METRICS))
    return bcWeights


def _runPPO(exp: dict, testID: str, expertPath: Path, bcWeights: Path) -> None:
    paths = Paths()
    paths.setTest(testID)
    ppoParams = PPO(paths)

    env = Environment(useExpert=False, render=False, stepsMax=1000)
    env.cycleWaypoints = False

    policy = Actor(dimState=10, dimAction=2, dimHidden=exp["dimsHidden"]).to(DEVICE)
    critic = Critic(dimState=10, dimHidden=exp["dimsHidden"]).to(DEVICE)

    if bcWeights.exists():
        ckpt = torch.load(bcWeights, map_location=DEVICE)
        policy.load_state_dict(ckpt["policy"])
        console.print(f"[bold magenta] [PPO][/bold magenta] Loaded BClone weights (Validation Loss: [green]{ckpt['lossVal']:.5f}[/green])")

    optPolicy = optim.Adam(policy.parameters(), lr=ppoParams.LR)
    optCritic = optim.Adam(critic.parameters(), lr=ppoParams.LR)

    buffer = RolloutBuffer(
        sBuffer=exp["sizeBuffer"],
        dimState=10,
        dimAction=2,
        gamma=ppoParams.GAMMA,
        lambdaGAE=ppoParams.LAMBDA_GAE,
        device=DEVICE,
    )

    eLoader = getExpertDataloader(str(expertPath), ppoParams.BATCH_SIZE, targetIdx=exp["wpIdx"])

    metrics = []
    pbar = tqdm(range(ppoParams.ITERATIONS), desc=f"[PPO:{exp['name']}]", ascii=False, ncols=150)

    for iteration in pbar:
        buffer.clear()
        state = _resetToWaypoint(env, exp["wpIdx"])
        stateTensor = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        currEpisodeReward = 0.0
        episodeRewards: list[float] = []
        done = False

        for _ in range(exp["sizeBuffer"]):
            with torch.no_grad():
                actionTensor, logProb = policy.forward(stateTensor)

            npAction = actionTensor.squeeze(0).cpu().numpy()
            nextState, reward, terminated, truncated, _ = env.step(npAction)
            done = terminated or truncated

            buffer.store(
                state=stateTensor.squeeze(0),
                action=actionTensor.squeeze(0),
                reward=reward,
                done=float(done),
                probs=logProb.squeeze(0),
            )
            currEpisodeReward += reward
            stateTensor = torch.as_tensor(nextState, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            if done:
                episodeRewards.append(currEpisodeReward)
                currEpisodeReward = 0.0
                state = _resetToWaypoint(env, exp["wpIdx"])
                stateTensor = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            lastVal = (
                torch.zeros((), device=DEVICE)
                if done
                else critic(stateTensor).squeeze(0)
            )
            values = critic(buffer.states[: buffer.ptr])

        buffer.computeAdvantages(values, lastVal, done)
        bufferData = buffer.getTensors()

        ppoStats = updatePPO(
            policy=policy,
            critic=critic,
            optPolicy=optPolicy,
            optCritic=optCritic,
            bufferData=bufferData,
            epochs=ppoParams.EPOCHS,
            sMinibatch=ppoParams.BATCH_SIZE,
            clipEps=ppoParams.CLIP_EPSILON,
            coeffValue=ppoParams.COEFF_VALUE,
            coeffEntropy=ppoParams.COEFF_ENTROPY,
            loader=eLoader,
            coeffBClone=exp["coeffBC"],
        )

        with torch.no_grad():
            predValues = critic(buffer.states[: buffer.ptr])
        returns = buffer.returns[: buffer.ptr]
        varExpl = 1 - torch.var(returns - predValues, unbiased=False) / (
            torch.var(returns, unbiased=False) + 1e-8
        )
        avgReward = sum(episodeRewards) / len(episodeRewards) if episodeRewards else currEpisodeReward

        pbar.set_postfix(
            LPol=f"{ppoStats['policyLoss']:.3f}",
            LVal=f"{ppoStats['valueLoss']:.3f}",
            Ent=f"{ppoStats['entropy']:.3f}",
            Reward=f"{avgReward:.2f}",
        )

        metrics.append({
            "Iter": iteration,
            "Average Reward": avgReward,
            "Loss:Actor": ppoStats["policyLoss"],
            "Loss:Value": ppoStats["valueLoss"],
            "Entropy": ppoStats["entropy"],
            "Explained Variance": varExpl.item(),
            "Buffer Size": buffer.ptr,
        })

    torch.save(
        {
            "policy": policy.state_dict(),
            "critic": critic.state_dict(),
            "optPolicy": optPolicy.state_dict(),
            "optCritic": optCritic.state_dict(),
            "params": {
                "dimsHidden": exp["dimsHidden"],
                "coeffBC": exp["coeffBC"],
                "wpIdx": exp["wpIdx"],
            },
        },
        ppoParams.WEIGHTS,
    )
    pd.DataFrame(metrics).to_csv(ppoParams.METRICS, index=False)
    console.print(f"\n[bold magenta] [PPO][/bold magenta] Complete → [dim]{ppoParams.WEIGHTS}[/dim]")


def main():
    console.print(Panel.fit(
        f"[bold blue]Device:[/bold blue] {DEVICE.type.upper()}\n"
        f"[bold blue]Running:[/bold blue] {len(EXPERIMENTS)} queued experiments",
        title="[bold green] Batch Experiment Runner [/bold green]",
        border_style="green"
    ))

    for i, exp in enumerate(EXPERIMENTS, start=1):
        console.print("\n")
        # Experiment configuration panel
        console.print(Panel(
            f"[bold]dimsHidden:[/bold] {exp['dimsHidden']} | "
            f"[bold]coeff\\[BClone]:[/bold] {exp['coeffBC']} | "
            f"[bold]wpIdx:[/bold] {exp['wpIdx']} for {WAYPOINTS[exp['wpIdx']]}",
            title=f"[bold cyan] Experiment {i}/{len(EXPERIMENTS)}: {exp['name']}[/bold cyan]",
            border_style="cyan"
        ))

        paths = Paths()
        paths._TESTS = paths._ROOT / "tests" / "experiments"
        expertPath = paths.EXPERT

        paths.createTest()
        # Use "experiments/<id>" so every Paths().setTest(testID) resolves to
        #   tests/experiments/<id>, including the one created inside BehaviorClone.
        testID = f"experiments/{paths.TEST.name}"
        testDir = paths._ROOT / "tests" / testID
        testDir.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold dim] Test directory:[/bold dim]  @ tests/{testID}/")

        _saveConfig(testDir, exp, expertPath)

        console.rule("[bold blue] TRAINING: Behavior Cloning[/bold blue]")
        bcWeights = _runBC(exp, testID, expertPath)

        console.rule("[bold magenta] TRAINING: PPO[/bold magenta]")
        _runPPO(exp, testID, expertPath, bcWeights)

        console.print(f"\n[bold green] DONE:[/bold green] '{exp['name']}' @ tests/{testID}/")

    console.print("\n")
    console.rule("[bold green] [EXPERIMENT] All experiment are completed! [/bold green]")


if __name__ == "__main__":
    main()