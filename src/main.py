import argparse
import os

import numpy as np
import pygame
import torch

from core.config import Colors, Physics, Screen
from core.env import Environment
from core.model.modules import Actor
from core.model.params import PPO, Paths

parser = argparse.ArgumentParser(description="Evaluate and log the trained drone policy.")
parser.add_argument(
    "-e", "--episodes",
    type=int, 
    default=10, 
    help="Number of episodes to record data for."
)
args = parser.parse_args()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

isRunning: bool = True

paths: Paths = Paths()
testID: str = paths.getCurrentTest()
print(f"[TEST] Test: {testID}")
paths.setTest(testID)

params = PPO(paths)

# Load the trained Actor
ckpt = torch.load(
    params.WEIGHTS,
    map_location='cpu',
    weights_only=False
)


policy: Actor = Actor(
    dimState=10,
    dimAction=2,
    # * Change according to the saved policy dimensions
    dimHidden=params.HIDDEN_DIMS
).to(DEVICE)

policy.load_state_dict(ckpt["policy"])
policy.eval()

environment: Environment = Environment(
    useExpert=False,
    render=True,
    stepsMax=1000
)

currState, _ = environment.reset(options={"randomStart": True})

episodeCount: int = 1
stepCount: int = 0
telemetryData: list = []

while isRunning:
    if environment.render:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                isRunning = False

    tState: torch.Tensor = torch.tensor(
        currState,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    with torch.no_grad():
        tAction, _ = policy(tState)

    # Get actions as an array
    action = tAction.squeeze(0).cpu().numpy()

    # Update the environment
    nextState, reward, terminated, truncated, info = environment.step(action)
    isDone: bool = terminated or truncated

    if args.episodes > 0:
        # Extract the state from the environment info dictionary
        vState = info.get("rawState", environment.drone.state)
        targetX, targetY = environment.waypoints[environment.currWaypointIdx]
        
        # Calculate the absolute distance to the target
        distance = np.hypot(
            vState[0] - targetX,
            vState[1] - targetY
        )

        telemetryData.append({
            "Episode": episodeCount,
            "Step": stepCount,
            # ------ State Values ------
            "X": vState[0],
            "Y": vState[1],
            "Distance": distance,
            "Vx": vState[2],
            "Vy": vState[3],
            "Theta": vState[4],
            "Omega": vState[5],
            "Wx": vState[6],
            "Wy": vState[7],
            # ------ Actions and Reward ------
            "LeftTh": action[0],
            "RightTh": action[1],
            "Reward": reward,
        })

    currState = nextState
    stepCount += 1

    # Reset the episode to start the new one
    if isDone:
        if args.episodes > 0:
            print(f"[DATA] Episode {episodeCount} completed.") 

        episodeCount += 1
        stepCount = 0

        if args.episodes > 0 and episodeCount > args.episodes:
            isRunning = False

        currState, _ = environment.reset(options={"randomStart": True})

    if environment.render:
        environment.screen.fill(
            (Colors.BLACK.R, Colors.BLACK.G, Colors.BLACK.B)
        )

        pygame.draw.rect(
            environment.screen,
            (Colors.GRAY.R, Colors.GRAY.G, Colors.GRAY.B),
            (0, 0, Screen.WIDTH, Screen.HEIGHT),
            2
        )

        environment._drawDrone()
        environment._drawHUD()

        pygame.display.flip()
        environment.clock.tick(Physics.FPS)

if environment.render:
    pygame.quit()
    os.system("clear")

# Save telemetry data if episode recording is done
if args.episodes > 0 and len(telemetryData) > 0:
    import pandas as pd

    df = pd.DataFrame(telemetryData)
    df.to_csv(paths.TEST / "telemetry.csv", index=False)
    print(f"[DATA] Telemetry data saved for {args.episodes} episodes")