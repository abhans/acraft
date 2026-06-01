import os

import pygame
import torch

from core.config import Colors, Physics, Screen
from core.env import Environment
from core.model.critic import Policy
from core.model.params import PPO, Paths

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

isRunning: bool = True

paths: Paths = Paths()
testID: str = paths.getCurrentTest()
print(f"[TEST] Test: {testID}")
paths.setTest(testID)

params = PPO(paths)

# Load the trained Policy
ckpt = torch.load(
    params.WEIGHTS,
    map_location='cpu',
    weights_only=False
)


policy: Policy = Policy(
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

    currState = nextState
    
    # Reset the episode to start the new one
    if isDone:
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