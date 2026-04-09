from core.env import DroneEnvironment

if __name__ == "__main__":
    env = DroneEnvironment(useExpert=True)
    env.runUserControl()
