# The Environment

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install it with the command below if its' not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the virtual environment and install all dependencies:

```bash
uv sync
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

## Running the Program

In order to run the simulation with the "trained" policy, use the command below:

```bash
python -W ignore -m src.main
```

A window will appear that simulates episodes with random initial starting coordinates and active wind.

![Environment Example](./docs/example.png)

To record telemetry data over a number of episodes, pass the `-e` flag with the desired episode count:

```bash
python -W ignore -m src.main -e 20
```

The simulation will run for the specified number of episodes and save per-step telemetry data (position, velocity, heading, thrust, and reward) to `telemetry.csv` inside the current test directory. 

The default number of episodes is `10`. Setting the flag to `0` disables data collection entirely.

## Training

To system to function properly and for training to be reliable, expert data must be collected first. This can be done by the command below:

```bash
python -W ignore -m core.env
```

The data will be saved to `data/expert.npz` directory.

![Example Training Results](./docs/results.png)

After the collection of expert data, the `BehaviorClone` can be trained with the command below:

```bash
python -W ignore -m core.model.train
```

Finally, PPO algorithm can be trained to achieve the trained policy which will be used in the real-time simulation. This can be done by the command below:

``` bash
python -W ignore -m src.train
```

### Chaining

This entire process can be executed with the `main.sh` script:

```bash
./main.sh
```

This will automatically collect expert data, train Behavior Cloning and PPO, simulating the learned policy real-time at the end.