# The Environment

In order to run the simulation with "trained" policy, use the command below:

```bash
python -W ignore -m src.main
```

A window will appear that simulates episodes with random initial starting coordinates and active wind.

<!-- Add example screenshot -->

## Training

To system to function properly and for training to be reliable, expert data must be collected first. This can be done by the command below:

```bash
python -W ignore -m core.env
```

The data will be saved to `data/expert.npz` directory.

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