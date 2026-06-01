"""
This file contains the training script for the 'Behavior Clone' model. 

It initializes the training process, trains the model, and saves the trained policy to disk.
"""
# To visualize the environment
from core.env import Environment
from core.model.bc import BehaviorClone
from core.model.params import BClone, Paths


def main():
    paths: Paths = Paths()
    # Create a new test directory to save the results
    paths.createTest()
    testID: str = paths.getCurrentTest()

    env = Environment(
        useExpert=False,
        # * Change the "render" to speed up training (disables visualization)
        render=False, 
        stepsMax=1000
    )

    params = BClone(paths)
    paths.ensure(params.WEIGHTS, params.METRICS)

    trainer = BehaviorClone(
        test=testID,
        pathExpert=str(paths.EXPERT),
        batchSize=params.BATCH_SIZE,
        epochs=params.EPOCHS,
        lr=params.LR,
        splits=params.SPLITS,
    )

    trainer.train(env)
    trainer.save(params.WEIGHTS.as_posix())
    # Save the training metrics
    trainer.saveMetrics(params.METRICS)

    print(f"\n[SYSTEM:::BCLONE] 'Behavior Clone' training complete. Policy saved to '{params.WEIGHTS}'.")


if __name__ == "__main__":
    main()