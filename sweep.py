import sys
from functools import partial

from absl import app

import wandb
from train_offline import main

if __name__ == "__main__":
    sweep_configuration = {
        "method": "random",
        "metric": {
            "name": "normalized_return",
            "goal": "maximize",
        },
        "parameters": {
            "alpha": {"min": 0.1, "max": 5.0},
            "beta": {"min": 0.1, "max": 10.0},
        },
    }
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project="sweep",
    )
    wandb.agent(sweep_id, partial(app.run, main, sys.argv), count=10)
