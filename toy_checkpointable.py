#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random
import time

import numpy as np

import ray
from ray.tune import Trainable, run_experiments, sample_from
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler


class MyTrainableClass(Trainable):
    """Example agent whose learning curve is a random sigmoid.

    The dummy hyperparameters "width" and "height" determine the slope and
    maximum reward value reached.
    """

    def _setup(self, config):
        self.timestep = 0

    def _train(self):
        self.timestep += 1
        v = np.tanh(float(self.timestep) / self.config["width"])
        v *= self.config["height"]
        time.sleep(5)

        # Here we use `episode_reward_mean`, but you can also report other
        # objectives such as loss or accuracy.
        return {"episode_reward_mean": v}

    def _save(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "w") as f:
            f.write(json.dumps({"timestep": self.timestep}))
        return path

    def _restore(self, checkpoint_path):
        with open(checkpoint_path) as f:
            self.timestep = json.loads(f.read())["timestep"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--redis-address",
        default=None,
        type=str,
        help="The Redis address of the cluster.")
    args = parser.parse_args()
    ray.init(redis_address=args.redis_address)

    # asynchronous hyperband early stopping, configured with
    # `episode_reward_mean` as the
    # objective and `training_iteration` as the time unit,
    # which is automatically filled by Tune.

    run_experiments(
        {
            "asynchyperband_test": {
                "run": MyTrainableClass,
                "stop": {
                    "training_iteration": 50
                },
                "checkpoint_freq": 2,
                "num_samples": 30,
                "resources_per_trial": {
                    "cpu": 4,
                    "gpu": 0
                },
                "config": {
                    "width": sample_from(
                        lambda spec: 10 + int(90 * random.random())),
                    "height": sample_from(
                        lambda spec: int(100 * random.random())),
                },
            }
        }, verbose=1)
