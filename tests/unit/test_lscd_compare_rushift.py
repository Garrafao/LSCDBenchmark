import sys
sys.path.insert(0, ".")

from hydra import initialize, compose, utils
from tests.utils import overrides, initialize_tests_hydra
from src.utils.runner import instantiate, run
from scipy import stats

import unittest
import pytest


initialize_tests_hydra(version_base=None, config_path="../conf", working_dir='results')


# Compose hydra config
config = compose(config_name="config", return_hydra_config=True, overrides=overrides(
            {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "bert-base-cased",
                    "task/lscd_graded@task.model": "apd_compare_all",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "task/wic/metric@task.model.wic.similarity_metric": "cosine",
                    "+task.model.layers":[12],
                    "dataset": "rusemshift_200",
                    "dataset/split": "test",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "raw",
                    "evaluation": "change_graded",
                    "evaluation/plotter": "none",
                }
        ))

score1 = run(*instantiate(config))
print("Spearman: ", score1)
