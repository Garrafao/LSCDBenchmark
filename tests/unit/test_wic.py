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
                "task": "wic",
                "task/wic@task.model": "contextual_embedder",
                "task.model.ckpt": "pierluigic/xl-lexeme",
                "task.model.embedding_scope": "sentence",
                "task/wic/metric@task.model.similarity_metric": "cosine",
                "dataset": "dwug_en_200",
                "dataset/split": "test",
                "dataset/spelling_normalization": "none",
                "dataset/preprocessing": "raw",
                "evaluation": "wic",
                "evaluation/metric": "spearman",
                "evaluation/plotter": "none",
            }
        ))

score1 = run(*instantiate(config))
print("Spearman: ", score1)
