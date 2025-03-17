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
                "task": "lscd_compare",
                "task/lscd_compare@task.model": "apd_compare_all",
                "task/wic@task.model.wic": "deepmistake",
                "task/wic/dm_ckpt@task.model.wic.ckpt": "WIC_DWUG+XLWSD",
                "dataset": "dwug_de_300",
                "dataset/split": "full",
                "dataset/spelling_normalization": "none",
                "dataset/preprocessing": "raw",
                "evaluation": "none",
            }
        ))

score1 = run(*instantiate(config))
print("Spearman: ", score1)
