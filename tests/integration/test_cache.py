import sys
sys.path.insert(0, ".")

from hydra import initialize, compose, utils
from tests.utils import overrides, initialize_tests_hydra
from src.utils.runner import instantiate, run
from scipy import stats

import unittest
import pytest

class TestCache(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        initialize_tests_hydra(version_base=None, config_path="../conf", working_dir='results')
        super().__init__(*args, **kwargs)    


    def test_bert_embedding_apd_change_graded_ger(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "bert-base-cased",
                    "task/lscd_graded@task.model": "apd_compare_all",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "task/wic/metric@task.model.wic.similarity_metric": "cosine",
                    "dataset": "dwug_de_210",
                    "dataset/split": "dev",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "normalization",
                    # These 2 words have extreme change_graded values in the gold data: 0.0 and 0.93
                    "dataset.test_on": ["Ackergerät", "Engpaß"],
                    "evaluation": "change_graded",
                    "evaluation/plotter": "none",
                }
            ),
        )

        # Run 1st time
        score1, predictions1 = run(*instantiate(config))
        # Assert that prediction corresponds to gold
        # Run 2nd time
        score2, predictions2 = run(*instantiate(config))
        # Assert that the result reproduces across runs
        assert score1 == score2
        assert predictions1 == predictions2


    def some_other_test_function(self) -> None:
        pass

if __name__ == '__main__':
    
    unittest.main()
