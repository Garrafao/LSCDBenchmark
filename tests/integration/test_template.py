import sys
sys.path.insert(0, ".")

from hydra import initialize, compose, utils
from tests.utils import overrides, initialize_tests_hydra
from src.utils.runner import instantiate, run
from scipy import stats

import unittest
import pytest

class TestModels(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        initialize_tests_hydra(version_base=None, config_path="../conf", working_dir='results') # initialize hydra parameters on each class instance
        super().__init__(*args, **kwargs)    
    
    def some_test_function(self) -> None:

        # Compose hydra config
        config = compose(config_name="config", return_hydra_config=True, overrides=overrides(
                    {
                        "task": "lscd_graded",
                        "task.model.wic.ckpt": "bert-base-german-cased",
                        "task/lscd_graded@task.model": "apd_compare_all",
                        "task/wic@task.model.wic": "contextual_embedder",
                        "task/wic/metric@task.model.wic.similarity_metric": "cosine",
                        "dataset": "dwug_de_210",                        
                        "dataset/split": "dev",
                        "dataset/spelling_normalization": "german",
                        "dataset/preprocessing": "normalization",
                        "dataset.test_on": ["Ackergerät", "Engpaß"],
                        "evaluation": "change_graded",
                        "evaluation/plotter": "none",
                    }
                ))

        # Run 
        score, predictions = run(*instantiate(config))
        # Assert that prediction corresponds to gold
        assert pytest.approx(1.0) == score

    def some_other_test_function(self) -> None:
        pass

if __name__ == '__main__':
    
    initialize_tests_hydra(version_base=None, config_path="../conf", working_dir='results')
    unittest.main()
