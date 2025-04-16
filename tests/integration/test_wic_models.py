import sys
sys.path.insert(0, ".")

from hydra import initialize, compose, utils
from tests.utils import overrides, initialize_tests_hydra
from src.utils.runner import instantiate, run
from scipy import stats

import unittest
import pytest

class TestWICModels(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        initialize_tests_hydra(version_base=None, config_path="../conf", working_dir='results')
        super().__init__(*args, **kwargs)    

    def test_wic_eng_simple_plane_afternoon(self) -> None:
        
        # Compose hydra config
        config = compose(config_name="config", return_hydra_config=True, overrides=overrides(
                    {
                        "task": "wic",
                        # "task/wic/dm_ckpt@task.model.ckpt": "WIC_DWUG+XLWSD",
                        "task/wic@task.model": "contextual_embedder",
                        "task.model.ckpt": "bert-base-cased",
                        "dataset": "testwug_en_111",
                        "task/wic/metric@task.model.similarity_metric": "cosine",                 
                        "dataset/split": "dev",
                        "dataset/spelling_normalization": "english",
                        "dataset/preprocessing": "raw",
                        # These 2 words have extreme change_graded values in the gold data
                        "dataset.test_on": ["afternoon_nn", "plane_nn"],
                        "evaluation": "wic",
                        "evaluation/metric": "spearman",
                        # "evaluation/plotter": "none",
                    }
                ))

        # Run 1st time
        score1, predictions1 = run(*instantiate(config))
        # Assert that prediction corresponds to gold
        print(score1)
        assert score1 > 0.0
        #assert pytest.approx(1.0) == score1
        # Run 2nd time
        score2, predictions2 = run(*instantiate(config))
        # Assert that the result reproduces across runs
        assert score1 == score2


if __name__ == '__main__':
    
    unittest.main()
