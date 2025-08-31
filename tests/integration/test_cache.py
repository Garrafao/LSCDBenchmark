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
        
    def test_bert_apd_change_graded_eng_simple_arm(self) -> None:

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
                    "dataset": "testwug_en_111",
                    "dataset/split": "full",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "raw",
                    # has very few usages
                    "dataset.test_on": ["arm"],
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


    def test_bert_apd_change_graded_ger(self) -> None:

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
        
    def test_bert_wic_eng_simple_arm(self) -> None:
        
        # Compose hydra config
        config = compose(config_name="config", return_hydra_config=True, overrides=overrides(
                    {
                        "task": "wic",
                        "task/wic@task.model": "contextual_embedder",
                        "task.model.ckpt": "bert-base-cased",
                        "dataset": "testwug_en_111",
                        "task/wic/metric@task.model.similarity_metric": "cosine",                 
                        "dataset/split": "full",
                        "dataset/spelling_normalization": "english",
                        "dataset/preprocessing": "raw",
                        "dataset.test_on": ["arm"],
                        "evaluation": "wic",
                        "evaluation/metric": "spearman",
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

        
    def test_bert_wic_eng_simple_plane_afternoon(self) -> None:
        
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
        #print(predictions1)
        assert score1 > 0.0
        #assert pytest.approx(1.0) == score1
        # Run 2nd time
        score2, predictions2 = run(*instantiate(config))
        # Assert that the result reproduces across runs
        assert score1 == score2
        
    def test_bert_wic_eng_simple_plane_afternoon_lemma(self) -> None:
        
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
        #print(predictions1)
        assert score1 > 0.0
        
        # Run 2nd time
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
                        "dataset/preprocessing": "lemmatization",
                        # These 2 words have extreme change_graded values in the gold data
                        "dataset.test_on": ["afternoon_nn", "plane_nn"],
                        "evaluation": "wic",
                        "evaluation/metric": "spearman",
                        # "evaluation/plotter": "none",
                    }
                ))
                
        score2, predictions2 = run(*instantiate(config))
        # Assert that the result reproduces across runs
        print(score2)
        assert score2 > 0.0
        assert score1 != score2        
        
    def test_bert_wic_eng_simple_plane_afternoon_similarity(self) -> None:
        
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
        #print(predictions1)
        assert score1 > 0.0
        
        # Run 2nd time
        # Compose hydra config
        config = compose(config_name="config", return_hydra_config=True, overrides=overrides(
                    {
                        "task": "wic",
                        # "task/wic/dm_ckpt@task.model.ckpt": "WIC_DWUG+XLWSD",
                        "task/wic@task.model": "contextual_embedder",
                        "task.model.ckpt": "bert-base-cased",
                        "dataset": "testwug_en_111",
                        "task/wic/metric@task.model.similarity_metric": "manhattan",                 
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
                
        score2, predictions2 = run(*instantiate(config))
        # Assert that the result reproduces across runs
        print(score2)
        assert score1 != score2            
        
    def test_bert_wic_eng_simple_plane_afternoon_model(self) -> None:
        
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
        #print(predictions1)
        assert score1 > 0.0
        
        # Run 2nd time
        # Compose hydra config
        config = compose(config_name="config", return_hydra_config=True, overrides=overrides(
                    {
                        "task": "wic",
                        # "task/wic/dm_ckpt@task.model.ckpt": "WIC_DWUG+XLWSD",
                        "task/wic@task.model": "contextual_embedder",
                        "task.model.ckpt": "pierluigic/xl-lexeme",
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
                
        score2, predictions2 = run(*instantiate(config))
        # Assert that the result reproduces across runs
        print(score2)
        assert score1 != score2              

if __name__ == '__main__':
    
    unittest.main()
