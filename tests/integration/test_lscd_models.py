import sys

sys.path.insert(0, ".")

from hydra import initialize, compose, utils
from tests.utils import overrides, initialize_tests_hydra
from src.utils.runner import instantiate, run
from scipy import stats
import numpy as np

import unittest
import pytest


class TestLSCDModels(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        initialize_tests_hydra(
            version_base=None, config_path="../conf", working_dir="results"
        )
        super().__init__(*args, **kwargs)

    # Minimal run of model on very small data set for frequent testing purposes
    def test_apd_change_graded_eng_simple_arm(self) -> None:

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

        # Run
        score, predictions = run(*instantiate(config))
        
    def test_apd_cosine_cut_change_graded_eng_simple_arm(self) -> None:

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
                    "task/wic/metric@task.model.wic.similarity_metric": "cosine_cut",
                    "task.model.wic.similarity_metric.thresholds": [0.32535901204859474, 0.48300452735758276, 0.6115888591099456], # English thresholds
                    "task.model.wic.similarity_metric.labels": [1, 2, 3, 4], # Can be omitted as specified as default in config
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

        # Run
        score, predictions = run(*instantiate(config))
        
    def test_apd_sampled_change_graded_eng_simple_arm(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "bert-base-cased",
                    "task/lscd_graded@task.model": "apd_compare_sampled",
                    "task.model.use_pairs.sample.n": "5",
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

        # Run
        score, predictions = run(*instantiate(config))

    def test_apd_change_graded_eng_simple_plane_afternoon(self) -> None:

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
                    "dataset": "testwug_en_111",  # todo: this should become testwug_en_1.1.1
                    "dataset/split": "full",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "lemmatization",
                    # These 2 words have extreme change_graded values in the gold data: 0.0 and 0.94
                    "dataset.test_on": ["afternoon_nn", "plane_nn"],
                    "evaluation": "change_graded",
                    "evaluation/plotter": "none",
                }
            ),
        )

        # Run
        score, predictions = run(*instantiate(config))
        # Assert that prediction corresponds to gold
        assert pytest.approx(1.0) == score

    # Minimal run of model on very small data set for frequent testing purposes
    def test_apd_change_graded_ger_simple(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "bert-base-german-cased",
                    "task/lscd_graded@task.model": "apd_compare_all",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "task/wic/metric@task.model.wic.similarity_metric": "cosine",
                    "dataset": "refwug_110",  # todo: this should become testwug_en_1.1.1
                    "dataset/split": "full",
                    "dataset/spelling_normalization": "german",
                    "dataset/preprocessing": "toklem",
                    # These 2 words have extreme change_graded values in the gold data: 0.0 and 0.87
                    "dataset.test_on": ["Reichstag", "Presse"],
                    "evaluation": "change_graded",
                    "evaluation/plotter": "none",
                }
            ),
        )

        # Run
        score, predictions = run(*instantiate(config))
        # Assert that prediction corresponds to gold
        assert pytest.approx(1.0) == score

    def test_apd_change_graded_ger(self) -> None:

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
                    # "dataset/_target_": "src.dataset.Dataset",
                    # "+dataset/name": "dwug_de_210", # is done as default in runner.instantiate()
                    "dataset": "dwug_de_210",
                    "dataset/split": "dev",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "normalization",
                    # These 2 words have extreme change_graded values in the gold data: 0.0 and 0.93
                    "dataset.test_on": ["Ackergerät", "Engpaß"],
                    # "dataset/filter_lemmas": "all",
                    # "evaluation": "wic",
                    # "evaluation/metric": "f1_score",
                    "evaluation": "change_graded",
                    "evaluation/plotter": "none",
                }
            ),
        )

        # Run 1st time
        score1, predictions = run(*instantiate(config))
        # Assert that prediction corresponds to gold
        assert pytest.approx(1.0) == score1
        
    def test_xllxm_apd_compare_ger_comedi_test(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "pierluigic/xl-lexeme",
                    "task/lscd_graded@task.model": "apd_compare_all",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "task/wic/metric@task.model.wic.similarity_metric": "cosine",
                    "dataset": "durel_300",
                    "dataset/split": "comedi_test",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "raw",
                    "evaluation": "compare",
                    "evaluation/plotter": "none",
                }
            ),
        )

        score1, predictions = run(*instantiate(config))
        print(score1)
        assert pytest.approx(-1.0) == score1        

    def test_xldurel_apd_compare_ger_comedi_test(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "sachinn1/xl-durel",
                    "task/lscd_graded@task.model": "apd_compare_all",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "task/wic/metric@task.model.wic.similarity_metric": "cosine",
                    "dataset": "durel_300",
                    "dataset/split": "comedi_test",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "raw",
                    "evaluation": "compare",
                    "evaluation/plotter": "none",
                }
            ),
        )

        score1, predictions = run(*instantiate(config))
        print(score1)
        assert pytest.approx(-0.7714285714285715) == score1        

    def test_xldurel_apd_cosine_cut_compare_ger_comedi_test(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "sachinn1/xl-durel",
                    "task/lscd_graded@task.model": "apd_compare_all",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "task/wic/metric@task.model.wic.similarity_metric": "cosine_cut",
                    "task.model.wic.similarity_metric.thresholds": [0.32997490400345, 0.46455512615346295, 0.5999736878501423], # German thresholds
                    "task.model.wic.similarity_metric.labels": [1, 2, 3, 4], # Can be omitted as specified as default in config
                    "dataset": "durel_300",
                    "dataset/split": "comedi_test",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "raw",
                    "evaluation": "compare",
                    "evaluation/plotter": "none",
                }
            ),
        )

        score1, predictions = run(*instantiate(config))
        print(score1)
        assert pytest.approx(-0.7714285714285715) == score1     
        
    def test_xldurel_jsddot_downsampled_cosine_cut_change_graded_eng_comedi_test(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "sachinn1/xl-durel",
                    "task/lscd_graded@task.model": "jsddot_all_downsampled",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "task/wic/metric@task.model.wic.similarity_metric": "cosine_cut",
                    "task.model.wic.similarity_metric.thresholds": [0.32535901204859474, 0.48300452735758276, 0.6115888591099456], # English thresholds
                    "task.model.wic.similarity_metric.labels": [1, 2, 3, 4], # Can be omitted as specified as default in config
                    "dataset": "dwug_en_300",
                    "dataset/split": "comedi_test",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "raw",
                    "evaluation": "change_graded",
                    "evaluation/plotter": "none",
                }
            ),
        )

        score1, predictions = run(*instantiate(config))
        print(score1)
        assert pytest.approx(0.705170430668875) == score1    
        
    def test_xldurel_apd_cosine_cut_compare_rus_comedi_test(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "sachinn1/xl-durel",
                    "task/lscd_graded@task.model": "apd_compare_all",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "task/wic/metric@task.model.wic.similarity_metric": "cosine_cut",
                    "task.model.wic.similarity_metric.thresholds": [0.25539005328206443, 0.49066763335561997, 0.6147420218754426], # Russian thresholds
                    "task.model.wic.similarity_metric.labels": [1, 2, 3, 4], # Can be omitted as specified as default in config
                    "dataset": "rusemshift_1_200",
                    "dataset/split": "comedi_test",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "raw",
                    "evaluation": "compare",
                    "evaluation/plotter": "none",
                }
            ),
        )

        score1, predictions = run(*instantiate(config))
        print(score1)
        assert pytest.approx(-0.9079538924922887) == score1        
        
    def test_xldurel_apd_cosine_cut_compare_rus2_comedi_test(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "sachinn1/xl-durel",
                    "task/lscd_graded@task.model": "apd_compare_all",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "task/wic/metric@task.model.wic.similarity_metric": "cosine_cut",
                    "task.model.wic.similarity_metric.thresholds": [0.25539005328206443, 0.49066763335561997, 0.6147420218754426], # Russian thresholds
                    "task.model.wic.similarity_metric.labels": [1, 2, 3, 4], # Can be omitted as specified as default in config
                    "dataset": "rushifteval2_200",
                    "dataset/split": "comedi_test",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "raw",
                    "evaluation": "compare",
                    "evaluation/plotter": "none",
                }
            ),
        )

        score1, predictions = run(*instantiate(config))
        print(score1)
        assert pytest.approx(-0.7825013313896345) == score1                
        
    def test_xldurel_apd_cosine_cut_change_graded_eng_comedi_test(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "sachinn1/xl-durel",
                    "task/lscd_graded@task.model": "apd_compare_all",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "task/wic/metric@task.model.wic.similarity_metric": "cosine_cut",
                    "task.model.wic.similarity_metric.thresholds": [0.32535901204859474, 0.48300452735758276, 0.6115888591099456], # English thresholds
                    "task.model.wic.similarity_metric.labels": [1, 2, 3, 4], # Can be omitted as specified as default in config
                    "dataset": "dwug_en_300",
                    "dataset/split": "comedi_test",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "raw",
                    "evaluation": "change_graded",
                    "evaluation/plotter": "none",
                }
            ),
        )

        score1, predictions = run(*instantiate(config))
        print(score1)
        assert pytest.approx(0.9240164263936982) == score1   
        
    def test_xldurel_apd_cosine_cut_scaled_compare_rus_comedi_test(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "sachinn1/xl-durel",
                    "task/lscd_graded@task.model": "apd_compare_all",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "task/wic/metric@task.model.wic.similarity_metric": "cosine_cut_scaled",
                    "task.model.wic.similarity_metric.thresholds": [0.25539005328206443, 0.49066763335561997, 0.6147420218754426], # Russian thresholds
                    "task.model.wic.similarity_metric.labels": [1, 2, 3, 4], # Can be omitted as specified as default in config
                    "dataset": "rusemshift_1_200",
                    "dataset/split": "comedi_test",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "raw",
                    "evaluation": "compare",
                    "evaluation/plotter": "none",
                }
            ),
        )

        score1, predictions = run(*instantiate(config))
        print(score1)
        assert pytest.approx(-0.9079538924922887) == score1        
         
    def test_diasense_change_graded_eng_simple_arm(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "bert-base-cased",
                    "task/lscd_graded@task.model": "diasense_all",
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

        # Run
        score, predictions = run(*instantiate(config))

    # def test_graded_apd_compare_all(self) -> None:
    #     with initialize(version_base=None, config_path="../../conf"):
    #         cfg = compose(
    #             config_name="config",
    #             overrides=overrides(
    #                 {
    #                     "task": "lscd_graded",
    #                     "task/lscd_graded@task.model": "apd_compare_all",
    #                     "task/wic@task.model.wic": "bert",
    #                     "dataset": "dwug_de",
    #                     "dataset.test_on": "3",
    #                 }
    #             ),
    #         )
    #         self.assertIsInstance(run(*instantiate(cfg)), float)

    # def test_binary_apd_compare_all() -> None:
    #     with initialize(version_base=None, config_path="../../conf"):
    #         cfg = compose(
    #             config_name="config",
    #             overrides=overrides(
    #                 {
    #                     "task": "lscd_binary",
    #                     "task/lscd_binary@task.model": "apd_compare_all",
    #                     "task/wic@task.model.graded_model.wic": "bert",
    #                     "task/lscd_binary/threshold_fn@task.model.threshold_fn": "mean_std",
    #                     "dataset": "dwug_de",
    #                     "dataset.test_on": "3",
    #                 }
    #             ),
    #         )
    #         assert isinstance(run(*instantiate(cfg)), float)

    # def test_binary_cos() -> None:
    #     with initialize(version_base=None, config_path="../../conf"):
    #         cfg = compose(
    #             config_name="config",
    #             overrides=overrides(
    #                 {
    #                     "task": "lscd_binary",
    #                     "task/lscd_binary@task.model": "cos",
    #                     "task/wic@task.model.graded_model.wic": "bert",
    #                     "task/lscd_binary/threshold_fn@task.model.threshold_fn": "mean_std",
    #                     "dataset": "dwug_de",
    #                     "dataset.test_on": "3",
    #                 }
    #             ),
    #         )
    #         assert isinstance(run(*instantiate(cfg)), float)

    def test_apd_change_graded_es_simple_actitud(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task/lscd_graded@task.model": "apd_compare_all",
                    "task/wic@task.model.wic": "deepmistake",
                    "task/wic/dm_ckpt@task.model.wic.ckpt": "WIC_DWUG+XLWSD",
                    "dataset": "dwug_es_300",
                    "dataset/split": "full",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "raw",
                    "dataset.test_on": ["actitud", "ataque"],
                    "evaluation": "change_graded",
                }
            ),
        )

        # Run
        score, predictions = run(*instantiate(config))

        assert pytest.approx(-1.0) == score

    def test_apd_compare_es_simple_recordar(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_compare",
                    "task/lscd_compare@task.model": "apd_compare_all",
                    "task/wic@task.model.wic": "deepmistake",
                    "task/wic/dm_ckpt@task.model.wic.ckpt": "WIC_DWUG+XLWSD",
                    "dataset": "dwug_es_300",
                    "dataset/split": "full",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "raw",
                    "dataset.test_on": ["recordar", "propiamente"],
                    "evaluation": "compare",
                }
            ),
        )

        # Run
        score, predictions = run(*instantiate(config))

        assert pytest.approx(1.0) == score


    ## resampled data german
    def test_apd_change_graded_de_Abgesang_Frechheit(self) -> None:
        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "bert-base-german-cased",
                    "task/lscd_graded@task.model": "apd_compare_all",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "task/wic/metric@task.model.wic.similarity_metric": "cosine",
                    "dataset": "dwug_de_resampled_100",
                    "dataset/split": "dev1",
                    "dataset/spelling_normalization": "german",
                    "dataset/preprocessing": "raw",
                    # These 2 words have extreme change_graded values in the gold data: 0.0 and 0.94
                    "dataset.test_on": ["Mulatte", "Frechheit"],
                    "evaluation": "change_graded",
                    "evaluation/plotter": "none",
                }
            ),
        )

        # Run
        score, predictions = run(*instantiate(config))
        # Assert that prediction corresponds to gold
        assert 0.0 >= score

     ## resampled data english
    def test_apd_change_graded_eng_attack_edge(self) -> None:
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
                    "dataset": "dwug_en_resampled_100",
                    "dataset/split": "full",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "raw",
                    # These 2 words have extreme change_graded values in the gold data: 0.0 and 0.94
                    "dataset.test_on": ["attack_nn", "edge_nn"],
                    "evaluation": "change_graded",
                    "evaluation/plotter": "none",
                }
            ),
        )

        # Run
        score, predictions = run(*instantiate(config))
        # Assert that prediction corresponds to gold
        assert 0.0 <= score
        
    def test_apd_change_graded_eng_graft_ounce(self) -> None:
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
                    "dataset": "dwug_en_resampled_100",
                    "dataset/split": "full",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "raw",
                    # These 2 words have extreme change_graded values in the gold data: 0.0 and 0.55
                    "dataset.test_on": ["ounce_nn", "twist_nn"],
                    "evaluation": "change_graded",
                    "evaluation/plotter": "none",
                }
            ),
        )

        # Run
        score, predictions = run(*instantiate(config))
        print(score)
        # Assert that prediction corresponds to gold
        assert 0.0 >= score
        
    # reampled data Swedish
    def test_apd_change_graded_sv_aktiv_krita(self) -> None:
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
                    "dataset": "dwug_sv_resampled_100",
                    "dataset/split": "full",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "raw",
                    # These 2 words have extreme change_graded values in the gold data: 0.0 and 0.94
                    "dataset.test_on": ["aktiv", "krita"],
                    "evaluation": "change_graded",
                    "evaluation/plotter": "none",
                }
            ),
        )

        # Run
        score, predictions = run(*instantiate(config))
        # Assert that prediction corresponds to gold
        assert 1.0 >= score


    def test_apd_downsampled_change_graded_eng_simple_plane_afternoon(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "bert-base-cased",
                    "task/lscd_graded@task.model": "apd_compare_all_downsampled",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "task/wic/metric@task.model.wic.similarity_metric": "cosine",
                    "dataset": "testwug_en_111",  # todo: this should become testwug_en_1.1.1
                    "dataset/split": "full",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "lemmatization",
                    # These 2 words have extreme change_graded values in the gold data: 0.0 and 0.94
                    "dataset.test_on": ["afternoon_nn", "plane_nn"],
                    "evaluation": "change_graded",
                    "evaluation/plotter": "none",
                }
            ),
        )

        # Run
        score, predictions = run(*instantiate(config))
        # Assert that prediction corresponds to gold
        assert pytest.approx(1.0) == score


    def test_jsddot_downsampled_change_graded_eng_simple_plane_afternoon(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "bert-base-cased",
                    "task/lscd_graded@task.model": "jsddot_all_downsampled",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "task/wic/metric@task.model.wic.similarity_metric": "cosine",
                    "dataset": "testwug_en_111",  # todo: this should become testwug_en_1.1.1
                    "dataset/split": "full",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "lemmatization",
                    # These 2 words have extreme change_graded values in the gold data: 0.0 and 0.94
                    "dataset.test_on": ["afternoon_nn", "plane_nn"],
                    "evaluation": "change_graded",
                    "evaluation/plotter": "none",
                }
            ),
        )

        # Run
        score, predictions = run(*instantiate(config))
        # Assert that prediction corresponds to gold
        assert pytest.approx(1.0) == score

    def test_jsddot_downsampled_change_graded_eng_simple_arm(self) -> None: # for fast testing

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "bert-base-cased",
                    "task/lscd_graded@task.model": "jsddot_all_downsampled",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "task/wic/metric@task.model.wic.similarity_metric": "cosine",
                    "dataset": "testwug_en_111", 
                    "dataset/split": "full",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "raw",
                    "dataset.test_on": ["arm"],
                    "evaluation": "change_graded",
                    "evaluation/plotter": "none",
                }
            ),
        )

        # Run
        score, predictions = run(*instantiate(config))
        # Assert that prediction corresponds to gold
        assert np.isnan(score)
        
    # Minimal run of model on very small data set for frequent testing purposes
    def test_cos_change_graded_eng_simple_arm(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "bert-base-cased",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "task/lscd_graded@task.model": "cos_all",
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

        # Run
        score, predictions = run(*instantiate(config))
       
    def test_cos_change_graded_eng_simple_plane_afternoon(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "bert-base-cased",
                    "task/lscd_graded@task.model": "cos_all",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "dataset": "testwug_en_111",
                    "dataset/split": "full",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "lemmatization",
                    # These 2 words have extreme change_graded values in the gold data: 0.0 and 0.94
                    "dataset.test_on": ["afternoon_nn", "plane_nn"],
                    "evaluation": "change_graded",
                    "evaluation/plotter": "none",
                }
            ),
        )

        # Run
        score, predictions = run(*instantiate(config))
        # Assert that prediction corresponds to gold
        assert pytest.approx(1.0) == score
        
    # Minimal run of model on very small data set for frequent testing purposes
    def test_jsdsoft_change_graded_eng_simple_arm(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "bert-base-cased",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "task/lscd_graded@task.model": "jsdsoft_all",
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

        # Run
        score, predictions = run(*instantiate(config))
       
    def test_jsdsoft_change_graded_eng_simple_plane_afternoon(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "bert-base-cased",
                    "task/lscd_graded@task.model": "jsdsoft_all",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "dataset": "testwug_en_111",
                    "dataset/split": "full",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "lemmatization",
                    # These 2 words have extreme change_graded values in the gold data: 0.0 and 0.94
                    "dataset.test_on": ["afternoon_nn", "plane_nn"],
                    "evaluation": "change_graded",
                    "evaluation/plotter": "none",
                }
            ),
        )

        # Run
        score, predictions = run(*instantiate(config))
        # Assert that prediction corresponds to gold
        assert pytest.approx(1.0) == score
        
    def test_xldurel_jsdsoft_cosine_change_graded_eng_comedi_test(self) -> None:

        # Compose hydra config
        config = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task.model.wic.ckpt": "sachinn1/xl-durel",
                    "task/lscd_graded@task.model": "jsdsoft_all",
                    "task/wic@task.model.wic": "contextual_embedder",
                    "dataset": "dwug_en_300",
                    "dataset/split": "comedi_test",
                    "dataset/spelling_normalization": "none",
                    "dataset/preprocessing": "raw",
                    "evaluation": "change_graded",
                    "evaluation/plotter": "none",
                }
            ),
        )

        # Run
        score, predictions = run(*instantiate(config))
        print(score)
        # Assert that prediction corresponds to gold
        assert pytest.approx(0.6930123197952737) == score
        

if __name__ == "__main__":

    unittest.main()
