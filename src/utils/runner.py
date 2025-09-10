import os
import csv
from pathlib import Path
from typing import Any, TypeAlias
from tqdm import tqdm
from hydra import utils
from hydra.core.hydra_config import HydraConfig
import yaml
import pandas as pd
from pandas import DataFrame, Series
from omegaconf import DictConfig, OmegaConf, open_dict
from src.dataset import Dataset
from src.evaluation import Evaluation
from src.lscd import GradedLSCDModel, BinaryThresholdModel
from src.wic import WICModel
from src.wic.contextual_embedder import ContextualEmbedder
from src.wic.model import NumpyEncoder
from src.utils.utils import path as get_path
from src.wsi import WSIModel
import numpy as np

OUTPUTS = get_path("outputs")
MULTIRUN = get_path("multirun")

Model: TypeAlias = WICModel | GradedLSCDModel | BinaryThresholdModel | WSIModel
# Model: TypeAlias = WICModel


def instantiate(
    config: DictConfig,
) -> tuple[Dataset | None, Model | None, Evaluation | None]:
    dataset = None
    model = None
    evaluation = None

    # The most simple instantiation would do this:            
    # dataset = utils.instantiate(config.dataset, _convert_="all")
    # model = utils.instantiate(config.task.model, _convert_="all")
    # evaluation = utils.instantiate(config.evaluation, _convert_="all")
       
    try:
        hydra_cfg = HydraConfig.get()
    except ValueError: # this should happen only in testing scenarios where hydra's compose is used
        # for testing don't use hydra config
        hydra_cfg = None

    if config.get("dataset") is not None:

        # Adds the data set name, in the dataset subdictionary, not sure if this is needed in the future
        with open_dict(config):
            config.dataset.name = config.dataset.path

        # I'm unsure what this block is doing, it seems to store the parameters, but this should be handled by Hydra
        if hydra_cfg is not None:

            choices = OmegaConf.to_container(hydra_cfg.runtime.choices)
            output_dir = Path(hydra_cfg.runtime.output_dir)

            OmegaConf.set_struct(config, True)
            with open_dict(config):
                config.dataset.name = choices["dataset"].split(".")[0]  # type: ignore, not sure if this is needed anymore
                with open(
                    file=output_dir / ".hydra" / "config.yaml", 
                    mode="w", 
                    encoding="utf8"
                ) as f:
                    yaml.safe_dump(
                        OmegaConf.to_object(config),
                        f,
                        allow_unicode=True,
                        default_flow_style=False,
                    )

        dataset = utils.instantiate(
            config.dataset,
            _convert_="all",
        )
    if config.get("task") is not None and config.task.get("model") is not None:
        model = utils.instantiate(config.task.model, _convert_="all")
    if config.get("evaluation") is not None:
        # after instantiation, could it still be None?
        evaluation = utils.instantiate(config.evaluation, _convert_="all")
        
    return dataset, model, evaluation


def run(
    dataset: Dataset | None, model: Model | None, evaluation: Evaluation | None
) -> float | None:

    score = None
    if model is not None and dataset is not None:
        predictions: Any = {}

        lemmas = dataset.filter_lemmas(dataset.lemmas)
        lemma_pbar = lemmas
        if isinstance(model, WICModel):
            assert (
                dataset.wic_use_pairs is not None
            ), "Please specify a set of options for use pairs in `dataset.wic_use_pairs`"
            group = dataset.wic_use_pairs.group
            sample = dataset.wic_use_pairs.sample

            if isinstance(model, ContextualEmbedder) and model.encode_only:
                uses = []
                lemma_pbar = tqdm(lemmas, leave=False, desc="Collecting uses")
                for lemma in lemma_pbar:
                    uses.extend(lemma.get_uses())
                uses = list(set(uses))
                model.encode_all(uses)
                return None

            use_pairs = dataset.use_pairs(group=group, sample=sample)
            id_pairs = [(use_0.identifier, use_1.identifier) for use_0, use_1 in use_pairs]
            predictions.update(dict(zip(id_pairs, model.predict_all(use_pairs))))

        elif isinstance(model, GradedLSCDModel):
            predictions.update(
                dict(zip([lemma.name for lemma in lemmas], model.predict_all(lemmas)))
            )
        elif isinstance(model, BinaryThresholdModel):
            graded_predictions = []
            lemma_names = [lemma.name for lemma in dataset.lemmas]
            for lemma in lemma_pbar:
                graded_predictions.append(model.graded_model.predict(lemma))
            predictions.update(
                dict(zip(lemma_names, model.predict(graded_predictions)))
            )
        elif isinstance(model, WSIModel):
            for lemma in lemma_pbar:
                uses = lemma.get_uses()
                ids = [use.identifier for use in uses]
                predictions.update(dict(zip(ids, model.predict(uses))))

        predictions_df = DataFrame(
            {
                "instance": list(predictions.keys()),
                "prediction": list(predictions.values()),
            }
        )

        first_key = list(predictions.keys())[0]
        if isinstance(first_key, (tuple, list, set)):
            new_cols = predictions_df.instance.apply(Series)
            new_cols.columns = [f"instance_{i}" for i in range(len(new_cols.columns))]  # type: ignore
            predictions_df.drop(columns=["instance"], inplace=True)
            predictions_df = pd.concat([new_cols, predictions_df], axis=1)
        predictions_df.to_csv(
            path_or_buf="predictions.csv", sep="\t", quoting=csv.QUOTE_NONE
        )

        if evaluation is not None:
            print(evaluation)
            labels = dataset.get_labels(evaluation_task=evaluation.task)
            score = evaluation(labels=labels, predictions=predictions)

        if score is None:
             score = np.nan
        
    return score, predictions
