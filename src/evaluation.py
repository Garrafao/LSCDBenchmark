import json
import numpy as np
from abc import ABC
from typing import Any, Callable, Literal, TypeAlias, TypedDict, TypeVar

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, Field

from src.plots import Plotter

EvaluationTask: TypeAlias = Literal[
    "wic", "binary_wic", "change_graded", "change_binary", "compare", "wsi"
]
K = TypeVar("K", str, tuple[str, str])
V = TypeVar("V", int, float)

class DatasetMetadata(TypedDict):
    """_summary_

    :param TypedDict: _description_
    :type TypedDict: _type_
    """    
    name: str
    version: str

class Evaluation(BaseModel, ABC):
    """_summary_

    :param BaseModel: _description_
    :type BaseModel: _type_
    :param ABC: _description_
    :type ABC: _type_
    :return: _description_
    :rtype: _type_
    """    
    task: EvaluationTask
    metric: Callable[[list[float | int], list[float | int]], Any]
    plotter: Plotter | None = Field(...)

    def __call__(self, predictions: dict[K, V], labels: dict[K, V]) -> int | float:
        if self.plotter is not None:
            self.plotter(predictions=predictions, labels=labels)

        results = self.combine_inputs(labels=labels, predictions=predictions)

        y_true = results.label.tolist()
        y_pred = results.prediction.tolist()
        #print(y_true)
        
        #assert not np.isnan(y_true).any() # there can be nan e.g. for 0-judgments in WiC evaluation
        assert not np.isnan(y_pred).any()
        
        results.to_csv("predictions.csv", sep="\t")

        result = {"score": self.metric(y_true, y_pred), "metric": self.metric.func.__name__}

        with open(file="result.json", mode="w", encoding="utf8") as f:
            f.write(json.dumps(result, indent=4))
        return result["score"]

    @staticmethod
    def combine_inputs(labels: dict[K, V], predictions: dict[K, V]) -> DataFrame:
        """Take results as input, combine the instance column and merge all other data. 

        :param labels: the true value
        :type labels: dict[K, V]
        :param predictions: the value predicted by model
        :type predictions: dict[K, V]
        :return: a dataframe with three columns: instances, predictions, labels
        :rtype: DataFrame

        """        
        labels_df = DataFrame(
            {"instance": list(labels.keys()), "label": list(labels.values())}
        )
        predictions_df = DataFrame(
            {
                "instance": list(predictions.keys()),
                "prediction": list(predictions.values()),
            }
        )
        merged = pd.merge(
            left=labels_df,
            right=predictions_df,
            how="right",
            on="instance",
            validate="one_to_one",
        )
        
        assert len(predictions_df) == len(merged)

        first_key = list(labels.keys())[0] # unclear what this code is doing, validate
        if isinstance(first_key, (tuple, list, set)):
            new_cols = merged.instance.apply(pd.Series)
            new_cols.columns = [f"instance_{i}" for i in range(len(new_cols.columns))]  # type: ignore
            merged.drop(columns=["instance"], inplace=True) 
            merged = pd.concat([new_cols, merged], axis=1)
            merged = merged[list(new_cols.columns) + ["prediction", "label"]]
        else:
            merged = merged[["instance", "prediction", "label"]]

        return merged