from abc import (
    ABC,
    abstractmethod,
)
import functools
from pathlib import Path
from typing import Any, Iterable
import json

import pandas as pd
import numpy as np
import more_itertools as mit
from pandas import DataFrame
from pydantic import BaseModel, Field, PrivateAttr
from tqdm import tqdm
from src.utils import utils
from sklearn import preprocessing

from src.use import (
    Use,
    UseID,
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.integer, np.floating)):
            return obj.item()
        return super().default(obj)
        
class PredictionCache(BaseModel):
    is_active: bool
    path_prefix: str
    metadata: dict[Any, Any]        
    _cache: pd.DataFrame = PrivateAttr(default=None)
    _cache_path: Path = PrivateAttr(default=None)
    def __init__(self, **data):
        super().__init__(**data)
        self._cache_path = utils.path(".wic/"+self.path_prefix) / "predictions.parquet"
        
        if self.is_active:
            try:
                self._cache = pd.read_parquet(self._cache_path)
            except FileNotFoundError:
                self._initialize_empty_cache()
        else:
            self._initialize_empty_cache()

    def _initialize_empty_cache(self):
        self._cache = pd.json_normalize(self.metadata).assign(id=None, target=None) # parallel solution to embedding cache
        try:
            self._cache['contextual_embedder.similarity_metric'] = self._cache['contextual_embedder.similarity_metric'].astype(str) # to avoid problems in conversion
        except KeyError:
            pass
        self._cache = self._cache.assign(use_0=None, use_1=None, prediction=None)
    
    def _prepare_query(self, left: list[str], right: list[str]) -> pd.DataFrame:
        query = pd.json_normalize(self.metadata).assign(id=None, target=None)
        try:
            query['contextual_embedder.similarity_metric'] = query['contextual_embedder.similarity_metric'].astype(str) # to avoid problems in conversion
        except KeyError:
            pass
        query = query.loc[query.index.repeat(len(left))]
        query = query.assign(use_0=left, use_1=right)
        return query

    def _merge_cache(self, query: pd.DataFrame) -> pd.DataFrame:
        assert not self._cache.duplicated().any()
        merged_df = self._cache.merge(query, how="right")
        assert not merged_df.duplicated(subset=['use_0', 'use_1']).any()
        return merged_df
    
    def _update_cache(self, non_cached: pd.DataFrame, new_predictions: list[float]) -> None:

        print(non_cached)
        for i, _ in non_cached.iterrows():
            non_cached.at[i, "prediction"] = new_predictions[i]
        print(new_predictions)
        print(non_cached)
        
        self._cache = pd.concat([self._cache, non_cached], ignore_index=True)
        assert not self._cache.duplicated().any()

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache.to_parquet(self._cache_path, index=False)

    def _get_non_cached(self, df: pd.DataFrame) -> pd.DataFrame:
        non_cached = df[df["prediction"].isna()].copy(deep=True).reset_index(drop=True)
        return non_cached

    def _filter_new_use_pairs(self, use_pairs: list[tuple[Use, Use]], non_cached: pd.DataFrame) -> list[tuple[Use, Use]]:
        new_left = list(non_cached["use_0"])
        new_right = list(non_cached["use_1"])
        new_use_pairs = [(use_0, use_1) for use_0, use_1 in use_pairs if use_0.identifier in new_left and use_1.identifier in new_right]
        return new_use_pairs        

    def _combine_predictions(self, non_cached: pd.DataFrame, new_predictions: list[float], merged_df: pd.DataFrame) -> dict:
        new_use_pair_ids = list(zip(list(non_cached["use_0"]), list(non_cached["use_1"])))
        cached = merged_df[~merged_df["prediction"].isna()].copy(deep=True).reset_index()
        old_use_pair_ids = list(zip(list(cached["use_0"]), list(cached["use_1"])))

        full_predictions = dict(zip(new_use_pair_ids, new_predictions))
        full_predictions.update(dict(zip(old_use_pair_ids, cached.prediction)))
        return full_predictions
        

class WICModel(BaseModel, ABC):
    prediction_cache: PredictionCache | None = Field(...) # contains currently only metadata, needs to be merged with external cache
    class Config:
        json_encoders = {
            functools.partial: lambda f: f.func.__name__,
            preprocessing.StandardScaler: lambda _: "standard",
            preprocessing.MaxAbsScaler: lambda _: "maxabs",
            preprocessing.MinMaxScaler: lambda _: "minmax",
            preprocessing.RobustScaler: lambda _: "robust"
        }
        
    scaler: Any = Field(default=None)  # should be a scikit-learn scaler
    predictions: dict[tuple[UseID, UseID], float] = Field(default_factory=dict, exclude=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
            
    @abstractmethod
    def as_df(self) -> DataFrame:
        ...
    
    @abstractmethod
    def predict(self, use_pairs: Iterable[tuple[Use, Use]], **kwargs) -> list[float]:
        ...

    def predict_all(self, use_pairs: list[tuple[Use, Use]]) -> list[float]:
            
        left, right = self._prepare_use_identifiers(use_pairs)
                
        # cache will be initialized empty internally if not activated
        query = self.prediction_cache._prepare_query(left, right)
        merged_df = self.prediction_cache._merge_cache(query)
        
        non_cached = self.prediction_cache._get_non_cached(merged_df)
        new_use_pairs = self.prediction_cache._filter_new_use_pairs(use_pairs, non_cached)

        new_predictions = self.predict(use_pairs=new_use_pairs)
        if self.scaler is not None:
            new_predictions = self._scale_predictions(new_predictions)
        
        full_predictions = self.prediction_cache._combine_predictions(non_cached, new_predictions, merged_df)

        # store cache only if activated
        if self.prediction_cache.is_active:
            self.prediction_cache._update_cache(non_cached, new_predictions)
            self.prediction_cache._save_cache()
       
        return list(full_predictions.values())

    def _prepare_use_identifiers(self, use_pairs: list[tuple[Use, Use]]) -> tuple[list[str], list[str]]:
        left = [use_0.identifier for use_0, _ in use_pairs]
        right = [use_1.identifier for _, use_1 in use_pairs]
        return left, right

    def _scale_predictions(self, new_predictions: list[float]) -> list[float]:
        new_predictions = np.array(new_predictions).reshape(-1, 1)
        new_predictions = self.scaler.fit_transform(new_predictions).flatten().tolist()
        with Path("scaler_parameters.json").open(mode="w", encoding="utf8") as f:
            as_json = json.dumps(self.scaler.__dict__, cls=NumpyEncoder, indent=4)
            f.write(as_json)
        return new_predictions
