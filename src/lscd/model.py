from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F
from pandas import DataFrame
from scipy.spatial import distance
from src.config import ID, Config
from src.use import Use
from src.vectorizer import Vectorizer
from torch import Tensor


class VectorModel:
    def __init__(
        self,
        config: Config,
        uses: List[Use],
        vectorizer: Vectorizer,
    ):
        self.config = config
        self.uses = uses
        self.vectorizer = vectorizer
        self._vectors = None
        self._id_to_row = None
        self._distances = None

    @property
    def id_to_row(self):
        if self._id_to_row is None:
            self._id_to_row = {use.identifier: i for i, use in enumerate(self.uses)}
        return self._id_to_row

    def distances(
        self,
        ids: List[Tuple[ID, ID]],
        method: Callable = lambda *args, **kwargs: 1 - F.cosine_similarity(*args, **kwargs),
        **kwargs
    ) -> Tensor:
        return torch.stack([
            method(
                self.vectors[self.id_to_row[id1]],
                self.vectors[self.id_to_row[id2]],
                **kwargs
            )
            for id1, id2 in ids
        ])

    @property
    def vectors(self):
        if self._vectors is None:
            self._vectors = self.vectorizer(
                contexts=[u.context_preprocessed for u in self.uses],
                target_indices=[
                    (u.target_index_begin, u.target_index_end) for u in self.uses
                ],
            )
        return self._vectors
