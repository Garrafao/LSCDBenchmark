import csv
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import (
    Dict,
    Literal,
)

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandera import (
    Column,
    DataFrameSchema,
)
from pydantic import (
    BaseModel,
    PrivateAttr,
    validate_arguments,
    DirectoryPath
)

from src.preprocessing import ContextPreprocessor
from src.use import (
    Use,
    UseID,
)

class RandomSampling(BaseModel):
    n: int
    replace: bool

Group = Literal["COMPARE", "EARLIER", "LATER", "ALL"]
Sample = Literal["all", "annotated", "predefined", "all_downsampled"] | RandomSampling

class UsePairOptions(BaseModel):
    group: Group
    sample: Sample

class Lemma(BaseModel):
    """Class representing one lemma in a DWUG-like dataset
    (i.e., one of the words represented as folders in the data/ directory)
    """

    groupings: tuple[str, str]
    """
    Each of the DWUG datasets consists of word usages from multiple groups.
    In most cases, there are only two, which represent time periods. In other
    datasets, there are more than two, in which case they represent regional variations.
    """

    path: DirectoryPath
    """
    The path to the directory containing the corresponding lemma within its dataset.
    Must be a valid existing directory.
    """

    preprocessing: ContextPreprocessor
    """
    A context preprocessing strategy
    """

    _uses_df: DataFrame = PrivateAttr(default=None)
    _annotated_pairs_df: DataFrame = PrivateAttr(default=None)
    _augmented_annotated_pairs: DataFrame = PrivateAttr(default=None)
    _predefined_use_pairs_df: DataFrame = PrivateAttr(default=None)
    _augmented_predefined_use_pairs_df: DataFrame = PrivateAttr(default=None)
    _clusters_df: DataFrame = PrivateAttr(default=None)

    @property
    def name(self) -> str:
        """The name of the lemma, based on instance's path"""
        return self.path.name

    @property
    def uses_df(self) -> DataFrame:
        """Cached property that collects the corresponding uses.csv files,
        as well as preprocesses each use based on the provided configuration.

        :return: The preprocessed DataFrame of uses for the corresponding lemma
        :rtype: DataFrame
        """
        if self._uses_df is None:
            # load uses
            path = self.path / "uses.csv"
            self._uses_df = pd.read_csv(path, delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE)
            # filter by grouping
            self._uses_df.grouping = self._uses_df.grouping.astype(str)
            self._uses_df = self._uses_df[self._uses_df.grouping.isin(self.groupings)]
            # preprocess uses
            self._uses_df = pd.concat(
                [
                    self._uses_df,
                    self._uses_df.apply(self.preprocessing.__call__, axis=1),
                ],
                axis=1,
            )
            self._uses_df['identifier'] = self._uses_df['identifier'].astype(str)
            self._uses_df = self.uses_schema.validate(self._uses_df)
        return self._uses_df

    @property
    def uses_schema(self) -> DataFrameSchema:
        return DataFrameSchema(
            {
                "identifier": Column(dtype=str, unique=True),
                "grouping": Column(dtype=str),
                "context_preprocessed": Column(str),
                "target_index_begin": Column(int),
                "target_index_end": Column(int),
            }
        )

    @property
    def annotated_pairs_df(self) -> DataFrame:
        """Property that collects the annotated pairs of the corresponding lemma
        from its judgments.csv file. It performs validation based on :attr:`annotated_pairs_schema`.

        :return: A DataFrame containing two columns (identifier1, identifier2)
        :rtype: DataFrame
        """
        if self._annotated_pairs_df is None:
            path = self.path / "judgments.csv"
            self._annotated_pairs_df = pd.read_csv(path, delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE, usecols=["identifier1", "identifier2"])
            self._annotated_pairs_df['identifier1'] = self._annotated_pairs_df['identifier1'].astype(str)
            self._annotated_pairs_df['identifier2'] = self._annotated_pairs_df['identifier2'].astype(str)
            self._annotated_pairs_df = self.annotated_pairs_schema.validate(self._annotated_pairs_df)
        return self._annotated_pairs_df

    @property
    def augmented_annotated_pairs_df(self) -> DataFrame:
        """A version of :attr:`annotated_pairs_df` that incorporates grouping information.
        The base :attr:`annotated_pairs_df` is expanded with the groupings oƒ each of the identifiers in each row. 

        :return: The expanded DataFrame
        :rtype: DataFrame
        """
        if self._augmented_annotated_pairs is None:
            self._augmented_annotated_pairs = pd.merge(
                self.annotated_pairs_df,
                self.uses_df,
                left_on="identifier1",
                right_on="identifier",
                how="left",
            )
            self._augmented_annotated_pairs = pd.merge(
                self._augmented_annotated_pairs,
                self.uses_df,
                left_on="identifier2",
                right_on="identifier",
                how="left",
            )
            drop_cols = [col for col in self._augmented_annotated_pairs.columns 
                         if col not in ["identifier1", "identifier2", "grouping_x", "grouping_y"]]
            self._augmented_annotated_pairs.drop(columns=drop_cols, inplace=True)

        return self._augmented_annotated_pairs

    @property
    def annotated_pairs_schema(self) -> DataFrameSchema:
        """Schema for validating that a judgments.csv file contains two columns (identifier1, identifier2)


        :return: The schema
        :rtype: DataFrameSchema
        """
        return DataFrameSchema(
            {
                "identifier1": Column(dtype=str),
                "identifier2": Column(dtype=str),
            }
        )

    def useid_to_grouping(self) -> Dict[UseID, str]:
        """Method to generate a dictionary from use identifiers to their corresponding groupings

        :return: A dictionary from use identifiers to use groupings
        :rtype: Dict[UseID, str]
        """
        return dict(zip(self.uses_df.identifier, self.uses_df.grouping))

    def grouping_to_useid(self) -> dict[str, list[UseID]]:
        """Method to generate a dictionary from use groupings to a 
        list of use identifiers corresponding to that grouping

        :return: A dictionary from groupings to list of use identifier
        :rtype: dict[str, list[UseID]]
        """
        grouping_to_useid = defaultdict(list)
        for useid, grouping in self.useid_to_grouping().items():
            grouping_to_useid[grouping].append(useid)
        return dict(grouping_to_useid)

    def _split_compare_uses(self) -> tuple[list[UseID], list[UseID]]:
        ids1 = self.uses_df[self.uses_df.grouping == self.groupings[0]]
        ids2 = self.uses_df[self.uses_df.grouping == self.groupings[1]]
        return ids1.identifier.tolist(), ids2.identifier.tolist()

    def _split_earlier_uses(self) -> tuple[list[UseID], list[UseID]]:
        ids = self.uses_df[self.uses_df.grouping == self.groupings[0]]
        return ids.identifier.tolist(), ids.identifier.tolist()

    def _split_later_uses(self) -> tuple[list[UseID], list[UseID]]:
        ids = self.uses_df[self.uses_df.grouping == self.groupings[1]]
        return ids.identifier.tolist(), ids.identifier.tolist()

    def split_uses(self, group: Group) -> tuple[list[UseID], list[UseID]]:
        """Splits the uses of a lemma into two separate lists of use identifiers, according to `pairing`

        :param group: A pairing strategy
        :type group: Group
        :return: _description_
        :rtype: tuple[list[UseID], list[UseID]]
        """
        match group:
            case "COMPARE":
                return self._split_compare_uses()
            case "EARLIER":
                return self._split_earlier_uses()
            case "LATER":
                return self._split_later_uses()
            case "ALL":
                compare_0, compare_1 = self._split_compare_uses()
                earlier_0, earlier_1 = self._split_earlier_uses()
                later_0, later_1 = self._split_later_uses()
                return (
                    compare_0 + earlier_0 + later_0, # this makes only limited sense, see below
                    compare_1 + earlier_1 + later_1
                )                

    def get_uses(self) -> list[Use]:
        return [Use.from_series(row) for _, row in self.uses_df.iterrows()]

    def use_pairs(self, group: Group, sample: Sample) -> list[tuple[Use, Use]]:
        match (sample, group):
            case ("annotated", p):
                ids1, ids2 = self._split_augmented_uses(p, self.augmented_annotated_pairs_df)
                use_pairs = list(zip(ids1, ids2))
            case ("predefined", p):
                ids1, ids2 = self._split_augmented_uses(p, self.augmented_predefined_use_pairs_df)
                use_pairs = list(zip(ids1, ids2))
            case ("all", p):
                ids1, ids2 = self.split_uses(p)
                use_pairs = list(product(ids1, ids2)) # in combination with split_uses this can lead to duplicate pairs, assert here that there are no duplicate pairs
            case ("all_downsampled", p): # this first downsamples uses randomly to equal number
                ids1 = self.uses_df[self.uses_df.grouping == self.groupings[0]]
                ids2 = self.uses_df[self.uses_df.grouping == self.groupings[1]]
                if len(ids1)>len(ids2):
                    ids1 = ids1.sample(n=len(ids2), replace=False)
                elif len(ids1)<len(ids2):
                    ids2 = ids2.sample(n=len(ids1), replace=False)
                else:
                    pass
                self._uses_df = pd.concat([ids1,ids2],ignore_index=True)
                ids1, ids2 = self.split_uses(p)
                use_pairs = list(product(ids1, ids2)) # in combination with split_uses this can lead to duplicate pairs, assert here that there are no duplicate pairs
            case (sampled, p):
                assert isinstance(sampled, RandomSampling)
                ids1, ids2 = self.split_uses(p)
                ids1 = [np.random.choice(ids1, replace=sampled.replace) for _ in range(sampled.n)] # there is a potential bug here
                ids2 = [np.random.choice(ids2, replace=sampled.replace) for _ in range(sampled.n)] # there is a potential bug here
                use_pairs = list(zip(ids1, ids2))

        use_pairs_instances = []
        for id1, id2 in use_pairs: # could be done more efficiently
            u1 = Use.from_series(self.uses_df[self.uses_df.identifier == id1].iloc[0])
            u2 = Use.from_series(self.uses_df[self.uses_df.identifier == id2].iloc[0])
            use_pairs_instances.append((u1, u2))

        return use_pairs_instances

    @property
    def predefined_use_pairs_df(self) -> DataFrame:
        if self._predefined_use_pairs_df is None:
            self._predefined_use_pairs_df = pd.read_csv(self.path / "use_pairs.csv", encoding="utf8", delimiter="\t", quoting=csv.QUOTE_NONE)
        return self._predefined_use_pairs_df
    
    @property
    def augmented_predefined_use_pairs_df(self) -> DataFrame:
        if self._augmented_predefined_use_pairs_df is None:
            self._augmented_predefined_use_pairs_df = self.predefined_use_pairs_df.merge(
                right=self.uses_df,
                left_on="identifier1",
                right_on="identifier",
                how="left",
            )
            self._augmented_predefined_use_pairs_df = self._augmented_predefined_use_pairs_df.merge(
                right=self.uses_df,
                left_on="identifier2",
                right_on="identifier",
                how="left",
            )

            drop_cols = [col for col in self._augmented_predefined_use_pairs_df.columns 
                        if col not in ["identifier1", "identifier2", "grouping_x", "grouping_y"]]
            self._augmented_predefined_use_pairs_df.drop(columns=drop_cols, inplace=True)
        return self._augmented_predefined_use_pairs_df
        
    def _split_augmented_uses(self, group: Group, augmented_uses: DataFrame) -> tuple[list[UseID], list[UseID]]:
        if group == "ALL":
            compare_0, compare_1 = self._split_augmented_uses("COMPARE", augmented_uses)
            earlier_0, earlier_1 = self._split_augmented_uses("EARLIER", augmented_uses)
            later_0, later_1 = self._split_augmented_uses("LATER", augmented_uses)
            return (
                compare_0 + earlier_0 + later_0,
                compare_1 + earlier_1 + later_1
            )

        match group:
            case "COMPARE":
                group_0, group_1 = self.groupings
            case "EARLIER":
                group_0, group_1 = self.groupings[0], self.groupings[0]
            case "LATER":
                group_0, group_1 = self.groupings[1], self.groupings[1]

        filtered = augmented_uses[
            (augmented_uses.grouping_x == group_0)
            & (augmented_uses.grouping_y == group_1)
        ]

        return (
            filtered.identifier1.tolist(),
            filtered.identifier2.tolist(),
        )
