from enum import Enum
import csv
import pickle
import json
import os
import shutil
import tempfile
import uuid
import zipfile
import yaml
from pathlib import Path
from typing import Any, Literal, TypedDict
import numpy as np
from git import Repo
import gdown

import pandas as pd
import pandera as pa
import requests
from pandas import (
    DataFrame,
    Series,
)
from pandera import (
    Column,
    DataFrameSchema,
)
from pydantic import BaseModel, PrivateAttr, HttpUrl, Field, validator
from tqdm import tqdm
from src.use import Use, UseID

import src.utils.utils as utils
from src.cleaning import Cleaning
from src.evaluation import EvaluationTask
from src.preprocessing import ContextPreprocessor, Raw
from src.lemma import Group, Lemma, Sample, UsePairOptions
import uuid


class StandardSplit(BaseModel):
    dev: list[str]
    dev1: list[str]
    dev2: list[str]
    test: list[str]
    full: list[str]


class UsePairCache(BaseModel):
    dataset: str
    split: str
    group: str
    sample: str


class Dataset(BaseModel):
    groupings: tuple[str, str]
    type: Literal["dev", "test"]
    split: Literal["dev", "dev1", "dev2", "test", "full"]
    exclude_annotators: list[str]
    path: Path
    name: str
    url: HttpUrl | None = Field(default=None)
    commit: str | None = Field(default=None)
    path_in_repo: Path | None = Field(default=None)
    standard_split: StandardSplit | None = Field(default=None)
    test_on: set[str] | int | None = Field(...)
    cleaning: Cleaning | None = Field(...)
    preprocessing: ContextPreprocessor | None = Field(default=None)
    wic_use_pairs: UsePairOptions | None = Field(default=None)

    _stats_groupings: DataFrame = PrivateAttr(default=None)
    _uses: DataFrame = PrivateAttr(default=None)
    _judgments: DataFrame = PrivateAttr(default=None)
    _agreements: DataFrame = PrivateAttr(default=None)
    _clusters: DataFrame = PrivateAttr(default=None)
    _lemmas: list[Lemma] = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)

        if not self.absolute_path.exists():
            self.__download()

        if self.standard_split is None:
            self.standard_split = self.get_standard_split()
            path = utils.path("conf") / "dataset" / f"{self.name}.yaml"
            with path.open(mode="r", encoding="utf8") as f:
                config = yaml.safe_load(f)
                config["standard_split"] = self.standard_split.dict()
            self.rewrite_config(new_config=config, path=path)
          
    @validator("preprocessing", always=True, pre=True)
    def set_preprocessing(cls, v) -> ContextPreprocessor:
        return v or Raw(spelling_normalization=None)

    def rewrite_config(self, new_config: dict[str, Any], path: Path) -> None:
        with path.open(mode="w", encoding="utf8") as f:
            yaml.safe_dump(new_config, f, encoding="utf8", allow_unicode=True, default_flow_style=False)

    @property
    def relative_path(self) -> Path:
        return self.path

    @property
    def absolute_path(self) -> Path:
        return self.data_dir / self.path

    @property
    def data_dir(self) -> Path:
        root = os.getenv("DATA_DIR")
        if root is None:
            root = "wug"
        return utils.path(root)

    # def __download_from_git(self) -> None:
    #     assert self.url is not None
    #     if(self.path_in_repo and self.commit is not None):
    #         self.data_dir.mkdir(parents=True, exist_ok=True)
    #         Repo.clone_from(url=self.url, to_path=self.data_dir / self.relative_path.parts[0])
    #         repo = Repo(self.data_dir / self.relative_path.parts[0])
    #         repo.git.checkout(self.commit)
    #         shutil.copytree(repo.working_dir / self.path_in_repo, self.absolute_path)
    #     else:
    #         Repo.clone_from(url=self.url, to_path=self.data_dir / self.relative_path.parts[0])
    def __download_from_git(self) -> None:
        assert self.url is not None, "URL must be provided."
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            
            repo_dir = tmp_dir_path / "repo"
            Repo.clone_from(url=self.url, to_path=repo_dir)
            
            if self.commit:
                repo = Repo(repo_dir)
                repo.git.checkout(self.commit)
            
            if self.path_in_repo:
                source_path = repo_dir / self.path_in_repo
                if source_path.exists():
                    shutil.copytree(source_path, self.absolute_path, dirs_exist_ok=True)
                else:
                    raise FileNotFoundError(f"Path {source_path} does not exist in the repository.")
            else:
                shutil.copytree(repo_dir, self.absolute_path, dirs_exist_ok=True)

    def __download_zip(self) -> None:
        assert self.url is not None
        
        if self.url.startswith("https://drive.google.com"):
            self.data_dir.mkdir(parents=True, exist_ok=True)
            zipped = self.absolute_path.with_suffix(".zip")
            gdown.download(self.url, str(zipped), quiet=False, fuzzy=True)
            shutil.unpack_archive(str(zipped), self.absolute_path)
            
        else:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            zipped = self.absolute_path.with_suffix(".zip")
            r = requests.get(self.url, stream=True)
            with open(file=zipped, mode="wb") as f:
                pbar = tqdm(
                    desc=f"Downloading dataset '{self.name}'",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    total=int(r.headers["Content-Length"]),
                    leave=False,
                )
                pbar.clear()  # clear 0% info
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        pbar.update(len(chunk))
                        f.write(chunk)
                pbar.close()
            self.__unzip(zip_file=zipped)

    def __download(self) -> None:
        assert self.url is not None, f"Could not find a download URL for dataset `{self.name}`"
        if self.url.endswith(".git"):
            self.__download_from_git()
        else:
            self.__download_zip()
        
        if self._check_duplicate_identifiers():
            self._patch_identifiers()
        else:
            print(f"No duplicates found; identifiers not patched for dataset: {self.name}")


    def __unzip(self, zip_file: Path) -> None:
        trans_table = {"ó": "ó", "á": "á", "é": "é", "ú": "ú"}
        self.absolute_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(file=zip_file) as z:
            namelist = z.namelist()
            root = self.data_dir
            self.absolute_path.mkdir(parents=True, exist_ok=True)

            for filename in tqdm(
                namelist,
                desc=f"Unzipping dataset '{self.name}'",
                leave=False
            ):

                filename_fixed = filename
                for k, v in trans_table.items():
                    filename_fixed = filename_fixed.replace(k, v)

                path = Path(filename_fixed)
                f_parts = list(path.parts)

                f_parts[0] = self.name
                target_path = root.joinpath(*f_parts)

                if not filename.endswith("/"):
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with target_path.open(mode="wb") as file_obj:
                        shutil.copyfileobj(z.open(filename, mode="r"), file_obj)

        zip_file.unlink()

    @property
    def stats_groupings_df(self) -> DataFrame:
        if self._stats_groupings is None:
            stats_groupings = "stats_groupings.csv"
            path = self.absolute_path / "stats" / "semeval" / stats_groupings
            if not path.exists():
                path = self.absolute_path / "stats" / "opt" / stats_groupings
            if not path.exists():
                path = self.absolute_path / "stats" / stats_groupings
            self._stats_groupings = pd.read_csv(
                path, delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE
            )
        return self._stats_groupings

    @stats_groupings_df.setter
    def stats_groupings_df(self, other: DataFrame) -> None:
        self._stats_groupings = other

    def get_stats_groupings_schema(
        self, evaluation_task: EvaluationTask
    ) -> DataFrameSchema:
        """
        Examples:
        >>> self.add(1, 2)
        """
        def validate_grouping(s: Series) -> bool:
            for _, item in s.items():
                parts = item.split("_")
                if len(parts) != 2:
                    return False
            return True

        schema = DataFrameSchema(
            {
                "lemma": Column(str),
                "grouping": Column(str, checks=pa.Check(check_fn=validate_grouping)),
            }
        )

        match evaluation_task:
            case "change_graded":
                return schema.add_columns({"change_graded": Column(float)})
            case "change_binary":
                return schema.add_columns({"change_binary": Column(int)})
            case "COMPARE":
                return schema.add_columns({"COMPARE": Column(float)})
            case _:
                return schema

    @property
    def graded_change_labels(self) -> dict[str, float]:
        stats_groupings = self.get_stats_groupings_schema("change_graded").validate(
            self.stats_groupings_df
        )
        return dict(zip(stats_groupings.lemma, stats_groupings.change_graded))

    @property
    def compare_labels(self) -> dict[str, float]:
        stats_groupings = self.get_stats_groupings_schema("COMPARE").validate(
            self.stats_groupings_df
        )
        return dict(zip(stats_groupings.lemma, stats_groupings.COMPARE))

    @property
    def binary_change_labels(self) -> dict[str, int]:
        stats_groupings = self.get_stats_groupings_schema("change_binary").validate(
            self.stats_groupings_df
        )
        return dict(zip(stats_groupings.lemma, stats_groupings.change_binary))

    @property
    def wic_labels(self) -> dict[tuple[UseID, UseID], float]:
        judgments = self.judgments_df[
            ~self.judgments_df["annotator"].isin(self.exclude_annotators)
        ].copy(deep=True)
        judgments.replace(to_replace=0, value=np.nan, inplace=True)
        judgments['i1_i2_pair'] = judgments.apply(lambda row: tuple(sorted([row['identifier1'], row['identifier2']])), axis=1)
        judgments = (
            judgments.groupby(by=['i1_i2_pair'])["judgment"]
            .median()
            .reset_index()
        )
        
        duplicates = judgments.duplicated(subset=['i1_i2_pair'], keep=False)
        return {
            (row['i1_i2_pair'][0], row['i1_i2_pair'][1]): row['judgment']
            for row in judgments.to_dict('records')
        }

    @property
    def binary_wic_labels(self) -> dict[tuple[UseID, UseID], float]:
        labels = self.wic_labels
        return {
            use_pair: judgment
            for use_pair, judgment in labels.items()
            if judgment in [4.0, 1.0]
        }

    @property
    def wsi_labels(self) -> dict[str, int]:
        clusters = self.clusters_df.replace(-1, np.nan)
        return dict(zip(clusters.identifier, clusters.cluster))

    def use_pairs(self, group: Group, sample: Sample) -> list[tuple[Use, Use]]:
        index_path = utils.path(".use_pairs") / "index.parquet"
        try:
            index = pd.read_parquet(index_path)
        except FileNotFoundError:
            index = pd.DataFrame(columns=["dataset", "split", "id", "group", "sample"])

        query = pd.DataFrame([{"dataset": self.name, "split": self.split, "group": group, "sample": sample}])
        df = index.merge(query)


        if df.empty or self.test_on is not None:
            use_pairs = [
                use_pair 
                for lemma in tqdm(self.filter_lemmas(self.lemmas), desc="Retrieving each lemma's use pairs", leave=False)
                for use_pair in lemma.use_pairs(group=group, sample=sample)
            ]
            if self.test_on is None:
                while True:
                    identifier = str(uuid.uuid4())
                    if identifier not in index.id.tolist():
                        index_path.parent.mkdir(exist_ok=True, parents=True)
                        index = pd.concat(
                            [
                                index,
                                pd.DataFrame([{
                                    "dataset": self.name,
                                    "split": self.split,
                                    "id": identifier,
                                    "group": group,
                                    "sample": sample
                                }])
                            ],
                            ignore_index=True,
                        )
                        index.to_parquet(index_path, index=False)
                        with open(file=index_path.parent / f"{identifier}.pkl", mode="wb") as f:
                            pickle.dump(use_pairs, f)

                        break
        else:
            id = df.id.iloc[0]
            with open(file=index_path.parent / f"{id}.pkl", mode="rb") as f:
                use_pairs = pickle.load(f)
        use_pairs = list(set(use_pairs))
        return use_pairs


        

    def get_labels(self, evaluation_task: EvaluationTask) -> dict[Any, Any]:
        # the get_*_labels methods return dictionaries from targets, identifiers or tuples of identifiers to labels
        # to be able to return the correct subset, we need the `keys` parameter
        # this value should be a list returned by any of the models
        match evaluation_task:
            case "change_graded":
                return self.graded_change_labels
            case "change_binary":
                return self.binary_change_labels
            case "compare":
                return self.compare_labels
            case "wic":
                return self.wic_labels
            case "binary_wic":
                return self.binary_wic_labels
            case "wsi":
                return self.wsi_labels
            case _:
                raise ValueError

    @property
    def stats_agreement_df(self) -> DataFrame:
        if self._agreements is None:
            path = self.absolute_path / "stats" / "stats_agreement.csv"
            self._agreements = pd.read_csv(
                path, delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE
            )
        return self._agreements

    @property
    def uses_df(self):
        if self._uses is None:
            self._uses = pd.concat([target.uses_df for target in self.lemmas])
        return self._uses

    @property
    def judgments_df(self):
        if self._judgments is None:
            tables = []
            for lemma in self.lemmas:
                path = self.absolute_path / "data" / lemma.name / "judgments.csv"
                judgments = pd.read_csv(
                    path, delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE
                )
                judgments["judgment"] = judgments["judgment"].astype(float)
                judgments["annotator"] = judgments["annotator"].astype(str)
                judgments.dropna(subset=["judgment"], inplace=True)
                judgments['identifier1'] = judgments['identifier1'].astype(str)
                judgments['identifier2'] = judgments['identifier2'].astype(str)
                judgments = self.judgments_schema.validate(judgments)
                tables.append(judgments)
            self._judgments = pd.concat(tables)
        return self._judgments
    

    @property
    def judgments_schema(self) -> DataFrameSchema:
        return DataFrameSchema(
            {
                "identifier1": Column(dtype=str),
                "identifier2": Column(dtype=str),
                "judgment": Column(dtype=float),
                "annotator": Column(dtype=str),
            }
        )

    @property
    def clusters_df(self):
        if self._clusters is None:
            tables = []
            for lemma in self.lemmas:
                path = self.absolute_path / "clusters" / "opt" / f"{lemma.name}.csv"
                clusters = pd.read_csv(
                    path, delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE
                )
                clusters = self.clusters_schema.validate(clusters)
                tables.append(clusters)
            self._clusters = pd.concat(tables)
        return self._clusters

    @property
    def clusters_schema(self) -> DataFrameSchema:
        return DataFrameSchema(
            {"identifier": Column(dtype=str, unique=True), "cluster": Column(int)}
        )

    def get_standard_split(self) -> StandardSplit:
        all_lemmas = [folder.name for folder in (self.absolute_path / "data").iterdir()]
        match self.type:
            case "dev":
                dev1 = Series(all_lemmas).sample(frac=0.5, replace=False).tolist()
                dev2 = list(set(all_lemmas).difference(dev1))
                return StandardSplit(dev=all_lemmas, dev1=dev1, dev2=dev2, test=[], full=all_lemmas)
            case "test":
                return StandardSplit(dev=[], dev1=[], dev2=[], test=all_lemmas, full=all_lemmas)

    def get_split(self) -> list[str]:
        assert self.standard_split is not None
        match self.split:
            case "full":
                return self.standard_split.full
            case "dev1":
                return self.standard_split.dev1
            case "dev2":
                return self.standard_split.dev2
            case "test":
                return self.standard_split.test
            case "dev":
                return self.standard_split.dev

    # def filter_lemmas(self, lemmas: list[Lemma]) -> list[Lemma]:
    #     if utils.is_str_set(self.test_on):
    #         keep = self.test_on
    #     elif utils.is_int(self.test_on):
    #         keep = set([lemma.name for lemma in lemmas[: self.test_on]])
    #     else:
    #         # keep all lemma names
    #         # keep = set([lemma.name for lemma in self.lemmas])
    #         keep = set(self.get_split())
    #         if self.cleaning is not None and len(self.cleaning.stats) > 0:
    #             # remove "data=full" row
    #             agreements = self.stats_agreement_df.iloc[1:, :].copy()
    #             agreements = self.cleaning(agreements)
    #             keep = keep.intersection(agreements.data.unique().tolist())

    #     return [lemma for lemma in lemmas if lemma.name in keep]

    def filter_lemmas(self, lemmas: list[Lemma]) -> list[Lemma]:
        print(f"Number of input lemmas: {len(lemmas)}")
        if not lemmas:
            print("Warning: Input lemmas list is empty.")

        if utils.is_str_set(self.test_on):
            keep = self.test_on
        elif utils.is_int(self.test_on):
            keep = set([lemma.name for lemma in lemmas[: self.test_on]])
        else:
            # Debug: Check self.get_split()
            split_lemmas = self.get_split()
            keep = set(split_lemmas)

            # Debug: Check self.cleaning
            if self.cleaning is not None and len(self.cleaning.stats) > 0:
                agreements = self.stats_agreement_df.iloc[1:, :].copy()
                agreements = self.cleaning(agreements)
                keep = keep.intersection(agreements.data.unique().tolist())

        # Debug: Check the lemma names in the input lemmas list
        lemma_names = [lemma.name for lemma in lemmas]
   

        # Filter lemmas based on the keep set
        filtered_lemmas = [lemma for lemma in lemmas if lemma.name in keep]
        return filtered_lemmas

    @property
    def lemmas(self) -> list[Lemma]:
        """Returns the list of lemmas in the dataset

        Returns:
            list[Lemma]: _description_

        Examples
        --------
        >>> np.angle([1.0, 1.0j, 1+1j])               # in radians
        array([ 0.        ,  1.57079633,  0.78539816]) # may vary
        >>> np.angle(1+1j, deg=True)                  # in degrees
        45.0
        """
        
        if self._lemmas is None:
            to_load = [folder for folder in (self.absolute_path / "data").iterdir()]
            assert self.preprocessing is not None, TypeError("Preprocessing should never be None")
            self._lemmas = [
                Lemma(
                    path=target,
                    groupings=self.groupings,
                    preprocessing=self.preprocessing,
                )
                for target in to_load
            ]

        return self._lemmas
    
    def _patch_identifiers(self):
        """Add lemma name to identifiers in uses.csv and judgments.csv after download."""
        data_path = self.absolute_path / "data"
        for lemma_dir in data_path.iterdir():
            if not lemma_dir.is_dir():
                continue

            lemma = lemma_dir.name
            uses_path = lemma_dir / "uses.csv"
            judgments_path = lemma_dir / "judgments.csv"

            if uses_path.exists():
                uses_df = pd.read_csv(uses_path, delimiter="\t")
                uses_df["identifier"] = f"{lemma}::" + uses_df["identifier"].astype(str)
                uses_df["lemma"] = lemma
                uses_df.to_csv(uses_path, sep="\t", index=False)

            if judgments_path.exists():
                judgments_df = pd.read_csv(judgments_path, delimiter="\t")
                judgments_df["identifier1"] = f"{lemma}::" + judgments_df["identifier1"].astype(str)
                judgments_df["identifier2"] = f"{lemma}::" + judgments_df["identifier2"].astype(str)
                judgments_df["lemma"] = lemma
                judgments_df.to_csv(judgments_path, sep="\t", index=False)

        print(f"Patched identifiers for dataset: {self.name}")

    def _check_duplicate_identifiers(self) -> bool:
        data_path = self.absolute_path / "data"
        pairs_dict = {}

        for lemma_dir in data_path.iterdir():
            if lemma_dir.is_dir():
                lemma = lemma_dir.name
                judgments_path = lemma_dir / "judgments.csv"
                
                if judgments_path.exists():
                    df = pd.read_csv(judgments_path, delimiter="\t", dtype=str)

                    for _, row in df.iterrows():
                        pair = (row["identifier1"], row["identifier2"])
                        pairs_dict.setdefault(pair, set()).add(lemma)

        duplicates = [
            (pair, lemmas) for pair, lemmas in pairs_dict.items() if len(lemmas) > 1
        ]

        if duplicates:
            print(f"Duplicate identifier pairs found across lemmas in dataset '{self.name}':")
            for (identifier1, identifier2), lemmas in duplicates:
                lemmas_list = ', '.join(lemmas)
                print(f"  - Pair ({identifier1}, {identifier2}) appears in lemmas: {lemmas_list}")
            return True  # Duplicates found
        else:
            print(f"No duplicate identifier pairs found across lemmas in dataset '{self.name}'.")
            return False  # No duplicates found
