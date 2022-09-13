import csv
import json
import os
import shutil
import zipfile
from fnmatch import translate
from gettext import translation
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pandera as pa
import requests
from pandas import DataFrame
from pandera import Column, DataFrameSchema
from tqdm import tqdm

import src.utils as utils
from src.config import Config, EvaluationTask, Task
from src.target import Target


class Dataset:

    def __init__(self, config: Config):
        self.config = config
        self.groupings = self.config.groupings
        self._stats_groupings = None
        self._uses = None
        self._judgments = None
        self._agreements = None
        self._labels = None
        self._gc_labels = None
        self._bc_labels = None
        self._sp_labels = None
        self._targets = None
        self._wug_to_url = None
        self._path = None
        self.__csv_params = dict(delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE)

        if not self.path.exists():
            if self.config.dataset.name not in self.config.dataset.wug_to_url:
                raise KeyError("dataset could not be found")
            self.path.parent.parent.mkdir(parents=True, exist_ok=True)
            self.__download()
            self.__unzip(self.path.parent.parent)

    @property
    def path(self) -> Path:
        if self._path is None:
            if self.config.dataset.path is not None:
                self._path = self.config.dataset.path
            else:
                self._path = utils.path("wug") / self.config.dataset.name / self.config.dataset.version
        return self._path
            


    @property
    def translation_table(self) -> Dict[str, str]:
        dataset2lang = {
            "dwug_de": "german",
            "dwug_en": "english",
            "dwug_sv": "swedish"
        }
        
        language = dataset2lang.get(self.config.dataset.name)
        if self.config.orthography.normalize:
            translation_table = self.config.orthography.translation_table.get(language, {})
            return translation_table
        return {}
        
        
    @property
    def url(self) -> str:
        return self.config.dataset.wug_to_url[self.config.dataset.name][self.config.dataset.version]
        
    @property
    def __zipped_filename(self) -> Path:
        return utils.path(f"{self.config.dataset.name}-{self.config.dataset.version}.zip")
    
    def __download(self) -> None:
        r = requests.get(self.url, stream=True)
        with open(file=self.__zipped_filename, mode="wb") as f:
            pbar = tqdm(
                desc=f"Downloading dataset '{self.config.dataset.name}' (version {self.config.dataset.version})",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                total=int(r.headers["Content-Length"]),
                leave=False
            )
            pbar.clear()  # clear 0% info
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    pbar.update(len(chunk))
                    f.write(chunk)
            pbar.close()

    def __unzip(self, output_dir: Path) -> None:
        zipped = self.__zipped_filename
        trans_table = {"ó": "ó", "á": "á", "é": "é", "ú": "ú"}

        with zipfile.ZipFile(file=zipped) as z:
            namelist = z.namelist()
            root = output_dir / namelist[0]
            root.mkdir(parents=True, exist_ok=True)

            for filename in tqdm(namelist, desc=f"Unzipping dataset '{self.config.dataset.name}' (version {self.config.dataset.version})"):

                filename_fixed = filename
                for k, v in trans_table.items():
                    filename_fixed = filename_fixed.replace(k, v)

                path = Path(filename_fixed)
                f_parts = list(path.parts)
                
                f_parts[f_parts.index(root.name)] = self.config.dataset.version
                target_path = root.joinpath(*f_parts)
                
                if not filename.endswith("/"):
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with target_path.open(mode="wb") as file_obj:
                        shutil.copyfileobj(z.open(filename, mode="r"), file_obj)
                    
        zipped.unlink()

    @property
    def stats_groupings(self) -> DataFrame:
        if self._stats_groupings is None:
            stats_groupings =  "stats_groupings.csv"
            path = self.path / "stats" / "semeval" / stats_groupings
            if not path.exists():
                path = self.path / "stats" / "opt" / stats_groupings
            if not path.exists():
                path = self.path / "stats" / stats_groupings
            self._stats_groupings = pd.read_csv(path, **self.__csv_params)
        return self._stats_groupings
    
    @property
    def stats_groupings_schema(self) -> DataFrameSchema:
        def validate_grouping(s: str) -> bool:
            parts = s.split("_")
            return len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()

        schema = DataFrameSchema({
            "lemma": Column(str),
            "grouping": Column(str, checks=pa.Check(check_fn=validate_grouping))
        })

        match self.config.evaluation.task:
            case EvaluationTask.GRADED_CHANGE:
                return schema.add_columns({
                    "change_graded": Column(float)
                })
            case EvaluationTask.BINARY_CHANGE:
                return schema.add_columns({
                    "change_binary": Column(float)
                })
            case _:
                return schema

    @property
    def graded_change_labels(self) -> Dict[str, float]:
        if self._gc_labels is None:
            self._gc_labels = dict(zip(
                self.stats_groupings.lemma, 
                self.stats_groupings.change_graded
            )) 
        return self._gc_labels
            
    @property
    def binary_change_labels(self) -> Dict[str, float]:
        if self._bc_labels is None:
            self._bc_labels = dict(zip(
                self.stats_groupings.lemma, 
                self.stats_groupings["change_binary"]
            )) 
        return self._bc_labels
        
    @property
    def semantic_proximity_labels(self) -> Dict[Tuple[str, str], float]:
        if self._sp_labels is None:
            annotated_pairs = list(zip(self.judgments.identifier1, self.judgments.identifier2))
            self._sp_labels = dict(zip(annotated_pairs, self.judgments["judgment"]))
        return self._sp_labels
    
    @property
    def labels(self) -> Dict[str | Tuple[str, str], float]:
        match self.config.evaluation.task:
            case EvaluationTask.GRADED_CHANGE:
                return self.stats_groupings_schema.validate(self.graded_change_labels)
            case EvaluationTask.BINARY_CHANGE:
                return self.stats_groupings_schema.validate(self.binary_change_labels)
            case EvaluationTask.SEMANTIC_PROXIMITY:
                # the judgments should already have been validated
                return self.semantic_proximity_labels

    @property
    def stats_agreement(self) -> DataFrame:
        if self._agreements is None:
            path = self.path / "stats" / "stats_agreement.csv"
            self._agreements = pd.read_csv(path, **self.__csv_params)
        return self._agreements
    
    @property
    def uses(self):
        if self._uses is None:
            self._uses = pd.concat([target.uses for target in self.targets])
        return self._uses

    @property
    def judgments(self):
        if self._judgments is None:
            self._judgments = pd.concat([target.judgments for target in self.targets])
        return self._judgments

    @property
    def clusters(self):
        if self._clusters is None:
            self._clusters = pd.concat([target.clusters for target in self.targets])
        return self._clusters

    @property
    def targets(self) -> List[Target]:
        if self._targets is None:
            to_load = []

            if len(self.config.cleaning.stats) > 0:
                agreements = self.stats_agreement.iloc[1:, :].copy()  # remove "data=full" row
                agreements = self.config.cleaning(agreements)
                to_load = agreements.data.unique().tolist()
            else:
                to_load = list(self.graded_change_labels.keys())

            if self.config.test_targets is not None:
                if isinstance(self.config.test_targets, int):
                    to_load = to_load[:self.config.test_targets]
                elif isinstance(self.config.test_targets, list):
                    to_load = self.config.test_targets
            
            trans_table = self.translation_table
            self._targets = [
                Target(config=self.config, name=target, translation_table=trans_table, path=self.path)
                for target in tqdm(to_load, desc="Building targets", leave=False)
            ]

        return self._targets
