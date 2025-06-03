import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from logging import getLogger
from pathlib import Path
from typing import Any, TypedDict, _TypedDict
import fnmatch

import numpy as np
import pandas as pd
import requests
import torch.cuda
from deepmistake.deepmistake import DeepMistakeWiC
from deepmistake.utils import DataProcessor, Example
from git import Repo
from pandas import DataFrame
from pydantic import BaseModel, Field, HttpUrl, PrivateAttr
from src.use import Use, UseID
from src.utils import utils
from src.wic.model import WICModel
from tqdm import tqdm

log = getLogger(__name__)


class Model(BaseModel):
    """ """

    name: str
    url: str
    fnpattern: str = None


def use_pair_group(use_pair: tuple[Use, Use]) -> str:
    """ """
    if use_pair[0].grouping != use_pair[1].grouping:
        return "COMPARE"
    else:
        if use_pair[0].grouping == 0 and use_pair[1].grouping == 0:
            return "EARLIER"
        else:
            return "LATER"


def to_data_format(use_pair: tuple[Use, Use]) -> Example:
    """ """
    return Example(
        **{
            "docId": f"{use_pair[0].target}.{np.random.randint(low=100000, high=1000000)}",
            "start_1": use_pair[0].indices[0],
            "end_1": use_pair[0].indices[1],
            "text_1": use_pair[0].context,
            "start_2": use_pair[1].indices[0],
            "end_2": use_pair[1].indices[1],
            "text_2": use_pair[1].context,
            "lemma": use_pair[0].target,
            "pos": "NOUN" if use_pair[0].pos == "NN" else use_pair[0].pos,
            "grp": use_pair_group(use_pair),
            "label": "F",
            "score": -1.0,
        }
    )


class DMWrapperClass(WICModel):
    """ """

    ckpt: Model

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        
        if not self.ckpt_dir.exists() or not (self.ckpt_dir / 'pytorch_model.bin').exists():
            self.init_ckpt()

    def __enter__(self) -> None:
        pass

    def as_df(self) -> DataFrame:
        """ """
        df = pd.json_normalize(data=json.loads(self.json(ensure_ascii=False)))
        return df

    @property
    def path(self) -> Path:
        """ """
        path = os.getenv("DEEPMISTAKE")
        if path is None:
            path = ".deepmistake"
        return utils.path(path)

    @property
    def ckpt_dir(self) -> Path:
        """ """
        return self.path / "checkpoints" / self.ckpt.name
    
    @property
    def dm_model(self) -> DeepMistakeWiC:
        """ """
        print('Loading ', self.ckpt_dir)
        return DeepMistakeWiC(self.ckpt_dir, device="cuda" if torch.cuda.is_available() else 'cpu')
    
    def init_ckpt(self) -> None:
        """ """
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        zipped = self.__download_ckpt()
        self.__unzip_ckpt(zipped)

    def __unzip_ckpt(self, zipped: Path) -> None:
        if zipped.suffix == '.zip':
            with zipfile.ZipFile(file=zipped) as z:
                namelist = z.namelist()[1:]  # remove root element

                for filename in tqdm(
                    namelist, desc="Unzipping checkpoint files", leave=False
                ):
                    filename_p = Path(filename)
                    path = self.ckpt_dir / filename_p.parts[-1]
                    with path.open(mode="wb") as file_obj:
                        shutil.copyfileobj(z.open(filename, mode="r"), file_obj)
        else:
            pat = self.ckpt.fnpattern
            with tarfile.open(zipped) as tar:
                namelist = [m for m in tar.getnames() ]
                for filename in tqdm(namelist, desc="Unzipping checkpoint files", leave=False):
                    if filename.endswith('/'): continue
                    if pat and not fnmatch.fnmatch(filename, pat):
                        print(f'{filename} does not match the pattern {pat}, skipping')
                        continue
                    filename_p = Path(filename)
                    path = self.ckpt_dir / filename_p.parts[-1]
                    src = tar.extractfile(filename)
                    if src is None:
                        print(f'{filename} is not a regular file or cannot be extracted, skipping')
                        continue
                    with path.open(mode="wb") as file_obj:
                        shutil.copyfileobj(src, file_obj)

        zipped.unlink()

    def __download_ckpt(self) -> Path:
        filename = self.ckpt.url.split("/")[-1]
        ckpt_dir = self.ckpt_dir
        path = ckpt_dir / filename
        if path.exists():
            return path
        assert filename.endswith(".zip") or filename.endswith(".tar.gz"), f'Incorrect URL {filename}'
        print('Downloading: ',self.ckpt.url)
        r = requests.get(self.ckpt.url, stream=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        with open(file=path, mode="wb") as f:
            pbar = tqdm(
                desc=f"Downloading checkpoint '{self.ckpt.name}'",
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
        return path

    def predict(self, use_pairs: list[tuple[Use, Use]]) -> list[float]:
        """ """
        if len(use_pairs) == 0:
            return []
        use_pairs_formatted = [to_data_format(up) for up in use_pairs]
        (
            scores,
            preds,
        ) = self.dm_model.predict_examples(use_pairs_formatted, log)
        return scores
