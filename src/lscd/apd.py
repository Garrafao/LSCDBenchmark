import numpy as np
import logging
from tqdm import tqdm
from functools import partial
from pydantic import Field

from src import wic
from src.lscd.model import GradedLSCDModel
from src.lemma import Lemma, UsePairOptions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APD(GradedLSCDModel):
    wic: wic.WICModel
    use_pair_options: UsePairOptions = Field(alias="use_pairs")

    def predict(self, lemma: Lemma) -> float:
        logger.info(f"[APD] Starting prediction for lemma: ")
        use_pairs = lemma.use_pairs(
            group=self.use_pair_options.group, 
            sample=self.use_pair_options.sample # sampling here will override sampling and predict_all, more of efficiency problem
        )
        logger.info(f"[APD] Got {len(use_pairs)} use pairs")
        similarities = self.wic.predict_all(use_pairs)
        logger.info(f"[APD] Computed similarities for lemma: ")
        result = -np.mean(similarities).item()
        logger.info(f"[APD] Final score for lemma: {result}")
        return result

    def predict_all(self, lemmas: list[Lemma]) -> list[float]:
        logger.info("[APD] Building use pairs for all lemmas")
        use_pairs_nested = [
            lemma.use_pairs(
                group=self.use_pair_options.group, 
                sample=self.use_pair_options.sample
            ) 
            for lemma in tqdm(lemmas, desc="Building use pairs", leave=False)
        ]
        logger.info("[APD] Flattening use pairs")
        use_pairs = [use_pair for sublist in use_pairs_nested for use_pair in sublist]
        id_pairs = [(use_0.identifier, use_1.identifier) for use_0, use_1 in use_pairs]
        logger.info("[APD] Running batch prediction")
        self.wic.predictions = dict(zip(id_pairs, self.wic.predict_all(use_pairs=use_pairs)))
        logger.info("[APD] Running per-lemma prediction")
        return [self.predict(lemma) for lemma in tqdm(lemmas, desc="Processing lemmas")]


class DiaSense(GradedLSCDModel):
    wic: wic.WICModel
    use_pair_options: UsePairOptions = Field(alias="use_pairs")

    def predict(self, lemma: Lemma) -> float:
        logger.info(f"[DiaSense] Starting prediction for lemma: ")
        use_pairs_0 = lemma.use_pairs(group="COMPARE", sample=self.use_pair_options.sample)
        logger.info(f"[DiaSense] Got {len(use_pairs_0)} COMPARE use pairs")
        use_pairs_1 = lemma.use_pairs(group="ALL", sample=self.use_pair_options.sample)
        logger.info(f"[DiaSense] Got {len(use_pairs_1)} ALL use pairs")
        similarities_0 = self.wic.predict(use_pairs_0)
        similarities_1 = self.wic.predict(use_pairs_1)
        result = -(np.mean(similarities_0).item() - np.mean(similarities_1).item())
        logger.info(f"[DiaSense] Final score for lemma: {result}")
        return result

    def predict_all(self, lemmas: list[Lemma]) -> list[float]:
        logger.info("[DiaSense] Building ALL use pairs for all lemmas")
        use_pairs_nested = [
            lemma.use_pairs(group="ALL", sample=self.use_pair_options.sample) 
            for lemma in tqdm(lemmas, desc="Building use pairs", leave=False)
        ]
        logger.info("[DiaSense] Combining and flattening use pairs")
        use_pairs = [use_pair for sublist in use_pairs_nested for use_pair in sublist]
        uses1, uses2 = zip(*use_pairs)
        id2use = {use.identifier:use for use in set(uses1+uses2)} 
        id_pairs = list(set([(use_0.identifier, use_1.identifier) for use_0, use_1 in use_pairs])) # set mapping added for efficiency, validate
        use_pairs = [(id2use[id_0], id2use[id_1]) for id_0, id_1 in id_pairs] # get subset of use pairs corresponding to id_pairs
        logger.info("[DiaSense] Running batch prediction")
        self.wic.predictions = dict(zip(id_pairs, self.wic.predict_all(use_pairs=use_pairs)))
        logger.info("[DiaSense] Running per-lemma prediction")
        return [self.predict(lemma) for lemma in tqdm(lemmas, desc="Processing lemmas")]


class JSDDOT(GradedLSCDModel):
    wic: wic.WICModel
    use_pair_options: UsePairOptions = Field(alias="use_pairs")

    def predict(self, lemma: Lemma) -> float:
        logger.info(f"[JSDDOT] Starting prediction for lemma: ")
        use_pairs_0 = lemma.use_pairs(group="EARLIER", sample=self.use_pair_options.sample)
        logger.info(f"[JSDDOT] Got {len(use_pairs_0)} EARLIER use pairs")
        use_pairs_1 = lemma.use_pairs(group="LATER", sample=self.use_pair_options.sample)
        logger.info(f"[JSDDOT] Got {len(use_pairs_1)} LATER use pairs")
        use_pairs_2 = lemma.use_pairs(group="ALL", sample=self.use_pair_options.sample)
        logger.info(f"[JSDDOT] Got {len(use_pairs_2)} ALL use pairs")
        similarities_0 = self.wic.predict(use_pairs_0)
        similarities_1 = self.wic.predict(use_pairs_1)
        similarities_2 = self.wic.predict(use_pairs_2)
        result = (-np.mean(similarities_2).item()) - (
            0.5 * ((-np.mean(similarities_0).item()) + (-np.mean(similarities_1).item()))
        )
        logger.info(f"[JSDDOT] Final score for lemma: {result}")
        return result

    def predict_all(self, lemmas: list[Lemma]) -> list[float]:
        logger.info("[JSDDOT] Building ALL use pairs for all lemmas")
        use_pairs_nested = [
            lemma.use_pairs(group="ALL", sample=self.use_pair_options.sample)
            for lemma in tqdm(lemmas, desc="Building use pairs", leave=False)
        ]
        logger.info("[JSDDOT] Flattening use pairs")
        use_pairs = [use_pair for sublist in use_pairs_nested for use_pair in sublist]
        uses1, uses2 = zip(*use_pairs)
        id2use = {use.identifier:use for use in set(uses1+uses2)} 
        id_pairs = list(set([(use_0.identifier, use_1.identifier) for use_0, use_1 in use_pairs])) # set mapping added for efficiency, validate
        use_pairs = [(id2use[id_0], id2use[id_1]) for id_0, id_1 in id_pairs] # get subset of use pairs corresponding to id_pairs
        logger.info("[JSDDOT] Running batch prediction")
        self.wic.predictions = dict(zip(id_pairs, self.wic.predict_all(use_pairs=use_pairs)))
        logger.info("[JSDDOT] Running per-lemma prediction")
        return [self.predict(lemma) for lemma in tqdm(lemmas, desc="Processing lemmas")]
