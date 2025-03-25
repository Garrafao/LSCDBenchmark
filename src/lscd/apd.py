import numpy as np
from tqdm import tqdm
from functools import partial
from pydantic import Field

from src import wic
from src.lscd.model import GradedLSCDModel
from src.lemma import Lemma, UsePairOptions

class APD(GradedLSCDModel):
    wic: wic.WICModel
    use_pair_options: UsePairOptions = Field(alias="use_pairs")

    def predict(self, lemma: Lemma) -> float:
        """Generates predictions of mean similarity in the compare use pair samples for input 
        lemma.

        :param lemma: lemma instance from data set
        :type lemma: Lemma
        :return: mean of pairwise distances
        :rtype: float
        """         
        use_pairs = lemma.use_pairs(
            group=self.use_pair_options.group, 
            sample=self.use_pair_options.sample
        )
        similarities = self.wic.predict_all(use_pairs)
        return -np.mean(similarities).item()

    def predict_all(self, lemmas: list[Lemma]) -> list[float]:
        use_pairs_nested = [
            lemma.use_pairs(
                group=self.use_pair_options.group, 
                sample=self.use_pair_options.sample
            ) 
            for lemma in tqdm(lemmas, desc="Building use pairs", leave=False)
        ]
        use_pairs = [use_pair for sublist in use_pairs_nested for use_pair in sublist]
        id_pairs = [(use_0.identifier, use_1.identifier) for use_0, use_1 in use_pairs]
        self.wic.predictions = dict(zip(id_pairs, self.wic.predict_all(use_pairs=use_pairs)))
        return [self.predict(lemma) for lemma in tqdm(lemmas, desc="Processing lemmas")]


class DiaSense(GradedLSCDModel):
    wic: wic.WICModel
    use_pair_options: UsePairOptions = Field(alias="use_pairs")

    def predict(self, lemma: Lemma) -> float:
        """Generates predictions of difference between two groups of use pair samples for input 
        lemma. The groups of use pair samples are 'compare' and 'campare + earlier + later'.

        :param lemma: lemma instance from data set
        :type lemma: Lemma
        :return: difference of means of pairwise distances
        :rtype: float
        """        
        use_pairs_0 = lemma.use_pairs(group="COMPARE", sample=self.use_pair_options.sample)
        use_pairs_1 = lemma.use_pairs(group="ALL", sample=self.use_pair_options.sample)
        similarities_0 = self.wic.predict(use_pairs_0)
        similarities_1 = self.wic.predict(use_pairs_1)
        return -(np.mean(similarities_0).item() - np.mean(similarities_1).item())

    def predict_all(self, lemmas: list[Lemma]) -> list[float]:
        use_pairs_nested_0 = [
            lemma.use_pairs(group="ALL", sample=self.use_pair_options.sample) 
            for lemma in tqdm(lemmas, desc="Building use pairs", leave=False)
        ]
        use_pairs_nested_1 = [
            lemma.use_pairs(group="COMPARE", sample=self.use_pair_options.sample) # We need this? Should be contained in ALL group
            
            for lemma in tqdm(lemmas, desc="Building use pairs", leave=False)
        ]
        use_pairs_nested = use_pairs_nested_0 + use_pairs_nested_1 # duplicates?
        use_pairs = [use_pair for sublist in use_pairs_nested for use_pair in sublist]
        id_pairs = [(use_0.identifier, use_1.identifier) for use_0, use_1 in use_pairs]
        self.wic.predictions = dict(zip(id_pairs, self.wic.predict_all(use_pairs=use_pairs)))
        return [self.predict(lemma) for lemma in tqdm(lemmas, desc="Processing lemmas")]
    

class JSDDOT(GradedLSCDModel):
    wic: wic.WICModel
    use_pair_options: UsePairOptions = Field(alias="use_pairs")

    def predict(self, lemma: Lemma) -> float:
        """Generates predictions of estimated JSD based on entropy estimates using the 
        mean of similarity predictions for corresponding groups.

        :param lemma: lemma instance from data set
        :type lemma: Lemma
        :return: difference of means of pairwise distances
        :rtype: float
        """        
        use_pairs_0 = lemma.use_pairs(group="EARLIER", sample=self.use_pair_options.sample)
        use_pairs_1 = lemma.use_pairs(group="LATER", sample=self.use_pair_options.sample)
        use_pairs_2 = lemma.use_pairs(group="ALL", sample=self.use_pair_options.sample)
        similarities_0 = self.wic.predict(use_pairs_0)
        similarities_1 = self.wic.predict(use_pairs_1)
        similarities_2 = self.wic.predict(use_pairs_2)
        return (-np.mean(similarities_2).item()) - (0.5*((-np.mean(similarities_0).item()) + (-np.mean(similarities_1).item())))

    def predict_all(self, lemmas: list[Lemma]) -> list[float]:
        use_pairs_nested = [
            lemma.use_pairs(group="ALL", sample=self.use_pair_options.sample) 
            for lemma in tqdm(lemmas, desc="Building use pairs", leave=False)
        ]
        use_pairs = [use_pair for sublist in use_pairs_nested for use_pair in sublist]
        id_pairs = [(use_0.identifier, use_1.identifier) for use_0, use_1 in use_pairs]
        self.wic.predictions = dict(zip(id_pairs, self.wic.predict_all(use_pairs=use_pairs)))
        return [self.predict(lemma) for lemma in tqdm(lemmas, desc="Processing lemmas")]
