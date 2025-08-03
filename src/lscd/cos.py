import numpy as np
import logging
from scipy.spatial import distance
from scipy.special import softmax
from scipy.stats import entropy
from tqdm import tqdm

from src.lscd.model import GradedLSCDModel
from src.lemma import Lemma
from src.use import Use
from src.wic import ContextualEmbedder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Cos(GradedLSCDModel):
    wic: ContextualEmbedder

    def predict(self, lemma: Lemma) -> float:
        earlier_df = lemma.uses_df[lemma.uses_df.grouping == lemma.groupings[0]]
        later_df = lemma.uses_df[lemma.uses_df.grouping == lemma.groupings[1]]

        earlier = [Use.from_series(s) for _, s in earlier_df.iterrows()]
        later = [Use.from_series(s) for _, s in later_df.iterrows()]

        with self.wic:
            earlier_vectors = np.vstack([self.wic.encode(use) for use in earlier])
            later_vectors = np.vstack([self.wic.encode(use) for use in later])

        earlier_avg = earlier_vectors.mean(axis=0)
        later_avg = later_vectors.mean(axis=0)
        cos = distance.cosine(earlier_avg, later_avg)
        return float(cos)
    
    def predict_all(self, lemmas: list[Lemma]) -> list[float]:
        return [self.predict(lemma) for lemma in lemmas]
 
 
class JSDSOFT(GradedLSCDModel):
    wic: ContextualEmbedder

    def predict(self, lemma: Lemma) -> float:
        earlier_df = lemma.uses_df[lemma.uses_df.grouping == lemma.groupings[0]]
        later_df = lemma.uses_df[lemma.uses_df.grouping == lemma.groupings[1]]

        earlier = [Use.from_series(s) for _, s in earlier_df.iterrows()]
        later = [Use.from_series(s) for _, s in later_df.iterrows()]

        with self.wic:
            earlier_vectors = np.vstack([self.wic.encode(use) for use in earlier])
            later_vectors = np.vstack([self.wic.encode(use) for use in later])

        earlier_avg = earlier_vectors.mean(axis=0)
        later_avg = later_vectors.mean(axis=0)
        mixture_avg = np.vstack([earlier_avg, later_avg]).mean(axis=0)
                
        earlier_avg_prob = softmax(earlier_avg)
        later_avg_prob = softmax(later_avg)
        mixture_avg_prob = softmax(mixture_avg)

        #print(earlier_avg[:10])  
        #print(earlier_avg_prob[:10])
        
        JSD = entropy(mixture_avg_prob, base=2.0) - (0.5 * (entropy(earlier_avg_prob, base= 2.0) + entropy(later_avg_prob, base= 2.0)))
        
        logger.info(f"[JSDSOFT] Final score for lemma: {JSD}")      
        
        return float(JSD)
    
    def predict_all(self, lemmas: list[Lemma]) -> list[float]:
        return [self.predict(lemma) for lemma in lemmas]        