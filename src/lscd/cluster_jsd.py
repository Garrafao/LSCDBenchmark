import scipy
from tqdm import tqdm

from src import wsi
from src.lscd.model import GradedLSCDModel
from src.lemma import Lemma


class ClusterJSD(GradedLSCDModel):
    wsi: wsi.WSIModel

    def predict(self, lemma: Lemma) -> float:
        uses = lemma.get_uses()
        clusters = dict(zip([use.identifier for use in uses], self.wsi.predict(uses)))
        c1, c2 = self.wsi.make_freq_dists(
            clusters, lemma.useid_to_grouping(), lemma.groupings
        )
        return scipy.spatial.distance.jensenshannon(c1, c2, base=2.0)  # type: ignore

    def predict_all(self, lemmas: list[Lemma]) -> list[float]:
        return [self.predict(lemma) for lemma in lemmas]