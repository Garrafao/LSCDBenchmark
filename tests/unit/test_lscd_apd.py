import sys
sys.path.insert(0, ".")
import unittest
from unittest.mock import patch
from scipy.spatial.distance import cosine
import os
import pathlib

from src.lscd.apd import APD, DiaSense
from src.wic.contextual_embedder import ContextualEmbedder
from src.lemma import Lemma, UsePairOptions
from src.preprocessing import Lemmatize
from src.dataset import Dataset

class TestAPD(unittest.TestCase):
    WIC = ContextualEmbedder(truncation_tokens_before_target=0.5,
                            similarity_metric=cosine, 
                            ckpt="bert-base-uncased",
                            layers=[1, 12],
                            embedding_cache=None,
                            layer_aggregation="average",
                            subword_aggregation="average",
                            encode_only=False
                            )
    UPO = UsePairOptions(group="EARLIER", sample='all')
    APD = APD(wic=WIC, use_pairs=UPO)

    # test with testwug_en_111, check if dataset exist
    if os.path.isdir('wug/testwug_en_111'):
        print('testwug_en_111 is ready!')
    else:
        print('testwug_en_111 does not exist')
        D = Dataset(path='testwug_en_111', 
                    groupings=tuple(['1', '2']), 
                    type='dev',
                    split='dev',
                    exclude_annotators=[],
                    name='testwug_en_111',
                    test_on=None,
                    cleaning=None,
                    url='https://zenodo.org/record/7946753/files/testwug_en.zip')
        D._Dataset__download_zip()
        print('testwug_en_111 is downloaded!')

    mock_testwug_path_0 = pathlib.Path('wug/testwug_en_111/data/afternoon_nn')
    mock_testwug_path_1 = pathlib.Path('wug/testwug_en_111/data/plane_nn')

    Lem = Lemmatize(spelling_normalization=None) 
    Lemma_0 = Lemma(groupings=('1', '2'), 
              path=mock_testwug_path_0,
              preprocessing=Lem)
    Lemma_1 = Lemma(groupings=('1', '2'), 
              path=mock_testwug_path_1,
              preprocessing=Lem)
    
    def test_predict(self):
        self.assertEqual(0.1879, round(self.APD.predict(lemma=self.Lemma_0), 4))
    
    def test_predict_all(self):
        self.APD.predict_all(lemmas=[self.Lemma_0, self.Lemma_1])
        # TODO: same error for all 'predict_all'
        # TypeError: Object of type 'function' is not JSON serializable

class TestDiaSense(unittest.TestCase):
    WIC = ContextualEmbedder(truncation_tokens_before_target=0.5,
                            similarity_metric=cosine, 
                            ckpt="bert-base-uncased",
                            layers=[1, 12],
                            embedding_cache=None,
                            layer_aggregation="average",
                            subword_aggregation="average",
                            encode_only=False
                            )
    DS = DiaSense(wic=WIC)

    # test with testwug_en_111, check if dataset exist
    if os.path.isdir('wug/testwug_en_111'):
        print('testwug_en_111 is ready!')
    else:
        print('testwug_en_111 does not exist')
        D = Dataset(path='testwug_en_111', 
                    groupings=tuple(['1', '2']), 
                    type='dev',
                    split='dev',
                    exclude_annotators=[],
                    name='testwug_en_111',
                    test_on=None,
                    cleaning=None,
                    url='https://zenodo.org/record/7946753/files/testwug_en.zip')
        D._Dataset__download_zip()
        print('testwug_en_111 is downloaded!')

    mock_testwug_path_0 = pathlib.Path('wug/testwug_en_111/data/afternoon_nn')
    mock_testwug_path_1 = pathlib.Path('wug/testwug_en_111/data/plane_nn')

    Lem = Lemmatize(spelling_normalization=None) 
    Lemma_0 = Lemma(groupings=('1', '2'), 
              path=mock_testwug_path_0,
              preprocessing=Lem)
    Lemma_1 = Lemma(groupings=('1', '2'), 
              path=mock_testwug_path_1,
              preprocessing=Lem)
    
    def test_predict(self):
        self.assertEqual(-0.0041, round(self.DS.predict(lemma=self.Lemma_0), 4))
    
    def test_predict_all(self):
        self.DS.predict_all(lemmas=[self.Lemma_0, self.Lemma_1])
        # TODO: same error for all 'predict_all'
        # TypeError: Object of type 'function' is not JSON serializable

if __name__ == '__main__':
    unittest.main()

