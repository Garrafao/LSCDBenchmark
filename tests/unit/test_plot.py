import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from typing import Callable
# from scipy.stats import spearmanr # not working for replacing Callable

import unittest
from unittest.mock import patch

from src.plots import Plotter

class TestPlotter(unittest.TestCase):
    max_alpha = 0.5
    default_alpha = 0.3
    min_boots_in_one_tail = 2
    mock_y_true = np.array([0.1, 0.2, 0.3])
    mock_y_pred = np.array([0.1, 0.4, 0.3])
    # TODO: Callable() takes no arguments --> when run 'self.metric(y_true, y_pred)' in self.P.metric_boot_histogram()
    metric=Callable(mock_y_true, mock_y_pred)
    P = Plotter(max_alpha=max_alpha, 
                default_alpha=default_alpha, 
                min_boots_in_one_tail=min_boots_in_one_tail, 
                metric=metric)
    # possible solution for Callable, but not working
    # P = Plotter(max_alpha=max_alpha, 
    #             default_alpha=default_alpha, 
    #             min_boots_in_one_tail=min_boots_in_one_tail, 
    #             metric=spearmanr)
    
    def test_validate_alphas(self):
        # when --> not 0 < default_alpha <= max_alpha
        with self.assertRaises(ValueError) as context_1:
            Plotter(max_alpha=0.3, 
                    default_alpha=0.5, 
                    min_boots_in_one_tail=int(), 
                    metric=Callable)
        self.assertIn('alpha=0.5 is outside allowed range: 0 < alpha <= 0.3', str(context_1.exception))
        # when --> default_alpha > max_alpha and 0 < (1 - default_alpha) <= max_alpha
        with self.assertRaises(ValueError) as context_2:
            Plotter(max_alpha=0.7, 
                    default_alpha=0.8, 
                    min_boots_in_one_tail=int(), 
                    metric=Callable)
        self.assertIn('alpha=0.8 > 0.5. Did you mean alpha=0.2?', str(context_2.exception))

    def test__functions(self):
        # test _min_n_boots_from()
        # min_boots_in_one_tail = 2
        # alpha = max_alpha = 0.5
        # int(np.ceil((min_boots_in_one_tail - 1) / (0.5 * alpha) + 1))
        self.assertEqual(self.P._min_n_boots, 5)
        self.assertEqual(self.P._n_boots, 8)

        # test _min_alpha_from()
        # n_boots = 8
        # 2 * (min_boots_in_one_tail - 1) / (n_boots - 1)
        self.assertEqual(self.P._alpha, 0.3)

    def test_dropna(self):
        df = pd.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
                           "toy": [np.nan, 'Batmobile', 'Bullwhip'],
                           "born": [pd.NaT, pd.Timestamp("1940-04-25"), pd.NaT]})
        self.assertEqual(list(self.P.preprocess_inputs(df)['name']), ['Batman'])
        self.assertEqual(list(self.P.preprocess_inputs(df)['toy']), ['Batmobile'])
        self.assertEqual(list(self.P.preprocess_inputs(df)['born']), [pd.Timestamp("1940-04-25")])

    def test_combine_inputs(self):
        mock_labels = {'key_1': 1,
                       'key_2': 2,
                       'key_3': 3}
        mock_preds = {'key_1': 3,
                      'key_2': 5,
                      'key_3': 6}
        merged = self.P.combine_inputs(labels=mock_labels, predictions=mock_preds)
        self.assertEqual(list(merged['target']), ['key_1', 'key_2', 'key_3'])
        self.assertEqual(list(merged['prediction']), [3, 5, 6])
        self.assertEqual(list(merged['label']), [1, 2, 3])
    
    def test__one_boot_and__boot_generator(self):
        one_boot_y_true, one_boot_y_pred = self.P._one_boot(self.mock_y_true, self.mock_y_pred)
        self.assertEqual(len(one_boot_y_true), 3)
        self.assertIsInstance(one_boot_y_pred, np.ndarray)
        self.assertIsInstance(list(self.P._boot_generator(self.mock_y_true, self.mock_y_pred))[0][0], np.ndarray)
        self.assertIsInstance(list(self.P._boot_generator(self.mock_y_true, self.mock_y_pred))[0][1], np.ndarray)

    @patch('src.plots.np.random.randint')
    def test_random_in__one_boot(self, mock_randint):
        one_boot_y_true, one_boot_y_pred = self.P._one_boot(self.mock_y_true, self.mock_y_pred)
        self.assertTrue(mock_randint.called)

    def test_metric_boot_histogram(self):
        # self.P.metric_boot_histogram(self.mock_y_true, self.mock_y_pred)
        # not working 
        pass

    def test___call__(self):
        # also test validate_alphas
        # also test combine_inputs
        # also test metric_boot_histogram
        pass

if __name__ == '__main__':
    unittest.main()

