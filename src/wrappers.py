from typing import Literal
import torch
import scipy
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from scipy import stats


def spearmanr(y_true: list[float], y_pred: list[float], **kwargs) -> float:
    corr, _ = stats.spearmanr(a=y_true, b=y_pred, **kwargs)
    return corr

def pearsonr(y_true: list[float], y_pred: list[float], **kwargs) -> float:
    corr, _ = stats.pearsonr(x=y_true, y=y_pred, **kwargs)
    return corr

def l1(v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(input=v, p=1, dim=0)

def l2(v: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(input=v, p=2, dim=0)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - scipy.spatial.distance.cosine(a, b)

def cosine_similarity_cut(a: np.ndarray, b: np.ndarray, thresholds: np.ndarray, labels: np.ndarray) -> float:
    similarities = [1 - scipy.spatial.distance.cosine(a, b)]
    bins = [-np.inf] + thresholds + [np.inf]
    binned_similarities = pd.cut(similarities, bins=bins, labels=labels)
    return binned_similarities

def cosine_similarity_cut_scaled(a: np.ndarray, b: np.ndarray, thresholds: np.ndarray, labels: np.ndarray) -> float:
    assert len(labels) == 4
    similarities = [1 - scipy.spatial.distance.cosine(a, b)]
    bins = [-np.inf] + thresholds + [np.inf]
    similarities_binned_scaled = []
    for s in similarities:
        scaler = MinMaxScaler()  
        if s < bins[1]: # Should be generalized to all numbers of labels
            scaler.fit(np.array([0.0,bins[1]]).reshape(-1, 1) ) 
            base = labels[0]
        elif s < bins[2]:
            scaler.fit(np.array([bins[1],bins[2]]).reshape(-1, 1) ) 
            base = labels[1]
        elif s < bins[3]:
            scaler.fit(np.array([bins[2],bins[3]]).reshape(-1, 1) ) 
            base = labels[2]
        else:
            scaler.fit(np.array([bins[3],1.0]).reshape(-1, 1) )
            base = labels[3]
        s = scaler.transform(np.array([s]).reshape(-1, 1)).flatten()[0]
        s += base
    similarities_binned_scaled.append(s)
    return similarities_binned_scaled
    
def euclidean_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return -scipy.spatial.distance.euclidean(a, b)