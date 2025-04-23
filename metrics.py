import os
import warnings
import numpy as np
import scipy.sparse as ssp
from sklearn.metrics import average_precision_score

__all__ = ['fmax', 'aupr', 'ROOT_GO_TERMS']
ROOT_GO_TERMS = {'GO:0003674', 'GO:0008150', 'GO:0005575'}


def fmax(targets: ssp.csr_matrix, scores: np.ndarray):
    fmax, tmax = 0.0, 0.0
    for t in ((i / 100.0) for i in range(1, 100)):
        t_scores = ssp.csr_matrix((scores >= t).astype(np.int32))
        tp = t_scores.multiply(targets).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # tp + fp scores 中的正例
            tp_sum_fp = t_scores.sum(axis=1)
            # tp + fn labels 中的正例
            tp_sum_fn = targets.sum(axis=1)

            precision = tp / tp_sum_fp
            precision = np.average(precision[np.invert(np.isnan(precision))])

            recall = np.average(tp / tp_sum_fn)

        if np.isnan(precision):
            continue
        try:
            if precision + recall > 0.0:
                f = (2 * precision * recall) / (precision + recall)
                fmax = max(fmax, f)
            else:
                fmax = max(fmax, 0.0)
        except ZeroDivisionError:
            pass
    return fmax


def aupr(targets: ssp.csr_matrix, scores: np.ndarray):
    targets = targets.toarray().flatten()
    scores = scores.flatten()

    # 处理缺失值和无穷大
    targets = np.nan_to_num(targets)
    scores = np.nan_to_num(scores)

    # 转换为 float64 以防数值过大
    targets = targets.astype(np.float64)
    scores = scores.astype(np.float64)

    return average_precision_score(targets, scores)
