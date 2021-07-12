import numpy as np
import torch
import torch.nn as nn


def getdcg(scores):
    return np.sum(np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)

def getndcgK(model_list, golden_list, K):
    """
    calculate ndcg

    :param Tensor score_list: the scores of items produced by model
    :param Tensor item_list: true item list that the usr interact with
    :return: ndcg score
    """
    model_list = model_list[:K]
    golden_list = golden_list[:K]
    relevant = np.ones_like(golden_list)
    it2rel = {it: rel for it, rel in zip(golden_list, relevant)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in model_list], dtype=np.float32)

    idcg = getdcg(relevant)
    dcg = getdcg(rank_scores)

    if dcg == 0:
        return 0.0
    return idcg / dcg

def getapk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    :param list actual: A list of elements that are to be predicted (order doesn't matter)
    :param list predicted : A list of predicted elements (order does matter)
    :param int k: The maximum number of predicted elements, optional
    :return double score: The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def getmapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    :param list actual: A list of lists of elements that are to be predicted (order doesn't matter in the lists)
    :param list predicted : A list of lists of predicted elements (order does matter in the lists)
    :param int k: The maximum number of predicted elements, optional
    :return double score: The mean average precision at k over the input lists
    """
    return np.mean([getapk(a, p, k) for a, p in zip(actual, predicted)])

def tied_rank(x):
    """
    Computes the tied rank of elements in x.
    This function computes the tied rank of elements in x.
    :param list x: list of numbers, numpy array
    :return list score: the tied rank f each element in x
    """
    sorted_x = sorted(zip(x, range(len(x))))
    r = [0 for k in x]
    cur_val = sorted_x[0][0]
    last_rank = 0
    for i in range(len(sorted_x)):
        if cur_val != sorted_x[i][0]:
            cur_val = sorted_x[i][0]
            for j in range(last_rank, i):
                r[sorted_x[j][1]] = float(last_rank + 1 + i) / 2.0
            last_rank = i
        if i == len(sorted_x) - 1:
            for j in range(last_rank, i + 1):
                r[sorted_x[j][1]] = float(last_rank + i + 2) / 2.0
    return r

def auc(actual, posterior):
    """
    Computes the area under the receiver-operater characteristic (AUC)
    This function computes the AUC error metric for binary classification.
    :param nparray actual: list of binary numbers, The ground truth value
    :param nparray posterior: Defines a ranking on the binary numbers, from most likely to be positive to least likely to be positive.
    :returns double score: The mean squared error between actual and posterior
    """
    r = tied_rank(posterior)
    num_positive = len([0 for x in actual if x == 1])
    num_negative = len(actual) - num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if actual[i] == 1])
    auc = ((sum_positive - num_positive * (num_positive + 1) / 2.0) / (num_negative * num_positive))
    return auc


