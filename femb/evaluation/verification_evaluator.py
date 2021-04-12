import torch
from tqdm import tqdm
import numpy as np
import logging

from itertools import combinations, product
from sklearn.metrics import roc_curve

from .similarity import get_similarity_function
from .evaluator import Evaluator


class VerificationEvaluator(Evaluator):

    def __init__(self, similarity, metrics=['eer'], limit=50, batch_size=32):
        self.similarity = similarity
        self.limit = limit


    def evaluate(self, features, labels):
        genuine_scores, imposter_scores = compute_comparison_scores(features, labels, self.similarity, self.limit)

        fpr, tpr, threshold = compute_roc(genuine_scores, imposter_scores)
        fnr = 1 - tpr

        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        stats = {
            'eer': eer,
            'eer_threshold': eer_threshold,
            # add other metrics here
        }
        return stats


def compute_comparison_scores(features, labels, similarity, limit):
    assert len(features) > 0

    similarity = get_similarity_function(similarity)

    distinct_labels = np.unique(labels).astype(int)

    genuine_idxs = {label: np.random.permutation(np.argwhere(labels == label))[:limit] for label in distinct_labels}
    imposter_idxs = {label: np.random.permutation(np.argwhere(labels != label))[:limit] for label in distinct_labels}

    genuine_scores = []
    imposter_scores = []

    for label in distinct_labels:

        for idx0, idx1 in combinations(genuine_idxs[label], r=2):
            score = similarity(features[idx0], features[idx1])
            genuine_scores.append(score)

        for idx0, idx1 in product(genuine_idxs[label], imposter_idxs[label]):
            score = similarity(features[idx0], features[idx1])

            imposter_scores.append(score)

    return np.nan_to_num(np.array(genuine_scores)), np.nan_to_num(np.array(imposter_scores))


def compute_roc(genuine, imposter):
    return roc_curve(np.hstack([np.ones(len(genuine)), np.zeros(len(imposter))]), np.hstack([genuine, imposter]))
