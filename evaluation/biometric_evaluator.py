import torch
from tqdm import tqdm
import numpy as np

from evaluation.similarity import get_similarity_function

from itertools import combinations, product
from sklearn.metrics import roc_curve


class BiometricEvaluator:

    def __init__(self, dataset, similarity, metrics=['eer'], limit=50, batch_size=32):

        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        self.similarity = similarity
        self.limit = limit



    def __get_features_and_labels(self, model, device):
        all_features = []
        all_labels = []

        model.eval()

        pbar = tqdm(enumerate(self.dataloader))

        with torch.no_grad():
            for step, batch in pbar:

                inputs = batch['image'].to(device)
                labels = batch['label'].view(-1)

                features = model(inputs)

                features = features.view(features.shape[0], features.shape[1]).cpu().numpy()

                all_features.extend([f for f in features])
                all_labels.extend([l for l in labels])

                pbar.set_description_str(f"[{step+1}/{len(self.dataloader)}] - Encoding evaluation set")


        return np.array(all_features), np.array(all_labels)


    def __evaluate(self, features, labels):
        genuine_scores, imposter_scores = compute_comparison_scores(features, labels, self.similarity, self.limit)

        fpr, tpr, threshold = compute_roc(genuine_scores, imposter_scores)
        fnr = 1 - tpr

        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        stats = {
            'eer': eer,
            'eer_threshold': eer_threshold,
        }


        return eer


    def __call__(self, model, epoch, device):

        features, labels = self.__get_features_and_labels(model=model, device=device)

        eer = self.__evaluate(features, labels)

        print("EER:", eer)



def compute_comparison_scores(features, labels, similarity, limit):
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

    return np.array(genuine_scores), np.array(imposter_scores)


def compute_roc(genuine, imposter):
    return roc_curve(np.hstack([np.ones(len(genuine)), np.zeros(len(imposter))]), np.hstack([genuine, imposter]))
