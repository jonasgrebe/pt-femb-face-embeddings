import numpy as np


def l2_sim(x, y):
    return 1. / (1 + np.linalg.norm(y-x))


def cos_sim(x, y):
    x, y = x.reshape(-1), y.reshape(-1)
    return np.dot(x, y) / (1e-16 + np.linalg.norm(x) * np.linalg.norm(y))


def get_similarity_function(similarity):
    similarity_fcts = {
        'l2': l2_sim,
        'cos': cos_sim
    }

    return similarity_fcts[similarity]
