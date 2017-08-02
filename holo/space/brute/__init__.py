import numpy as np

from ..space import Space


def normalize_l2(vector):
    norm = np.linalg.norm(vector)
    # assert np.isfinite(norm) and norm != 0.
    return vector / norm


def distance_l2(a, b):
    return np.linalg.norm(a - b)


class BruteSpace(Space):
    name = 'brute'

    def __init__(self, ids, vectors):
        super().__init__(ids, vectors)
        normed = []
        for vector in vectors:
            normed.append(normalize_l2(vector))
        self.normed = np.array(normed)

    def get_nearest(self, vector, limit):
        query = normalize_l2(vector)
        dists_indexes = []
        for i, vector in enumerate(self.normed):
            dist = distance_l2(query, vector)
            dists_indexes.append((dist, i))
        dists_indexes.sort()
        dists = []
        ids = []
        for dist, index in dists_indexes:
            dists.append(dist)
            ids.append(self.ids[index])
        return np.array(ids), np.array(dists)
