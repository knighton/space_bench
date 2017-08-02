from copy import deepcopy
import numpy as np


class Space(object):
    def __init__(self, ids, vectors):
        self.ids = deepcopy(ids)
        self.vectors = deepcopy(vectors)

    def get_nearest(self, vector, limit):
        """
        vector, limit -> ids, dists
        """
        raise NotImplementedError

    def batch_get_nearest(self, vectors, limit):
        id_lists = []
        dist_lists = []
        for vector in vectors:
            ids, dists = self.get_nearest(vector, limit)
            id_lists.append(ids)
            dist_lists.append(dists)
        id_lists = np.array(id_lists)
        dist_lists = np.array(dist_lists)
        return id_lists, dist_lists
