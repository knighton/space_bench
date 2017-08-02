import numpy as np
from wurlitzer import pipes

from ..space import Space


with pipes(stdout=None):
    import nmslib


class NMSLibSpace(Space):
    name = 'nmslib'

    def __init__(self, ids, vectors):
        super().__init__(ids, vectors)
        with pipes(stdout=None):
            self.index = nmslib.init(method='hnsw', space='cosinesimil')
            self.index.addDataPointBatch(vectors)
            self.index.createIndex({'post': 2}, print_progress=False)

    def get_nearest(self, vector, limit):
        indexes, dists = self.index.knnQuery(vector, limit)
        for i, index in enumerate(indexes):
            indexes[i] = self.ids[index]
        return indexes, dists

    def batch_get_nearest(self, vectors, limit, num_threads=0):
        pairs = self.index.knnQueryBatch(vectors, limit, num_threads)
        id_lists = []
        dist_lists = []
        for indexes, dists in pairs:
            ids = []
            for index in indexes:
                ids.append(self.ids[index])
            id_lists.append(ids)
            dist_lists.append(dists)
        id_matrix = np.array(id_lists)
        dist_matrix = np.array(dist_lists)
        return id_matrix, dist_matrix
