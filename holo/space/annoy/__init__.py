from annoy import AnnoyIndex
import numpy as np

from ..space import Space


class AnnoySpace(Space):
    name = 'annoy'

    def __init__(self, ids, vectors, k=32):
        super().__init__(ids, vectors)
        self.index = AnnoyIndex(vectors.shape[1], metric='angular')
        for i, vector in enumerate(vectors):
            self.index.add_item(i, vector)
        self.index.build(k)

    def get_nearest(self, vector, limit):
        indexes, dists = self.index.get_nns_by_vector(
            vector, limit, include_distances=True)
        ids = []
        for index in indexes:
            ids.append(self.ids[index])
        return np.array(ids), dists
