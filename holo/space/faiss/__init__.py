import numpy as np
from wurlitzer import pipes

from ..space import Space
from .faiss import IndexFlatL2


class FaissSpace(Space):
    name = 'faiss'

    def __init__(self, ids, vectors):
        super().__init__(ids, vectors)
        self.index = IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)

    def get_nearest(self, vector, limit):
        vectors = np.expand_dims(vector, 0)
        dist_matrix, index_matrix = self.index.search(vectors, limit)
        for i, index in enumerate(index_matrix[0]):
            index_matrix[0][i] = self.ids[index]
        return index_matrix[0], dist_matrix[0]

    def batch_get_nearest(self, vectors, limit):
        dist_matrix, index_matrix = self.index.search(vectors, limit)
        for i in range(index_matrix.shape[0]):
            for j in range(index_matrix.shape[1]):
                index = index_matrix[i, j]
                index_matrix[i, j] = self.ids[index]
        return index_matrix, dist_matrix
