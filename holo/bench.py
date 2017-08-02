import json
import numpy as np
from time import time

from .space import get


def load():
    f = 'data/vectors.npy'
    t0 = time()
    vectors = np.load(f)
    t = time() - t0

    total_count = len(vectors)
    vectors = list(vectors)
    for i in range(len(vectors)):
        norm = np.linalg.norm(vectors[i])
        if not np.isfinite(norm) or norm == 0.:
            vectors[i] = None
    vectors = list(filter(lambda v: v is not None, vectors))
    vectors = np.array(vectors).astype('float32')
    good_count = len(vectors)

    print('Loading vectors took %.3f sec (%d good / %d total).' %
          (t, good_count, total_count))

    return vectors


def main():
    num_queries = 1000
    results_per_query = 100

    vectors = load()

    ids = np.arange(len(vectors))

    selected_ids = np.random.choice(len(vectors), num_queries)
    selected_vectors = []
    for index in selected_ids:
        selected_vectors.append(vectors[index])
    selected_vectors = np.array(selected_vectors)

    names_kwargss = [
        # ('brute', None),
        ('faiss', None),
        ('annoy', {'k': 1}),
        ('annoy', {'k': 2}),
        ('annoy', {'k': 4}),
        ('annoy', {'k': 8}),
        ('annoy', {'k': 16}),
        ('annoy', {'k': 32}),
        #('annoy', {'k': 64}),
        ('nmslib', None),
    ]

    for name, kwargs in names_kwargss:
        if kwargs is None:
            kwargs = {}
        print('Evaluating: %s with %s' % (name, kwargs))

        t0 = time()
        space = get(name)(ids, vectors, **kwargs)
        construct_time = time() - t0

        t0 = time()
        space.batch_get_nearest(selected_vectors, results_per_query)
        lookup_time = time() - t0

        d = {
            'name': name,
            'kwargs': kwargs,
            'construct_time': construct_time,
            'lookup_time': lookup_time,
        }
        print(json.dumps(d, indent=4, sort_keys=True))


if __name__ == '__main__':
    main()
