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


def evaluate(pred_ids, pred_dists, true_ids, true_dists):
    top_5 = []
    top_10 = []
    top_20 = []
    top_100 = []
    top_1000 = []
    for i in range(len(pred_ids)):
        sub_top_5 = len(set(pred_ids[i][:5]) & set(true_ids[i][:5]))
        sub_top_10 = len(set(pred_ids[i][:10]) & set(true_ids[i][:10]))
        sub_top_20 = len(set(pred_ids[i][:20]) & set(true_ids[i][:20]))
        sub_top_100 = len(set(pred_ids[i][:100]) & set(true_ids[i][:100]))
        sub_top_1000 = len(set(pred_ids[i][:1000]) & set(true_ids[i][:1000]))
        top_5.append(sub_top_5 / 5.)
        top_10.append(sub_top_10 / 10.)
        top_20.append(sub_top_20 / 20.)
        top_100.append(sub_top_100 / 100.)
        top_1000.append(sub_top_1000 / 1000.)
    top_5 = np.array(top_5)
    top_10 = np.array(top_10)
    top_20 = np.array(top_20)
    top_100 = np.array(top_100)
    top_1000 = np.array(top_1000)
    return {
        'top_5_mean': top_5.mean(),
        'top_5_std': top_5.std(),
        'top_10_mean': top_10.mean(),
        'top_10_std': top_10.std(),
        'top_20_mean': top_20.mean(),
        'top_20_std': top_20.std(),
        'top_100_mean': top_100.mean(),
        'top_100_std': top_100.std(),
        'top_1000_mean': top_1000.mean(),
        'top_1000_std': top_1000.std(),
    }


def main():
    num_queries = 1000
    limit = 100

    vectors = load()

    ids = np.arange(len(vectors))

    selected_ids = np.random.choice(len(vectors), num_queries)
    selected_vectors = []
    for index in selected_ids:
        selected_vectors.append(vectors[index])
    selected_vectors = np.array(selected_vectors)

    names_kwargss = [
        # ('brute', None),
        ('brute_faiss', None),
        ('lsh_faiss', {'nbits': 1}),
        ('lsh_faiss', {'nbits': 2}),
        ('lsh_faiss', {'nbits': 4}),
        ('lsh_faiss', {'nbits': 8}),
        ('lsh_faiss', {'nbits': 16}),
        ('lsh_faiss', {'nbits': 32}),
        ('lsh_faiss', {'nbits': 64}),
        ('lsh_faiss', {'nbits': 128}),
        ('lsh_faiss', {'nbits': 256}),
        ('lsh_faiss', {'nbits': 512}),
        ('lsh_faiss', {'nbits': 1024}),
        ('annoy', {'k': 1}),
        ('annoy', {'k': 2}),
        ('annoy', {'k': 4}),
        ('annoy', {'k': 8}),
        ('annoy', {'k': 16}),
        ('annoy', {'k': 32}),
        #('annoy', {'k': 64}),
        ('nmslib', None),
    ]

    faiss = get('brute_faiss')(ids, vectors)
    true_ids, true_dists = faiss.batch_get_nearest(selected_vectors, limit)

    for name, kwargs in names_kwargss:
        if kwargs is None:
            kwargs = {}
        print('Evaluating: %s with %s' % (name, kwargs))

        t0 = time()
        space = get(name)(ids, vectors, **kwargs)
        construct_time = time() - t0

        t0 = time()
        pred_ids, pred_dists = space.batch_get_nearest(selected_vectors, limit)
        lookup_time = time() - t0

        acc = evaluate(pred_ids, pred_dists, true_ids, true_dists)

        d = {
            'name': name,
            'kwargs': kwargs,
            'construct_time': construct_time,
            'lookup_time': lookup_time,
            'accuracy': acc,
        }
        print(json.dumps(d, indent=4, sort_keys=True))


if __name__ == '__main__':
    main()
