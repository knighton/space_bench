import json
import numpy as np
import sys
from time import time
import ujson as json

from .space import get


def load(f):
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
    top_5_in_5s = []
    top_5_in_10s = []
    top_5_in_20s = []
    top_5_in_50s = []
    top_5_in_100s = []
    top_20_in_20s = []
    top_20_in_50s = []
    top_20_in_100s = []
    top_100_in_100s = []
    top_100_in_1000s = []
    for i in range(len(pred_ids)):
        top_5_in_5 = len(set(pred_ids[i][:5]) & set(true_ids[i][:5])) / 5.
        top_5_in_10 = len(set(pred_ids[i][:5]) & set(true_ids[i][:10])) / 5.
        top_5_in_20 = len(set(pred_ids[i][:5]) & set(true_ids[i][:20])) / 5.
        top_5_in_50 = len(set(pred_ids[i][:5]) & set(true_ids[i][:50])) / 5.
        top_5_in_100 = len(set(pred_ids[i][:5]) & set(true_ids[i][:100])) / 5.
        top_20_in_20 = len(set(pred_ids[i][:20]) & set(true_ids[i][:20])) / 20.
        top_20_in_50 = len(set(pred_ids[i][:20]) & set(true_ids[i][:50])) / 20.
        top_20_in_100 = len(set(pred_ids[i][:20]) &
                            set(true_ids[i][:100])) / 20.
        top_5_in_5s.append(top_5_in_5)
        top_5_in_10s.append(top_5_in_10)
        top_5_in_20s.append(top_5_in_20)
        top_5_in_50s.append(top_5_in_50)
        top_5_in_100s.append(top_5_in_100)
        top_20_in_20s.append(top_20_in_20)
        top_20_in_50s.append(top_20_in_50)
        top_20_in_100s.append(top_20_in_100)
    top_5_in_5 = np.array(top_5_in_5s)
    top_5_in_10 = np.array(top_5_in_10s)
    top_5_in_20 = np.array(top_5_in_20s)
    top_5_in_50 = np.array(top_5_in_50s)
    top_5_in_100 = np.array(top_5_in_100s)
    top_20_in_20 = np.array(top_20_in_20s)
    top_20_in_50 = np.array(top_20_in_50s)
    top_20_in_100 = np.array(top_20_in_100s)
    return {
        '5_5_mean': top_5_in_5.mean(),
        '5_5_std': top_5_in_5.std(),
        '5_10_mean': top_5_in_10.mean(),
        '5_10_std': top_5_in_10.std(),
        '5_20_mean': top_5_in_20.mean(),
        '5_20_std': top_5_in_20.std(),
        '5_50_mean': top_5_in_50.mean(),
        '5_50_std': top_5_in_50.std(),
        '5_100_mean': top_5_in_100.mean(),
        '5_100_std': top_5_in_100.std(),
        '20_20_mean': top_20_in_20.mean(),
        '20_20_std': top_20_in_20.std(),
        '20_50_mean': top_20_in_50.mean(),
        '20_50_std': top_20_in_50.std(),
        '20_100_mean': top_20_in_100.mean(),
        '20_100_std': top_20_in_100.std(),
    }


def main():
    f = 'data/vectors.npy'
    num_queries = 10000
    limit = 100
    out = 'data/out.txt'

    vectors = load(f)

    ids = np.arange(len(vectors))

    selected_ids = np.random.choice(len(vectors), num_queries)
    selected_vectors = []
    for index in selected_ids:
        selected_vectors.append(vectors[index])
    selected_vectors = np.array(selected_vectors)

    faiss_trials = [
        ('Faiss (Brute Force)', 'brute_faiss', None),
        ('Faiss (LSH @ 32 bits)', 'lsh_faiss', {'nbits': 32}),
        ('Faiss (LSH @ 64 bits)', 'lsh_faiss', {'nbits': 64}),
        ('Faiss (LSH @ 128 bits)', 'lsh_faiss', {'nbits': 128}),
        ('Faiss (LSH @ 256 bits)', 'lsh_faiss', {'nbits': 256}),
        ('Faiss (LSH @ 512 bits)', 'lsh_faiss', {'nbits': 512}),
        ('Faiss (LSH @ 1024 bits)', 'lsh_faiss', {'nbits': 1024}),
        ('Faiss (LSH @ 2048 bits)', 'lsh_faiss', {'nbits': 2048}),
    ]

    nmslib_trials = [
        ('NMSLib (M = 2)', 'nmslib', {'m': 2}),
        ('NMSLib (M = 4)', 'nmslib', {'m': 4}),
        ('NMSLib (M = 8)', 'nmslib', {'m': 8}),
        ('NMSLib (M = 16)', 'nmslib', {'m': 16}),
        ('NMSLib (M = 32)', 'nmslib', {'m': 32}),
        ('NMSLib (M = 64)', 'nmslib', {'m': 64}),
    ]

    annoy_trials = [
        ('Annoy (1 tree)', 'annoy', {'k': 1}),
        ('Annoy (2 trees)', 'annoy', {'k': 2}),
        ('Annoy (4 trees)', 'annoy', {'k': 4}),
        ('Annoy (8 trees)', 'annoy', {'k': 8}),
        ('Annoy (16 trees)', 'annoy', {'k': 16}),
        ('Annoy (32 trees)', 'annoy', {'k': 32}),
    ]

    trials = faiss_trials + nmslib_trials + annoy_trials

    faiss = get('brute_faiss')(ids, vectors)
    true_ids, true_dists = faiss.batch_get_nearest(selected_vectors, limit)

    out = open(out, 'wb')
    for pretty, class_name, kwargs in trials:
        if kwargs is None:
            kwargs = {}
        sys.stdout.write('%s...' % pretty)
        sys.stdout.flush()

        t0 = time()
        space = get(class_name)(ids, vectors, **kwargs)
        construct_time = time() - t0

        t0 = time()
        pred_ids, pred_dists = space.batch_get_nearest(selected_vectors, limit)
        search_time = time() - t0

        acc = evaluate(pred_ids, pred_dists, true_ids, true_dists)

        d = {
            'pretty': pretty,
            'class_name': class_name,
            'kwargs': kwargs,
            'construct_time': construct_time,
            'search_time': search_time,
            'accuracy': acc,
        }
        line = json.dumps(d, sort_keys=True, double_precision=4) + '\n'
        out.write(line.encode('utf-8'))
        out.flush()

        sys.stdout.write(' build %.3fs, search %.3fs, acc 5/10 %.3f, acc '
                         '5/100 %.3f\n' % (construct_time, search_time,
                         acc['5_10_mean'], acc['5_100_mean']))


if __name__ == '__main__':
    main()
