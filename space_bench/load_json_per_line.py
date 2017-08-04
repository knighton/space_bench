from argparse import ArgumentParser
import json
import numpy as np


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--in', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    return ap.parse_args()


def run(args):
    fff = []
    bad = 0
    good = 0
    for line in open(getattr(args, 'in')):
        x = json.loads(line)
        ff = x['embedding']
        ff = np.array(ff)
        norm = np.linalg.norm(ff)
        if not np.isfinite(norm) or norm == 0.:
            bad += 1
            continue
        fff.append(ff)
        good += 1
    fff = np.array(fff)
    fff.dump(open(args.out, 'wb'))
    print('%d bad, %d good.' % (bad, good))


if __name__ == '__main__':
    run(parse_args())
