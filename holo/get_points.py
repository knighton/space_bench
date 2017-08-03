from collections import defaultdict
import json
import numpy as np

import plotly.offline as offline
import plotly.graph_objs as go
import plotly.plotly as py


def main():
    class2prettys_times_accs = defaultdict(list)
    for line in open('data/out.txt'):
        x = json.loads(line)
        class_name = x['class_name']
        pretty = x['pretty']
        acc = x['accuracy']['5_20_mean']
        time = x['search_time']
        class2prettys_times_accs[class_name].append((pretty, time, acc))

    traces = []
    for class_name in sorted(class2prettys_times_accs):
        prettys_times_accs = class2prettys_times_accs[class_name]
        for pretty, time, acc in prettys_times_accs:
            xx = np.array([acc])
            yy = np.array([time])
            trace = go.Scatter(name=pretty, x=xx, y=yy, mode='markers')
            traces.append(trace)

    offline.plot(traces, filename='out.png')


if __name__ == '__main__':
    main()
