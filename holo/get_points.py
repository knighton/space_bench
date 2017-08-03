from collections import defaultdict
import json
import numpy as np

import plotly.offline as offline
import plotly.graph_objs as go
import plotly.plotly as py


def main():
    class2pretty2time_acc = defaultdict(dict)
    for line in open('data/out.txt'):
        x = json.loads(line)
        class_name = x['class_name']
        pretty = x['pretty']
        acc = x['accuracy']['5_20_mean']
        time = x['search_time']
        class2pretty2time_acc[class_name][pretty] = time, acc

    traces = []
    for class_name in sorted(class2pretty2time_acc):
        pretty2time_acc = class2pretty2time_acc[class_name]
        for pretty in sorted(pretty2time_acc):
            time, acc = pretty2time_acc[pretty]
            xx = np.array([acc])
            yy = np.array([time])
            trace = go.Scatter(name=pretty, x=xx, y=yy, mode='markers')
            traces.append(trace)

    offline.plot(traces, filename='out.png')


if __name__ == '__main__':
    main()
