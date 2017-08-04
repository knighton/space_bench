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
        acc = x['accuracy']['5_10_mean']
        time = x['search_time']
        class2prettys_times_accs[class_name].append((pretty, time, acc))

    name2dark = {
        'annoy': '#08f',
        'nmslib': '#f80',
        'lsh_faiss': '#0b0',
        'brute_faiss': 'red',
    }

    name2light = {
        'annoy': '#ace',
        'nmslib': '#fb8',
        'lsh_faiss': '#8f8',
        'brute_faiss': 'red',
    }

    traces = []
    for class_name in sorted(class2prettys_times_accs):
        prettys_times_accs = class2prettys_times_accs[class_name]
        xx = []
        yy = []
        if class_name == 'brute_faiss':
            xx.append(0.)
            yy.append(prettys_times_accs[0][1])
        for pretty, time, acc in prettys_times_accs:
            xx.append(acc)
            yy.append(time)
        color = name2light[class_name]
        trace = go.Scatter(name=class_name, x=xx, y=yy, mode='lines',
                           line=dict(color=color))
        traces.append(trace)
        for pretty, time, acc in prettys_times_accs:
            xx = np.array([acc])
            yy = np.array([time])
            color = name2dark[class_name]
            trace = go.Scatter(name=pretty, x=xx, y=yy, mode='markers',
                               line=dict(color=color))
            traces.append(trace)

    layout = go.Layout(**{
        'title': 'Accuracy vs Latency Tradeoffs',
        'xaxis': {'title': 'Accuracy (% of predicted top 5 in the true top '
                           '10)'},
        'yaxis': {'title': 'Latency (of 10,000 queries for 100 nearest '
                           'videos, in seconds)'},
    })
    figure = go.Figure(data=traces, layout=layout)
    offline.plot(figure, filename='data/out.html')


if __name__ == '__main__':
    main()
