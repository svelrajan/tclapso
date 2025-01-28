#!/usr/bin/python3

import networkx as nx
import math
import re
import networkx.algorithms.approximation as nxaa
import random 
import pandas as pd  # To read data
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import time
import sys
import logging 

sla_threshold = 70
def plot_performance_graph(graph_type, datafile):

    # Import pandas
    import pandas as pd
 
    data = pd.read_csv(data_file)

    nsamples= len(data.index)
    if (nsamples < 10):
        return

    # I is the iteration / C is the CPU utilisation / D is disk utilisation / N is network utilisation
    # S is the # of user sessions /  M is the Mean column / Q is the QoE column

    I = data['ite']
    T = data['timestamp']
    C = data['cpu']
    D = data['disk']
    N = data['nw']
    M = data['mean_ru']
    Q = data['qoe']
    S = data['sessions']

    if (re.match(graph_type, "aps")):
        APS = [None] * nsamples
        APS = (Q/M)+(M)
        data['APS'] = APS

    x_axis = APS
    xmin = min(x_axis) #min(x_axis)
    xmax = max(x_axis) #max(x_axis)
    xlimit = 40

    y_axis = S
    ymin = min(y_axis)
    ymax = max(y_axis)

    # Display graphs where there are enough instances of data samples
    if (xmax < xlimit):
        return

    fig, ax = plt.subplots(dpi=300)

    plt.xlim(xmin, xmax)
    plt.ylim(0, 100)
    plt.vlines(x_axis, 0, M, colors='orange', lw=2, label='App. RU')

    ax.set_xlabel("Performance Scale", fontsize=12)

    ax.set_ylabel("Sessions", fontsize=12)

    ax.plot(x_axis, y_axis, color='grey', marker='', linestyle='--', markersize='5', lw='2', label="APS")

    # Shrink the graph vertically to create space for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.9])

#    ax2 = ax.twinx()
#    ax2.set_ylabel("No. of user sessions", fontsize=12)
#    ax2.plot(x_axis, S, color='blue', marker='', linestyle='-.', label="User Sessions")
#    ax2.set_ylim(0, 180)
#    ax2.set_position([box.x0, box.y0 + box.height * 0.2,
#                 box.width, box.height * 0.9])

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    new_file = data_file + "-sessions-" + ".png"
    plt.savefig(new_file)

#    plt.show()
    plt.close()
    return

# End of plot_graph_sessions()
if (len(sys.argv) > 1):
   graph_type=sys.argv[1]
   data_file=sys.argv[2]
   plot_performance_graph(graph_type, data_file)
else:
   print("Error: Need graph type (aps/static) and source data file as arguments")
