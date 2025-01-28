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

    x_axis = T/1000
    
    xmin = min(x_axis)
    xmax = max(x_axis)

    # Display graphs where there are enough instances of data samples
    if (xmax < 40):
        return

    fig, ax = plt.subplots()#dpi=300)

    plt.xlim(xmin, xmax)
    plt.ylim(0, 100)
#    plt.vlines(x_axis, 0, M, colors='orange', lw=2, label='App. RU')

    ax.set_xlabel("Time Instance", fontsize=12)

    ax.set_ylabel("Performance Scale", fontsize=12)

    plt.axhline(y = sla_threshold, color = 'r', label="SLA Limit", linestyle = 'dotted', lw=3)
    plt.text(50, 73, 'SLA Limit = 70%', color='r')

    # Plotting Application QoS/QoE
    ax.plot(x_axis, Q, color='darkblue', marker='+', linestyle='-', markersize="7", lw='2', label="App. QoS")

    if (re.match(graph_type, "aps")):
        ax.plot(x_axis, APS, color='green', marker='', linestyle='--', markersize='5', lw='2', label="App. Perf. Score (APS)")

    # Shrink the graph vertically to create space for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.9])

    ax2 = ax.twinx()
    ax2.set_ylabel("No. of user sessions", fontsize=12)
    ax2.plot(x_axis, S, color='orange', marker='', linestyle='-.', label="User Sessions")
    ax2.set_ylim(0, max(S))
    ax2.set_position([box.x0, box.y0 + box.height * 0.2,
                 box.width, box.height * 0.9])

    lhandle1, label1= ax.get_legend_handles_labels()
    lhandle2, label2= ax2.get_legend_handles_labels()

    if (re.match(graph_type, "aps")):
#        ax.legend([lhandle1[0]] + [lhandle1[1]] + [lhandle1[2]] + [lhandle1[3]] + lhandle2, [label1[0]] + [label1[1]] + [label1[2]] + [label1[3]] + label2, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3, fontsize=12)
        ax.legend([lhandle1[0]] + [lhandle1[1]] + [lhandle1[2]] + lhandle2, [label1[0]] + [label1[1]] + [label1[2]] + label2, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=1, fontsize=12)
    else:
        ax.legend([lhandle1[0]] + [lhandle1[1]] + [lhandle1[2]] + lhandle2, [label1[0]] + [label1[1]] + [label1[2]] + label2, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=2, fontsize=12)

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)


    new_file = data_file + ".png"
#    plt.savefig(new_file)

    plt.show()
    plt.close()
    return

# End of plot_graph_sessions()
if (len(sys.argv) > 1):
   graph_type=sys.argv[1]
   data_file=sys.argv[2]
   plot_performance_graph(graph_type, data_file)
else:
   print("Error: Need graph type (aps/static) and source data file as arguments")
