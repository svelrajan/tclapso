import networkx as nx
import re
import networkx.algorithms.approximation as nxaa
import random
import pandas as pd  # To read data
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import time
import sys
import logging

weight={}
weight['iot']={'cpu':0, 'disk':1, 'mem':0, 'nw':1}

def plot_ru_graph():

    # Import pandas
    import pandas as pd
    global data_file

    # reading csv file
    data = pd.read_csv(data_file)
#    print(data)
    samples_count = len(data.index)

    I = data['ite']
    C = data['cpu']
    D = data['disk']
    MM = data['mem']
    N = data['nw']
    M = data['mean_ru']
    Q = data['qoe']
    S = data['sessions']
    W = data['mean_ru']
#    APS = [None] * samples_count

    for id in range(samples_count):
        W[id] = round(((weight['iot']['cpu']*C[id]) + (weight['iot']['disk']*D[id]) + (weight['iot']['mem']*MM[id]) + (weight['iot']['nw']*N[id]))/(weight['iot']['cpu'] + weight['iot']['disk'] + weight['iot']['mem'] + weight['iot']['nw']))
#        APS[id] = (W[id] / Q[id]) *100
#        print("APS= " + str(APS[id]) + " R= " + str(W[id]) + " Q= " + str(Q[id]))
    
#    print(W)
    xmin = min(I)
    xmax = max(I)

    fig, ax = plt.subplots(dpi=300)

    plt.xlim(xmin, xmax)
    plt.ylim(0, 100)


    ax.set_xlabel("Time Instance", fontsize=12)
    ax.set_ylabel("Performance Scale", fontsize=12)

    # Plotting Application QoS/QoE
    ax.plot(I, C, color='green', marker='', linestyle=':', markersize='3', lw='3', label="CPU %")
    ax.plot(I, D, color='orange', marker='', linestyle='--', markersize='5', lw='2', label="Disk %")
    ax.plot(I, MM, color='blue', marker='+', linestyle='-', markersize="7", lw='2', label="Mem %")
    ax.plot(I, N, color='black', marker='', linestyle='-.', label="Network %")
    ax.plot(I, W, color='red', marker='', linestyle='-', lw='3', label="App RU%")
#    ax.plot(I, Q, color='darkblue', marker='', label="App QoS")
#    ax.plot(I, APS, color='red', marker='o', label="APS")

    ax.legend(loc='best', fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

#    ax2 = ax.twinx()
#    ax2.set_ylabel("No. of user sessions", fontsize=12)
#    ax2.plot(I, S, color='yellow', marker='', linestyle='-', lw='3', label="Sessions")

    print(plt.gcf().get_size_inches())
    new_file = data_file + ".png"
#    plt.savefig(new_file)

    plt.show()

if (len(sys.argv) > 1):
   data_file=sys.argv[1]

plot_ru_graph()
