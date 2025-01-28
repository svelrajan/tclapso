import networkx as nx
import math
import re
import networkx.algorithms.approximation as nxaa
import random
import pandas as pd  # To read data
import matplotlib.pyplot as plt
import time
import sys
import logging
import tasklib
import operator
import pickle

from itertools import chain

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from datetime import datetime

data_file=sys.argv[1]

def plot_tasks():
    global data_file

    task_data = pd.read_csv(data_file)
    simstart_timestamp = task_data['mig_start_timestamp'][0]
    print("Sim start TS: " + str(simstart_timestamp))

    fig, ax = plt.subplots()

#    plt.xlim(xmin, xmax)

    print(task_data)

    task_data['actual_migration_time'] = round(((task_data['mig_end_timestamp'] *1000) - (task_data['mig_start_timestamp'] *1000))/1000, 2)
    task_data['time_for_migration'] = round(task_data['time_for_migration'], 2)
    print(task_data)

    task_data['mig_start_timestamp'] = round(task_data['mig_start_timestamp'] - simstart_timestamp)
    task_data['mig_start_timestamp'] /= 1000

    ymin = min(chain(task_data['time_for_migration'] , task_data['actual_migration_time']))
    ymax = max(chain(task_data['time_for_migration'] , task_data['actual_migration_time']))
    print("**************")
    print(ymin, ymax)
    print("**************")
    plt.xlim(min(task_data['mig_start_timestamp']), max(task_data['mig_start_timestamp'])+5)
    plt.ylim(0, ymax+1000)

    print(task_data['mig_start_timestamp'])
    task_data['mig_end_timestamp'] = task_data['mig_start_timestamp']
    bar_width = 0.5
    items_count=len(task_data)
    print("Total items: " + str(items_count))
    start_time = 1
    for i in range(0, items_count):
        task_data['mig_end_timestamp'][i] =  start_time + (2 * bar_width) + 1.2
        start_time = task_data['mig_end_timestamp'][i]
    
    print(task_data['mig_end_timestamp'])
    ax.bar(task_data['mig_end_timestamp'], task_data['time_for_migration'], bar_width, label = "Time allotted")
    ax.bar(task_data['mig_end_timestamp'] + bar_width, task_data['actual_migration_time'], bar_width, label = "Time taken")

#    plt.plot(task_data['mig_start_timestamp'], task_data['time_for_migration'], color='grey', marker='', linestyle='--', markersize='5', lw='2',label="Time allotted")
#    plt.plot(task_data['mig_start_timestamp'], task_data['actual_migration_time'], color='red',marker='', linestyle='--', markersize='5', lw='2', label="Time taken")

#    plt.vlines(task_data['mig_start_timestamp'], 0, task_data['sessions_migrated'], colors='grey', lw=2, label='Sessions Migrated')
#    plt.vlines(task_data['mig_start_timestamp'], 0, task_data['sla_miss'], colors='red', lw=2, label='SLA Miss')

#    plt.vlines(task_data['sessions_migrated'], 0, task_data['time_for_migration'], colors='grey', lw=2, label='Tasks Assigned')

    ax.set_xlabel("Migration Instance ", fontsize=12)
#    ax.set_ylabel("Tasks Count", fontsize=12)
#    ax.set_ylabel("Sessions Migrated", fontsize=12)
    ax.set_ylabel("Migration Duration (Allotted vs. Actual)", fontsize=12)

    plt.legend(loc="upper left")

    plt.show()

#    plt.axhline(y = sla_threshold, color = 'r', label="SLA Limit", linestyle = 'dotted', lw=3)

# End of plot_tasks

plot_tasks()
