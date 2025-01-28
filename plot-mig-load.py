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

data_file="/tmp/task_data.txt"

def plot_tasks():
    global data_file

    task_data = pd.read_csv(data_file)
    simstart_timestamp = task_data['mig_start_timestamp'][0]
    print("Sim start TS: " + str(simstart_timestamp))

    fig, ax = plt.subplots()

#    plt.xlim(xmin, xmax)

    print(task_data)

    task_data['mean_ru_before_migration']
    task_data['mean_ru_after_migration']

    print("Printing...")
    print(task_data.index.values)

    task_data['mig_start_timestamp'] /= 1000

    ymin = min(chain(task_data['mean_ru_before_migration'] , task_data['mean_ru_after_migration']))
    ymax = max(chain(task_data['mean_ru_before_migration'] , task_data['mean_ru_after_migration']))

    x_axis_data = (list(map(lambda i : i + 5, task_data.index.values)))

    print("**************")
    print(ymin, ymax)
    print("**************")
#    plt.xlim(min(task_data['mig_start_timestamp']), max(task_data['mig_start_timestamp'])+5)

    bar_width = 0.35

    task_data['mig_start_timestamp'][0] = task_data['mig_start_timestamp'][0] + 5

    ax.bar(task_data['mig_start_timestamp'], task_data['mean_ru_before_migration'], bar_width, label = "Load Before Migration")
    ax.bar(task_data['mig_start_timestamp'] + bar_width, task_data['mean_ru_after_migration'], bar_width, label = "Load After Migration")

#    ax.bar(list(task_data.index.values), task_data['mean_ru_before_migration'], bar_width, label = "Load Before Migration")
#    ax.bar(x_axis_data, task_data['mean_ru_after_migration'], bar_width, label = "Load After Migration")
#    plt.plot(task_data['mig_start_timestamp'], task_data['time_for_migration'], color='grey', marker='', linestyle='--', markersize='5', lw='2',label="Time allotted")
#    plt.plot(task_data['mig_start_timestamp'], task_data['actual_migration_time'], color='red',marker='', linestyle='--', markersize='5', lw='2', label="Time taken")

#    plt.vlines(task_data['mig_start_timestamp'], 0, task_data['sessions_migrated'], colors='grey', lw=2, label='Sessions Migrated')
#    plt.vlines(task_data['mig_start_timestamp'], 0, task_data['sla_miss'], colors='red', lw=2, label='SLA Miss')

#    plt.vlines(task_data['sessions_migrated'], 0, task_data['time_for_migration'], colors='grey', lw=2, label='Tasks Assigned')

    ax.set_xlabel("Time Instance ", fontsize=12)
#    ax.set_ylabel("Tasks Count", fontsize=12)
#    ax.set_ylabel("Sessions Migrated", fontsize=12)
    ax.set_ylabel("Load (Before vs. After)", fontsize=12)

    plt.title('Migration Load')
    plt.legend(loc="upper left")

    plt.show()

#    plt.axhline(y = sla_threshold, color = 'r', label="SLA Limit", linestyle = 'dotted', lw=3)

# End of plot_tasks

plot_tasks()
