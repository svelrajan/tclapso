#!/usr/bin/python3

import math
import re
import random 
import pandas as pd  # To read data
import matplotlib.pyplot as plt
import time
import sys
import logging 
import tasklib as tasks

###############################
### Variables Definitions
###############################

# Recommendations 10 x 5 x 5 (20000 tasks)
# Default mix 3 x 3 x 3 (2000 tasks)
# Number of edges should be 2 or above
no_of_edges=3
max_hosts_per_edge=3
max_vms_per_host=3
max_apps_per_vm=3

app_type = "surveillance"
app_type = "video"
app_type = "smartmeter"

tasks_file_dir="."
# Tasks File name will follow the syntax tasks-<number of tasks>.txt
#tasks_file="./tasks-2000.txt"

attr_list = {'cpu', 'disk', 'mem', 'nw', 'sessions'}

############################################
# Application Profiles Definition
############################################
weight = {}
weight['smartmeter']={'cpu':0, 'disk':1, 'mem':0, 'nw':1, 'sessions':0}
weight['video']={'cpu':0, 'disk':1, 'mem':1, 'nw':1, 'sessions':0}
weight['compression']={'cpu':1, 'disk':1, 'mem':0, 'nw':1, 'sessions':0}
weight['surveillance']={'cpu':0, 'disk':1, 'mem':0, 'nw':1, 'sessions':0}

multiplier = {}
multiplier['smartmeter']={'cpu':0.5, 'disk':0.5, 'mem':0.5, 'nw':0.25, 'sessions':1.5}
multiplier['video']={'cpu':0.5, 'disk':1, 'mem':0.5, 'nw':1, 'sessions':0.5}
multiplier['compression']={'cpu':1, 'disk':1, 'mem':1.5, 'nw':1, 'sessions':1}
multiplier['surveillance']={'cpu':1, 'disk':1, 'mem':1, 'nw':1, 'sessions':0.75}

task_gen_method="random"
task_size = "small"

def get_random_task():
    tasks_list=list(tasks.task_def.keys())
    return random.choice(tasks_list)

def get_random_task_op():
      return tasks.task_ops[random.randint(0, len(tasks.task_ops)-1)]

def calc_load(app_type, cpu, disk, mem, nw):
    mean_ru = round(((weight[app_type]['cpu']*cpu) + (weight[app_type]['disk']*disk) + (weight[app_type]['mem']*mem) + (weight[app_type]['nw']*nw))/(weight[app_type]['cpu'] + weight[app_type]['disk'] + weight[app_type]['mem'] + weight[app_type]['nw']))
#    mean_ru = round((cpu + disk + mem + nw)/4)
    return mean_ru

# Format of tasks in tasks file
# taskid:task_type:cpu:disk:mem:nw:sessions:task_status:task_op:task_expiry

def tasks_gen(app_type, tasks_gen_method, task_size):
    task_id = 0
    task_status = "active"
    task_expiry = 0
    system_load = 0
    system_avg_load = 0
    global sessions_multiplier

    f = open(tasks_file, "w")

    f.write("task_id:task_type:cpu:disk:mem:nw:sessions:task_op\n")
    while (1):
        if (re.match(tasks_gen_method, "random")):
            task_type=get_random_task()

            f.write(str(task_id) + ":" + task_type + ":")
            ## SARO: Python doesn't maintain the sequence of lists !! :(
            for attr in ('cpu', 'disk', 'mem', 'nw', 'sessions'):
                f.write(str(tasks.task_def[task_type][attr] * multiplier[app_type][attr]))
                f.write(":")
                print(str(tasks.task_def[task_type][attr] * multiplier[app_type][attr]), end="")
                print(":", end="")

            print("")
            task_op = get_random_task_op()
            f.write(task_op + "\n")

            task_load = calc_load(app_type, tasks.task_def[task_type]['cpu'], tasks.task_def[task_type]['disk'], tasks.task_def[task_type]['mem'], tasks.task_def[task_type]['nw'])

            if (re.match(task_op, "assign")):
                system_load = system_load + task_load
            else:
                system_load = system_load - task_load

            system_avg_load = round(system_load/total_app_instances)

            if (system_avg_load > system_avg_load_max):
                break
        task_id = task_id + 1
    
    print("Tasks Count: " + str(task_id) + " " + "App load avg: " + str(system_avg_load))
    f.close()

def read_tasks():
    import pandas as pd
    system_load=0

    data = pd.read_table(tasks_file, delimiter =":")
    data['mean_ru'] = (data['cpu'] + data['disk'] + data['mem'] + data['nw'])/4

    data_cum = data.copy()
    for attr in attr_list:
        data_cum[attr] = data[attr].cumsum()
    data_cum['mean_ru'] = data_cum['mean_ru'].cumsum()

    system_load =  data_cum['mean_ru'].iloc[-1]

    print(system_load)
    print(data_cum)

    plt.plot(data_cum['task_id'], data_cum['mean_ru'], color='red', label="Tasks")
    plt.plot(data_cum['task_id'], data_cum['cpu'], color='lightblue', label="CPU")
    plt.plot(data_cum['task_id'], data_cum['disk'], color='grey', label="Disk")
    plt.plot(data_cum['task_id'], data_cum['mem'], color='yellow', label="Mem")
    plt.plot(data_cum['task_id'], data_cum['nw'], color='orange', label="NW")
    plt.plot(data_cum['task_id'], data_cum['sessions'], color='blue', label="Sessions")
    plt.xlabel("Task ID")
    plt.ylabel("Average RU")
    plt.legend(loc='best')
    plt.show()

################################################
# Main program starts here
################################################

if (len(sys.argv) > 1):
    system_avg_load_max=int(sys.argv[1])
    tasks_file=sys.argv[2]
else:
    system_avg_load_max=65
    tasks_file="/tmp/tasks.txt"


total_app_instances = no_of_edges * max_hosts_per_edge * max_vms_per_host * max_apps_per_vm

print(total_app_instances)
tasks_gen(app_type, "random", "small")
read_tasks()
