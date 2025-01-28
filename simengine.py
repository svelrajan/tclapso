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

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

##############################################
# Debugs Definition
##############################################
DEBUG = "" 
INFO = "" 
DEBUG = True 
INFO = True 

def log_debug(*s):
    if DEBUG:
#        print("DEBUG", end=" ")
        print(*s)

def log_info(*s):
    if INFO:
#        print(end=" ")
        print(*s)


no_of_edges = 2
max_hosts_per_edge=2
max_vms_per_host=6
max_apps_per_vm=1

display_nodes = []

early_alerts = 0

# migration time in milli seconds
#MIN_MIGRATION_TIME_PER_UE = 80
#MAX_MIGRATION_TIME_PER_UE = 100

node_attrs={'cpu':0, 'disk':0, 'mem':0, 'nw':0, 'sessions':0, 'migration_time':0, 'mon_interval':0}
total_node_instances = 0
total_app_instances = (no_of_edges * max_hosts_per_edge * max_vms_per_host * max_apps_per_vm)
total_vm_instances = (no_of_edges * max_hosts_per_edge * max_vms_per_host)
total_host_instances = (no_of_edges * max_hosts_per_edge)
total_edge_instances = (no_of_edges)
total_cloud_instances = 1

total_node_instances = total_app_instances + total_vm_instances + total_host_instances + total_edge_instances + total_cloud_instances

sla_threshold = 70
ru_min_threshold = 60
ru_default_threshold = 67
ru_max_threshold = 70

# As you decrease this value, CLAPSOD peforms better (decreasing the value provides enough runway migration time to avoid a SLA violation)
# When you increase this, CLAPSO performs better - need to validate this hypothesis
ru_migration_threshold = 65
aps_threshold = 65

total_node_migrations = total_task_migrations = total_qoe_violations= total_node_instances_with_violations = total_tasks_reassigned = total_mtime_violations = 0
total_load_reduced = 0

NO_TREND_SEEN = 1
TREND_SEEN = 0

cloud_node_str="cloud" # Centralized Cloud
host_node_str="host"   # Edge host
edge_node_str="ec"     # Edge cloud
vm_node_str="vm"       # Edge VM
app_node_str="app"     # Edge app
ue_node_str="ue"       # UE Task Node

type_of_node = app_node_str

MAX_RESOURCE_FOR_APP=100
MAX_SESSIONS_FOR_APP=10000    # Unlimited

node_resource_attrs = ["cpu", "disk", "mem", "nw", "sessions"]

############################################
# SIM Clocks / Time Initialization
############################################
sim_clock_time = simstart_timestamp = round(time.time())

############################################
# SIM Stats Definition
############################################

# sim_stats - RU stats 
# mig_stats - Migration time stats
# 
# sim_stats[node][0] {'ite':0, 'cpu':cpu, 'disk':disk, 'mem': mem, 'nw':nw, 'sessions':sessions, 'mean_ru':0, 'qoe':0, 'best_ru':0, 'best_qoe':0}
# mig_stats[node][<value_type>] = {'migration_time_low':migration_time, 'migration_time_high':migration_time}
# <value_type> for apps is app_mig_time or ue_mig_time
# <value_type> for vm is vm_mig_time

sim_stats = {}
mig_stats = {}
pso_stats = {}
task_stats = {}
task_stats_counter = 0
misc_stats = {'simstart_timestamp':0, 'simend_timestamp':0, 'end_taskid':0}

# For each node there is a sim_stats_counter
sim_stats_counter = {}

# SLA Violation stats
sla_stats_attrs = ["cpu", "disk", "mem", "nw", "total"]
sla_stats = {}

# Stats file
stats_lock_file = "/tmp/sim_stats.txt.lock"
stats_file = "/tmp/sim_stats.txt"

# Cache Files
topo_db="./simdb/topo-file.db"
stats_db="./simdb/stats-file.db"
sla_stats_db="./simdb/sla_stats-file.db"
pso_stats_db="./simdb/pso_stats-file.db"
sim_stats_counter_db="./simdb/sim_stats_counter-file.db"
misc_stats_db="./simdb/misc_stats-file.db"

############################################
# QoE Profiles Definition
############################################
#   0 is no-service
#  20 is poor
#  40 is bad
#  60 is acceptable
#  80 is good
# 100 is great

qoe_pattern = [ {'min_ru':0, 'max_ru':20, 'min_qoe':95, 'max_qoe':99},
                {'min_ru':20, 'max_ru':40, 'min_qoe':90, 'max_qoe':95},
                {'min_ru':40, 'max_ru':60, 'min_qoe':80, 'max_qoe':90},
                {'min_ru':60, 'max_ru':70, 'min_qoe':70, 'max_qoe':80},
                {'min_ru':70, 'max_ru':100, 'min_qoe':5, 'max_qoe':70}
              ]

############################################
# Application Profiles Definition
############################################
weight = {}
weight['smartmeter']={'cpu':0, 'disk':1, 'mem':0, 'nw':1, 'sessions':4}
weight['video']={'cpu':0, 'disk':1, 'mem':1, 'nw':1, 'sessions':1}
weight['compression']={'cpu':1, 'disk':1, 'mem':0, 'nw':1, 'sessions':1}
weight['surveillance']={'cpu':0, 'disk':1, 'mem':0, 'nw':1, 'sessions':1}

# Default migration time in milliseconds
def_migration_time_min = {}
def_migration_time_max = {}

def_migration_time_min['smartmeter'] = 1
def_migration_time_max['smartmeter'] = 10

def_migration_time_min['video'] = 80
def_migration_time_max['video'] = 100

def_migration_time_min['compression'] = 60
def_migration_time_max['compression'] = 80

def_migration_time_min['surveillance'] = 40
def_migration_time_max['surveillance'] = 60

mon_interval_default = 10

resource_limit = {}
resource_limit = {'cpu':MAX_RESOURCE_FOR_APP, 'disk':MAX_RESOURCE_FOR_APP, 'mem': MAX_RESOURCE_FOR_APP, 'nw':MAX_RESOURCE_FOR_APP, 'sessions':MAX_SESSIONS_FOR_APP}

############################################
# Tasks Interval Definition
############################################
TASK_ASSIGNMENT_INTERVAL_MIN = 0     # milliseconds
TASK_ASSIGNMENT_INTERVAL_MAX = 10   # milliseconds
MOD_TASK_COUNT_MIN = 5
MOD_TASK_COUNT_MAX = 100

global task_def
global tasks_count

def increment_sim_time():
    global sim_clock_time
    #sim_clock_time = sim_clock_time + random.randint(1, 100)
    sim_clock_time = sim_clock_time + 100

def get_sim_time():
    global sim_clock_time
    return sim_clock_time

def get_edge_from_app(app_node):
    vm_node = get_vm_from_app(app_node)
    host_node = get_host_from_vm(vm_node)
    edge = get_edge_from_host(host_node)

    return edge

def get_edge_from_vm(vm_node):
    host_node = get_host_from_vm(vm_node)
    edge = get_edge_from_host(host_node)

    return edge

def get_edge_from_host(host_node):
    edge = ""
    if re.search(host_node_str, host_node):
        for node in G.neighbors(host_node):
           if re.search(edge_node_str, node):
               edge = node
               break
    return edge

def get_host_from_vm(vm_node):
    host = ""
    is_vm = 0
    if re.search(vm_node_str, vm_node):
        is_vm = 1
        for node in G.neighbors(vm_node):
           if re.search(host_node_str, node):
               host = node
               break
    return host

def get_vm_from_app(app):
    vm = ""
    is_app = 0
    if re.search(app_node_str, app):
        is_app = 1
        for node in G.neighbors(app):
           if re.search(vm_node_str, node):
               vm = node
               break
    return vm

def get_ues_from_app(app):
    ues = []
    is_app = 0
    if re.search(app_node_str, app):
        is_app = 1
        for node in G.neighbors(app):
            if re.search(ue_node_str, node):
                ues.append(node)
    return ues
# End of get_ues_from_app()

#SARO: Have to find subset sum using knapsack or dynamic programming
def get_min_subset_max_sum (nums, sum):

    # sort numbers in descending order
    nums.sort()
    print(nums)

    return A
# End of get_min_subset_max_sum


#####################################
# Greedy Knapsack Algorithm
# Maximize "profit" (Load) - More load has to be reduced from the app
# Ensure that "weight" (Migration time) is lesser than the "allotted" migration time (constraint)
#####################################

def knapsack(weights, values, capacity, ids):
      prioritised_svcs = []
      res = 0
      load_total = 0
      mt_total = 0
      pairs = sorted(zip(weights, values, ids), reverse=True)
      for pair in pairs:
         if (pair[0] <= capacity) and ((load_total + pair[1]) > load_total):
            capacity -= pair[0]
            load_total += pair[1]
            mt_total += pair[0]
            #print("Migration Time - " + str(pair[0]) + " Load - " + str(pair[1]))
            print("Id: " + str(pair[2]) + " Migration Time (Total) - " + str(pair[0]) + " (" + str(mt_total) + ") " + " Load (Total) - " + str(pair[1]) + " (" + str(load_total) + ")")
            prioritised_svcs.append(pair[2])
            
      return (prioritised_svcs, load_total, mt_total)

#End of knapsack

def get_ues_from_app_based_on_knapsack(app, time_for_migration):
    ues_load = []
    ues_migration_time = []
    src_ues_list = []

    for node in G.neighbors(app):
        if re.search(ue_node_str, node):
            src_ues_list.append(node)
            ues_migration_time.append(get_migration_time_from_node(node))
            ues_load.append(calc_node_load(node))
    print("**** Sessions ****")
    print(src_ues_list)
    print("**** Load ****")
    print(ues_load)
    print("**** Migration Time ****")
    print(ues_migration_time)

    print("**** Total migration time is " + str(time_for_migration) + " ****")
    (ues_list, load_total, mt_total) = knapsack(ues_migration_time, ues_load, time_for_migration, src_ues_list)

    print("Allotted Migration Time: " + str(time_for_migration) + " Identified UEs for migration time / load " + str(mt_total) + " / " + str(load_total))

    return ues_list

# End of get_ues_from_app_based_on_knapsack

def get_ues_from_app_based_on_migration_time(app, time_for_migration):
    ues_dict = {}
    ues_list = []
    for node in G.neighbors(app):
        if re.search(ue_node_str, node):
            ue_node = node
            ues_dict[ue_node] = get_migration_time_from_node(ue_node)

    # Sort the UEs based on migration time (low migration time first, high migration time at the end)
    s_ues_dict = dict(sorted(ues_dict.items(), key=lambda x:x[1], reverse=True))

    t_migration_time = 0
    # Loop until the sum of numbers is greater than sum
    for (ue_node, migration_time) in s_ues_dict.items():
        if (t_migration_time > time_for_migration):
            break
        if ((t_migration_time + migration_time) <= time_for_migration):
            ues_list.append(ue_node)
            t_migration_time += migration_time
        else:
            break

#    print("UE list : " + str(ues_dict))
#    print("Sorted list : " + str(s_ues_dict))
    print("Allotted Migration Time: " + str(time_for_migration) + " Identified UEs for migration time " + str(t_migration_time))
#    print("UE Shortlist : " + str(ues_list))

    return ues_list
# End of get_ues_from_app_based_on_migration_time

def initialise_tasks():
    import pandas as pd
    global tasks
    global tasks_count

    tasks = pd.read_table(tasks_file, delimiter =":")
    tasks_count = len(tasks.index)

def initialise_pso_stats():
    pso_stats.clear()
   
    for node in G.nodes():
        if not (re.search(ue_node_str, node)):
            # Resonance is the number of times the ru or mt is put to use and
            # it helped avoiding SLA violations
            # best_
            pso_stats.setdefault(node, {'best_ru':0, 'ru_per_task':0, 'mt_per_task':0, 'best_ru_resonance': 0, 'ru_per_task_resonance':0, 'mt_per_task_resonance':0})
            print("Node: ", node, pso_stats[node])


def initialise_sla_stats():
    sla_stats.clear()
    sla_stats.setdefault(app_node_str, {})
    sla_stats.setdefault(vm_node_str, {})
    sla_stats.setdefault(host_node_str, {})
    sla_stats.setdefault(edge_node_str, {})

    for attr in sla_stats_attrs:
        sla_stats[app_node_str].setdefault(attr, 0)
        sla_stats[vm_node_str].setdefault(attr, 0)
        sla_stats[host_node_str].setdefault(attr, 0)
        sla_stats[edge_node_str].setdefault(attr, 0)

def initialise_stats():

    global task_stats_counter
    sim_stats.clear()
    sim_stats_counter.clear()

    task_stats.clear()
#    task_stats[task_stats_counter] = {'mig_start_timestamp': time.time(), 'mig_end_timestamp':0, 'tasks_assigned':0, 'sessions_migrated':0, 'sla_miss':0, 'time_for_migration':0}
#    task_stats_counter = task_stats_counter + 1 

    # setdefault creates the nodes, if they do not exist already
    # update() function expects the nodes to be present already for updating them with values
    # update() function would throw an exception if the key / index is not present in the dict
    # ite value is also included in the stats, so that it is easy to convert to Pandas array
    #    for linear regressions

    nodes = G.nodes()
    for node in nodes:
        sim_stats_counter[node] = 0
        sim_stats.setdefault(node, {})

#        (cpu, disk, mem, nw, sessions) = get_resource_attrs_from_node(node)
#        mean_ru = calc_node_load(node)
#        qoe = gen_qoe(mean_ru)

        if re.search(app_node_str, node):
            migration_time = get_migration_time_from_node(node)
            mon_interval = mon_interval_default

#        sim_stats[node][0] = {'ite':0, 'timestamp': 0, 'cpu':cpu, 'disk':disk, 'mem': mem, 'nw':nw, 'sessions':sessions, 'mean_ru':mean_ru, 'qoe':qoe, 'best_ru':0, 'best_qoe':0}
#        print("Added ... for node " + str(node))
#        print(sim_stats[node][0])

        if re.search(app_node_str, node):
            mig_stats[node] = {'app_mig_time':migration_time, 'ue_mig_time': migration_time}
# Saro: Temporarily commenting this... till p3-sim works (date Jan 8th 2023)
#        if re.search(vm_node_str, node):
#            mig_stats.setdefault(node, {})
#            mig_stats[node] = {'vm_mig_time':migration_time}
        
def get_resource_attrs_from_node (node):
    (cpu, disk, mem, nw, sessions, migration_time, mon_interval) = G.nodes[node].values()
    return (cpu, disk, mem, nw, sessions)

def get_app_number_from_name(app_name):
    app_number = re.findall("[0-9]+", app_name)
    return(app_number[0])

def get_taskid_from_ue(ue_name):
    (ue_name, taskid) = ue_name.split(".")
    return taskid

def add_ue_node(app_name, taskid):

    avg_migration_time_per_ue = 0

    app_number=get_app_number_from_name(app_name)
    task = tasks.iloc[taskid]
    task_type = task['task_type']
    task_op   = task['task_op']

    ue_node_name=ue_node_str + str(app_number) + "." + str(taskid)
    G.add_nodes_from([(ue_node_name, tasklib.task_def[task_type])])
    avg_migration_time_per_ue = random.randint(def_migration_time_min[app_type], def_migration_time_max[app_type])
    G.nodes[ue_node_name]['migration_time'] = tasklib.task_def[task_type]['sessions'] * avg_migration_time_per_ue
    G.nodes[ue_node_name]['mon_interval'] = mon_interval_default
    G.add_edge(app_name, ue_node_name, weight="100")
    return ue_node_name

def del_ue_node(ue_name):
    G.remove_node(ue_name)

def assign_task_to_app (app, taskid):
    error = 0

    task = tasks.iloc[taskid]
    task_type = task['task_type']
    task_op   = task['task_op']

    # Check if the app has resources to accept a new task
    for attr in node_resource_attrs:
        if ((G.nodes[app][attr] + task[attr]) > resource_limit[attr]):
            sla_stats[app_node_str][attr] += 1
            sla_stats[app_node_str]['total'] += 1
            error = 1
            break
    if (error != 1):
        # Check if there are sufficient resources at VM / Host / Edge levels
        error = update_topo_stats(app, task, operator.add)

    if (error != 1):
        for attr in node_resource_attrs:
            G.nodes[app][attr] += task[attr]
        ue_node = add_ue_node(app, taskid)
        G.nodes[app]['migration_time'] += G.nodes[ue_node]['migration_time']
        G.nodes[app]['mon_interval'] = mon_interval_default
        update_topo_attrs()
    
    return error

def unassign_task(app, task_type, ue_name):
    global task_def

    error = 0 # 0 indicates successful unassignment

    for attr in node_resource_attrs:
        if ((G.nodes[app][attr] - tasklib.task_def[task_type][attr]) < 0):
            error = 1
            break

    if (error != 1):
        for attr in node_resource_attrs:
            G.nodes[app][attr] -= tasklib.task_def[task_type][attr]

        G.nodes[app]['mon_interval'] = mon_interval_default
        G.nodes[app]['migration_time'] -= G.nodes[ue_name]['migration_time']
        error = update_topo_stats(app, tasklib.task_def[task_type], operator.sub)
        del_ue_node(ue_name)
        update_topo_attrs()
    return error
# end of unassign_task()


def get_non_ue_nodes_count(G):
    infra_nodes = list(filter(lambda x: not ue_node_str in x, G.nodes))
    return len(infra_nodes)

def get_random_app(G):
    app_nodes = []
    edge_nodes = get_edges_from_cloud(cloud_node_str)
    total_apps_per_edge = max_hosts_per_edge * max_vms_per_host * max_apps_per_vm
    # New logic introduced to keep some spare capacity for migrations
    for edge in edge_nodes:
        t_apps = get_apps_from_edge(edge)
        app_nodes = app_nodes + t_apps[0:total_apps_per_edge - 2]
    #print("Nodes: " + str(app_nodes) + " Len: " + str(len(app_nodes)) + " vs. total of " + str(2*total_apps_per_edge))
    return (app_nodes[random.randint(0,len(app_nodes)-1)])

def get_least_loaded_app(G):
    app_nodes = list(filter(lambda x: app_node_str in x, G.nodes))
    for app_node in app_nodes:
        if (calc_node_load(app_node) < ru_max_threshold):
            return app_node
    return (app_nodes[random.randint(0,len(app_nodes)-1)])

def get_random_app_from_list(app_nodes):
    return (app_nodes[random.randint(0,len(app_nodes)-1)])

def gen_qoe(ru):
    qoe = 0
    for key in range(len(qoe_pattern)):
       if ((ru >= qoe_pattern[key]['min_ru']) and (ru <= qoe_pattern[key]['max_ru'])):
           qoe = random.uniform(qoe_pattern[key]['min_qoe'], qoe_pattern[key]['max_qoe'])
           break
    return qoe

def calc_node_load_from_stats(stats):
    cpu = stats["cpu"]
    disk = stats["disk"]
    mem = stats["mem"]
    nw = stats["nw"]

    mean_ru = round(((weight[app_type]['cpu']*cpu) + (weight[app_type]['disk']*disk) + (weight[app_type]['mem']*mem) + (weight[app_type]['nw']*nw))/(weight[app_type]['cpu'] + weight[app_type]['disk'] + weight[app_type]['mem'] + weight[app_type]['nw']))
    return mean_ru

def calc_node_load(node):
    cpu = G.nodes[node]["cpu"]
    disk = G.nodes[node]["disk"]
    mem = G.nodes[node]["mem"]
    nw = G.nodes[node]["nw"]

    mean_ru = round(((weight[app_type]['cpu']*cpu) + (weight[app_type]['disk']*disk) + (weight[app_type]['mem']*mem) + (weight[app_type]['nw']*nw))/(weight[app_type]['cpu'] + weight[app_type]['disk'] + weight[app_type]['mem'] + weight[app_type]['nw']))
    return mean_ru

def is_node_load_high(node):
    if (calc_node_load(node) > ru_max_threshold):
        return 1
    else:
        return 0

def is_node_load_low(node):
    if (calc_node_load(node) < ru_min_threshold):
        return 1
    else:
        return 0

def initialise_topology():
    G.clear()


    G.add_node(cloud_node_str, **node_attrs)
    for w in range(0,no_of_edges):
        edge_node=edge_node_str + str(w)
        G.add_node(edge_node, **node_attrs)
        G.add_edge(cloud_node_str, edge_node, weight="100")
        sim_stats_counter[edge_node] = 0
        sim_stats.setdefault(edge_node, {})

        for x in range(0,max_hosts_per_edge):
            host_node=host_node_str + str(w) + str(x)
            G.add_node(host_node, **node_attrs)
            G.add_edge(edge_node, host_node, weight="100")
            sim_stats_counter[host_node] = 0
            sim_stats.setdefault(host_node, {})

            for y in range(0,max_vms_per_host):
                vm_node=vm_node_str + str(w) + str(x) + str(y)
                G.add_node(vm_node, **node_attrs)
                G.add_edge(host_node, vm_node, weight="100")
                sim_stats_counter[vm_node] = 0
                sim_stats.setdefault(vm_node, {})

                for z in range(0,max_apps_per_vm):
                    app_node=app_node_str + str(w) + str(x) + str(y) + str(z)

#                    a_cpu=random.randint(0,int(MAX_RESOURCE_FOR_APP/max_apps_per_vm))
#                    a_disk=random.randint(0,int(MAX_RESOURCE_FOR_APP/max_apps_per_vm))
#                    a_mem=random.randint(0,int(MAX_RESOURCE_FOR_APP/max_apps_per_vm))
#                    a_nw=random.randint(0,int(MAX_RESOURCE_FOR_APP/max_apps_per_vm))
#                    a_cpu=random.randint(0,30)
#                    a_disk=random.randint(0,30)
#                    a_mem=random.randint(0,30)
#                    a_nw=random.randint(0,30)
#                    a_sessions=1

                    a_cpu = a_disk = a_mem = a_nw = a_sessions = 0
                    a_migration_time = 0
                    a_mon_interval = mon_interval_default

                    app_node_attrs={'cpu':a_cpu, 'disk':a_disk, 'mem':a_mem, 'nw':a_nw, 'sessions':a_sessions, 'migration_time':a_migration_time, 'mon_interval': a_mon_interval}
                    G.add_node(app_node, **app_node_attrs)
                    G.add_edge(vm_node, app_node, weight="100")

                    sim_stats_counter[app_node] = 0
                    sim_stats.setdefault(app_node, {})
                    task = {}
                    (task['cpu'], task['disk'], task['mem'], task['nw'], task['sessions']) = (a_cpu, a_disk, a_mem, a_nw, a_sessions)
                    update_topo_stats(app_node, task, operator.add)

    update_topo_attrs()
    print(G.nodes())

def update_node_stats(node, stats, operator, timestamp):
    error = 0
    divisor = 1  # Default divisor for an app node
    global sim_stats

    if re.search(vm_node_str, node):
        divisor = max_apps_per_vm
    elif re.search(host_node_str, node):
        divisor = max_vms_per_host * max_apps_per_vm
    elif re.search(edge_node_str, node):
        divisor = max_hosts_per_edge * max_vms_per_host * max_apps_per_vm

    sample_id = sim_stats_counter[node]

    # Whenever you are copying from one dict to another, do a "copy()"
    # Just assigning directly points to the address
    # Future modifications to destn dict is impacting the values in the source dict

    if (sample_id == 0):
        sim_stats_node = {'ite':0, 'timestamp': 0, 'cpu':0, 'disk':0, 'mem': 0, 'nw':0, 'sessions':0, 'mean_ru':0, 'qoe':0, 'best_ru':0, 'best_qoe':0,'migration_time':0, 'mon_interval':0}
    else:
        sim_stats_node = sim_stats[node][sample_id - 1]

    node_curr_stats = sim_stats_node.copy()

    node_curr_stats['ite']      = sample_id
    node_curr_stats['timestamp']    = timestamp

    node_curr_stats['cpu']      = operator(node_curr_stats['cpu'] , (stats['cpu']/divisor))
    node_curr_stats['disk']     = operator(node_curr_stats['disk'] , (stats['disk']/divisor))
    node_curr_stats['mem']      = operator(node_curr_stats['mem'] , (stats['mem']/divisor))
    node_curr_stats['nw']       = operator(node_curr_stats['nw'] , (stats['nw']/divisor))
    node_curr_stats['sessions'] = operator(node_curr_stats['sessions'] , stats['sessions'])

    node_curr_stats['mean_ru']  = calc_node_load_from_stats(node_curr_stats)
    node_curr_stats['qoe']      = gen_qoe(node_curr_stats['mean_ru'])
    node_curr_stats['migration_time']      = get_migration_time_from_node(node)
    node_curr_stats['mon_interval']      = get_mon_interval_from_node(node)

    sim_stats[node][sample_id] = node_curr_stats.copy()
    sim_stats_counter[node]       += 1

    return error

def update_leaf_nodes_attrs_to_root_node(root_node, leaf_nodes):
    global G
    t_root = {}

    for attr in node_attrs:
        t_root[attr] = 0

    if re.search(vm_node_str, root_node):
        divisor = max_apps_per_vm
    elif re.search(host_node_str, root_node):
        divisor = max_vms_per_host
    elif re.search(edge_node_str, root_node):
        divisor = max_hosts_per_edge

    for leaf_node in leaf_nodes:
        for attr in node_attrs:
            t_root[attr] += G.nodes[leaf_node][attr]

    # Just do an average only for the resource attrs. 
    # Do not average for the migration time attr
    # Migration time attr should be a sum of ALL leaf-level migration times
    for attr in node_resource_attrs:
        G.nodes[root_node][attr] = round(t_root[attr]/divisor)
    G.nodes[root_node]['migration_time'] = t_root['migration_time']

    # Mon interval attribute is applicable only for APP nodes
    G.nodes[root_node]['mon_interval'] = mon_interval_default

# End of update_leaf_nodes_attrs_to_root_node(root_node, leaf_nodes)

def update_topo_attrs():
    global G
    error = 0

    t_edge= {}
    t_host = {}
    t_vm= {}

    edges = get_edges_from_cloud(cloud_node_str)
    for edge in edges:
        hosts = get_hosts_from_edge(edge)
        for host_node in hosts:
            vms = get_vms_from_host(host_node)
            for vm_node in vms:
                apps = get_apps_from_vm(vm_node)
                update_leaf_nodes_attrs_to_root_node(vm_node, apps)
            update_leaf_nodes_attrs_to_root_node(host_node, vms)
        update_leaf_nodes_attrs_to_root_node(edge, hosts)

# End of update_topo_attrs()

def update_topo_stats(app_node, task, operator):
    error=0
    stats = {}

    (stats['cpu'], stats['disk'], stats['mem'], stats['nw'], stats['sessions']) = (task['cpu'], task['disk'], task['mem'], task['nw'], task['sessions'])

    if (re.search(app_node_str, app_node)):
        stats['mean_ru'] = calc_node_load(app_node)
        stats['qoe'] = gen_qoe(stats['mean_ru'])
        stats['migration_time'] = get_migration_time_from_node(app_node)
        stats['mon_interval'] = get_mon_interval_from_node(app_node)

    # Store the incremental timestamp in seconds
    timestamp = get_sim_time() - simstart_timestamp

    if (update_node_stats(app_node, stats, operator, timestamp)):
        error = 1
        return error

    vm_node = get_vm_from_app(app_node)
    if (update_node_stats(vm_node, stats, operator, timestamp)):
        error = 1
        return error

    host_node = get_host_from_vm(vm_node)
    if (update_node_stats(host_node, stats, operator, timestamp)):
        error = 1
        return error

    edge_node = get_edge_from_host(host_node)
    if (update_node_stats(edge_node, stats, operator, timestamp)):
        error = 1
        return error

    return error

def print_topology_simple(G):
    for w in range(0,no_of_edges):
        edge_node=edge_node_str + str(w)
        print(edge_node + " (Load: " + str(calc_node_load(edge_node)) + ") " + str(G.nodes[edge_node]))
        for x in range(0,max_hosts_per_edge):
            host_node=host_node_str + str(w) + str(x)
            print(host_node +  " (Load: " + str(calc_node_load(host_node)) + ") " + str(G.nodes[host_node]))
            for y in range(0,max_vms_per_host):
                vm_node=vm_node_str + str(w) + str(x) + str(y)
                print(vm_node +  " (Load: " + str(calc_node_load(vm_node)) + ") " + str(G.nodes[vm_node]))
                for z in range(0,max_apps_per_vm):
                    app_node=app_node_str + str(w) + str(x) + str(y) + str(z)
                    print(app_node +  " (Load: " + str(calc_node_load(app_node)) + ") " + str(G.nodes[app_node]))
                    #print(G.nodes[app_node])

def print_topology(G):
    for w in range(0,no_of_edges):
        edge_node=edge_node_str + str(w)
        print(edge_node + " (Load: " + str(calc_node_load(edge_node)) + ") " + str(G.nodes[edge_node]))
        for x in range(0,max_hosts_per_edge):
            host_node=host_node_str + str(w) + str(x)
            print("|---" + host_node +  " (Load: " + str(calc_node_load(host_node)) + ") " + str(G.nodes[host_node]))
            for y in range(0,max_vms_per_host):
                vm_node=vm_node_str + str(w) + str(x) + str(y)
                print("|   " + "|---" + vm_node +  " (Load: " + str(calc_node_load(vm_node)) + ") " + str(G.nodes[vm_node]))
                for z in range(0,max_apps_per_vm):
                    app_node=app_node_str + str(w) + str(x) + str(y) + str(z)
                    print("|   " + "|   " + "|---" + app_node +  " (Load: " + str(calc_node_load(app_node)) + ") " + str(G.nodes[app_node]))
                    #print(G.nodes[app_node])

def print_sim_stats(node):
    print(sim_stats_counter[node])
    for i in range(0, sim_stats_counter[node]):
        print(node + " : " + str(i) + " : " + str(sim_stats[node][i]))

def write_sim_stats_to_file():

    import pathlib

    from filelock import FileLock

    with FileLock(stats_lock_file):
    # work with the file as it is now locked
        data = pd.DataFrame.from_dict(sim_stats)
        data.to_csv(stats_file, sep=',')

def linear_regression(X, Y):
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    return (Y_pred,linear_regressor.coef_)

def find_min_count(arr):
    min_count = 0
     
    #Loop through the array    
    for i in range(0, len(arr)):
        #Find elements that are violating the threshold
        if(arr[i] < sla_threshold):
            min_count = min_count + 1

    return min_count

#End of find_min_count

def calc_qoe_violations(stats):

    # Import pandas
    import pandas as pd
    global total_qoe_violations
    global total_node_instances_with_violations
 
    t_data = pd.DataFrame(stats)
    if t_data.empty:
        return

    data = t_data.transpose()

    nsamples= len(data.index)
    if (nsamples < 10):
        return

    #log_debug(data)

    I = data['ite']
    C = data['cpu']
    D = data['disk']
    N = data['nw']
    M = data['mean_ru']
    Q = data['qoe']


    xmin = min(I)
    xmax = max(I)
    # Do not consider smaller # of samples
    if (xmax < 40):
        return

    # Find the minimum QoE value
    min_count = find_min_count(Q)
    # Check if the QoE value violates SLA threshold
    if (min_count):
        total_node_instances_with_violations = total_node_instances_with_violations + 1
        total_qoe_violations = total_qoe_violations + min_count

# End of calc_qoe_violations

def check_node_perf_trend_STATIC(node):
    import numpy as np

    trend = NO_TREND_SEEN

    stats = sim_stats[node]
    t_data = pd.DataFrame(stats)
    if t_data.empty:
        return (NO_TREND_SEEN)

    data = t_data.transpose()
    data_filtered = data[data['mean_ru'] != 0]
    data = data_filtered

    nsamples= len(data.index)
    if (nsamples < 10):
        print("Return - 1")
        return (NO_TREND_SEEN)

#    data.round()

    # I is the iteration (Column #0)        / T is the Timestamp in ms (Column #1)
    # C is the CPU utilisation (Column #2)  / D is disk utilisation (Column #3)
    # MEM is memory utilisation (Column #4) / N is network utilisation (Column #5)
    # S is the user sessions (Column #6)
    # M is the Mean RU column (Column #7)   / Q is the QoE column (Column #8)
    # Best_ru is Column #9                  / Best QoE is Column #10
    # Migration Time is Column #11           / Monitoring Interval is Column #12
    # APS is Column #13                     / R is the Rate of change of RU (Column #14)
    # EWMA is Column #15

    I   = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
    T   = data.iloc[:, 1].values.reshape(-1, 1)  # values converts it into a numpy array
    C   = data.iloc[:, 2].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    D   = data.iloc[:, 3].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    MEM = data.iloc[:, 4].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    N   = data.iloc[:, 5].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    S   = data.iloc[:, 6].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    M   = data.iloc[:, 7].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    Q   = data.iloc[:, 8].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column


    APS = [None] * nsamples
    APS = (Q/M)+(M)
    data['APS'] = APS

    data.replace([np.inf, -np.inf], 0, inplace=True)
    data.dropna(inplace=True)

    if (all(i < aps_threshold for i in APS)):
        print("Return - 2")
        return (NO_TREND_SEEN)

    import numpy as np

    p_data = pd.DataFrame(M)

    data['EWMA'] =  p_data.ewm(com=0.5).mean()

    data.to_csv("graphs-data/gsim_stats.txt", sep=',')

    # Saro: Dummy... main purpose is to record the stats in the file
    return (TREND_SEEN)

def check_node_perf_trend_EWMA3(node):
    import numpy as np

    trend = NO_TREND_SEEN

    stats = sim_stats[node]
    t_data = pd.DataFrame(stats)
    if t_data.empty:
        return (NO_TREND_SEEN)

    data = t_data.transpose()
    data_filtered = data[data['mean_ru'] != 0]
    data = data_filtered

    nsamples= len(data.index)
    if (nsamples < 10):
        print("Return - 1")
        return (NO_TREND_SEEN)

#    data.round()

    # I is the iteration (Column #0)        / T is the Timestamp in ms (Column #1)
    # C is the CPU utilisation (Column #2)  / D is disk utilisation (Column #3)
    # MEM is memory utilisation (Column #4) / N is network utilisation (Column #5)
    # S is the user sessions (Column #6)
    # M is the Mean RU column (Column #7)   / Q is the QoE column (Column #8)
    # Best_ru is Column #9                  / Best QoE is Column #10
    # Migration Time is Column #11           / Monitoring Interval is Column #12
    # APS is Column #13                     / R is the Rate of change of RU (Column #14)
    # EWMA is Column #15

    I   = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
    T   = data.iloc[:, 1].values.reshape(-1, 1)  # values converts it into a numpy array
    C   = data.iloc[:, 2].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    D   = data.iloc[:, 3].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    MEM = data.iloc[:, 4].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    N   = data.iloc[:, 5].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    S   = data.iloc[:, 6].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    M   = data.iloc[:, 7].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    Q   = data.iloc[:, 8].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column


    APS = [None] * nsamples
    APS = (Q/M)+(M)
    data['APS'] = APS

    data.replace([np.inf, -np.inf], 0, inplace=True)
    data.dropna(inplace=True)

    if (all(i < aps_threshold for i in APS)):
        print("Return - 2")
        return (NO_TREND_SEEN)

    import numpy as np

    p_data = pd.DataFrame(M)

    data['EWMA'] =  p_data.ewm(com=0.5).mean()

    data.to_csv("graphs-data/gsim_stats.txt", sep=',')

    import  pandas as pdtrim

    total_samples = nsamples

    if (re.match(migration_algorithm, "ewma")):

        forecast_data = data['EWMA']
        forecast_data = forecast_data[~np.isnan(forecast_data)]
        print("Forecast Data")
        print(forecast_data)
        df_train = forecast_data[-10:]

        df_train.dropna()
        df_train.reset_index(drop=True)

        print("Train Data")
        print(df_train.values)

        # triple exponential smoothing - holtwinters
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        fitted_model = ExponentialSmoothing(df_train.values,trend='mul',seasonal='mul',seasonal_periods=5).fit()
        df_test_predictions = fitted_model.forecast(5)

        # If you give 10 numbers for training, it forecasts the 10, 11, 12, 13 and 14th numbers
        forecasted_threshold = df_test_predictions[0]
#        print("Fcast " + " " + str(forecasted_threshold) + " " + "Pred[14] " + " " + str(df_test_predictions[14]))
        if (forecasted_threshold > aps_threshold):
            print("Return - 2.6")
            trend=TREND_SEEN
            return (TREND_SEEN)
    # End of if condition for ewma

    if (trend == TREND_SEEN):
       print("Return - 4 " + " Trend Seen")

    return (trend)

# End of check_node_perf_trend_EWMA3()

def check_node_perf_trend_LR(node):
    import numpy as np

    trend = NO_TREND_SEEN

    stats = sim_stats[node]
    t_data = pd.DataFrame(stats)
    if t_data.empty:
        return (NO_TREND_SEEN)

    data = t_data.transpose()
    data_filtered = data[data['mean_ru'] != 0]
    data = data_filtered

    nsamples= len(data.index)
    if (nsamples < 10):
        print("Return - 1")
        return (NO_TREND_SEEN)

#    data.round()

    # I is the iteration (Column #0)        / T is the Timestamp in ms (Column #1)
    # C is the CPU utilisation (Column #2)  / D is disk utilisation (Column #3)
    # MEM is memory utilisation (Column #4) / N is network utilisation (Column #5)
    # S is the user sessions (Column #6)
    # M is the Mean RU column (Column #7)   / Q is the QoE column (Column #8)
    # Best_ru is Column #9                  / Best QoE is Column #10
    # Migration Time is Column #11           / Monitoring Interval is Column #12
    # APS is Column #13                     / R is the Rate of change of RU (Column #14)
    # EWMA is Column #15

    I   = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
    T   = data.iloc[:, 1].values.reshape(-1, 1)  # values converts it into a numpy array
    C   = data.iloc[:, 2].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    D   = data.iloc[:, 3].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    MEM = data.iloc[:, 4].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    N   = data.iloc[:, 5].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    S   = data.iloc[:, 6].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    M   = data.iloc[:, 7].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    Q   = data.iloc[:, 8].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

    APS = [None] * nsamples
    APS = (Q/M)+(M)
    data['APS'] = APS

    data.replace([np.inf, -np.inf], 0, inplace=True)
    data.dropna(inplace=True)

    if (all(i < aps_threshold for i in APS)):
        print("Return - 2")
        return (NO_TREND_SEEN)

    import numpy as np

    # SARO: This logic has further options for optimization
    # and converting it into a separate research paper

    # R is the rate of change of Resource Utilisation

    p_data = pd.DataFrame(M)
    R = p_data.pct_change(fill_method ='ffill').to_numpy()*100
    data['R'] = R

    r_data = pd.DataFrame(R)

    # Moving average of R
    data['EWMA'] =  r_data.ewm(com=0.5).mean()

    data.to_csv("graphs-data/gsim_stats.txt", sep=',')

    import  pandas as pdtrim

    total_samples = nsamples
    while (total_samples >= 5):
        # Start with the recent samples
        pdtrim = data.tail(total_samples)

        I1 = pdtrim.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
        M1 = pdtrim.iloc[:, 7].values.reshape(-1, 1)  # values converts it into a numpy array
        APS1 = pdtrim.iloc[:, 13].values.reshape(-1, 1)  # values converts it into a numpy array
 
        if (all(i < aps_threshold for i in APS1)):
            # Samples do not indicate high APS
            # No need to run LR algorithm
            total_samples = round(total_samples/2)
            continue

        (Y_pred, slope) = linear_regression(I1,APS1)
        score=r2_score(APS1, Y_pred)

        if ((score >= 0.8) and (slope > 0)):
            # R2Score of > 0.8 indicates a trend of increase in resource utilisation (RU)
            # Positive slope indicates that the RU is going up.
            log_info("TREND - SEEN (samples count)", node, total_samples)
            trend=TREND_SEEN
            break

        # End of if condition
        total_samples = round(total_samples/2)
    # End of while loop

    if ((trend == TREND_SEEN) and (all(i < aps_threshold for i in APS1))):
        print("Return - 3")
        return (NO_TREND_SEEN)

    if (trend == TREND_SEEN):
       print("Return - 4 " + " Trend Seen")

    return (trend)

# End of check_node_perf_trend()
    
def forecast_sarimax(forecast_param):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as f_plt
    import sys
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Read the dataset
    sim_data = pd.read_csv('graphs-data/gsim_stats.txt',
					index_col ='ite',
					parse_dates = True)

    # order (0,1,1) refers to exponential smoothing of values
    # Train the model on the full dataset
    model = SARIMAX(sim_data[forecast_param],
                        order = (0, 1, 1), 
                        seasonal_order =(2, 1, 1, 12), enforce_stationarity=False)
    result = model.fit()
  
    sim_data_len = len(sim_data)
    predictions_count = 10
    prediction_start_value = sim_data_len
    prediction_end_value   = (sim_data_len - 1) + predictions_count

    # Forecast for the next 10 iterations/cycles
    forecast = result.predict(start = prediction_start_value,
                          end = prediction_end_value,
                          typ = 'levels').rename('Forecast')
  
    # Plot the forecast values
    #sim_data[forecast_param].plot(figsize = (12, 5), legend = True)
    #forecast.plot(legend = True)
    #f_plt.show()

    return (forecast)

def get_pso_stats_for_node (node, param):
    return (pso_stats[node][param])
# End of get_pso_stats_for_node

def get_ru_threshold_for_migration(node):

    ru_for_app   = ru_for_vm = ru_for_host = ru_for_edge = 0

    # Special logic for vm node
    if re.search(vm_node_str, node):
        vm_node = node
        host_node = get_host_from_vm(vm_node)
        edge = get_edge_from_host(host_node)

        ru_for_vm   = get_pso_stats_for_node(vm_node, 'best_ru')
        ru_for_host = get_pso_stats_for_node(host_node, 'best_ru')
        ru_for_edge = get_pso_stats_for_node(edge, 'best_ru')
        ru_for_global = get_pso_stats_for_node(cloud_node_str, 'best_ru')
    # End of Special logic for vm node
    elif re.search(app_node_str, node):
        vm_node = get_vm_from_app(node)
        host_node = get_host_from_vm(vm_node)
        edge = get_edge_from_host(host_node)

        ru_for_app   = get_pso_stats_for_node(node, 'best_ru')
        ru_for_vm   = get_pso_stats_for_node(vm_node, 'best_ru')
        ru_for_host = get_pso_stats_for_node(host_node, 'best_ru')
        ru_for_edge = get_pso_stats_for_node(edge, 'best_ru')
        ru_for_global = get_pso_stats_for_node(cloud_node_str, 'best_ru')

    if (ru_for_global == 0):
        ru_for_global = ru_default_threshold
    if (ru_for_edge == 0):
        ru_for_edge = ru_default_threshold
    if (ru_for_host== 0):
        ru_for_host = ru_default_threshold
    if (ru_for_vm == 0):
        ru_for_vm = ru_default_threshold
    if (ru_for_app == 0):
        ru_for_app = ru_default_threshold

# SARO: Scope for playing around with this logic to get the best method
# SARO: This needs some checking
    ru_threshold = min(ru_for_global, ru_for_edge, ru_for_host, ru_for_vm, ru_for_app)

    if (ru_threshold == ru_default_threshold):
        ru_entity = "default"
    elif (ru_threshold == ru_for_global):
        ru_entity = "cloud"
    elif (ru_threshold == ru_for_edge):
        ru_entity = "edge"
    elif (ru_threshold == ru_for_host):
        ru_entity = "host"
    elif (ru_threshold == ru_for_vm):
        ru_entity = "vm"
    elif (ru_threshold == ru_for_app):
        ru_entity = "app"
    else:
        print("ERROR: Using None... something terribly wrong")

    print(node + ": ru_for_app: " + str(ru_for_app) + " ru_for_vm: " + str(ru_for_vm) + " ru_for_host: " + str(ru_for_host) + " ru_for_edge: " + str(ru_for_edge) + " ru_for_global: " + str(ru_for_global) + " ru_entity: " + str(ru_entity))

    return (ru_threshold, ru_entity)
# End of get_ru_threshold_for_migration()

def get_host_for_task_assignment(edge):

    host_nodes = get_hosts_from_edge(edge)
    for host_node in host_nodes:
        if (is_node_load_low(host_node)):
                return host_node
    return ""
# End of get_host_for_task_assignment

def get_vm_for_task_assignment(host_node):

    if (is_node_load_low(host_node)):
        vm_nodes = get_vms_from_host(host_node)
        for vm_node in vm_nodes:
            if (is_node_load_low(vm_node)):
                return vm_node
    return ""

# End of get_vm_for_task_assignment

def get_app_for_task_assignment(vm_node):
    app_nodes = get_apps_from_vm(vm_node)
    for app_node in app_nodes:
        if (is_node_load_low(app_node)):
            return app_node
    return ""

# End of get_app_for_task_assignment

def get_app_for_task_assignment_in_edge(edge, src_app_node):
    app_nodes = get_apps_from_edge(edge)
    for app_node in app_nodes:
        if (re.search(app_node, src_app_node)):
            ## Src and Dst apps cannot be the same
            continue
        print("get_app_for_task_assignment_in_edge: " + str(app_node) + " load is " + str(calc_node_load(app_node)))

        if (is_node_load_low(app_node)):
            return app_node
    return ""

# End of get_app_for_task_assignment

def print_load_and_migration_time_for_apps(app_nodes):
    print("Printing migration time: ")
    if (len(app_nodes) == 0):
        print("No nodes available to migrate within the stipulated time")
        return
    for app_node in app_nodes:
        print("Node: " + str(app_node) + " Load: " + str(calc_node_load(app_node)) + " Migration Time: " + str(G.nodes[app_node]['migration_time']))

#End of print_migration_time_for_apps(app_nodes):

def get_apps_from_vm_based_on_load(vm_node):
    apps_dict = {}
    app_nodes = get_apps_from_vm(vm_node)
    for app_node in app_nodes:
        app_load = calc_node_load(app_node)
        apps_dict[app_node]=app_load
    # Sort the apps based on load (highly loaded apps first, least loaded apps at the end)
    s_apps_dict = dict(sorted(apps_dict.items(), key=lambda x:x[1], reverse=True))

    #print("Load Apps list : " + str(apps_dict))
    #print("Load Sorted list : " + str(s_apps_dict))
    #print("Load Sorted apps list: " + str(s_apps_dict.keys()))

    return s_apps_dict.keys()
# End of get_apps_from_vm_based_on_load

def get_apps_from_vm_based_on_migration_time(vm_node, migration_time):

    get_apps_from_vm_based_on_load(vm_node)
    apps_dict = {}
    t_apps_dict = {}

    # SARO: Apr 19th, 2023 - this requires a cleanup

    app_nodes = get_apps_from_vm(vm_node)
    print("Allotted time for migration is " + str(migration_time))
    for app_node in app_nodes:
        # Pick a loaded node with lesser migration time
        apps_dict[app_node] = G.nodes[app_node]['migration_time']
        if ((calc_node_load(app_node) > ru_default_threshold) and
            (G.nodes[app_node]['migration_time'] < migration_time)):
            t_apps_dict[app_node] = G.nodes[app_node]['migration_time']

    # Sort the apps based on migration time (lesser migration time first, higher migration time at the end)
    s_apps_dict = dict(sorted(apps_dict.items(), key=lambda x:x[1], reverse=False))
    t_s_apps_dict = dict(sorted(t_apps_dict.items(), key=lambda x:x[1], reverse=False))

    #print("Time & Load Sorted list : " + str(t_s_apps_dict))
    #print("Time Sorted list : " + str(s_apps_dict))
    #print("Time Sorted apps list: " + str(s_apps_dict.keys()))
    #sys.exit()

    return t_apps_dict.keys()
# End of get_apps_from_vm_based_on_migration_time

def get_apps_from_edge_based_on_migration_time(edge, migration_time):
    apps = [] 
    host_nodes = get_hosts_from_edge(edge)
    for host_node in host_nodes:
        vm_nodes = get_vms_from_host(host_node)
        for vm_node in vm_nodes:
            app_nodes = get_apps_from_vm(vm_node)
            for app_node in app_nodes:
                if (G.nodes[app_node]['migration_time'] < migration_time):
                    apps.append(app_node)
    return apps
# End of get_apps_from_edge_based_on_migration_time

def get_apps_from_edge(edge):
    apps = [] 
    host_nodes = get_hosts_from_edge(edge)
    for host_node in host_nodes:
        vm_nodes = get_vms_from_host(host_node)
        for vm_node in vm_nodes:
            app_nodes = get_apps_from_vm(vm_node)
            for app_node in app_nodes:
                apps.append(app_node)
    return apps
#End of get_apps_from_edge

def get_apps_from_vm(vm_node):
    apps = [] 
    if re.search(vm_node_str, vm_node):
        for node in G.neighbors(vm_node):
            if re.search(app_node_str, node):
                apps.append(node)
    return apps 
#End of get_apps_from_vm

def get_vms_from_host(host_node):
    vms = [] 
    if re.search(host_node_str, host_node):
        for node in G.neighbors(host_node):
            if re.search(vm_node_str, node):
                vms.append(node)
    return vms
#End of get_vms_from_host

def get_host_from_app(app):
    vm_node = get_vm_from_app(app)
    host_node = get_host_from_vm(vm_node)
    return host_node
# End of get_host_from_app 

def get_hosts_from_edge(edge):
    hosts = [] 
    if re.search(edge_node_str, edge): 
        for node in G.neighbors(edge):
            if re.search(host_node_str, node):
                hosts.append(node)
    return hosts
# End of get_hosts_from_edge

def get_edges_from_cloud(cloud):
    edges = [] 
    if re.search(cloud_node_str, cloud):
        for node in G.neighbors(cloud):
            if re.search(edge_node_str, node):
                edges.append(node)
    return edges
#End of get_edges_from_cloud

def update_pso_stats_for_root_node(root_node, leaf_nodes, param):
# For example, root node = vm_node / leaf node = app_nodes
#              root node = host_node / leaf node = vm_nodes

        best_node_param = 0
        best_node  = ""
        best_node_param_resonance = 0

        resonance = param + "_resonance"
        for node in leaf_nodes:
            if (pso_stats[node][resonance] > best_node_param_resonance):
                best_node_param_resonance = pso_stats[node][resonance]
                best_node_param = pso_stats[node][param]
                best_node = node

        if (pso_stats[root_node][param] == best_node_param):
            pso_stats[root_node][resonance] += 1
        elif ((pso_stats[root_node][param] < best_node_param) and (pso_stats[root_node][resonance] <= best_node_param_resonance)):
            pso_stats[root_node][param] = best_node_param
            pso_stats[root_node][resonance] = 1

# End of update_pso_stats_for_root_node


def record_stats(task_id, app):

    (cpu, disk, mem, nw, sessions) = get_resource_attrs_from_node(app)
    app_mean_ru = calc_node_load(app)
    app_qoe = gen_qoe(app_mean_ru)
    app_migration_time = get_migration_time_from_node(app)
    app_mon_interval = get_mon_interval_from_node(app)
    timestamp = get_sim_time() - simstart_timestamp

    best_qoe = best_ru = 0

    stats = {'timestamp': timestamp, 'cpu': cpu, 'disk': disk, 'mem': mem, 'nw': nw, 'sessions':sessions, 'mean_ru':app_mean_ru, 'qoe':app_qoe, 'best_ru':best_ru, 'best_qoe':best_qoe,'migration_time':app_migration_time, 'mon_interval':app_mon_interval}

    # Record the stats for the app
    sample_id=sim_stats_counter[app]
    a_stats = {}
    h_stats = {}
    a_stats['ite'] = sample_id
    h_stats['ite'] = task_id

    for key in stats.keys():
        a_stats[key] = stats[key]
    sim_stats[app][sample_id] = a_stats
    sim_stats_counter[app] += 1
    
    vm_node = get_vm_from_app(app)

    # Record the stats for the host (corresponding to the app)
    host = get_host_from_vm(vm_node)
    (h_cpu, h_disk, h_mem, h_nw, h_sessions) = get_resource_attrs_from_node(host)
    h_mean_ru = calc_node_load(host)

    # Record the stats for the "region" edge cloud (corresponding to the app)
    ec = get_edge_from_host(host)
    (e_cpu, e_disk, e_mem, e_nw, e_sessions) = get_resource_attrs_from_node(ec)
    e_mean_ru = calc_node_load(ec)

    # Calculate the Avg QoE of ALL Apps in the Host and record that at host level
    # Calculate the total migration time for all apps and record that at host level

    sum_host_qoe = 0
    no_of_apps = 0
    sum_host_migration_time = 0
    for app in get_apps_from_vm(vm_node):
        no_of_apps += 1
        sample_id = sim_stats_counter[app]
        if (sim_stats[app] and sample_id):
            #print("Latest App QoE: ", sim_stats[app][sample_id - 1]['qoe'])
            sum_host_qoe += sim_stats[app][sample_id - 1]['qoe']
            sum_host_migration_time += sim_stats[app][sample_id - 1]['migration_time']
    h_qoe = 0
    if (no_of_apps > 0):
        h_qoe = sum_host_qoe/no_of_apps

    t_stats = {'timestamp': timestamp, 'cpu': h_cpu, 'disk': h_disk, 'mem': h_mem, 'nw': h_nw, 'sessions':h_sessions, 'mean_ru':h_mean_ru, 'qoe':h_qoe, 'best_ru':best_ru, 'best_qoe':best_qoe, 'migration_time':sum_host_migration_time}
    for key in t_stats.keys():
        h_stats[key] = t_stats[key]
    sample_id = sim_stats_counter[host]
    sim_stats[host][sample_id] = h_stats
    sim_stats_counter[host] += 1

# End of record_stats()

def record_pso_stats(node, param, value):

    print("Updating PSO stats for node - " + node + " migration algorithm is " + str(migration_algorithm))

    if re.search(app_node_str, node):
        # In addition to updating the stats for the App,
        # record the stats for the VM and host (corresponding to the app)
        vm_node = get_vm_from_app(node)
    elif re.search(vm_node_str, node):
        vm_node = node

    host_node = get_host_from_vm(vm_node)
    edge = get_edge_from_host(host_node)

    app_nodes = get_apps_from_vm(vm_node)
    vm_nodes  = get_vms_from_host(host_node)
    host_nodes = get_hosts_from_edge(edge)
    edge_nodes = get_edges_from_cloud(cloud_node_str)

    value = round(value)

    resonance = param + "_resonance"
    node_mean_ru = calc_node_load(node)

    if (re.fullmatch("clapso", migration_algorithm) or re.fullmatch("clapsot", migration_algorithm)):

        if (node_mean_ru >= ru_max_threshold):
            # This condition would not be hit mostly
            return

        # Update the APP level threshold
        if re.search(app_node_str, node):
            if (pso_stats[node][param] == value):
                pso_stats[node][resonance] += 1
            elif (pso_stats[node][param] < value):
                pso_stats[node][param] = value
                pso_stats[node][resonance] = 1

        # Update the VM level threshold
        update_pso_stats_for_root_node(vm_node, app_nodes, param)

        # Update the host level threshold 
        update_pso_stats_for_root_node(host_node, vm_nodes, param)

        # Update the edge level threshold 
        update_pso_stats_for_root_node(edge, host_nodes, param)

        # Update the global level threshold 
        if (pso_stats[cloud_node_str][param] == value):
            pso_stats[cloud_node_str][resonance] += 1
        # Note: This is for CLAPSOT feedback (> instead of <)
        #elif (pso_stats[cloud_node_str][param] > value):
        else:
            pso_stats[cloud_node_str][param] = value
            pso_stats[cloud_node_str][resonance] = 1

        for t_node in G.nodes():
            if not (re.search(ue_node_str, t_node)):
                if (pso_stats[t_node][param] or pso_stats[t_node][resonance]):
                    print("node: " + t_node + " " + param + ": " + str(pso_stats[t_node][param]) + " " + resonance + ": " + str(pso_stats[t_node][resonance]))

    elif (re.fullmatch("regpso", migration_algorithm)):
        if (pso_stats[node][param] < value):
            pso_stats[node][param] = value

        if (pso_stats[vm_node][param] < value):
            pso_stats[vm_node][param] = value

        if (pso_stats[host_node][param] < value):
            pso_stats[host_node][param] = value 

        if (pso_stats[edge][param] < value):
            pso_stats[edge][param] = value 

        if (pso_stats[cloud_node_str][param] < value):
            pso_stats[cloud_node_str][param] = value 

# End of record_pso_stats()

def print_pso_stats(iteration, taskid, param):
    print("PSO Stats")
    node = cloud_node_str
    print("Iteration: ", iteration, "Taskid: ", taskid, node + " : " + " RU: " + str(round(pso_stats[node][param],1)))

    ec = list(filter(lambda x: edge_node_str in x, G.nodes))
    for node in ec:
        log_debug("Iteration: ", iteration, "Taskid: ", taskid, node + " : " + " RU: " + str(round(pso_stats[node][param],1)))

    hosts = list(filter(lambda x: host_node_str in x, G.nodes))
    for node in hosts:
        if not re.search(ue_node_str, node):
            log_debug("Iteration: ", iteration, "Taskid: ", taskid, node + " : " + " RU: " + str(round(pso_stats[node][param],1)))

    vms = list(filter(lambda x: vm_node_str in x, G.nodes))
    for node in vms:
        if not re.search(ue_node_str, node):
            log_debug("Iteration: ", iteration, "Taskid: ", taskid, node + " : " + " RU: " + str(round(pso_stats[node][param],1)))

    app_nodes = list(filter(lambda x: app_node_str in x, G.nodes))
    for node in app_nodes:
        log_debug("Iteration: ", iteration, "Taskid: ", taskid, node + " : " + "Load Status: " + str(is_node_load_high(node)) + " RU: " + str(round(pso_stats[node][param],1)))


def get_attrs_node(G, node):
    return G.nodes[node].values();

#def migrate_tasks_from_vm (src_vm_node, time_for_migration):
#
#    global total_mtime_violations
#    migration_count = 0
#    load_reduced = 0
#    vm_load_initial = 0
#
#    if re.fullmatch("clapsot", migration_algorithm):
#        app_nodes = get_apps_from_vm_based_on_migration_time(src_vm_node, time_for_migration)
#        print ("List of app nodes based on migration time " + str(time_for_migration) + " ms : " + str(app_nodes))
#        if (len(app_nodes) == 0):
#            print ("Falling back to load based migration")
#            app_nodes = get_apps_from_vm_based_on_load(src_vm_node)
#            print ("List of app nodes based on load: " + str(app_nodes))
#    elif re.fullmatch("clapso", migration_algorithm):
#        # Plain clapso
#        app_nodes = get_apps_from_vm_based_on_load(src_vm_node)
##        print ("List of app nodes based on load: " + str(app_nodes))
#    else:
#        app_nodes = get_apps_from_vm_based_on_load(src_vm_node)
#
#    if (len(app_nodes) == 0):
#        # If there are no nodes that match the migration_time, then it is considered as an SLA violation
#        total_mtime_violations += 1
#
#    vm_load_initial = calc_node_load(src_vm_node)
#    print("VM Node " + src_vm_node + " load (initial) " + str(vm_load_initial))
#    vm_load_curr = 0
#
#    for app_node in app_nodes:
#        print("Selected " + app_node + " for task migration" + " app load is " + str(calc_node_load(app_node)))
#        print("VM Node " + src_vm_node + " load (interim) " + str(calc_node_load(src_vm_node)))
#
#        initial_app_sla_violations  = sla_stats[app_node_str]['total']
#        migration_count += migrate_tasks_from_app (app_node, time_for_migration)
#        vm_load_curr = calc_node_load(src_vm_node)
#        final_app_sla_violations  = sla_stats[app_node_str]['total']
#        new_app_sla_violations = final_app_sla_violations - initial_app_sla_violations
#        
#        if (new_app_sla_violations):
#            sla_stats[vm_node_str]['total'] = sla_stats[vm_node_str]['total'] + new_app_sla_violations
#
#        if (vm_load_curr < ru_migration_threshold):
#            load_reduced = 1
#            break
#
#    if (load_reduced and (vm_load_initial - vm_load_curr)):
#        load_reduced = 1
#    else:
#        load_reduced = 0
#
#    print(src_vm_node + " VM Node load (initial) " + str(vm_load_initial) + " VM Node load (final) " + str(calc_node_load(src_vm_node)), end = "")
#
#    if (load_reduced):
#        print(" Load Reduced below " + str(ru_migration_threshold))
#    else:
#        print(" Load NOT Reduced below " + str(ru_migration_threshold))
#    return migration_count
#

def migrate_tasks_from_app (src_app_node, time_for_migration):
    global total_tasks_reassigned
    global total_mtime_violations
    global tasks_assigned_count
    global task_stats_counter
    global total_node_migrations
    global total_load_reduced
    global ru_migration_threshold

    ret = 1
    task_migrations_count = 0
    migration_start_time = migration_end_time = 0
    sla_miss = 0
    migrated_sessions_count = 0
    load_reduced = 0


    migration_start_time = get_sim_time()
    total_node_migrations = total_node_migrations + 1

    print("App migration time:" + str(G.nodes[src_app_node]['migration_time']) + " " + "Time for migration: " + str(time_for_migration))

    load_before_migration = str(get_attrs_node(G, src_app_node))
    mean_ru_before_migration = calc_node_load(src_app_node)

    log_debug("App: ", src_app_node , "Load B4: ", load_before_migration)

    edge = get_edge_from_app(src_app_node)

    # Perform task migrations till the load is reduced

    print("Node " + src_app_node + " load is " + str(calc_node_load(src_app_node)))

    migration_time = 0
    while (calc_node_load(src_app_node) >= ru_migration_threshold):
        print("Came here - 1 ")
        
        if (re.fullmatch("clapso", migration_algorithm) or re.fullmatch("ewma", migration_algorithm) or re.fullmatch("static", migration_algorithm) or re.fullmatch("regpso", migration_algorithm)):
            ue_nodes = get_ues_from_app(src_app_node)
        elif (re.fullmatch("clapsot", migration_algorithm)):
            # If the entire app can be migrated within the allotted time, then do that
            if (time_for_migration > get_migration_time_from_node(src_app_node)):
                ue_nodes = get_ues_from_app(src_app_node)
                print("full migrations performed")
            else:
                print("partial migrations performed")
                # Get the partial list of UEs which can be migrated within the allotted time
                # ue_nodes = get_ues_from_app_based_on_migration_time(src_app_node, time_for_migration)
                ue_nodes = get_ues_from_app_based_on_knapsack(src_app_node, time_for_migration)
  
        for ue_node in ue_nodes:

            m_time         = get_migration_time_from_node(ue_node)
            sessions_count = get_sessions_count_from_ue(ue_node)

            if (re.fullmatch("clapsot", migration_algorithm)):
                # Time taken for migration greater than time allotted for migration
                if (migration_time + m_time > time_for_migration):
                    print("Came here - 1.5 (partial migration)")
                    ret = 1
                    break
                else:
                    print("App: " + src_app_node + " UE: " + ue_node +  " Actual Time: " + str(migration_time) + " Allotted Time: " + str(time_for_migration))

            print("Came here - 2 ")
            task_type=get_task_type_from_ue (ue_node)

            if (task_type is None):
                print("Task type is None. ERROR")
                # Ideally, it should not come here !
                # All UEs should map to a specific task type
                continue
            ret = unassign_task(src_app_node, task_type, ue_node)

            log_debug("App ", src_app_node , "UE Node: ", ue_node, "Unassigned Status: ", ret)

            if not (ret):
                # Task successfully unassigned
                # Now, reassign the task to a different app in the same edge cloud

                dst_app_node = get_app_for_task_assignment_in_edge(edge, src_app_node)
                log_debug("Result of low ru app search : ", dst_app_node)
                if (dst_app_node == ""):
                    sla_stats[app_node_str]['total'] += 1
                    ret = 1
                    break

                log_debug("Low RU App ", dst_app_node, "load before migration", get_attrs_node(G, dst_app_node))

                (app_nr, taskid) = re.findall("[0-9]+", ue_node)
                app_nr = get_app_number_from_name(dst_app_node)
                n_ue_name = ue_node_str + app_nr + "." + taskid
   
                ret = assign_task_to_app(dst_app_node, int(taskid))

                if not (ret):
                    # Task successfully re-assigned / migrated to a new app in the same host
                    log_debug("Low RU App ", dst_app_node, "load after migration", get_attrs_node(G, dst_app_node))
                    log_debug("Migrating task " + "from " + src_app_node + " to " + dst_app_node + " Old UE: " + ue_node + " New UE: " + n_ue_name)
                    task_migrations_count += 1
                    total_tasks_reassigned += 1

            migrated_sessions_count += sessions_count
            migration_time += m_time

            # Break out of UE Nodes loop, if the app load is reduced
            if (calc_node_load(src_app_node) <= ru_min_threshold):
                print("Came here - 2.6 ")
                break

            # Break out of UE Nodes loop, if the system cannot unassign/reassign a task
            if (ret):
                print("Came here - 2.5 ")
                break
        # End of UE nodes loop

        # Break out of App node loop , if the system cannot unassign/reassign a task
        if (ret):
            print("Came here - 3 ")
            break
    # End of App node loop

    # Migration end time is computed by adding the time taken to migrate ALL tasks from an APP
    migration_end_time = migration_start_time + migration_time

    # Check if the time allotted for migration was breached
    if (migration_time > time_for_migration):
        print("Time taken for migration is greater than the allotted migration time")
        sla_miss = 1
        total_mtime_violations += 1

    load_after_migration = str(get_attrs_node(G, src_app_node))
    mean_ru_after_migration = calc_node_load(src_app_node)

    task_stats[task_stats_counter] = {'mig_start_timestamp': migration_start_time, 'mig_end_timestamp': migration_end_time, 'tasks_assigned':tasks_assigned_count, 'sessions_migrated': migrated_sessions_count, 'sla_miss':sla_miss, 'mean_ru_before_migration': mean_ru_before_migration, 'mean_ru_after_migration': mean_ru_after_migration, 'time_for_migration': time_for_migration}
    task_stats_counter = task_stats_counter + 1

    tasks_assigned_count = 0

    print("Came here - 4 ")

    if (calc_node_load(src_app_node) < ru_max_threshold):
        load_reduced = 1
        total_load_reduced = total_load_reduced + 1

    log_info("High RU App ", src_app_node, "load b4 migration", load_before_migration, " RU", mean_ru_before_migration)
    log_info("High RU App ", src_app_node, "load a4 migration", load_after_migration, " RU", mean_ru_after_migration, " Count", task_migrations_count, " Load reduced: ", load_reduced)

    return (task_migrations_count, migration_time)

def migrate_tasks(src_node, node_type, time_for_migration):
    global total_tasks_reassigned
    global ru_migration_threshold
    global G

    task_migrations_count = 0
    task_migration_time = 0
    ret = 1

    # Noting down the SLA violations before migration is initiated

    initial_sla_violations  = sla_stats[node_type]['total']

    # Noting down the Mean RU before migration is initiated
    start_node_mean_ru = calc_node_load(src_node)

    # Noticing an increasing RU trend for the node
    print("Noticing an increasing RU trend for ", src_node)

    # Normalise the load by migrating tasks
    if (re.search(node_type, app_node_str)):
        (task_migrations_count, task_migration_time) = migrate_tasks_from_app(src_node, time_for_migration)
        print("I came in search of apps")
    elif (re.search(node_type, vm_node_str)):
        task_migrations_count = migrate_tasks_from_vm(src_node, time_for_migration)
        print("I came in search of vm")
    else:
        print("ERROR: Wrong node type for migration")
        sys.exit()

    # Noting down the SLA violations after the migration is performed
    final_sla_violations  = sla_stats[node_type]['total']

    print("Initial violations: " + str(initial_sla_violations) + " final_sla_violations: " + str(final_sla_violations))
    # Noting down the Mean RU after migration is complete
    final_node_mean_ru = calc_node_load(src_node)

    if (task_migrations_count):
        ru_per_task = round((start_node_mean_ru - final_node_mean_ru)/task_migrations_count)
        mt_per_task = round(task_migration_time/task_migrations_count)

        # SARO: Temporarily it is here - 7th Jul 2023
        # If there are violations (due to migration time), then we need to adjust the starting ru threshold
        # i.e., starts migrations in advance to prevent a SLA violation (due to time)

        if (task_migrations_count and re.fullmatch("clapsot", migration_algorithm) and (task_migration_time > time_for_migration)):
            time_diff = task_migration_time - time_for_migration
            print(task_migrations_count, task_migration_time, time_for_migration, time_diff)
            # In 1 secs, ((start_node_mean_ru - final_node_mean_ru)/task_migration_time)) units of load can be reduced
            # In time_diff secs, how much load can be reduced
            delta_load = (time_diff * ((start_node_mean_ru - final_node_mean_ru)/task_migration_time))
            print("Sarolog:", task_migrations_count, task_migration_time, time_for_migration, time_diff, delta_load)
            if (delta_load):
                # Logic for backtracking the PSO particles
                print(src_node + ": Start threshold for migration should be: " + str(round(start_node_mean_ru - delta_load)) + " instead of " + str(start_node_mean_ru))
                start_node_mean_ru = round(start_node_mean_ru - delta_load)

    # If migrations are performed and if the migrations did not result in any SLA violations
    if (task_migrations_count):
        # For CLAPSOT, update the PSO stats, even if there is an SLA violation during migration
        # Remember, CLAPSOT shares negative feedback too to the particles

        if (re.fullmatch("clapsot", migration_algorithm) or ((final_sla_violations - initial_sla_violations) == 0)):
            print("PSO Stats updated. Task migrations: " + str(task_migrations_count) + " Violations: " + str((final_sla_violations - initial_sla_violations)))
        
            record_pso_stats(src_node, 'best_ru', start_node_mean_ru)
            record_pso_stats(src_node, 'mt_per_task', mt_per_task)
            if (start_node_mean_ru - final_node_mean_ru):
                record_pso_stats(src_node, 'ru_per_task', ru_per_task)
            print("PSO Stats Update: " + "best_ru " + str(start_node_mean_ru) + " mt_per_task " + str(mt_per_task) + " ru_per_task " + str(ru_per_task))
        else:
            print("PSO Stats NOT updated. Task migrations: " + str(task_migrations_count) + " Violations: " + str((final_sla_violations - initial_sla_violations)))


    return task_migrations_count

# End of migrate_tasks()

def get_task_type_from_ue(ue_node):
    task_type_match = "" 
    for task_type in tasklib.task_def.keys():
        attrs = G.nodes[ue_node].keys()
        attrs_match = 1
        for attr in node_resource_attrs:
            if (G.nodes[ue_node][attr] != tasklib.task_def[task_type][attr]):
                attrs_match = 0
                break
        if (attrs_match):
            task_type_match = task_type
            break;
    return task_type_match
# End of get_task_type_from_ue()

def get_sessions_count_from_ue(ue_node):
    return (G.nodes[ue_node]['sessions'])
# End of get_sessions_count_from_ue()

def get_migration_time_from_node(node):
    return (G.nodes[node]['migration_time'])

def set_mon_interval_for_node(node, mon_interval):
    G.nodes[node]['mon_interval'] = mon_interval

def get_mon_interval_from_node(node):
    return (G.nodes[node]['mon_interval'])

def plot_tasks(task_stats):
    from itertools import chain

    t_task_data = pd.DataFrame.from_dict(task_stats)
    task_data = t_task_data.transpose()
    fig, ax = plt.subplots()


    print(task_data)
    if (task_stats_counter <=0):
        print("No task migrations to plot")
        return

    task_data['actual_migration_time'] = round(((task_data['mig_end_timestamp'] *1000) - (task_data['mig_start_timestamp'] *1000))/1000, 2)
    task_data['time_for_migration'] = round(task_data['time_for_migration'], 2)
    print(task_data)

    task_stats_file = "graphs-data/task-stats-" + migration_algorithm + "-" + app_type + ".txt"
    task_data.to_csv(task_stats_file, sep=',')

    # Plot from 1st data point to the last (smoothen the values)
    task_data['mig_start_timestamp'] = round(task_data['mig_start_timestamp'] - task_data['mig_start_timestamp'][0])
    task_data['mig_start_timestamp'] /= 1000

#    plt.xlim(min(task_data['mig_start_timestamp']), max(task_data['mig_start_timestamp']))
#    plt.ylim(min(chain(task_data['time_for_migration'], task_data['actual_migration_time'])), max(chain(task_data['time_for_migration'], task_data['actual_migration_time'])))

    ymin = min(chain(task_data['time_for_migration'] , task_data['actual_migration_time']))
    ymax = max(chain(task_data['time_for_migration'] , task_data['actual_migration_time']))

    plt.xlim(min(task_data['mig_start_timestamp']), max(task_data['mig_start_timestamp'])+5)
    plt.ylim(0, ymax + 1000)

    bar_width = 0.35

    task_data['mig_start_timestamp'][0] = task_data['mig_start_timestamp'][0] + 5

    ax.bar(task_data['mig_start_timestamp'], task_data['time_for_migration'], bar_width, label = "Time allotted")
    ax.bar(task_data['mig_start_timestamp'] + bar_width, task_data['actual_migration_time'], bar_width, label = "Time taken")

#    plt.plot(task_data['mig_start_timestamp'], task_data['time_for_migration'], color='grey', marker='', linestyle='--', markersize='5', lw='2',label="Time allotted")
#    plt.plot(task_data['mig_start_timestamp'], task_data['actual_migration_time'], color='red',marker='', linestyle='--', markersize='5', lw='2', label="Time taken")

#    plt.vlines(task_data['mig_start_timestamp'], 0, task_data['sessions_migrated'], colors='grey', lw=2, label='Sessions Migrated')
#    plt.vlines(task_data['mig_start_timestamp'], 0, task_data['sla_miss'], colors='red', lw=2, label='SLA Miss')

#    plt.vlines(task_data['sessions_migrated'], 0, task_data['time_for_migration'], colors='grey', lw=2, label='Tasks Assigned')

    ax.set_xlabel("Time Instance ", fontsize=12)
#    ax.set_ylabel("Tasks Count", fontsize=12)
#    ax.set_ylabel("Sessions Migrated", fontsize=12)
    ax.set_ylabel("Migration Duration (Allotted vs. Actual)", fontsize=12)

    plt.title('Migration Interval')
    plt.legend(loc="upper left")

    plt.show()

#    plt.axhline(y = sla_threshold, color = 'r', label="SLA Limit", linestyle = 'dotted', lw=3)

# End of plot_tasks

def plot_graph_sessions(stats, node):

    # Import pandas
    import pandas as pd
 
    t_data = pd.DataFrame(stats)
    if t_data.empty:
        return

    data = t_data.transpose()

    nsamples= len(data.index)
    if (nsamples < 20):
        return
    d2 = data.tail(-10)
    data = d2

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

    APS = [None] * nsamples
    APS = (Q/M)+(M)

    # Convert infinity to zeros
    data['APS'] = APS

    x_axis = T/1000
    xmax_limit = 0

    #log_debug(data)
    data.to_csv("graphs-data/node-sim-stats.txt", sep=',')

    xmin = min(x_axis)
    xmax = max(x_axis)

    # Display graphs where there are enough instances of data samples
    if (xmax < xmax_limit):
        return

#    fig, ax = plt.subplots(dpi=300)
    fig, ax = plt.subplots()

    plt.xlim(xmin, xmax)
    plt.ylim(0, 100)

    plt.vlines(x_axis, 0, M, colors='orange', lw=2, label='App. RU')

    ax.set_xlabel("Time Instance " + node, fontsize=12)
    ax.set_ylabel("Performance Scale", fontsize=12)
    plt.axhline(y = sla_threshold, color = 'r', label="SLA Limit", linestyle = 'dotted', lw=3)

    # Plotting Application QoS/QoE
    ax.plot(x_axis, Q, color='darkblue', marker='+', linestyle='-', markersize="7", lw='2', label="App. QoS")

    # Plotting APS
    ax.plot(x_axis, APS, color='grey', marker='', linestyle='--', markersize='5', lw='2', label="APS")

    # Shrink the graph vertically to create space for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.9])
    
    ax2 = ax.twinx()
    ax2.set_ylabel("No. of user sessions", fontsize=12)
    ax2.plot(x_axis, S, color='blue', marker='', linestyle='-.', label="User Sessions")
    ax2.set_ylim(min(S), max(S))
    ax2.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.9])

    lhandle1, label1= ax.get_legend_handles_labels()
    lhandle2, label2= ax2.get_legend_handles_labels()

    ax.legend([lhandle1[0]] + [lhandle1[1]] + [lhandle1[2]] + [lhandle1[3]] + lhandle2, [label1[0]] + [label1[1]] + [label1[2]] + [label1[3]] + label2, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3, fontsize=12)

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # R is the rate of change of Resource Utilisation

    p_data = pd.DataFrame(M)
    R = p_data.pct_change(fill_method ='ffill').to_numpy()*100
    data['R'] = R
    ymin = min(R)
    ymax = max(R)

    plt.show()
#    plt.close()
    return

# End of plot_graph_sessions()


def plot_graph_sessions_migration_time(stats, node):

    # Import pandas
    import pandas as pd
 
    t_data = pd.DataFrame(stats)
    if t_data.empty:
        return

    data = t_data.transpose()

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
    Mt = data['migration_time']

    APS = [None] * nsamples
    APS = (Q/M)+(M)

    # Convert infinity to zeros
    data['APS'] = APS

    xmax_limit = 0

    #log_debug(data)
    data.to_csv("graphs-data/node-sim-stats.txt", sep=',')

#    xmin = min(x_axis)
#    xmax = max(x_axis)

    # Display graphs where there are enough instances of data samples
#    if (xmax < xmax_limit):
#        return

#    fig, ax = plt.subplots(dpi=300)
    fig, ax = plt.subplots()

#    plt.xlim(xmin, xmax)
    plt.ylim(0, 100)

#    plt.vlines(x_axis, 0, M, colors='orange', lw=2, label='App. RU')

    ax.set_xlabel("Sessions" + node, fontsize=12)
    ax.set_ylabel("Migration Time", fontsize=12)

    # Plotting Application Migration Time
    ax.plot(S, Mt, color='darkblue', marker='+', linestyle='-', markersize="7", lw='2', label="App. QoS")

    # Shrink the graph vertically to create space for the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.9])
    

#    ax2 = ax.twinx()
#    ax2.set_ylabel("No. of user sessions", fontsize=12)
#    ax2.plot(x_axis, S, color='blue', marker='', linestyle='-.', label="User Sessions")
#    ax2.set_ylim(min(S), max(S))
#    ax2.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.9])

#    lhandle1, label1= ax.get_legend_handles_labels()
#    lhandle2, label2= ax2.get_legend_handles_labels()

#    ax.legend([lhandle1[0]] + [lhandle1[1]] + [lhandle1[2]] + [lhandle1[3]] + lhandle2, [label1[0]] + [label1[1]] + [label1[2]] + [label1[3]] + label2, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=3, fontsize=12)

#    ax.tick_params(axis='x', labelsize=12)
#    ax.tick_params(axis='y', labelsize=12)

#    # R is the rate of change of Resource Utilisation
#
#    p_data = pd.DataFrame(M)
#    R = p_data.pct_change(fill_method ='ffill').to_numpy()*100
#    data['R'] = R
#    ymin = min(R)
#    ymax = max(R)

    plt.show()
#    plt.close()
    return

# End of plot_graph_sessions()
def plot_load_graph_for_vm(vm_node):
    global sim_stats

    # Import pandas
    import pandas as pd
    import numpy as np
 
    plot_field = 'mean_ru'
    data = {}

    t_data = pd.DataFrame(sim_stats[vm_node])
    if t_data.empty:
        return
    data[vm_node] = t_data.transpose()
    total_samples = len(data[vm_node].index)
    print("Total samples: " + str(total_samples))

    app_id = 0
    for app_node in get_apps_from_vm(vm_node):
        print("Printing data for " + app_node)
        t_data = pd.DataFrame(sim_stats[app_node])
        print_sim_stats(app_node)
        if t_data.empty:
            return
        data[app_id] = t_data.transpose()
        app_id += 1
    # End of for loop
    
    app_id = 0
    nsamples = {}

    for app_id in range(max_apps_per_vm):
        for timestamp in data[vm_node]['timestamp']:
            append_data = pd.DataFrame(0, index=np.arange(1), columns=data[app_id].columns)
            ts_index = (data[app_id][data[app_id]['timestamp'] == timestamp].index)
            if (len(ts_index) == 0):
                append_data['timestamp'] = timestamp
                data[app_id] = data[app_id].append(append_data)
        data[app_id] = data[app_id].sort_values(by = 'timestamp')

    for app_id in range(max_apps_per_vm):
        last_row_data = data[app_id].iloc[0]
        for ite in range(0, total_samples - 1):
            if (data[app_id].iloc[ite]['ite'] == 0):
                timestamp = data[app_id].iloc[ite]['timestamp']
                # for empty rows, copy the previous valid load data sample
                data[app_id].iloc[ite] = last_row_data
                data[app_id].iloc[ite]['timestamp'] = timestamp
            last_row_data = data[app_id].iloc[ite]

    for app_id in range(max_apps_per_vm):
        APS = [None] * total_samples
        APS = (data[app_id]['qoe']/data[app_id]['mean_ru'])+(data[app_id]['mean_ru'])

        # Convert infinity to zeros
        data[app_id]['APS'] = (APS)/max_apps_per_vm
        data[app_id][plot_field] = data[app_id][plot_field]/max_apps_per_vm
        data[app_id]['APS'].replace([np.inf, -np.inf], 0, inplace=True)
#        # First row has a high APS value. Hence dropping first row
#        data[app_id].drop(index=data[app_id].index[0], axis=0, inplace=True)

    #End of for loop

#    for ite in range(0, total_samples - 1):
#        for app_id in range(max_apps_per_vm):
#            print (str(data[app_id].iloc[ite]['mean_ru']) + " ", end =" ")
#        print (str(data[vm_node].iloc[ite]['mean_ru']))

    for app_id in range(max_apps_per_vm):
        filename="app" + str(app_id) + ".csv"
        data[app_id]['mean_ru'].to_csv(filename)

    data[vm_node]['mean_ru'].to_csv('vm.csv')

    # I is the iteration / C is the CPU utilisation / D is disk utilisation / N is network utilisation
    # S is the # of user sessions /  M is the Mean column / Q is the QoE column
    #I = data[app_node]['ite']
    #T = data[app_node]['timestamp']
    #C = data[app_node]['cpu']
    #D = data[app_node]['disk']
    #N = data[app_node]['nw']
    #M = data[app_node]['mean_ru']
    #Q = data[app_node]['qoe']
    #S = data[app_node]['sessions'] - 80

    xmax_limit = 0
    # SARO:Need to check if this would be sufficient
    x_axis = data[vm_node]['timestamp']
    xmin = min(x_axis)
    xmax = max(x_axis)
    width = 1

    # Display graphs where there are enough instances of data samples
    if (xmax < xmax_limit):
        return

    ylimit = 100

    fig = plt.subplots()

    plt.xlim(xmin, xmax)
    plt.ylim(0, ylimit)

    ind = data[vm_node]['timestamp']

    p1 = plt.bar(ind, data[0][plot_field], width)
#    p2 = plt.bar(ind, data[1][plot_field], width, bottom = data[0][plot_field])
#    p3 = plt.bar(ind, data[2][plot_field], width, bottom = np.array(data[0][plot_field]) + np.array(data[1][plot_field]))
#    p4 = plt.bar(ind, data[3][plot_field], width, bottom = np.array(data[0][plot_field]) + np.array(data[1][plot_field]) + np.array(data[2][plot_field]))
#    p5 = plt.bar(ind, data[4][plot_field], width, bottom = np.array(data[0][plot_field]) + np.array(data[1][plot_field]) + np.array(data[2][plot_field]) + np.array(data[3][plot_field]))

    plt.ylabel(plot_field)
    plt.xlabel('Timestamp')
    plt.title('Load of VM ' + vm_node)
#    plt.legend((p1[0]), ('app1'))
#    plt.legend((p1[0], p2[0], p3[0]), ('app1', 'app2', 'app3'))
#    plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0]), ('app1', 'app2', 'app3', 'app4', 'app5'))

    plt.axhline(y = sla_threshold, color = 'r', label="SLA Limit", linestyle = 'dotted', lw=3)
    plt.plot(ind, data[vm_node][plot_field], color='red', marker='', linestyle='--', markersize='5', lw='2', label=plot_field)

    plt.show()
    plt.close()
    return

# End of plot_load_graph_for_vm()

def print_node_stats(node):
    print("Node: " + node + " Total stats entries: "  + str(sim_stats_counter[node]))
    for sample_id in range(0,sim_stats_counter[node]):
            print(node + " : " + str(sample_id) + " : " + str(sim_stats[node][sample_id]))

def print_node_load():
    edges = get_edges_from_cloud(cloud_node_str)
    for edge in edges:
        hosts = get_hosts_from_edge(edge)
        for host_node in hosts:
            vms = get_vms_from_host(host_node)
            for vm_node in vms:
                apps = get_apps_from_vm(vm_node)
                for app_node in apps:
                    print("Node: " + app_node + " " + str(calc_node_load(app_node)))
                print("Node: " + vm_node + " " + str(calc_node_load(vm_node)))
            print("Node: " + host_node + " " + str(calc_node_load(host_node)))
        print("Node: " + edge + " " + str(calc_node_load(edge)))

###################################################
### Main
###################################################

migration_algorithm="static"
app_type="video"
migration_algorithm = "regpso"
migration_algorithm = "clapso"

tasks_file = "tasks.txt"
if (len(sys.argv) > 2):
   app_type=sys.argv[1]
   migration_algorithm=sys.argv[2]
tasks_file = "tasks-" + app_type + ".txt"

###################################################
### Building the topology and initialising values
###################################################

initialise_tasks()

# Start a fresh run
G = nx.Graph() # Nodes Graph - contains nodes / edges / attributes for nodes
initialise_topology()
initialise_sla_stats()
initialise_stats()
initialise_pso_stats()
begin_task = 0

#print_topology(G)

time_for_migration = 0

tasks_assigned_count = 0
for taskid in range(begin_task, tasks_count):
    app_node = get_random_app(G)
    assign_task_to_app(app_node, taskid)

#    print_topology(G)
#    print_node_load()

    tasks_assigned_count = tasks_assigned_count + 1

    # Increment the clock for each task
    increment_sim_time()

    if (taskid % 10 == 0):
        print("Taskid: ", str(taskid), "Time is: ", str(get_sim_time()))

        t_nodes = list(filter(lambda x: type_of_node in x, G.nodes))

        for t_node in t_nodes:
            if (is_node_load_low(t_node)):
                continue
            #print_node_stats(t_node)

            node_mean_ru = calc_node_load(t_node)

            print("Node: " + str(t_node) + " Load: " + str(node_mean_ru))

            node_perf_trend = NO_TREND_SEEN

            if (re.search("pso", migration_algorithm)):
                # Use LR for all the PSO variants
                node_perf_trend = check_node_perf_trend_LR(t_node)
                (ru_migration_threshold, ru_entity) = get_ru_threshold_for_migration(t_node)
            elif (re.fullmatch("ewma", migration_algorithm)):
                # Use Exponential Weighted Moving Average based migrations for EWMA
                node_perf_trend = check_node_perf_trend_EWMA3(t_node)
                (ru_migration_threshold, ru_entity) = (ru_default_threshold, "default")
            elif (re.fullmatch("static", migration_algorithm)):
                check_node_perf_trend_STATIC(t_node)
                if (calc_node_load(t_node) > ru_max_threshold):
                    node_perf_trend = TREND_SEEN
                (ru_migration_threshold, ru_entity) = (ru_default_threshold, "default")

            # End of logic to check performance trend


            print("RU Migration Threshold: " + str(ru_migration_threshold))
            if ((node_perf_trend == TREND_SEEN) and (node_mean_ru > ru_migration_threshold)):
                # This portion keeps a record of migration times
                # for benchmarking the different algorithms

                last_sample_timestamp = sim_stats[t_node][sim_stats_counter[t_node] - 1]['timestamp']

                print("Increasing trend seen in node" + " " + t_node + " Load is " + str(node_mean_ru))

                print("Predicting future values " + "Last sample timestamp is " + str(last_sample_timestamp))

                import numpy as np

                forecast = forecast_sarimax('mean_ru')
                forecast_mean_ru = np.array(forecast)

                sla_violation_timestamp_index = np.nonzero(forecast_mean_ru > aps_threshold)

                early_alerts = 0
                if ((len(sla_violation_timestamp_index) == 0) or (len(sla_violation_timestamp_index[0]) == 0)):
                    # An early indicator of increasing trend
                    early_alerts += 1
                    continue

                # There is really a SLA violation possibility.
                # Now, predict "when" it is going to happen

                forecast = forecast_sarimax('timestamp')
                forecast_timestamp = np.array(forecast)

                forecast_values = np.transpose([forecast_timestamp, forecast_mean_ru])
                print("Forecast values - ts, mru " + str(forecast_values))

                print("SLA violation timstamp index - " + str(sla_violation_timestamp_index))
                sla_violation_timestamp = (forecast_timestamp[sla_violation_timestamp_index[0][0]])

                time_for_migration = sla_violation_timestamp - last_sample_timestamp
                if (time_for_migration < 0):
                    print("Abs function had to be invoked to fix negative timestamp")
                    time_for_migration = abs(sla_violation_timestamp - last_sample_timestamp)

                print("---- Curr Timestamp: " + str(last_sample_timestamp) + " SLA violation will happen at timestamp : " + str(sla_violation_timestamp) + " Diff: " + str(sla_violation_timestamp - last_sample_timestamp))

            # End of IF condition for migration time calculations

            # Migrations are performed in this section using migrate_tasks()
            # The mechanisms to trigger migrations, vary from algorithm to algorithm

            if ((node_perf_trend == TREND_SEEN) and (node_mean_ru > ru_migration_threshold)):
                migrations_count = migrate_tasks(t_node, type_of_node, time_for_migration)
                if (migrations_count):
                    # Display only the nodes where some migrations happened
                    display_nodes.append(t_node)
                if (re.fullmatch("ewma", migration_algorithm) or re.fullmatch("static", migration_algorithm)):
                    record_stats(0, t_node)
                total_task_migrations += migrations_count
            # End of logic for migrating tasks()

        #write_sim_stats_to_file()
# End of for loop

if (re.search(type_of_node, app_node_str)):
    v_nodes = list(filter(lambda x: app_node_str in x, G.nodes))
elif (re.search(type_of_node, vm_node_str)):
    v_nodes = list(filter(lambda x: vm_node_str in x, G.nodes))

for v_node in v_nodes:
    calc_qoe_violations(sim_stats[v_node])
#    print("Printing sim_stats for " + v_node)
#    plot_graph_sessions(sim_stats[v_node], v_node)
#    plot_graph_sessions_migration_time(sim_stats[v_node], v_node)
#    print_sim_stats(v_node)
#    plot_graph_sessions(sim_stats[v_node], v_node)

#plot_tasks(task_stats)

# remove duplicates from the list
uniq_nodes = []
[uniq_nodes.append(x) for x in display_nodes if x not in uniq_nodes]
uniq_nodes.sort()

print("Graphs will be generated for " + str(len(uniq_nodes)) + " / " + str(len(v_nodes)) + " nodes." + str(uniq_nodes))
for u_node in uniq_nodes:
    t = 0
#    plot_graph_sessions(sim_stats[u_node], u_node)
#    plot_load_graph_for_vm(u_node)
#print_topology(G)
print_topology_simple(G)

if re.search(type_of_node,vm_node_str):
    total_node_instances = total_vm_instances
elif re.search(type_of_node,vm_node_str):
    total_node_instances = total_app_instances

print("Algorithm: " + migration_algorithm + " App Type: " + app_type)
print("Total Nodes that had task migrations: " + str(total_node_migrations))
print(" Total Node instances violating SLA : " + str(total_node_instances_with_violations) + " / " + str(total_node_instances) + " Total QoE Violations: " + str(total_qoe_violations) + " Total task migrations: " + str(total_task_migrations))
print(" SLA Violations " + " Node: " + str(total_node_instances_with_violations) + " APP: " + str(sla_stats[app_node_str]['total']), end = " ")
print(" | Migration time: " + str(total_mtime_violations))
#print(" Total Load Reduced / Actual Node Migrations: " + str(total_load_reduced) + " / " + str(total_node_migrations))

topo_stats_file = "graphs-data/topo-stats-" + migration_algorithm + "-" + app_type + ".txt"
nx.write_gml(G, topo_stats_file) 
