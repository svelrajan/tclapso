''' tasklib.py '''

task_def = {}
task_def['cputask-1']= {'cpu':2, 'disk':0, 'mem':0, 'nw':0, 'sessions':1}
task_def['cputask-2']= {'cpu':4, 'disk':0, 'mem':0, 'nw':0, 'sessions':1}
task_def['cputask-3']= {'cpu':6, 'disk':0, 'mem':0, 'nw':0, 'sessions':1}
#task_def['cputask-4']= {'cpu':8, 'disk':0, 'mem':0, 'nw':0, 'sessions':1}

task_def['disktask-1']= {'cpu':0, 'disk':2, 'mem':0, 'nw':0, 'sessions':1}
task_def['disktask-2']= {'cpu':0, 'disk':4, 'mem':0, 'nw':0, 'sessions':1}
task_def['disktask-3']= {'cpu':0, 'disk':6, 'mem':0, 'nw':0, 'sessions':1}
#task_def['disktask-4']= {'cpu':0, 'disk':8, 'mem':0, 'nw':0, 'sessions':1}

task_def['memtask-1']= {'cpu':0, 'disk':0, 'mem':2, 'nw':0, 'sessions':1}
task_def['memtask-2']= {'cpu':0, 'disk':0, 'mem':4, 'nw':0, 'sessions':1}
task_def['memtask-3']= {'cpu':0, 'disk':0, 'mem':6, 'nw':0, 'sessions':1}
#task_def['memtask-4']= {'cpu':0, 'disk':0, 'mem':8, 'nw':0, 'sessions':1}

task_def['nwtask-1']= {'cpu':0, 'disk':0, 'mem':0, 'nw':2, 'sessions':1}
task_def['nwtask-2']= {'cpu':0, 'disk':0, 'mem':0, 'nw':4, 'sessions':1}
task_def['nwtask-3']= {'cpu':0, 'disk':0, 'mem':0, 'nw':6, 'sessions':1}
#task_def['nwtask-4']= {'cpu':0, 'disk':0, 'mem':0, 'nw':8, 'sessions':1}

task_def['sesstask-1']= {'cpu':1, 'disk':1, 'mem':1, 'nw':1, 'sessions':2}
task_def['sesstask-2']= {'cpu':1, 'disk':1, 'mem':1, 'nw':1, 'sessions':4}
task_def['sesstask-3']= {'cpu':1, 'disk':1, 'mem':1, 'nw':1, 'sessions':6}
#task_def['sesstask-4']= {'cpu':1, 'disk':1, 'mem':1, 'nw':1, 'sessions':8}

# 30% assign and 20% unassign
task_ops = ["assign", "unassign", "assign", "assign", "unassign"]
task_ops = ["assign", "assign", "assign", "assign", "assign"]
