# tclapso
Service Migration Time Aware Closed Loop Adaptive Particle Swarm Optimisation Algorithm

The following are the new scripts developed to simulate the edge clusters, generate traffic and simulate load conditions.

**taskgen.py** - Generates tasks emulating Video, IoT and FTP Traffic

**simengine.py** - Creates the topology of edge clusters, configures the clusters and generates traffic. Data generated by the simulation engine is stored locally                     in log files

**plot-mig-load.py** - Used to plot the load before and after the service migration

**plot-mig-time.py** - Used to plot the migration time available vs. taken

**plot-ru-graph.py **- Used to plot the resource utilization graphs

**plot-sess-mt-from-node-sim-stats.py**  - Used to plot the sessions migrated within the migration interval

**plot-sess-vs-load-graph.py** - Used to plot the number of sessions vs. the load

**plot_aps_graph.py** - Used to plot the Application Performance Score for the application intance
