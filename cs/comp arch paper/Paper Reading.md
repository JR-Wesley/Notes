# Efficient Orchestrated AI Workflows Execution on Scale-out Spatial Architecture

Introduction
AI models and general processing tasks are defined as Orchestrated AI workflows, such as recognition, recommendation.
This research present
- the Orchestrated Workflow Graph (OWG) that systematically describes the inherent logical connections and embedded characteristics of Orchestrated AI Workflows.
- the concept of Task Blocks (TB) to encapsulate AI and general computational tasks
- Control Blocks (CB) are used to delineate logical decision-making processes
Orchestrated AI Workflows exhibit what we call Dual Dynamicity:
 1) certain TBs exhibit dynamic execution times at a fine granularity, and 
 2) multiple TBs display dynamic execution frequencies at a coarse granularity
Dual Dynamicity presents significant challenges to exist-
ing heterogeneous architectures during the execution of Orchestrated AI Workflows.
the static nature of the compilation process: Once compiled, TBs cannot dynamically adjust to changing hardware resource demands during execution, which can lead to resource idleness. Spatial architectures are promising for addressing these challenges due to their flexible computational features.
Existing spatial architectures  fundamentally fail to perform efficient dynamic scheduling when managing the aforementioned Dual Dynamicity.
 (1) Indiscriminate Resource Allocation arises because the current
spatial architecture allocates schedulable tasks to two distinct
exploitable PEA resources without differentiation; (2) Reactive
Load Rebalancing involves awaiting threshold-triggered work-
load imbalances before responding, resulting in prolonged data
transfers and heightened idleness; and (3) Contagious PEA
Idleness refers to idle states spreading among interconnected
PEAs due to a lack of timely rescheduling. What makes it
worse is that Orchestrated AI workflows tend more often to de-
ploy on large-scale spatial architecture, such as chiplet system
or even wafer-scale system [34]–[38], the above mentioned
challenges will exacerbate.