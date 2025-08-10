---
dateCreated: 2025-08-09
dateModified: 2025-08-09
---

This document provides a high-level overview of NCCL's system architecture, showing how kernel planning, transport management, and topology optimization work together to orchestrate GPU communication operations. It covers the core flow from user API calls through task preparation to kernel execution.

For specific transport implementations, see [Transport Layer](https://deepwiki.com/NVIDIA/nccl/3-transport-layer). For detailed topology management, see [Topology Management](https://deepwiki.com/NVIDIA/nccl/4-topology-management). For device-side execution details, see [Device-Side Operations](https://deepwiki.com/NVIDIA/nccl/5-device-side-operations).

## System Overview

NCCL's core architecture revolves around transforming user-submitted collective and point-to-point operations into optimized GPU kernel execution plans. The system consists of several interconnected subsystems that handle task preparation, transport selection, topology optimization, and execution orchestration.

![](assets/Core%20Architecture.assets/System%20Overview.png)

**Core Architecture Flow** Sources: [](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L331-L519)[src/enqueue.cc331-519](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L331-L519)[](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L529-L784)

[src/enqueue.cc529-784](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L529-L784)

## Task Preparation System

The task preparation system transforms user operations into internal task representations and determines optimal algorithms and protocols.

![](assets/Core%20Architecture.assets/Task%20Preparation%20System.png)

### Task Collection and Sorting

**Task Collection and Algorithm Selection** Sources: [](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L335-L385)[src/enqueue.cc335-385](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L335-L385)[](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L388-L449)

[src/enqueue.cc388-449](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L388-L449)

The `ncclPrepareTasks` function processes tasks in size-descending order from `ncclTaskCollSorter`, then bins them by `(func, op, type)` tuples. For each bin, it aggregates operations within 4X size of each other and computes optimal algorithms using `getAlgoInfo`. Tasks are then categorized by scheduling constraints (collnet vs nvls) for proper channel assignment.

### Buffer Registration and Work Creation

The system registers buffers and creates device work structures for efficient GPU execution:

![](assets/Core%20Architecture.assets/Buffer%20Registration%20and%20Work%20Creation.png)

**Buffer Registration and Work Creation** Sources: [](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L267-L327)[src/enqueue.cc267-327](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L267-L327)[](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L455-L517)

[src/enqueue.cc455-517](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L455-L517)

## Kernel Planning System

The kernel planning system organizes tasks into executable batches and manages GPU kernel arguments.

### Work Batch Organization

![](assets/Core%20Architecture.assets/Work%20Batch%20Organization.png)

**Work Batch Management** Sources:[](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L98-L163)

[src/enqueue.cc98-163](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L98-L163)

The `addWorkBatchToPlan` function manages work batch creation with strict constraints. New batches are created when work types or function IDs differ, or when size limits are exceeded. Extension batches handle non-contiguous work offsets, using a 63-item offset limit and bitset encoding for efficient device-side processing.

### Plan Finalization

![](assets/Core%20Architecture.assets/Plan%20Finalization.png)

**Plan Finalization Process** Sources:[](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L165-L238)

[src/enqueue.cc165-238](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L165-L238)

The `finishPlan` function finalizes kernel execution plans by organizing batches in round-robin order across channels and merge-sorting proxy operations by `opCount`. It determines optimal memory storage (in kernel args vs external) and sets up device-side navigation structures with `nextJump` pointers for batch traversal.

## Scheduling Systems

NCCL implements separate scheduling systems for collective and point-to-point operations, each optimized for different communication patterns.

### Collective Operation Scheduling

![](assets/Core%20Architecture.assets/Collective%20Operation%20Scheduling.png)

**Collective Scheduling Logic** Sources: [](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L529-L784)[src/enqueue.cc529-784](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L529-L784)[](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L608-L738)

[src/enqueue.cc608-738](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L608-L738)

The `scheduleCollTasksToPlan` function implements sophisticated traffic distribution for collective operations. For CollNet operations, it uses `calcCollChunking` to determine optimal chunk sizes. For standard operations, it employs cell-based partitioning with minimum traffic thresholds (16KB) and distributes work across channels using count-based distribution (CBD) with `countLo`, `countMid`, and `countHi` values.

### Point-to-Point Operation Scheduling

![](assets/Core%20Architecture.assets/Point-to-Point%20Operation%20Scheduling.png)

**Point-to-Point Scheduling** Sources: [](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L793-L1016)[src/enqueue.cc793-1016](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L793-L1016)[](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L1018-L1064)

[src/enqueue.cc1018-1064](https://github.com/NVIDIA/nccl/blob/7c12c627/src/enqueue.cc#L1018-L1064)

The `addP2pToPlan` function handles P2P operation scheduling with protocol selection based on size thresholds and transport capabilities. It optimizes chunk sizes for network operations, manages buffer registration for both network and IPC transports, and distributes work across multiple channels using `ncclP2pPartBounds` for optimal bandwidth utilization.

## Transport Integration

The core architecture integrates with multiple transport layers through a unified interface system.

### Transport Selection and Configuration

![](assets/Core%20Architecture.assets/Transport%20Selection%20and%20Configuration.png)

**Transport System Integration** Sources: [](https://github.com/NVIDIA/nccl/blob/7c12c627/src/transport/net.cc#L154-L268)[src/transport/net.cc154-268](https://github.com/NVIDIA/nccl/blob/7c12c627/src/transport/net.cc#L154-L268)[](https://github.com/NVIDIA/nccl/blob/7c12c627/src/transport/coll_net.cc#L141-L209)

[src/transport/coll_net.cc141-209](https://github.com/NVIDIA/nccl/blob/7c12c627/src/transport/coll_net.cc#L141-L209)

## Memory and Resource Management

The architecture implements sophisticated memory management for optimal GPU communication performance.

### Buffer Management Strategy

![](assets/Core%20Architecture.assets/Buffer%20Management%20Strategy.png)

**Memory Management Architecture** Sources: [](https://github.com/NVIDIA/nccl/blob/7c12c627/src/transport/net.cc#L63-L91)[src/transport/net.cc63-91](https://github.com/NVIDIA/nccl/blob/7c12c627/src/transport/net.cc#L63-L91)[](https://github.com/NVIDIA/nccl/blob/7c12c627/src/transport/net.cc#L270-L306)

[src/transport/net.cc270-306](https://github.com/NVIDIA/nccl/blob/7c12c627/src/transport/net.cc#L270-L306)

The system uses a sophisticated memory mapping strategy with `connectMap` structures that bank different memory types using offset encoding. The upper 3 bits of offsets indicate memory bank types, enabling efficient pointer resolution across different memory spaces while supporting both same-process and cross-process scenarios.

This architecture provides the foundation for NCCL's high-performance GPU communication by efficiently orchestrating task preparation, transport selection, and execution planning while maintaining optimal memory utilization patterns.
