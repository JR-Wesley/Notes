---
dateCreated: 2025-04-15
dateModified: 2025-04-15
---

Synthesis Lectures on Computer Architecture

# Abstract & Preface

This Synthesis Lecture focuses on techniques for efficient data orchestration within DNN accelerators.

It is well known that the cost of data movement today surpasses the cost of the actual computation; therefore, DNN accelerators require careful orchestration of data across on-chip compute, network, and memory elements to minimize the number of accesses to external DRAM. The book covers **DNN dataflows, data reuse, buffer hierarchies, networks-on-chip, and automated design-space exploration**. It concludes with data orchestration challenges with compressed and sparse DNNs and future trends.

The goal of this Synthesis Lecture is to dissect and describe the key building blocks and design flows common across DL inference accelerators. In particular, we focus on **efficient mechanisms to manage data orchestration—i.e., systematically staging fine-grained data movement within an accelerator** for performance and energy efficiency.
