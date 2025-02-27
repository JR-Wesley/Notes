RTL Design Using Verilog

主要内容：ASIC design concepts, semi-custom ASIC design flow and the case studies
有一个总体的介绍，不过很笼统。没有具体细节，也没有EDA的使用。

Chapter 1: Introduction
Chapter 2: ASIC Design Flow
Chapter 3: Let Us Build Design Foundation:  role of the design elements in the ASIC design
Chapter 4: Sequential Design Concepts
Chapter 5: Important Design Considerations:  timing, skew, latency, and other design considerations such as parallelism and concurrency
Chapter 6: Important Considerations for ASIC Designs
Chapter 7: Multiple Clock Domain Designs
Chapter 8: Low Power Design Considerations
Chapter 9: Architecture and Micro-architecture Design
Chapter 10: Design Constraints and SDC Commands
Chapter 11: Design Synthesis and Optimization Using RTL Tweaks
Chapter 12: Synthesis and Optimization Techniques: 
Chapter 13: Design Optimization and Scenarios:
Chapter 14: Design for Testability: 
Chapter 15: Timing Analysis: 
Chapter 16: Physical Design: 
Chapter 17: Case Study: Processor ASIC Implementation: 
Chapter 18: Programmable ASIC: 
Chapter 19: Prototyping Design
Chapter 20: Case Study: IP Design and Development


# ch2
 the ASICs can be of type full-custom, semi-custom, gate array-based ASICs
## 2.1 ASIC Design Flow
semi-custom ASIC: the standard cells and macros which are prevalidated is used
1. Market Survey and Specification Extraction
2. Design Planning
3. Logic Design
(a) RTL design
(b) RTL verification
(c) Synthesis
(d) DFT and scan insertion
(e) Equivalence checking
(f) Pre-layout STA
4. Physical Design and GDSII(gate-level -> GDSII)
(a) Floor planning
(b) Power planning
(c) CTS
(d) Place and route
(e) LVS
(f) DRC
(g) Signoff STA
(h) GDSII
5. chip



# ch16 Physical Design
common issues:
1. Congestion
2. Routing issues and routing delays
3. Issues during the distribution of the clock and clock skew
4. Issues due to the net delays and parasitic
5. Meeting of the chip-level constraints such as timing and maximum frequency
6. Issues due to noise and derate of the timing
7. Design rule check fails
8. LVS issues due to the routing.


## 16.1 Physical Design Flow
input -  netlist with chip constraints and required libraries 
1 start with the floor planning: planning of the design mapping so that there should not be congestion while routing of the design and the logic blocks or functional blocks should meet the aspect ratio
2 the power planning is to plan for the power rings(VDD & VSS) and power straps depending on the power requirements
3 clock tree synthesis to balance the clock skew and to distribute the clock to functional blocks
4 PnR to have the layout of the chip which needs to be checked to verify the 
(a) Foundry rules that is DRC
(b) LVS that is checking of the layout versus the schematic, and the intent is to verify the layout with the gate-level netlist.
5 signoff STA
6 GDSII(the Generic or Geometric Data Structure Information Interchange) generation

## 16.2 Foundation and Important Terms
Initially, the area estimation for the chip is unknown and with initial floorplan we can estimate the rough area utilization
(a) Chip-Level Utilization:
(Area (Standard Cells) + Area (Macros) + Area (Pad Cells))/Area (chip)
(b) Floorplan Utilization: 
((Area (Standard Cells) + Area (Macros) + Area (Pad Cells))/(Area (Chip)) − Area (sub floorplan))
(c) Cell Row Utilization: 
Area (Standard Cells)/(Area (Chip) − Area (Macro) − Area (Region Blockages))

## 16.3 Floor Planning and Power Planning
netlist gives information about: Various design and functional blocks/ Macros/ Memories/ Interconnection between these block
- What should be the strategies for the best floor plan?
objective
1. Use of the minimum area
2. Strategy to have floor plan so the congestion can be very minimum
3. The delays due to routing can be minimized due to better floor plan.
 important tasks
1. The chip area and size estimation
2. Strategy to arrange various functional blocks on the silicon
3. Strategy for the pin assignment
4. Planning for the IOs.
- Floorplan needs use of the important elements!
Standard cells/ IO cells/ Macros

## 16.4 Power Planning
Power Rings: Carries VDD and VSS around the chip
Power Stripes: Carries VDD and VSS from rings across the chip
Power Rails: Connect VDD and VSS to the standard cell VDD and VSS
![[Power Planning.png]]

## 16.5 Clock Tree Synthesis



# ADVANCED ASIC CHIP SYNTHESIS
Using Synopsys® Design CompilerTM Physical CompilerTM and PrimeTime®
详细介绍ASIC芯片设计的流程，具体的操作。

CHAPTER 1: ASIC DESIGN METHODOLOGY
CHAPTER 2: TUTORIAL
CHAPTER 3: BASIC CONCEPTS
CHAPTER 4: SYNOPSYS TECHNOLOGY LIBRARY
CHAPTER 5: PARTITIONING AND CODING STYLES
CHAPTER 6: CONSTRAINING DESIGNS
CHAPTER 7: OPTIMIZING DESIGNS
CHAPTER 8: DESIGN FOR TEST
CHAPTER 9: LINKS TO LAYOUT & POST LAYOUT OPT.
CHAPTER 10: PHYSICAL SYNTHESIS
CHAPTER 11: SDF GENERATION
CHAPTER 12: PRIMETIME BASICS
CHAPTER 13: STATIC TIMING ANALYSIS



