

# 1 new code com algorithm & decompressor in FPGA
![[Pasted image 20240122145815.png]]

# 2 deflate com&decom
![[Pasted image 20240122145936.png]]



# LZ77 
![[Pasted image 20240122150055.png]]
## abstract
- a "refine and recycle" method applicable to LZ77-type decompressors & implementation in reconfigurable logic
- refine the write commands(for literal tokens) & read commands(for copy tokens) a set of commands that target a single bank of block ram, and rather than performing all the dependency calculations saves logic by recycling (read) commands that return with an invalid result.
- A single “Snappy” decompressor implemented in reconfigurable logic leveraging this method is capable of processing multiple literal or copy tokens per cycle and achieves up to 7.2GB/s, which can keep pace with an NVMe device.
- 



## 6 results
### 6.1 setup
- target
the Xilinx Virtex Ultrascale VU3P- 2 device on an AlphaData ADM-PCIE-9V3 board and integrated with the POWER9 CAPI 2.0 [27] interface
- compared
an optimized software Snappy decompression implementation [1] compiled by gcc 7.3.0 with “O3” option and running on a POWER9 CPU in little endian mode with Ubuntu 18.04.1 LTS.
- benchmarks
1-3 “lineitem” table of the TPC-H benchmarks
Wiki [22] is an XML file dump from Wikipedia
Matrix[2] is a sparse matrix from the Matrix Market
high compression ratio file (Geo) that stores geographic information
![[Pasted image 20240122151209.png]]
### 6.2 integration
decompressor communicates with host memory through the CAPI 2.0 interface

### 6.3 resource utilization

![[Pasted image 20240122152015.png]]
### 6.4 end-to-end throughput performance

![[Pasted image 20240122152147.png]]

### 6.5 impacts of # of BCPs


### 6.6 comparison of decompression accelerators

By using 6 BCPs, output up to 31B per cycle at a clock frequency of 250MHz

around 14.5x and 3.7x faster then the prior work on ZLIB[18] and Snappy
more area-efficient, measured in MB/s per 1K LUTs and MB/s per BRAM (36kb), which is 1.4x more LUT efficient than the ZLIB implementation in [18] and 2.4x more BRAM efficient than the Snappy implementation in [25].
the Vitis Data Compression Library (VDCL) 25x and 4x faster than a single engine implementation and an 8-engine implementation
![[Pasted image 20240122152654.png]]





# A FPGA-based Snappy decompressor-Filter


## 5 results
### setup

![[Pasted image 20240122160731.png]]
[26] sample videos, \benchmark2," http://www.sample-videos.com/
download-sample-text-le.php, accessed Dec 17, 2017.
[27] applied maths, \benchmark1," http://www.applied-maths.com/download/
sample-data, accessed Dec 17, 2017.
![[Pasted image 20240122161424.png]]



# HW of LZMA
![[Pasted image 20240122163931.png]]

CR = 57%

