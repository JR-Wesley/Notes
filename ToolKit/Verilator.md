
# 
`Verilator` is a tool that compiles `Verilog` and `SystemVerilog` sources to highly optimized (and optionally multithreaded) cycle-accurate `C++` or `SystemC` code. The converted modules can be instantiated and used in a C++ or a SystemC testbench, for verification and/or modelling purposes.

More information can be found at [the official Verilator website](https://www.veripool.org/verilator/) and [the official manual](https://verilator.org/guide/latest/).
Verilator is a [cycle-based](https://www.asic-world.com/verilog/verifaq3.html) simulator, which means it does not evaluate time within a single clock cycle, and does not simulate exact circuit timing. Instead, the circuit state is typically evaluated once per clock-cycle, so any intra-period glitches cannot be observed, and timed signal delays are not supported. This has both benefits and drawbacks when comparing Verilator to other simulators.