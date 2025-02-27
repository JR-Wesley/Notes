---
dateCreated: 2024-11-28
dateModified: 2024-12-04
---

<a href=" https://pysv.readthedocs.io/index.html">pysv: running python code in SV</a>

<a href="https://docs.cocotb.org/en/stable/">cocotb</a>

<a href=" https://www.zhihu.com/column/c_1545791526281879552">关于对 Python 进行 ASIC 验证的讨论</a>

<a href=" https://www.zhihu.com/column/c_1400489023509618688">利用 TVM/Verilator，进行深度学习相关部署</a>

# Verilator

<a href=" https://ioyoi.me/tech/verilator-tutorial/">verilator 入门</a>

# Cocotb

**cocotb** is a _COroutine_ based _COsimulation_ _TestBench_ environment for verifying VHDL and SystemVerilog RTL using <a href="https://www.python.org/">Python</a>. cocotb requires a <a href=" https://docs.cocotb.org/en/stable/simulator_support.html#simulator-support">simulator</a> to simulate the HDL design.

A test is simply a Python function. At any given time either the simulator is advancing time or the Python code is executing. The [`await`]( https://docs.python.org/3/reference/expressions.html#await " (in Python v3.13)") keyword is used to indicate when to pass control of execution back to the simulator. A test can spawn multiple coroutines, allowing for independent flows of execution.

## Quickstart

In cocotb, you can access all internals of your design, e.g. signals, ports, parameters, etc. through an object that is passed to each test. In the following we’ll call this object `dut`.

- Use `.value` to get a signal’s current value.
- Use the `@cocotb.test()` decorator to mark the test function to be run.

## NOTE

> [!装饰器（Decorator）]
> 1. **增加功能**：装饰器可以在不修改原函数代码的情况下，给函数增加新的功能。例如，记录函数的执行时间、检查函数的输入参数、处理函数的异常等。
> 2. **代码复用**：通过装饰器，可以将一些通用的功能抽象出来，然后在多个函数上复用这些功能，避免代码重复。
> 3. **解耦功能**：装饰器可以将业务逻辑和横切关注点（如日志、权限检查、事务管理等）解耦，使得代码更加清晰和模块化。
> 4. **扩展性**：装饰器提供了一种灵活的方式来扩展函数的功能，可以在运行时动态地添加或修改装饰器。
> 5. **提高代码可读性**：通过使用装饰器，可以将一些复杂的逻辑封装起来，使得代码更加简洁和易于理解。
> 6. **实现 AOP（面向切面编程）**：装饰器是实现 AOP 的一种方式，可以在不改变业务逻辑代码的情况下，对函数的执行过程进行干预，例如在函数执行前后添加日志、进行权限检查等。

参考：<a href=" https://blog.csdn.net/zhh763984017/article/details/120072425">弄懂 Python 装饰器</a>

装饰器是给现有的模块**增添新的功能**，可以对原函数进行功能扩展，而且还不需要修改原函数的内容，也不需要修改原函数的调用。

> [!装饰器的使用符合了面向对象编程的开放封闭原则。]
> 开放封闭原则主要体现在两个方面：
> 对扩展开放，意味着有新的需求或变化时，可以对现有代码进行扩展，以适应新的情况。
> 对修改封闭，意味着类一旦设计完成，就可以独立其工作，而不要对类尽任何修改。

### Test

Since generating a clock is such a common task, cocotb provides a helper for it - [`cocotb. clock. Clock`]( https://docs.cocotb.org/en/stable/library_reference.html#cocotb.clock.Clock "cocotb. clock. Clock").

### Runners (experimental) or Makefile

Running a test involves three steps:

- first the runner is instantiated with `get_runner` based on the default simulator to use,
- then the [HDL](https://docs.cocotb.org/en/stable/glossary.html#term-HDL) is built using the design sources and the toplevel with `runner.build`,
- finally, the module containing tests to run are passed to `runner.test`

## Writing Testbench

When cocotb initializes it finds the toplevel instantiation in the simulator and creates a _handle_ called `dut`. Using "dot" notation to access object and signals inside.

```python
# Get a reference to the "clk" signal and assign a value
# writes are not applied immediately, but delayed until the next write cycle.
clk = dut.clk
clk.value = 1

# set a new value immediately
sig.setimmediatevalue(new_val)

# Direct assignment through the hierarchy
dut.input_signal.value = 12
```

- reading values

```python
# Read a value back from the DUT
>>> count = dut.counter.value
>>> print(count.binstr)
1X1010
>>> # Resolve the value to an integer (X or Z treated as 0)
>>> print(count.integer)
42
>>> # Show number of bits in a value
>>> print(count.n_bits)
6
```

Using `@cocotb.test` to identify tests.
