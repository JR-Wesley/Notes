# Digital System Test and Testable Design Using HDL Models and Architectures

## ch2 Verilog HDL for Design and Test
## 2.7 Adding Machine


## ch8 Standard IEEE Test Access Methods
Scan(previous chapter) focus on the inside of chip or a core
Boundary scan targets the boundary of a CUT (IEEE std.11.49.1) and doesn't interfere in the design of a core. The purpose is to isolate the core being tested from other devices.

## 8.1 Boundary Scan Basics
JTAG(BS-1149.1) mainly consists of a scan register on the ports of a component for testing its interconnects and core logic
![[general structure of BS.jpg]]
Boundary scan uses **a chain of scan flip-flops** to shift test data into the inputs of a core logic being tested, and uses the same mechanism to move test response out. The scan flip-flops isolate a core from its neighboring cores, and testing of each core is done independently.
2 modes of BS works:
- **noninvasive**: independent from board or chip core logic, the test hardware (BS-1149.1) communicates with the outside world for bringing in test data, or transmitting response out of the system. This is done while the rest of the system performs its normal functions
- **pin-permission**: the BS-1149.1 hardware takes over input and output pins of a core logic for testing its interconnects. In this mode, the core logic is disconnected from its environments and is only operated by test logic. After completion of a pin-permission mode operation, it is important for the test hardware to be put back in the noninvasive mode to avoid bus conflicts while the system performs its normal functions.

## 8.2 Boundary Scan Architecture
Test hardware consists of controllers, registers, decoders(like a single testable package)
![[Details of BS standard.jpg]]
### 8.2.1 Test Access Port
BS-1149.1 adds several pins for test data and control (4 and 1 optional, cannot be used by the core logic) . Pins not used should be left floating high to prevent interference in the normal functionality

| Name | Function                    | Active  | default | neccessity |
| ---- | --------------------------- | ------- | ------- | ---------- |
| TCK  | clk                         | 50%     |         | √          |
| TDI  | serial data in              | rising  | 1       | √          |
| TMS  | mode select                 | rsing   | 1       | √          |
| TDO  | serial data out             | falling | Z       | √          |
| TRST | reset into noninvasive mode | low     | 1       | optional   |
Several registers of the main hardware are categorized into instruction and data registers.

### 8.2.2 BS-1149.1 Registers
#### Instruction Register
An instruction register of at least 2 bits holds the instructions and is a mandatory part of this standard.
the standard full-feature instruction register cell consists of two flip-flops, one for shift or capture and another for update

![[Instruction register cell structure.png]]


| 信号名             | 作用                                  |
| --------------- | ----------------------------------- |
| ShiftIR         | 移位使能                                |
| PI Data         | 并行输入数据                              |
| From last cell  | 扫描输入（数据来自上一扫描单元的扫描输出端口）             |
| ClockIR         | 扫描单元时钟信号，仅在Capture-IR和Shift-IR状态时有效 |
| Update IR       | 指令位触发器时钟信号，仅在Update-IR状态时有效         |
| Reset           | 复位信号，用于强制性指令解码（BYPASS或IDCODE）       |
| To next cell    | 扫描输出（连接到下一扫描单元的扫描输入端口）              |
| Instruction bit | 指令位，指示是否为该指令位对应的指令                  |


#### Data Registers
The instruction that is loaded in the instruction register causes one of the data registers to go between TDI and TDO serial input and serial output

- Bypass register(mandatory): bypass a core from scan chain so that serially shifted data can reach the target core quicker

![[Bypass register cell structure.png]]

Device identification register(optional)

- Boundary scan register
Din/Dout go between the interconnect and the core logic. Sin/Sout are used for shifting serial data that enter the register on TDI and exit the register on the TDO.
The BS register cell has a shift or capture FF and an update one.
ModeControl = 0 -> noninvasive mode
ModeControl = 1 -> pin-permission mode
![[BS cell.png]]



### 8.2.3 TAP(test access port) controller
All BS operations are controlled by a FSM with 16 states. TCLK for the clk, TMS for the input, TRST for reseting.
![[TAP controller.png|JTAG/TAP controller.png]]
*Test_Logic_Reset*: entered by issuing TRST or by TMS being 1 for 5 consecutive clocks. A RstBar reset signal is issued to all BS components, loading a null pattern in the instruction register.
*Run_Test_Idle*: contents of registers remain the same as in the previous state. In the state, the core logic can perform its own self-test operations.
*Select_DR_Scan*/*Select_IR_Scan*: a temporary state
*Capture_IR*: the instruction register ClockIR is issued that caused it to perform it to perform a parallel load. This will load "01" in the LSB of the instruction register. If the instruction register is longer than 2 bits, the rest of the bits will receive a predefined value that is treated as a null instruction
*Capture_DR* and other data register-related control states: a specific data register selected is targeted. ClockDR is issued. The boundary scan data register captures data.
*Shift_IR*: the instruction register is placed between TDI and TDO. While in this state, rising edges of ClockIR cause captured data to be shifted out on Sout and new serial data moved from Sin. In this state a new instruction bit pattern that appears on TDI will be shifted in the capture flip-flops. After completion of shifting the proper bit pattern, the instruction is loaded when TAP controller goes into the Update_IR state.
*Shift_DR*: serial data are shifted in the selected data register. ShiftDR signal is set to 1 and shifting occurs on the rising edge of ClockDR
*Exit1_DR/Pause_DR/Exit2_DR*: allows a pause while the external tester fetches more data for its buffer memory
*Exit1_IR/Pause_IR/Exit2_IR*: may be eliminated cause instructions are short
*Update_IR*: after an instruction is shifted in the instruction register's capture FF, or a complete test vector is shifted in the BS register's capture FF. UpdateIR signal is issued and on its rising edge, the bit pattern in the shift register chain loads into the instruction register as the current instruction. Once done, signals corresponding to this new instruction are activated, and a specific data register will be selected. The decoder uses the instruction register and TAP signals to issue selection and clocking signals to data registers
*Update_DR*: loads test data bits that have been shifted in the shift register chain of a
data register into its update FF.

### 8.2.4 The Decoder Unit
instruction in the instruction register and signals from TAP -> signals to data register

### 8.2.5 Select and Other Units
MUX, FF, tristate buffer(puts TDO in the floating state) lead to TDO output

## 8.3 Boundary Scan Test Instructions


### 8.3.1 Mandatory Instructions
- Bypass: shortening the scan path and bypassing the units not participating in a certain round of test
![[Fig. 8.12 Bypass instruction execution.jpg]]
- Sample: in the noninvasive mode and takes a snap-shot of the input interconnect values and outputs of core logic
![[Fig. 8.13 Sample instruction execution.jpg]]
- Preload: loading the corresponding bit pattern and initializing the scan cell
![[Fig. 8.14 Preload instruction execution.jpg]]
- Extest: in pin-permission mode, takeover the interconnects
The first time Extest instruction is being executed it must follow the complete execution of Preload. Preload loads test data into update FF of the BS cells. After Extest, new test data will be shifted as test response from the previous round of testing is being shifted out
![[Fig. 8.15 Output cells updating test data in Extest.jpg]]
![[Fig. 8.16 Input cells capturing test data in Extest.jpg]]


- Intest: applies test data to inpus of a chip and reads out the test response, in pin-permission mode
The Preload precedes Intest
![[Fig. 8.18 Intest input and output cells.jpg]]
## 8.4 Board Level Scan Chain Structure
At the board or chip level, various arrangements of scan registers can play an important role in saving test hardware and test time
