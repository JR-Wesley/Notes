# Verilog高级数字

# RTL

数字电路设计技术：状态机、FIFO、仲裁、同步

芯片专业知识：综合、时序分析、可测性设计，ASIC/SOC

函数：可用来生成可综合组合逻辑。具有一个函数名，一个或多个输入信号，以及一个输出信号，内部不能出现时间控制语句

任务：可以定义输入输出端口，定时抗旨，任务按顺序执行，可以可综合也可以不可综合，更多用于testbench

generate

```verilog
generate
    genvar i;
    for(i = 0; i < PTR; i = i + 1)
        begin
            ;
        end
endgenerate

generate
    genvar n;
    for(n = 0; n < NUM; n = n + 1)
        begin:switch_port_inst
            swith_port u
            (.a(in1[n])
             .b(out1[n]));
        end
endgenerate
```

## 用于验证

```verilog
initial
    begin
    end
$finish
// 完成仿真退出
$stop
// 停止命令行输入.继续
$display("");
$monitor("");
// monitor在监视的信号数值变化时才显示
$time
// 返回当前时间
$realtime
// 返回时间带小数
$random/$random(seed)
// 返回32带符号随机整数，seed指出范围
$save("file_name");
// 将仿真器当前仿真状态信息保存
$readmemh/$writememh
// 读取或写入文本
$fopen/$fclose

while/for/repeat
// 循环
    
force/release
// 将一个固定值强制赋予一个reg或wire变量，执行后，变量值不会再改变

fork/join
// 内部语句并发执行，当内部多条语句都执行完后，继续执行后面语句。主要用于testbench
```

# 数字电路



