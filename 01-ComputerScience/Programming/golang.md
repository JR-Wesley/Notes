---
dateCreated: 2024-11-07
dateModified: 2025-08-16
---
ref:

https://zhuanlan.zhihu.com/p/403114396

https://golang.google.cn/learn/

https://www.runoob.com/go/go-tutorial.html

<a href=" https://denganliang.github.io/the-way-to-go_ZH_CN/directory.html">The way to go</a>

1. 静态语言：
	1. 一般都需要通过编译器（compiler）将源代码翻译成机器码，之后才能执行。程序被编译之后无论是程序中的数据类型还是程序的结构都不可以被改变
	2. 静态语言的性能和安全性都非常好, 例如 C 和 C++、Go, 但是 C 和 C++ 的缺点是开发速度慢, 维护成本高
2. .动态语言
	1. 一般不需要通过编译器将源代码翻译成机器码，在运行程序的时候才逐行翻译。程序在运行的过程中可以动态修改程序中的数据类型和程序的结构
	2. 动态语言开发速度快,维护成本低,例如 Ruby 和 Python, 但是 Ruby 和 Python 的性能和安全性又略低

 Go 语言 (Golang) 是 Google 公司 2009 年推出的一门 " 高级编程言语 ", 目的是为了解决:

- " 现有主流编程语言 " 明显**落后于硬件发展速度**的问题
- **不能合理利用多核 CPU**的优势提升软件系统性能的问题
- 软件复杂度越来越高, **_ 维护成本也越来越高 _**的问题
- 企业开发中不得不在**_ 快速开发和性能之间艰难抉择 _**的问题
- Go 语言专门针对多核 CPU 进行了优化, **能够充分使用硬件多核 CPU 的优势**, 使得通过 Go 语言编写的**软件系统性能能够得到很大提升**
- Go 语言编写的程序,既可以媲美 C 或 C++ 代码的运行速度, 也可以媲美 Ruby 或 Python 开发的效率
- 所以 Go 语言很好的解决了 " 现有主流编程语言 " 存在的问题, 被誉 " 现代化的编程语言 "

优势

- 丰富的标准库：Go 目前已经内置了大量的库，特别是网络库非常强大；Go 里面也可以直接包含 c 代码，利用现有的丰富的 C 库
- 跨平台编译和部署：Go 代码可直接编译成机器码，不依赖其他库。并且 Go 代码还可以做到跨平台编译 (例如: window 系统编译 linux 的应用)
- 内置强大的工具：Go 语言里面内置了很多工具链，最好的应该是 gofmt 工具，自动化格式化代码，能够让团队 review 变得简单
- 性能优势: Go 极其地快。其性能与 C 或 C++ 相似。在我们的使用中，Go 一般比 Python 要快 30 倍左右
- 语言层面支持并发，这个就是 Go 最大的特色，天生的支持并发，可以充分的利用多核，很容易的使用并发
- 内置 runtime，支持垃圾回收

## 语言的核心结构与技术

`scanner_test.go`

25 个关键字或保留字和 36 个预定义标识符

|          |             |        |           |        |
| -------- | ----------- | ------ | --------- | ------ |
| break    | default     | func   | interface | select |
| case     | defer       | go     | map       | struct |
| chan     | else        | goto   | package   | switch |
| const    | fallthrough | if     | range     | type   |
| continue | for         | import | return    | var    |

|        |         |         |         |        |         |           |            |         |
| ------ | ------- | ------- | ------- | ------ | ------- | --------- | ---------- | ------- |
| append | bool    | byte    | cap     | close  | complex | complex64 | complex128 | uint16  |
| copy   | false   | float32 | float64 | imag   | int     | int8      | int16      | uint32  |
| int32  | int64   | iota    | len     | make   | new     | nil       | panic      | uint64  |
| print  | println | real    | recover | string | true    | uint      | uint8      | uintptr |

# The way to Go CN
# 第 2 章：安装与运行环境

目前有 2 个版本的编译器：Go 原生编译器 gc 和非原生编译器 gccgo

> [!note]
当你在创建目录时，文件夹名称永远不应该包含空格，而应该使用下划线 “_” 或者其它一般符号代替。

# 第 4 章：基本结构和基本数据类型

文件名均由小写字母组成，不包含空格或其他特殊字符。

# 语法整理

### **一、基础语法**
1. **程序结构与包管理**
   - **包（package）**：理解 `package main` 作为程序入口，`import` 导入标准库或第三方包。

     ```go
     package main
     import "fmt"
     func main() {
         fmt.Println("Hello, Go!")
     }
     ```

   - **文件结构**：了解 `.go` 文件的组织方式，命名规则（首字母大写导出）。

2. **变量与数据类型**
   - **变量声明**：
     - 显式声明：`var a int = 10`
     - 短变量声明：`b := 20`（仅限函数内）
     - 批量声明：`var (x int; y string = "Go")`
   - **数据类型**：
     - 基本类型：`int`, `float64`, `string`, `bool`
     - 复合类型：`array`, `slice`, `map`, `struct`
     - 特殊类型：`interface{}`（空接口）、`uintptr`（指针类型）
   - **类型推导**：`:=` 自动推断类型。
   - **常量**：`const pi = 3.14159`（编译时常量）。

3. **控制结构**
   - **条件语句（if-else）**：

     ```go
     if x > 10 {
         fmt.Println("x is large")
     } else {
         fmt.Println("x is small")
     }
     ```

   - **循环（for）**：
     - 标准循环：`for i := 0; i < 10; i++ {}`
     - 类似 `while`：`for x < 100 { x *= 2 }`
     - 无限循环：`for {}`（需手动 `break`）
   - **Switch 语句**：

     ```go
     switch num {
     case 1:
         fmt.Println("One")
     default:
         fmt.Println("Other")
     }
     ```

4. **函数与方法**
   - **函数定义**：支持多返回值。

     ```go
     func add(a, b int) (int, string) {
         return a + b, "Result"
     }
     ```

   - **匿名函数与闭包**：

     ```go
     func main() {
         add := func(x, y int) int { return x + y }
         fmt.Println(add(2, 3)) // 输出 5
     }
     ```

   - **方法接收器**：通过 `func (receiver T) Method()` 定义方法。

     ```go
     type Rectangle struct{ width, height float64 }
     func (r Rectangle) Area() float64 {
         return r.width * r.height
     }
     ```

5. **结构体与接口**
   - **结构体（struct）**：定义复杂数据类型。

     ```go
     type User struct {
         Name string
         Age  int
     }
     ```

   - **接口（interface）**：定义行为规范。

     ```go
     type Animal interface {
         Speak() string
     }
     ```

6. **错误处理**
   - **错误类型**：`error` 接口。

     ```go
     func Divide(a, b float64) (float64, error) {
         if b == 0 {
             return 0, errors.New("division by zero")
         }
         return a / b, nil
     }
     ```

   - **panic 和 recover**：处理严重错误。

     ```go
     defer func() {
         if r := recover(); r != nil {
             fmt.Println("Recovered:", r)
         }
     }()
     panic("Something went wrong!")
     ```

---

### **二、进阶语法**
1. **并发编程（Goroutine 与 Channel）**
   - **Goroutine**：轻量级线程。

     ```go
     go func() {
         fmt.Println("Running in a goroutine")
     }()
     ```

   - **Channel**：协程间通信。

     ```go
     ch := make(chan int)
     go func() {
         ch <- 42 // 发送数据
     }()
     fmt.Println(<-ch) // 接收数据
     ```

   - **同步机制**：
     - `sync.WaitGroup`：等待所有协程完成。
     - `sync.Mutex`：互斥锁保护共享资源。
     - `select`：多通道操作选择。

       ```go
       select {
       case msg1 := <-ch1:
           fmt.Println("Received from ch1:", msg1)
       case msg2 := <-ch2:
           fmt.Println("Received from ch2:", msg2)
       }
       ```

2. **切片（Slice）与映射（Map）**
   - **切片操作**：

     ```go
     s := []int{1, 2, 3}
     s = append(s, 4) // 添加元素
     s = s[1:]        // 截取子切片
     ```

   - **映射操作**：

     ```go
     m := map[string]int{"a": 1, "b": 2}
     m["c"] = 3 // 添加/修改
     delete(m, "a") // 删除
     ```

3. **指针与内存管理**
   - **指针**：Go 没有指针运算，但支持指针类型。

     ```go
     var p *int
     a := 10
     p = &a
     fmt.Println(*p) // 输出 10
     ```

4. **反射（Reflection）**
   - 使用 `reflect` 包动态操作类型和值。

     ```go
     v := reflect.ValueOf(x)
     t := reflect.TypeOf(x)
     ```

5. **接口与类型断言**
   - **类型断言**：判断接口值的具体类型。

     ```go
     var i interface{} = "hello"
     s, ok := i.(string)
     if ok {
         fmt.Println("String:", s)
     }
     ```

---

### 项目开发

1. **模块化与依赖管理**
   - **Go Modules**：管理依赖版本。

     ```bash
     go mod init mymodule
     go get github.com/example/dependency
     ```

2. **标准库应用**
   - **网络编程**：`net/http` 构建 HTTP 服务。

     ```go
     http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
         fmt.Fprintf(w, "Hello, World!")
     })
     http.ListenAndServe(":8080", nil)
     ```

   - **文件操作**：`os`, `io/ioutil` 读写文件。

     ```go
     data, _ := os.ReadFile("file.txt")
     os.WriteFile("output.txt", data, 0644)
     ```

3. **测试与调试**
   - **单元测试**：使用 `testing` 包。

     ```go
     func TestAdd(t *testing.T) {
         if add(2, 3) != 5 {
             t.Fail()
         }
     }
     ```

   - **性能分析**：`pprof` 工具分析 CPU/内存占用。
