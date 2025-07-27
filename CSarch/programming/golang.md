
ref:
https://zhuanlan.zhihu.com/p/403114396
https://golang.google.cn/learn/
https://www.runoob.com/go/go-tutorial.html

<a href=" https://denganliang.github.io/the-way-to-go_ZH_CN/directory.html">The way to go</a>

1. 静态语言： 
	1. 一般都需要通过编译器（compiler）将源代码翻译成机器码，之后才能执行。程序被编译之后无论是程序中的数据类型还是程序的结构都不可以被改变 
	2. 静态语言的性能和安全性都非常好, 例如C和C++、Go, 但是C和C++的缺点是开发速度慢, 维护成本高
2. .动态语言
	1. 一般不需要通过编译器将源代码翻译成机器码，在运行程序的时候才逐行翻译。程序在运行的过程中可以动态修改程序中的数据类型和程序的结构
	2. 动态语言开发速度快,维护成本低,例如Ruby和Python, 但是Ruby和Python的性能和安全性又略低

 Go语言(Golang)是Google公司2009年推出的一门"高级编程言语", 目的是为了解决:  
- "现有主流编程语言"明显**落后于硬件发展速度**的问题
- **不能合理利用多核CPU**的优势提升软件系统性能的问题
- 软件复杂度越来越高, **_维护成本也越来越高_**的问题
- 企业开发中不得不在**_快速开发和性能之间艰难抉择_**的问题
- Go语言专门针对多核CPU进行了优化, **能够充分使用硬件多核CPU的优势**, 使得通过Go语言编写的**软件系统性能能够得到很大提升**
- Go语言编写的程序,既可以媲美C或C++代码的运行速度, 也可以媲美Ruby或Python开发的效率
- 所以Go语言很好的解决了"现有主流编程语言"存在的问题, 被誉"现代化的编程语言"

优势

- 丰富的标准库  ：Go目前已经内置了大量的库，特别是网络库非常强大；Go里面也可以直接包含c代码，利用现有的丰富的C库
- 跨平台编译和部署：Go代码可直接编译成机器码，不依赖其他库。并且Go代码还可以做到跨平台编译(例如: window系统编译linux的应用)
- 内置强大的工具 ：Go语言里面内置了很多工具链，最好的应该是gofmt工具，自动化格式化代码，能够让团队review变得简单
- 性能优势: Go 极其地快。其性能与 C 或 C++相似。在我们的使用中，Go 一般比 Python 要快 30 倍左右  
- 语言层面支持并发，这个就是Go最大的特色，天生的支持并发，可以充分的利用多核，很容易的使用并发
- 内置runtime，支持垃圾回收

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

# The way to go CN
# 第 2 章：安装与运行环境
目前有2个版本的编译器：Go 原生编译器 gc 和非原生编译器 gccgo
>[!note]
当你在创建目录时，文件夹名称永远不应该包含空格，而应该使用下划线 “_” 或者其它一般符号代替。

# 第 4 章：基本结构和基本数据类型
文件名均由小写字母组成，不包含空格或其他特殊字符。