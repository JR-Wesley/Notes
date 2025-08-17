---
dateCreated: 2025-08-16
dateModified: 2025-08-16
---
# 语法整理

### **一、基础语法（入门阶段）**
1. **程序结构与包管理**
   - **包管理工具 Cargo**：创建项目、管理依赖、构建和运行程序。

     ```bash
     cargo new my_project
     cargo build
     cargo run
     ```

   - **模块组织**：使用 `mod` 定义模块，`pub` 导出可见性。

     ```rust
     mod math {
         pub fn add(a: i32, b: i32) -> i32 { a + b }
     }
     ```

2. **变量与数据类型**
   - **变量声明**：
     - 默认不可变：`let x = 5;`
     - 可变变量：`let mut y = 10;`
   - **数据类型**：
     - **标量类型**：整数（`i8`, `u32`, `isize`）、浮点数（`f32`, `f64`）、布尔值（`bool`）、字符（`char`）。

       ```rust
       let i32_num: i32 = -12345;
       let f64_num: f64 = 3.14;
       let is_true: bool = true;
       let c: char = 'R';
       ```

     - **复合类型**：元组（`tuple`）、数组（`array`）、切片（`slice`）。

       ```rust
       let tup: (i32, f64, char) = (500, 6.4, 'c');
       let arr = [1, 2, 3, 4, 5];
       let slice = &arr[1..3]; // 引用数组的第2和第3个元素
       ```

3. **控制流**
   - **条件语句**：`if-else`、`if let`。

     ```rust
     let num = 3;
     if num > 5 {
         println!("大于5");
     } else if num == 5 {
         println!("等于5");
     } else {
         println!("小于5");
     }
     ```

   - **循环语句**：`loop`、`while`、`for`。

     ```rust
     for i in 0..5 {
         println!("{}", i); // 遍历 0 到 4
     }
     ```

4. **函数与方法**
   - **函数定义**：支持多返回值。

     ```rust
     fn add(a: i32, b: i32) -> i32 {
         a + b
     }
     ```

   - **方法接收器**：通过 `self` 定义方法。

     ```rust
     struct Rectangle {
         width: u32,
         height: u32,
     }
     impl Rectangle {
         fn area(&self) -> u32 {
             self.width * self.height
         }
     }
     ```

5. **错误处理**
   - **Result 和 Option 类型**：

     ```rust
     fn divide(a: f64, b: f64) -> Result<f64, String> {
         if b == 0.0 {
             Err("Division by zero".to_string())
         } else {
             Ok(a / b)
         }
     }
     ```

   - **匹配模式**：`match` 或 `if let` 处理结果。

     ```rust
     match divide(10.0, 2.0) {
         Ok(result) => println!("Result: {}", result),
         Err(e) => println!("Error: {}", e),
     }
     ```

---

### **二、核心概念（进阶阶段）**
1. **所有权与借用**
   - **所有权规则**：
     - 每个值有一个所有者。
     - 所有权转移（Move）：赋值或参数传递时默认转移所有权。

       ```rust
       let s1 = String::from("hello");
       let s2 = s1; // s1 的所有权转移给 s2
       // println!("{}", s1); // 错误：s1 不再有效
       ```

     - 克隆（Clone）：显式复制值。

       ```rust
       let s1 = String::from("hello");
       let s2 = s1.clone(); // 显式克隆
       println!("s1: {}, s2: {}", s1, s2); // 有效
       ```

   - **借用（Borrowing）**：
     - 不可变借用：`&T`。
     - 可变借用：`&mut T`。

     ```rust
     fn calculate_length(s: &String) -> usize {
         s.len()
     }
     let s = String::from("hello");
     let len = calculate_length(&s);
     println!("Length: {}", len);
     ```

2. **生命周期（Lifetimes）**
   - 解决悬垂引用问题，通过标注生命周期参数确保引用有效性。

     ```rust
     fn longest<'a>(s1: &'a str, s2: &'a str) -> &'a str {
         if s1.len() > s2.len() { s1 } else { s2 }
     }
     ```

3. **智能指针**
   - `**Box<T>**`：堆分配内存。
   - `**Rc<T>**`：引用计数指针（多所有权）。
   - `**RefCell<T>**`：运行时借用检查（可变借用）。

     ```rust
     use std::rc:: Rc;
     let a = Rc:: new (vec![1, 2, 3]);
     let b = a.clone (); // 共享所有权
     ```

4. **泛型与 trait**
   - **泛型函数/结构体**：

     ```rust
     fn largest<T: PartialOrd + Copy>(list: &[T]) -> T {
         *list.iter (). max (). unwrap ()
     }
     ```

   - **trait**：定义行为接口。

     ```rust
     trait Summary {
         fn summarize (&self) -> String;
     }
     impl Summary for String {
         fn summarize (&self) -> String {
             format! ("String: {}", self[.. 10])
         }
     }
     ```

5. **并发编程**
   - **线程**：使用 `std::thread::spawn` 创建线程。

     ```rust
     use std:: thread;
     thread:: spawn (|| {
         println! ("Hello from a thread!");
     }). join (). unwrap ();
     ```

   - **通道（Channel）**：线程间通信。

     ```rust
     use std::sync:: mpsc;
     let (tx, rx) = mpsc:: channel ();
     thread:: spawn (move || {
         tx.send ("Hello"). unwrap ();
     });
     println! ("Received: {}", rx.recv (). unwrap ());
     ```

   - **异步编程**：使用 `async/await`。

     ```rust
     use std::future:: Future;
     async fn fetch_data () -> String {
         "Data". to_string ()
     }
     ```

---

### **三、高级语法与标准库**
1. **宏（Macros）**
   - 定义宏：`macro_rules!`。

     ```rust
     macro_rules! create_function {
         ($func_name:ident) => {
             fn $func_name () {
                 println! ("You called {}", stringify! ($func_name));
             }
         };
     }
     create_function! (hello);
     ```

2. **模式匹配**
   - 使用 `match` 解构复杂类型。

     ```rust
     enum Message {
         Quit,
         Move { x: i 32, y: i 32 },
     }
     let msg = Message:: Move { x: 1, y: 2 };
     match msg {
         Message:: Quit => println! ("Quit"),
         Message:: Move { x, y } => println! ("Move to ({}, {})", x, y),
     }
     ```

3. **字符串与集合**
   - **字符串**：`String`（可变）与 `&str`（不可变）。

     ```rust
     let s 1 = String:: from ("hello");
     let s 2 = &s 1; // 不可变借用
     ```

   - **集合**：`Vec<T>`（动态数组）、`HashMap<K, V>`。

     ```rust
     use std::collections:: HashMap;
     let mut map = HashMap:: new ();
     map.insert ("key", "value");
     ```

4. **文件与 I/O**
   - 读写文件：`std::fs::File`。

     ```rust
     use std::fs:: File;
     use std::io::prelude::*;
     let mut file = File:: create ("example. txt")?;
     file. write_all (b"Hello, Rust!")?;
     ```

---

### **四、工程化与实践**
1. **Cargo 工具链**
   - **依赖管理**：`Cargo. toml` 配置依赖。

     ```toml
     [dependencies]
     serde = "1.0"
     ```

   - **测试与文档**：`cargo test` 运行测试，`cargo doc` 生成文档。

2. **实际项目开发**
   - **命令行工具**：使用 `clap` 或 `structopt` 解析参数。
   - **Web 服务**：使用 `Actix-web` 或 `Rocket` 框架。
   - **嵌入式开发**：通过 `embedded-hal` 支持硬件交互。

---

### **五、学习资源与路径**
1. **官方文档**
   - [The Rust Programming Language（中英文版）](https://kaisery.github.io/trpl-zh-cn/)
   - [Rust 标准库文档](https://doc.rust-lang.org/std/)
2. **书籍推荐**
   - 《Rust 编程之道》：系统讲解 Rust 核心语法与项目实践。
   - 《Rust in Action》：实战案例驱动学习。
3. **编译器错误**：Rust 编译器错误信息详细，需仔细阅读并修改代码。
4. **调试工具**：使用 `gdb`/`lldb` 或 `rust-gdb` 调试。
5. **性能优化**：通过 `cargo bench` 进行基准测试，优化关键代码。
