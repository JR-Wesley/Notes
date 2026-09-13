---
dateCreated: 2025-08-19
dateModified: 2025-08-19
---

# Vector

`std::vector` 是 C++ 标准模板库 (STL) 中最常用和最重要的容器之一。它的设计目标是提供一个动态数组，结合了 C 风格数组的高效随机访问特性和动态内存管理的便利性。

以下是 `std::vector` 的核心实现原理：

## 1. 核心数据结构：三个指针

`std::vector` 通常由三个指针（或等效的迭代器）来管理其内部存储：

- **`start` / `begin`**: 指向当前已分配内存块中第一个元素的指针。
- **`finish` / `end`**: 指向当前已分配内存块中最后一个**已构造**元素的下一个位置的指针。`finish - start` 就是 `size()`。
- **`end_of_storage` / `capacity_end`**: 指向当前已分配内存块末尾的指针。`end_of_storage - start` 就是 `capacity()`。

这三个指针共同定义了 `vector` 的状态：

- **大小 (Size)**: `finish - start`
- **容量 (Capacity)**: `end_of_storage - start`

## 2. 动态增长策略

当向 `vector` 添加元素（如 `push_back`）导致 `size()` 即将超过 `capacity()` 时，需要进行扩容。

- **分配新内存**: 分配一块比当前容量更大的新内存块。标准**不规定**具体的增长因子，但实践中（如 GCC 的 libstdc++ 和 Clang 的 libc++）通常采用 **1.5 倍或 2 倍**的增长策略。选择 2 倍增长是为了摊销插入操作的平均时间复杂度为 O(1)。
- **移动/复制元素**: 将旧内存块中的所有现有元素**移动构造**（如果元素类型支持移动语义）或**复制构造**到新内存块中。
- **析构并释放旧内存**: 调用旧内存块中所有元素的析构函数，并释放旧的内存块。
- **更新指针**: 更新 `start`, `finish`, `end_of_storage` 指向新的内存块。

**关键点**:

- **移动语义**: C++11 引入移动语义后，如果元素类型有高效的移动构造函数（如 `std::string`, `std::vector` 本身），扩容时会优先使用移动，这比复制要快得多。
- **异常安全**: 扩容过程需要保证强异常安全。如果在移动/复制过程中抛出异常，`vector` 的状态（大小、元素）必须保持不变（通常通过先分配新内存，再复制/移动，成功后再释放旧内存来实现）。
- **迭代器失效**: 扩容会导致**所有**指向该 `vector` 的迭代器、指针和引用失效，因为底层内存地址改变了。

## 3. 内存管理

- **连续存储**: `std::vector` 保证其元素在内存中是**连续存储**的。这是它与 `std::list` 或 `std::deque` 的关键区别，也是它能提供高效随机访问（O(1)）和良好缓存局部性的基础。
- **`operator new`**: 通常使用 `operator new` 或 `std::allocator` 来分配原始内存块。`std::allocator` 是标准的内存分配器，负责分配和释放未初始化的内存。
- **对象构造/析构**:
    - **分配内存**和**构造对象**是分开的。`vector` 先分配一块原始内存（使用 `allocator::allocate`），然后在需要时（如 `push_back`, `resize`）在特定位置使用 `allocator::construct`（或 `placement new`）来构造对象。
    - 在对象被移除或 `vector` 销毁时，使用 `allocator::destroy`（或显式调用析构函数）来析构对象，然后才释放内存。

## 4. 关键操作的时间复杂度

- **随机访问 (`operator[]`, `at`, `front`, `back`)**: O(1)
- **在末尾插入 (`push_back`)**:
    - 平摊 O(1)（因为扩容是摊销成本）
    - 最坏情况 O(n)（当发生扩容时）
- **在末尾删除 (`pop_back`)**: O(1)
- **在任意位置插入/删除 (`insert`, `erase`)**: O(n)（因为需要移动插入点之后的所有元素）
- **获取大小 (`size`)**: O(1)
- **获取容量 (`capacity`)**: O(1)
- **调整大小 (`resize`)**:
    - 如果新大小 <= 容量: O(|new_size - old_size|)（构造或析构元素）
    - 如果新大小 > 容量: O(n)（需要扩容和移动所有元素）
- **预留空间 (`reserve`)**:
    - 如果 `n <= capacity()`: O(1)
    - 如果 `n > capacity()`: O(n)（需要扩容）

## 5. 与 `std::array` 和 `std::deque` 的对比

- **`std::array`**: 固定大小，栈上或静态存储，无动态分配，`size()` == `capacity()`。
- **`std::deque`**: 双端队列，非连续存储（通常由分段连续块组成），支持在两端高效插入/删除（O(1)），但随机访问略慢于 `vector`，迭代器可能在插入/删除时失效。

**总结**: `std::vector` 的核心是**连续的动态数组**，通过**三个指针**管理大小和容量，利用**摊销策略**（如 2 倍增长）保证平摊高效的插入性能，并严格分离**内存分配**和**对象构造**。其连续存储特性使其成为需要高效随机访问和缓存友好的场景的首选容器。

# 哈希表

C++ 中的哈希表主要通过标准库中的 `std::unordered_map` 和 `std::unordered_set`（以及它们的多值版本 `std::unordered_multimap`/`std::unordered_multiset`）来实现。其核心原理基于**哈希函数**和**处理冲突的策略**。

以下是哈希表的主要实现原理：

## 1. 核心思想

哈希表的目标是实现**平均 O(1)** 时间复杂度的插入、删除和查找操作。

*   **键 (Key)**: 用于查找的值。
*   **值 (Value)**: 与键关联的数据（`unordered_map` 有值，`unordered_set` 的键就是值）。
*   **哈希函数 (Hash Function)**: 一个函数，将任意类型的键映射到一个**哈希值**（通常是 `size_t` 类型的整数）。
*   **桶 (Bucket)**: 哈希表内部通常是一个**数组**，数组的每个元素称为一个“桶”。
*   **槽 (Slot)**: 每个桶中可以存储一个或多个键值对（元素）。

**基本流程**：
1.  给定一个键 `key`。
2.  计算哈希值：`hash_value = hash_function(key)`。
3.  计算索引：`index = hash_value % bucket_count`。这个 `index` 决定了该键值对应该被存储在哪个桶里。
4.  在 `index` 对应的桶中查找、插入或删除键为 `key` 的元素。

## 2. 处理哈希冲突 (Collision)

不同的键可能计算出相同的 `index`（即哈希冲突），这是不可避免的。哈希表必须有策略来处理冲突。最常用的策略是**链地址法 (Separate Chaining)**。

*   **链地址法 (Separate Chaining)**:
    * 每个桶（数组元素）是一个**链表**（或其他容器，如动态数组）的头。
    * 当发生冲突时，新的元素被插入到对应桶的链表中。
    *   **查找**: 计算 `index`，然后在 `index` 对应的链表中线性搜索目标键。
    *   **插入**: 计算 `index`，检查链表中是否已存在相同键（对于 `map/set` 通常是不允许重复键的），若不存在则插入链表。
    *   **删除**: 计算 `index`，在链表中找到目标节点并删除。
    *   **优点**: 实现相对简单，删除操作容易。
    *   **缺点**: 当链表过长时，查找性能退化为 O(n)。需要额外的指针开销。

*   **开放寻址法 (Open Addressing)**:
    * 所有元素都存储在哈希表数组本身中。
    * 当发生冲突时（目标桶已被占用），按照某种**探测序列**（Probing Sequence）寻找下一个“空”的桶。
    * 常见的探测方法：
        *   **线性探测 (Linear Probing)**: `index, index+1, index+2, ...` (模 `bucket_count`)。简单但容易产生“聚集”（Clustering）。
        *   **二次探测 (Quadratic Probing)**: `index, index+1², index+2², ...`。能减少线性聚集。
        *   **双重哈希 (Double Hashing)**: 使用第二个哈希函数 `h2(key)` 来计算步长：`index, index+h2(key), index+2*h2(key), ...`。通常能提供更好的分布。
    *   **查找**: 计算 `index`，如果桶中的键不匹配，则按探测序列继续查找，直到找到匹配的键或遇到空桶（表示未找到）。
    *   **插入**: 类似查找，找到一个空桶（或标记为删除的桶）插入。
    *   **删除**: 不能简单地将桶置空，因为这会中断探测序列。通常需要将桶标记为“已删除”（Tombstone），查找时遇到“已删除”桶会继续探测，插入时可以覆盖“已删除”桶。
    *   **优点**: 空间利用率高（无指针开销），缓存局部性好（元素连续存储）。
    *   **缺点**: 实现更复杂，删除操作麻烦，负载因子不能太高（否则探测序列会很长），可能发生“聚集”。

**C++ 标准库实现**: `std::unordered_*` 容器**通常使用链地址法**。例如，GCC 的 libstdc++ 和 Clang 的 libc++ 都采用基于链表的分离链接。

## 3. 哈希函数 (Hash Function)

*   **`std::hash`**: C++ 标准库为基本类型（`int`, `float`, `char`, `std::string` 等）和一些标准容器提供了特化的 `std::hash` 模板。`std::unordered_map/set` 默认使用 `std::hash<Key>` 作为其哈希函数。
*   **自定义哈希函数**: 对于自定义类型，需要提供一个可调用对象（函数指针、函数对象、Lambda）作为模板参数，或者为 `std::hash` 特化。
*   **好哈希函数的要求**:
    *   **确定性**: 相同的输入总是产生相同的输出。
    *   **均匀分布**: 尽可能将不同的键均匀地分散到所有可能的哈希值上，以减少冲突。
    *   **高效**: 计算速度快。

## 4. 负载因子 (Load Factor) 与再哈希 (Rehashing)

*   **负载因子 (α)**: `α = 元素总数 / 桶的数量 (bucket_count)`。它衡量了哈希表的“拥挤”程度。
*   **最大负载因子 (max_load_factor)**: `std::unordered_map/set` 有一个可配置的最大负载因子（默认通常为 1.0）。当 `α > max_load_factor` 时，哈希表会自动进行**再哈希 (Rehashing)**。
*   **再哈希 (Rehashing)**:
    1.  分配一个**更大**的桶数组（通常是原大小的 2 倍或某个增长因子）。
    2.  遍历旧哈希表中的**所有元素**。
    3.  对每个元素的键重新计算哈希值和新的 `index`。
    4.  将元素插入到新桶数组的相应位置。
    5.  释放旧桶数组。
*   **目的**: 保持较低的负载因子，从而控制链表长度（链地址法）或探测序列长度（开放寻址法），保证平均 O(1) 的性能。
*   **代价**: 再哈希是 O(n) 操作，且会**使所有迭代器、指针、引用失效**（因为元素内存地址可能改变）。

## 5. 关键操作的时间复杂度

*   **平均情况**:
    * 查找 (find): O(1)
    * 插入 (insert): O(1) (平摊，因为再哈希是摊销成本)
    * 删除 (erase): O(1)
*   **最坏情况** (所有键都哈希到同一个桶):
    * 查找、插入、删除: O(n)
*   **保证**:
    * 标准库保证平均情况下的 O(1) 性能，但这依赖于**好的哈希函数**和**适当的负载因子管理**。如果哈希函数很差，性能会退化到 O(n)。

## 6. `std::map std::unordered_map` 的对比

| 特性 | `std::unordered_map` (哈希表) | `std::map` (基于红黑树) |
| :--- | :--- | :--- |
| **底层结构** | 哈希表 (数组 + 链表/其他) | 平衡二叉搜索树 (通常是红黑树) |
| **排序** | **无序** (元素顺序未定义) | **有序** (按键的 `<` 比较排序) |
| **时间复杂度** | **平均 O(1)** (查找/插入/删除) | **O(log n)** (查找/插入/删除) |
| **内存开销** | 通常较低 (但有桶数组) | 较高 (每个节点有左右子节点和颜色指针) |
| **缓存局部性** | 较好 (桶数组连续，但链表节点可能分散) | 较差 (树节点动态分配，可能分散) |
| **哈希函数** | 需要为键类型提供 `std::hash` 或自定义 | 需要为键类型提供 `operator<` 或自定义比较函数 |
| **迭代器失效** | 插入可能导致所有迭代器失效 (再哈希)，删除只影响被删元素的迭代器 | 插入通常不使迭代器失效，删除只影响被删元素的迭代器 |

## 总结

C++ 哈希表 (`std::unordered_map/set`) 的核心是：

1.  **哈希函数**将键映射到桶索引。
2.  **链地址法**（主流实现）处理哈希冲突，在每个桶中用链表存储冲突的元素。
3.  **负载因子**监控表的拥挤程度，超过阈值时触发**再哈希**（扩容并重新分配所有元素），以维持平均 O(1) 的性能。
4.  其性能高度依赖于**哈希函数的质量**。

选择 `unordered_map` 还是 `map` 取决于需求：需要**极致的平均查找速度**且**不关心顺序**时选前者；需要**有序遍历**或**最坏情况性能保证**时选后者。

# 关联容器和无须关联容器

## **一、关联容器（有序关联容器）**

### **1. 基本概念**

- **底层实现**：基于 **红黑树**（自平衡二叉搜索树），保证元素按顺序存储。
- **主要容器**：
    - `std::map` / `std::multimap`：存储键值对（Key-Value），按键排序。
    - `std::set` / `std::multiset`：存储唯一/重复的键（Key），按键排序。
- **操作时间复杂度**：插入、删除、查找为 **O(log N)**。

### **2. 行为与使用方法**

- **有序性**：
    
    - 元素按键的升序（默认）排列，支持范围查询（如 `lower_bound` 和 `upper_bound`）。
    - 示例：

        ```cpp
        std::map<std::string, int> word_count;
        word_count["apple"] = 5;  // 插入或更新
        for (const auto& [key, value] : word_count) {
            std::cout << key << ": " << value << "\n";  // 按键排序输出
        }
        ```

- **插入与查找**：
    
    - **插入**：使用 `insert` 或 `operator[]`。

        ```cpp
        word_count.insert({"banana", 3});  // 返回 pair<iterator, bool>
        ```

    - **查找**：使用 `find` 或 `operator[]`。

        ```cpp
        auto it = word_count.find("apple");
        if (it != word_count.end()) {
            std::cout << it->second;  // 输出 5
        }
        ```

- **范围操作**：
    
    - 使用 `lower_bound` 和 `upper_bound` 查找键的区间。

        ```cpp
        auto lb = word_count.lower_bound("apple");  // 第一个 >= "apple" 的元素
        auto ub = word_count.upper_bound("apple");  // 第一个 > "apple" 的元素
        ```

- **限制**：
    
    - **键不可修改**：直接修改键会破坏红黑树的有序性。需先删除再插入。
    - **键唯一性**：`map` 和 `set` 不允许重复键，`multimap` 和 `multiset` 允许。

### **3. 高级特性**

- **异类查找（C++14+）**：
    - 允许使用与键类型不同的对象进行查找（需重载比较运算符）。
    - 示例：

        ```cpp
        struct Key {
            int id;
            bool operator<(const Key& other) const { return id < other.id; }
        };
        std::map<Key, std::string> m;
        m[Key{1}] = "A";
        auto it = m.find(Key{1});  // 查找成功
        ```

---

## **二、无序关联容器（哈希表）**

### **1. 基本概念**

- **底层实现**：基于 **哈希表**，元素无序存储。
- **主要容器**：
    - `std::unordered_map` / `std::unordered_multimap`：存储键值对。
    - `std::unordered_set` / `std::unordered_multiset`：存储唯一/重复的键。
- **操作时间复杂度**：平均 **O(1)**，最坏 **O(N)**（哈希冲突严重时）。

### **2. 行为与使用方法**

- **无序性**：
    
    - 元素无固定顺序，输出顺序可能与插入顺序不同。
    - 示例：

        ```cpp
        std::unordered_map<std::string, int> word_count;
        word_count["apple"] = 5;
        word_count["banana"] = 3;
        for (const auto& [key, value] : word_count) {
            std::cout << key << ": " << value << "\n";  // 输出顺序可能为 banana, apple
        }
        ```

- **插入与查找**：
    
    - **插入**：使用 `insert` 或 `operator[]`。

        ```cpp
        word_count.insert({"orange", 2});  // 返回 pair<iterator, bool>
        ```

    - **查找**：使用 `find` 或 `operator[]`。

        ```cpp
        auto it = word_count.find("apple");
        if (it != word_count.end()) {
            std::cout << it->second;  // 输出 5
        }
        ```

- **桶管理**：
    
    - **扩容**：当负载因子（`size / bucket_count`）超过阈值时自动扩容。
    - **桶操作**：可获取桶数量、元素分布等信息。

        ```cpp
        std::cout << word_count.bucket_count();  // 当前桶数
        ```

- **自定义哈希和比较函数**：
    
    - 对于自定义类型，需提供哈希函数和相等比较函数。
    - 示例：

        ```cpp
        struct Key {
            int id;
            bool operator==(const Key& other) const { return id == other.id; }
        };
        
        struct Hash {
            std::size_t operator()(const Key& key) const {
                return std::hash<int>()(key.id);
            }
        };
        
        std::unordered_set<Key, Hash> my_set;
        my_set.insert({1});
        ```

### **3. 高级特性**

- **异构查找（C++20+）**：
    - 允许使用与键类型不同的对象进行查找（需定义 `is_transparent` 标记）。
    - 示例：

        ```cpp
        struct StringHash {
            using is_transparent = void;
            std::size_t operator()(const std::string_view sv) const {
                return std::hash<std::string_view>{}(sv);
            }
        };
        
        struct StringEqual {
            using is_transparent = void;
            bool operator()(const std::string_view a, const std::string_view b) const {
                return a == b;
            }
        };
        
        std::unordered_map<std::string, int, StringHash, StringEqual> umap;
        umap["banana"] = 2;
        auto it = umap.find("banana");  // 可使用 std::string_view 查找
        ```

---

## **三、性能对比与选择**

|**特性**|**关联容器（红黑树）**|**无序关联容器（哈希表）**|
|---|---|---|
|**查找/插入/删除**|O(log N)|平均 O(1)，最坏 O(N)|
|**有序性**|是（支持范围查询）|否|
|**内存开销**|每个节点存储指针和颜色属性|桶数组和链表/开放寻址的额外空间|
|**适用场景**|需要有序性或范围查询（如 `lower_bound`）|高效查找和插入，不关心顺序|
|**自定义类型支持**|需要重载比较运算符|需要提供哈希函数和相等比较函数|
|**扩容与迭代器失效**|插入/删除不导致迭代器失效|扩容可能导致所有迭代器失效|

### **性能测试示例**

- **插入**：
    - 小数据量（<256）：`vector` 和 `unordered_set` 性能相近。
    - 大数据量（>15000）：`unordered_set` 显著优于 `map` 和 `set`。
- **查找**：
    - 无序容器平均更快，但哈希冲突严重时可能退化为 O(N)。
- **遍历**：
    - 关联容器按顺序输出，无序容器输出顺序不确定。

---

## **四、注意事项**

1. **关联容器的键不可变**：
    
    - 直接修改键会破坏红黑树的有序性。需先删除再插入。
    - 示例：

        ```
        std::map<std::string, int> m;
        m["apple"] = 5;
        m["apple"].id = 10;  // 若键是结构体，修改会破坏有序性！
        ```

2. **无序容器的哈希冲突**：
    
    - 哈希函数设计不良可能导致大量冲突，影响性能。
    - 可通过调整 `max_load_factor` 和手动扩容优化：

        ```
        word_count.max_load_factor(0.5);  // 降低负载因子阈值
        word_count.rehash(1000);         // 手动调整桶数
        ```

3. **迭代器失效**：
    
    - **关联容器**：插入/删除不导致迭代器失效（除删除当前节点）。
    - **无序容器**：扩容会导致所有迭代器失效。
4. **选择容器的策略**：
    
    - **有序性需求**：使用关联容器（如 `map`/`set`）。
    - **快速查找**：使用无序容器（如 `unordered_map`/`unordered_set`）。
    - **范围查询**：仅关联容器支持（如 `lower_bound`/`upper_bound`）。

---

## **五、典型应用场景**

### **1. 关联容器**

- **字典**：统计文本中单词出现次数（按字母顺序输出）。

- **优先队列**：维护动态更新的排名（如排行榜）。

### **2. 无序关联容器**

- **缓存**：快速查找缓存项（如 LRU 缓存的辅助结构）。

    ```
    std::unordered_map<int, std::string> cache;
    cache[123] = "data";
    ```

- **去重**：快速判断元素是否存在（如 URL 去重）。

    ```
    std::unordered_set<std::string> visited_urls;
    if (!visited_urls.count(url)) {
        visited_urls.insert(url);
    }
	```

---

## **六、总结**

- **关联容器**（红黑树）：适合需要有序性、范围查询的场景，操作时间复杂度为 O(log N)。
- **无序关联容器**（哈希表）：适合高效查找和插入，平均时间复杂度为 O(1)，但元素无序。
- **选择依据**：根据是否需要有序性、数据规模、自定义类型的支持等因素权衡。
- **高级特性**：C++14/20 的异类查找和自定义哈希函数提升了灵活性，但需注意实现细节。

# 红黑树和哈希表

在 C++ 中，**红黑树**和**哈希表**是两种核心的数据结构，广泛应用于标准库的关联式容器（如 `std::map`、`std::set`）和无序容器（如 `std::unordered_map`、`std::unordered_set`）。以下从原理和实现细节两方面深入解析它们的工作机制。

---

## **一、红黑树的实现原理**

### **1. 红黑树的基本性质**

红黑树是一种自平衡的二叉搜索树（BST），通过颜色属性（红色或黑色）和特定规则维持树的平衡。其核心性质如下：

1. **每个节点要么是红色，要么是黑色。**
2. **根节点必须是黑色。**
3. **每个红色节点的子节点必须是黑色**（即不能有两个连续的红色节点）。
4. **从任意节点到其子树中所有 `NULL` 叶子节点的路径上，黑色节点的数量相同**（称为“黑色高度”）。

这些规则确保了红黑树的最长路径不超过最短路径的两倍，从而保证查找、插入和删除操作的时间复杂度为 **O(log N)**。

---

### **2. 红黑树的插入与调整**

插入新节点时，红黑树需要通过**变色**和**旋转**操作维持平衡。以下是关键步骤：

#### **插入规则**

1. 新节点默认为红色。
2. 如果父节点是黑色，直接插入即可。
3. 如果父节点是红色，则需要调整树的结构。

#### **调整情况**

根据叔叔节点（父节点的兄弟节点）的颜色和位置，分为三种情况：

1. **情况 1：叔叔节点存在且为红色**
    
    - **操作**：将父节点和叔叔节点变为黑色，祖父节点变为红色。
    - **效果**：解决连续红色问题，但祖父节点可能变成红色，需继续向上调整。
    - **示例**：

        ```
        // 假设 cur 是新插入的红色节点，p 是父节点，g 是祖父节点，u 是叔叔节点
        if (u != nullptr && u->_col == RED) {
            p->_col = BLACK;
            u->_col = BLACK;
            g->_col = RED;
            cur = g; // 继续向上调整
        }
        ```

2. **情况 2：叔叔节点不存在或为黑色（且新节点在父节点的内侧）**
    
    - **操作**：进行一次单旋（左旋或右旋）后变色。
    - **示例**（父节点是左孩子，新节点是右孩子）：

        ```
        if (cur == p->_right) {
            RotateL(p); // 左旋
            cur = p;
        }
        RotateR(g); // 右旋
        p->_col = BLACK;
        g->_col = RED;
        ```

3. **情况 3：叔叔节点不存在或为黑色（且新节点在父节点的外侧）**
    
    - **操作**：进行一次双旋（左右或右左）后变色。
    - **示例**（父节点是左孩子，新节点是右孩子）：

        ```
        RotateL(p); // 先左旋
        RotateR(g); // 再右旋
        cur->_col = BLACK;
        g->_col = RED;
        ```

#### **旋转操作**

- **左旋**（Left Rotation）：将右子树提升为父节点。
- **右旋**（Right Rotation）：将左子树提升为父节点。
    - 旋转操作会调整节点的父子关系，并保持红黑树的性质。

---

### **3. 红黑树的应用**

- **C++ 标准库中的关联式容器**（如 `std::map`、`std::set`）均基于红黑树实现。
- **优势**：支持有序遍历（通过中序遍历）、动态插入/删除高效。
- **缺点**：相比哈希表，查找效率略低（O(log N) vs O(1)）。

---

## **二、哈希表的实现原理**

### **1. 哈希表的核心思想**

哈希表通过**哈希函数**将键（Key）映射到存储桶（Bucket）中，实现快速查找。其核心步骤如下：

1. **哈希函数**：将键转换为整数索引（例如 `index = key % table_size`）。
2. **冲突解决**：当多个键映射到同一索引时，采用链地址法或开放寻址法处理冲突。

---

### **2. 哈希表的冲突解决方法**

#### **链地址法（Chaining）**

- 每个桶存储一个链表（或动态数组），冲突的键值对以链表形式存储。
- **优点**：实现简单，适合动态数据量。
- **缺点**：链表过长时查找效率下降。
- **C++ 标准库中的无序容器**（如 `std::unordered_map`）通常采用链地址法。

#### **开放寻址法（Open Addressing）**

- 冲突时，按一定规则（如线性探测、二次探测）寻找下一个空桶。
- **优点**：内存连续，缓存友好。
- **缺点**：需要动态扩容，删除操作复杂。
- **示例**（线性探测）：

    ```
    int index = hash(key);
    while (table[index] != EMPTY && table[index].key != key) {
        index = (index + 1) % table_size;
    }
    ```

---

### **3. 负载因子与扩容机制**

- **负载因子**（Load Factor）：哈希表中元素数量与桶数的比值（`load_factor = size / bucket_count`）。
- **扩容条件**：当负载因子超过阈值（如 1.0）时，哈希表会自动扩容（通常为原大小的 2 倍）。
- **扩容步骤**：
    1. 创建新桶数组（大小为原数组的 2 倍）。
    2. 重新计算所有键的哈希值，并插入到新桶中。
    3. 释放旧桶数组。
- **C++ 标准库实现**：`std::unordered_map` 的扩容策略类似 Java HashMap，使用 2 的幂次方大小的桶数组，并优化哈希函数计算。

---

### **4. 哈希表的应用**

- **C++ 标准库中的无序容器**（如 `std::unordered_map`、`std::unordered_set`）基于哈希表实现。
- **优势**：查找、插入、删除操作平均时间复杂度为 **O(1)**。
- **缺点**：不支持有序遍历，依赖良好的哈希函数设计。

---

## **三、红黑树与哈希表的对比**

| **特性**       | **红黑树**                     | **哈希表**          |
| ------------ | --------------------------- | ---------------- |
| **数据结构类型**   | 二叉搜索树                       | 数组 + 链表/开放寻址     |
| **查找时间复杂度**  | O(log N)                    | 平均 O(1)，最坏 O(N)  |
| **插入/删除复杂度** | O(log N)                    | 平均 O(1)，最坏 O(N)  |
| **是否支持有序遍历** | 是（中序遍历）                     | 否                |
| **内存开销**     | 每个节点需存储指针和颜色属性              | 桶数组和链表/开放寻址的额外空间 |
| **适用场景**     | 需要有序性或范围查询（如 `lower_bound`） | 高效查找和插入，不关心顺序    |

---

## **四、C++ 标准库中的实现示例**

### **1. 红黑树的模拟实现**

```
// 红黑树节点定义
struct RBTreeNode {
    int key;
    int value;
    RBTreeNode* left;
    RBTreeNode* right;
    RBTreeNode* parent;
    bool color; // true 表示红色，false 表示黑色
};

// 插入操作（简化版）
void insert(RBTreeNode*& root, int key, int value) {
    // 插入新节点并调整颜色和旋转
    // 具体调整逻辑参考上述插入与调整部分
}
```

### **2. 哈希表的模拟实现**

```
// 哈希表节点定义（链地址法）
struct HashNode {
    int key;
    int value;
    HashNode* next;
};

// 哈希表类定义
class HashTable {
private:
    std::vector<HashNode*> buckets;
    int size;
    int bucket_count;
    float load_factor_threshold = 1.0;

    // 哈希函数
    int hash(int key) { return key % bucket_count; }

public:
    HashTable(int initial_bucket_count = 16) 
        : bucket_count(initial_bucket_count), size(0) {
        buckets.resize(bucket_count, nullptr);
    }

    // 插入操作
    void insert(int key, int value) {
        if ((float)size / bucket_count >= load_factor_threshold) {
            resize();
        }
        int index = hash(key);
        HashNode* node = new HashNode{key, value, buckets[index]};
        buckets[index] = node;
        size++;
    }

    // 扩容操作
    void resize() {
        int new_bucket_count = bucket_count * 2;
        std::vector<HashNode*> new_buckets(new_bucket_count, nullptr);
        for (auto& bucket : buckets) {
            HashNode* current = bucket;
            while (current) {
                int new_index = current->key % new_bucket_count;
                HashNode* next = current->next;
                current->next = new_buckets[new_index];
                new_buckets[new_index] = current;
                current = next;
            }
        }
        buckets = std::move(new_buckets);
        bucket_count = new_bucket_count;
    }
};
```

---

## **五、总结**

- **红黑树**通过颜色属性和旋转操作维持平衡，适合需要有序遍历和动态插入/删除的场景。
- **哈希表**通过哈希函数和冲突解决策略实现高效查找，适合无序数据的快速访问。
- C++ 标准库根据需求选择不同的实现：`std::map`/`std::set` 使用红黑树，`std::unordered_map`/`std::unordered_set` 使用哈希表。
- 两者各有优劣，选择时需根据具体需求（如是否需要有序性、数据规模等）权衡。
