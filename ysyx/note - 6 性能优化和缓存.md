---
banner: "[[../300-以影像之/Vertin.jpg|../../300-以影像之/Vertin.jpg]]"
dateCreated: 2024-11-27
dateModified: 2025-06-02
---
# 讲义内容

对应学习讲义 B 阶段 B 3。

# Slides

# 缓存的验证

## ## 测试 vs. 证明
- 测试 - 判断**给定**的输入是否能运行正确
- 证明 - 判断**所有**的输入是否能运行正确
- DiffTest 属于测试
    - 你可能遇到过: cputest 都对, 但跑超级玛丽就会出错
- UVM 也属于测试
    - 不要迷信 UVM 的 100% 覆盖率报告

光靠测试无法证明 DUT 的正确性, 除非测试用例覆盖了所有输入。需要借助软件测试理论的等价类测试方法降低测试集大小。

## 求解器 - 把问题当作方程来解

在给定约束条件下寻找可行解的数学工具 (类似解方程组或线性规划)。

<a href=" https://github.com/Z3Prover/z3">Z3</a>是一个 SMT (Satisfiability Modulo Theories, 可满足性模理论) 求解器。<a href=" https://github.com/Z3Prover/z3/wiki#background">github wiki</a>

- 求解包含实数, 整数, 比特, 字符, 数组, 字符串等内容的命题是否成立
- 能将问题表达成一阶逻辑语言的某个子集, 就能让 SMT 求解器求解
- 广泛应用于定理自动证明, 程序分析, 程序验证和软件测试等领域
