---
dateCreated: 2025-05-31
dateModified: 2025-05-31
---
# 深入理解计算机系统

<a href=" https://fengmuzi2003.gitbook.io/csapp3e">重点导读</a>

# A Tour of Computer Systems

## 1.9 Important Themes

### 1.9.1 Amdahl's Law

The main idea is that when we speed up one part of a system, the effect on the overall system performance depends on both how significant this part was and how much it sped up. Consider a system in which executing some application requires time $T_{old}$ . Suppose some part of the system requires a fraction α of this time, and that we improve its performance by a factor of $k$.

The overall execution time would thus be

$$
T_{new} = (1-\alpha)T_{old}+\alpha T_{old}/k=T_{old}[(1-\alpha)+\alpha/k]x
$$

From this, we can compute the speedup as $S=T_{old}/T_{new}=\frac{1}{(1-\alpha)+\alpha/k}$.

This is the major insight of Amdahl’s law—to significantly speed up the entire system, we must improve the speed of a very large fraction of the overall system.
# 2 Representing and Manipulating Information


