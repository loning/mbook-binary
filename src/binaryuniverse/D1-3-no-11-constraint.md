# D1-3：no-11约束定义

## 定义概述

no-11约束是二进制编码系统中的关键限制条件，要求编码序列中不能出现连续的"11"模式。该约束为φ-表示系统的构造提供基础。

## 形式化定义

### 定义1.3（no-11约束）

对于二进制编码系统，no-11约束定义为：

$$
\text{No11Constraint}(\text{Encode}) \equiv \forall s \in S: \text{Encode}(s) \notin \text{Pattern}_{11}
$$

其中$\text{Pattern}_{11}$表示包含连续"11"子串的所有二进制串的集合：
$$
\text{Pattern}_{11} = \{w \in \{0,1\}^* : w = u \cdot 11 \cdot v \text{ for some } u,v \in \{0,1\}^*\}
$$

## 三种等价表述

### 表述1：正则表达式形式

满足no-11约束的字符串集合可表示为：
$$
\text{Valid}_{no11} = (0|10)^* \cdot (1|\varepsilon)
$$

### 表述2：递归生成规则

$$
\begin{aligned}
\text{ValidString} &::= \varepsilon \\
&\quad | \; 0 \cdot \text{ValidString} \\
&\quad | \; 10 \cdot \text{ValidString} \\
&\quad | \; 1
\end{aligned}
$$

### 表述3：有限状态自动机

状态集合：$Q = \{q_0, q_1, q_{reject}\}$

转移函数：
$$
\delta(q_0, 0) = q_0, \quad \delta(q_0, 1) = q_1
$$
$$
\delta(q_1, 0) = q_0, \quad \delta(q_1, 1) = q_{reject}
$$

接受状态：$F = \{q_0, q_1\}$

## 约束的基本性质

### 性质1.3.1（前缀封闭性）

no-11约束具有前缀封闭性：
$$
s \in \text{Valid}_{no11} \land p \text{ is prefix of } s \Rightarrow p \in \text{Valid}_{no11}
$$

### 性质1.3.2（扩展规则）

对于满足约束的字符串$s$：
- 总是可以添加"0"：$s0 \in \text{Valid}_{no11}$
- 仅当$s$不以"1"结尾时可以添加"1"：$s1 \in \text{Valid}_{no11} \iff s \text{ does not end with } 1$

### 性质1.3.3（计数序列）

长度为$n$的满足no-11约束的字符串数量记为$F_{n+2}$，其中$F_k$是第$k$个Fibonacci数：
$$
F_0 = 0, \quad F_1 = 1, \quad F_k = F_{k-1} + F_{k-2} \text{ for } k \geq 2
$$

递归关系：
$$
N(0) = 1, \quad N(1) = 2, \quad N(n) = N(n-1) + N(n-2) \text{ for } n \geq 2
$$

### 性质1.3.4（信息容量）

no-11约束下的渐近信息容量为：
$$
C_{no11} = \lim_{n \to \infty} \frac{\log_2 F_{n+2}}{n} = \log_2 \phi
$$
其中$\phi = \frac{1+\sqrt{5}}{2}$是黄金比例。

## 与φ-表示的关系

### 双射对应

存在双射映射：
$$
\text{Valid}_{no11} \leftrightarrow \text{Zeckendorf representations}
$$

其中Zeckendorf表示是使用非连续Fibonacci数的唯一表示法。

### φ-编码函数

对于无"11"的二进制串$b_n b_{n-1} \cdots b_1$：
$$
\phi\text{-value}(b_n b_{n-1} \cdots b_1) = \sum_{i=1}^n b_i \cdot F_i
$$

## 算法复杂性

### 验证算法

检验字符串是否满足no-11约束：
- **时间复杂度**：$O(n)$
- **空间复杂度**：$O(1)$

### 枚举算法

生成所有长度为$n$的满足约束的字符串：
- **时间复杂度**：$O(F_{n+2})$
- **空间复杂度**：$O(n \cdot F_{n+2})$

## 应用范围

no-11约束在以下场景中发挥作用：
- 前缀码构造
- 信息压缩算法
- 自指系统的稳定编码
- 黄金比例进制系统

## 符号约定

- $u \cdot v$：字符串连接
- $\varepsilon$：空字符串
- $|s|$：字符串$s$的长度
- $F_k$：第$k$个Fibonacci数

---

**依赖关系**：
- **基于**：D1-2 (二进制表示定义)
- **支持**：D1-8 (φ-表示定义)

**引用文件**：
- 引理L1-4将证明no-11约束的最优性
- 引理L1-5将证明Fibonacci结构的涌现
- 定理T2-3将建立约束优化定理

**形式化特征**：
- **类型**：定义 (Definition)
- **编号**：D1-3
- **状态**：完整形式化定义
- **验证**：符合严格定义标准

**注记**：本定义提供no-11约束的精确数学表述，所有最优性证明和必然性推导将在相应的引理和定理文件中完成。