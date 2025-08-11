# D1-8：φ-表示定义

## 定义概述

φ-表示系统是基于Fibonacci数列的位值编码系统，对应于满足no-11约束的二进制串与正整数的Zeckendorf表示之间的双射关系。该系统为自指完备系统提供最优编码。

## 形式化定义

### 定义1.8（φ-表示系统）

φ-表示系统是一个四元组：
$$
\Phi\text{-System} = (\mathcal{F}, \mathcal{B}, \text{encode}_\phi, \text{decode}_\phi)
$$

其中：
- $\mathcal{F} = \{F_n\}_{n=1}^{\infty}$：Fibonacci数列
- $\mathcal{B}$：满足no-11约束的二进制串集合
- $\text{encode}_\phi$：编码函数
- $\text{decode}_\phi$：解码函数

## Fibonacci数列定义

### 用于φ-表示的Fibonacci数列

为了保证与正整数的双射，我们使用修改的Fibonacci序列：

$$
F_1 = 1, \quad F_2 = 2, \quad F_n = F_{n-1} + F_{n-2} \text{ for } n \geq 3
$$

序列开始为：$1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...$

### 索引约定

$$
\mathcal{F} = \{F_1=1, F_2=2, F_3=3, F_4=5, F_5=8, F_6=13, ...\}
$$

**注意**：这与标准Fibonacci数列（从1,1开始）略有不同，但对于建立双射是必要的。

### Fibonacci数列的性质

**递归关系**：$F_n = F_{n-1} + F_{n-2}$

**Binet公式**：
$$
F_n = \frac{\phi^n - \psi^n}{\sqrt{5}}
$$
其中$\phi = \frac{1+\sqrt{5}}{2}$，$\psi = \frac{1-\sqrt{5}}{2}$

**渐近行为**：
$$
F_n \sim \frac{\phi^n}{\sqrt{5}} \text{ as } n \to \infty
$$

## 编码函数

### 二进制串到正整数

对于满足no-11约束的二进制串$b = b_n b_{n-1} \cdots b_1$：
$$
\text{encode}_\phi(b) = \sum_{i=1}^n b_i \cdot F_i
$$

其中$b_i \in \{0,1\}$且不存在相邻的1。

### 编码示例

- $\text{encode}_\phi(1) = 1 \cdot F_1 = 1$
- $\text{encode}_\phi(10) = 1 \cdot F_2 + 0 \cdot F_1 = 2$
- $\text{encode}_\phi(100) = 1 \cdot F_3 = 3$
- $\text{encode}_\phi(101) = 1 \cdot F_3 + 0 \cdot F_2 + 1 \cdot F_1 = 3 + 1 = 4$
- $\text{encode}_\phi(1000) = 1 \cdot F_4 = 5$
- $\text{encode}_\phi(1010) = 1 \cdot F_4 + 0 \cdot F_3 + 1 \cdot F_2 + 0 \cdot F_1 = 5 + 2 = 7$

## 解码函数

### 正整数到二进制串

对于正整数$n$，解码过程使用贪心算法：

**算法1.8.1（φ-解码算法）**
```
Input: 正整数 n
Output: 满足no-11约束的二进制串

1. 初始化：result = "", remaining = n
2. 找到最大的 k 使得 F_k ≤ remaining
3. While remaining > 0:
   a. 从位置 k 到 1：
      - 如果 F_i ≤ remaining：
        * result[i] = '1'
        * remaining -= F_i
      - 否则：
        * result[i] = '0'
      - i -= 1
4. Return result
```

### 解码示例

- $\text{decode}_\phi(1) = 1$ （使用F₁=1）
- $\text{decode}_\phi(2) = 10$ （使用F₂=2）
- $\text{decode}_\phi(3) = 100$ （使用F₃=3）
- $\text{decode}_\phi(4) = 101$ （使用F₃+F₁=3+1）
- $\text{decode}_\phi(5) = 1000$ （使用F₄=5）
- $\text{decode}_\phi(6) = 1001$ （使用F₄+F₁=5+1）
- $\text{decode}_\phi(7) = 1010$ （使用F₄+F₂=5+2）

## Zeckendorf表示

### Zeckendorf定理

**定理（Zeckendorf，1972）**：每个正整数都有唯一的表示为不连续Fibonacci数的和。

形式化表述：
$$
\forall n \in \mathbb{Z}^+: \exists! I \subset \mathbb{N}: n = \sum_{i \in I} F_i \text{ 且 } \forall i,j \in I: |i-j| \geq 2
$$

### φ-表示与Zeckendorf的等价性

φ-表示系统与Zeckendorf表示建立双射：
$$
\{b \in \mathcal{B}\} \leftrightarrow \{\text{Zeckendorf representations}\} \leftrightarrow \mathbb{Z}^+
$$

## 系统性质

### 性质1.8.1（双射性）

编码和解码函数构成双射：
$$
\text{decode}_\phi \circ \text{encode}_\phi = \text{id}_{\mathcal{B}}
$$
$$
\text{encode}_\phi \circ \text{decode}_\phi = \text{id}_{\mathbb{Z}^+}
$$

### 性质1.8.2（唯一性）

每个正整数有唯一的φ-表示：
$$
\forall n \in \mathbb{Z}^+: \exists! b \in \mathcal{B}: \text{encode}_\phi(b) = n
$$

### 性质1.8.3（保序性）

φ-表示保持数值的大小关系：
$$
n_1 < n_2 \Leftrightarrow \text{encode}_\phi(\text{decode}_\phi(n_1)) < \text{encode}_\phi(\text{decode}_\phi(n_2))
$$

### 性质1.8.4（紧致性）

φ-表示是最紧致的满足no-11约束的表示：
$$
|\text{decode}_\phi(n)| = \lfloor \log_\phi n \rfloor + 1
$$

## 算术运算

### 加法运算

φ-表示中的加法：
$$
\text{add}_\phi(b_1, b_2) = \text{decode}_\phi(\text{encode}_\phi(b_1) + \text{encode}_\phi(b_2))
$$

### 进位规则

φ-表示的进位规则基于Fibonacci恒等式：
$$
F_i + F_{i+1} = F_{i+2}
$$

### 比较运算

φ-表示的字典序对应数值大小：
$$
b_1 <_{\text{lex}} b_2 \Leftrightarrow \text{encode}_\phi(b_1) < \text{encode}_\phi(b_2)
$$

## 信息容量

### 渐近信息容量

φ-表示系统的信息容量为：
$$
C_\phi = \lim_{n \to \infty} \frac{\log_2 F_{n+2}}{n} = \log_2 \phi \approx 0.694
$$

### 效率比较

与标准二进制的效率比：
$$
\eta_\phi = \frac{C_\phi}{1} = \log_2 \phi \approx 69.4\%
$$

### 压缩性能

φ-表示在某些结构化数据上优于标准二进制：
- 斐波那契数：压缩率接近100%
- 递归结构：显著压缩优势
- 自相似数据：自然适配

## 与黄金比例的关系

### 黄金比例的自指性

黄金比例满足自指方程：
$$
\phi = 1 + \frac{1}{\phi}
$$

### φ-表示中的黄金比例

φ-表示系统的容量直接关联黄金比例：
$$
\text{capacity} = \log_2 \phi
$$

### 连分数表示

黄金比例的连分数：
$$
\phi = 1 + \cfrac{1}{1 + \cfrac{1}{1 + \cfrac{1}{1 + \cdots}}}
$$

这种无限递归结构体现了自指完备性。

## 计算复杂度

### 编码复杂度

将二进制串编码为整数：
- **时间复杂度**：$O(n)$
- **空间复杂度**：$O(1)$

### 解码复杂度

将整数解码为二进制串：
- **时间复杂度**：$O(\log n)$
- **空间复杂度**：$O(\log n)$

### 运算复杂度

φ-表示中的算术运算：
- **加法**：$O(\log \max(a,b))$
- **比较**：$O(\min(|a|,|b|))$

## 应用领域

### 数据压缩

φ-表示在以下场景中优于标准编码：
- 递归数据结构
- 分形图像
- 自相似信号

### 错误检测

φ-表示的冗余性提供天然的错误检测能力：
- 非法模式（包含"11"）立即可检测
- 局部错误不会传播

### 密码学

φ-表示的数学性质可用于：
- 密钥生成
- 随机数生成
- 零知识证明

## 符号约定

- $\mathcal{F}$：Fibonacci数列集合
- $\mathcal{B}$：无"11"二进制串集合
- $\phi$：黄金比例
- $F_n$：第n个Fibonacci数
- $\mathbb{Z}^+$：正整数集合

---

**依赖关系**：
- **基于**：D1-2 (二进制表示定义)，D1-3 (no-11约束定义)
- **完成**：D1系列基础定义的最后一个

**引用文件**：
- 引理L1-6将证明φ-表示的唯一性
- 定理T2-4将建立φ-表示系统定理
- 定理T2-5将建立Zeckendorf对应定理

**形式化特征**：
- **类型**：定义 (Definition)
- **编号**：D1-8
- **状态**：完整形式化定义
- **验证**：符合严格定义标准

**注记**：本定义完成了D1系列基础定义，提供了φ-表示系统的完整数学框架。φ-表示的最优性证明和与自指完备系统的深层联系将在相应的引理和定理文件中建立。