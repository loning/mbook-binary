# D1-2：二进制表示定义

## 定义概述

二进制表示是自指完备系统中信息编码的基础形式。该定义建立在D1-1自指完备性的基础上，为后续的约束条件和φ-表示系统提供数学基础。

## 形式化定义

### 定义1.2（二进制表示）

对于自指完备系统S，其二进制表示定义为：

$$
\text{BinaryRep}(S) \equiv \exists \text{Encode}: S \to \{0,1\}^* \text{ 满足四个条件：}
$$

**条件1：编码完整性（唯一可解码性）**
$$
\forall s_1, s_2 \in S: s_1 \neq s_2 \Rightarrow \text{Encode}(s_1) \neq \text{Encode}(s_2)
$$
编码函数在S上是单射的。

**条件2：前缀自由性（即时可解码）**
$$
\forall s_1, s_2 \in S: \text{Encode}(s_1) \text{ 不是 } \text{Encode}(s_2) \text{ 的前缀}
$$
任何码字都不是其他码字的前缀。

**条件3：自嵌入性（自指完备性要求）**
$$
\text{Encode} \in \text{Domain}(\text{Encode}) \land \text{Encode}(\text{Encode}) \in \text{Range}(\text{Encode})
$$
编码函数能够编码自身。

**条件4：编码封闭性**
$$
\text{Encode}(s) \in \{0,1\}^* \subseteq \mathcal{L} \subseteq \mathcal{S}
$$
编码结果是形式语言的元素，也是可能的系统状态。

## 基本性质

### 性质1.2.1（编码空间）

二进制字符串空间$\{0,1\}^*$包括：
- 空串：$\varepsilon$
- 有限长度串：$\{0,1\}^n$ for all $n \geq 1$
- 所有有限二进制串的集合

### 性质1.2.2（解码存在性）

若$\text{Encode}$满足定义1.2的四个条件，则存在解码函数：
$$
\text{Decode}: \text{Range}(\text{Encode}) \to S
$$
使得$\text{Decode} \circ \text{Encode} = \text{id}_S$

### 性质1.2.3（长度下界）

对于包含$|S|$个状态的系统，任何满足条件1-2的编码必须满足：
$$
\max_{s \in S} |\text{Encode}(s)| \geq \lceil \log_2 |S| \rceil
$$

### 性质1.2.4（自指对偶性）

在二进制表示中，符号0和1构成完全对偶：
$$
0 \equiv \neg 1, \quad 1 \equiv \neg 0
$$
这种对偶性反映了自指系统的内在对称性。

## 符号约定

- $\{0,1\}^*$：所有有限长度二进制串的集合
- $|s|$：字符串s的长度
- $\varepsilon$：空字符串
- $s_1 \circ s_2$：字符串连接
- $\text{prefix}(s_1, s_2)$：$s_1$是$s_2$的前缀

## 与其他定义的关系

### 依赖关系

- **基于**：D1-1 (自指完备性定义)
- **支持**：D1-3 (no-11约束定义)，D1-8 (φ-表示定义)

### 引用文件

- 引理L1-2将证明二进制基底的必然性
- 定理T2-2将证明二进制表示的唯一性
- 推论C2-1将建立二进制同构性质

---

**形式化特征**：
- **类型**：定义 (Definition)
- **编号**：D1-2
- **状态**：完整形式化定义
- **验证**：符合严格定义标准

**注记**：本定义提供二进制表示的形式化框架，所有必然性证明和推导将在相应的引理和定理文件中完成。