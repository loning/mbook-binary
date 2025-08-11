# P1: 二元区分命题

## 依赖关系
- 基于: [D1-1-self-referential-completeness.md](D1-1-self-referential-completeness.md), [L1-2-binary-necessity.md](L1-2-binary-necessity.md)
- 类型: 基础命题

## 命题陈述

**命题1** (二元区分): 任何区分的最小形式是二元的。

形式化表述：
$$
\forall \text{Distinction} \exists \text{Binary}: \text{Distinction} \equiv \text{Binary}
$$

## 证明

### 步骤1：区分的定义

区分要求存在至少两个不同的状态：
$$
\text{Distinction} \Rightarrow \exists (A, B): A \neq B
$$

### 步骤2：最小性论证

任何区分都可以映射到二元区分$(0,1)$：
$$
\text{Map}: \{A, B\} \to \{0, 1\}
$$

### 步骤3：等价性

更多元的区分可以分解为二元区分的组合：
$$
\{A, B, C\} \equiv \{(0,0), (0,1), (1,0)\}
$$

∎

## 应用

### 应用1：逻辑基础

建立逻辑系统的二元基础。

### 应用2：信息论

证明bit作为信息单位的基础性。

---

**形式化特征**：
- **类型**：命题 (Proposition)
- **编号**：P1
- **状态**：完整证明
- **验证**：符合严格推导标准