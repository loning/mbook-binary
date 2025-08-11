# C1-3: 信息密度推论

## 推论陈述

**推论 C1-3**（信息密度推论）：φ-表示系统在 no-11 约束条件下达到最大信息密度。

## 形式化表述

设 $\mathcal{S}_n$ 是长度为 $n$ 的满足 no-11 约束的二进制序列集合。则信息密度 $\rho_n$ 满足：

$$
\rho_n = \frac{\log_2 |\mathcal{S}_n|}{n} \to \log_2 \varphi \text{ as } n \to \infty
$$
其中 $\varphi = \frac{1+\sqrt{5}}{2}$ 是黄金比例。

## 证明

**证明**：

1. **序列计数**：
   - 定义 $F_n$ 为长度为 $n$ 的满足 no-11 约束的序列数
   - 由 L1-5（斐波那契涌现引理），$F_n$ 遵循修正斐波那契递推关系
   - $F_n = F_{n-1} + F_{n-2}$，其中 $F_1 = 2$, $F_2 = 3$

2. **渐近行为**：
   - 由斐波那契数列的性质，$F_n \sim \frac{\varphi^n}{\sqrt{5}}$
   - 因此 $|\mathcal{S}_n| = F_n \sim \frac{\varphi^n}{\sqrt{5}}$

3. **信息密度计算**：
   - $\rho_n = \frac{\log_2 F_n}{n}$
   - $\lim_{n \to \infty} \rho_n = \lim_{n \to \infty} \frac{\log_2(\varphi^n/\sqrt{5})}{n}$
   - $= \lim_{n \to \infty} \frac{n \log_2 \varphi - \log_2 \sqrt{5}}{n}$
   - $= \log_2 \varphi$

4. **最大性证明**：
   - 考虑任意其他约束 $\mathcal{C}$
   - 如果 $\mathcal{C}$ 比 no-11 更严格，则 $|\mathcal{S}_n^{\mathcal{C}}| < |\mathcal{S}_n|$
   - 如果 $\mathcal{C}$ 比 no-11 更宽松，则违反了 L1-3（约束必要性引理）
   - 因此 no-11 约束下的信息密度是最大的

5. **熵的观点**：
   - 定义熵：$H_n = \log_2 |\mathcal{S}_n|$
   - 熵密度：$h_n = \frac{H_n}{n} = \rho_n$
   - $\lim_{n \to \infty} h_n = \log_2 \varphi$

6. **与标准二进制的比较**：
   - 标准二进制：$\rho_{\text{binary}} = \log_2 2 = 1$
   - φ-表示：$\rho_\varphi = \log_2 \varphi \approx 0.694$
   - 虽然 $\rho_\varphi < 1$，但 φ-表示具有结构优势

∎

## 物理意义

此推论揭示了：
- 黄金比例在信息理论中的基础地位
- 约束条件与信息密度之间的权衡关系
- 自然系统的信息编码倾向

## 应用价值

1. **信道容量**：约束信道的容量上界
2. **编码理论**：结构化编码的设计
3. **复杂系统**：自组织系统的信息特征

## 关联定理

- 依赖于：L1-3, L1-5, T2-10, C1-1, C1-2
- 应用于：C2-1（观测效应推论）