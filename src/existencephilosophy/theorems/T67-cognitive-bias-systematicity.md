# T67：认知偏误系统性定理 (Cognitive Bias Systematicity Theorem)  

**定理陈述**: 认知偏误不是认知系统的缺陷，而是在资源有限环境中的适应性特征  

## 推导依据
- **T13 (方法优化定理)**: 系统在约束条件下优化处理方法
- **T14 (认知涌现定理)**: 认知特征从底层约束涌现

## 依赖理论
- T13：有限资源下的优化策略
- T14：认知特征的涌现机制

## 形式化表述  
```  
设B为认知偏误集合，E为环境，R为资源约束，定义：

偏误作为适应性启发式：
Cognitive_Biases = Adaptive_Heuristics(Resource_Constraints, Environmental_Structure)  
∀偏误b ∈ B: Utility(b, E_ancestral) > 0 [在祖先环境中有正效用]

生态理性：
Ecological_Rationality = Performance(Heuristic, Environment)
Bounded_Rationality ≠ Irrationality
Fast_and_Frugal > Slow_and_Optimal (in most contexts)

偏误系统性：
Pattern(Biases) = f(Cognitive_Architecture, Environmental_Regularities)
Predictability(Bias) = High [偏误具有系统可预测性]
```  

## 严格证明  

### 前提引入
1. **P1 (T13)**: 系统优化受资源约束 Optimization → Resource_Bounded
2. **P2 (T14)**: 认知模式从约束涌现 Cognitive_Patterns ← Constraints
3. **P3**: 计算资源有限 Computational_Resources < ∞

### 推导步骤1：资源约束的必然影响
从P3和计算复杂性：
- 时间约束：Decision_Time < Response_Deadline
- 注意力限制：Attention_Capacity ≈ 7±2 items
- 记忆限制：Working_Memory < Long_Term_Memory < ∞

最优化问题：
```
Optimal_Decision = argmax Expected_Utility
但计算Optimal需要：
- 完整信息：Information = Complete (不可能)
- 无限计算：Computation = Unlimited (不现实)
- 完美记忆：Memory = Perfect (不存在)
∴ 需要满意解而非最优解
```

### 推导步骤2：启发式的适应性价值
从P1（资源优化）：
- 启发式减少计算：Heuristic → Reduced_Computation
- 速度优于精度：Speed > Accuracy (在多数情境)
- 简单规则稳健：Simple_Rules → Robust_Performance

具体启发式分析：
```
可得性启发式：
Recent/Vivid → Higher_Probability_Estimate
适应价值：快速评估常见风险

代表性启发式：
Similarity → Category_Membership
适应价值：快速模式识别

锚定效应：
Initial_Value → Reference_Point
适应价值：在不确定中建立基准
```

### 推导步骤3：进化环境的塑造作用
从进化心理学：
- 认知在特定环境进化：Cognition_Evolved_in_EEA
- EEA（进化适应环境）≠ 现代环境
- 偏误反映祖先环境规律：Biases → Ancestral_Regularities

进化逻辑：
```
在EEA中：
- 错误拒绝（False_Negative）代价 >> 错误接受（False_Positive）
- 例：将树枝误认为蛇 vs 将蛇误认为树枝
∴ 演化出过度检测偏误（Hyperactive_Agency_Detection）

生存压力：
Fast_Decision + 70%_Accuracy > Slow_Decision + 100%_Accuracy
因为：Predator_Won't_Wait
```

### 推导步骤4：偏误的系统性组织
从P2（模式涌现）：
- 偏误非随机分布：Bias_Distribution ≠ Random
- 遵循认知架构：Biases_Follow_Architecture
- 可预测和一致：Predictable + Consistent

系统性证据：
```
双系统组织：
System1_Biases: 快速、自动、直觉
System2_Corrections: 缓慢、受控、分析

领域特异性：
Social_Biases: 面孔识别、意图归因
Spatial_Biases: 距离判断、方向偏好
Temporal_Biases: 现时偏好、规划谬误

文化普遍性：
Core_Biases = Universal_Across_Cultures
表明深层认知架构的共性
```

实验验证：
- Kahneman-Tversky研究程序：系统映射偏误
- Gigerenzer生态理性：环境匹配时偏误变优势
- 计算模型：偏误emerge从理性代理+资源限制

∴ 认知偏误是认知系统在资源约束下的适应性设计特征 □  