# BDAG 文件命名规范

## 文件名格式

### 完整格式
```
[层级代码][序号]__[节点名]__[操作类型]__FROM__[输入节点信息]__TO__[输出类型]__ATTR__[属性标签].md
```

### 格式说明

#### 1. 层级代码 (1字符)
```
A = Axiom Layer      (公理层)
B = Basic Layer      (基础操作层) 
C = Composite Layer  (复合操作层)
E = Emergent Layer   (涌现操作层)
U = Unified Layer    (统一应用层)
```

#### 2. 序号 (3位数字)
```
001-999: 每层内的唯一序号，按创建顺序递增
例如：A001, A002, B101, B102, C201...
```

#### 3. 节点名 (PascalCase)
```
- 使用PascalCase命名法（首字母大写的驼峰命名）
- 描述张量的核心概念或实体
- 避免使用缩写，保持语义清晰
- 例如：SelfReference, Phi, ObserverDifferentiation, QuantumState
```

#### 4. 操作类型 (UPPERCASE)
```
DEFINE     = 定义基础张量或常数
APPLY      = 将函数应用到单个张量
TRANSFORM  = 张量的结构变换
COMBINE    = 多个张量的组合操作
EMERGE     = 新属性或现象的涌现
DERIVE     = 从现有结构推导新关系
```

#### 5. 输入节点信息 (FROM部分)
```
FROM__[输入1]__[输入2]__...

基础输入类型：
- Axiom        : 来自公理
- Constant     : 来自数学常数
- Sequence     : 来自数学序列
- Constraint   : 来自约束条件

节点输入格式：
- A001_SelfReference     : 单一输入节点
- B101_Entropy__B103_Encode : 多输入节点（双下划线分隔）
```

#### 6. 输出类型 (TO部分)
```
TO__[OutputType]

输出类型命名：
- 使用PascalCase
- 明确描述输出张量的类型
- 例如：SelfRefTensor, EntropyTensor, QuantumState, MetricSpace
```

#### 7. 属性标签 (ATTR部分)
```
ATTR__[属性1]_[属性2]_...

常见属性：
数学属性：
- Recursive, Irreversible, Monotonic, Continuous, Discrete
- Unitary, Hermitian, Symmetric, Antisymmetric
- Complete, Compact, Connected, Bounded

物理属性：
- Entropic, Causal, Quantum, Classical, Relativistic
- Conservative, Dissipative, Coherent, Decoherent
- Local, Nonlocal, Gauge, Topological

信息属性：
- Compressed, Encoded, Measured, Observed
- Correlated, Entangled, Separable, Mixed
```

## 命名示例

### A层示例（公理层）
```
A001__SelfReference__DEFINE__FROM__Axiom__TO__SelfRefTensor__ATTR__Recursive_Entropic.md
A002__Phi__DEFINE__FROM__Constant__TO__GoldenRatio__ATTR__Irrational_Algebraic.md
A003__Fibonacci__DEFINE__FROM__Sequence__TO__FibonacciTensor__ATTR__Recursive_Growth.md
A004__No11Constraint__DEFINE__FROM__Constraint__TO__BinaryConstraint__ATTR__Forbidden_Consecutive.md
```

### B层示例（基础操作层）
```
B101__EntropyIncrease__APPLY__FROM__A001_SelfReference__TO__EntropyTensor__ATTR__Monotonic_Irreversible.md
B102__TimeEmergence__APPLY__FROM__A001_SelfReference__TO__TimeTensor__ATTR__Quantum_Discrete.md
B103__PhiEncoding__APPLY__FROM__A002_Phi__TO__ZeckendorfSystem__ATTR__Unique_Optimal.md
B106__ObserverDifferentiation__TRANSFORM__FROM__A001_SelfReference__TO__ObserverTensor__ATTR__Separated_Measuring.md
```

### C层示例（复合操作层）
```
C201__InformationEntropy__COMBINE__FROM__B101_EntropyIncrease__B103_PhiEncoding__TO__InfoTensor__ATTR__Quantized_Compressed.md
C202__ZeckendorfMetric__COMBINE__FROM__B103_PhiEncoding__B105_BinarySpace__TO__MetricSpace__ATTR__Complete_Contracted.md
C203__QuantumState__COMBINE__FROM__B106_ObserverDifferentiation__B103_PhiEncoding__TO__QuantumTensor__ATTR__Superposed_Entangled.md
```

### E层示例（涌现操作层）
```
E301__SpacetimeGeometry__EMERGE__FROM__C202_ZeckendorfMetric__C203_QuantumState__TO__SpacetimeTensor__ATTR__Curved_Quantized.md
E302__MeasurementCollapse__EMERGE__FROM__C203_QuantumState__B106_ObserverDifferentiation__TO__CollapseTensor__ATTR__Stochastic_Irreversible.md
```

## 验证规则

### 1. 语法验证
- 文件名必须完全匹配指定格式
- 所有分隔符必须正确使用
- 层级代码和序号必须有效

### 2. 语义验证
- 操作类型必须与层级相符
- 输入节点必须存在且在较低层级
- 输出类型必须与操作类型一致

### 3. DAG验证
- 不能存在循环依赖
- 输入节点的层级必须低于当前节点
- 序号必须在层级内唯一

## 正则表达式

### 完整验证正则
```regex
^([ABCEU])(\d{3})__([A-Z][a-zA-Z0-9]*)__([A-Z]+)__FROM__((?:[A-Z]\d{3}_[A-Z][a-zA-Z0-9]*(?:__[A-Z]\d{3}_[A-Z][a-zA-Z0-9]*)*)|(?:Axiom|Constant|Sequence|Constraint))__TO__([A-Z][a-zA-Z0-9]*)__ATTR__([A-Z][a-zA-Z]*(?:_[A-Z][a-zA-Z]*)*)\.md$
```

### 分组验证
```regex
层级代码: ^[ABCEU]$
序号:     ^\d{3}$
节点名:   ^[A-Z][a-zA-Z0-9]*$
操作类型: ^(DEFINE|APPLY|TRANSFORM|COMBINE|EMERGE|DERIVE)$
输出类型: ^[A-Z][a-zA-Z0-9]*$
属性:     ^[A-Z][a-zA-Z]*(?:_[A-Z][a-zA-Z]*)*$
```

## 错误示例与修正

### 错误示例1：格式不规范
```
❌ 错误：T203_Quantum_Combine_T106_T103.md
✅ 正确：C203__QuantumState__COMBINE__FROM__B106_ObserverDifferentiation__B103_PhiEncoding__TO__QuantumTensor__ATTR__Superposed_Entangled.md
```

### 错误示例2：分隔符错误
```
❌ 错误：A001-SelfReference-DEFINE-FROM-Axiom.md
✅ 正确：A001__SelfReference__DEFINE__FROM__Axiom__TO__SelfRefTensor__ATTR__Recursive_Entropic.md
```

### 错误示例3：层级不匹配
```
❌ 错误：A101__Entropy__APPLY__FROM__A001_SelfReference...  (A层不应有APPLY操作)
✅ 正确：B101__EntropyIncrease__APPLY__FROM__A001_SelfReference...
```

### 错误示例4：依赖关系错误
```
❌ 错误：B101__Entropy__APPLY__FROM__C201_InfoEntropy...  (不能依赖更高层级)
✅ 正确：B101__EntropyIncrease__APPLY__FROM__A001_SelfReference...
```

## 最佳实践

### 1. 命名一致性
- 相同概念在不同文件中使用相同名称
- 避免同义词混用
- 保持命名风格统一

### 2. 语义清晰性  
- 节点名应准确反映其数学或物理含义
- 避免过于抽象或模糊的命名
- 输出类型应明确描述结果

### 3. 属性标签使用
- 选择最重要的2-4个属性
- 按重要性排序
- 避免冗余或显而易见的属性

### 4. 版本控制
- 文件名一旦确定不应随意更改
- 如需修改应创建新版本文件
- 保持向后兼容性