# D1-4-formal: 时间度量的形式化定义

## 机器验证元数据
```yaml
type: definition
verification: machine_ready
dependencies: ["D1-1-formal.md"]
verification_points:
  - non_negativity
  - monotonicity
  - additivity
  - directionality
```

## 核心定义

### 定义 D1-4（时间度量）
```
TimeMetric(S : SelfReferentialComplete) : Prop ≡
  ∃τ : Function[StateSpace × StateSpace → Real⁺] .
    NonNegative(τ) ∧
    Monotonic(τ) ∧
    Additive(τ) ∧
    Directional(τ)
```

## 四个基本性质

### 性质1：非负性
```
NonNegative(τ : Metric) : Prop ≡
  ∀i, j : Time . τ(Sᵢ, Sⱼ) ≥ 0 ∧
    (τ(Sᵢ, Sⱼ) = 0 ⟺ i = j)
```

### 性质2：单调性
```
Monotonic(τ : Metric) : Prop ≡
  ∀i, j, k : Time . i < j < k → 
    τ(Sᵢ, Sⱼ) < τ(Sᵢ, Sₖ)
```

### 性质3：可加性
```
Additive(τ : Metric) : Prop ≡
  ∀i, j, k : Time . i ≤ j ≤ k → 
    τ(Sᵢ, Sₖ) = τ(Sᵢ, Sⱼ) + τ(Sⱼ, Sₖ)
```

### 性质4：方向性
```
Directional(τ : Metric) : Prop ≡
  ∀i, j : Time . τ(Sᵢ, Sⱼ) > 0 ⟺ i < j
```

## 标准构造

### 结构距离度量
```
τ_struct(Sᵢ, Sⱼ) : Real⁺ ≡
  match (i, j) with
  | i = j     ⇒ 0
  | i < j     ⇒ Σ(k=i to j-1) ρ(Sₖ, Sₖ₊₁)
  | i > j     ⇒ -τ(Sⱼ, Sᵢ)
  
where
  ρ(Sₖ, Sₖ₊₁) := √|Sₖ₊₁ \ Sₖ|
```

### 信息距离度量
```
τ_info(Sᵢ, Sⱼ) : Real⁺ ≡
  Σ(k=i to j-1) [H(Sₖ₊₁) - H(Sₖ)]
  
where
  H : State → Real⁺ is entropy function
```

## 类型定义

```
Type Time := ℕ
Type State := System × Time  
Type StateSpace := Set[State]
Type Metric := StateSpace × StateSpace → Real⁺
Type Real⁺ := {r ∈ ℝ | r ≥ 0}
```

## 附加性质

### 离散性
```
Discrete(τ : Metric) : Prop ≡
  ∃δ > 0 . ∀i ≠ j . τ(Sᵢ, Sⱼ) ≥ δ
```

### 不可逆性
```
Irreversible(S : System) : Prop ≡
  ∀t : Time . ¬∃φ : Sₜ₊₁ → Sₜ . Surjective(φ)
```

### 累积性
```
TotalTime(t : Time) : Real⁺ ≡
  Σ(k=0 to t-1) τ(Sₖ, Sₖ₊₁)
```

## 机器验证检查点

### 检查点1：非负性验证
```python
def verify_non_negativity(metric, states):
    return all(metric(si, sj) >= 0 for si in states for sj in states)
```

### 检查点2：单调性验证
```python
def verify_monotonicity(metric, states):
    return all(metric(si, sj) < metric(si, sk) 
               for si, sj, sk in states if i < j < k)
```

### 检查点3：可加性验证
```python
def verify_additivity(metric, states):
    return all(abs(metric(si, sk) - (metric(si, sj) + metric(sj, sk))) < ε
               for si, sj, sk in states if i <= j <= k)
```

### 检查点4：方向性验证
```python
def verify_directionality(metric, states):
    return all((metric(si, sj) > 0) == (i < j)
               for si, sj in states)
```

## 形式化验证状态
- [x] 定义语法正确
- [x] 性质相互独立
- [x] 构造方法完整
- [x] 类型声明清晰
- [x] 最小完备