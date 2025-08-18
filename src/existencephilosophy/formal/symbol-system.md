# 存在哲学的严格符号系统 (Rigorous Symbol System for Existence Philosophy)  

## 基础符号定义  

### 1. 原始概念 (Primitive Concepts)  

#### 1.1 论域 (Domain of Discourse)  
```  
𝔻 = {x | x可在此理论中被指称和讨论}  
```  
- 包含：常量符号、变量、谓词所涉及的对象  
- 排除：形式矛盾的符号组合  
- 注：这是形式系统的论域，不预设本体论承诺  

#### 1.2 基本常量 (Basic Constants)  
```  
E ∈ 𝔻  : 指称存在本体的常量符号  
⊥ ∈ 𝔻  : 指称矛盾的常量符号  
```  
注：E和⊥的具体性质由公理系统定义，此处仅给出符号。  

#### 1.3 变量约定 (Variable Conventions)  
```  
x, y, z ∈ 𝔻     : 一般存在域变量  
s, t ∈ States   : 状态变量  
o ∈ Observers  : 观察者变量  
i, j ∈ Info     : 信息变量  
n, m ∈ ℕ       : 自然数（用于层次索引）  
```  

### 2. 基本谓词 (Basic Predicates)  

#### 2.1 存在谓词  
```  
Exists: 𝔻 → {True, False}  
Exists(x) ≡ "x存在"  
```  
定义域：𝔻中的所有元素  
值域：真值集合  

#### 2.2 定义谓词  
```  
Def: 𝔻 × 𝔻 → {True, False}  
Def(x, y) ≡ "x定义y"  
```  
定义域：𝔻 × 𝔻的有序对  
约束：∀x,y • Def(x, y) → Exists(x)  

#### 2.3 自指谓词  
```  
SelfDef: 𝔻 → {True, False}  
SelfDef(x) ≡ Def(x, x)  
```  
定义域：𝔻中的所有元素  
等价定义：x自我定义当且仅当x定义x  

#### 2.4 展开谓词  
```  
Unfold: 𝔻 → 𝒫(𝔻)  
Unfold(x) ≡ "x展开产生的元素集合"  
```  
定义域：𝔻中的所有元素  
值域：𝔻的幂集  

#### 2.5 觉知谓词  
```  
Aware: Observers × Info → {True, False}  
Aware(o, i) ≡ "观察者o觉知信息i"  
```  
定义域：Observers × Info  
前提条件：Exists(o) ∧ Exists(i)  

#### 2.6 超越谓词  
```  
Transcend: States × States → {True, False}  
Transcend(s₁, s₂) ≡ "状态s₁超越状态s₂"  
```  
定义域：States × States  
性质：反自反、传递但非对称  

### 3. 导出概念 (Derived Concepts)  

#### 3.1 子域定义  
```  
注：以下子域由公理推导确定，此处仅给出形式结构  

Observers ⊆ 𝔻  (观察者子域)  
Info ⊆ 𝔻       (信息子域)  
States ⊆ 𝔻     (状态子域)  
Time ⊆ 𝔻       (时间子域)  

具体元素由A3展开公理和A4观察公理确定  
```  

#### 3.2 复合谓词  
```  
注：以下谓词基于基本谓词定义，避免循环  

Depends: 𝔻 × 𝔻 → {True, False}  
Independent: 𝔻 → {True, False}  
SelfAware: 𝔻 → {True, False}  

具体定义由公理系统的推导确定，此处仅给出谓词签名  
```  

### 4. 逻辑连接词 (Logical Connectives)  

#### 4.1 经典逻辑连接词  
```  
¬ : 否定 (Negation)  
∧ : 合取 (Conjunction)  
∨ : 析取 (Disjunction)  
→ : 蕴含 (Implication)  
↔ : 等价 (Biconditional)  
```  

#### 4.2 量词及其定义域  
```  
∀x ∈ 𝔻 • P(x) : 全称量化，x遍历整个存在域  
∃x ∈ 𝔻 • P(x) : 存在量化，在存在域中至少存在一个x  
∃!x ∈ 𝔻 • P(x) : 唯一存在量化，恰好存在一个x满足P(x)  
```  

#### 4.3 模态算子（扩展用）  
```  
□ : 必然 (Necessity)  
◇ : 可能 (Possibility)  
```  
定义：  
- □P ≡ "P在所有可能世界中为真"  
- ◇P ≡ "P在至少一个可能世界中为真"  

### 5. 函数符号 (Function Symbols)  

#### 5.1 状态函数  
```  
State: 𝔻 × Time → States  
State(x, t) = "对象x在时刻t的状态"  
```  

#### 5.2 层次函数  
```  
Level: ℕ → 𝒫(States)  
Level(n) = "第n层次的所有状态集合"  
```  

#### 5.3 信息提取函数  
```  
Extract: Unfold(E) → Info  
Extract(u) = "从展开结果u中提取的信息"  
```  

### 6. 关系符号 (Relation Symbols)  

#### 6.1 超越关系（替代简单序关系）  
```  
Transcend : States × States → {True, False}  
Transcend(s₂, s₁) ≡ "状态s₂超越状态s₁"  
```  
性质：反自反、传递，但非连接（并非所有状态对都可比较）  

#### 6.2 等价关系  
```  
≡ : 𝔻 × 𝔻 → {True, False}  
x ≡ y ≡ "x与y逻辑等价"  
```  
性质：等价关系（自反、对称、传递）  

#### 6.3 同构关系  
```  
≅ : 𝔻 × 𝔻 → {True, False}  
x ≅ y ≡ "x与y结构同构"  
```  

### 7. 严格定义规则  

#### 7.1 良构公式 (Well-Formed Formulas)  
一个公式是良构的当且仅当：  
1. 所有变量都有明确的定义域  
2. 所有函数应用都在其定义域内  
3. 所有谓词应用都满足其前提条件  

#### 7.2 类型一致性  
```  
Type-Check规则：  
- 如果P(x)且x ∈ A，则P的定义域必须包含A  
- 如果f(x) = y且x ∈ A，则f: A → B且y ∈ B  
```  

#### 7.3 存在性前提  
```  
存在性规则：  
- 使用Def(x, y)前，必须确立Exists(x)  
- 使用Aware(o, i)前，必须确立o ∈ Observers ∧ i ∈ Info  
- 使用Transcend(s₁, s₂)前，必须确立s₁, s₂ ∈ States  
```  

### 8. 符号使用约定  

#### 8.1 优先级约定  
```  
括号 > 量词 > 否定 > 合取 > 析取 > 蕴含 > 等价  
```  

#### 8.2 缩写约定  
```  
∀x • P(x) ≡ ∀x ∈ 𝔻 • P(x) （当上下文明确时）  
P ∧ Q ∧ R ≡ (P ∧ Q) ∧ R （左结合）  
P → Q → R ≡ P → (Q → R) （右结合）  
```  

#### 8.3 符号一致性要求  
1. 同一证明中，同一符号必须指代同一对象  
2. 量化变量不能与自由变量重名  
3. 嵌套量词必须使用不同的变量名  

### 9. 元语言约定  

#### 9.1 证明标记  
```  
⊢ : 可证明 (Provable)  
⊨ : 语义蕴含 (Semantic entailment)  
⊥ : 矛盾 (Contradiction)  
QED : 证明完成 (Quod erat demonstrandum)  
```  

#### 9.2 推理步骤标记  
```  
[前提] : 给定的前提  
[定义] : 根据定义展开  
[MP] : Modus Ponens应用  
[∀E] : 全称量词消除  
[∃I] : 存在量词引入  
[反证] : 反证法  
```  

### 10. 语义解释  

#### 10.1 标准模型  
```  
标准模型 M = ⟨𝔻, E, I⟩ 其中：  
- 𝔻是非空存在域  
- E是存在本体的解释  
- I是解释函数，将符号映射到其语义  
```  

#### 10.2 真值条件  
```  
M ⊨ Exists(x) iff x在模型M中有对应的存在物  
M ⊨ Def(x, y) iff x在模型M中定义了y  
M ⊨ P ∧ Q iff M ⊨ P 且 M ⊨ Q  
M ⊨ ∀x • P(x) iff 对所有d ∈ 𝔻, M ⊨ P(d)  
```  

这个符号系统为存在哲学提供了严格的形式基础，消除了歧义，确保推理的精确性。  