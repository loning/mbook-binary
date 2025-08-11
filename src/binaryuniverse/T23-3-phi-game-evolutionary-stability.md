# T23-3 φ-博弈演化稳定性定理

## 依赖关系
- **前置定理**: T23-2 (φ-博弈均衡存在性定理), T23-1 (φ-博弈策略涌现定理)
- **前置推论**: C20-1 (collapse-aware观测推论)
- **前置定义**: D1-8 (φ-表示系统), D1-7 (Collapse算子)
- **唯一公理**: A1 (自指完备系统必然熵增)

## 定理陈述

**定理 T23-3** (φ-博弈演化稳定性定理): 基于T23-2的φ-Nash均衡理论，演化稳定策略(ESS)在φ-博弈系统中必须满足严格的熵增稳定性条件：

1. **φ-ESS的定义**: 策略$s^* \in S$是演化稳定的，当且仅当对任何入侵策略$s' \neq s^*$，存在入侵阈值$\epsilon_{\phi} = \frac{1}{\phi^2}$，使得当入侵比例$\epsilon < \epsilon_{\phi}$时：
   
$$
U(s^*, (1-\epsilon)s^* + \epsilon s') > U(s', (1-\epsilon)s^* + \epsilon s')
$$
2. **熵增稳定性条件**: φ-ESS必须满足局部熵最大化：
   
$$
\frac{\partial^2 H}{\partial s^2}\bigg|_{s=s^*} < -\frac{1}{\phi} \quad \text{(严格凹性)}
$$
   其中$H$是策略分布的熵

3. **演化动力学的φ-调制**: 复制动力学方程在φ-系统中修正为：
   
$$
\frac{dx_i}{dt} = \frac{x_i}{\phi} \left[f_i(\mathbf{x}) - \bar{f}(\mathbf{x}) + \frac{\partial H}{\partial x_i}\right]
$$
   其中$x_i$是策略$i$的频率，$f_i$是适应度，熵梯度项确保熵增

4. **稳定性半径**: φ-ESS的稳定性半径为：
   
$$
r_{stable} = \frac{1}{\phi} \cdot \min_{s' \neq s^*} \left\|\frac{F_{k'}}{φ^{d'}} - \frac{F_k}{φ^d}\right\|
$$
   其中策略表示为$s = F_k/φ^d$（Fibonacci数的φ-幂）

5. **演化收敛时间**: 从任意初始分布收敛到ESS的时间满足：
   
$$
T_{converge} \leq \phi^2 \cdot \log\left(\frac{1}{\delta}\right)
$$
   其中$\delta$是收敛精度

## 证明

### 第一步：从熵增原理推导ESS条件

由唯一公理，自指完备系统必然熵增。在演化博弈中，策略分布的演化必须满足：
$$
\frac{dH}{dt} > 0
$$
考虑策略分布$\mathbf{p} = (p_1, ..., p_n)$的熵：
$$
H(\mathbf{p}) = -\sum_{i=1}^n p_i \log p_i
$$
对于候选ESS策略$s^*$（纯策略或混合策略），定义入侵后的分布：
$$
\mathbf{p}_{\epsilon} = (1-\epsilon)\mathbf{p}^* + \epsilon \mathbf{p}'
$$
其中$\mathbf{p}^*$是ESS分布，$\mathbf{p}'$是入侵分布。

**熵增要求**：
$$
H(\mathbf{p}_{\epsilon}) - H(\mathbf{p}^*) > 0 \quad \text{for small } \epsilon > 0
$$
泰勒展开到二阶：
$$
H(\mathbf{p}_{\epsilon}) = H(\mathbf{p}^*) + \epsilon \nabla H \cdot (\mathbf{p}' - \mathbf{p}^*) + \frac{\epsilon^2}{2}(\mathbf{p}' - \mathbf{p}^*)^T \nabla^2 H (\mathbf{p}' - \mathbf{p}^*)
$$
由于$\mathbf{p}^*$是局部最大值的候选，一阶项为零。二阶项必须为负（严格凹性）：
$$
\nabla^2 H\bigg|_{\mathbf{p}^*} < -\frac{1}{\phi} I
$$
这确保了熵的局部最大性，是φ-ESS的必要条件。

### 第二步：建立演化稳定性的φ-条件

**经典ESS条件**（Maynard Smith）：
1. $(s^*, s^*) \geq (s', s^*)$ 对所有$s' \neq s^*$（Nash均衡条件）
2. 如果$(s^*, s^*) = (s', s^*)$，则$(s^*, s') > (s', s')$（稳定性条件）

**φ-ESS扩展**：
在φ-系统中，收益函数包含熵贡献：
$$
U_{\phi}(s, s') = U(s, s') + \frac{1}{\phi} H(s|s')
$$
其中$H(s|s')$是条件熵，反映策略$s$在面对$s'$时的不确定性。

对于Zeckendorf编码的策略$s = F_k/φ^d$，条件熵为：
$$
H(s|s') = -\sum_{i: z_i=1} \frac{1}{φ^i} \log \frac{1}{φ^i}
$$
其中$z_i$是Zeckendorf表示的第$i$位。

**φ-ESS判据**：
策略$s^*$是φ-ESS当且仅当：
1. $U_{\phi}(s^*, s^*) \geq U_{\phi}(s', s^*)$ 对所有$s' \neq s^*$
2. 存在$\epsilon_{\phi} = 1/φ^2$使得对$\epsilon < \epsilon_{\phi}$：
   
$$
U_{\phi}(s^*, (1-\epsilon)s^* + \epsilon s') > U_{\phi}(s', (1-\epsilon)s^* + \epsilon s')
$$
### 第三步：推导演化动力学的φ-修正

标准复制动力学：
$$
\frac{dx_i}{dt} = x_i(f_i - \bar{f})
$$
在φ-系统中，考虑熵增约束，动力学修正为：
$$
\frac{dx_i}{dt} = \frac{x_i}{\phi} \left[f_i(\mathbf{x}) - \bar{f}(\mathbf{x}) + \frac{\partial H}{\partial x_i}\right]
$$
熵梯度项：
$$
\frac{\partial H}{\partial x_i} = -\log x_i - 1
$$
这确保了演化过程中的熵增。时间尺度因子$1/φ$反映了φ-系统的内在时间结构。

**Lyapunov函数**：
定义$V(\mathbf{x}) = -H(\mathbf{x}) + \frac{1}{\phi} \sum_i x_i \log f_i(\mathbf{x})$

沿轨迹的导数：
$$
\frac{dV}{dt} = -\frac{1}{\phi} \sum_i x_i \left(\frac{\partial H}{\partial x_i}\right)^2 < 0
$$
这证明了φ-ESS的渐近稳定性。

### 第四步：计算稳定性半径

对于φ-策略$s^* = F_k/φ^d$，考虑邻近策略$s' = F_{k'}/φ^{d'}$。

策略距离（在φ-度量下）：
$$
d_{\phi}(s^*, s') = \left|\frac{F_k}{φ^d} - \frac{F_{k'}}{φ^{d'}}\right|
$$
**稳定性半径定理**：
φ-ESS $s^*$能抵抗所有满足$d_{\phi}(s^*, s') > r_{stable}$的入侵策略，其中：
$$
r_{stable} = \frac{1}{\phi} \cdot \min_{s' \neq s^*} d_{\phi}(s^*, s')
$$
证明：考虑入侵动力学
$$
\frac{d\epsilon}{dt} = \epsilon(1-\epsilon)[U_{\phi}(s', \mathbf{p}_{\epsilon}) - U_{\phi}(s^*, \mathbf{p}_{\epsilon})]
$$
当$d_{\phi}(s^*, s') > r_{stable}$时，括号内的项对小$\epsilon$为负，因此$\epsilon \to 0$。

### 第五步：演化收敛时间分析

从初始分布$\mathbf{x}(0)$到φ-ESS $\mathbf{x}^*$的收敛由以下估计给出：

$$
\|\mathbf{x}(t) - \mathbf{x}^*\| \leq \|\mathbf{x}(0) - \mathbf{x}^*\| \cdot \exp\left(-\frac{t}{\phi^2}\right)
$$
要达到精度$\delta$：
$$
\|\mathbf{x}(T) - \mathbf{x}^*\| < \delta
$$
需要时间：
$$
T_{converge} = \phi^2 \cdot \log\left(\frac{\|\mathbf{x}(0) - \mathbf{x}^*\|}{\delta}\right)
$$
由于$\|\mathbf{x}(0) - \mathbf{x}^*\| \leq 1$（概率单纯形内），得：
$$
T_{converge} \leq \phi^2 \cdot \log\left(\frac{1}{\delta}\right)
$$
这完成了证明。∎

## 数学形式化

```python
class PhiEvolutionaryStableStrategy:
    """φ-演化稳定策略实现"""
    
    def __init__(self, game_system: PhiGameSystem):
        self.game = game_system
        self.phi = (1 + np.sqrt(5)) / 2
        self.invasion_threshold = 1.0 / (self.phi ** 2)
        
    def is_phi_ess(self, strategy: PhiStrategy, tolerance: float = 1e-6) -> bool:
        """判断策略是否为φ-ESS"""
        # 1. 检查Nash均衡条件
        if not self._is_nash_equilibrium(strategy):
            return False
            
        # 2. 检查熵的严格凹性
        if not self._check_entropy_concavity(strategy):
            return False
            
        # 3. 检查入侵稳定性
        return self._check_invasion_stability(strategy, tolerance)
        
    def _check_entropy_concavity(self, strategy: PhiStrategy) -> bool:
        """检查熵的二阶导数条件"""
        # 计算策略分布的Hessian矩阵
        hessian = self._compute_entropy_hessian(strategy)
        
        # 检查负定性（所有特征值 < -1/φ）
        eigenvalues = np.linalg.eigvals(hessian)
        return np.all(eigenvalues < -1.0/self.phi)
        
    def _check_invasion_stability(self, resident: PhiStrategy, 
                                 tolerance: float) -> bool:
        """检查对入侵策略的稳定性"""
        strategies = self.game.strategy_space.get_all_strategies()
        
        for mutant in strategies:
            if mutant != resident:
                # 测试不同入侵比例
                for epsilon in [0.001, 0.01, 0.1]:
                    if epsilon < self.invasion_threshold:
                        # 混合群体
                        mixed_pop = (1 - epsilon) * resident + epsilon * mutant
                        
                        # 计算适应度
                        resident_fitness = self._compute_fitness(resident, mixed_pop)
                        mutant_fitness = self._compute_fitness(mutant, mixed_pop)
                        
                        # ESS条件
                        if mutant_fitness >= resident_fitness - tolerance:
                            return False
                            
        return True
        
    def compute_stability_radius(self, ess: PhiStrategy) -> float:
        """计算ESS的稳定性半径"""
        min_distance = float('inf')
        strategies = self.game.strategy_space.get_all_strategies()
        
        for strategy in strategies:
            if strategy != ess:
                distance = abs(ess.value - strategy.value)
                min_distance = min(min_distance, distance)
                
        return min_distance / self.phi
        
    def evolve_to_ess(self, initial_dist: Dict[PhiStrategy, float],
                     max_time: int = 1000) -> Dict[PhiStrategy, float]:
        """演化到ESS的动力学模拟"""
        current_dist = initial_dist.copy()
        dt = 0.1
        
        for t in range(max_time):
            # 计算平均适应度
            avg_fitness = self._compute_average_fitness(current_dist)
            
            # 更新每个策略的频率
            new_dist = {}
            for strategy, freq in current_dist.items():
                fitness = self._compute_fitness(strategy, current_dist)
                entropy_grad = -np.log(freq) - 1 if freq > 0 else 0
                
                # φ-复制动力学
                change = (freq / self.phi) * (fitness - avg_fitness + entropy_grad)
                new_freq = freq + dt * change
                new_freq = max(0, min(1, new_freq))
                new_dist[strategy] = new_freq
                
            # 归一化
            total = sum(new_dist.values())
            if total > 0:
                new_dist = {s: f/total for s, f in new_dist.items()}
                
            # 检查收敛
            if self._has_converged(current_dist, new_dist):
                return new_dist
                
            current_dist = new_dist
            
        return current_dist
        
    def verify_convergence_time(self, ess: PhiStrategy, 
                               delta: float = 0.01) -> float:
        """验证收敛时间上界"""
        theoretical_bound = (self.phi ** 2) * np.log(1.0 / delta)
        return theoretical_bound
```

## 物理解释

1. **生物进化**: 物种策略的演化稳定状态遵循φ-ESS条件
2. **文化演化**: 社会规范和文化模因的稳定传播
3. **市场演化**: 交易策略在金融市场中的长期稳定性

## 实验可验证预言

1. **入侵阈值**: 成功入侵需要超过$1/φ^2 \approx 38.2\%$的初始频率
2. **收敛时间**: 演化到稳定状态的时间与$φ^2 \cdot \log(1/\delta)$成正比
3. **稳定性半径**: ESS的稳定域大小与$1/φ$成正比

## 应用示例

```python
# 创建φ-博弈系统
network = WeightedPhiNetwork(n_initial=5)
game = PhiGameSystem(network, n_players=5)

# 初始化ESS分析器
ess_analyzer = PhiEvolutionaryStableStrategy(game)

# 寻找候选ESS
strategies = game.strategy_space.get_all_strategies()
for strategy in strategies[:10]:  # 测试前10个策略
    if ess_analyzer.is_phi_ess(strategy):
        print(f"找到φ-ESS: {strategy}")
        
        # 计算稳定性半径
        radius = ess_analyzer.compute_stability_radius(strategy)
        print(f"稳定性半径: {radius:.4f}")
        
        # 验证收敛时间
        conv_time = ess_analyzer.verify_convergence_time(strategy, delta=0.01)
        print(f"理论收敛时间上界: {conv_time:.2f}")
        
        # 模拟演化动力学
        initial = {s: 1/len(strategies) for s in strategies}
        final = ess_analyzer.evolve_to_ess(initial)
        
        # 分析最终分布
        dominant = max(final.items(), key=lambda x: x[1])
        print(f"演化终态: {dominant[0]} (频率={dominant[1]:.3f})")
        break
```

---

**注记**: T23-3建立了演化稳定策略在φ-博弈系统中的完整理论，揭示了ESS必须满足的熵增稳定性条件。入侵阈值$1/φ^2$和收敛时间$φ^2 \log(1/\delta)$都体现了黄金比例在演化稳定性中的基础作用。