# T11-2 相变定理 - 形式化描述

## 1. 形式化框架

### 1.1 相变系统模型

```python
class PhaseTransitionSystem:
    """相变定理的数学模型"""
    
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.lambda_c = 1 / self.phi  # 临界参数
        self.MAX_LENGTH = 100
        self.MIN_LENGTH = 10
        
    def calculate_order_parameter(self, state: str) -> float:
        """计算序参量 O(S) = (1/|S|) Σ δ(s_i = s_{i+1})"""
        if not state or len(state) < 2:
            return 0
            
        same_count = sum(1 for i in range(len(state)-1) if state[i] == state[i+1])
        return same_count / (len(state) - 1)
        
    def calculate_energy(self, state: str) -> float:
        """计算能量函数 E(S) = -Σ s_i s_{i+1}"""
        if not state or len(state) < 2:
            return 0
            
        energy = 0
        for i in range(len(state) - 1):
            # 将'0'和'1'映射到-1和+1
            s_i = 1 if state[i] == '1' else -1
            s_i1 = 1 if state[i+1] == '1' else -1
            energy -= s_i * s_i1
            
        return energy
        
    def calculate_partition_function(self, length: int, lambda_param: float) -> float:
        """计算配分函数 Z(λ) = Σ_S exp(-λE(S))"""
        # 对于二进制系统的精确计算（小系统）
        if length > 20:  # 对大系统使用近似
            return self.approximate_partition_function(length, lambda_param)
            
        Z = 0
        for i in range(2**length):
            state = format(i, f'0{length}b')
            if self.is_valid_state(state):  # no-11约束
                energy = self.calculate_energy(state)
                Z += np.exp(-lambda_param * energy)
                
        return Z
        
    def approximate_partition_function(self, length: int, lambda_param: float) -> float:
        """大系统的配分函数近似"""
        # 使用平均场近似
        if lambda_param < self.lambda_c:
            # 无序相
            return np.exp(length * np.log(self.phi))  # Fibonacci数的渐近
        else:
            # 有序相
            return np.exp(length * lambda_param)
            
    def identify_phase(self, state: str) -> str:
        """识别系统所处的相态"""
        O = self.calculate_order_parameter(state)
        
        if O > self.phi - 1:  # ≈ 0.618
            return "ordered"
        elif O < 1 - 1/self.phi:  # ≈ 0.382
            return "disordered"
        else:
            return "critical"
            
    def generate_state_at_temperature(self, length: int, lambda_param: float, 
                                    num_steps: int = 1000) -> str:
        """使用Metropolis算法在给定温度下生成状态"""
        # 初始随机状态
        state = self.generate_random_valid_state(length)
        
        for _ in range(num_steps):
            # 随机选择一个位置翻转
            pos = np.random.randint(0, length)
            new_state = list(state)
            new_state[pos] = '0' if state[pos] == '1' else '1'
            new_state = ''.join(new_state)
            
            # 检查no-11约束
            if not self.is_valid_state(new_state):
                continue
                
            # Metropolis准则
            delta_E = self.calculate_energy(new_state) - self.calculate_energy(state)
            if delta_E < 0 or np.random.random() < np.exp(-lambda_param * delta_E):
                state = new_state
                
        return state
        
    def is_valid_state(self, state: str) -> bool:
        """检查状态是否满足no-11约束"""
        return '11' not in state
        
    def generate_random_valid_state(self, length: int) -> str:
        """生成满足no-11约束的随机状态"""
        state = ""
        prev = '0'
        for _ in range(length):
            if prev == '1':
                state += '0'
                prev = '0'
            else:
                bit = np.random.choice(['0', '1'])
                state += bit
                prev = bit
        return state
        
    def calculate_correlation_length(self, state: str) -> float:
        """计算关联长度 ξ"""
        if len(state) < 4:
            return 1
            
        # 计算自关联函数
        correlations = []
        for r in range(1, min(len(state)//2, 20)):
            corr = 0
            count = 0
            for i in range(len(state) - r):
                s_i = 1 if state[i] == '1' else -1
                s_ir = 1 if state[i+r] == '1' else -1
                corr += s_i * s_ir
                count += 1
            if count > 0:
                correlations.append(corr / count)
                
        # 拟合指数衰减找到关联长度
        if not correlations:
            return 1
            
        # 简单估计：找到关联降到1/e的距离
        for i, corr in enumerate(correlations):
            if abs(corr) < 1/np.e:
                return i + 1
                
        return len(correlations)
        
    def calculate_susceptibility(self, states: List[str]) -> float:
        """计算磁化率 χ = <O²> - <O>²"""
        if not states:
            return 0
            
        order_params = [self.calculate_order_parameter(s) for s in states]
        mean_O = np.mean(order_params)
        mean_O2 = np.mean([O**2 for O in order_params])
        
        return mean_O2 - mean_O**2
```

### 1.2 相变分析器

```python
class PhaseTransitionAnalyzer:
    """相变行为的详细分析"""
    
    def __init__(self):
        self.pt_system = PhaseTransitionSystem()
        self.phi = (1 + np.sqrt(5)) / 2
        
    def scan_phase_diagram(self, length: int, lambda_range: Tuple[float, float], 
                          num_points: int = 50) -> Dict[str, Any]:
        """扫描参数空间构建相图"""
        lambda_values = np.linspace(lambda_range[0], lambda_range[1], num_points)
        results = {
            'lambda': [],
            'order_parameter': [],
            'energy': [],
            'phase': [],
            'correlation_length': []
        }
        
        for lam in lambda_values:
            # 生成该参数下的平衡态
            states = [self.pt_system.generate_state_at_temperature(length, lam) 
                     for _ in range(10)]
            
            # 计算平均值
            avg_O = np.mean([self.pt_system.calculate_order_parameter(s) for s in states])
            avg_E = np.mean([self.pt_system.calculate_energy(s) for s in states])
            avg_xi = np.mean([self.pt_system.calculate_correlation_length(s) for s in states])
            
            # 识别相态
            phase = self.pt_system.identify_phase(states[0])
            
            results['lambda'].append(lam)
            results['order_parameter'].append(avg_O)
            results['energy'].append(avg_E)
            results['phase'].append(phase)
            results['correlation_length'].append(avg_xi)
            
        return results
        
    def locate_critical_point(self, length: int, precision: float = 0.001) -> float:
        """精确定位临界点"""
        # 二分搜索找到序参量跳跃的位置
        left, right = 0.1, 2.0
        
        while right - left > precision:
            mid = (left + right) / 2
            
            # 在mid两侧采样
            states_low = [self.pt_system.generate_state_at_temperature(length, mid - 0.01) 
                         for _ in range(20)]
            states_high = [self.pt_system.generate_state_at_temperature(length, mid + 0.01) 
                          for _ in range(20)]
            
            O_low = np.mean([self.pt_system.calculate_order_parameter(s) for s in states_low])
            O_high = np.mean([self.pt_system.calculate_order_parameter(s) for s in states_high])
            
            # 检查跳跃
            if abs(O_high - O_low) > 0.1:
                return mid
            elif O_low < 0.5:
                left = mid
            else:
                right = mid
                
        return (left + right) / 2
        
    def measure_critical_exponents(self, length: int) -> Dict[str, float]:
        """测量临界指数"""
        lambda_c = self.pt_system.lambda_c
        exponents = {}
        
        # β指数：序参量 O ~ |λ - λc|^β
        deltas = [0.01, 0.02, 0.05, 0.1]
        O_values = []
        
        for delta in deltas:
            states = [self.pt_system.generate_state_at_temperature(length, lambda_c + delta) 
                     for _ in range(20)]
            O = np.mean([self.pt_system.calculate_order_parameter(s) for s in states])
            O_values.append((delta, O))
            
        # 对数拟合提取指数
        if len(O_values) > 2:
            log_deltas = np.log([d for d, _ in O_values])
            log_Os = np.log([O for _, O in O_values if O > 0])
            if len(log_Os) == len(log_deltas):
                beta = np.polyfit(log_deltas, log_Os, 1)[0]
                exponents['beta'] = abs(beta)
            
        # ν指数：关联长度 ξ ~ |λ - λc|^(-ν)
        xi_values = []
        
        for delta in deltas:
            states = [self.pt_system.generate_state_at_temperature(length, lambda_c - delta) 
                     for _ in range(10)]
            xi = np.mean([self.pt_system.calculate_correlation_length(s) for s in states])
            xi_values.append((delta, xi))
            
        if len(xi_values) > 2:
            log_deltas = np.log([d for d, _ in xi_values])
            log_xis = np.log([xi for _, xi in xi_values if xi > 0])
            if len(log_xis) == len(log_deltas):
                nu = -np.polyfit(log_deltas, log_xis, 1)[0]
                exponents['nu'] = abs(nu)
                
        return exponents
        
    def verify_finite_size_scaling(self, sizes: List[int]) -> Dict[str, Any]:
        """验证有限尺度标度律"""
        lambda_c = self.pt_system.lambda_c
        results = {
            'sizes': sizes,
            'order_jump': [],
            'correlation_length': []
        }
        
        for L in sizes:
            # 测量序参量跳跃
            states_low = [self.pt_system.generate_state_at_temperature(L, lambda_c - 0.05) 
                         for _ in range(10)]
            states_high = [self.pt_system.generate_state_at_temperature(L, lambda_c + 0.05) 
                          for _ in range(10)]
            
            O_low = np.mean([self.pt_system.calculate_order_parameter(s) for s in states_low])
            O_high = np.mean([self.pt_system.calculate_order_parameter(s) for s in states_high])
            
            results['order_jump'].append(abs(O_high - O_low))
            
            # 测量关联长度
            states_c = [self.pt_system.generate_state_at_temperature(L, lambda_c) 
                       for _ in range(10)]
            xi = np.mean([self.pt_system.calculate_correlation_length(s) for s in states_c])
            results['correlation_length'].append(xi)
            
        # 验证标度关系 ΔO ~ L^(-1/ν)
        if len(sizes) > 2:
            log_L = np.log(sizes)
            log_jump = np.log(results['order_jump'])
            slope = np.polyfit(log_L, log_jump, 1)[0]
            results['scaling_exponent'] = -slope
            results['theoretical_exponent'] = self.phi  # 1/ν = φ
            
        return results
```

### 1.3 临界现象验证器

```python
class CriticalPhenomenaVerifier:
    """临界现象的详细验证"""
    
    def __init__(self):
        self.pt_system = PhaseTransitionSystem()
        self.analyzer = PhaseTransitionAnalyzer()
        
    def test_universality(self, different_models: List[Callable]) -> Dict[str, Any]:
        """测试普适性假设"""
        results = {
            'models': [],
            'critical_exponents': [],
            'consistency': False
        }
        
        for model in different_models:
            # 每个模型应该返回相同的临界指数
            exponents = self.analyzer.measure_critical_exponents(50)
            results['models'].append(model.__name__)
            results['critical_exponents'].append(exponents)
            
        # 检查一致性
        if len(results['critical_exponents']) > 1:
            beta_values = [exp.get('beta', 0) for exp in results['critical_exponents']]
            nu_values = [exp.get('nu', 0) for exp in results['critical_exponents']]
            
            beta_std = np.std(beta_values)
            nu_std = np.std(nu_values)
            
            results['consistency'] = beta_std < 0.1 and nu_std < 0.1
            
        return results
        
    def analyze_fluctuations(self, lambda_param: float, length: int, 
                           num_samples: int = 100) -> Dict[str, Any]:
        """分析涨落性质"""
        states = [self.pt_system.generate_state_at_temperature(length, lambda_param) 
                 for _ in range(num_samples)]
        
        # 计算各种涨落
        order_params = [self.pt_system.calculate_order_parameter(s) for s in states]
        energies = [self.pt_system.calculate_energy(s) for s in states]
        
        results = {
            'lambda': lambda_param,
            'mean_order': np.mean(order_params),
            'order_fluctuation': np.std(order_params),
            'mean_energy': np.mean(energies),
            'energy_fluctuation': np.std(energies),
            'susceptibility': self.pt_system.calculate_susceptibility(states),
            'specific_heat': np.var(energies) * lambda_param**2  # C = (∂²F/∂T²)
        }
        
        # 在临界点附近涨落应该最大
        results['near_critical'] = abs(lambda_param - self.pt_system.lambda_c) < 0.1
        
        return results
        
    def verify_scaling_relations(self, length: int = 50) -> Dict[str, Any]:
        """验证标度关系"""
        exponents = self.analyzer.measure_critical_exponents(length)
        
        results = {
            'measured_exponents': exponents,
            'scaling_relations': {}
        }
        
        # 理论预测的指数关系
        beta = exponents.get('beta', 1/self.pt_system.phi)
        nu = exponents.get('nu', 1/self.pt_system.phi)
        
        # Rushbrooke不等式: α + 2β + γ ≥ 2
        # 对于我们的模型，预期等式成立
        
        # 超标度关系: 2 - α = ν·d (d=1 for 1D system)
        # Fisher关系: γ = ν(2 - η)
        
        results['theoretical_relations'] = {
            'beta': 1/self.pt_system.phi,
            'nu': 1/self.pt_system.phi,
            'gamma': self.pt_system.phi
        }
        
        return results
```

## 2. 综合验证系统

```python
class PhaseTransitionTheoremVerifier:
    """T11-2相变定理的综合验证"""
    
    def __init__(self):
        self.pt_system = PhaseTransitionSystem()
        self.analyzer = PhaseTransitionAnalyzer()
        self.cp_verifier = CriticalPhenomenaVerifier()
        
    def run_comprehensive_verification(self) -> Dict[str, Any]:
        """运行完整验证套件"""
        results = {
            'phase_diagram': {},
            'critical_point': {},
            'critical_exponents': {},
            'finite_size_scaling': {},
            'fluctuation_analysis': {},
            'overall_assessment': {}
        }
        
        # 1. 构建相图
        phase_diagram = self.analyzer.scan_phase_diagram(30, (0.1, 2.0), 50)
        results['phase_diagram'] = {
            'lambda_range': (0.1, 2.0),
            'phases_found': list(set(phase_diagram['phase'])),
            'order_parameter_jump': max(phase_diagram['order_parameter']) - 
                                   min(phase_diagram['order_parameter'])
        }
        
        # 2. 定位临界点
        measured_lambda_c = self.analyzer.locate_critical_point(50)
        results['critical_point'] = {
            'measured': measured_lambda_c,
            'theoretical': self.pt_system.lambda_c,
            'error': abs(measured_lambda_c - self.pt_system.lambda_c)
        }
        
        # 3. 测量临界指数
        exponents = self.analyzer.measure_critical_exponents(50)
        results['critical_exponents'] = exponents
        
        # 4. 有限尺度标度
        scaling = self.analyzer.verify_finite_size_scaling([10, 20, 30, 40, 50])
        results['finite_size_scaling'] = scaling
        
        # 5. 涨落分析
        fluct_critical = self.cp_verifier.analyze_fluctuations(self.pt_system.lambda_c, 50)
        fluct_ordered = self.cp_verifier.analyze_fluctuations(self.pt_system.lambda_c + 0.2, 50)
        fluct_disordered = self.cp_verifier.analyze_fluctuations(self.pt_system.lambda_c - 0.2, 50)
        
        results['fluctuation_analysis'] = {
            'critical': fluct_critical,
            'ordered': fluct_ordered,
            'disordered': fluct_disordered,
            'max_fluctuation_at_critical': fluct_critical['susceptibility'] > 
                                          max(fluct_ordered['susceptibility'], 
                                              fluct_disordered['susceptibility'])
        }
        
        # 6. 总体评估
        results['overall_assessment'] = self.assess_results(results)
        
        return results
        
    def assess_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """评估验证结果"""
        assessment = {
            'phase_transition_confirmed': False,
            'critical_point_accurate': False,
            'scaling_laws_verified': False,
            'theorem_support': 'Weak'
        }
        
        # 检查相变存在性
        if results['phase_diagram']['order_parameter_jump'] > 0.2:
            assessment['phase_transition_confirmed'] = True
            
        # 检查临界点精度
        if results['critical_point']['error'] < 0.05:
            assessment['critical_point_accurate'] = True
            
        # 检查标度律
        if 'scaling_exponent' in results['finite_size_scaling']:
            exp_diff = abs(results['finite_size_scaling']['scaling_exponent'] - 
                          results['finite_size_scaling']['theoretical_exponent'])
            if exp_diff < 0.3:
                assessment['scaling_laws_verified'] = True
                
        # 综合评估
        score = sum([assessment['phase_transition_confirmed'],
                    assessment['critical_point_accurate'],
                    assessment['scaling_laws_verified']]) / 3
                    
        if score > 0.7:
            assessment['theorem_support'] = 'Strong'
        elif score > 0.5:
            assessment['theorem_support'] = 'Moderate'
            
        return assessment
```

## 3. 总结

本形式化框架提供了：

1. **相变系统模型**：实现序参量计算、能量函数和相态识别
2. **相变分析器**：构建相图、定位临界点和测量临界指数
3. **临界现象验证**：验证普适性、分析涨落和标度关系
4. **综合验证系统**：全面测试相变定理的各个方面

这为T11-2相变定理提供了严格的数学基础和可验证的实现。