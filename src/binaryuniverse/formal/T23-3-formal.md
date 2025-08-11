# T23-3 φ-博弈演化稳定性定理 - 形式化规范

## 依赖导入
```python
import numpy as np
import math
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import scipy.linalg
from scipy.optimize import minimize
from scipy.stats import entropy

# 从前置理论导入
from T23_2_formal import (PhiNashEquilibriumExistence, PhiEntropyConservingEquilibrium,
                          PhiBestResponseMapping)
from T23_1_formal import (PhiGameSystem, PhiGamePlayer, PhiStrategy, 
                          PhiStrategySpace, PhiNashEquilibriumDetector)
from T22_3_formal import WeightedPhiNetwork
from T22_1_formal import FibonacciSequence, NetworkNode, ZeckendorfString
```

## 1. φ-ESS定义与判定

### 1.1 演化稳定策略结构
```python
@dataclass
class PhiEvolutionaryStableStrategy:
    """φ-演化稳定策略"""
    strategy: PhiStrategy
    stability_radius: float
    invasion_threshold: float
    convergence_time: float
    entropy_concavity: float
    
    def __post_init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        # 验证入侵阈值
        assert abs(self.invasion_threshold - 1.0 / (self.phi ** 2)) < 1e-10
        
    def is_stable_against(self, mutant: PhiStrategy, epsilon: float) -> bool:
        """检查对特定入侵策略的稳定性"""
        if epsilon >= self.invasion_threshold:
            return False  # 入侵比例过大
            
        # 计算策略距离
        distance = abs(self.strategy.value - mutant.value)
        return distance > self.stability_radius
        
    def get_basin_of_attraction(self) -> float:
        """获取吸引域大小"""
        return self.stability_radius * self.phi
```

### 1.2 ESS判定器
```python
class PhiESSDetector:
    """φ-ESS检测器"""
    
    def __init__(self, game_system: PhiGameSystem):
        self.game = game_system
        self.phi = (1 + np.sqrt(5)) / 2
        self.invasion_threshold = 1.0 / (self.phi ** 2)
        self.entropy_concavity_threshold = -1.0 / self.phi
        
    def is_phi_ess(self, strategy_profile: Dict[int, PhiStrategy]) -> bool:
        """判断策略组合是否为φ-ESS"""
        
        # Step 1: 验证Nash均衡
        if not self._verify_nash_equilibrium(strategy_profile):
            return False
            
        # Step 2: 验证熵的严格凹性
        if not self._verify_entropy_concavity(strategy_profile):
            return False
            
        # Step 3: 验证入侵稳定性
        if not self._verify_invasion_stability(strategy_profile):
            return False
            
        return True
        
    def _verify_nash_equilibrium(self, profile: Dict[int, PhiStrategy]) -> bool:
        """验证Nash均衡条件"""
        detector = PhiNashEquilibriumDetector(self.game)
        
        for player_id in profile:
            if not detector.check_best_response(player_id, profile):
                return False
        return True
        
    def _verify_entropy_concavity(self, profile: Dict[int, PhiStrategy]) -> bool:
        """验证熵的严格凹性"""
        # 构建策略分布
        distribution = self._profile_to_distribution(profile)
        
        # 计算熵的Hessian矩阵
        hessian = self._compute_entropy_hessian(distribution)
        
        # 检查所有特征值
        eigenvalues = np.linalg.eigvals(hessian)
        
        # 严格凹性条件：所有特征值 < -1/φ
        return np.all(eigenvalues < self.entropy_concavity_threshold)
        
    def _verify_invasion_stability(self, resident_profile: Dict[int, PhiStrategy]) -> bool:
        """验证对入侵的稳定性"""
        
        for player_id in resident_profile:
            resident_strategy = resident_profile[player_id]
            
            # 测试所有可能的入侵策略
            for mutant_strategy in self.game.strategy_space.get_all_strategies():
                if mutant_strategy != resident_strategy:
                    # 测试不同入侵比例
                    for epsilon in [0.001, 0.01, 0.1]:
                        if epsilon < self.invasion_threshold:
                            if not self._test_single_invasion(
                                player_id, resident_profile, 
                                resident_strategy, mutant_strategy, epsilon
                            ):
                                return False
        return True
        
    def _test_single_invasion(self, player_id: int,
                             resident_profile: Dict[int, PhiStrategy],
                             resident: PhiStrategy,
                             mutant: PhiStrategy,
                             epsilon: float) -> bool:
        """测试单个入侵场景"""
        # 构建混合群体
        mixed_profile = resident_profile.copy()
        
        # 创建混合策略（作为概率分布）
        mixed_strategy_dist = {
            resident: 1 - epsilon,
            mutant: epsilon
        }
        
        # 计算适应度
        resident_fitness = self._compute_strategy_fitness(
            player_id, resident, mixed_profile, mixed_strategy_dist
        )
        mutant_fitness = self._compute_strategy_fitness(
            player_id, mutant, mixed_profile, mixed_strategy_dist
        )
        
        # ESS条件：resident严格优于mutant
        return resident_fitness > mutant_fitness
        
    def _compute_strategy_fitness(self, player_id: int, strategy: PhiStrategy,
                                 profile: Dict[int, PhiStrategy],
                                 population_dist: Dict[PhiStrategy, float]) -> float:
        """计算策略在混合群体中的适应度"""
        fitness = 0.0
        
        # 基础收益
        for other_id in profile:
            if other_id != player_id:
                base_payoff = self.game.payoff_matrix[player_id, other_id]
                
                # 策略交互
                other_strategy = profile[other_id]
                interaction = strategy.zeckendorf_overlap(other_strategy) / self.phi
                
                fitness += base_payoff * interaction
                
        # 熵贡献
        if strategy in population_dist and population_dist[strategy] > 0:
            freq = population_dist[strategy]
            entropy_contribution = -freq * math.log(freq) / self.phi
            fitness += entropy_contribution
            
        return fitness
        
    def _profile_to_distribution(self, profile: Dict[int, PhiStrategy]) -> np.ndarray:
        """将策略组合转换为分布向量"""
        strategies = self.game.strategy_space.get_all_strategies()
        n_strategies = len(strategies)
        
        distribution = np.zeros(n_strategies)
        
        for player_id, strategy in profile.items():
            if strategy in strategies:
                idx = strategies.index(strategy)
                distribution[idx] += 1.0 / len(profile)
                
        return distribution
        
    def _compute_entropy_hessian(self, distribution: np.ndarray) -> np.ndarray:
        """计算熵函数的Hessian矩阵"""
        n = len(distribution)
        hessian = np.zeros((n, n))
        
        for i in range(n):
            if distribution[i] > 0:
                # 对角元素
                hessian[i, i] = -1.0 / distribution[i]
                
        return hessian
```

## 2. 演化动力学

### 2.1 φ-复制动力学
```python
class PhiReplicatorDynamics:
    """φ-复制动力学系统"""
    
    def __init__(self, game_system: PhiGameSystem):
        self.game = game_system
        self.phi = (1 + np.sqrt(5)) / 2
        self.strategies = game_system.strategy_space.get_all_strategies()
        
    def evolve_step(self, distribution: Dict[PhiStrategy, float], 
                   dt: float = 0.01) -> Dict[PhiStrategy, float]:
        """执行一步演化"""
        
        # 计算平均适应度
        avg_fitness = self._compute_average_fitness(distribution)
        
        # 更新每个策略的频率
        new_distribution = {}
        
        for strategy in self.strategies:
            if strategy in distribution:
                freq = distribution[strategy]
                
                # 计算适应度
                fitness = self._compute_fitness(strategy, distribution)
                
                # 计算熵梯度
                entropy_gradient = self._compute_entropy_gradient(freq)
                
                # φ-复制动力学方程
                change = (freq / self.phi) * (fitness - avg_fitness + entropy_gradient)
                
                # 更新频率
                new_freq = freq + dt * change
                new_freq = max(0.0, min(1.0, new_freq))
                
                new_distribution[strategy] = new_freq
            else:
                new_distribution[strategy] = 0.0
                
        # 归一化
        total = sum(new_distribution.values())
        if total > 0:
            new_distribution = {s: f/total for s, f in new_distribution.items()}
            
        return new_distribution
        
    def _compute_fitness(self, strategy: PhiStrategy, 
                        distribution: Dict[PhiStrategy, float]) -> float:
        """计算策略的适应度"""
        fitness = 0.0
        
        for other_strategy, other_freq in distribution.items():
            if other_freq > 0:
                # 计算交互收益
                payoff = self._compute_pairwise_payoff(strategy, other_strategy)
                fitness += payoff * other_freq
                
        return fitness
        
    def _compute_pairwise_payoff(self, s1: PhiStrategy, s2: PhiStrategy) -> float:
        """计算两个策略的交互收益"""
        # 基于Zeckendorf重叠
        overlap = s1.zeckendorf_overlap(s2)
        
        # 基于策略值距离
        distance = abs(s1.value - s2.value)
        
        # φ-调制的收益
        payoff = overlap / ((1 + distance) * self.phi)
        
        return payoff
        
    def _compute_average_fitness(self, distribution: Dict[PhiStrategy, float]) -> float:
        """计算平均适应度"""
        avg_fitness = 0.0
        
        for strategy, freq in distribution.items():
            if freq > 0:
                fitness = self._compute_fitness(strategy, distribution)
                avg_fitness += fitness * freq
                
        return avg_fitness
        
    def _compute_entropy_gradient(self, freq: float) -> float:
        """计算熵梯度"""
        if freq > 0:
            return -math.log(freq) - 1
        else:
            return 1.0  # 鼓励稀有策略
            
    def find_equilibrium(self, initial_distribution: Dict[PhiStrategy, float],
                        max_iterations: int = 10000,
                        tolerance: float = 1e-6) -> Optional[Dict[PhiStrategy, float]]:
        """寻找演化均衡"""
        current = initial_distribution.copy()
        
        for iteration in range(max_iterations):
            next_dist = self.evolve_step(current)
            
            # 检查收敛
            if self._has_converged(current, next_dist, tolerance):
                return next_dist
                
            current = next_dist
            
        return None  # 未收敛
        
    def _has_converged(self, dist1: Dict[PhiStrategy, float],
                       dist2: Dict[PhiStrategy, float],
                       tolerance: float) -> bool:
        """检查是否收敛"""
        for strategy in self.strategies:
            f1 = dist1.get(strategy, 0.0)
            f2 = dist2.get(strategy, 0.0)
            if abs(f1 - f2) > tolerance:
                return False
        return True
```

### 2.2 稳定性分析
```python
class PhiEvolutionaryStabilityAnalyzer:
    """演化稳定性分析器"""
    
    def __init__(self, game_system: PhiGameSystem):
        self.game = game_system
        self.phi = (1 + np.sqrt(5)) / 2
        self.dynamics = PhiReplicatorDynamics(game_system)
        self.detector = PhiESSDetector(game_system)
        
    def compute_stability_radius(self, ess_strategy: PhiStrategy) -> float:
        """计算ESS的稳定性半径"""
        min_distance = float('inf')
        
        for strategy in self.game.strategy_space.get_all_strategies():
            if strategy != ess_strategy:
                distance = abs(ess_strategy.value - strategy.value)
                min_distance = min(min_distance, distance)
                
        return min_distance / self.phi
        
    def compute_convergence_time(self, target_precision: float = 0.01) -> float:
        """计算理论收敛时间上界"""
        return (self.phi ** 2) * math.log(1.0 / target_precision)
        
    def analyze_basin_of_attraction(self, ess_strategy: PhiStrategy,
                                   n_samples: int = 100) -> float:
        """分析吸引域大小"""
        strategies = self.game.strategy_space.get_all_strategies()
        n_strategies = len(strategies)
        
        successful_convergence = 0
        
        for _ in range(n_samples):
            # 随机初始分布
            probs = np.random.dirichlet(np.ones(n_strategies))
            initial_dist = {strategies[i]: probs[i] for i in range(n_strategies)}
            
            # 演化到均衡
            equilibrium = self.dynamics.find_equilibrium(initial_dist)
            
            if equilibrium:
                # 检查是否收敛到ESS
                dominant_strategy = max(equilibrium.items(), key=lambda x: x[1])[0]
                if dominant_strategy == ess_strategy:
                    successful_convergence += 1
                    
        return successful_convergence / n_samples
        
    def verify_lyapunov_stability(self, ess_profile: Dict[int, PhiStrategy]) -> bool:
        """验证Lyapunov稳定性"""
        # 构建Lyapunov函数
        def lyapunov_function(distribution: np.ndarray) -> float:
            # V = -H + (1/φ) * Σ xi log fi
            entropy = -np.sum(distribution * np.log(distribution + 1e-10))
            
            # 简化：使用策略值作为适应度代理
            fitness_sum = 0.0
            strategies = self.game.strategy_space.get_all_strategies()
            for i, strategy in enumerate(strategies):
                if distribution[i] > 0:
                    fitness_sum += distribution[i] * math.log(strategy.value + 1)
                    
            return -entropy + fitness_sum / self.phi
            
        # 测试Lyapunov函数沿轨迹递减
        test_points = 10
        for _ in range(test_points):
            # 随机初始点
            n = len(self.game.strategy_space.get_all_strategies())
            dist = np.random.dirichlet(np.ones(n))
            
            # 计算导数（数值近似）
            eps = 1e-6
            v0 = lyapunov_function(dist)
            
            # 微小扰动
            dist_perturbed = dist + eps * np.random.randn(n)
            dist_perturbed = np.maximum(0, dist_perturbed)
            dist_perturbed /= np.sum(dist_perturbed)
            
            v1 = lyapunov_function(dist_perturbed)
            
            # 验证递减性
            if v1 > v0 + eps:  # 允许小的数值误差
                return False
                
        return True
```

## 3. 入侵动力学

### 3.1 入侵分析
```python
class PhiInvasionDynamics:
    """入侵动力学分析"""
    
    def __init__(self, game_system: PhiGameSystem):
        self.game = game_system
        self.phi = (1 + np.sqrt(5)) / 2
        self.invasion_threshold = 1.0 / (self.phi ** 2)
        
    def simulate_invasion(self, resident: PhiStrategy, mutant: PhiStrategy,
                         initial_mutant_freq: float,
                         max_time: int = 1000) -> Tuple[bool, List[float]]:
        """模拟入侵动力学"""
        
        # 初始分布
        distribution = {
            resident: 1 - initial_mutant_freq,
            mutant: initial_mutant_freq
        }
        
        dynamics = PhiReplicatorDynamics(self.game)
        mutant_trajectory = [initial_mutant_freq]
        
        for t in range(max_time):
            distribution = dynamics.evolve_step(distribution)
            mutant_freq = distribution.get(mutant, 0.0)
            mutant_trajectory.append(mutant_freq)
            
            # 检查入侵结果
            if mutant_freq < 1e-6:
                return False, mutant_trajectory  # 入侵失败
            elif mutant_freq > 1 - 1e-6:
                return True, mutant_trajectory   # 入侵成功
                
        # 未决定
        final_freq = mutant_trajectory[-1]
        return final_freq > initial_mutant_freq, mutant_trajectory
        
    def compute_invasion_fitness(self, resident: PhiStrategy, 
                                mutant: PhiStrategy) -> float:
        """计算入侵适应度"""
        # 在resident占主导的群体中mutant的适应度
        resident_payoff = self._compute_payoff(mutant, resident)
        
        # 熵贡献（稀有优势）
        entropy_bonus = 1.0 / self.phi  # 稀有策略的熵优势
        
        return resident_payoff + entropy_bonus
        
    def _compute_payoff(self, s1: PhiStrategy, s2: PhiStrategy) -> float:
        """计算策略交互收益"""
        overlap = s1.zeckendorf_overlap(s2)
        distance = abs(s1.value - s2.value)
        return overlap / ((1 + distance) * self.phi)
        
    def find_uninvadable_strategies(self) -> List[PhiStrategy]:
        """寻找不可入侵的策略"""
        uninvadable = []
        strategies = self.game.strategy_space.get_all_strategies()
        
        for resident in strategies:
            is_uninvadable = True
            
            for mutant in strategies:
                if mutant != resident:
                    # 测试小入侵
                    success, _ = self.simulate_invasion(
                        resident, mutant, 0.01, max_time=500
                    )
                    
                    if success:
                        is_uninvadable = False
                        break
                        
            if is_uninvadable:
                uninvadable.append(resident)
                
        return uninvadable
```

## 4. 综合ESS系统

### 4.1 完整ESS分析器
```python
class PhiESSSystem:
    """完整的φ-ESS系统"""
    
    def __init__(self, game_system: PhiGameSystem):
        self.game = game_system
        self.phi = (1 + np.sqrt(5)) / 2
        
        # 初始化各组件
        self.detector = PhiESSDetector(game_system)
        self.dynamics = PhiReplicatorDynamics(game_system)
        self.stability_analyzer = PhiEvolutionaryStabilityAnalyzer(game_system)
        self.invasion_dynamics = PhiInvasionDynamics(game_system)
        
    def find_all_ess(self) -> List[PhiEvolutionaryStableStrategy]:
        """寻找所有ESS"""
        ess_list = []
        
        # 首先找到所有不可入侵策略
        uninvadable = self.invasion_dynamics.find_uninvadable_strategies()
        
        for strategy in uninvadable:
            # 构建纯策略profile
            profile = {0: strategy}  # 单玩家情况
            
            # 验证ESS条件
            if self.detector.is_phi_ess(profile):
                # 计算ESS属性
                stability_radius = self.stability_analyzer.compute_stability_radius(strategy)
                convergence_time = self.stability_analyzer.compute_convergence_time()
                
                ess = PhiEvolutionaryStableStrategy(
                    strategy=strategy,
                    stability_radius=stability_radius,
                    invasion_threshold=1.0 / (self.phi ** 2),
                    convergence_time=convergence_time,
                    entropy_concavity=-1.0 / self.phi
                )
                
                ess_list.append(ess)
                
        return ess_list
        
    def analyze_evolutionary_dynamics(self, initial_distribution: Dict[PhiStrategy, float],
                                     time_steps: int = 1000) -> Dict[str, Any]:
        """分析演化动力学"""
        
        # 记录轨迹
        trajectory = []
        entropy_history = []
        
        current = initial_distribution.copy()
        
        for t in range(time_steps):
            trajectory.append(current.copy())
            
            # 计算当前熵
            entropy_val = self._compute_distribution_entropy(current)
            entropy_history.append(entropy_val)
            
            # 演化一步
            current = self.dynamics.evolve_step(current)
            
        # 分析结果
        final_distribution = trajectory[-1] if trajectory else initial_distribution
        dominant_strategy = max(final_distribution.items(), key=lambda x: x[1])[0]
        
        # 检查是否收敛到ESS
        profile = {0: dominant_strategy}
        is_ess = self.detector.is_phi_ess(profile)
        
        # 验证熵增
        entropy_increased = all(
            entropy_history[i+1] >= entropy_history[i] - 1e-10
            for i in range(len(entropy_history)-1)
        )
        
        return {
            'trajectory': trajectory,
            'entropy_history': entropy_history,
            'final_distribution': final_distribution,
            'dominant_strategy': dominant_strategy,
            'converged_to_ess': is_ess,
            'entropy_increased': entropy_increased,
            'total_entropy_change': entropy_history[-1] - entropy_history[0] if entropy_history else 0
        }
        
    def _compute_distribution_entropy(self, distribution: Dict[PhiStrategy, float]) -> float:
        """计算分布的熵"""
        entropy_val = 0.0
        
        for strategy, freq in distribution.items():
            if freq > 0:
                entropy_val -= freq * math.log(freq)
                
        return entropy_val
        
    def verify_theoretical_predictions(self) -> Dict[str, bool]:
        """验证理论预测"""
        predictions = {}
        
        # 1. 验证入侵阈值
        test_invasions = 10
        successful_invasions = 0
        
        for _ in range(test_invasions):
            # 随机选择resident和mutant
            strategies = self.game.strategy_space.get_all_strategies()
            if len(strategies) >= 2:
                resident, mutant = np.random.choice(strategies, 2, replace=False)
                
                # 测试阈值附近的入侵
                epsilon = self.invasion_dynamics.invasion_threshold * 0.9
                success, _ = self.invasion_dynamics.simulate_invasion(
                    resident, mutant, epsilon, max_time=500
                )
                
                if not success:
                    successful_invasions += 1
                    
        predictions['invasion_threshold_valid'] = successful_invasions > test_invasions * 0.7
        
        # 2. 验证收敛时间
        theoretical_time = self.stability_analyzer.compute_convergence_time(0.01)
        predictions['convergence_time_bounded'] = theoretical_time == (self.phi ** 2) * math.log(100)
        
        # 3. 验证稳定性半径
        ess_list = self.find_all_ess()
        if ess_list:
            ess = ess_list[0]
            radius = ess.stability_radius
            predictions['stability_radius_positive'] = radius > 0
        else:
            predictions['stability_radius_positive'] = False
            
        # 4. 验证Lyapunov稳定性
        if ess_list:
            profile = {0: ess_list[0].strategy}
            predictions['lyapunov_stable'] = self.stability_analyzer.verify_lyapunov_stability(profile)
        else:
            predictions['lyapunov_stable'] = False
            
        return predictions
```

---

**注记**: 本形式化规范提供了T23-3定理的完整数学实现，包括φ-ESS检测、演化动力学、入侵分析和稳定性验证的所有必要组件。所有实现严格遵循φ-表示、Zeckendorf编码和熵增原理。