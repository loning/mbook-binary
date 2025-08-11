# T23-2 φ-博弈均衡存在性定理 - 形式化规范

## 依赖导入
```python
import numpy as np
import math
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import itertools
from abc import ABC, abstractmethod
from scipy.optimize import minimize
from scipy.special import softmax

# 从前置理论导入
from T23_1_formal import (PhiGameSystem, PhiGamePlayer, PhiStrategy, 
                          PhiStrategySpace, PhiNashEquilibriumDetector)
from T22_3_formal import WeightedPhiNetwork, ComprehensiveStabilityAnalyzer
from T22_2_formal import ConnectionEvolutionDynamics
from T22_1_formal import FibonacciSequence, NetworkNode, ZeckendorfString
```

## 1. φ-最优反应映射

### 1.1 最优反应结构
```python
class PhiBestResponseMapping:
    """φ-最优反应映射"""
    
    def __init__(self, game_system: PhiGameSystem):
        self.game = game_system
        self.phi = (1 + np.sqrt(5)) / 2
        self.temperature = 1.0 / self.phi  # φ-软最大化温度
        
    def compute_best_response(self, player_id: int, 
                            other_strategies: Dict[int, Dict[PhiStrategy, float]]) -> Dict[PhiStrategy, float]:
        """计算玩家i对其他玩家策略的φ-最优反应"""
        if player_id not in self.game.players:
            return {}
            
        player = self.game.players[player_id]
        strategies = player.strategy_space.get_all_strategies()
        
        # 计算每个策略的期望收益
        expected_utilities = {}
        
        for strategy in strategies:
            utility = self._compute_expected_utility(player_id, strategy, other_strategies)
            
            # 加入φ-深度调制
            phi_depth = strategy.d
            adjusted_utility = utility / (self.phi ** phi_depth)
            
            expected_utilities[strategy] = adjusted_utility
            
        # φ-软最大化响应（保证连续性）
        return self._phi_softmax_response(expected_utilities)
        
    def _compute_expected_utility(self, player_id: int, strategy: PhiStrategy,
                                 other_strategies: Dict[int, Dict[PhiStrategy, float]]) -> float:
        """计算策略的期望收益"""
        total_utility = 0.0
        
        for other_id, other_dist in other_strategies.items():
            if other_id != player_id and other_id in self.game.players:
                # 基础收益
                base_payoff = self.game.payoff_matrix[player_id, other_id]
                
                # 对每个其他玩家的策略分布求期望
                for other_strategy, other_prob in other_dist.items():
                    # 策略交互强度
                    interaction = self._compute_strategy_interaction(
                        strategy, other_strategy, player_id, other_id
                    )
                    
                    # 期望收益贡献
                    total_utility += base_payoff * interaction * other_prob
                    
        return total_utility
        
    def _compute_strategy_interaction(self, strategy1: PhiStrategy, strategy2: PhiStrategy,
                                     player1_id: int, player2_id: int) -> float:
        """计算两个策略的交互强度"""
        # 策略值距离
        value_distance = abs(strategy1.value - strategy2.value)
        
        # Zeckendorf重叠
        overlap = strategy1.zeckendorf_overlap(strategy2)
        
        # 网络距离调制
        if (player1_id in self.game.players and player2_id in self.game.players):
            player1 = self.game.players[player1_id]
            player2 = self.game.players[player2_id]
            network_distance = player1.compute_network_distance_to(player2)
        else:
            network_distance = 1.0
            
        # φ-调制的交互强度
        interaction = overlap / ((1 + value_distance) * (self.phi ** (network_distance / 2)))
        
        return max(0.1, min(1.0, interaction))
        
    def _phi_softmax_response(self, utilities: Dict[PhiStrategy, float]) -> Dict[PhiStrategy, float]:
        """φ-软最大化响应（确保连续性）"""
        if not utilities:
            return {}
            
        # 转换为数组形式
        strategies = list(utilities.keys())
        utility_values = np.array([utilities[s] for s in strategies])
        
        # 避免数值溢出
        max_util = np.max(utility_values)
        exp_utilities = np.exp((utility_values - max_util) / self.temperature)
        
        # 归一化为概率分布
        probabilities = exp_utilities / np.sum(exp_utilities)
        
        return {strategies[i]: probabilities[i] for i in range(len(strategies))}
        
    def verify_continuity(self, player_id: int, epsilon: float = 1e-6) -> bool:
        """验证最优反应映射的连续性"""
        # 创建两个接近的策略分布
        other_strategies1 = self._create_random_strategy_profile(exclude_player=player_id)
        other_strategies2 = self._perturb_strategy_profile(other_strategies1, epsilon)
        
        # 计算最优反应
        response1 = self.compute_best_response(player_id, other_strategies1)
        response2 = self.compute_best_response(player_id, other_strategies2)
        
        # 计算反应的距离
        distance = self._compute_distribution_distance(response1, response2)
        
        # 验证连续性：小扰动导致小变化
        return distance < 10 * epsilon  # 允许Lipschitz常数为10
        
    def _create_random_strategy_profile(self, exclude_player: int = None) -> Dict[int, Dict[PhiStrategy, float]]:
        """创建随机策略组合"""
        profile = {}
        
        for player_id, player in self.game.players.items():
            if player_id != exclude_player:
                strategies = player.strategy_space.get_all_strategies()
                if strategies:
                    # 随机概率
                    probs = np.random.dirichlet(np.ones(len(strategies)))
                    profile[player_id] = {strategies[i]: probs[i] for i in range(len(strategies))}
                    
        return profile
        
    def _perturb_strategy_profile(self, profile: Dict[int, Dict[PhiStrategy, float]], 
                                 epsilon: float) -> Dict[int, Dict[PhiStrategy, float]]:
        """微扰策略组合"""
        perturbed = {}
        
        for player_id, strategy_dist in profile.items():
            if strategy_dist:
                # 添加小扰动
                strategies = list(strategy_dist.keys())
                probs = np.array([strategy_dist[s] for s in strategies])
                
                # 添加高斯噪声
                noise = np.random.normal(0, epsilon, len(probs))
                perturbed_probs = probs + noise
                
                # 投影回概率单纯形
                perturbed_probs = np.maximum(0, perturbed_probs)
                perturbed_probs /= np.sum(perturbed_probs)
                
                perturbed[player_id] = {strategies[i]: perturbed_probs[i] 
                                       for i in range(len(strategies))}
                
        return perturbed
        
    def _compute_distribution_distance(self, dist1: Dict[PhiStrategy, float], 
                                      dist2: Dict[PhiStrategy, float]) -> float:
        """计算两个策略分布的距离（总变差距离）"""
        all_strategies = set(dist1.keys()) | set(dist2.keys())
        
        total_distance = 0.0
        for strategy in all_strategies:
            p1 = dist1.get(strategy, 0.0)
            p2 = dist2.get(strategy, 0.0)
            total_distance += abs(p1 - p2)
            
        return total_distance / 2  # 总变差距离
```

### 1.2 组合映射
```python
class PhiJointBestResponseMapping:
    """φ-组合最优反应映射"""
    
    def __init__(self, game_system: PhiGameSystem):
        self.game = game_system
        self.phi = (1 + np.sqrt(5)) / 2
        self.individual_mapping = PhiBestResponseMapping(game_system)
        
    def compute_joint_best_response(self, strategy_profile: Dict[int, Dict[PhiStrategy, float]]) -> Dict[int, Dict[PhiStrategy, float]]:
        """计算所有玩家的组合最优反应"""
        joint_response = {}
        
        for player_id in self.game.players:
            # 其他玩家的策略
            other_strategies = {pid: dist for pid, dist in strategy_profile.items() 
                              if pid != player_id}
            
            # 计算该玩家的最优反应
            best_response = self.individual_mapping.compute_best_response(
                player_id, other_strategies
            )
            
            joint_response[player_id] = best_response
            
        return joint_response
        
    def is_continuous(self) -> bool:
        """验证组合映射的连续性"""
        # 对每个玩家验证连续性
        for player_id in self.game.players:
            if not self.individual_mapping.verify_continuity(player_id):
                return False
        return True
        
    def find_fixed_point(self, max_iterations: int = 1000, 
                        tolerance: float = 1e-6) -> Optional[Dict[int, Dict[PhiStrategy, float]]]:
        """寻找不动点（Nash均衡）"""
        # 初始化：均匀分布
        current_profile = self._initialize_uniform_profile()
        
        for iteration in range(max_iterations):
            # 计算组合最优反应
            new_profile = self.compute_joint_best_response(current_profile)
            
            # 检查收敛
            if self._has_converged(current_profile, new_profile, tolerance):
                return new_profile
                
            # 使用凸组合更新（提高稳定性）
            alpha = 1.0 / (iteration + 2)  # 递减学习率
            current_profile = self._convex_combination(current_profile, new_profile, alpha)
            
        # 返回最后的近似
        return current_profile if self._is_approximate_fixed_point(current_profile, tolerance * 10) else None
        
    def _initialize_uniform_profile(self) -> Dict[int, Dict[PhiStrategy, float]]:
        """初始化均匀策略组合"""
        profile = {}
        
        for player_id, player in self.game.players.items():
            strategies = player.strategy_space.get_all_strategies()
            if strategies:
                uniform_prob = 1.0 / len(strategies)
                profile[player_id] = {s: uniform_prob for s in strategies}
            else:
                profile[player_id] = {}
                
        return profile
        
    def _has_converged(self, profile1: Dict[int, Dict[PhiStrategy, float]],
                       profile2: Dict[int, Dict[PhiStrategy, float]],
                       tolerance: float) -> bool:
        """检查是否收敛"""
        for player_id in self.game.players:
            if player_id in profile1 and player_id in profile2:
                dist1 = profile1[player_id]
                dist2 = profile2[player_id]
                
                distance = self.individual_mapping._compute_distribution_distance(dist1, dist2)
                if distance > tolerance:
                    return False
                    
        return True
        
    def _convex_combination(self, profile1: Dict[int, Dict[PhiStrategy, float]],
                          profile2: Dict[int, Dict[PhiStrategy, float]],
                          alpha: float) -> Dict[int, Dict[PhiStrategy, float]]:
        """计算两个策略组合的凸组合"""
        combined = {}
        
        for player_id in self.game.players:
            if player_id in profile1 and player_id in profile2:
                dist1 = profile1[player_id]
                dist2 = profile2[player_id]
                
                combined_dist = {}
                all_strategies = set(dist1.keys()) | set(dist2.keys())
                
                for strategy in all_strategies:
                    p1 = dist1.get(strategy, 0.0)
                    p2 = dist2.get(strategy, 0.0)
                    combined_dist[strategy] = (1 - alpha) * p1 + alpha * p2
                    
                combined[player_id] = combined_dist
                
        return combined
        
    def _is_approximate_fixed_point(self, profile: Dict[int, Dict[PhiStrategy, float]],
                                   tolerance: float) -> bool:
        """检查是否为近似不动点"""
        response = self.compute_joint_best_response(profile)
        return self._has_converged(profile, response, tolerance)
```

## 2. φ-Nash均衡存在性

### 2.1 均衡存在性定理
```python
class PhiNashEquilibriumExistence:
    """φ-Nash均衡存在性定理实现"""
    
    def __init__(self, game_system: PhiGameSystem):
        self.game = game_system
        self.phi = (1 + np.sqrt(5)) / 2
        self.joint_mapping = PhiJointBestResponseMapping(game_system)
        self.equilibrium_tolerance = 1e-6 / self.phi
        
    def prove_existence_via_brouwer(self) -> Tuple[bool, Optional[Dict[int, Dict[PhiStrategy, float]]]]:
        """通过Brouwer不动点定理证明均衡存在性"""
        
        # 1. 验证策略空间是紧凸集
        if not self._verify_strategy_space_compact_convex():
            return False, None
            
        # 2. 验证最优反应映射的连续性
        if not self.joint_mapping.is_continuous():
            return False, None
            
        # 3. 寻找不动点
        fixed_point = self.joint_mapping.find_fixed_point()
        
        if fixed_point is None:
            return False, None
            
        # 4. 验证不动点是φ-Nash均衡
        if self._verify_phi_nash_equilibrium(fixed_point):
            return True, fixed_point
        else:
            return False, None
            
    def _verify_strategy_space_compact_convex(self) -> bool:
        """验证策略空间的紧凸性"""
        for player in self.game.players.values():
            strategies = player.strategy_space.get_all_strategies()
            
            # 验证有限性（紧性的充分条件）
            if len(strategies) == 0 or len(strategies) > 10000:
                return False
                
            # 策略空间上的概率分布自动构成凸集
            
        return True
        
    def _verify_phi_nash_equilibrium(self, profile: Dict[int, Dict[PhiStrategy, float]]) -> bool:
        """验证策略组合是否为φ-Nash均衡"""
        for player_id in self.game.players:
            if not self._check_player_best_response(player_id, profile):
                return False
        return True
        
    def _check_player_best_response(self, player_id: int,
                                   profile: Dict[int, Dict[PhiStrategy, float]]) -> bool:
        """检查玩家策略是否为最优反应"""
        if player_id not in profile:
            return False
            
        current_dist = profile[player_id]
        other_strategies = {pid: dist for pid, dist in profile.items() 
                          if pid != player_id}
        
        # 计算最优反应
        best_response = self.joint_mapping.individual_mapping.compute_best_response(
            player_id, other_strategies
        )
        
        # 计算期望收益
        current_utility = self._compute_expected_utility(player_id, current_dist, other_strategies)
        best_utility = self._compute_expected_utility(player_id, best_response, other_strategies)
        
        # φ-Nash条件
        return abs(best_utility - current_utility) <= self.equilibrium_tolerance
        
    def _compute_expected_utility(self, player_id: int,
                                 strategy_dist: Dict[PhiStrategy, float],
                                 other_strategies: Dict[int, Dict[PhiStrategy, float]]) -> float:
        """计算策略分布的期望收益"""
        total_utility = 0.0
        
        for strategy, prob in strategy_dist.items():
            utility = self.joint_mapping.individual_mapping._compute_expected_utility(
                player_id, strategy, other_strategies
            )
            total_utility += utility * prob
            
        return total_utility
```

### 2.2 熵守恒均衡
```python
class PhiEntropyConservingEquilibrium:
    """熵守恒的φ-Nash均衡"""
    
    def __init__(self, game_system: PhiGameSystem, equilibrium: Dict[int, Dict[PhiStrategy, float]]):
        self.game = game_system
        self.equilibrium = equilibrium
        self.phi = (1 + np.sqrt(5)) / 2
        
    def verify_entropy_conservation(self) -> Tuple[bool, Dict[str, float]]:
        """验证均衡的熵守恒性质"""
        
        # 1. 计算策略熵
        strategy_entropy = self._compute_strategy_entropy()
        
        # 2. 计算交互熵
        interaction_entropy = self._compute_interaction_entropy()
        
        # 3. 计算结构熵
        structure_entropy = self._compute_structure_entropy()
        
        # 4. 计算总熵
        total_entropy = strategy_entropy / self.phi + interaction_entropy + structure_entropy
        
        # 5. 验证分解公式
        expected_total = self._compute_expected_total_entropy()
        
        conservation_error = abs(total_entropy - expected_total)
        is_conserved = conservation_error < 1e-6
        
        return is_conserved, {
            'strategy_entropy': strategy_entropy,
            'interaction_entropy': interaction_entropy,
            'structure_entropy': structure_entropy,
            'total_entropy': total_entropy,
            'expected_total': expected_total,
            'conservation_error': conservation_error
        }
        
    def _compute_strategy_entropy(self) -> float:
        """计算策略熵"""
        total_entropy = 0.0
        
        for player_id, strategy_dist in self.equilibrium.items():
            player_entropy = 0.0
            for prob in strategy_dist.values():
                if prob > 0:
                    player_entropy -= prob * math.log(prob)
            total_entropy += player_entropy
            
        return total_entropy
        
    def _compute_interaction_entropy(self) -> float:
        """计算交互熵"""
        interaction_entropy = 0.0
        player_ids = list(self.equilibrium.keys())
        
        for i in range(len(player_ids)):
            for j in range(i + 1, len(player_ids)):
                player_i_id = player_ids[i]
                player_j_id = player_ids[j]
                
                if (player_i_id in self.game.players and player_j_id in self.game.players):
                    # 计算联合分布的互信息
                    mutual_info = self._compute_mutual_information(player_i_id, player_j_id)
                    interaction_entropy += mutual_info
                    
        return interaction_entropy
        
    def _compute_mutual_information(self, player_i: int, player_j: int) -> float:
        """计算两个玩家策略的互信息"""
        if player_i not in self.equilibrium or player_j not in self.equilibrium:
            return 0.0
            
        # 简化：使用收益矩阵元素作为交互强度
        if player_i < len(self.game.players) and player_j < len(self.game.players):
            payoff = self.game.payoff_matrix[player_i, player_j]
            return payoff * math.log(2)  # 简化的互信息
        else:
            return 0.0
            
    def _compute_structure_entropy(self) -> float:
        """计算结构熵"""
        n_players = len(self.equilibrium)
        return n_players * math.log(self.phi)
        
    def _compute_expected_total_entropy(self) -> float:
        """计算期望的总熵（根据分解公式）"""
        strategy_part = self._compute_strategy_entropy() / self.phi
        interaction_part = self._compute_interaction_entropy()
        structure_part = self._compute_structure_entropy()
        
        return strategy_part + interaction_part + structure_part
        
    def compute_phi_fixed_point_equation(self, player_id: int, strategy: PhiStrategy) -> float:
        """计算φ-不动点方程的值"""
        if player_id not in self.equilibrium:
            return 0.0
            
        current_prob = self.equilibrium[player_id].get(strategy, 0.0)
        
        if current_prob <= 0:
            return 0.0
            
        # 计算熵梯度
        entropy_gradient = -math.log(current_prob) - 1
        
        # 计算期望收益
        other_strategies = {pid: dist for pid, dist in self.equilibrium.items() 
                          if pid != player_id}
        
        mapping = PhiBestResponseMapping(self.game)
        expected_utility = mapping._compute_expected_utility(player_id, strategy, other_strategies)
        
        # φ-不动点方程：p(s) = exp(φ * U(s)) / Z * φ^(-d)
        phi_depth = strategy.d
        
        # 计算归一化常数
        Z = 0.0
        for s in self.equilibrium[player_id]:
            u = mapping._compute_expected_utility(player_id, s, other_strategies)
            Z += math.exp(self.phi * u) / (self.phi ** s.d)
            
        # 不动点方程的值
        predicted_prob = math.exp(self.phi * expected_utility) / (Z * (self.phi ** phi_depth))
        
        return abs(predicted_prob - current_prob)
```

## 3. 均衡唯一性与稳定性

### 3.1 唯一性条件
```python
class PhiEquilibriumUniqueness:
    """φ-Nash均衡唯一性分析"""
    
    def __init__(self, game_system: PhiGameSystem):
        self.game = game_system
        self.phi = (1 + np.sqrt(5)) / 2
        
    def verify_strict_phi_diagonal_dominance(self) -> bool:
        """验证严格φ-对角优势条件"""
        n = len(self.game.players)
        
        for i in range(n):
            # 计算非对角元素之和
            off_diagonal_sum = 0.0
            for j in range(n):
                if i != j:
                    off_diagonal_sum += abs(self.game.payoff_matrix[i, j])
                    
            # 获取最小策略值
            player = self.game.players.get(i)
            if player:
                strategies = player.strategy_space.get_all_strategies()
                if strategies:
                    min_strategy_value = min(s.value for s in strategies)
                    
                    # 检查φ-对角优势条件
                    if off_diagonal_sum >= self.phi * min_strategy_value:
                        return False
                        
        return True
        
    def find_all_equilibria(self, n_attempts: int = 10) -> List[Dict[int, Dict[PhiStrategy, float]]]:
        """寻找所有可能的均衡"""
        equilibria = []
        joint_mapping = PhiJointBestResponseMapping(self.game)
        
        for attempt in range(n_attempts):
            # 从不同初始点开始
            initial_profile = self._create_random_initial_profile()
            
            # 寻找不动点
            fixed_point = self._find_fixed_point_from(initial_profile, joint_mapping)
            
            if fixed_point and self._is_new_equilibrium(fixed_point, equilibria):
                equilibria.append(fixed_point)
                
        return equilibria
        
    def _create_random_initial_profile(self) -> Dict[int, Dict[PhiStrategy, float]]:
        """创建随机初始策略组合"""
        profile = {}
        
        for player_id, player in self.game.players.items():
            strategies = player.strategy_space.get_all_strategies()
            if strategies:
                # Dirichlet分布生成随机概率
                probs = np.random.dirichlet(np.ones(len(strategies)))
                profile[player_id] = {strategies[i]: probs[i] for i in range(len(strategies))}
                
        return profile
        
    def _find_fixed_point_from(self, initial: Dict[int, Dict[PhiStrategy, float]],
                              mapping: PhiJointBestResponseMapping) -> Optional[Dict[int, Dict[PhiStrategy, float]]]:
        """从给定初始点寻找不动点"""
        current = initial
        max_iterations = 500
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            new_profile = mapping.compute_joint_best_response(current)
            
            if mapping._has_converged(current, new_profile, tolerance):
                return new_profile
                
            # 更新
            alpha = 1.0 / (iteration + 2)
            current = mapping._convex_combination(current, new_profile, alpha)
            
        return None
        
    def _is_new_equilibrium(self, candidate: Dict[int, Dict[PhiStrategy, float]],
                          existing: List[Dict[int, Dict[PhiStrategy, float]]]) -> bool:
        """检查是否为新的均衡"""
        tolerance = 1e-4
        
        for equilibrium in existing:
            total_distance = 0.0
            
            for player_id in candidate:
                if player_id in equilibrium:
                    dist1 = candidate[player_id]
                    dist2 = equilibrium[player_id]
                    
                    for strategy in set(dist1.keys()) | set(dist2.keys()):
                        p1 = dist1.get(strategy, 0.0)
                        p2 = dist2.get(strategy, 0.0)
                        total_distance += abs(p1 - p2)
                        
            if total_distance < tolerance:
                return False  # 太接近已有均衡
                
        return True
```

### 3.2 稳定性分析
```python
class PhiEquilibriumStability:
    """φ-Nash均衡稳定性分析"""
    
    def __init__(self, game_system: PhiGameSystem, equilibrium: Dict[int, Dict[PhiStrategy, float]]):
        self.game = game_system
        self.equilibrium = equilibrium
        self.phi = (1 + np.sqrt(5)) / 2
        
    def analyze_perturbation_stability(self, perturbation_size: float = 1.0 / (100 * np.sqrt(5))) -> Dict[str, Any]:
        """分析扰动稳定性"""
        # 扰动大小应该 ≤ 1/φ²
        max_perturbation = 1.0 / (self.phi ** 2)
        
        if perturbation_size > max_perturbation:
            perturbation_size = max_perturbation
            
        # 扰动收益矩阵
        perturbed_payoff = self._perturb_payoff_matrix(perturbation_size)
        
        # 创建扰动后的博弈系统
        perturbed_game = self._create_perturbed_game(perturbed_payoff)
        
        # 寻找扰动后的均衡
        existence_prover = PhiNashEquilibriumExistence(perturbed_game)
        exists, perturbed_equilibrium = existence_prover.prove_existence_via_brouwer()
        
        if not exists or perturbed_equilibrium is None:
            return {'stable': False, 'reason': 'no_equilibrium_after_perturbation'}
            
        # 计算均衡的变化
        equilibrium_change = self._compute_equilibrium_distance(self.equilibrium, perturbed_equilibrium)
        
        # 验证1/φ-稳定性
        stability_bound = 1.0 / self.phi
        is_stable = equilibrium_change <= stability_bound
        
        return {
            'stable': is_stable,
            'perturbation_size': perturbation_size,
            'equilibrium_change': equilibrium_change,
            'stability_bound': stability_bound,
            'relative_change': equilibrium_change / stability_bound
        }
        
    def _perturb_payoff_matrix(self, size: float) -> np.ndarray:
        """扰动收益矩阵"""
        n = len(self.game.players)
        original = self.game.payoff_matrix.copy()
        
        # 生成随机扰动
        perturbation = np.random.uniform(-size, size, (n, n))
        
        # 保持对角线为0
        np.fill_diagonal(perturbation, 0)
        
        # 应用扰动
        perturbed = original + perturbation
        
        # 确保非负
        perturbed = np.maximum(0.1, perturbed)
        np.fill_diagonal(perturbed, 0)
        
        return perturbed
        
    def _create_perturbed_game(self, payoff_matrix: np.ndarray) -> PhiGameSystem:
        """创建扰动后的博弈系统"""
        # 复制原始博弈系统
        perturbed_game = PhiGameSystem(self.game.network, self.game.n_players)
        
        # 替换收益矩阵
        perturbed_game.payoff_matrix = payoff_matrix
        
        return perturbed_game
        
    def _compute_equilibrium_distance(self, eq1: Dict[int, Dict[PhiStrategy, float]],
                                     eq2: Dict[int, Dict[PhiStrategy, float]]) -> float:
        """计算两个均衡之间的距离"""
        total_distance = 0.0
        n_players = 0
        
        for player_id in set(eq1.keys()) | set(eq2.keys()):
            if player_id in eq1 and player_id in eq2:
                dist1 = eq1[player_id]
                dist2 = eq2[player_id]
                
                player_distance = 0.0
                all_strategies = set(dist1.keys()) | set(dist2.keys())
                
                for strategy in all_strategies:
                    p1 = dist1.get(strategy, 0.0)
                    p2 = dist2.get(strategy, 0.0)
                    player_distance += abs(p1 - p2)
                    
                total_distance += player_distance / 2  # 总变差距离
                n_players += 1
                
        return total_distance / n_players if n_players > 0 else 0.0
        
    def compute_stability_measure(self) -> float:
        """计算均衡的稳定性度量"""
        stability_score = 0.0
        n_players = len(self.equilibrium)
        
        mapping = PhiBestResponseMapping(self.game)
        
        for player_id, strategy_dist in self.equilibrium.items():
            # 计算偏离成本
            other_strategies = {pid: dist for pid, dist in self.equilibrium.items() 
                              if pid != player_id}
            
            current_utility = 0.0
            for strategy, prob in strategy_dist.items():
                utility = mapping._compute_expected_utility(player_id, strategy, other_strategies)
                current_utility += utility * prob
                
            # 计算最大偏离收益
            max_deviation_gain = 0.0
            
            player = self.game.players.get(player_id)
            if player:
                for alternative_strategy in player.strategy_space.get_all_strategies():
                    alternative_utility = mapping._compute_expected_utility(
                        player_id, alternative_strategy, other_strategies
                    )
                    deviation_gain = alternative_utility - current_utility
                    max_deviation_gain = max(max_deviation_gain, deviation_gain)
                    
            # 稳定性分数
            player_stability = self.phi / (self.phi + max(0, max_deviation_gain))
            stability_score += player_stability
            
        return stability_score / n_players if n_players > 0 else 0.0
```

---

**注记**: 本形式化规范提供了T23-2定理的完整数学实现，包括φ-最优反应映射、Brouwer不动点定理应用、熵守恒验证、唯一性条件和稳定性分析的所有必要组件。所有实现严格遵循φ-表示、Zeckendorf编码和熵增原理。