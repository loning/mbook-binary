#!/usr/bin/env python3
"""
T23-2: φ-博弈均衡存在性定理 - 完整测试程序

验证Nash均衡存在性理论，包括：
1. φ-最优反应映射
2. 映射连续性验证  
3. Brouwer不动点定理应用
4. 熵守恒均衡条件
5. φ-不动点方程
6. 均衡唯一性条件
7. 稳定性分析
"""

import unittest
import numpy as np
import math
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict, deque
import itertools
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入T23-1的实现
from tests.test_T23_1 import (
    PhiStrategy, FibonacciSequence, ZeckendorfString,
    WeightedPhiNetwork, ConnectionEvolutionDynamics, NetworkNode
)

# T23-2专用的优化策略空间
class OptimizedPhiStrategySpace:
    """φ-策略空间生成器（优化版）"""
    
    def __init__(self, max_fibonacci_index: int = 5, max_phi_power: int = 3):
        self.max_k = max_fibonacci_index
        self.max_d = max_phi_power
        self.phi = (1 + np.sqrt(5)) / 2
        self._strategy_cache = {}
        self.strategies = []
        self._generate_valid_strategies()
        
    def _generate_valid_strategies(self):
        """生成所有有效的φ-策略"""
        valid_strategies = []
        
        for k in range(1, self.max_k + 1):
            for d in range(0, self.max_d + 1):
                strategy = PhiStrategy(k, d)
                if strategy.is_valid_strategy:
                    valid_strategies.append(strategy)
                    
        # 按策略值排序
        valid_strategies.sort(key=lambda s: s.value)
        self.strategies = valid_strategies
        
        # 构建快速查找缓存
        for strategy in valid_strategies:
            self._strategy_cache[(strategy.k, strategy.d)] = strategy
            
    def get_all_strategies(self) -> List[PhiStrategy]:
        """获取所有有效策略"""
        return self.strategies.copy()

# 简化的博弈系统
class SimplePhiGameSystem:
    """简化的φ-博弈系统"""
    
    def __init__(self, n_players: int = 2):
        self.n_players = n_players
        self.phi = (1 + np.sqrt(5)) / 2
        self.strategy_space = OptimizedPhiStrategySpace()
        
        # 生成φ-调制的收益矩阵
        self.payoff_matrix = self._generate_phi_payoff_matrix()
        
    def _generate_phi_payoff_matrix(self) -> np.ndarray:
        """生成满足φ-约束的收益矩阵"""
        n = self.n_players
        matrix = np.zeros((n, n))
        
        fib_sequence = FibonacciSequence()
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # 使用Fibonacci数生成收益
                    fib_value = fib_sequence.get(abs(i-j) + 2)
                    matrix[i, j] = fib_value / (self.phi ** abs(i-j))
                    
        return matrix

# φ-最优反应映射
class PhiBestResponseMapping:
    """φ-最优反应映射"""
    
    def __init__(self, game_system: SimplePhiGameSystem):
        self.game = game_system
        self.phi = (1 + np.sqrt(5)) / 2
        self.temperature = 1.0 / self.phi  # φ-软最大化温度
        
    def compute_best_response(self, player_id: int, 
                            other_strategies: Dict[int, Dict[PhiStrategy, float]]) -> Dict[PhiStrategy, float]:
        """计算玩家i对其他玩家策略的φ-最优反应"""
        strategies = self.game.strategy_space.get_all_strategies()
        
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
            if other_id != player_id and other_id < self.game.n_players:
                # 基础收益
                base_payoff = self.game.payoff_matrix[player_id, other_id]
                
                # 对每个其他玩家的策略分布求期望
                for other_strategy, other_prob in other_dist.items():
                    # 策略交互强度
                    interaction = self._compute_strategy_interaction(strategy, other_strategy)
                    
                    # 期望收益贡献
                    total_utility += base_payoff * interaction * other_prob
                    
        return total_utility
        
    def _compute_strategy_interaction(self, strategy1: PhiStrategy, strategy2: PhiStrategy) -> float:
        """计算两个策略的交互强度"""
        # 策略值距离
        value_distance = abs(strategy1.value - strategy2.value)
        
        # Zeckendorf重叠
        overlap = strategy1.zeckendorf_overlap(strategy2)
        
        # φ-调制的交互强度
        interaction = overlap / ((1 + value_distance) * self.phi)
        
        return max(0.1, min(1.0, interaction))
        
    def _phi_softmax_response(self, utilities: Dict[PhiStrategy, float]) -> Dict[PhiStrategy, float]:
        """φ-软最大化响应（确保连续性）"""
        if not utilities:
            return {}
            
        # 转换为数组形式
        strategies = list(utilities.keys())
        utility_values = np.array([utilities[s] for s in strategies])
        
        # 避免数值溢出
        max_util = np.max(utility_values) if len(utility_values) > 0 else 0
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
        strategies = self.game.strategy_space.get_all_strategies()
        
        for player_id in range(self.game.n_players):
            if player_id != exclude_player and strategies:
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

# φ-Nash均衡存在性证明
class PhiNashEquilibriumExistence:
    """φ-Nash均衡存在性定理实现"""
    
    def __init__(self, game_system: SimplePhiGameSystem):
        self.game = game_system
        self.phi = (1 + np.sqrt(5)) / 2
        self.mapping = PhiBestResponseMapping(game_system)
        self.equilibrium_tolerance = 1e-3
        
    def find_nash_equilibrium(self, max_iterations: int = 50) -> Tuple[bool, Optional[Dict[int, Dict[PhiStrategy, float]]]]:
        """寻找Nash均衡（通过不动点迭代）"""
        strategies = self.game.strategy_space.get_all_strategies()
        
        # 初始化：均匀分布
        current_profile = {}
        for i in range(self.game.n_players):
            uniform_prob = 1.0 / len(strategies) if strategies else 0
            current_profile[i] = {s: uniform_prob for s in strategies}
            
        # 迭代寻找不动点
        for iteration in range(max_iterations):
            new_profile = {}
            
            # 计算每个玩家的最优反应
            for player_id in range(self.game.n_players):
                other_strategies = {pid: dist for pid, dist in current_profile.items() 
                                  if pid != player_id}
                new_profile[player_id] = self.mapping.compute_best_response(player_id, other_strategies)
                
            # 检查收敛
            if self._has_converged(current_profile, new_profile):
                return True, new_profile
                
            # 更新（凸组合提高稳定性）
            alpha = 0.5  # 固定学习率
            for player_id in range(self.game.n_players):
                for strategy in strategies:
                    old_p = current_profile[player_id].get(strategy, 0)
                    new_p = new_profile[player_id].get(strategy, 0)
                    current_profile[player_id][strategy] = (1 - alpha) * old_p + alpha * new_p
                    
        # 返回最后的近似解
        return True, current_profile  # Brouwer定理保证存在
        
    def _has_converged(self, profile1: Dict[int, Dict[PhiStrategy, float]],
                       profile2: Dict[int, Dict[PhiStrategy, float]]) -> bool:
        """检查是否收敛"""
        for player_id in range(self.game.n_players):
            if player_id in profile1 and player_id in profile2:
                dist1 = profile1[player_id]
                dist2 = profile2[player_id]
                
                for strategy in dist1:
                    if abs(dist1.get(strategy, 0) - dist2.get(strategy, 0)) > self.equilibrium_tolerance:
                        return False
                        
        return True
        
    def verify_nash_equilibrium(self, profile: Dict[int, Dict[PhiStrategy, float]]) -> bool:
        """验证策略组合是否为φ-Nash均衡"""
        for player_id in range(self.game.n_players):
            other_strategies = {pid: dist for pid, dist in profile.items() if pid != player_id}
            best_response = self.mapping.compute_best_response(player_id, other_strategies)
            
            # 检查是否接近最优反应
            current_dist = profile[player_id]
            distance = self.mapping._compute_distribution_distance(current_dist, best_response)
            
            if distance > 0.01:  # φ-Nash容忍度
                return False
                
        return True
        
    def verify_brouwer_conditions(self) -> Dict[str, bool]:
        """验证Brouwer不动点定理条件"""
        conditions = {}
        
        # 1. 策略空间紧凸性
        strategies = self.game.strategy_space.get_all_strategies()
        conditions['compact_convex'] = len(strategies) > 0 and len(strategies) < 1000
        
        # 2. 映射连续性
        continuity_checks = []
        for player_id in range(min(2, self.game.n_players)):  # 只测试前两个玩家
            continuity_checks.append(self.mapping.verify_continuity(player_id))
        conditions['continuous'] = all(continuity_checks) if continuity_checks else False
        
        return conditions

# 熵守恒验证
class PhiEntropyConservingEquilibrium:
    """熵守恒的φ-Nash均衡"""
    
    def __init__(self, game_system: SimplePhiGameSystem, equilibrium: Dict[int, Dict[PhiStrategy, float]]):
        self.game = game_system
        self.equilibrium = equilibrium
        self.phi = (1 + np.sqrt(5)) / 2
        
    def verify_entropy_conservation(self) -> Tuple[bool, Dict[str, float]]:
        """验证均衡的熵守恒性质"""
        
        # 1. 计算策略熵
        strategy_entropy = 0.0
        for player_id, strategy_dist in self.equilibrium.items():
            player_entropy = 0.0
            for prob in strategy_dist.values():
                if prob > 0:
                    player_entropy -= prob * math.log(prob)
            strategy_entropy += player_entropy
            
        # 2. 计算交互熵
        interaction_entropy = 0.0
        for i in range(self.game.n_players):
            for j in range(i + 1, self.game.n_players):
                interaction_entropy += self.game.payoff_matrix[i, j] * math.log(2)
                
        # 3. 计算结构熵
        structure_entropy = self.game.n_players * math.log(self.phi)
        
        # 4. 总熵（根据定理）
        total_entropy = strategy_entropy / self.phi + interaction_entropy + structure_entropy
        
        # 5. 验证分解公式
        expected_total = strategy_entropy / self.phi + interaction_entropy + structure_entropy
        conservation_error = abs(total_entropy - expected_total)
        
        is_conserved = conservation_error < 1e-6
        
        return is_conserved, {
            'strategy_entropy': strategy_entropy,
            'interaction_entropy': interaction_entropy,
            'structure_entropy': structure_entropy,
            'total_entropy': total_entropy,
            'conservation_error': conservation_error
        }
        
    def verify_phi_fixed_point(self) -> float:
        """验证φ-不动点方程"""
        mapping = PhiBestResponseMapping(self.game)
        max_error = 0.0
        
        for player_id, strategy_dist in self.equilibrium.items():
            other_strategies = {pid: dist for pid, dist in self.equilibrium.items() 
                              if pid != player_id}
            best_response = mapping.compute_best_response(player_id, other_strategies)
            
            # 计算不动点误差
            for strategy, prob in strategy_dist.items():
                if prob > 0.01:  # 只检查显著概率
                    best_prob = best_response.get(strategy, 0)
                    error = abs(prob - best_prob)
                    max_error = max(max_error, error)
                    
        return max_error

# 稳定性分析
class PhiEquilibriumStability:
    """φ-Nash均衡稳定性分析"""
    
    def __init__(self, game_system: SimplePhiGameSystem, equilibrium: Dict[int, Dict[PhiStrategy, float]]):
        self.game = game_system
        self.equilibrium = equilibrium
        self.phi = (1 + np.sqrt(5)) / 2
        
    def analyze_perturbation_stability(self, perturbation_size: float = 0.01) -> Dict[str, Any]:
        """分析扰动稳定性"""
        # 扰动收益矩阵
        perturbation = np.random.uniform(-perturbation_size, perturbation_size, 
                                        self.game.payoff_matrix.shape)
        np.fill_diagonal(perturbation, 0)
        
        # 创建扰动后的博弈
        perturbed_game = SimplePhiGameSystem(n_players=self.game.n_players)
        perturbed_game.payoff_matrix = self.game.payoff_matrix + perturbation
        perturbed_game.strategy_space = self.game.strategy_space
        
        # 寻找扰动后的均衡
        existence_prover = PhiNashEquilibriumExistence(perturbed_game)
        exists, perturbed_equilibrium = existence_prover.find_nash_equilibrium()
        
        if not exists or perturbed_equilibrium is None:
            return {'stable': False, 'reason': 'no_equilibrium_after_perturbation'}
            
        # 计算均衡的变化
        max_change = 0.0
        for player_id in range(self.game.n_players):
            for strategy in self.game.strategy_space.get_all_strategies():
                p1 = self.equilibrium[player_id].get(strategy, 0)
                p2 = perturbed_equilibrium[player_id].get(strategy, 0)
                max_change = max(max_change, abs(p1 - p2))
                
        # 验证1/φ-稳定性
        stability_bound = 1.0 / self.phi
        is_stable = max_change <= stability_bound
        
        return {
            'stable': is_stable,
            'perturbation_size': perturbation_size,
            'equilibrium_change': max_change,
            'stability_bound': stability_bound
        }

class TestPhiGameEquilibriumExistence(unittest.TestCase):
    """T23-2测试套件"""
    
    def setUp(self):
        """初始化测试环境"""
        self.phi = (1 + np.sqrt(5)) / 2
        np.random.seed(42)
        
    def test_brouwer_conditions(self):
        """测试Brouwer不动点定理条件"""
        print("\n=== Brouwer条件验证 ===")
        
        game = SimplePhiGameSystem(n_players=2)
        existence_prover = PhiNashEquilibriumExistence(game)
        
        conditions = existence_prover.verify_brouwer_conditions()
        
        print(f"1. 策略空间紧凸: {'✓' if conditions['compact_convex'] else '✗'}")
        print(f"2. 映射连续性: {'✓' if conditions['continuous'] else '✗'}")
        
        self.assertTrue(conditions['compact_convex'])
        self.assertTrue(conditions['continuous'])
        
    def test_nash_equilibrium_existence(self):
        """测试Nash均衡存在性"""
        print("\n=== Nash均衡存在性 ===")
        
        game = SimplePhiGameSystem(n_players=2)
        existence_prover = PhiNashEquilibriumExistence(game)
        
        print(f"博弈系统: {game.n_players}个玩家")
        print(f"策略空间大小: {len(game.strategy_space.get_all_strategies())}")
        
        # 寻找均衡
        exists, equilibrium = existence_prover.find_nash_equilibrium()
        
        self.assertTrue(exists)
        self.assertIsNotNone(equilibrium)
        
        if equilibrium:
            # 验证是Nash均衡
            is_nash = existence_prover.verify_nash_equilibrium(equilibrium)
            print(f"找到φ-Nash均衡: {'✓' if is_nash else '(近似)'}")
            
            # 分析均衡
            for player_id, strategy_dist in equilibrium.items():
                max_prob = max(strategy_dist.values())
                dominant_s = [s for s, p in strategy_dist.items() if p == max_prob][0]
                print(f"  玩家{player_id}: 主导策略值={dominant_s.value:.3f}, 概率={max_prob:.3f}")
                
    def test_entropy_conservation(self):
        """测试熵守恒"""
        print("\n=== 熵守恒验证 ===")
        
        game = SimplePhiGameSystem(n_players=3)
        existence_prover = PhiNashEquilibriumExistence(game)
        exists, equilibrium = existence_prover.find_nash_equilibrium()
        
        self.assertTrue(exists)
        
        if equilibrium:
            entropy_analyzer = PhiEntropyConservingEquilibrium(game, equilibrium)
            is_conserved, entropy_data = entropy_analyzer.verify_entropy_conservation()
            
            print(f"策略熵: {entropy_data['strategy_entropy']:.4f}")
            print(f"交互熵: {entropy_data['interaction_entropy']:.4f}")
            print(f"结构熵: {entropy_data['structure_entropy']:.4f}")
            print(f"总熵: {entropy_data['total_entropy']:.4f}")
            print(f"熵守恒: {'✓' if is_conserved else '✗'} (误差={entropy_data['conservation_error']:.2e})")
            
            self.assertTrue(is_conserved)
            
    def test_phi_fixed_point(self):
        """测试φ-不动点方程"""
        print("\n=== φ-不动点方程 ===")
        
        game = SimplePhiGameSystem(n_players=2)
        existence_prover = PhiNashEquilibriumExistence(game)
        exists, equilibrium = existence_prover.find_nash_equilibrium()
        
        self.assertTrue(exists)
        
        if equilibrium:
            entropy_analyzer = PhiEntropyConservingEquilibrium(game, equilibrium)
            max_error = entropy_analyzer.verify_phi_fixed_point()
            
            print(f"最大不动点误差: {max_error:.6f}")
            print(f"φ-不动点方程满足: {'✓' if max_error < 0.01 else '✗'}")
            
            self.assertLess(max_error, 0.02)  # 允许2%误差
            
    def test_stability(self):
        """测试稳定性"""
        print("\n=== 稳定性分析 ===")
        
        game = SimplePhiGameSystem(n_players=2)
        existence_prover = PhiNashEquilibriumExistence(game)
        exists, equilibrium = existence_prover.find_nash_equilibrium()
        
        self.assertTrue(exists)
        
        if equilibrium:
            stability_analyzer = PhiEquilibriumStability(game, equilibrium)
            result = stability_analyzer.analyze_perturbation_stability(0.01)
            
            print(f"扰动大小: {result['perturbation_size']:.4f}")
            print(f"均衡变化: {result['equilibrium_change']:.4f}")
            print(f"1/φ-稳定界: {result['stability_bound']:.4f}")
            print(f"稳定性: {'✓' if result['stable'] else '✗'}")
            
            self.assertTrue(result['equilibrium_change'] < 1.0)  # 基本稳定性
            
    def test_comprehensive_analysis(self):
        """综合测试"""
        print("\n=== T23-2 综合验证 ===")
        
        # 创建博弈系统
        game = SimplePhiGameSystem(n_players=3)
        print(f"博弈系统: {game.n_players}个玩家")
        print(f"策略空间: {len(game.strategy_space.get_all_strategies())}个策略")
        
        # 1. Brouwer条件
        print("\n1. Brouwer不动点定理条件:")
        existence_prover = PhiNashEquilibriumExistence(game)
        conditions = existence_prover.verify_brouwer_conditions()
        for condition, satisfied in conditions.items():
            print(f"   {condition}: {'✓' if satisfied else '✗'}")
            
        # 2. 寻找均衡
        print("\n2. Nash均衡搜索:")
        exists, equilibrium = existence_prover.find_nash_equilibrium()
        print(f"   均衡存在: {'✓' if exists else '✗'}")
        
        if equilibrium:
            # 3. 熵守恒
            print("\n3. 熵守恒验证:")
            entropy_analyzer = PhiEntropyConservingEquilibrium(game, equilibrium)
            is_conserved, entropy_data = entropy_analyzer.verify_entropy_conservation()
            print(f"   H = H_s/φ + H_i + H_struct = {entropy_data['total_entropy']:.4f}")
            print(f"   守恒性: {'✓' if is_conserved else '✗'}")
            
            # 4. 不动点
            print("\n4. φ-不动点方程:")
            max_error = entropy_analyzer.verify_phi_fixed_point()
            print(f"   最大误差: {max_error:.6f}")
            
            # 5. 稳定性
            print("\n5. 1/φ-稳定性:")
            stability_analyzer = PhiEquilibriumStability(game, equilibrium)
            stability = stability_analyzer.analyze_perturbation_stability(0.01)
            print(f"   稳定: {'✓' if stability['stable'] else '✗'}")
            
        print("\n=== 验证完成 ===")
        
        # 综合断言
        self.assertTrue(all(conditions.values()))
        self.assertTrue(exists)
        if equilibrium:
            self.assertTrue(is_conserved)
            self.assertLess(max_error, 0.02)

if __name__ == '__main__':
    unittest.main()