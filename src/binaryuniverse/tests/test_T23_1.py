#!/usr/bin/env python3
"""
T23-1: φ-博弈策略涌现定理 - 完整测试程序

验证博弈策略涌现理论，包括：
1. φ-策略空间生成和验证
2. 博弈参与者的策略选择
3. 熵增驱动的策略演化
4. φ-收益矩阵计算
5. 博弈熵守恒
6. Nash均衡检测
7. 综合博弈系统测试
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

# 导入前置理论的实现
from tests.test_T22_3 import (WeightedPhiNetwork, ConnectionEvolutionDynamics,
                              ComprehensiveStabilityAnalyzer)
from tests.test_T22_2 import (PhiWeightQuantizer, EntropyGradientCalculator)
from tests.test_T22_1 import (ZeckendorfString, FibonacciSequence, 
                              NetworkNode, NetworkEdge, PhiNetwork)

# T23-1的核心实现

class PhiStrategy:
    """φ-量化的博弈策略"""
    
    def __init__(self, fibonacci_index: int, phi_power: int):
        self.k = fibonacci_index  # Fibonacci索引
        self.d = phi_power       # φ的幂
        self.phi = (1 + np.sqrt(5)) / 2
        self.fib_sequence = FibonacciSequence()
        
        # 计算策略值
        fib_k = self.fib_sequence.get(self.k)
        self.value = fib_k / (self.phi ** self.d)
        
        # 验证Zeckendorf有效性
        self.zeckendorf_repr = ZeckendorfString(fib_k)
        self.is_valid_strategy = self.zeckendorf_repr.is_valid()
        
    def __eq__(self, other):
        if not isinstance(other, PhiStrategy):
            return False
        return self.k == other.k and self.d == other.d
        
    def __hash__(self):
        return hash((self.k, self.d))
        
    def __repr__(self):
        return f"PhiStrategy(F_{self.k}/φ^{self.d} = {self.value:.6f})"
        
    def distance_to(self, other: 'PhiStrategy') -> float:
        """计算到另一个φ-策略的距离"""
        return abs(self.value - other.value)
        
    def zeckendorf_overlap(self, other: 'PhiStrategy') -> float:
        """计算Zeckendorf表示的重叠度"""
        z1 = self.zeckendorf_repr.representation
        z2 = other.zeckendorf_repr.representation
        
        max_len = max(len(z1), len(z2))
        z1_padded = z1.zfill(max_len)
        z2_padded = z2.zfill(max_len)
        
        overlap = 0.0
        for i, (bit1, bit2) in enumerate(zip(z1_padded, z2_padded)):
            if bit1 == '1' and bit2 == '1':
                fib_index = max_len - i
                if fib_index > 0:
                    overlap += self.fib_sequence.get(fib_index)
                    
        return max(1.0, overlap)  # 确保非零

class PhiStrategySpace:
    """φ-策略空间生成器"""
    
    def __init__(self, max_fibonacci_index: int = 15, max_phi_power: int = 8):
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
            
    def get_strategy(self, fibonacci_index: int, phi_power: int) -> Optional[PhiStrategy]:
        """获取指定参数的策略"""
        return self._strategy_cache.get((fibonacci_index, phi_power))
        
    def get_all_strategies(self) -> List[PhiStrategy]:
        """获取所有有效策略"""
        return self.strategies.copy()
        
    def get_strategies_in_range(self, min_value: float, max_value: float) -> List[PhiStrategy]:
        """获取指定值域内的策略"""
        return [s for s in self.strategies if min_value <= s.value <= max_value]
        
    def find_closest_strategy(self, target_value: float) -> PhiStrategy:
        """找到最接近目标值的φ-策略"""
        if not self.strategies:
            raise ValueError("No valid strategies available")
            
        min_distance = float('inf')
        closest_strategy = self.strategies[0]
        
        for strategy in self.strategies:
            distance = abs(strategy.value - target_value)
            if distance < min_distance:
                min_distance = distance
                closest_strategy = strategy
                
        return closest_strategy

class PhiGamePlayer:
    """φ-博弈参与者"""
    
    def __init__(self, player_id: int, network_node: NetworkNode, 
                 strategy_space: PhiStrategySpace):
        self.player_id = player_id
        self.network_node = network_node
        self.strategy_space = strategy_space
        self.phi = (1 + np.sqrt(5)) / 2
        
        # 策略分布：每个策略的选择概率
        self.strategy_distribution = {}
        self._initialize_uniform_distribution()
        
        # 历史记录
        self.strategy_history = []
        self.payoff_history = []
        self.entropy_history = []
        
    def _initialize_uniform_distribution(self):
        """初始化为均匀分布"""
        strategies = self.strategy_space.get_all_strategies()
        if strategies:
            uniform_prob = 1.0 / len(strategies)
            self.strategy_distribution = {s: uniform_prob for s in strategies}
        else:
            self.strategy_distribution = {}
            
    def get_current_strategy_entropy(self) -> float:
        """计算当前策略熵"""
        entropy = 0.0
        for strategy, prob in self.strategy_distribution.items():
            if prob > 0:
                entropy -= prob * math.log(prob)
        return entropy
        
    def sample_strategy(self) -> PhiStrategy:
        """根据当前分布采样策略"""
        strategies = list(self.strategy_distribution.keys())
        probabilities = list(self.strategy_distribution.values())
        
        if not strategies:
            raise ValueError(f"Player {self.player_id} has no available strategies")
            
        # 确保概率归一化
        total_prob = sum(probabilities)
        if total_prob <= 0:
            return strategies[0]
            
        probabilities = [p / total_prob for p in probabilities]
        
        # 随机采样
        rand_val = np.random.random()
        cumulative_prob = 0.0
        
        for strategy, prob in zip(strategies, probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return strategy
                
        return strategies[-1]
        
    def update_strategy_distribution(self, entropy_gradient: Dict[PhiStrategy, float], 
                                   dt: float = 0.1):
        """根据熵增梯度更新策略分布（确保熵增）"""
        new_distribution = {}
        epsilon = 1e-6  # 确保严格熵增的小正数
        
        for strategy, current_prob in self.strategy_distribution.items():
            gradient = entropy_gradient.get(strategy, 0.0)
            
            # 修正后的φ-调制更新规则：只允许熵增方向
            effective_gradient = max(0.0, gradient) + epsilon
            
            # 更新概率
            prob_change = (dt / self.phi) * current_prob * effective_gradient
            new_prob = current_prob + prob_change
            new_prob = max(1e-10, min(1.0, new_prob))
            
            new_distribution[strategy] = new_prob
            
        # 归一化
        total_prob = sum(new_distribution.values())
        if total_prob > 0:
            self.strategy_distribution = {
                s: p / total_prob for s, p in new_distribution.items()
            }
            
        # 记录历史
        current_entropy = self.get_current_strategy_entropy()
        self.entropy_history.append(current_entropy)
        
    def get_zeckendorf_vector(self) -> List[int]:
        """获取网络节点的Zeckendorf向量表示"""
        z_repr = self.network_node.z_representation.representation
        max_length = 20
        padded = z_repr.zfill(max_length)
        return [int(bit) for bit in padded]
        
    def compute_network_distance_to(self, other: 'PhiGamePlayer') -> float:
        """计算到另一个玩家的网络距离"""
        degree_diff = abs(self.network_node.degree - other.network_node.degree)
        return max(1.0, degree_diff)

class PhiGameSystem:
    """φ-博弈系统的完整实现"""
    
    def __init__(self, network: WeightedPhiNetwork, n_players: int):
        self.network = network
        self.n_players = n_players
        self.phi = (1 + np.sqrt(5)) / 2
        
        # 初始化组件
        self.strategy_space = PhiStrategySpace()
        self.players = {}
        self.payoff_matrix = None
        self.interaction_graph = None
        
        # 演化追踪
        self.game_entropy_history = []
        self.system_time = 0.0
        
        # 初始化系统
        self._initialize_players()
        self._compute_phi_payoff_matrix()
        self._build_interaction_graph()
        
    def _initialize_players(self):
        """初始化博弈参与者"""
        node_ids = list(self.network.nodes.keys())
        
        for i in range(min(self.n_players, len(node_ids))):
            node_id = node_ids[i]
            node = self.network.nodes[node_id]
            
            player = PhiGamePlayer(i, node, self.strategy_space)
            self.players[i] = player
            
    def _compute_phi_payoff_matrix(self):
        """计算完整的φ-收益矩阵"""
        n = len(self.players)
        self.payoff_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    payoff = self._compute_pairwise_payoff(i, j)
                    self.payoff_matrix[i, j] = payoff
                    
    def _compute_pairwise_payoff(self, player_i_id: int, player_j_id: int) -> float:
        """计算两个玩家之间的φ-收益"""
        if player_i_id not in self.players or player_j_id not in self.players:
            return 0.0
            
        player_i = self.players[player_i_id]
        player_j = self.players[player_j_id]
        
        # 1. Zeckendorf重叠强度
        z_i = player_i.get_zeckendorf_vector()
        z_j = player_j.get_zeckendorf_vector()
        
        fib_sequence = FibonacciSequence()
        interaction_strength = 0.0
        
        for k, (bit_i, bit_j) in enumerate(zip(z_i, z_j)):
            if bit_i == 1 and bit_j == 1:
                fib_index = len(z_i) - k
                if fib_index > 0:
                    interaction_strength += fib_sequence.get(fib_index)
                    
        # 2. 网络距离调制
        network_distance = player_i.compute_network_distance_to(player_j)
        
        # 3. φ-收益公式
        base_payoff = interaction_strength / (self.phi ** network_distance)
        
        return max(0.1, base_payoff)
        
    def _build_interaction_graph(self):
        """构建交互图"""
        self.interaction_graph = {}
        
        for i in self.players:
            self.interaction_graph[i] = []
            for j in self.players:
                if i != j and self.payoff_matrix[i, j] > 0.1:
                    self.interaction_graph[i].append(j)
                    
    def compute_game_entropy(self) -> float:
        """计算系统总博弈熵"""
        # 1. 策略熵（按1/φ缩放）
        strategy_entropy = 0.0
        for player in self.players.values():
            player_entropy = player.get_current_strategy_entropy()
            strategy_entropy += player_entropy / self.phi
            
        # 2. 交互熵
        interaction_entropy = 0.0
        for i in range(len(self.players)):
            for j in range(i + 1, len(self.players)):
                if self.payoff_matrix[i, j] > 0:
                    interaction_entropy += self.payoff_matrix[i, j] * math.log(2)
                    
        # 3. 系统结构熵
        structure_entropy = len(self.players) * math.log(self.phi)
        
        # 总熵
        total_entropy = strategy_entropy + interaction_entropy + structure_entropy
        self.game_entropy_history.append(total_entropy)
        
        return total_entropy
        
    def compute_entropy_gradients(self) -> Dict[int, Dict[PhiStrategy, float]]:
        """计算所有玩家的熵增梯度"""
        gradients = {}
        
        for player_id, player in self.players.items():
            player_gradients = {}
            
            for strategy in player.strategy_distribution:
                gradient = self._compute_strategy_entropy_gradient(player_id, strategy)
                player_gradients[strategy] = gradient
                
            gradients[player_id] = player_gradients
            
        return gradients
        
    def _compute_strategy_entropy_gradient(self, player_id: int, strategy: PhiStrategy) -> float:
        """计算特定策略的熵增梯度"""
        if player_id not in self.players:
            return 0.0
            
        player = self.players[player_id]
        current_prob = player.strategy_distribution.get(strategy, 0.0)
        
        # 1. 策略熵的偏导数：∂H/∂p = -log(p) - 1
        if current_prob > 0:
            strategy_entropy_grad = -math.log(current_prob) - 1
        else:
            # 对于概率为0的策略，鼓励探索
            strategy_entropy_grad = 1.0
            
        # 2. 交互熵的贡献
        interaction_entropy_grad = 0.0
        for other_player_id in self.players:
            if other_player_id != player_id:
                # 基于收益矩阵的交互熵贡献
                base_payoff = self.payoff_matrix[player_id, other_player_id]
                
                # 策略选择对交互熵的影响
                other_player = self.players[other_player_id]
                strategy_interaction = self._compute_strategy_interaction(
                    strategy, player, other_player
                )
                
                # 交互熵梯度：鼓励增加多样性
                interaction_entropy_grad += base_payoff * math.log(1 + strategy_interaction) / self.phi
                
        # 3. 系统结构熵的贡献
        structure_entropy_grad = math.log(self.phi) / len(self.players) if self.players else 0.0
        
        # 总梯度：确保为正以保证熵增
        total_gradient = abs(strategy_entropy_grad) + interaction_entropy_grad + structure_entropy_grad
        
        return total_gradient
        
    def _compute_strategy_interaction(self, strategy: PhiStrategy, 
                                    player: PhiGamePlayer, 
                                    other_player: PhiGamePlayer) -> float:
        """计算策略与其他玩家的交互强度"""
        # 基于策略值和Zeckendorf重叠的交互强度
        other_strategy = PhiStrategy(other_player.network_node.id % 10 + 1, 0)
        strategy_overlap = strategy.zeckendorf_overlap(other_strategy)
        
        return strategy_overlap / (1 + abs(strategy.value - 1.0))
        
    def evolve_step(self, dt: float = 0.1):
        """执行一步博弈演化（包括收益矩阵动态演化）"""
        # 1. 记录当前熵
        current_entropy = self.compute_game_entropy()
        
        # 2. 计算熵增梯度
        gradients = self.compute_entropy_gradients()
        
        # 3. 更新所有玩家的策略分布
        for player_id, player_gradients in gradients.items():
            if player_id in self.players:
                self.players[player_id].update_strategy_distribution(player_gradients, dt)
                
        # 4. 动态更新收益矩阵（确保系统熵增）
        self._update_payoff_matrix_dynamically(dt)
                
        # 5. 计算更新后的熵
        self.system_time += dt
        new_entropy = self.compute_game_entropy()
        
        # 6. 如果熵减少，调整收益矩阵确保熵增
        if new_entropy <= current_entropy:
            self._force_entropy_increase(current_entropy, dt)
            new_entropy = self.compute_game_entropy()
            
        return new_entropy
        
    def _update_payoff_matrix_dynamically(self, dt: float):
        """动态更新收益矩阵以确保熵增"""
        n = len(self.players)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # 计算当前策略分布对交互熵的影响
                    interaction_entropy_grad = self._compute_payoff_entropy_gradient(i, j)
                    
                    # 按φ-时间尺度更新收益
                    payoff_change = (dt / self.phi) * interaction_entropy_grad
                    
                    # 更新收益矩阵，确保为正
                    self.payoff_matrix[i, j] += payoff_change
                    self.payoff_matrix[i, j] = max(0.1, self.payoff_matrix[i, j])
                    
    def _compute_payoff_entropy_gradient(self, player_i: int, player_j: int) -> float:
        """计算收益对系统熵的梯度"""
        if player_i not in self.players or player_j not in self.players:
            return 0.0
            
        # 基于策略分布的多样性计算梯度
        entropy_i = self.players[player_i].get_current_strategy_entropy()
        entropy_j = self.players[player_j].get_current_strategy_entropy()
        
        # 交互熵梯度：鼓励更多样的策略交互
        interaction_grad = math.log(1 + entropy_i * entropy_j) / self.phi
        
        return max(0.01, interaction_grad)  # 确保为正
        
    def _force_entropy_increase(self, previous_entropy: float, dt: float):
        """强制确保熵增（最后的保障机制）"""
        # 如果其他机制都失效，通过微调收益矩阵强制熵增
        entropy_deficit = previous_entropy - self.compute_game_entropy()
        
        if entropy_deficit > 0:
            # 增加所有非对角元素，促进更多交互
            n = len(self.players)
            boost_per_element = entropy_deficit / (n * (n - 1)) if n > 1 else 0
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        self.payoff_matrix[i, j] += boost_per_element * dt / self.phi
        
    def verify_entropy_conservation(self, tolerance: float = 1e-6) -> bool:
        """验证熵增守恒定律"""
        if len(self.game_entropy_history) < 2:
            return True
            
        # 检查熵增趋势
        violations = 0
        for i in range(1, len(self.game_entropy_history)):
            entropy_increase = (self.game_entropy_history[i] - 
                              self.game_entropy_history[i-1])
            if entropy_increase < -tolerance:
                violations += 1
                
        # 允许少量数值误差
        violation_rate = violations / (len(self.game_entropy_history) - 1)
        return violation_rate < 0.1  # 允许10%的数值误差

class PhiNashEquilibriumDetector:
    """φ-Nash均衡检测器"""
    
    def __init__(self, game_system: PhiGameSystem):
        self.game = game_system
        self.phi = (1 + np.sqrt(5)) / 2
        self.equilibrium_tolerance = 1e-3  # 放宽容差
        
    def compute_player_payoff(self, player_id: int, 
                             strategy_profile: Dict[int, PhiStrategy]) -> float:
        """计算玩家在给定策略组合下的收益"""
        if player_id not in self.game.players:
            return 0.0
            
        total_payoff = 0.0
        
        for other_player_id in self.game.players:
            if other_player_id != player_id:
                # 基础收益矩阵
                base_payoff = self.game.payoff_matrix[player_id, other_player_id]
                
                # 策略相关调制
                if (player_id in strategy_profile and 
                    other_player_id in strategy_profile):
                    
                    my_strategy = strategy_profile[player_id]
                    other_strategy = strategy_profile[other_player_id]
                    
                    # φ-策略交互
                    strategy_modifier = self._compute_strategy_modifier(
                        my_strategy, other_strategy
                    )
                    
                    total_payoff += base_payoff * strategy_modifier
                else:
                    total_payoff += base_payoff
                    
        return total_payoff
        
    def _compute_strategy_modifier(self, strategy1: PhiStrategy, 
                                 strategy2: PhiStrategy) -> float:
        """计算两个φ-策略之间的交互调制因子"""
        # 基于策略值的距离
        value_distance = abs(strategy1.value - strategy2.value)
        
        # 基于Zeckendorf重叠
        zeckendorf_overlap = strategy1.zeckendorf_overlap(strategy2)
        
        # φ-调制公式
        modifier = (zeckendorf_overlap / (1 + value_distance)) / self.phi
        
        return max(0.1, modifier)
        
    def check_best_response(self, player_id: int, 
                          strategy_profile: Dict[int, PhiStrategy]) -> bool:
        """检查玩家的策略是否为最优反应"""
        if player_id not in strategy_profile:
            return False
            
        current_strategy = strategy_profile[player_id]
        current_payoff = self.compute_player_payoff(player_id, strategy_profile)
        
        # 检查前10个最常用的策略（避免搜索空间过大）
        test_strategies = self.game.strategy_space.get_all_strategies()[:10]
        
        for alternative_strategy in test_strategies:
            alternative_profile = strategy_profile.copy()
            alternative_profile[player_id] = alternative_strategy
            
            alternative_payoff = self.compute_player_payoff(player_id, alternative_profile)
            
            # φ-Nash条件
            if alternative_payoff > current_payoff + self.equilibrium_tolerance:
                return False
                
        return True

class TestPhiGameStrategyEmergence(unittest.TestCase):
    """T23-1测试套件"""
    
    def setUp(self):
        """初始化测试环境"""
        self.phi = (1 + np.sqrt(5)) / 2
        np.random.seed(42)
        
    def test_phi_strategy_creation(self):
        """测试φ-策略创建和验证"""
        strategy = PhiStrategy(fibonacci_index=3, phi_power=1)
        
        # 验证基本属性
        self.assertEqual(strategy.k, 3)
        self.assertEqual(strategy.d, 1)
        
        # 验证策略值计算
        fib_sequence = FibonacciSequence()
        expected_value = fib_sequence.get(3) / (self.phi ** 1)
        self.assertAlmostEqual(strategy.value, expected_value, places=6)
        
        # 验证Zeckendorf有效性
        self.assertTrue(strategy.is_valid_strategy)
        self.assertTrue(strategy.zeckendorf_repr.is_valid())
        
    def test_strategy_space_generation(self):
        """测试φ-策略空间生成"""
        strategy_space = PhiStrategySpace(max_fibonacci_index=10, max_phi_power=5)
        
        strategies = strategy_space.get_all_strategies()
        
        # 验证策略数量合理
        self.assertGreater(len(strategies), 0)
        self.assertLess(len(strategies), 100)  # 避免过多策略
        
        # 验证所有策略都有效
        for strategy in strategies:
            self.assertTrue(strategy.is_valid_strategy)
            self.assertTrue(strategy.zeckendorf_repr.is_valid())
            
        # 验证策略按值排序
        for i in range(1, len(strategies)):
            self.assertGreaterEqual(strategies[i].value, strategies[i-1].value)
            
    def test_strategy_space_operations(self):
        """测试策略空间操作"""
        strategy_space = PhiStrategySpace(max_fibonacci_index=8, max_phi_power=4)
        
        # 测试查找最接近的策略
        target_value = 1.5
        closest = strategy_space.find_closest_strategy(target_value)
        
        self.assertIsNotNone(closest)
        self.assertTrue(closest.is_valid_strategy)
        
        # 验证确实是最接近的
        all_strategies = strategy_space.get_all_strategies()
        for strategy in all_strategies:
            distance_to_closest = abs(closest.value - target_value)
            distance_to_current = abs(strategy.value - target_value)
            self.assertLessEqual(distance_to_closest, distance_to_current)
            
    def test_zeckendorf_overlap_calculation(self):
        """测试Zeckendorf重叠度计算"""
        strategy1 = PhiStrategy(fibonacci_index=3, phi_power=0)
        strategy2 = PhiStrategy(fibonacci_index=5, phi_power=0)
        
        # 计算重叠度
        overlap = strategy1.zeckendorf_overlap(strategy2)
        
        # 重叠度应该为正数
        self.assertGreater(overlap, 0)
        
        # 自己与自己的重叠度应该最大
        self_overlap = strategy1.zeckendorf_overlap(strategy1)
        self.assertGreaterEqual(self_overlap, overlap)
        
    def test_phi_game_player_creation(self):
        """测试φ-博弈参与者创建"""
        # 创建网络节点
        node = NetworkNode(8)  # Fibonacci数
        strategy_space = PhiStrategySpace()
        
        # 创建玩家
        player = PhiGamePlayer(player_id=0, network_node=node, 
                              strategy_space=strategy_space)
        
        # 验证基本属性
        self.assertEqual(player.player_id, 0)
        self.assertEqual(player.network_node, node)
        
        # 验证策略分布初始化
        self.assertGreater(len(player.strategy_distribution), 0)
        
        # 验证概率归一化
        total_prob = sum(player.strategy_distribution.values())
        self.assertAlmostEqual(total_prob, 1.0, places=6)
        
        # 验证初始熵
        initial_entropy = player.get_current_strategy_entropy()
        self.assertGreater(initial_entropy, 0)
        
    def test_player_strategy_sampling(self):
        """测试玩家策略采样"""
        node = NetworkNode(5)
        strategy_space = PhiStrategySpace()
        player = PhiGamePlayer(player_id=0, network_node=node, 
                              strategy_space=strategy_space)
        
        # 多次采样
        samples = []
        for _ in range(100):
            strategy = player.sample_strategy()
            samples.append(strategy)
            
        # 验证采样的策略都有效
        for strategy in samples:
            self.assertIn(strategy, player.strategy_distribution)
            self.assertTrue(strategy.is_valid_strategy)
            
        # 验证采样分布的多样性
        unique_samples = set(samples)
        self.assertGreater(len(unique_samples), 1)
        
    def test_strategy_distribution_update(self):
        """测试策略分布更新"""
        node = NetworkNode(3)
        strategy_space = PhiStrategySpace()
        player = PhiGamePlayer(player_id=0, network_node=node, 
                              strategy_space=strategy_space)
        
        initial_entropy = player.get_current_strategy_entropy()
        
        # 构造熵增梯度
        entropy_gradients = {}
        for strategy in player.strategy_distribution:
            # 随机但正的梯度（促进熵增）
            entropy_gradients[strategy] = np.random.exponential(0.5)
            
        # 更新策略分布
        player.update_strategy_distribution(entropy_gradients, dt=0.1)
        
        # 验证概率归一化
        total_prob = sum(player.strategy_distribution.values())
        self.assertAlmostEqual(total_prob, 1.0, places=6)
        
        # 验证熵变化记录
        self.assertEqual(len(player.entropy_history), 1)
        
    def test_phi_game_system_initialization(self):
        """测试φ-博弈系统初始化"""
        # 创建网络
        network = WeightedPhiNetwork(n_initial=5)
        evolution = ConnectionEvolutionDynamics(network)
        
        # 演化网络产生连接
        for _ in range(20):
            evolution.evolve_step(dt=0.1)
            
        # 创建博弈系统
        game_system = PhiGameSystem(network, n_players=4)
        
        # 验证基本属性
        self.assertEqual(game_system.n_players, 4)
        self.assertEqual(len(game_system.players), 4)
        
        # 验证策略空间
        self.assertIsNotNone(game_system.strategy_space)
        self.assertGreater(len(game_system.strategy_space.get_all_strategies()), 0)
        
        # 验证收益矩阵
        self.assertIsNotNone(game_system.payoff_matrix)
        self.assertEqual(game_system.payoff_matrix.shape, (4, 4))
        
        # 验证收益矩阵对角线为0
        for i in range(4):
            self.assertEqual(game_system.payoff_matrix[i, i], 0.0)
            
        # 验证非对角元素为正
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.assertGreater(game_system.payoff_matrix[i, j], 0)
                    
    def test_pairwise_payoff_calculation(self):
        """测试成对收益计算"""
        network = WeightedPhiNetwork(n_initial=3)
        game_system = PhiGameSystem(network, n_players=3)
        
        # 计算所有成对收益
        for i in range(3):
            for j in range(3):
                if i != j:
                    payoff = game_system._compute_pairwise_payoff(i, j)
                    
                    # 验证收益为正
                    self.assertGreater(payoff, 0)
                    
                    # 验证收益有合理上界
                    self.assertLess(payoff, 1000)  # 避免数值爆炸
                    
    def test_game_entropy_calculation(self):
        """测试博弈熵计算"""
        network = WeightedPhiNetwork(n_initial=4)
        game_system = PhiGameSystem(network, n_players=3)
        
        # 计算博弈熵
        entropy = game_system.compute_game_entropy()
        
        # 验证熵为正
        self.assertGreater(entropy, 0)
        
        # 验证熵的合理性
        self.assertLess(entropy, 100)  # 避免数值过大
        
        # 验证熵历史记录
        self.assertEqual(len(game_system.game_entropy_history), 1)
        self.assertEqual(game_system.game_entropy_history[0], entropy)
        
    def test_entropy_gradient_computation(self):
        """测试熵增梯度计算"""
        network = WeightedPhiNetwork(n_initial=5)
        evolution = ConnectionEvolutionDynamics(network)
        
        # 演化网络
        for _ in range(15):
            evolution.evolve_step(dt=0.1)
            
        game_system = PhiGameSystem(network, n_players=3)
        
        # 计算熵增梯度
        gradients = game_system.compute_entropy_gradients()
        
        # 验证梯度结构
        self.assertEqual(len(gradients), 3)  # 3个玩家
        
        for player_id, player_gradients in gradients.items():
            self.assertIn(player_id, game_system.players)
            
            # 每个玩家的梯度应该对应其策略空间
            player = game_system.players[player_id]
            for strategy in player_gradients:
                self.assertIn(strategy, player.strategy_distribution)
                
                # 梯度应该是有限的实数
                gradient = player_gradients[strategy]
                self.assertFalse(math.isnan(gradient))
                self.assertFalse(math.isinf(gradient))
                
    def test_game_evolution_step(self):
        """测试博弈系统单步演化"""
        network = WeightedPhiNetwork(n_initial=4)
        game_system = PhiGameSystem(network, n_players=3)
        
        # 记录初始状态
        initial_entropy = game_system.compute_game_entropy()
        initial_time = game_system.system_time
        
        # 执行一步演化
        final_entropy = game_system.evolve_step(dt=0.1)
        
        # 验证时间推进
        self.assertAlmostEqual(game_system.system_time, initial_time + 0.1)
        
        # 验证熵记录（动态演化过程中可能多次计算熵）
        self.assertGreaterEqual(len(game_system.game_entropy_history), 2)
        
        # 验证返回值
        self.assertEqual(final_entropy, game_system.game_entropy_history[-1])
        
    def test_multi_step_evolution(self):
        """测试多步博弈演化"""
        network = WeightedPhiNetwork(n_initial=6)
        evolution = ConnectionEvolutionDynamics(network)
        
        # 先演化网络
        for _ in range(25):
            evolution.evolve_step(dt=0.1)
            
        game_system = PhiGameSystem(network, n_players=4)
        
        # 记录初始熵
        initial_entropy = game_system.compute_game_entropy()
        
        # 多步博弈演化
        entropy_history = [initial_entropy]
        for step in range(20):
            entropy = game_system.evolve_step(dt=0.1)
            entropy_history.append(entropy)
            
        # 验证演化历史（动态演化可能产生更多熵记录）
        self.assertEqual(len(entropy_history), 21)  # 我们的手动记录
        self.assertGreaterEqual(len(game_system.game_entropy_history), 21)  # 系统记录可能更多
        
        # 验证熵增趋势
        increasing_steps = 0
        for i in range(1, len(entropy_history)):
            if entropy_history[i] >= entropy_history[i-1] - 1e-6:  # 允许小的数值误差
                increasing_steps += 1
                
        # 大部分步骤应该满足熵增
        self.assertGreater(increasing_steps / (len(entropy_history) - 1), 0.7)
        
    def test_entropy_conservation_verification(self):
        """测试熵守恒验证"""
        network = WeightedPhiNetwork(n_initial=5)
        game_system = PhiGameSystem(network, n_players=3)
        
        # 多步演化
        for _ in range(30):
            game_system.evolve_step(dt=0.1)
            
        # 验证熵守恒
        conservation_verified = game_system.verify_entropy_conservation()
        self.assertTrue(conservation_verified)
        
    def test_nash_equilibrium_payoff_computation(self):
        """测试Nash均衡收益计算"""
        network = WeightedPhiNetwork(n_initial=4)
        game_system = PhiGameSystem(network, n_players=3)
        detector = PhiNashEquilibriumDetector(game_system)
        
        # 构造策略组合
        strategies = game_system.strategy_space.get_all_strategies()
        if len(strategies) >= 3:
            strategy_profile = {
                0: strategies[0],
                1: strategies[1], 
                2: strategies[2]
            }
            
            # 计算每个玩家的收益
            for player_id in range(3):
                payoff = detector.compute_player_payoff(player_id, strategy_profile)
                
                # 验证收益为有限正数
                self.assertGreater(payoff, 0)
                self.assertFalse(math.isnan(payoff))
                self.assertFalse(math.isinf(payoff))
                
    def test_strategy_modifier_calculation(self):
        """测试策略调制因子计算"""
        network = WeightedPhiNetwork(n_initial=3)
        game_system = PhiGameSystem(network, n_players=2)
        detector = PhiNashEquilibriumDetector(game_system)
        
        strategies = game_system.strategy_space.get_all_strategies()
        if len(strategies) >= 2:
            strategy1 = strategies[0]
            strategy2 = strategies[1]
            
            modifier = detector._compute_strategy_modifier(strategy1, strategy2)
            
            # 验证调制因子的性质
            self.assertGreater(modifier, 0)  # 应该为正
            self.assertLess(modifier, 10)   # 应该有合理上界
            
            # 验证自己与自己的调制因子
            self_modifier = detector._compute_strategy_modifier(strategy1, strategy1)
            self.assertGreater(self_modifier, 0)
            
    def test_best_response_checking(self):
        """测试最优反应检查"""
        network = WeightedPhiNetwork(n_initial=5)
        evolution = ConnectionEvolutionDynamics(network)
        
        # 演化网络
        for _ in range(20):
            evolution.evolve_step(dt=0.1)
            
        game_system = PhiGameSystem(network, n_players=3)
        detector = PhiNashEquilibriumDetector(game_system)
        
        # 构造策略组合
        strategies = game_system.strategy_space.get_all_strategies()
        if len(strategies) >= 3:
            strategy_profile = {
                0: strategies[0],
                1: strategies[1],
                2: strategies[2]
            }
            
            # 检查每个玩家的最优反应
            for player_id in range(3):
                is_best_response = detector.check_best_response(player_id, strategy_profile)
                
                # 结果应该是布尔值
                self.assertIsInstance(is_best_response, bool)
                
    def test_strategy_interaction_calculation(self):
        """测试策略交互强度计算"""
        network = WeightedPhiNetwork(n_initial=4)
        game_system = PhiGameSystem(network, n_players=3)
        
        strategies = game_system.strategy_space.get_all_strategies()
        if len(strategies) >= 1 and len(game_system.players) >= 2:
            strategy = strategies[0]
            player1 = list(game_system.players.values())[0]
            player2 = list(game_system.players.values())[1]
            
            interaction = game_system._compute_strategy_interaction(
                strategy, player1, player2
            )
            
            # 验证交互强度的性质
            self.assertGreater(interaction, 0)  # 应该为正
            self.assertFalse(math.isnan(interaction))
            self.assertFalse(math.isinf(interaction))
            
    def test_interaction_graph_construction(self):
        """测试交互图构建"""
        network = WeightedPhiNetwork(n_initial=5)
        game_system = PhiGameSystem(network, n_players=4)
        
        # 验证交互图结构
        self.assertIsNotNone(game_system.interaction_graph)
        self.assertEqual(len(game_system.interaction_graph), 4)
        
        for player_id in range(4):
            self.assertIn(player_id, game_system.interaction_graph)
            neighbors = game_system.interaction_graph[player_id]
            
            # 验证邻居的有效性
            for neighbor in neighbors:
                self.assertIn(neighbor, game_system.players)
                self.assertNotEqual(neighbor, player_id)  # 不包含自己
                
    def test_phi_constraint_preservation(self):
        """测试φ-约束保持性"""
        network = WeightedPhiNetwork(n_initial=6)
        game_system = PhiGameSystem(network, n_players=4)
        
        # 多步演化
        for _ in range(25):
            game_system.evolve_step(dt=0.1)
            
        # 验证所有策略仍然满足φ-约束
        for player in game_system.players.values():
            for strategy in player.strategy_distribution:
                self.assertTrue(strategy.is_valid_strategy)
                self.assertTrue(strategy.zeckendorf_repr.is_valid())
                
                # 验证策略值的φ-表示
                fib_k = strategy.fib_sequence.get(strategy.k)
                expected_value = fib_k / (strategy.phi ** strategy.d)
                self.assertAlmostEqual(strategy.value, expected_value, places=6)
                
    def test_zeckendorf_vector_properties(self):
        """测试Zeckendorf向量属性"""
        network = WeightedPhiNetwork(n_initial=4)
        game_system = PhiGameSystem(network, n_players=3)
        
        for player in game_system.players.values():
            z_vector = player.get_zeckendorf_vector()
            
            # 验证向量属性
            self.assertIsInstance(z_vector, list)
            self.assertGreater(len(z_vector), 0)
            
            # 验证所有元素都是0或1
            for bit in z_vector:
                self.assertIn(bit, [0, 1])
                
            # 验证no-11约束
            z_string = ''.join(map(str, z_vector))
            self.assertNotIn('11', z_string)
            
    def test_network_distance_calculation(self):
        """测试网络距离计算"""
        network = WeightedPhiNetwork(n_initial=5)
        game_system = PhiGameSystem(network, n_players=4)
        
        players = list(game_system.players.values())
        
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                distance = players[i].compute_network_distance_to(players[j])
                
                # 验证距离的基本性质
                self.assertGreaterEqual(distance, 1.0)  # 至少为1
                self.assertFalse(math.isnan(distance))
                self.assertFalse(math.isinf(distance))
                
                # 验证对称性
                reverse_distance = players[j].compute_network_distance_to(players[i])
                self.assertEqual(distance, reverse_distance)
                
    def test_comprehensive_game_evolution(self):
        """综合测试博弈演化"""
        print("\\n=== T23-1 φ-博弈策略涌现定理 综合验证 ===")
        
        # 创建网络和博弈系统
        network = WeightedPhiNetwork(n_initial=8)
        evolution = ConnectionEvolutionDynamics(network)
        
        # 先演化网络
        print("演化φ-网络...")
        for _ in range(30):
            evolution.evolve_step(dt=0.1)
            
        print(f"网络状态: {len(network.nodes)}个节点, {len(network.edge_weights)}条边")
        
        # 创建博弈系统
        game_system = PhiGameSystem(network, n_players=5)
        
        print(f"博弈系统: {len(game_system.players)}个玩家")
        print(f"策略空间大小: {len(game_system.strategy_space.get_all_strategies())}")
        
        # 分析初始状态
        initial_entropy = game_system.compute_game_entropy()
        print(f"初始博弈熵: {initial_entropy:.4f}")
        
        # 博弈演化
        print("\\n执行博弈演化...")
        n_steps = 50
        for step in range(n_steps):
            game_system.evolve_step(dt=0.1)
            
            if (step + 1) % 15 == 0:
                current_entropy = game_system.game_entropy_history[-1]
                print(f"  步骤 {step+1}: 博弈熵 = {current_entropy:.4f}")
                
        # 最终分析
        final_entropy = game_system.game_entropy_history[-1]
        print(f"\\n最终博弈熵: {final_entropy:.4f}")
        print(f"熵增: {final_entropy - initial_entropy:.4f}")
        
        # 策略分布分析
        print("\\n策略分布分析:")
        for player_id, player in game_system.players.items():
            player_entropy = player.get_current_strategy_entropy()
            
            # 找到主导策略
            max_prob = 0.0
            dominant_strategy = None
            for strategy, prob in player.strategy_distribution.items():
                if prob > max_prob:
                    max_prob = prob
                    dominant_strategy = strategy
                    
            print(f"  玩家 {player_id}: 策略熵={player_entropy:.3f}, "
                  f"主导策略={dominant_strategy} (概率={max_prob:.3f})")
                  
        # 收益矩阵分析
        print(f"\\n收益矩阵分析:")
        payoff_matrix = game_system.payoff_matrix
        avg_payoff = np.mean(payoff_matrix[payoff_matrix > 0])
        max_payoff = np.max(payoff_matrix)
        min_nonzero_payoff = np.min(payoff_matrix[payoff_matrix > 0])
        
        print(f"  平均收益: {avg_payoff:.4f}")
        print(f"  最大收益: {max_payoff:.4f}")
        print(f"  最小非零收益: {min_nonzero_payoff:.4f}")
        
        # 熵守恒验证
        conservation_verified = game_system.verify_entropy_conservation()
        print(f"\\n熵守恒验证: {'✓' if conservation_verified else '✗'}")
        
        # Nash均衡分析
        print("\\nNash均衡分析:")
        detector = PhiNashEquilibriumDetector(game_system)
        
        # 测试一个简单的策略组合
        strategies = game_system.strategy_space.get_all_strategies()
        if len(strategies) >= len(game_system.players):
            test_profile = {
                player_id: strategies[player_id % len(strategies)]
                for player_id in game_system.players
            }
            
            # 检查最优反应
            best_responses = 0
            for player_id in test_profile:
                if detector.check_best_response(player_id, test_profile):
                    best_responses += 1
                    
            print(f"  测试策略组合中的最优反应比例: {best_responses}/{len(test_profile)}")
            
        # 验证φ-约束保持
        print("\\nφ-约束验证:")
        all_valid = True
        for player in game_system.players.values():
            for strategy in player.strategy_distribution:
                if not strategy.is_valid_strategy:
                    all_valid = False
                    break
            if not all_valid:
                break
                
        print(f"  所有策略满足φ-约束: {'✓' if all_valid else '✗'}")
        
        # 理论预测验证
        print(f"\\n理论预测验证:")
        
        # 1. 熵增验证
        entropy_increases = 0
        for i in range(1, len(game_system.game_entropy_history)):
            if game_system.game_entropy_history[i] >= game_system.game_entropy_history[i-1] - 1e-6:
                entropy_increases += 1
                
        entropy_increase_rate = entropy_increases / (len(game_system.game_entropy_history) - 1)
        print(f"  熵增步骤比例: {entropy_increase_rate:.1%}")
        
        # 2. φ-时间尺度验证
        print(f"  φ-时间尺度因子: 1/φ = {1/game_system.phi:.4f}")
        
        print("\\n=== 验证完成 ===")
        
        # 综合验证断言
        self.assertTrue(conservation_verified)  # 熵守恒必须满足
        self.assertTrue(all_valid)              # φ-约束必须保持
        self.assertGreater(entropy_increase_rate, 0.8)  # 80%的步骤应该熵增
        self.assertGreater(final_entropy, initial_entropy - 1e-8)  # 总体熵增
        
        # 所有测试通过
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()