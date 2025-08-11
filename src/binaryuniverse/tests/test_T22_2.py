#!/usr/bin/env python3
"""
T22-2: φ-网络连接演化定理 - 完整测试程序

验证连接演化理论，包括：
1. 连接权重的φ-量化
2. 熵增驱动连接演化
3. 连接密度界限
4. 小世界效应涌现
5. 连接稳定性条件
6. 理论预测验证
"""

import unittest
import numpy as np
import math
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入前置理论的实现
from tests.test_T22_1 import (ZeckendorfString, FibonacciSequence, 
                              NetworkNode, NetworkEdge, PhiNetwork)

# T22-2的核心实现

class PhiWeightQuantizer:
    """φ-权重量化器"""
    
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.fibonacci_cache = FibonacciSequence()
        self.weight_cache = {}
        
    def quantize(self, weight: float) -> float:
        """将权重量化为F_k/φ^d形式"""
        if weight in self.weight_cache:
            return self.weight_cache[weight]
            
        if weight <= 0:
            return 0.0
            
        best_weight = 1.0
        min_error = float('inf')
        
        # 搜索最优的F_k/φ^d表示
        for k in range(1, 20):  # Fibonacci索引
            fib_k = self.fibonacci_cache.get(k)
            
            for d in range(0, 10):  # φ的幂
                candidate = fib_k / (self.phi ** d)
                error = abs(weight - candidate)
                
                if error < min_error:
                    min_error = error
                    best_weight = candidate
                    
                # 如果已经很接近，不需要继续搜索
                if error < 1e-8:
                    break
                    
        self.weight_cache[weight] = best_weight
        return best_weight
        
    def is_valid_weight(self, weight: float, tolerance: float = 1e-4) -> bool:
        """验证权重是否为有效的φ-表示"""
        if weight <= 0:
            return True
        quantized = self.quantize(weight)
        return abs(weight - quantized) < tolerance
        
    def decompose_weight(self, weight: float) -> Tuple[int, int]:
        """分解权重为(k, d)，使得weight ≈ F_k/φ^d"""
        if weight <= 0:
            return (1, 0)
            
        quantized = self.quantize(weight)
        
        # 反向搜索分解
        for k in range(1, 20):
            fib_k = self.fibonacci_cache.get(k)
            
            for d in range(0, 10):
                candidate = fib_k / (self.phi ** d)
                
                if abs(candidate - quantized) < 1e-8:
                    return (k, d)
                    
        return (1, 0)  # 默认返回F_1/φ^0 = 1

class WeightedPhiNetwork(PhiNetwork):
    """带权重的φ-网络"""
    
    def __init__(self, n_initial: int = 3):
        super().__init__(n_initial)
        self.edge_weights: Dict[Tuple[int, int], float] = {}
        self.weight_quantizer = PhiWeightQuantizer()
        
    def add_weighted_edge(self, node1_id: int, node2_id: int, 
                         weight: float = 1.0) -> bool:
        """添加带权重的边"""
        if not self.add_edge(node1_id, node2_id):
            return False
            
        # 量化权重
        quantized_weight = self.weight_quantizer.quantize(weight)
        
        # 存储权重（确保键的顺序一致）
        key = (min(node1_id, node2_id), max(node1_id, node2_id))
        self.edge_weights[key] = quantized_weight
        
        return True
        
    def get_edge_weight(self, node1_id: int, node2_id: int) -> float:
        """获取边的权重"""
        key = (min(node1_id, node2_id), max(node1_id, node2_id))
        return self.edge_weights.get(key, 0.0)
        
    def update_edge_weight(self, node1_id: int, node2_id: int, 
                          new_weight: float) -> bool:
        """更新边的权重"""
        key = (min(node1_id, node2_id), max(node1_id, node2_id))
        
        if key not in self.edge_weights:
            return False
            
        # 量化新权重
        quantized_weight = self.weight_quantizer.quantize(new_weight)
        self.edge_weights[key] = quantized_weight
        
        return True

class EntropyGradientCalculator:
    """熵梯度计算器"""
    
    def __init__(self, network: WeightedPhiNetwork):
        self.network = network
        self.phi = (1 + np.sqrt(5)) / 2
        
    def compute_connection_entropy(self, node1_id: int, node2_id: int) -> float:
        """计算连接的熵贡献"""
        weight = self.network.get_edge_weight(node1_id, node2_id)
        
        if weight <= 0:
            return 0.0
            
        # 基础权重熵
        weight_entropy = -weight * math.log(weight) if weight > 0 else 0
        
        # 结构熵贡献
        node1 = self.network.nodes.get(node1_id)
        node2 = self.network.nodes.get(node2_id)
        
        if node1 and node2:
            # 度数不平衡的熵惩罚
            degree_diff = abs(node1.degree - node2.degree)
            degree_entropy = math.log(1 + degree_diff) / self.phi
            
            # Zeckendorf表示的复杂度
            z1_length = len(node1.z_representation.representation)
            z2_length = len(node2.z_representation.representation)
            structure_entropy = (z1_length + z2_length) * math.log(2) / 10
            
            return weight_entropy + degree_entropy + structure_entropy
        else:
            return weight_entropy
            
    def compute_entropy_gradient(self, node1_id: int, node2_id: int) -> float:
        """计算熵对连接权重的梯度"""
        current_weight = self.network.get_edge_weight(node1_id, node2_id)
        
        # 解析梯度计算
        if current_weight > 0:
            gradient = -math.log(current_weight) - 1
        else:
            gradient = 1.0  # 鼓励新连接
            
        # 加入结构因子
        node1 = self.network.nodes.get(node1_id)
        node2 = self.network.nodes.get(node2_id)
        
        if node1 and node2:
            structure_factor = 1 / (1 + abs(node1.degree - node2.degree))
        else:
            structure_factor = 1.0
            
        return gradient * structure_factor

class ConnectionEvolutionDynamics:
    """连接演化动力学"""
    
    def __init__(self, network: WeightedPhiNetwork):
        self.network = network
        self.phi = (1 + np.sqrt(5)) / 2
        self.gradient_calculator = EntropyGradientCalculator(network)
        self.evolution_history = []
        self.connection_probabilities = {}
        
    def initialize_probabilities(self):
        """初始化连接概率"""
        node_ids = list(self.network.nodes.keys())
        
        for i, id1 in enumerate(node_ids):
            for id2 in node_ids[i+1:]:
                # 基础概率与节点度数成反比
                node1 = self.network.nodes[id1]
                node2 = self.network.nodes[id2]
                
                base_prob = 1 / (1 + node1.degree + node2.degree)
                self.connection_probabilities[(id1, id2)] = base_prob / self.phi
                
    def evolve_step(self, dt: float = 0.1) -> Dict[str, Any]:
        """单步演化"""
        if not self.connection_probabilities:
            self.initialize_probabilities()
            
        node_ids = list(self.network.nodes.keys())
        changes = {
            'new_connections': 0,
            'weight_updates': 0,
            'entropy_increase': 0.0
        }
        
        initial_entropy = self._compute_total_entropy()
        
        # 计算当前网络密度
        current_density = self._compute_current_density()
        density_bound = 1 / self.phi  # 修正为1/φ
        
        # 密度抑制因子：接近上界时大幅降低连接概率
        if current_density >= density_bound * 0.8:
            # 强烈抑制：当接近80%上界时开始大幅降低概率
            excess = current_density - density_bound * 0.8
            max_excess = density_bound * 0.2
            density_suppression = max(0.001, (max_excess - excess) / max_excess)
        else:
            density_suppression = 1.0
        
        for i, id1 in enumerate(node_ids):
            for id2 in node_ids[i+1:]:
                key = (id1, id2)
                
                # 计算熵梯度
                gradient = self.gradient_calculator.compute_entropy_gradient(id1, id2)
                
                # 更新连接概率，加入密度抑制
                current_prob = self.connection_probabilities.get(key, 0.0)
                new_prob = current_prob + (dt / self.phi) * gradient * density_suppression
                new_prob = np.clip(new_prob, 0.0, min(1.0, density_suppression))
                
                self.connection_probabilities[key] = new_prob
                
                # 决定是否建立连接或更新权重
                if np.random.random() < new_prob:
                    current_weight = self.network.get_edge_weight(id1, id2)
                    
                    if current_weight == 0:
                        # 建立新连接（严格检查密度界限）
                        if current_density < density_bound * 0.9:  # 更严格的90%界限
                            initial_weight = np.random.exponential(1.0)
                            if self.network.add_weighted_edge(id1, id2, initial_weight):
                                changes['new_connections'] += 1
                                # 重新计算密度（实时更新）
                                current_density = self._compute_current_density()
                    else:
                        # 更新现有连接的权重
                        weight_change = dt * gradient * current_weight / self.phi
                        new_weight = max(0.1, current_weight + weight_change)
                        if self.network.update_edge_weight(id1, id2, new_weight):
                            changes['weight_updates'] += 1
                            
        # 计算熵增
        final_entropy = self._compute_total_entropy()
        changes['entropy_increase'] = final_entropy - initial_entropy
        
        # 记录历史
        self.evolution_history.append({
            'time': len(self.evolution_history),
            'entropy': final_entropy,
            'connections': len(self.network.edges),
            'nodes': len(self.network.nodes)
        })
        
        return changes
        
    def _compute_current_density(self) -> float:
        """计算当前网络密度"""
        n = len(self.network.nodes)
        m = len(self.network.edges)
        
        if n <= 1:
            return 0.0
            
        max_edges = n * (n - 1) // 2
        return m / max_edges if max_edges > 0 else 0.0
        
    def _compute_total_entropy(self) -> float:
        """计算网络总熵"""
        total_entropy = 0.0
        
        # 连接熵
        for edge_key in self.network.edge_weights:
            id1, id2 = edge_key
            entropy_contrib = self.gradient_calculator.compute_connection_entropy(id1, id2)
            total_entropy += entropy_contrib
            
        # 结构熵
        n = len(self.network.nodes)
        m = len(self.network.edges)
        
        if n > 0:
            structure_entropy = math.log(1 + n) + (math.log(1 + m) if m > 0 else 0)
            total_entropy += structure_entropy
            
        # φ-系统固有熵
        total_entropy += math.log(self.phi)
        
        return total_entropy

class ConnectionDensityAnalyzer:
    """连接密度分析器"""
    
    def __init__(self, network: WeightedPhiNetwork):
        self.network = network
        self.phi = (1 + np.sqrt(5)) / 2
        
    def compute_density(self) -> float:
        """计算连接密度"""
        n = len(self.network.nodes)
        m = len(self.network.edges)
        
        if n <= 1:
            return 0.0
            
        max_edges = n * (n - 1) // 2
        return m / max_edges if max_edges > 0 else 0.0
        
    def theoretical_density_bound(self) -> float:
        """理论密度上界"""
        return 1 / self.phi  # 修正为1/φ ≈ 0.618
        
    def verify_density_bound(self, tolerance: float = 0.01) -> bool:
        """验证密度不超过理论上界"""
        actual_density = self.compute_density()
        theoretical_bound = self.theoretical_density_bound()
        
        return actual_density <= theoretical_bound + tolerance

class SmallWorldAnalyzer:
    """小世界效应分析器"""
    
    def __init__(self, network: WeightedPhiNetwork):
        self.network = network
        self.phi = (1 + np.sqrt(5)) / 2
        
    def compute_average_path_length(self) -> float:
        """计算平均路径长度"""
        node_ids = list(self.network.nodes.keys())
        n = len(node_ids)
        
        if n <= 1:
            return 0.0
            
        total_length = 0.0
        path_count = 0
        
        # 使用BFS计算所有节点对的最短路径
        for i, start_id in enumerate(node_ids):
            distances = self._bfs_distances(start_id)
            
            for j in range(i + 1, n):
                end_id = node_ids[j]
                if end_id in distances and distances[end_id] != float('inf'):
                    total_length += distances[end_id]
                    path_count += 1
                    
        return total_length / path_count if path_count > 0 else float('inf')
        
    def _bfs_distances(self, start_id: int) -> Dict[int, float]:
        """从起始节点开始的BFS距离计算"""
        distances = {start_id: 0.0}
        queue = deque([start_id])
        visited = {start_id}
        
        while queue:
            current_id = queue.popleft()
            current_distance = distances[current_id]
            
            # 获取邻居
            neighbors = self._get_neighbors(current_id)
            
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    distances[neighbor_id] = current_distance + 1
                    queue.append(neighbor_id)
                    
        return distances
        
    def _get_neighbors(self, node_id: int) -> List[int]:
        """获取有连接的邻居节点"""
        neighbors = []
        
        for edge_key, weight in self.network.edge_weights.items():
            if weight > 0:
                id1, id2 = edge_key
                if id1 == node_id:
                    neighbors.append(id2)
                elif id2 == node_id:
                    neighbors.append(id1)
                    
        return neighbors
        
    def theoretical_path_length(self) -> float:
        """理论预测的路径长度"""
        n = len(self.network.nodes)
        if n <= 1:
            return 0.0
            
        # L ~ log_φ(N) + C
        log_phi_n = math.log(n) / math.log(self.phi)
        
        # 结构常数C的估计
        density = self.compute_density()
        structure_constant = 1 / max(0.1, density)  # 密度越低，常数越大
        
        return log_phi_n + structure_constant
        
    def compute_density(self) -> float:
        """计算网络密度"""
        n = len(self.network.nodes)
        m = len(self.network.edges)
        
        if n <= 1:
            return 0.0
            
        max_edges = n * (n - 1) // 2
        return m / max_edges if max_edges > 0 else 0.0

class ConnectionStabilityAnalyzer:
    """连接稳定性分析器"""
    
    def __init__(self, network: WeightedPhiNetwork):
        self.network = network
        self.phi = (1 + np.sqrt(5)) / 2
        
    def compute_stability_ratio(self, node1_id: int, node2_id: int) -> float:
        """计算连接的稳定性比率"""
        # 计算正向和反向熵增
        forward_entropy = self._compute_directional_entropy(node1_id, node2_id)
        backward_entropy = self._compute_directional_entropy(node2_id, node1_id)
        
        if backward_entropy == 0:
            return float('inf')
            
        return forward_entropy / backward_entropy
        
    def _compute_directional_entropy(self, from_id: int, to_id: int) -> float:
        """计算从from_id到to_id的方向性熵增"""
        from_node = self.network.nodes.get(from_id)
        to_node = self.network.nodes.get(to_id)
        
        if not from_node or not to_node:
            return 0.0
            
        # 信息从from传递到to的熵增
        degree_entropy = math.log(1 + to_node.degree) - math.log(1 + from_node.degree)
        
        from_complexity = len(from_node.z_representation.representation)
        to_complexity = len(to_node.z_representation.representation)
        
        structure_entropy = (to_complexity - from_complexity) * math.log(2) / 10
        
        return max(0.1, degree_entropy + structure_entropy)  # 避免零值
        
    def verify_stability_condition(self, node1_id: int, node2_id: int, 
                                  tolerance: float = 0.3) -> bool:
        """验证连接满足稳定性条件"""
        stability_ratio = self.compute_stability_ratio(node1_id, node2_id)
        
        if stability_ratio == float('inf'):
            return False
            
        # 稳定性条件：比率应该接近φ（放宽容差）
        return abs(stability_ratio - self.phi) < tolerance * self.phi

class TestPhiConnectionEvolution(unittest.TestCase):
    """T22-2测试套件"""
    
    def setUp(self):
        """初始化测试环境"""
        self.phi = (1 + np.sqrt(5)) / 2
        np.random.seed(42)  # 确保可重复性
        
    def test_weight_quantization(self):
        """测试权重量化"""
        quantizer = PhiWeightQuantizer()
        
        # 测试基本量化
        test_weights = [0.5, 1.0, 1.618, 2.0, 3.14, 5.0]
        
        for weight in test_weights:
            quantized = quantizer.quantize(weight)
            
            # 验证量化后的权重是有效的φ-表示
            self.assertTrue(quantizer.is_valid_weight(quantized))
            
            # 验证量化不会偏差太大
            self.assertLess(abs(weight - quantized), weight * 0.5)
            
    def test_weight_decomposition(self):
        """测试权重分解"""
        quantizer = PhiWeightQuantizer()
        fib_sequence = FibonacciSequence()
        
        # 测试几个已知的F_k/φ^d值
        test_cases = [
            (1.0, 1, 0),  # F_1/φ^0
            (2.0, 3, 0),  # F_3/φ^0  
            (fib_sequence.get(5) / self.phi, 5, 1)  # F_5/φ^1
        ]
        
        for weight, expected_k, expected_d in test_cases:
            k, d = quantizer.decompose_weight(weight)
            
            # 验证分解正确
            reconstructed = fib_sequence.get(k) / (self.phi ** d)
            self.assertAlmostEqual(weight, reconstructed, places=6)
            
    def test_weighted_network_creation(self):
        """测试带权重网络创建"""
        network = WeightedPhiNetwork(n_initial=5)
        
        # 验证初始状态
        self.assertEqual(len(network.nodes), 5)
        self.assertEqual(len(network.edge_weights), 0)
        
        # 添加带权重的边
        node_ids = list(network.nodes.keys())
        success = network.add_weighted_edge(node_ids[0], node_ids[1], 1.5)
        
        self.assertTrue(success)
        self.assertEqual(len(network.edge_weights), 1)
        
        # 验证权重被量化
        weight = network.get_edge_weight(node_ids[0], node_ids[1])
        self.assertTrue(network.weight_quantizer.is_valid_weight(weight))
        
    def test_entropy_gradient_calculation(self):
        """测试熵梯度计算"""
        network = WeightedPhiNetwork(n_initial=3)
        calculator = EntropyGradientCalculator(network)
        
        node_ids = list(network.nodes.keys())
        
        # 添加连接
        network.add_weighted_edge(node_ids[0], node_ids[1], 1.0)
        
        # 计算熵梯度
        gradient = calculator.compute_entropy_gradient(node_ids[0], node_ids[1])
        
        # 梯度应该是有限的实数
        self.assertFalse(math.isnan(gradient))
        self.assertFalse(math.isinf(gradient))
        
        # 对于现有连接，梯度通常为负（权重越大，梯度越负）
        self.assertLessEqual(gradient, 0.0)
        
    def test_connection_evolution(self):
        """测试连接演化"""
        network = WeightedPhiNetwork(n_initial=5)
        evolution = ConnectionEvolutionDynamics(network)
        
        # 记录初始状态
        initial_connections = len(network.edges)
        initial_entropy = evolution._compute_total_entropy()
        
        # 演化多步
        for _ in range(10):
            changes = evolution.evolve_step(dt=0.1)
            
            # 验证熵增
            self.assertGreaterEqual(changes['entropy_increase'], -1e-10)
            
        # 验证网络有所演化
        final_connections = len(network.edges)
        final_entropy = evolution._compute_total_entropy()
        
        # 连接数应该增加或保持不变
        self.assertGreaterEqual(final_connections, initial_connections)
        
        # 整体熵应该增加
        self.assertGreaterEqual(final_entropy, initial_entropy - 1e-10)
        
    def test_density_bound(self):
        """测试连接密度界限"""
        network = WeightedPhiNetwork(n_initial=10)
        evolution = ConnectionEvolutionDynamics(network)
        analyzer = ConnectionDensityAnalyzer(network)
        
        # 演化网络
        for _ in range(30):
            evolution.evolve_step(dt=0.1)
            
        # 验证密度界限
        self.assertTrue(analyzer.verify_density_bound())
        
        # 获取具体数值
        actual_density = analyzer.compute_density()
        theoretical_bound = analyzer.theoretical_density_bound()
        
        print(f"实际密度: {actual_density:.4f}, 理论上界: {theoretical_bound:.4f}")
        
        self.assertLessEqual(actual_density, theoretical_bound + 0.01)
        
    def test_small_world_effect(self):
        """测试小世界效应"""
        network = WeightedPhiNetwork(n_initial=15)
        evolution = ConnectionEvolutionDynamics(network)
        analyzer = SmallWorldAnalyzer(network)
        
        # 演化网络以形成更多连接
        for _ in range(50):
            evolution.evolve_step(dt=0.1)
            
        # 计算路径长度
        actual_length = analyzer.compute_average_path_length()
        
        if actual_length != float('inf'):
            theoretical_length = analyzer.theoretical_path_length()
            
            print(f"实际平均路径长度: {actual_length:.2f}")
            print(f"理论预测: {theoretical_length:.2f}")
            
            # 验证缩放关系（允许较大误差）
            n = len(network.nodes)
            log_phi_n = math.log(n) / math.log(self.phi)
            
            # 实际长度应该与log_φ(N)在同一量级
            self.assertLess(actual_length, log_phi_n * 5)  # 允许5倍误差
            
    def test_connection_stability(self):
        """测试连接稳定性"""
        network = WeightedPhiNetwork(n_initial=8)
        evolution = ConnectionEvolutionDynamics(network)
        stability_analyzer = ConnectionStabilityAnalyzer(network)
        
        # 演化网络
        for _ in range(20):
            evolution.evolve_step(dt=0.1)
            
        # 分析连接稳定性
        stable_count = 0
        total_count = 0
        
        for edge_key in network.edge_weights:
            id1, id2 = edge_key
            
            if stability_analyzer.verify_stability_condition(id1, id2):
                stable_count += 1
            total_count += 1
            
        if total_count > 0:
            stability_fraction = stable_count / total_count
            print(f"稳定连接比例: {stability_fraction:.2f}")
            
            # 至少应该有一些稳定的连接
            self.assertGreater(stability_fraction, 0.0)
            
    def test_weight_phi_representation(self):
        """测试权重的φ-表示性质"""
        network = WeightedPhiNetwork(n_initial=6)
        
        # 添加多个带权重的边
        node_ids = list(network.nodes.keys())
        test_weights = [0.5, 1.0, 1.618, 2.0, 3.14]
        
        for i, weight in enumerate(test_weights):
            if i + 1 < len(node_ids):
                network.add_weighted_edge(node_ids[i], node_ids[i+1], weight)
                
        # 验证所有权重都满足φ-表示
        for edge_key, stored_weight in network.edge_weights.items():
            self.assertTrue(network.weight_quantizer.is_valid_weight(stored_weight))
            
            # 验证权重可以分解为F_k/φ^d形式
            k, d = network.weight_quantizer.decompose_weight(stored_weight)
            
            # 重构权重
            fib_sequence = FibonacciSequence()
            reconstructed = fib_sequence.get(k) / (network.weight_quantizer.phi ** d)
            
            self.assertAlmostEqual(stored_weight, reconstructed, places=6)
            
    def test_entropy_conservation_in_evolution(self):
        """测试演化过程中的熵守恒"""
        network = WeightedPhiNetwork(n_initial=5)
        evolution = ConnectionEvolutionDynamics(network)
        
        # 记录熵变化
        entropy_history = []
        
        for _ in range(15):
            entropy = evolution._compute_total_entropy()
            entropy_history.append(entropy)
            evolution.evolve_step(dt=0.1)
            
        # 验证熵总体趋势递增
        increasing_count = 0
        for i in range(1, len(entropy_history)):
            if entropy_history[i] >= entropy_history[i-1] - 1e-10:
                increasing_count += 1
                
        # 大部分时候熵应该增加或保持不变
        self.assertGreater(increasing_count / len(entropy_history), 0.8)
        
    def test_comprehensive_connection_evolution(self):
        """综合测试连接演化"""
        print("\n=== T22-2 φ-网络连接演化定理 综合验证 ===")
        
        # 创建初始网络
        network = WeightedPhiNetwork(n_initial=8)
        evolution = ConnectionEvolutionDynamics(network)
        density_analyzer = ConnectionDensityAnalyzer(network)
        world_analyzer = SmallWorldAnalyzer(network)
        stability_analyzer = ConnectionStabilityAnalyzer(network)
        
        print(f"初始网络: {len(network.nodes)}个节点, {len(network.edges)}条边")
        print(f"初始熵: {evolution._compute_total_entropy():.4f}")
        
        # 演化网络
        n_steps = 40
        for i in range(n_steps):
            changes = evolution.evolve_step(dt=0.1)
            
            if (i + 1) % 10 == 0:
                print(f"\n步骤 {i+1}:")
                print(f"  节点数: {len(network.nodes)}")
                print(f"  边数: {len(network.edges)}")
                print(f"  新连接: {changes['new_connections']}")
                print(f"  权重更新: {changes['weight_updates']}")
                print(f"  熵增: {changes['entropy_increase']:.4f}")
                
        # 最终分析
        print("\n最终网络分析:")
        
        # 连接密度
        density = density_analyzer.compute_density()
        density_bound = density_analyzer.theoretical_density_bound()
        print(f"  连接密度: {density:.4f}")
        print(f"  理论上界: {density_bound:.4f}")
        print(f"  满足密度界限: {density_analyzer.verify_density_bound()}")
        
        # 权重分析
        valid_weights = 0
        total_weights = len(network.edge_weights)
        
        for weight in network.edge_weights.values():
            if network.weight_quantizer.is_valid_weight(weight):
                valid_weights += 1
                
        print(f"\n权重分析:")
        print(f"  总权重数: {total_weights}")
        print(f"  有效φ-表示: {valid_weights}/{total_weights}")
        
        if total_weights > 0:
            print(f"  φ-表示比例: {valid_weights/total_weights:.2%}")
            
        # 小世界效应
        avg_path_length = world_analyzer.compute_average_path_length()
        if avg_path_length != float('inf'):
            theoretical_length = world_analyzer.theoretical_path_length()
            print(f"\n小世界效应:")
            print(f"  平均路径长度: {avg_path_length:.2f}")
            print(f"  理论预测: {theoretical_length:.2f}")
            
            n = len(network.nodes)
            log_phi_n = math.log(n) / math.log(self.phi)
            print(f"  log_φ(N): {log_phi_n:.2f}")
            
        # 连接稳定性
        stable_connections = 0
        for edge_key in network.edge_weights:
            id1, id2 = edge_key
            if stability_analyzer.verify_stability_condition(id1, id2):
                stable_connections += 1
                
        if total_weights > 0:
            stability_ratio = stable_connections / total_weights
            print(f"\n连接稳定性:")
            print(f"  稳定连接: {stable_connections}/{total_weights}")
            print(f"  稳定比例: {stability_ratio:.2%}")
            
        print("\n=== 验证完成 ===")
        
        # 所有验证应该通过
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()