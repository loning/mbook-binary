# T22-2 φ-网络连接演化定理 - 形式化规范

## 依赖导入
```python
import numpy as np
import math
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import networkx as nx

# 从前置理论导入
from T22_1_formal import PhiNetwork, NetworkNode, NetworkEdge, FibonacciSequence
from T20_2_formal import TraceStructure
from C20_1_formal import CollapseObserver
```

## 1. 连接权重量化

### 1.1 φ-权重表示
```python
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
            
        best_weight = 1.0
        min_error = float('inf')
        
        # 搜索最优的F_k/φ^d表示
        for k in range(1, 25):  # Fibonacci索引
            fib_k = self.fibonacci_cache.get(k)
            
            for d in range(0, 15):  # φ的幂
                candidate = fib_k / (self.phi ** d)
                error = abs(weight - candidate)
                
                if error < min_error:
                    min_error = error
                    best_weight = candidate
                    
                # 如果已经很接近，不需要继续搜索
                if error < 1e-10:
                    break
                    
        self.weight_cache[weight] = best_weight
        return best_weight
        
    def is_valid_weight(self, weight: float, tolerance: float = 1e-6) -> bool:
        """验证权重是否为有效的φ-表示"""
        quantized = self.quantize(weight)
        return abs(weight - quantized) < tolerance
        
    def decompose_weight(self, weight: float) -> Tuple[int, int]:
        """分解权重为(k, d)，使得weight ≈ F_k/φ^d"""
        quantized = self.quantize(weight)
        
        # 反向搜索分解
        for k in range(1, 25):
            fib_k = self.fibonacci_cache.get(k)
            
            for d in range(0, 15):
                candidate = fib_k / (self.phi ** d)
                
                if abs(candidate - quantized) < 1e-10:
                    return (k, d)
                    
        return (1, 0)  # 默认返回F_1/φ^0 = 1
```

### 1.2 加权网络结构
```python
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
```

## 2. 熵增驱动的连接演化

### 2.1 熵梯度计算
```python
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
        
        # 数值梯度计算
        epsilon = 1e-6
        
        # 当前熵
        current_entropy = self.compute_connection_entropy(node1_id, node2_id)
        
        # 微扰后的熵
        if current_weight > 0:
            # 现有连接：计算权重变化的影响
            perturbed_weight = current_weight + epsilon
            # 临时更新权重计算熵
            original_weight = current_weight
            
            # 简化：直接计算解析梯度
            if current_weight > 0:
                gradient = -math.log(current_weight) - 1
            else:
                gradient = 1.0  # 鼓励新连接
                
        else:
            # 新连接：鼓励建立连接
            gradient = 1.0
            
        # 加入结构因子
        node1 = self.network.nodes.get(node1_id)
        node2 = self.network.nodes.get(node2_id)
        
        if node1 and node2:
            structure_factor = 1 / (1 + abs(node1.degree - node2.degree))
        else:
            structure_factor = 1.0
            
        return gradient * structure_factor
```

### 2.2 连接演化动力学
```python
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
        
        for i, id1 in enumerate(node_ids):
            for id2 in node_ids[i+1:]:
                key = (id1, id2)
                
                # 计算熵梯度
                gradient = self.gradient_calculator.compute_entropy_gradient(id1, id2)
                
                # 更新连接概率
                current_prob = self.connection_probabilities.get(key, 0.0)
                new_prob = current_prob + (dt / self.phi) * gradient
                new_prob = np.clip(new_prob, 0.0, 1.0)
                
                self.connection_probabilities[key] = new_prob
                
                # 决定是否建立连接或更新权重
                if np.random.random() < new_prob:
                    current_weight = self.network.get_edge_weight(id1, id2)
                    
                    if current_weight == 0:
                        # 建立新连接
                        initial_weight = np.random.exponential(1.0)
                        if self.network.add_weighted_edge(id1, id2, initial_weight):
                            changes['new_connections'] += 1
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
        
    def _compute_total_entropy(self) -> float:
        """计算网络总熵"""
        total_entropy = 0.0
        
        # 连接熵
        for edge in self.network.edges:
            entropy_contrib = self.gradient_calculator.compute_connection_entropy(
                edge.node1.id, edge.node2.id)
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
```

## 3. 连接密度分析

### 3.1 密度界限验证
```python
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
        return 1 / (self.phi ** 2)
        
    def verify_density_bound(self, tolerance: float = 0.01) -> bool:
        """验证密度不超过理论上界"""
        actual_density = self.compute_density()
        theoretical_bound = self.theoretical_density_bound()
        
        return actual_density <= theoretical_bound + tolerance
        
    def compute_local_densities(self) -> Dict[int, float]:
        """计算每个节点的局部密度"""
        local_densities = {}
        
        for node_id, node in self.network.nodes.items():
            # 获取邻居节点
            neighbors = self._get_neighbors(node_id)
            k = len(neighbors)
            
            if k <= 1:
                local_densities[node_id] = 0.0
                continue
                
            # 计算邻居间的连接数
            neighbor_connections = 0
            for i, neighbor1 in enumerate(neighbors):
                for neighbor2 in neighbors[i+1:]:
                    if self.network.get_edge_weight(neighbor1, neighbor2) > 0:
                        neighbor_connections += 1
                        
            # 局部密度 = 实际连接数 / 最大可能连接数
            max_connections = k * (k - 1) // 2
            local_densities[node_id] = neighbor_connections / max_connections if max_connections > 0 else 0.0
            
        return local_densities
        
    def _get_neighbors(self, node_id: int) -> List[int]:
        """获取节点的邻居"""
        neighbors = []
        
        for edge_key, weight in self.network.edge_weights.items():
            if weight > 0:
                id1, id2 = edge_key
                if id1 == node_id:
                    neighbors.append(id2)
                elif id2 == node_id:
                    neighbors.append(id1)
                    
        return neighbors
```

## 4. 小世界效应

### 4.1 路径长度分析
```python
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
        density = len(self.network.edges) / (n * (n - 1) / 2) if n > 1 else 0
        structure_constant = 1 / max(0.1, density)  # 密度越低，常数越大
        
        return log_phi_n + structure_constant
        
    def verify_small_world_scaling(self, tolerance: float = 0.5) -> bool:
        """验证小世界缩放律"""
        actual_length = self.compute_average_path_length()
        theoretical_length = self.theoretical_path_length()
        
        if actual_length == float('inf') or theoretical_length == float('inf'):
            return False
            
        # 允许较大的相对误差
        relative_error = abs(actual_length - theoretical_length) / max(actual_length, theoretical_length)
        return relative_error < tolerance
        
    def compute_clustering_coefficient(self) -> float:
        """计算聚集系数"""
        node_ids = list(self.network.nodes.keys())
        total_clustering = 0.0
        node_count = 0
        
        for node_id in node_ids:
            neighbors = self._get_neighbors(node_id)
            k = len(neighbors)
            
            if k < 2:
                continue
                
            # 计算邻居间的连接数
            neighbor_connections = 0
            for i, neighbor1 in enumerate(neighbors):
                for neighbor2 in neighbors[i+1:]:
                    if self.network.get_edge_weight(neighbor1, neighbor2) > 0:
                        neighbor_connections += 1
                        
            # 局部聚集系数
            max_connections = k * (k - 1) // 2
            local_clustering = neighbor_connections / max_connections
            
            total_clustering += local_clustering
            node_count += 1
            
        return total_clustering / node_count if node_count > 0 else 0.0
```

## 5. 连接稳定性

### 5.1 稳定性条件验证
```python
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
        # 基于度数差异和Zeckendorf表示复杂度
        
        degree_entropy = math.log(1 + to_node.degree) - math.log(1 + from_node.degree)
        
        from_complexity = len(from_node.z_representation.representation)
        to_complexity = len(to_node.z_representation.representation)
        
        structure_entropy = (to_complexity - from_complexity) * math.log(2) / 10
        
        return degree_entropy + structure_entropy
        
    def verify_stability_condition(self, node1_id: int, node2_id: int, 
                                  tolerance: float = 0.1) -> bool:
        """验证连接满足稳定性条件"""
        stability_ratio = self.compute_stability_ratio(node1_id, node2_id)
        
        if stability_ratio == float('inf'):
            return False
            
        # 稳定性条件：比率应该接近φ
        return abs(stability_ratio - self.phi) < tolerance * self.phi
        
    def analyze_network_stability(self) -> Dict[str, Any]:
        """分析整个网络的稳定性"""
        stable_connections = 0
        unstable_connections = 0
        stability_ratios = []
        
        for edge_key in self.network.edge_weights:
            id1, id2 = edge_key
            
            if self.verify_stability_condition(id1, id2):
                stable_connections += 1
            else:
                unstable_connections += 1
                
            ratio = self.compute_stability_ratio(id1, id2)
            if ratio != float('inf'):
                stability_ratios.append(ratio)
                
        total_connections = stable_connections + unstable_connections
        
        return {
            'stable_fraction': stable_connections / total_connections if total_connections > 0 else 0,
            'mean_stability_ratio': np.mean(stability_ratios) if stability_ratios else 0,
            'std_stability_ratio': np.std(stability_ratios) if stability_ratios else 0,
            'total_connections': total_connections
        }
```

---

**注记**: 本形式化规范提供了T22-2定理的完整数学实现，包括权重量化、熵增演化、密度分析、小世界效应和稳定性验证的所有必要组件。所有实现严格遵循φ-表示和熵增原理。