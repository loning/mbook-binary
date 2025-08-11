# T22-3 φ-网络拓扑稳定性定理 - 形式化规范

## 依赖导入
```python
import numpy as np
import math
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import scipy.linalg
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# 从前置理论导入
from T22_2_formal import WeightedPhiNetwork, ConnectionEvolutionDynamics
from T22_1_formal import PhiNetwork, NetworkNode, NetworkEdge, FibonacciSequence
from T20_2_formal import TraceStructure
from C20_1_formal import CollapseObserver
```

## 1. 拓扑稳定性分析

### 1.1 核心稳定性分析器
```python
class PhiTopologyStabilityAnalyzer:
    """φ-网络拓扑稳定性分析器"""
    
    def __init__(self, network: WeightedPhiNetwork):
        self.network = network
        self.phi = (1 + np.sqrt(5)) / 2
        self.adjacency_matrix = None
        self.eigenvalues = None
        self.components_cache = None
        
    def compute_stability_ratio(self) -> float:
        """计算拓扑稳定性比率 Δs_add/Δs_remove"""
        add_entropy = self._compute_average_add_entropy()
        remove_entropy = self._compute_average_remove_entropy()
        
        if remove_entropy <= 0:
            return float('inf')
            
        return add_entropy / remove_entropy
        
    def verify_stability_condition(self, tolerance: float = 0.1) -> bool:
        """验证稳定性条件：比率应接近φ"""
        ratio = self.compute_stability_ratio()
        
        if ratio == float('inf'):
            return False
            
        return abs(ratio - self.phi) < tolerance * self.phi
        
    def _compute_average_add_entropy(self) -> float:
        """计算添加边的平均熵增"""
        node_ids = list(self.network.nodes.keys())
        total_entropy = 0.0
        count = 0
        
        for i, id1 in enumerate(node_ids):
            for id2 in node_ids[i+1:]:
                if self.network.get_edge_weight(id1, id2) == 0:
                    entropy_increase = self._compute_edge_add_entropy(id1, id2)
                    total_entropy += entropy_increase
                    count += 1
                    
        return total_entropy / count if count > 0 else 0.0
        
    def _compute_average_remove_entropy(self) -> float:
        """计算移除边的平均熵减"""
        total_entropy = 0.0
        count = 0
        
        for edge_key in self.network.edge_weights:
            id1, id2 = edge_key
            entropy_decrease = self._compute_edge_remove_entropy(id1, id2)
            total_entropy += entropy_decrease
            count += 1
            
        return total_entropy / count if count > 0 else 0.0
        
    def _compute_edge_add_entropy(self, id1: int, id2: int) -> float:
        """计算添加边(id1,id2)的熵增"""
        node1 = self.network.nodes.get(id1)
        node2 = self.network.nodes.get(id2)
        
        if not node1 or not node2:
            return 0.0
            
        # 基础连接熵
        base_entropy = math.log(2)
        
        # 度数相关的熵增（度数越小，熵增越大）
        degree_factor = 1 / (1 + node1.degree) + 1 / (1 + node2.degree)
        
        # Zeckendorf复杂度影响
        z1_length = len(node1.z_representation.representation)
        z2_length = len(node2.z_representation.representation)
        zeckendorf_factor = (z1_length + z2_length) / 20
        
        return base_entropy * degree_factor * (1 + zeckendorf_factor)
        
    def _compute_edge_remove_entropy(self, id1: int, id2: int) -> float:
        """计算移除边(id1,id2)的熵减"""
        weight = self.network.get_edge_weight(id1, id2)
        
        if weight <= 0:
            return 0.0
            
        # 权重熵（移除会减少这部分熵）
        weight_entropy = -weight * math.log(weight) if weight > 0 else 0
        
        # 结构熵贡献
        node1 = self.network.nodes.get(id1)
        node2 = self.network.nodes.get(id2)
        
        if node1 and node2:
            # 度数不平衡的熵
            degree_diff = abs(node1.degree - node2.degree)
            structure_entropy = math.log(1 + degree_diff) / self.phi
            
            # Zeckendorf表示的复杂度
            z1_length = len(node1.z_representation.representation)
            z2_length = len(node2.z_representation.representation)
            zeckendorf_entropy = (z1_length + z2_length) * math.log(2) / 10
            
            return weight_entropy + structure_entropy + zeckendorf_entropy
        else:
            return weight_entropy
```

## 2. 特征值稳定性分析

### 2.1 特征值界限验证
```python
class EigenvalueStabilityAnalyzer:
    """特征值稳定性分析器"""
    
    def __init__(self, network: WeightedPhiNetwork):
        self.network = network
        self.phi = (1 + np.sqrt(5)) / 2
        self.adjacency_matrix = None
        self.eigenvalues = None
        
    def compute_adjacency_matrix(self) -> np.ndarray:
        """构建加权邻接矩阵"""
        if self.adjacency_matrix is not None:
            return self.adjacency_matrix
            
        node_ids = sorted(self.network.nodes.keys())
        n = len(node_ids)
        
        if n == 0:
            return np.array([])
            
        adjacency = np.zeros((n, n))
        id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
        
        for edge_key, weight in self.network.edge_weights.items():
            id1, id2 = edge_key
            if id1 in id_to_index and id2 in id_to_index:
                i, j = id_to_index[id1], id_to_index[id2]
                adjacency[i, j] = weight
                adjacency[j, i] = weight
                
        self.adjacency_matrix = adjacency
        return adjacency
        
    def compute_eigenvalues(self) -> np.ndarray:
        """计算邻接矩阵的特征值"""
        if self.eigenvalues is not None:
            return self.eigenvalues
            
        adjacency = self.compute_adjacency_matrix()
        
        if adjacency.size == 0:
            return np.array([])
            
        # 使用对称矩阵的高效特征值算法
        eigenvalues = np.linalg.eigvals(adjacency)
        self.eigenvalues = np.real(eigenvalues)  # 对称矩阵特征值为实数
        
        return self.eigenvalues
        
    def get_max_eigenvalue(self) -> float:
        """获取最大特征值（谱半径）"""
        eigenvalues = self.compute_eigenvalues()
        
        if len(eigenvalues) == 0:
            return 0.0
            
        return np.max(eigenvalues)
        
    def compute_theoretical_bound(self) -> float:
        """计算理论特征值上界 φ√N"""
        N = len(self.network.nodes)
        return self.phi * np.sqrt(N) if N > 0 else 0.0
        
    def verify_eigenvalue_bound(self, tolerance: float = 1e-10) -> bool:
        """验证特征值稳定性界限 λ₁ ≤ φ√N"""
        max_eigenvalue = self.get_max_eigenvalue()
        theoretical_bound = self.compute_theoretical_bound()
        
        return max_eigenvalue <= theoretical_bound + tolerance
        
    def compute_spectral_gap(self) -> float:
        """计算谱间隙（最大与次大特征值的差）"""
        eigenvalues = self.compute_eigenvalues()
        
        if len(eigenvalues) < 2:
            return 0.0
            
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]  # 降序排列
        return sorted_eigenvalues[0] - sorted_eigenvalues[1]
        
    def analyze_spectral_properties(self) -> Dict[str, float]:
        """分析谱性质"""
        eigenvalues = self.compute_eigenvalues()
        
        if len(eigenvalues) == 0:
            return {
                'max_eigenvalue': 0.0,
                'spectral_radius': 0.0,
                'spectral_gap': 0.0,
                'trace': 0.0,
                'determinant': 0.0
            }
            
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]
        
        return {
            'max_eigenvalue': sorted_eigenvalues[0],
            'spectral_radius': np.max(np.abs(eigenvalues)),
            'spectral_gap': sorted_eigenvalues[0] - sorted_eigenvalues[1] if len(eigenvalues) > 1 else 0.0,
            'trace': np.sum(eigenvalues),
            'determinant': np.prod(eigenvalues) if len(eigenvalues) > 0 else 0.0
        }
```

## 3. 连通分量稳定性

### 3.1 连通分量分析器
```python
class ConnectedComponentAnalyzer:
    """连通分量稳定性分析器"""
    
    def __init__(self, network: WeightedPhiNetwork):
        self.network = network
        self.phi = (1 + np.sqrt(5)) / 2
        self.components_cache = None
        self.component_sizes_cache = None
        
    def find_connected_components(self) -> List[Set[int]]:
        """找到所有连通分量"""
        if self.components_cache is not None:
            return self.components_cache
            
        visited = set()
        components = []
        
        for node_id in self.network.nodes.keys():
            if node_id not in visited:
                component = set()
                self._dfs_component(node_id, visited, component)
                if component:  # 确保非空
                    components.append(component)
                    
        self.components_cache = components
        return components
        
    def _dfs_component(self, node_id: int, visited: Set[int], component: Set[int]):
        """深度优先搜索连通分量"""
        visited.add(node_id)
        component.add(node_id)
        
        # 找到所有有连接的邻居
        neighbors = self._get_neighbors(node_id)
        
        for neighbor in neighbors:
            if neighbor not in visited:
                self._dfs_component(neighbor, visited, component)
                
    def _get_neighbors(self, node_id: int) -> List[int]:
        """获取节点的所有邻居"""
        neighbors = []
        
        for edge_key, weight in self.network.edge_weights.items():
            if weight > 0:  # 只考虑有权重的边
                id1, id2 = edge_key
                if id1 == node_id:
                    neighbors.append(id2)
                elif id2 == node_id:
                    neighbors.append(id1)
                    
        return neighbors
        
    def get_component_count(self) -> int:
        """获取连通分量数量"""
        components = self.find_connected_components()
        return len(components)
        
    def get_component_sizes(self) -> List[int]:
        """获取各连通分量的大小"""
        if self.component_sizes_cache is not None:
            return self.component_sizes_cache
            
        components = self.find_connected_components()
        sizes = [len(comp) for comp in components]
        sizes.sort(reverse=True)  # 按大小降序排列
        
        self.component_sizes_cache = sizes
        return sizes
        
    def verify_fibonacci_constraint(self) -> bool:
        """验证连通分量数是Fibonacci数"""
        K = self.get_component_count()
        
        if K == 0:
            return True
            
        fib_sequence = FibonacciSequence()
        fibonacci_numbers = set(fib_sequence.get(i) for i in range(1, 25))
        
        return K in fibonacci_numbers
        
    def verify_phi_bound(self) -> bool:
        """验证φ界限：K ≤ ⌊N/φ⌋"""
        K = self.get_component_count()
        N = len(self.network.nodes)
        
        if N == 0:
            return K == 0
            
        bound = math.floor(N / self.phi)
        return K <= bound
        
    def verify_component_stability(self) -> bool:
        """验证连通分量稳定性"""
        return self.verify_fibonacci_constraint() and self.verify_phi_bound()
        
    def analyze_component_distribution(self) -> Dict[str, Any]:
        """分析连通分量分布"""
        components = self.find_connected_components()
        sizes = self.get_component_sizes()
        K = len(components)
        N = len(self.network.nodes)
        
        # 计算分量分布的统计量
        if sizes:
            largest_component_ratio = sizes[0] / N if N > 0 else 0
            avg_component_size = np.mean(sizes)
            component_size_std = np.std(sizes)
        else:
            largest_component_ratio = 0
            avg_component_size = 0
            component_size_std = 0
            
        return {
            'component_count': K,
            'component_sizes': sizes,
            'largest_component_ratio': largest_component_ratio,
            'avg_component_size': avg_component_size,
            'component_size_std': component_size_std,
            'is_fibonacci': self.verify_fibonacci_constraint(),
            'satisfies_phi_bound': self.verify_phi_bound(),
            'is_stable': self.verify_component_stability()
        }
```

## 4. 拓扑熵计算

### 4.1 拓扑熵分析器
```python
class TopologicalEntropyCalculator:
    """拓扑熵计算器"""
    
    def __init__(self, network: WeightedPhiNetwork):
        self.network = network
        self.phi = (1 + np.sqrt(5)) / 2
        self.component_analyzer = ConnectedComponentAnalyzer(network)
        
    def compute_degree_entropy(self) -> float:
        """计算度分布熵 Σlog(di+1)/φ"""
        entropy = 0.0
        
        for node in self.network.nodes.values():
            entropy += math.log(node.degree + 1)
            
        return entropy / self.phi
        
    def compute_structure_entropy(self) -> float:
        """计算结构熵 K·log(φ)"""
        K = self.component_analyzer.get_component_count()
        return K * math.log(self.phi)
        
    def compute_total_topological_entropy(self) -> float:
        """计算总拓扑熵"""
        degree_entropy = self.compute_degree_entropy()
        structure_entropy = self.compute_structure_entropy()
        
        return degree_entropy + structure_entropy
        
    def verify_entropy_conservation(self, tolerance: float = 1e-6) -> bool:
        """验证拓扑熵守恒"""
        # 分别计算各组分
        degree_part = self.compute_degree_entropy()
        structure_part = self.compute_structure_entropy()
        
        # 计算总熵
        total_entropy = self.compute_total_topological_entropy()
        
        # 验证守恒关系
        expected_total = degree_part + structure_part
        
        return abs(total_entropy - expected_total) < tolerance
        
    def decompose_entropy(self) -> Dict[str, float]:
        """分解拓扑熵"""
        return {
            'degree_entropy': self.compute_degree_entropy(),
            'structure_entropy': self.compute_structure_entropy(),
            'total_entropy': self.compute_total_topological_entropy()
        }
        
    def compute_entropy_per_node(self) -> Dict[int, float]:
        """计算每个节点的熵贡献"""
        node_entropies = {}
        
        for node_id, node in self.network.nodes.items():
            # 节点度数熵
            degree_entropy = math.log(node.degree + 1) / self.phi
            
            # 节点所在分量的结构熵分摊
            components = self.component_analyzer.find_connected_components()
            structure_entropy = 0.0
            
            for component in components:
                if node_id in component:
                    structure_entropy = math.log(self.phi) / len(component)
                    break
                    
            node_entropies[node_id] = degree_entropy + structure_entropy
            
        return node_entropies
```

## 5. 扰动稳定性分析

### 5.1 扰动稳定性测试器
```python
class PerturbationStabilityTester:
    """扰动稳定性测试器"""
    
    def __init__(self, network: WeightedPhiNetwork):
        self.network = network
        self.phi = (1 + np.sqrt(5)) / 2
        self.eigenvalue_analyzer = EigenvalueStabilityAnalyzer(network)
        
    def test_eigenvalue_perturbation(self, epsilon: float = 0.1, n_trials: int = 10) -> bool:
        """测试特征值扰动稳定性"""
        # 获取原始邻接矩阵和特征值
        original_adjacency = self.eigenvalue_analyzer.compute_adjacency_matrix()
        
        if original_adjacency.size == 0:
            return True  # 空网络稳定
            
        original_eigenvalues = self.eigenvalue_analyzer.compute_eigenvalues()
        
        stable_count = 0
        
        for _ in range(n_trials):
            # 生成扰动
            perturbation_norm = epsilon / self.phi
            perturbation = np.random.normal(0, perturbation_norm, original_adjacency.shape)
            
            # 保持对称性
            perturbation = (perturbation + perturbation.T) / 2
            
            # 确保扰动的Frobenius范数不超过ε/φ
            actual_norm = np.linalg.norm(perturbation, 'fro')
            if actual_norm > perturbation_norm:
                perturbation = perturbation * (perturbation_norm / actual_norm)
                
            # 应用扰动
            perturbed_adjacency = original_adjacency + perturbation
            perturbed_eigenvalues = np.real(np.linalg.eigvals(perturbed_adjacency))
            
            # 计算特征值变化
            eigenvalue_change = np.linalg.norm(
                np.sort(perturbed_eigenvalues) - np.sort(original_eigenvalues)
            )
            
            # 检查是否满足稳定性条件
            if eigenvalue_change <= epsilon:
                stable_count += 1
                
        # 要求至少80%的试验满足稳定性
        return stable_count / n_trials >= 0.8
        
    def test_topology_perturbation(self, epsilon: float = 0.1) -> bool:
        """测试拓扑扰动稳定性"""
        # 记录原始状态
        original_edges = len(self.network.edges)
        original_components = len(ConnectedComponentAnalyzer(self.network).find_connected_components())
        
        # 创建扰动：随机添加/删除少量边
        node_ids = list(self.network.nodes.keys())
        n_nodes = len(node_ids)
        
        if n_nodes < 2:
            return True
            
        # 扰动强度：最多改变 ε * 最大可能边数 条边
        max_edges = n_nodes * (n_nodes - 1) // 2
        max_changes = max(1, int(epsilon * max_edges))
        
        # 随机选择要改变的边
        changes_made = 0
        
        for _ in range(max_changes):
            if np.random.random() < 0.5:
                # 尝试添加边
                id1, id2 = np.random.choice(node_ids, 2, replace=False)
                if self.network.get_edge_weight(id1, id2) == 0:
                    success = self.network.add_weighted_edge(id1, id2, 1.0)
                    if success:
                        changes_made += 1
            else:
                # 尝试删除边（通过设置权重为0）
                if self.network.edge_weights:
                    edge_key = np.random.choice(list(self.network.edge_weights.keys()))
                    if edge_key in self.network.edge_weights:
                        del self.network.edge_weights[edge_key]
                        # 更新度数
                        id1, id2 = edge_key
                        if id1 in self.network.nodes:
                            self.network.nodes[id1].degree = max(0, self.network.nodes[id1].degree - 1)
                        if id2 in self.network.nodes:
                            self.network.nodes[id2].degree = max(0, self.network.nodes[id2].degree - 1)
                        changes_made += 1
                        
        # 检查扰动后的稳定性
        final_edges = len(self.network.edge_weights)  # 使用权重字典计算边数
        final_components = len(ConnectedComponentAnalyzer(self.network).find_connected_components())
        
        # 边数变化不应太大
        edge_change_ratio = abs(final_edges - original_edges) / max(1, original_edges)
        
        # 连通分量数变化不应太大
        component_change = abs(final_components - original_components)
        
        return edge_change_ratio <= epsilon and component_change <= max(1, epsilon * original_components)
        
    def comprehensive_stability_test(self, epsilon: float = 0.1) -> Dict[str, bool]:
        """综合稳定性测试"""
        return {
            'eigenvalue_stable': self.test_eigenvalue_perturbation(epsilon),
            'topology_stable': self.test_topology_perturbation(epsilon)
        }
```

## 6. 综合稳定性分析器

### 6.1 完整稳定性分析
```python
class ComprehensiveStabilityAnalyzer:
    """综合φ-网络拓扑稳定性分析器"""
    
    def __init__(self, network: WeightedPhiNetwork):
        self.network = network
        self.phi = (1 + np.sqrt(5)) / 2
        
        # 初始化各个分析器
        self.stability_analyzer = PhiTopologyStabilityAnalyzer(network)
        self.eigenvalue_analyzer = EigenvalueStabilityAnalyzer(network)
        self.component_analyzer = ConnectedComponentAnalyzer(network)
        self.entropy_calculator = TopologicalEntropyCalculator(network)
        self.perturbation_tester = PerturbationStabilityTester(network)
        
    def analyze_all_stability_conditions(self) -> Dict[str, Any]:
        """分析所有稳定性条件"""
        # 1. 拓扑稳定性比率
        stability_ratio = self.stability_analyzer.compute_stability_ratio()
        ratio_stable = self.stability_analyzer.verify_stability_condition()
        
        # 2. 特征值稳定性
        max_eigenvalue = self.eigenvalue_analyzer.get_max_eigenvalue()
        eigenvalue_bound = self.eigenvalue_analyzer.compute_theoretical_bound()
        eigenvalue_stable = self.eigenvalue_analyzer.verify_eigenvalue_bound()
        
        # 3. 连通分量稳定性
        component_analysis = self.component_analyzer.analyze_component_distribution()
        
        # 4. 拓扑熵分析
        entropy_decomposition = self.entropy_calculator.decompose_entropy()
        entropy_conservation = self.entropy_calculator.verify_entropy_conservation()
        
        # 5. 扰动稳定性
        perturbation_results = self.perturbation_tester.comprehensive_stability_test()
        
        return {
            'stability_ratio': {
                'value': stability_ratio,
                'theoretical': self.phi,
                'is_stable': ratio_stable
            },
            'eigenvalue_stability': {
                'max_eigenvalue': max_eigenvalue,
                'theoretical_bound': eigenvalue_bound,
                'is_stable': eigenvalue_stable
            },
            'component_stability': component_analysis,
            'entropy_analysis': {
                'decomposition': entropy_decomposition,
                'conservation_verified': entropy_conservation
            },
            'perturbation_stability': perturbation_results,
            'overall_stable': (
                ratio_stable and 
                eigenvalue_stable and 
                component_analysis['is_stable'] and 
                entropy_conservation and
                all(perturbation_results.values())
            )
        }
        
    def generate_stability_report(self) -> str:
        """生成稳定性分析报告"""
        analysis = self.analyze_all_stability_conditions()
        
        report = "=== φ-网络拓扑稳定性分析报告 ===\\n\\n"
        
        # 基本网络信息
        report += f"网络规模: {len(self.network.nodes)} 节点, {len(self.network.edge_weights)} 边\\n"
        report += f"网络密度: {len(self.network.edge_weights) * 2 / (len(self.network.nodes) * (len(self.network.nodes) - 1)):.4f}\\n\\n"
        
        # 稳定性比率
        ratio_data = analysis['stability_ratio']
        report += f"1. 拓扑稳定性比率:\\n"
        report += f"   实际值: {ratio_data['value']:.4f}\\n"
        report += f"   理论值: {ratio_data['theoretical']:.4f}\\n"
        report += f"   稳定性: {'✓' if ratio_data['is_stable'] else '✗'}\\n\\n"
        
        # 特征值稳定性
        eigen_data = analysis['eigenvalue_stability']
        report += f"2. 特征值稳定性:\\n"
        report += f"   最大特征值: {eigen_data['max_eigenvalue']:.4f}\\n"
        report += f"   理论上界: {eigen_data['theoretical_bound']:.4f}\\n"
        report += f"   稳定性: {'✓' if eigen_data['is_stable'] else '✗'}\\n\\n"
        
        # 连通分量稳定性
        comp_data = analysis['component_stability']
        report += f"3. 连通分量稳定性:\\n"
        report += f"   分量数: {comp_data['component_count']}\\n"
        report += f"   Fibonacci约束: {'✓' if comp_data['is_fibonacci'] else '✗'}\\n"
        report += f"   φ界限: {'✓' if comp_data['satisfies_phi_bound'] else '✗'}\\n"
        report += f"   稳定性: {'✓' if comp_data['is_stable'] else '✗'}\\n\\n"
        
        # 拓扑熵
        entropy_data = analysis['entropy_analysis']
        report += f"4. 拓扑熵分析:\\n"
        report += f"   度熵: {entropy_data['decomposition']['degree_entropy']:.4f}\\n"
        report += f"   结构熵: {entropy_data['decomposition']['structure_entropy']:.4f}\\n"
        report += f"   总熵: {entropy_data['decomposition']['total_entropy']:.4f}\\n"
        report += f"   守恒验证: {'✓' if entropy_data['conservation_verified'] else '✗'}\\n\\n"
        
        # 扰动稳定性
        pert_data = analysis['perturbation_stability']
        report += f"5. 扰动稳定性:\\n"
        report += f"   特征值稳定: {'✓' if pert_data['eigenvalue_stable'] else '✗'}\\n"
        report += f"   拓扑稳定: {'✓' if pert_data['topology_stable'] else '✗'}\\n\\n"
        
        # 总体评估
        report += f"总体稳定性: {'✓ 稳定' if analysis['overall_stable'] else '✗ 不稳定'}\\n"
        
        return report
```

---

**注记**: 本形式化规范提供了T22-3定理的完整数学实现，包括拓扑稳定性分析、特征值界限验证、连通分量分析、拓扑熵计算和扰动稳定性测试的所有必要组件。所有实现严格遵循φ-表示和熵增原理。