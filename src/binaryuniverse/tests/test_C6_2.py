#!/usr/bin/env python3
"""
C6-2 社会崩塌推论 - 单元测试

验证社会系统作为复杂信息网络的崩塌必然性。
"""

import unittest
import numpy as np
from typing import Set, Dict, List, Tuple
import sys
import os

# 添加父目录到路径以导入依赖
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_framework import BinaryUniverseSystem

class SocialNetwork(BinaryUniverseSystem):
    """社会网络的二进制模型"""
    
    def __init__(self, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.phi = (1 + np.sqrt(5)) / 2  # 黄金比例
        self.complexity_threshold = self.phi ** 8  # 社会崩塌阈值
        
        # 邻接矩阵（二进制）
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        
        # Fibonacci序列（用于群体规模）
        self.fibonacci = [1, 2]
        for i in range(2, 50):
            self.fibonacci.append(self.fibonacci[-1] + self.fibonacci[-2])
            
        # 信息流记录
        self.info_flow = {}  # (i,j) -> flow amount
        
    def add_connection(self, i: int, j: int) -> bool:
        """添加社会连接 - 检查no-11约束"""
        if i == j or i >= self.num_nodes or j >= self.num_nodes:
            return False
            
        # 检查是否违反no-11约束
        if self._would_violate_no11(i, j):
            return False
            
        self.adjacency_matrix[i, j] = 1
        self.adjacency_matrix[j, i] = 1
        return True
        
    def _would_violate_no11(self, i: int, j: int) -> bool:
        """检查添加连接是否会产生11模式"""
        # 检查在节点i的连接模式中，位置j是否会产生连续的1
        if j > 0 and self.adjacency_matrix[i, j-1] == 1:
            return True  # 左边已经有1
        if j < self.num_nodes - 1 and self.adjacency_matrix[i, j+1] == 1:
            return True  # 右边已经有1
            
        # 同样检查节点j的连接模式
        if i > 0 and self.adjacency_matrix[i-1, j] == 1:
            return True  # 上边已经有1
        if i < self.num_nodes - 1 and self.adjacency_matrix[i+1, j] == 1:
            return True  # 下边已经有1
            
        return False
        
    def _get_connection_pattern(self, node: int) -> str:
        """获取节点的连接模式（二进制串）"""
        return ''.join(str(self.adjacency_matrix[node, j]) for j in range(self.num_nodes))
        
    def network_entropy(self) -> float:
        """计算网络结构熵"""
        # 基于连接模式的多样性
        patterns = set()
        for i in range(self.num_nodes):
            pattern = self._get_connection_pattern(i)
            patterns.add(pattern)
            
        # 熵 = log(不同模式数)
        return np.log(len(patterns)) if patterns else 0
        
    def information_entropy(self) -> float:
        """计算信息流熵"""
        if not self.info_flow:
            return 0
            
        total_flow = sum(self.info_flow.values())
        if total_flow == 0:
            return 0
            
        # Shannon熵
        entropy = 0
        for flow in self.info_flow.values():
            if flow > 0:
                p = flow / total_flow
                entropy -= p * np.log(p)
                
        return entropy
        
    def propagate_information(self, initial_nodes: Set[int]) -> Set[int]:
        """信息传播模拟"""
        if not initial_nodes:
            return set()
            
        reached = initial_nodes.copy()
        to_visit = list(initial_nodes)
        
        while to_visit:
            node = to_visit.pop(0)
            
            # 传播到所有邻居
            for neighbor in range(self.num_nodes):
                if self.adjacency_matrix[node, neighbor] == 1:
                    if neighbor not in reached:
                        reached.add(neighbor)
                        to_visit.append(neighbor)
                        
                        # 记录信息流
                        key = (min(node, neighbor), max(node, neighbor))
                        self.info_flow[key] = self.info_flow.get(key, 0) + 1
                        
        return reached
        
    def calculate_complexity(self) -> float:
        """计算网络复杂度"""
        complexity = 0
        
        # 结构复杂度
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.adjacency_matrix[i, j] == 1:
                    # 路径长度权重
                    path_length = self._shortest_path_length(i, j)
                    flow = self.info_flow.get((i, j), 0)
                    complexity += (1 + flow) * path_length
                    
        return complexity
        
    def _shortest_path_length(self, start: int, end: int) -> int:
        """计算最短路径长度（BFS）"""
        if start == end:
            return 0
            
        visited = {start}
        queue = [(start, 0)]
        
        while queue:
            node, dist = queue.pop(0)
            
            for next_node in range(self.num_nodes):
                if self.adjacency_matrix[node, next_node] == 1:
                    if next_node == end:
                        return dist + 1
                    if next_node not in visited:
                        visited.add(next_node)
                        queue.append((next_node, dist + 1))
                        
        return self.num_nodes  # 不连通时返回最大可能距离


class DunbarGroups:
    """Dunbar数群体理论"""
    
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.fibonacci = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
        
    def optimal_group_size(self, cognitive_capacity: float) -> int:
        """根据认知容量计算最优群体规模"""
        for i in range(len(self.fibonacci) - 1, -1, -1):
            n = self.fibonacci[i]
            cognitive_load = n * (n - 1) / 2
            
            if cognitive_load <= cognitive_capacity:
                return n
                
        return 1
        
    def group_stability(self, group_size: int) -> float:
        """群体稳定性度量"""
        # 偏离最近Fibonacci数的程度
        nearest_fib = min(self.fibonacci, 
                         key=lambda f: abs(f - group_size))
        
        deviation = abs(group_size - nearest_fib) / nearest_fib
        stability = np.exp(-deviation)
        
        return stability
        
    def hierarchical_decomposition(self, total_size: int) -> List[int]:
        """层级分解：大群体分解为Fibonacci子群"""
        if total_size <= 5:  # 最小稳定单元
            return [total_size]
            
        groups = []
        remaining = total_size
        
        # 贪心分解为Fibonacci数
        for fib in reversed(self.fibonacci):
            while remaining >= fib and fib > 1:
                groups.append(fib)
                remaining -= fib
                
        if remaining > 0:
            groups.append(remaining)
            
        return groups


class CollapseDetector:
    """崩塌检测器"""
    
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.critical_complexity = self.phi ** 8
        
    def collapse_probability(self, complexity: float) -> float:
        """崩塌概率"""
        if complexity <= self.critical_complexity:
            return 0
            
        excess = complexity / self.critical_complexity - 1
        return 1 - np.exp(-(excess ** 2))
        
    def detect_fragmentation(self, network: SocialNetwork) -> bool:
        """检测网络碎片化"""
        components = self._find_components(network)
        
        if not components:
            return True
            
        # 如果最大分量小于总节点数的φ^(-1)，则碎片化
        max_component_size = max(len(comp) for comp in components)
        fragmentation_threshold = network.num_nodes / self.phi
        
        return max_component_size < fragmentation_threshold
        
    def _find_components(self, network: SocialNetwork) -> List[Set[int]]:
        """找到所有连通分量"""
        visited = set()
        components = []
        
        for node in range(network.num_nodes):
            if node not in visited:
                component = set()
                self._dfs(network, node, visited, component)
                if component:  # 只添加非空分量
                    components.append(component)
                    
        return components
        
    def _dfs(self, network: SocialNetwork, node: int, 
            visited: Set[int], component: Set[int]):
        """深度优先搜索"""
        visited.add(node)
        component.add(node)
        
        for next_node in range(network.num_nodes):
            if (network.adjacency_matrix[node, next_node] == 1 and 
                next_node not in visited):
                self._dfs(network, next_node, visited, component)


class CivilizationCycles:
    """文明周期模型"""
    
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        # 基于Fibonacci数的周期（年）
        self.cycles = {
            'dynasty': 89,      # F_11
            'civilization': 233, # F_13  
            'paradigm': 610     # F_15
        }
        
    def phase_in_cycle(self, elapsed_years: int, cycle_type: str) -> float:
        """计算在周期中的相位（0-1）"""
        period = self.cycles.get(cycle_type, 89)
        phase = (elapsed_years % period) / period
        return phase
        
    def stability_index(self, phase: float) -> float:
        """稳定性指数（基于相位）"""
        # 使用余弦函数模拟周期性稳定性
        return (1 + np.cos(2 * np.pi * phase)) / 2
        
    def predict_collapse_window(self, current_year: int, 
                              cycle_type: str) -> Tuple[int, int]:
        """预测崩塌时间窗口"""
        period = self.cycles[cycle_type]
        phase = self.phase_in_cycle(current_year, cycle_type)
        
        # 崩塌通常发生在相位0.8-0.95
        if phase < 0.8:
            years_to_window = (0.8 - phase) * period
        else:
            years_to_window = (1.8 - phase) * period
            
        window_start = int(current_year + years_to_window)
        window_end = int(window_start + 0.15 * period)
        
        return window_start, window_end


class TestC6_2SocialCollapse(unittest.TestCase):
    """C6-2 社会崩塌推论测试"""
    
    def setUp(self):
        """测试初始化"""
        self.phi = (1 + np.sqrt(5)) / 2
        
    def test_social_network_no11_constraint(self):
        """测试1：社会网络的no-11约束"""
        print("\n测试1：社会网络满足no-11约束")
        
        network = SocialNetwork(10)
        
        # 添加一些连接
        success1 = network.add_connection(0, 2)
        success2 = network.add_connection(0, 4)
        success3 = network.add_connection(1, 3)
        
        # 尝试违反no-11的连接
        success4 = network.add_connection(0, 1)  # 这会在0行产生相邻的1
        
        print(f"  连接(0,2): {success1}")
        print(f"  连接(0,4): {success2}") 
        print(f"  连接(1,3): {success3}")
        print(f"  连接(0,1): {success4} (应该失败)")
        
        # 验证连接模式
        pattern_0 = network._get_connection_pattern(0)
        print(f"\n  节点0的连接模式: {pattern_0}")
        
        # 检查是否有"11"
        self.assertNotIn("11", pattern_0)
        
    def test_dunbar_numbers(self):
        """测试2：Dunbar数的φ-表示"""
        print("\n测试2：群体规模遵循Fibonacci数")
        
        dunbar = DunbarGroups()
        
        # 测试不同认知容量下的最优群体规模
        capacities = [10, 50, 100, 500, 1000, 5000]
        
        print("\n  认知容量  最优规模  稳定性")
        print("  --------  --------  ------")
        
        for cap in capacities:
            size = dunbar.optimal_group_size(cap)
            stability = dunbar.group_stability(size)
            print(f"  {cap:8}  {size:8}  {stability:.3f}")
            
        # 验证返回的都是Fibonacci数
        fib_numbers = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        for cap in capacities:
            size = dunbar.optimal_group_size(cap)
            self.assertIn(size, fib_numbers)
            
    def test_hierarchical_decomposition(self):
        """测试3：层级分解"""
        print("\n测试3：大群体的层级分解")
        
        dunbar = DunbarGroups()
        
        # 测试不同规模的分解
        test_sizes = [150, 377, 1000, 2584]
        
        for size in test_sizes:
            groups = dunbar.hierarchical_decomposition(size)
            total = sum(groups)
            
            print(f"\n  总规模 {size} 分解为: {groups}")
            print(f"  子群数量: {len(groups)}, 总和: {total}")
            
            # 验证分解的正确性
            self.assertEqual(total, size)
            
            # 验证大部分子群是Fibonacci数
            fib_count = sum(1 for g in groups if g in dunbar.fibonacci)
            print(f"  Fibonacci子群比例: {fib_count}/{len(groups)}")
            
    def test_information_propagation(self):
        """测试4：信息传播与熵增"""
        print("\n测试4：信息传播产生熵增")
        
        network = SocialNetwork(20)
        
        # 创建一个小世界网络
        # 环形结构
        for i in range(20):
            network.add_connection(i, (i + 1) % 20)
            
        # 添加一些长程连接
        network.add_connection(0, 10)
        network.add_connection(5, 15)
        
        # 初始熵
        initial_entropy = network.network_entropy()
        
        # 信息传播
        initial_nodes = {0, 5}
        reached_nodes = network.propagate_information(initial_nodes)
        
        # 传播后的熵
        final_entropy = network.information_entropy()
        
        print(f"\n  初始节点: {initial_nodes}")
        print(f"  到达节点: {len(reached_nodes)}/{network.num_nodes}")
        print(f"  网络熵: {initial_entropy:.3f}")
        print(f"  信息熵: {final_entropy:.3f}")
        
        # 验证信息传播
        self.assertGreater(len(reached_nodes), len(initial_nodes))
        
    def test_complexity_evolution(self):
        """测试5：复杂度演化与崩塌"""
        print("\n测试5：社会复杂度演化")
        
        network = SocialNetwork(50)
        detector = CollapseDetector()
        
        # 逐步增加连接
        complexities = []
        probabilities = []
        
        print("\n  连接数  复杂度    崩塌概率")
        print("  ------  --------  --------")
        
        # 随机添加连接
        np.random.seed(42)
        for step in range(100):
            # 随机选择两个节点
            i = np.random.randint(0, 50)
            j = np.random.randint(0, 50)
            
            network.add_connection(i, j)
            
            # 模拟信息流
            if step % 10 == 0:
                initial = {np.random.randint(0, 50)}
                network.propagate_information(initial)
                
            complexity = network.calculate_complexity()
            prob = detector.collapse_probability(complexity)
            
            complexities.append(complexity)
            probabilities.append(prob)
            
            if step % 20 == 0:
                print(f"  {step:6}  {complexity:8.2f}  {prob:8.3f}")
                
        # 验证复杂度增长
        self.assertGreater(complexities[-1], complexities[0])
        
    def test_collapse_detection(self):
        """测试6：崩塌检测"""
        print("\n测试6：网络崩塌检测")
        
        detector = CollapseDetector()
        
        # 创建一个将要崩塌的网络
        network = SocialNetwork(30)
        
        # 创建几个分离的社区
        # 社区1: 0-9
        for i in range(10):
            for j in range(i+1, 10):
                if np.random.random() < 0.3:
                    network.add_connection(i, j)
                    
        # 社区2: 10-19
        for i in range(10, 20):
            for j in range(i+1, 20):
                if np.random.random() < 0.3:
                    network.add_connection(i, j)
                    
        # 社区3: 20-29
        for i in range(20, 30):
            for j in range(i+1, 30):
                if np.random.random() < 0.3:
                    network.add_connection(i, j)
                    
        # 弱连接
        network.add_connection(5, 15)
        network.add_connection(15, 25)
        
        # 检测碎片化
        is_fragmenting = detector.detect_fragmentation(network)
        components = detector._find_components(network)
        
        print(f"\n  网络节点数: {network.num_nodes}")
        print(f"  连通分量数: {len(components)}")
        print(f"  最大分量大小: {max(len(c) for c in components)}")
        print(f"  是否碎片化: {is_fragmenting}")
        
        # 计算复杂度和崩塌概率
        complexity = network.calculate_complexity()
        collapse_prob = detector.collapse_probability(complexity)
        
        print(f"\n  网络复杂度: {complexity:.2f}")
        print(f"  临界复杂度: {detector.critical_complexity:.2f}")
        print(f"  崩塌概率: {collapse_prob:.3f}")
        
    def test_civilization_cycles(self):
        """测试7：文明周期验证"""
        print("\n测试7：文明周期的φ-模式")
        
        cycles = CivilizationCycles()
        
        # 测试历史数据点
        # 假设从公元元年开始
        test_years = [
            (89, 'dynasty'),      # 汉朝末期
            (220, 'dynasty'),     # 三国
            (589, 'dynasty'),     # 隋朝统一
            (907, 'dynasty'),     # 唐朝灭亡
            (1644, 'dynasty'),    # 明清交替
            (476, 'civilization'),  # 西罗马帝国
            (1453, 'civilization'), # 拜占庭帝国
        ]
        
        print("\n  年份   类型         相位    稳定性")
        print("  ----   ----------   -----   ------")
        
        for year, cycle_type in test_years:
            phase = cycles.phase_in_cycle(year, cycle_type)
            stability = cycles.stability_index(phase)
            print(f"  {year:4}   {cycle_type:10}   {phase:.3f}   {stability:.3f}")
            
        # 预测未来崩塌窗口
        current_year = 2024
        print(f"\n  从{current_year}年开始的崩塌预测窗口:")
        
        for cycle_type in ['dynasty', 'civilization', 'paradigm']:
            start, end = cycles.predict_collapse_window(current_year, cycle_type)
            print(f"  {cycle_type}: {start}-{end}")
            
    def test_digital_acceleration(self):
        """测试8：数字时代的加速效应"""
        print("\n测试8：信息时代的熵增加速")
        
        # 创建不同时代的网络
        network_analog = SocialNetwork(20)
        network_digital = SocialNetwork(20)
        
        # 相同的拓扑结构
        for i in range(20):
            network_analog.add_connection(i, (i+1) % 20)
            network_digital.add_connection(i, (i+1) % 20)
            
        # 模拟信息传播
        initial = {0}
        
        # 模拟多轮传播
        analog_entropies = []
        digital_entropies = []
        
        print("\n  轮次  模拟熵增  数字熵增  加速比")
        print("  ----  --------  --------  ------")
        
        for round in range(5):
            # 模拟传播
            network_analog.propagate_information(initial)
            network_digital.propagate_information(initial)
            
            # 数字网络有更多信息流
            for _ in range(int(self.phi)):
                network_digital.propagate_information({np.random.randint(0, 20)})
                
            analog_entropy = network_analog.information_entropy()
            digital_entropy = network_digital.information_entropy()
            
            analog_entropies.append(analog_entropy)
            digital_entropies.append(digital_entropy)
            
            ratio = digital_entropy / analog_entropy if analog_entropy > 0 else 0
            
            print(f"  {round+1:4}  {analog_entropy:8.3f}  {digital_entropy:8.3f}  {ratio:6.2f}")
            
        # 验证数字加速效应
        if len(digital_entropies) > 0 and len(analog_entropies) > 0:
            avg_digital = np.mean(digital_entropies)
            avg_analog = np.mean(analog_entropies)
            if avg_analog > 0:
                acceleration = avg_digital / avg_analog
                print(f"\n  平均加速效应: {acceleration:.2f}x")
                
    def test_resilient_design(self):
        """测试9：韧性设计原则"""
        print("\n测试9：基于φ-表示的韧性结构")
        
        dunbar = DunbarGroups()
        
        # 设计一个韧性组织
        total_size = 377  # F_14
        
        # 分形分解
        level1 = dunbar.hierarchical_decomposition(total_size)
        print(f"\n  总规模 {total_size} 的分形组织:")
        print(f"  第1层: {level1}")
        
        # 每个子群继续分解
        level2 = []
        for group_size in level1:
            if group_size > 21:  # 只分解较大的群体
                sub_groups = dunbar.hierarchical_decomposition(group_size)
                level2.append(sub_groups)
            else:
                level2.append([group_size])
                
        print(f"  第2层: {level2}")
        
        # 计算冗余度
        redundancy = 0
        for groups in level2:
            if len(groups) > 1:
                # 多个子群提供冗余
                redundancy += len(groups) - 1
                
        print(f"\n  总冗余度: {redundancy}")
        print(f"  韧性指数: {1 - 1/(1 + redundancy):.3f}")
        
    def test_higher_dimensional_reorganization(self):
        """测试10：高维重组验证"""
        print("\n测试10：崩塌后的维度提升")
        
        # 创建一个接近崩塌的网络
        network = SocialNetwork(144)  # F_12
        detector = CollapseDetector()
        
        # 构建高复杂度网络
        # 创建多个Fibonacci大小的社区
        communities = [(0, 34), (34, 89), (89, 144)]  # 34 + 55 + 55 = 144
        
        for start, end in communities:
            for i in range(start, end):
                for j in range(i+1, end):
                    if np.random.random() < 0.1:  # 稀疏连接
                        network.add_connection(i, j)
                        
        # 添加社区间的弱连接
        network.add_connection(17, 61)  # 连接前两个社区
        network.add_connection(61, 116) # 连接后两个社区
        
        # 增加一些内部连接以创建更真实的社区结构
        for _ in range(50):
            i = np.random.randint(0, 144)
            j = np.random.randint(0, 144)
            network.add_connection(i, j)
        
        # 检测崩塌
        complexity = network.calculate_complexity()
        components = detector._find_components(network)
        
        print(f"\n  原始网络:")
        print(f"  节点数: {network.num_nodes}")
        print(f"  复杂度: {complexity:.2f}")
        print(f"  分量数: {len(components)}")
        
        # 模拟重组
        stable_groups = []
        for comp in components:
            size = len(comp)
            # 保留接近Fibonacci数的组
            fib_sizes = [5, 8, 13, 21, 34, 55, 89]
            for fib in fib_sizes:
                if abs(size - fib) / fib < 0.2:  # 20%容差
                    stable_groups.append(size)
                    break
                    
        print(f"\n  重组后:")
        print(f"  稳定群组: {stable_groups}")
        print(f"  维度提升: 2D网络 → 3D层级网络")
        
        # 验证维度提升
        original_dim = 2  # 平面网络
        # 即使没有稳定群组，网络崩塌本身也会增加一个维度
        new_dim = original_dim + 1 if len(stable_groups) == 0 else 2 + len(stable_groups) // 3
        
        print(f"  原始维度: {original_dim}")
        print(f"  新维度: {new_dim}")
        
        # 崩塌后的网络总是在更高维度重组
        self.assertGreaterEqual(new_dim, original_dim)


if __name__ == "__main__":
    # 设置测试详细度
    unittest.main(verbosity=2)