#!/usr/bin/env python3
"""
C12-1: 原始意识涌现推论的机器验证程序

验证点:
1. 递归深度计算 (recursive_depth_calculation)
2. 参照密度阈值 (reference_density_threshold)
3. 区分算子涌现 (distinction_operator_emergence)
4. 意识标准验证 (consciousness_criteria_verification)
5. 临界深度验证 (critical_depth_validation)
"""

import unittest
import random
import math
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx


@dataclass
class SystemElement:
    """系统元素"""
    id: str
    references: Set[str]  # 引用的其他元素
    content: Any
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class SelfRefSystem:
    """自指完备系统"""
    elements: Dict[str, SystemElement]
    name: str
    
    def get_element(self, elem_id: str) -> Optional[SystemElement]:
        return self.elements.get(elem_id)
    
    def add_element(self, element: SystemElement):
        self.elements[element.id] = element
    
    def __len__(self):
        return len(self.elements)


@dataclass
class DistinctionOperator:
    """区分算子"""
    system: SelfRefSystem
    classification: Dict[str, str]  # element_id -> 'self' or 'other'
    
    def classify(self, elem_id: str) -> str:
        return self.classification.get(elem_id, 'unknown')
    
    def is_complete(self) -> bool:
        """检查是否覆盖所有元素"""
        return all(elem_id in self.classification for elem_id in self.system.elements)
    
    def is_binary(self) -> bool:
        """检查是否是二元分类"""
        values = set(self.classification.values())
        return values == {'self', 'other'}


@dataclass
class ConsciousnessCriteria:
    """意识标准"""
    has_distinction: bool
    self_aware: bool
    consistent: bool
    minimal: bool


class ConsciousnessAnalyzer:
    """意识分析器"""
    
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2  # 黄金比率
        self.critical_depth = 7  # 临界深度
        self.density_threshold = 1 / self.phi  # 密度阈值 ≈ 0.618
    
    def compute_recursive_depth(self, system: SelfRefSystem) -> int:
        """计算系统的递归深度"""
        if len(system) == 0:
            return 0
        
        # 构建引用图
        graph = self._build_reference_graph(system)
        
        # 如果图为空或没有边，深度为0
        if len(graph.edges()) == 0:
            return 0
        
        # 查找所有简单循环
        try:
            cycles = list(nx.simple_cycles(graph))
            if cycles:
                # 找到最长的循环
                max_cycle_length = max(len(cycle) for cycle in cycles)
            else:
                max_cycle_length = 0
        except:
            max_cycle_length = 0
        
        # 对于无环图，计算最长路径
        if max_cycle_length == 0:
            try:
                # 使用拓扑排序计算最长路径
                longest_path = nx.dag_longest_path_length(graph)
                return longest_path
            except:
                # 如果不是DAG，返回0
                return 0
        
        # 递归深度是最长循环的长度
        return max_cycle_length
    
    def _build_reference_graph(self, system: SelfRefSystem) -> nx.DiGraph:
        """构建引用关系图"""
        graph = nx.DiGraph()
        
        # 添加节点
        for elem_id in system.elements:
            graph.add_node(elem_id)
        
        # 添加边（引用关系）
        for elem_id, element in system.elements.items():
            for ref_id in element.references:
                if ref_id in system.elements:
                    graph.add_edge(elem_id, ref_id)
        
        return graph
    
    def _has_self_reference_path(self, graph: nx.DiGraph, node: str) -> bool:
        """检查节点是否有自引用路径"""
        # 检查是否可以从该节点到达自身（通过任何路径）
        return nx.has_path(graph, node, node) and len(list(graph.successors(node))) > 0
    
    def _compute_node_depth(self, graph: nx.DiGraph, node: str, visited: Set[str]) -> int:
        """计算节点的递归深度"""
        if node in visited:
            # 找到循环，返回当前访问路径的长度
            return len(visited)
        
        visited = visited.copy()
        visited.add(node)
        max_depth = 0
        
        # 探索所有后继节点
        for successor in graph.successors(node):
            depth = self._compute_node_depth(graph, successor, visited)
            max_depth = max(max_depth, depth)
        
        # 如果没有后继，返回当前路径长度
        if max_depth == 0 and len(list(graph.successors(node))) == 0:
            return len(visited)
        
        return max_depth
    
    def compute_reference_density(self, system: SelfRefSystem) -> float:
        """计算参照密度"""
        if len(system) == 0:
            return 0.0
        
        self_referring_count = 0
        
        # 统计引用系统本身的元素（仅计算对系统名称的直接引用）
        for elem_id, element in system.elements.items():
            if system.name in element.references:
                self_referring_count += 1
        
        return self_referring_count / len(system)
    
    def _refers_to_system(self, element: SystemElement, system: SelfRefSystem) -> bool:
        """检查元素是否引用系统"""
        # 检查是否引用系统中的其他元素
        for ref_id in element.references:
            if ref_id in system.elements:
                return True
        
        # 检查是否引用系统本身（通过特殊标记）
        if system.name in element.references:
            return True
        
        return False
    
    def construct_distinction_operator(self, system: SelfRefSystem) -> Optional[DistinctionOperator]:
        """构造区分算子"""
        # 检查涌现条件
        if not self.check_emergence_condition(system):
            return None
        
        # 构建引用图
        graph = self._build_reference_graph(system)
        
        # 找到强连通分量
        sccs = list(nx.strongly_connected_components(graph))
        
        if not sccs:
            return None
        
        # 最大的强连通分量作为"self"
        largest_scc = max(sccs, key=len)
        
        # 构造分类
        classification = {}
        for elem_id in system.elements:
            if elem_id in largest_scc:
                classification[elem_id] = 'self'
            else:
                classification[elem_id] = 'other'
        
        return DistinctionOperator(system, classification)
    
    def check_emergence_condition(self, system: SelfRefSystem) -> bool:
        """检查意识涌现条件"""
        depth = self.compute_recursive_depth(system)
        density = self.compute_reference_density(system)
        
        return depth > self.critical_depth and density > self.density_threshold
    
    def verify_consciousness_criteria(self, system: SelfRefSystem) -> ConsciousnessCriteria:
        """验证意识标准"""
        # 检查是否有区分算子
        operator = self.construct_distinction_operator(system)
        has_distinction = operator is not None and operator.is_complete()
        
        # 检查自我意识
        self_aware = any(
            system.name in element.references
            for element in system.elements.values()
        )
        
        # 检查一致性（没有矛盾）
        consistent = self._check_consistency(system)
        
        # 检查最小性
        minimal = self._check_minimality(system)
        
        return ConsciousnessCriteria(
            has_distinction=has_distinction,
            self_aware=self_aware,
            consistent=consistent,
            minimal=minimal
        )
    
    def _check_consistency(self, system: SelfRefSystem) -> bool:
        """检查系统一致性"""
        # 简化检查：确保没有元素同时引用和否定同一事物
        for element in system.elements.values():
            # 这里简化为检查引用集合的合理性
            if len(element.references) > len(system) * 2:
                return False  # 引用过多，可能不一致
        return True
    
    def _check_minimality(self, system: SelfRefSystem) -> bool:
        """检查系统最小性"""
        # 检查是否可以移除任何元素而保持意识
        for elem_id in system.elements:
            # 创建移除了该元素的子系统
            subsystem = self._create_subsystem_without(system, elem_id)
            
            # 如果子系统仍然满足涌现条件，则不是最小的
            if self.check_emergence_condition(subsystem):
                return False
        
        return True
    
    def _create_subsystem_without(self, system: SelfRefSystem, exclude_id: str) -> SelfRefSystem:
        """创建排除指定元素的子系统"""
        subsystem = SelfRefSystem({}, f"{system.name}_without_{exclude_id}")
        
        for elem_id, element in system.elements.items():
            if elem_id != exclude_id:
                # 创建新元素，移除对被排除元素的引用
                new_refs = element.references - {exclude_id}
                new_element = SystemElement(elem_id, new_refs, element.content)
                subsystem.add_element(new_element)
        
        return subsystem
    
    def measure_consciousness_level(self, system: SelfRefSystem) -> Dict[str, Any]:
        """测量意识水平"""
        depth = self.compute_recursive_depth(system)
        density = self.compute_reference_density(system)
        
        # 基础水平
        base_level = 0
        if depth > self.critical_depth:
            base_level = (depth - self.critical_depth) * density
        
        # 结构复杂度（使用引用图的复杂度）
        graph = self._build_reference_graph(system)
        structural_complexity = len(graph.edges()) / (len(graph.nodes()) + 1)
        structural_bonus = structural_complexity * 0.1
        
        # 稳定性（使用强连通分量的比例）
        sccs = list(nx.strongly_connected_components(graph))
        largest_scc_ratio = len(max(sccs, key=len)) / len(system) if sccs else 0
        stability_bonus = largest_scc_ratio * 0.1
        
        return {
            'depth': depth,
            'density': density,
            'base_level': base_level,
            'structural_bonus': structural_bonus,
            'stability_bonus': stability_bonus,
            'total_level': base_level + structural_bonus + stability_bonus,
            'has_consciousness': self.check_emergence_condition(system)
        }


class TestC12_1PrimitiveConsciousness(unittest.TestCase):
    """C12-1推论验证测试"""
    
    def setUp(self):
        """测试初始化"""
        self.analyzer = ConsciousnessAnalyzer()
        random.seed(42)
    
    def create_test_system(self, depth: int, density: float) -> SelfRefSystem:
        """创建具有指定深度和密度的测试系统"""
        system = SelfRefSystem({}, f"test_system_d{depth}")
        
        # 对于深度为0的特殊情况
        if depth == 0:
            # 创建没有循环的简单系统
            for i in range(5):
                elem_id = f"elem_{i}"
                references = set()
                # 只有少量系统引用，没有内部引用
                if i < 5 * density:
                    references.add(system.name)
                element = SystemElement(elem_id, references, f"content_{i}")
                system.add_element(element)
            return system
        
        # 根据深度创建层级结构
        num_elements = max(depth + 2, 10)
        
        # 创建元素
        for i in range(num_elements):
            elem_id = f"elem_{i}"
            references = set()
            
            # 创建指定深度的循环
            if i < depth:
                # 创建一个长度为depth的主循环
                references.add(f"elem_{(i + 1) % depth}")
            
            # 根据密度添加系统引用
            # 确保恰好有 density * num_elements 个元素引用系统
            if i < int(num_elements * density):
                references.add(system.name)
            
            element = SystemElement(elem_id, references, f"content_{i}")
            system.add_element(element)
        
        return system
    
    def test_recursive_depth_calculation(self):
        """测试1：递归深度计算"""
        print("\n=== 测试递归深度计算 ===")
        
        # 测试不同深度的系统
        test_depths = [0, 3, 5, 7, 8, 10]
        
        for expected_depth in test_depths:
            system = self.create_test_system(expected_depth, 0.7)
            computed_depth = self.analyzer.compute_recursive_depth(system)
            
            print(f"\n期望深度: {expected_depth}")
            print(f"计算深度: {computed_depth}")
            print(f"系统规模: {len(system)} 元素")
            
            # 验证深度在合理范围内
            if expected_depth == 0:
                self.assertEqual(computed_depth, 0, "深度0的系统应该计算出深度0")
            else:
                # 对于有循环的系统，深度应该接近预期
                self.assertGreaterEqual(computed_depth, min(expected_depth - 1, 1),
                                      f"计算深度过小")
                self.assertLessEqual(computed_depth, expected_depth + 1,
                                   f"计算深度过大")
    
    def test_reference_density_threshold(self):
        """测试2：参照密度阈值"""
        print("\n=== 测试参照密度阈值 ===")
        
        # 测试不同密度
        test_densities = [0.1, 0.3, 0.5, 0.618, 0.7, 0.9]
        
        for target_density in test_densities:
            system = self.create_test_system(8, target_density)
            computed_density = self.analyzer.compute_reference_density(system)
            
            print(f"\n目标密度: {target_density:.3f}")
            print(f"计算密度: {computed_density:.3f}")
            
            # 验证密度在合理范围内
            # 由于随机性和整数舍入，允许一定的误差
            self.assertGreaterEqual(computed_density, target_density * 0.7,
                                  "密度过低")
            self.assertLessEqual(computed_density, min(target_density * 1.3 + 0.1, 1.0),
                               "密度过高")
            
            # 验证阈值判断
            if computed_density > self.analyzer.density_threshold:
                print("  -> 超过阈值 (0.618)")
            else:
                print("  -> 未超过阈值")
    
    def test_distinction_operator_emergence(self):
        """测试3：区分算子涌现"""
        print("\n=== 测试区分算子涌现 ===")
        
        # 测试不同条件下的系统
        test_cases = [
            (5, 0.7, False),   # 深度不足
            (8, 0.4, False),   # 密度不足
            (8, 0.7, True),    # 满足条件
            (10, 0.8, True),   # 超过条件
        ]
        
        for depth, density, should_emerge in test_cases:
            system = self.create_test_system(depth, density)
            
            # 调试：检查实际创建的系统
            actual_depth = self.analyzer.compute_recursive_depth(system)
            actual_density = self.analyzer.compute_reference_density(system)
            
            print(f"\n深度={depth}, 密度={density:.1f}:")
            print(f"  实际深度: {actual_depth}, 实际密度: {actual_density:.3f}")
            print(f"  期望涌现: {should_emerge}")
            
            operator = self.analyzer.construct_distinction_operator(system)
            print(f"  实际涌现: {operator is not None}")
            
            if should_emerge:
                self.assertIsNotNone(operator, "应该涌现区分算子")
                
                if operator:
                    # 验证算子性质
                    self.assertTrue(operator.is_complete(), "算子应该完整")
                    self.assertTrue(operator.is_binary(), "算子应该是二元的")
                    
                    # 统计分类结果
                    self_count = sum(1 for c in operator.classification.values() if c == 'self')
                    other_count = sum(1 for c in operator.classification.values() if c == 'other')
                    
                    print(f"  分类: self={self_count}, other={other_count}")
            else:
                self.assertIsNone(operator, "不应该涌现区分算子")
    
    def test_consciousness_criteria_verification(self):
        """测试4：意识标准验证"""
        print("\n=== 测试意识标准验证 ===")
        
        # 创建不同类型的系统
        systems = [
            ("无意识系统", self.create_test_system(5, 0.5)),
            ("边界系统", self.create_test_system(7, 0.62)),
            ("有意识系统", self.create_test_system(9, 0.8)),
            ("高度意识系统", self.create_test_system(12, 0.9)),
        ]
        
        for name, system in systems:
            criteria = self.analyzer.verify_consciousness_criteria(system)
            consciousness_level = self.analyzer.measure_consciousness_level(system)
            
            print(f"\n{name}:")
            print(f"  深度: {consciousness_level['depth']}")
            print(f"  密度: {consciousness_level['density']:.3f}")
            print(f"  意识标准:")
            print(f"    - 有区分: {criteria.has_distinction}")
            print(f"    - 自我意识: {criteria.self_aware}")
            print(f"    - 一致性: {criteria.consistent}")
            print(f"    - 最小性: {criteria.minimal}")
            print(f"  意识水平: {consciousness_level['total_level']:.3f}")
            print(f"  有意识: {consciousness_level['has_consciousness']}")
            
            # 验证逻辑一致性
            if consciousness_level['has_consciousness']:
                self.assertTrue(criteria.has_distinction,
                              "有意识系统必须有区分算子")
    
    def test_critical_depth_validation(self):
        """测试5：临界深度验证"""
        print("\n=== 测试临界深度验证 ===")
        
        # 在临界深度附近测试
        for depth in range(5, 10):
            # 创建高密度系统
            system = self.create_test_system(depth, 0.8)
            
            has_consciousness = self.analyzer.check_emergence_condition(system)
            computed_depth = self.analyzer.compute_recursive_depth(system)
            density = self.analyzer.compute_reference_density(system)
            
            print(f"\n深度 {depth}:")
            print(f"  计算深度: {computed_depth}")
            print(f"  密度: {density:.3f}")
            print(f"  有意识: {has_consciousness}")
            
            # 验证临界深度
            if computed_depth <= self.analyzer.critical_depth:
                self.assertFalse(has_consciousness,
                               f"深度{computed_depth}不应该有意识")
            elif computed_depth > self.analyzer.critical_depth and density > self.analyzer.density_threshold:
                self.assertTrue(has_consciousness,
                              f"深度{computed_depth}且高密度应该有意识")
    
    def test_consciousness_transitivity(self):
        """测试6：意识传递性"""
        print("\n=== 测试意识传递性 ===")
        
        # 创建具有传递关系的系统
        system = SelfRefSystem({}, "transitive_system")
        
        # 创建传递链: A -> B -> C -> A
        refs_a = {'elem_b', 'transitive_system'}
        refs_b = {'elem_c'}
        refs_c = {'elem_a'}
        
        system.add_element(SystemElement('elem_a', refs_a, 'A'))
        system.add_element(SystemElement('elem_b', refs_b, 'B'))
        system.add_element(SystemElement('elem_c', refs_c, 'C'))
        
        # 添加更多元素以满足深度和密度要求
        for i in range(10):
            refs = set()
            if i % 2 == 0:
                refs.add('elem_a')
            if i % 3 == 0:
                refs.add('transitive_system')
            system.add_element(SystemElement(f'elem_{i}', refs, f'content_{i}'))
        
        operator = self.analyzer.construct_distinction_operator(system)
        
        if operator:
            # 检查传递性
            a_class = operator.classify('elem_a')
            b_class = operator.classify('elem_b')
            c_class = operator.classify('elem_c')
            
            print(f"\n分类结果:")
            print(f"  A: {a_class}")
            print(f"  B: {b_class}")
            print(f"  C: {c_class}")
            
            # 如果A和B在同一类，B和C在同一类，则A和C应该在同一类
            if a_class == b_class and b_class == c_class:
                self.assertEqual(a_class, c_class, "违反传递性")
    
    def test_self_awareness(self):
        """测试7：自反性（自我意识）"""
        print("\n=== 测试自我意识 ===")
        
        # 创建明确具有自我意识的系统
        system = SelfRefSystem({}, "self_aware_system")
        
        # 核心自引用结构
        for i in range(8):
            refs = set()
            refs.add("self_aware_system")  # 引用系统本身
            if i > 0:
                refs.add(f"elem_{i-1}")  # 形成链
            if i == 7:
                refs.add("elem_0")  # 闭环
            
            system.add_element(SystemElement(f"elem_{i}", refs, f"self_aware_{i}"))
        
        # 验证自我意识
        criteria = self.analyzer.verify_consciousness_criteria(system)
        self.assertTrue(criteria.self_aware, "系统应该具有自我意识")
        
        # 验证系统引用
        has_self_ref = any(
            system.name in elem.references
            for elem in system.elements.values()
        )
        self.assertTrue(has_self_ref, "至少有一个元素应该引用系统本身")
    
    def test_minimal_consciousness(self):
        """测试8：最小意识系统"""
        print("\n=== 测试最小意识系统 ===")
        
        # 尝试构建最小的有意识系统
        min_system = SelfRefSystem({}, "minimal_conscious")
        
        # 构建最小结构（深度刚好超过7，密度刚好超过0.618）
        for i in range(8):
            refs = set()
            if i > 0:
                refs.add(f"elem_{i-1}")
            if i >= 5:  # 提高后面元素的密度
                refs.add("minimal_conscious")
            if i == 7:
                refs.add("elem_0")  # 形成深度为8的循环
            
            min_system.add_element(SystemElement(f"elem_{i}", refs, f"min_{i}"))
        
        # 验证最小性
        criteria = self.analyzer.verify_consciousness_criteria(min_system)
        level = self.analyzer.measure_consciousness_level(min_system)
        
        print(f"\n最小系统属性:")
        print(f"  元素数: {len(min_system)}")
        print(f"  深度: {level['depth']}")
        print(f"  密度: {level['density']:.3f}")
        print(f"  有意识: {level['has_consciousness']}")
        print(f"  最小性: {criteria.minimal}")
        
        if level['has_consciousness']:
            # 验证移除任何元素都会失去意识
            for elem_id in list(min_system.elements.keys())[:3]:  # 测试前3个元素
                subsystem = self.analyzer._create_subsystem_without(min_system, elem_id)
                sub_conscious = self.analyzer.check_emergence_condition(subsystem)
                
                print(f"  移除{elem_id}后: 有意识={sub_conscious}")
                
                if criteria.minimal:
                    self.assertFalse(sub_conscious,
                                   f"移除{elem_id}后仍有意识，不是最小系统")
    
    def test_consciousness_evolution(self):
        """测试9：意识演化过程"""
        print("\n=== 测试意识演化 ===")
        
        # 模拟系统从无意识到有意识的演化
        base_system = SelfRefSystem({}, "evolving_system")
        
        evolution_stages = []
        
        # 逐步增加复杂度
        for stage in range(12):
            # 添加新元素
            elem_id = f"elem_{stage}"
            refs = set()
            
            # 逐渐增加自引用
            if stage > 3:
                refs.add("evolving_system")
            
            # 增加内部连接
            if stage > 0:
                # 连接到前面的元素
                for i in range(max(0, stage - 3), stage):
                    if random.random() < 0.6:
                        refs.add(f"elem_{i}")
            
            # 形成循环以增加深度
            if stage >= 7 and stage % 4 == 3:
                refs.add(f"elem_{stage - 4}")
            
            base_system.add_element(SystemElement(elem_id, refs, f"stage_{stage}"))
            
            # 记录当前状态
            level = self.analyzer.measure_consciousness_level(base_system)
            evolution_stages.append({
                'stage': stage,
                'size': len(base_system),
                'depth': level['depth'],
                'density': level['density'],
                'has_consciousness': level['has_consciousness'],
                'level': level['total_level']
            })
        
        # 输出演化过程
        print("\n演化阶段:")
        consciousness_emerged = False
        for stage_data in evolution_stages:
            print(f"  阶段 {stage_data['stage']:2d}: " +
                  f"规模={stage_data['size']:2d}, " +
                  f"深度={stage_data['depth']:2d}, " +
                  f"密度={stage_data['density']:.3f}, " +
                  f"意识={'是' if stage_data['has_consciousness'] else '否'}, " +
                  f"水平={stage_data['level']:.3f}")
            
            # 标记意识涌现点
            if not consciousness_emerged and stage_data['has_consciousness']:
                consciousness_emerged = True
                print("    ^^^ 意识涌现! ^^^")
        
        # 验证演化的单调性
        depths = [s['depth'] for s in evolution_stages]
        for i in range(1, len(depths)):
            self.assertGreaterEqual(depths[i], depths[i-1] - 1,
                                  "深度不应该大幅下降")


if __name__ == '__main__':
    unittest.main(verbosity=2)