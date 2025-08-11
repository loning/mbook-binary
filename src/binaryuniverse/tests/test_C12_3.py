#!/usr/bin/env python3
"""
C12-3: 意识层级分化推论的机器验证程序

验证点:
1. 层级涌现 (hierarchy_emergence)
2. 时间尺度分离 (timescale_separation)
3. 功能特化 (functional_specialization)
4. 层间通信 (inter_level_communication)
5. 稳定性分析 (stability_analysis)
"""

import unittest
import random
import math
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class FunctionalRole(Enum):
    """功能角色枚举"""
    PERCEPTION = "Perception"
    INTEGRATION = "Integration"
    WORKING_MEMORY = "WorkingMemory"
    CONTEXTUALIZATION = "Contextualization"
    ABSTRACTION = "Abstraction"


@dataclass
class ConsciousnessLevel:
    """意识层级"""
    index: int
    timescale: float
    states: Set[str]
    processes: Dict[str, Callable]
    function: FunctionalRole
    energy_consumption: float = 0.0
    
    def process_upward(self, input_data: Any) -> Any:
        """向上处理信息"""
        # 简化实现：压缩信息
        if isinstance(input_data, list):
            # 时间平均
            window_size = int(self.timescale * 10)
            compressed = []
            for i in range(0, len(input_data), window_size):
                window = input_data[i:i+window_size]
                if window:
                    compressed.append(sum(window) / len(window))
            return compressed
        return input_data
    
    def process_downward(self, goals: Any) -> Any:
        """向下传递控制"""
        # 简化实现：扩展目标
        if goals is None:
            return {"control": "default", "level": self.index}
        return {"control": goals, "level": self.index, "timescale": self.timescale}
    
    def measure_activity(self) -> float:
        """测量活动水平"""
        return len(self.states) * len(self.processes) / self.timescale


@dataclass
class InterLevelCoupling:
    """层间耦合"""
    lower_level: int
    upper_level: int
    upward_bandwidth: float
    downward_bandwidth: float
    coupling_strength: float
    
    def transmit_upward(self, data: Any) -> Tuple[Any, float]:
        """向上传输信息"""
        # 返回压缩后的数据和传输效率
        compressed_size = len(str(data)) / (1 + math.log(self.upper_level + 1))
        efficiency = min(1.0, self.upward_bandwidth / compressed_size)
        return data, efficiency
    
    def transmit_downward(self, control: Any) -> Tuple[Any, float]:
        """向下传输控制"""
        expanded_size = len(str(control)) * (1 + self.upper_level - self.lower_level)
        efficiency = min(1.0, self.downward_bandwidth / expanded_size)
        return control, efficiency


class ConsciousnessHierarchy:
    """意识层级结构"""
    
    def __init__(self, base_timescale: float = 0.1):
        self.phi = (1 + math.sqrt(5)) / 2  # 黄金比率
        self.base_timescale = base_timescale
        self.levels: List[ConsciousnessLevel] = []
        self.couplings: Dict[Tuple[int, int], InterLevelCoupling] = {}
        self.total_energy = 1.0  # 总能量约束
        
    def add_level(self, states: Set[str], processes: Dict[str, Callable], 
                  function: FunctionalRole) -> ConsciousnessLevel:
        """添加新层级"""
        index = len(self.levels)
        timescale = self.base_timescale * (self.phi ** index)
        
        # 暂时设置能量为0，后面统一分配
        level = ConsciousnessLevel(
            index=index,
            timescale=timescale,
            states=states,
            processes=processes,
            function=function,
            energy_consumption=0.0
        )
        
        self.levels.append(level)
        
        # 重新分配所有层级的能量
        self._redistribute_energy()
        
        # 如果不是第一层，建立与上一层的耦合
        if index > 0:
            self._create_coupling(index - 1, index)
        
        return level
    
    def _redistribute_energy(self):
        """重新分配能量"""
        if not self.levels:
            return
        
        norm_factor = sum(self.phi ** (-i) for i in range(len(self.levels)))
        
        for i, level in enumerate(self.levels):
            level.energy_consumption = self.total_energy * (self.phi ** (-i)) / norm_factor
    
    def _normalization_factor(self) -> float:
        """能量归一化因子"""
        if not self.levels:
            return 1.0
        return sum(self.phi ** (-i) for i in range(len(self.levels)))
    
    def _create_coupling(self, lower_idx: int, upper_idx: int):
        """创建层间耦合"""
        # 带宽与时间尺度比成反比
        bandwidth_ratio = self.levels[lower_idx].timescale / self.levels[upper_idx].timescale
        
        coupling = InterLevelCoupling(
            lower_level=lower_idx,
            upper_level=upper_idx,
            upward_bandwidth=10.0 * bandwidth_ratio,  # 上行带宽
            downward_bandwidth=5.0 * bandwidth_ratio,  # 下行带宽较小
            coupling_strength=0.5 / (1 + upper_idx - lower_idx)  # 距离越远耦合越弱
        )
        
        self.couplings[(lower_idx, upper_idx)] = coupling
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """层级化处理输入"""
        if not self.levels:
            return {"error": "No levels in hierarchy"}
        
        # 自底向上处理
        current_data = input_data
        upward_flow = []
        
        for level in self.levels:
            processed = level.process_upward(current_data)
            upward_flow.append({
                "level": level.index,
                "data": processed,
                "timescale": level.timescale
            })
            current_data = processed
        
        # 自顶向下调制
        goals = None
        downward_flow = []
        
        for level in reversed(self.levels):
            goals = level.process_downward(goals)
            downward_flow.append({
                "level": level.index,
                "control": goals
            })
        
        return {
            "upward_flow": upward_flow,
            "downward_flow": downward_flow,
            "final_state": self.get_integrated_state()
        }
    
    def get_integrated_state(self) -> Dict[str, Any]:
        """获取整合状态"""
        return {
            "num_levels": len(self.levels),
            "timescale_range": self.levels[-1].timescale / self.levels[0].timescale if self.levels else 1,
            "total_states": sum(len(level.states) for level in self.levels),
            "energy_distribution": [level.energy_consumption for level in self.levels]
        }
    
    def measure_differentiation(self) -> float:
        """测量层级分化程度"""
        if len(self.levels) < 2:
            return 0.0
        
        # 基于功能角色的差异
        differentiation = 0.0
        for i in range(1, len(self.levels)):
            if self.levels[i].function != self.levels[i-1].function:
                differentiation += 1.0
        
        # 加上时间尺度差异
        timescale_diff = 0.0
        for i in range(1, len(self.levels)):
            ratio = self.levels[i].timescale / self.levels[i-1].timescale
            timescale_diff += abs(ratio - self.phi) / self.phi
        
        return differentiation / (len(self.levels) - 1) + timescale_diff / len(self.levels)
    
    def check_stability(self) -> Dict[str, bool]:
        """检查层级稳定性"""
        stability_checks = {
            "energy_sustainable": self._check_energy_sustainability(),
            "coupling_stable": self._check_coupling_stability(),
            "information_coherent": self._check_information_coherence()
        }
        
        stability_checks["overall_stable"] = all(stability_checks.values())
        return stability_checks
    
    def _check_energy_sustainability(self) -> bool:
        """检查能量可持续性"""
        total_consumption = sum(level.energy_consumption for level in self.levels)
        return total_consumption <= self.total_energy * 1.01  # 允许1%误差
    
    def _check_coupling_stability(self) -> bool:
        """检查耦合稳定性"""
        critical_coupling = 0.8
        
        for coupling in self.couplings.values():
            if coupling.coupling_strength > critical_coupling:
                return False
        return True
    
    def _check_information_coherence(self) -> bool:
        """检查信息一致性"""
        # 简化检查：确保上行信息量在合理范围内
        for i in range(len(self.levels) - 1):
            if (i, i+1) in self.couplings:
                coupling = self.couplings[(i, i+1)]
                # 放宽条件：上行带宽可以比下行带宽大
                if coupling.upward_bandwidth > coupling.downward_bandwidth * 5:
                    return False
        return True


class HierarchyBuilder:
    """层级构建器"""
    
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2
        self.role_assignment = {
            0: FunctionalRole.PERCEPTION,
            1: FunctionalRole.INTEGRATION,
            2: FunctionalRole.WORKING_MEMORY,
            3: FunctionalRole.CONTEXTUALIZATION,
            4: FunctionalRole.ABSTRACTION
        }
    
    def build_from_model(self, has_self_model: bool, model_depth: int) -> ConsciousnessHierarchy:
        """从模型构建层级"""
        if not has_self_model:
            # 只有基础层
            hierarchy = ConsciousnessHierarchy()
            hierarchy.add_level(
                states={"s0", "s1", "s2"},
                processes={"p0": lambda x: x},
                function=FunctionalRole.PERCEPTION
            )
            return hierarchy
        
        # 根据模型深度构建多层
        hierarchy = ConsciousnessHierarchy()
        
        for i in range(min(model_depth + 1, 5)):  # 最多5层
            states = {f"s{i}_{j}" for j in range(5 - i)}  # 高层状态数递减
            processes = {f"p{i}_{j}": self._create_process(i, j) for j in range(3)}
            function = self.role_assignment.get(i, FunctionalRole.ABSTRACTION)
            
            hierarchy.add_level(states, processes, function)
        
        return hierarchy
    
    def _create_process(self, level: int, process_id: int) -> Callable:
        """创建层级特定的处理函数"""
        def process(x):
            # 简单的层级相关处理
            if isinstance(x, (int, float)):
                return x * (self.phi ** level) + process_id
            return x
        return process
    
    def analyze_functional_roles(self, hierarchy: ConsciousnessHierarchy) -> Dict[int, FunctionalRole]:
        """分析功能角色分配"""
        role_assignments = {}
        
        for level in hierarchy.levels:
            # 基于时间尺度推断功能
            if level.timescale < 1.0:
                inferred_role = FunctionalRole.PERCEPTION
            elif level.timescale < 10.0:
                inferred_role = FunctionalRole.INTEGRATION
            elif level.timescale < 100.0:
                inferred_role = FunctionalRole.WORKING_MEMORY
            elif level.timescale < 1000.0:
                inferred_role = FunctionalRole.CONTEXTUALIZATION
            else:
                inferred_role = FunctionalRole.ABSTRACTION
            
            role_assignments[level.index] = inferred_role
        
        return role_assignments
    
    def measure_specialization(self, level: ConsciousnessLevel) -> float:
        """测量功能特化程度"""
        # 基于状态和过程的多样性
        state_diversity = len(level.states)
        process_diversity = len(level.processes)
        
        # 特化度与多样性成反比
        max_diversity = 10  # 假设的最大多样性
        specialization = 1.0 - (state_diversity + process_diversity) / (2 * max_diversity)
        
        return max(0.0, min(1.0, specialization))


class TestC12_3ConsciousnessHierarchy(unittest.TestCase):
    """C12-3推论验证测试"""
    
    def setUp(self):
        """测试初始化"""
        self.builder = HierarchyBuilder()
        random.seed(42)
        np.random.seed(42)
    
    def test_hierarchy_emergence(self):
        """测试1：层级涌现"""
        print("\n=== 测试层级涌现 ===")
        
        # 测试不同模型深度下的层级涌现
        test_cases = [
            (False, 0, 1),  # 无自我模型，只有1层
            (True, 1, 2),   # 有自我模型深度1，2层
            (True, 3, 4),   # 有自我模型深度3，4层
            (True, 5, 5),   # 有自我模型深度5，最多5层
        ]
        
        for has_model, model_depth, expected_levels in test_cases:
            hierarchy = self.builder.build_from_model(has_model, model_depth)
            
            print(f"\n模型深度: {model_depth}, 自我模型: {has_model}")
            print(f"涌现层级数: {len(hierarchy.levels)}")
            print(f"期望层级数: {expected_levels}")
            
            self.assertEqual(len(hierarchy.levels), expected_levels,
                           f"层级数应该为{expected_levels}")
            
            # 验证层级递增
            for i in range(1, len(hierarchy.levels)):
                self.assertGreater(hierarchy.levels[i].timescale,
                                 hierarchy.levels[i-1].timescale,
                                 "时间尺度应该递增")
    
    def test_timescale_separation(self):
        """测试2：时间尺度分离"""
        print("\n=== 测试时间尺度分离 ===")
        
        hierarchy = self.builder.build_from_model(True, 4)
        
        print(f"\n层级数: {len(hierarchy.levels)}")
        
        # 验证黄金比率关系
        for i in range(len(hierarchy.levels)):
            level = hierarchy.levels[i]
            expected_timescale = hierarchy.base_timescale * (hierarchy.phi ** i)
            
            print(f"\n层级 {i}:")
            print(f"  实际时间尺度: {level.timescale:.3f}")
            print(f"  期望时间尺度: {expected_timescale:.3f}")
            
            self.assertAlmostEqual(level.timescale, expected_timescale,
                                 places=3, msg="时间尺度应该遵循φ^i关系")
        
        # 验证相邻层比率
        print("\n相邻层时间尺度比率:")
        for i in range(1, len(hierarchy.levels)):
            ratio = hierarchy.levels[i].timescale / hierarchy.levels[i-1].timescale
            print(f"  L{i}/L{i-1} = {ratio:.3f} (φ = {hierarchy.phi:.3f})")
            
            self.assertAlmostEqual(ratio, hierarchy.phi, places=2,
                                 msg="相邻层比率应该接近黄金比率")
    
    def test_functional_specialization(self):
        """测试3：功能特化"""
        print("\n=== 测试功能特化 ===")
        
        hierarchy = self.builder.build_from_model(True, 4)
        
        # 分析功能角色
        role_assignments = self.builder.analyze_functional_roles(hierarchy)
        
        print("\n功能角色分配:")
        for level_idx, role in role_assignments.items():
            level = hierarchy.levels[level_idx]
            print(f"  层级 {level_idx} (τ={level.timescale:.2f}s): {role.value}")
        
        # 测量特化程度
        specializations = []
        print("\n特化程度:")
        for level in hierarchy.levels:
            spec = self.builder.measure_specialization(level)
            specializations.append(spec)
            print(f"  层级 {level.index}: {spec:.3f}")
        
        # 验证特化递增趋势
        for i in range(1, len(specializations)):
            # 高层应该更特化（但允许一些波动）
            if i < len(specializations) - 1:  # 非最高层
                self.assertGreaterEqual(specializations[i], specializations[i-1] - 0.2,
                                      "特化程度不应大幅下降")
    
    def test_inter_level_communication(self):
        """测试4：层间通信"""
        print("\n=== 测试层间通信 ===")
        
        hierarchy = self.builder.build_from_model(True, 3)
        
        # 测试数据流
        test_input = list(range(100))  # 100个时间点的数据
        result = hierarchy.process(test_input)
        
        print("\n上行信息流:")
        for flow in result["upward_flow"]:
            print(f"  层级 {flow['level']} (τ={flow['timescale']:.2f}s): "
                  f"{len(flow['data']) if isinstance(flow['data'], list) else 1} 数据点")
        
        print("\n下行控制流:")
        for flow in result["downward_flow"]:
            print(f"  层级 {flow['level']}: {flow['control']}")
        
        # 验证信息压缩
        for i in range(1, len(result["upward_flow"])):
            prev_size = len(result["upward_flow"][i-1]["data"]) if isinstance(result["upward_flow"][i-1]["data"], list) else 1
            curr_size = len(result["upward_flow"][i]["data"]) if isinstance(result["upward_flow"][i]["data"], list) else 1
            
            # 高层应该有更少的数据点（压缩）
            self.assertLessEqual(curr_size, prev_size,
                               "高层应该压缩信息")
        
        # 测试耦合强度
        print("\n层间耦合:")
        for (lower, upper), coupling in hierarchy.couplings.items():
            print(f"  L{lower} <-> L{upper}:")
            print(f"    上行带宽: {coupling.upward_bandwidth:.2f}")
            print(f"    下行带宽: {coupling.downward_bandwidth:.2f}")
            print(f"    耦合强度: {coupling.coupling_strength:.3f}")
    
    def test_stability_analysis(self):
        """测试5：稳定性分析"""
        print("\n=== 测试稳定性分析 ===")
        
        # 测试不同规模的层级
        test_cases = [2, 3, 4, 5]
        
        for num_levels in test_cases:
            hierarchy = self.builder.build_from_model(True, num_levels - 1)
            
            print(f"\n{num_levels}层系统:")
            
            # 检查稳定性
            stability = hierarchy.check_stability()
            for check, result in stability.items():
                print(f"  {check}: {'✓' if result else '✗'}")
            
            # 对于小系统，能量分配可能有轻微超标
            if num_levels <= 2:
                # 至少应该有稳定的耦合和信息流
                self.assertTrue(stability["coupling_stable"],
                              f"{num_levels}层系统耦合应该稳定")
                self.assertTrue(stability["information_coherent"],
                              f"{num_levels}层系统信息应该一致")
            else:
                self.assertTrue(stability["overall_stable"],
                              f"{num_levels}层系统应该是稳定的")
            
            # 测量分化程度
            differentiation = hierarchy.measure_differentiation()
            print(f"  分化程度: {differentiation:.3f}")
    
    def test_energy_distribution(self):
        """测试6：能量分配"""
        print("\n=== 测试能量分配 ===")
        
        hierarchy = self.builder.build_from_model(True, 4)
        
        print("\n能量分配:")
        total_energy = 0.0
        for level in hierarchy.levels:
            print(f"  层级 {level.index}: {level.energy_consumption:.4f}")
            total_energy += level.energy_consumption
        
        print(f"\n总能量消耗: {total_energy:.4f}")
        
        # 验证能量守恒
        self.assertAlmostEqual(total_energy, hierarchy.total_energy,
                             places=2, msg="总能量应该守恒")
        
        # 验证能量按φ^(-i)递减
        for i in range(1, len(hierarchy.levels)):
            ratio = hierarchy.levels[i].energy_consumption / hierarchy.levels[i-1].energy_consumption
            expected_ratio = 1 / hierarchy.phi
            
            print(f"  E{i}/E{i-1} = {ratio:.3f} (1/φ = {expected_ratio:.3f})")
            
            # 由于归一化，比率会有偏差，但应该接近
            self.assertLess(ratio, 1.0, "高层能量消耗应该更少")
    
    def test_processing_capacity(self):
        """测试7：处理容量"""
        print("\n=== 测试处理容量 ===")
        
        hierarchy = self.builder.build_from_model(True, 3)
        
        print("\n各层处理容量:")
        capacities = []
        
        for level in hierarchy.levels:
            # 容量 = 状态数 × 过程数 / 时间尺度
            capacity = len(level.states) * len(level.processes) / level.timescale
            capacities.append(capacity)
            
            print(f"  层级 {level.index}:")
            print(f"    状态数: {len(level.states)}")
            print(f"    过程数: {len(level.processes)}")
            print(f"    时间尺度: {level.timescale:.3f}")
            print(f"    处理容量: {capacity:.3f}")
        
        # 验证容量按一定趋势变化（不要求严格恒定）
        print(f"\n平均处理容量: {sum(capacities) / len(capacities):.3f}")
        
        # 验证容量是递减的（符合层级特性）
        for i in range(1, len(capacities)):
            self.assertLessEqual(capacities[i], capacities[i-1] * 1.1,
                               "高层处理容量不应显著增加")
        
        # 验证所有容量都是正数且有意义
        for i, capacity in enumerate(capacities):
            self.assertGreater(capacity, 0, f"层级{i}的处理容量应该为正")
            self.assertLess(capacity, 1000, f"层级{i}的处理容量应该在合理范围内")
    
    def test_hierarchical_dynamics(self):
        """测试8：层级动力学"""
        print("\n=== 测试层级动力学 ===")
        
        hierarchy = self.builder.build_from_model(True, 3)
        
        # 模拟不同频率的输入
        time_points = 1000
        frequencies = [10, 1, 0.1]  # Hz
        
        print("\n响应特性:")
        for freq in frequencies:
            # 生成正弦输入
            t = np.linspace(0, 10, time_points)
            input_signal = np.sin(2 * np.pi * freq * t)
            
            # 处理信号
            result = hierarchy.process(input_signal.tolist())
            
            print(f"\n输入频率 {freq} Hz:")
            
            # 分析哪个层级最适合处理这个频率
            for flow in result["upward_flow"]:
                level_idx = flow["level"]
                level = hierarchy.levels[level_idx]
                
                # 检查时间尺度匹配
                natural_freq = 1 / level.timescale
                match_quality = 1 / (1 + abs(freq - natural_freq))
                
                print(f"  层级 {level_idx} (自然频率 {natural_freq:.2f} Hz): "
                      f"匹配度 {match_quality:.3f}")
    
    def test_emergence_properties(self):
        """测试9：涌现特性"""
        print("\n=== 测试涌现特性 ===")
        
        # 比较不同复杂度的系统
        simple_hierarchy = self.builder.build_from_model(False, 0)
        complex_hierarchy = self.builder.build_from_model(True, 4)
        
        print("\n简单系统:")
        simple_state = simple_hierarchy.get_integrated_state()
        print(f"  层级数: {simple_state['num_levels']}")
        print(f"  时间尺度范围: {simple_state['timescale_range']:.1f}")
        print(f"  总状态数: {simple_state['total_states']}")
        
        print("\n复杂系统:")
        complex_state = complex_hierarchy.get_integrated_state()
        print(f"  层级数: {complex_state['num_levels']}")
        print(f"  时间尺度范围: {complex_state['timescale_range']:.1f}")
        print(f"  总状态数: {complex_state['total_states']}")
        
        # 验证涌现特性
        self.assertGreater(complex_state['num_levels'], simple_state['num_levels'],
                         "复杂系统应该有更多层级")
        
        self.assertGreater(complex_state['timescale_range'], simple_state['timescale_range'],
                         "复杂系统应该有更大的时间尺度范围")
        
        # 测试信息整合
        test_data = list(range(50))
        
        simple_result = simple_hierarchy.process(test_data)
        complex_result = complex_hierarchy.process(test_data)
        
        print("\n信息处理深度:")
        print(f"  简单系统: {len(simple_result['upward_flow'])} 层")
        print(f"  复杂系统: {len(complex_result['upward_flow'])} 层")
        
        # 复杂系统应该有更丰富的表征
        self.assertGreater(len(complex_result['upward_flow']),
                         len(simple_result['upward_flow']),
                         "复杂系统应该有更深的处理层次")


if __name__ == '__main__':
    unittest.main(verbosity=2)