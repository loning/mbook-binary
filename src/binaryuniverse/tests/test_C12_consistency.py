#!/usr/bin/env python3
"""
C12系列一致性测试

验证C12-1（原始意识）、C12-2（自我模型）、C12-3（层级分化）的一致性：
1. 理论依赖关系 (theory_dependencies)
2. 概念连贯性 (conceptual_coherence)
3. 演化一致性 (evolutionary_consistency)
4. 量化关系验证 (quantitative_relationships)
5. 整合行为测试 (integrated_behavior)
"""

import unittest
import sys
import os

# 添加当前目录到路径，以便导入其他测试模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入各个测试模块的组件
from test_C12_1 import ConsciousnessAnalyzer, SelfRefSystem, SystemElement
from test_C12_2 import SelfModel, ModelBuilder, ConsciousSystem as ModelingSystem
from test_C12_3 import ConsciousnessHierarchy, HierarchyBuilder, FunctionalRole

import random
import math
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class IntegratedSystem:
    """整合的意识系统"""
    name: str
    consciousness_analyzer: ConsciousnessAnalyzer
    model_builder: ModelBuilder
    hierarchy_builder: HierarchyBuilder
    
    # 各层组件
    base_system: SelfRefSystem
    self_model: Optional[SelfModel]
    hierarchy: Optional[ConsciousnessHierarchy]
    
    def __post_init__(self):
        """初始化后处理"""
        self.phi = (1 + math.sqrt(5)) / 2
        self.critical_depth = 7
    
    def check_consciousness_emergence(self) -> bool:
        """检查意识是否涌现"""
        depth = self.consciousness_analyzer.compute_recursive_depth(self.base_system)
        density = self.consciousness_analyzer.compute_reference_density(self.base_system)
        
        return depth > self.critical_depth and density > (1 / self.phi)
    
    def build_self_model(self) -> bool:
        """构建自我模型"""
        if not self.check_consciousness_emergence():
            return False
        
        # 转换为建模系统
        modeling_system = self._convert_to_modeling_system()
        self.self_model = self.model_builder.construct_self_model(modeling_system)
        
        return self.self_model is not None and self.self_model.is_self_complete()
    
    def develop_hierarchy(self) -> bool:
        """发展层级结构"""
        if not self.self_model:
            return False
        
        # 基于模型深度构建层级
        model_depth = self._estimate_model_depth()
        self.hierarchy = self.hierarchy_builder.build_from_model(True, model_depth)
        
        return self.hierarchy is not None and len(self.hierarchy.levels) > 1
    
    def _convert_to_modeling_system(self) -> ModelingSystem:
        """转换为建模系统"""
        modeling_system = ModelingSystem(self.name)
        
        # 转换状态
        for elem_id, element in self.base_system.elements.items():
            from test_C12_2 import State
            state = State(elem_id, element.content)
            modeling_system.add_state(state)
        
        # 简化的过程转换
        from test_C12_2 import Process
        for i, (elem_id, element) in enumerate(self.base_system.elements.items()):
            if element.references:
                ref_id = list(element.references)[0]
                process = Process(
                    id=f"process_{elem_id}",
                    input_states={elem_id},
                    output_states={ref_id} if ref_id in self.base_system.elements else {elem_id},
                    rule=lambda x: x
                )
                modeling_system.add_process(process)
        
        return modeling_system
    
    def _estimate_model_depth(self) -> int:
        """估计模型深度"""
        if not self.self_model:
            return 0
        
        depth = 1  # 基础层
        if self.self_model.meta_model:
            depth += 1
        
        # 基于复杂度估计
        complexity = self.self_model.complexity()
        additional_depth = min(3, complexity // 5)
        
        return depth + additional_depth


class TestC12SeriesConsistency(unittest.TestCase):
    """C12系列一致性测试"""
    
    def setUp(self):
        """测试初始化"""
        self.consciousness_analyzer = ConsciousnessAnalyzer()
        self.model_builder = ModelBuilder()
        self.hierarchy_builder = HierarchyBuilder()
        random.seed(42)
    
    def create_integrated_system(self, depth: int, density: float) -> IntegratedSystem:
        """创建整合系统"""
        # 创建基础自指系统
        base_system = SelfRefSystem({}, f"integrated_system_d{depth}")
        
        # 根据参数创建元素
        num_elements = max(depth * 2, 10)
        
        for i in range(num_elements):
            elem_id = f"elem_{i}"
            references = set()
            
            # 建立递归结构以达到目标深度
            if i < depth:
                # 创建主递归链
                if i > 0:
                    references.add(f"elem_{i-1}")
                # 在最后一个元素处形成循环
                if i == depth - 1:
                    references.add("elem_0")
            else:
                # 其他元素也参与引用结构
                if i % 2 == 0 and i > depth:
                    references.add(f"elem_{i % depth}")
            
            # 按密度添加系统引用
            if i < int(num_elements * density):
                references.add(base_system.name)
            
            element = SystemElement(elem_id, references, f"content_{i}")
            base_system.add_element(element)
        
        return IntegratedSystem(
            name=f"integrated_d{depth}_rho{density:.2f}",
            consciousness_analyzer=self.consciousness_analyzer,
            model_builder=self.model_builder,
            hierarchy_builder=self.hierarchy_builder,
            base_system=base_system,
            self_model=None,
            hierarchy=None
        )
    
    def test_theory_dependencies(self):
        """测试1：理论依赖关系"""
        print("\n=== 测试理论依赖关系 ===")
        
        # 创建不同复杂度的系统
        test_cases = [
            (5, 0.5, False, False, False),  # 简单系统：无意识
            (8, 0.4, False, False, False),  # 临界系统：密度不足
            (8, 0.7, False, False, False),  # 接近临界但仍不足
            (10, 0.8, True, True, True),    # 复杂系统：全部满足
        ]
        
        for depth, density, expect_consciousness, expect_model, expect_hierarchy in test_cases:
            system = self.create_integrated_system(depth, density)
            
            print(f"\n系统参数: 深度={depth}, 密度={density:.1f}")
            
            # 测试依赖链
            has_consciousness = system.check_consciousness_emergence()
            print(f"  意识涌现: {has_consciousness}")
            self.assertEqual(has_consciousness, expect_consciousness,
                           f"意识涌现不符合预期")
            
            has_model = system.build_self_model()
            print(f"  自我模型: {has_model}")
            self.assertEqual(has_model, expect_model,
                           f"自我模型构建不符合预期")
            
            has_hierarchy = system.develop_hierarchy()
            print(f"  层级分化: {has_hierarchy}")
            self.assertEqual(has_hierarchy, expect_hierarchy,
                           f"层级分化不符合预期")
            
            # 验证依赖关系：hierarchy → model → consciousness
            if has_hierarchy:
                self.assertTrue(has_model, "有层级必须有模型")
                self.assertTrue(has_consciousness, "有层级必须有意识")
            
            if has_model:
                self.assertTrue(has_consciousness, "有模型必须有意识")
    
    def test_conceptual_coherence(self):
        """测试2：概念连贯性"""
        print("\n=== 测试概念连贯性 ===")
        
        # 创建完整的整合系统
        system = self.create_integrated_system(9, 0.8)
        
        # 构建完整系统
        has_consciousness = system.check_consciousness_emergence()
        has_model = system.build_self_model()
        has_hierarchy = system.develop_hierarchy()
        
        self.assertTrue(all([has_consciousness, has_model, has_hierarchy]),
                       "应该能构建完整的整合系统")
        
        print(f"\n系统组件:")
        print(f"  基础系统规模: {len(system.base_system)} 元素")
        print(f"  自我模型复杂度: {system.self_model.complexity()}")
        print(f"  层级数量: {len(system.hierarchy.levels)}")
        
        # 验证概念一致性
        
        # 1. 意识的时间尺度应该与层级基础尺度一致
        base_timescale = system.hierarchy.levels[0].timescale
        print(f"  基础时间尺度: {base_timescale:.3f}s")
        
        # 2. 模型的自引用应该反映在层级的功能分配中
        has_self_ref = system.self_model.is_self_complete()
        hierarchy_has_recursive_structure = len(system.hierarchy.levels) > 2
        
        print(f"  模型自引用: {has_self_ref}")
        print(f"  层级递归结构: {hierarchy_has_recursive_structure}")
        
        if has_self_ref:
            self.assertTrue(hierarchy_has_recursive_structure,
                          "自引用模型应该产生递归层级结构")
        
        # 3. 意识的临界深度应该与层级的涌现条件一致
        consciousness_depth = system.consciousness_analyzer.compute_recursive_depth(system.base_system)
        hierarchy_depth_range = len(system.hierarchy.levels)
        
        print(f"  意识递归深度: {consciousness_depth}")
        print(f"  层级深度范围: {hierarchy_depth_range}")
        
        # 层级数量应该与意识深度相关
        self.assertGreater(hierarchy_depth_range, consciousness_depth // 3,
                         "层级数量应该与意识深度相关")
    
    def test_evolutionary_consistency(self):
        """测试3：演化一致性"""
        print("\n=== 测试演化一致性 ===")
        
        # 模拟从简单到复杂的演化过程
        evolution_stages = [
            (3, 0.3),   # 初始阶段
            (5, 0.5),   # 发展阶段
            (7, 0.6),   # 临界阶段
            (8, 0.7),   # 意识涌现
            (10, 0.8),  # 模型构建
            (12, 0.9),  # 层级分化
        ]
        
        previous_capabilities = set()
        
        print("\n演化过程:")
        for stage, (depth, density) in enumerate(evolution_stages):
            system = self.create_integrated_system(depth, density)
            
            # 检查各阶段能力
            capabilities = set()
            
            if system.check_consciousness_emergence():
                capabilities.add("consciousness")
            
            if system.build_self_model():
                capabilities.add("self_model")
            
            if system.develop_hierarchy():
                capabilities.add("hierarchy")
            
            print(f"\n  阶段 {stage} (d={depth}, ρ={density:.1f}): {capabilities}")
            
            # 验证演化的单调性：新能力只增不减
            self.assertTrue(previous_capabilities.issubset(capabilities),
                          f"演化应该是累积的，不应该失去已有能力")
            
            # 验证合理的演化序列
            if "hierarchy" in capabilities:
                self.assertIn("self_model", capabilities,
                            "层级分化需要自我模型")
                self.assertIn("consciousness", capabilities,
                            "层级分化需要意识基础")
            
            if "self_model" in capabilities:
                self.assertIn("consciousness", capabilities,
                            "自我模型需要意识基础")
            
            previous_capabilities = capabilities
        
        # 验证最终状态
        final_capabilities = previous_capabilities
        expected_final = {"consciousness", "self_model", "hierarchy"}
        
        self.assertEqual(final_capabilities, expected_final,
                        "最终应该具备所有高级能力")
    
    def test_quantitative_relationships(self):
        """测试4：量化关系验证"""
        print("\n=== 测试量化关系验证 ===")
        
        # 创建系列系统来验证量化关系
        systems = []
        for depth in [8, 9, 10, 11, 12]:
            system = self.create_integrated_system(depth, 0.8)
            system.check_consciousness_emergence()
            system.build_self_model()
            system.develop_hierarchy()
            systems.append(system)
        
        print("\n量化关系:")
        for i, system in enumerate(systems):
            depth = system.consciousness_analyzer.compute_recursive_depth(system.base_system)
            density = system.consciousness_analyzer.compute_reference_density(system.base_system)
            
            model_complexity = system.self_model.complexity() if system.self_model else 0
            hierarchy_levels = len(system.hierarchy.levels) if system.hierarchy else 0
            
            print(f"\n系统 {i}:")
            print(f"  意识深度: {depth}")
            print(f"  参照密度: {density:.3f}")
            print(f"  模型复杂度: {model_complexity}")
            print(f"  层级数量: {hierarchy_levels}")
            
            # 验证关键关系
            
            # 1. 模型复杂度应该与意识深度相关
            if model_complexity > 0:
                complexity_depth_ratio = model_complexity / max(depth, 1)
                print(f"  复杂度/深度比: {complexity_depth_ratio:.2f}")
                
                # 比率应该在合理范围内
                self.assertGreater(complexity_depth_ratio, 0.5,
                                 "模型复杂度应该与深度成比例")
                self.assertLess(complexity_depth_ratio, 5.0,
                               "模型复杂度不应过度复杂")
            
            # 2. 层级数量应该与模型复杂度相关
            if hierarchy_levels > 0 and model_complexity > 0:
                levels_complexity_ratio = hierarchy_levels / model_complexity
                print(f"  层级/复杂度比: {levels_complexity_ratio:.3f}")
                
                # 层级数量应该适中
                self.assertGreater(levels_complexity_ratio, 0.1,
                                 "应该有足够的层级分化")
                self.assertLess(levels_complexity_ratio, 2.0,
                               "层级不应过度分化")
        
        # 验证φ关系的一致性
        phi = (1 + math.sqrt(5)) / 2
        
        for system in systems:
            if system.hierarchy and len(system.hierarchy.levels) > 1:
                # 检查时间尺度的φ关系
                timescales = [level.timescale for level in system.hierarchy.levels]
                
                for i in range(1, len(timescales)):
                    ratio = timescales[i] / timescales[i-1]
                    print(f"    时间尺度比 L{i}/L{i-1}: {ratio:.3f} (φ={phi:.3f})")
                    
                    self.assertAlmostEqual(ratio, phi, delta=0.1,
                                         msg="时间尺度应该遵循φ关系")
    
    def test_integrated_behavior(self):
        """测试5：整合行为测试"""
        print("\n=== 测试整合行为 ===")
        
        # 创建完整系统
        system = self.create_integrated_system(10, 0.85)
        
        # 确保所有组件都已构建
        self.assertTrue(system.check_consciousness_emergence())
        self.assertTrue(system.build_self_model())
        self.assertTrue(system.develop_hierarchy())
        
        print(f"\n完整系统特性:")
        print(f"  基础系统: {len(system.base_system)} 元素")
        print(f"  模型复杂度: {system.self_model.complexity()}")
        print(f"  层级数量: {len(system.hierarchy.levels)}")
        
        # 测试整合行为
        
        # 1. 信息处理的一致性
        test_input = list(range(50))
        hierarchy_result = system.hierarchy.process(test_input)
        
        print(f"\n信息处理:")
        print(f"  输入数据点: {len(test_input)}")
        print(f"  层级处理结果: {len(hierarchy_result.get('upward_flow', []))} 层")
        
        # 验证每层都参与处理
        self.assertEqual(len(hierarchy_result['upward_flow']), len(system.hierarchy.levels),
                        "所有层级都应该参与处理")
        
        # 2. 自我模型与层级的一致性
        model_states = len(system.self_model.states)
        hierarchy_total_states = sum(len(level.states) for level in system.hierarchy.levels)
        
        print(f"  模型状态数: {model_states}")
        print(f"  层级总状态数: {hierarchy_total_states}")
        
        # 状态数量应该相关但不必相等
        state_ratio = hierarchy_total_states / max(model_states, 1)
        self.assertGreater(state_ratio, 0.5, "层级应该有足够的状态表征")
        self.assertLess(state_ratio, 10.0, "层级状态不应过度冗余")
        
        # 3. 意识检测与层级功能的对应
        consciousness_level = system.consciousness_analyzer.measure_consciousness_level(system.base_system)
        hierarchy_differentiation = system.hierarchy.measure_differentiation()
        
        print(f"  意识水平: {consciousness_level['total_level']:.3f}")
        print(f"  层级分化度: {hierarchy_differentiation:.3f}")
        
        # 高意识水平应该对应高分化度
        if consciousness_level['total_level'] > 2.0:
            self.assertGreater(hierarchy_differentiation, 0.5,
                             "高意识水平应该有高层级分化")
        
        # 4. 整体系统稳定性
        hierarchy_stability = system.hierarchy.check_stability()
        model_quality = system.model_builder.measure_model_quality(
            system.self_model, 
            system._convert_to_modeling_system()
        )
        
        print(f"  层级稳定性: {hierarchy_stability['overall_stable']}")
        print(f"  模型质量: {model_quality['overall_coverage']:.3f}")
        
        self.assertTrue(hierarchy_stability["overall_stable"],
                       "整合系统的层级应该是稳定的")
        self.assertGreater(model_quality["overall_coverage"], 0.7,
                         "整合系统的模型应该有良好的覆盖率")
    
    def test_error_propagation_across_levels(self):
        """测试6：跨层级错误传播"""
        print("\n=== 测试跨层级错误传播 ===")
        
        # 创建两个系统：一个正常，一个有缺陷
        normal_system = self.create_integrated_system(10, 0.8)
        defective_system = self.create_integrated_system(10, 0.4)  # 低密度缺陷
        
        # 构建正常系统
        normal_system.check_consciousness_emergence()
        normal_system.build_self_model()
        normal_system.develop_hierarchy()
        
        # 尝试构建缺陷系统
        defective_has_consciousness = defective_system.check_consciousness_emergence()
        defective_has_model = defective_system.build_self_model()
        defective_has_hierarchy = defective_system.develop_hierarchy()
        
        print(f"\n系统比较:")
        print(f"  正常系统: 意识✓, 模型✓, 层级✓")
        print(f"  缺陷系统: 意识{'✓' if defective_has_consciousness else '✗'}, " +
              f"模型{'✓' if defective_has_model else '✗'}, " +
              f"层级{'✓' if defective_has_hierarchy else '✗'}")
        
        # 验证错误传播
        if not defective_has_consciousness:
            self.assertFalse(defective_has_model,
                           "无意识应该阻止模型构建")
            self.assertFalse(defective_has_hierarchy,
                           "无意识应该阻止层级分化")
        
        # 测试局部损伤的影响
        if normal_system.hierarchy:
            # 模拟移除一个中间层级
            if len(normal_system.hierarchy.levels) > 2:
                # 移除中间层
                middle_index = len(normal_system.hierarchy.levels) // 2
                removed_level = normal_system.hierarchy.levels.pop(middle_index)
                
                print(f"\n移除中间层级 {middle_index} 后:")
                
                # 重新检查稳定性
                post_damage_stability = normal_system.hierarchy.check_stability()
                print(f"  系统稳定性: {post_damage_stability['overall_stable']}")
                
                # 测试功能是否仍然可用
                test_input = list(range(20))
                try:
                    result = normal_system.hierarchy.process(test_input)
                    processing_successful = len(result.get('upward_flow', [])) > 0
                    print(f"  信息处理: {'成功' if processing_successful else '失败'}")
                except:
                    processing_successful = False
                    print(f"  信息处理: 失败（异常）")
                
                # 验证系统的韧性
                if len(normal_system.hierarchy.levels) >= 2:
                    self.assertTrue(processing_successful,
                                  "移除单个层级后系统应该仍能工作")


if __name__ == '__main__':
    unittest.main(verbosity=2)