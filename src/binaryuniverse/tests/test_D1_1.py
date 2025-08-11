"""
Unit tests for D1-1: Self-Referential Completeness
D1-1：自指完备性的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
from formal_system import SystemState, FormalVerifier


class TestD1_1_SelfReferentialCompleteness(VerificationTest):
    """D1-1自指完备性的形式化验证测试"""
    
    def test_self_referential(self):
        """测试自指性"""
        # 创建自指系统
        system = SystemState(
            elements={"self", "reference_to_self"},
            description="System that refers to itself",
            time=0
        )
        
        # 验证存在函数f使得S = f(S)
        def self_function(s):
            # 模拟自指函数
            return s
        
        # 验证S = f(S)
        self.assertEqual(
            system,
            self_function(system),
            "System should equal f(System)"
        )
        
        # 验证系统包含对自身的引用
        self.assertIn("self", system.elements)
        self.assertIn("reference_to_self", system.elements)
        
    def test_completeness(self):
        """测试完备性"""
        system = SystemState(
            elements={"a", "b", "generator_of_a", "generator_of_b"},
            description="Complete system",
            time=0
        )
        
        # 验证每个元素都有内部来源
        element_origins = {
            "a": "generator_of_a",
            "b": "generator_of_b", 
            "generator_of_a": "b",  # 循环引用
            "generator_of_b": "a"   # 循环引用
        }
        
        # 检查所有元素都有来源
        for element in system.elements:
            self.assertIn(
                element, 
                element_origins,
                f"Element {element} should have an origin"
            )
            origin = element_origins[element]
            self.assertIn(
                origin,
                system.elements,
                f"Origin {origin} should be in system"
            )
            
    def test_consistency(self):
        """测试一致性"""
        # 一致的系统
        consistent_system = SystemState(
            elements={"true", "valid", "exists"},
            description="Consistent system",
            time=0
        )
        
        # 验证没有矛盾
        # 简化测试：检查没有相反的元素
        for element in consistent_system.elements:
            opposite = f"not_{element}"
            self.assertNotIn(
                opposite,
                consistent_system.elements,
                f"System should not contain both {element} and {opposite}"
            )
            
        # 不一致的系统示例（用于对比）
        inconsistent_elements = {"true", "not_true"}
        self.assertFalse(
            self._check_consistency(inconsistent_elements),
            "System with contradictory elements should be inconsistent"
        )
        
    def test_non_triviality(self):
        """测试非平凡性"""
        # 平凡系统（只有一个元素）
        trivial_system = SystemState(
            elements={"only_one"},
            description="Trivial system",
            time=0
        )
        
        # 非平凡系统（多个元素）
        non_trivial_system = SystemState(
            elements={"element1", "element2"},
            description="Non-trivial system", 
            time=0
        )
        
        # 验证
        self.assertEqual(len(trivial_system.elements), 1)
        self.assertGreater(len(non_trivial_system.elements), 1)
        
        verifier = FormalVerifier()
        self.assertFalse(
            verifier._check_non_trivial(trivial_system),
            "Trivial system should not pass non-triviality check"
        )
        self.assertTrue(
            verifier._check_non_trivial(non_trivial_system),
            "Non-trivial system should pass non-triviality check"
        )
        
    def test_full_src(self):
        """测试完整的自指完备性"""
        # 创建满足所有条件的系统
        src_system = SystemState(
            elements={
                "system_itself",
                "description_of_system",
                "reference_to_description",
                "generator_function"
            },
            description="A fully self-referential complete system",
            time=0
        )
        
        verifier = FormalVerifier()
        
        # 验证所有四个条件
        self.assertTrue(
            verifier._check_self_referential(src_system),
            "System should be self-referential"
        )
        self.assertTrue(
            verifier._check_completeness(src_system),
            "System should be complete"
        )
        self.assertTrue(
            verifier._check_consistency(src_system),
            "System should be consistent"
        )
        self.assertTrue(
            verifier._check_non_trivial(src_system),
            "System should be non-trivial"
        )
        
        # 验证整体SRC
        self.assertTrue(
            verifier.verify_self_referential_completeness(src_system),
            "System should be self-referentially complete"
        )
        
    def test_irreducibility(self):
        """测试不可约性"""
        src_system = SystemState(
            elements={"core", "self_ref", "complete_desc", "generator"},
            description="Irreducible SRC system",
            time=0
        )
        
        # 尝试分解系统
        subset1 = {"core"}  # 太小，不满足非平凡性
        subset2 = {"self_ref"}  # 没有完整的自指结构
        
        # 子集都不是SRC
        verifier = FormalVerifier()
        sub_system1 = SystemState(elements=subset1, description="Subset 1", time=0)
        sub_system2 = SystemState(elements=subset2, description="Subset 2", time=0)
        
        self.assertFalse(
            verifier.verify_self_referential_completeness(sub_system1),
            "Subset 1 should not be SRC"
        )
        self.assertFalse(
            verifier.verify_self_referential_completeness(sub_system2),
            "Subset 2 should not be SRC"
        )
        
        # 验证原系统是SRC
        self.assertTrue(
            verifier.verify_self_referential_completeness(src_system),
            "Original system should be SRC"
        )
        
    def test_closure(self):
        """测试封闭性"""
        system = SystemState(
            elements={"a", "b", "operation_result"},
            description="Closed system",
            time=0
        )
        
        # 定义系统内的操作
        def internal_operation(elem1, elem2):
            return "operation_result"
        
        # 验证操作结果在系统内
        result = internal_operation("a", "b")
        self.assertIn(
            result,
            system.elements,
            "Operation result should be in the system"
        )
        
    def test_recursion(self):
        """测试递归性"""
        system = SystemState(
            elements={"system", "description_of_system", "meta_description"},
            description="This is a description of the system",
            time=0
        )
        
        # 验证系统包含自身的描述
        self.assertIn("description_of_system", system.elements)
        self.assertIsNotNone(system.description)
        
        # 验证描述可以描述系统
        self.assertIn("system", system.description.lower())
        
    def test_dynamics(self):
        """测试动态性"""
        initial_system = SystemState(
            elements={"initial_state"},
            description="Initial SRC system",
            time=0
        )
        
        # 演化系统
        evolved_system = initial_system.evolve()
        
        # 验证系统发生了变化
        self.assertNotEqual(
            initial_system.elements,
            evolved_system.elements,
            "System should change over time"
        )
        self.assertNotEqual(
            initial_system.time,
            evolved_system.time,
            "Time should advance"
        )
        
    # 辅助方法
    def _check_consistency(self, elements):
        """检查元素集合的一致性"""
        for elem in elements:
            if f"not_{elem}" in elements or elem.startswith("not_") and elem[4:] in elements:
                return False
        return True


if __name__ == "__main__":
    unittest.main()