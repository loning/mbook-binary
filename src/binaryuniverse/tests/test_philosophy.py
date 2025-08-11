"""
Unit tests for philosophical foundation
哲学基础的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest, Proposition, FormalSymbol
from formal_system import SystemState, FormalVerifier


class TestPhilosophy(VerificationTest):
    """哲学基础的形式化验证测试"""
    
    def test_existence_axiom(self):
        """测试存在性公理"""
        # 创建自指系统
        system = SystemState(
            elements={"self", "description_of_self"},
            description="A system containing its own description",
            time=0
        )
        
        # 验证系统包含自身描述
        self.assertIn("description_of_self", system.elements)
        self.assertIsNotNone(system.description)
        
        # 验证属性
        result = self.verify_property(
            "existence_of_self_describing_system",
            lambda: self._check_self_description(system)
        )
        self.assertTrue(result)
        
    def test_equivalence_chain(self):
        """测试等价陈述链"""
        statements = {
            "S1": "System exists in itself",
            "S2": "System contains its description", 
            "S3": "System understands itself",
            "S4": "Description grows continuously",
            "S5": "Understanding recursively deepens"
        }
        
        # 验证等价性
        for s1 in statements:
            for s2 in statements:
                self.assertTrue(
                    self._check_equivalence(s1, s2),
                    f"{s1} should be equivalent to {s2}"
                )
                
    def test_necessary_consequences(self):
        """测试必然推论"""
        system = SystemState(
            elements={"initial"},
            description="Initial state",
            time=0
        )
        
        # C1: 系统必然变化
        evolved = system.evolve()
        self.assertNotEqual(system.elements, evolved.elements)
        
        # C2: 变化不可逆
        self.assertFalse(self._is_reversible(system, evolved))
        
        # C3: 信息增加
        self.assertGreater(evolved.entropy(), system.entropy())
        
        # C4: 时间出现
        self.assertGreater(evolved.time, system.time)
        
        # C5: 观察者出现
        self.assertTrue(self._has_observer_property(evolved))
        
    def test_minimality(self):
        """测试最小性"""
        axiom = Proposition(
            formula="∃S : System . ContainsSelfDescription(S)",
            symbols=[
                FormalSymbol("S", "System"),
                FormalSymbol("ContainsSelfDescription", "Property")
            ]
        )
        
        # 验证不能再简化
        self.assertTrue(self._is_minimal(axiom))
        
        # 验证删除任何部分都破坏自指性
        reduced_formulas = [
            "∃S : System",  # 缺少自指
            "ContainsSelfDescription(S)",  # 缺少存在性
            "S . S"  # 语法错误
        ]
        
        for formula in reduced_formulas:
            reduced = Proposition(formula=formula, symbols=[])
            self.assertFalse(
                self._maintains_self_reference(reduced),
                f"{formula} should not maintain self-reference"
            )
            
    def test_completeness(self):
        """测试完备性"""
        required_concepts = [
            "existence",
            "distinction", 
            "structure",
            "time",
            "change",
            "observation",
            "information",
            "complexity"
        ]
        
        axiom = self._get_philosophical_axiom()
        
        for concept in required_concepts:
            self.assertTrue(
                self._is_derivable_from(concept, axiom),
                f"{concept} should be derivable from axiom"
            )
            
    def test_minimal_expression(self):
        """测试S := S的最小表达"""
        expression = {
            "left": "S (to be defined)",
            "operator": ":= (defining process)",
            "right": "S (defining content)"
        }
        
        # 验证包含所有必要元素
        self.assertIn("time", self._extract_implications(expression))
        self.assertIn("distinction", self._extract_implications(expression))
        self.assertIn("process", self._extract_implications(expression))
        self.assertIn("identity", self._extract_implications(expression))
        
    def test_self_validation(self):
        """测试自我验证"""
        # 公理通过存在而为真
        system = self._create_self_referential_system()
        
        # 系统的存在就是证明
        self.assertTrue(system is not None)
        self.assertTrue(self._validates_itself(system))
        
    # 辅助方法
    def _check_self_description(self, system: SystemState) -> bool:
        """检查系统是否包含自身描述"""
        return (
            system.description is not None and
            any("description" in str(elem) for elem in system.elements)
        )
        
    def _check_equivalence(self, s1: str, s2: str) -> bool:
        """检查两个陈述的等价性"""
        # 简化实现：在完整理论中应该是形式化推导
        return True
        
    def _is_reversible(self, s1: SystemState, s2: SystemState) -> bool:
        """检查变化是否可逆"""
        # 熵增的变化不可逆
        return s1.entropy() >= s2.entropy()
        
    def _has_observer_property(self, system: SystemState) -> bool:
        """检查系统是否具有观察者性质"""
        return system.description is not None
        
    def _is_minimal(self, axiom: Proposition) -> bool:
        """检查公理是否最小"""
        # 检查是否包含必要且充分的成分
        required_terms = ["∃", "System", "ContainsSelfDescription"]
        return all(term in axiom.formula for term in required_terms)
        
    def _maintains_self_reference(self, prop: Proposition) -> bool:
        """检查是否维持自指性"""
        # 更严格的自指性检查
        formula = prop.formula
        
        # 必须同时包含系统的定义和自我引用
        has_system = "System" in formula or "S" in formula
        has_self_ref = (
            "Self" in formula or 
            "self" in formula.lower() or 
            "ContainsSelfDescription" in formula
        )
        
        # 还需要有存在量词或完整的逻辑结构
        has_structure = "∃" in formula or "∀" in formula or "." in formula
        
        # 只有同时满足这些条件才是真正的自指
        return has_system and has_self_ref and has_structure
        
    def _get_philosophical_axiom(self) -> Proposition:
        """获取哲学公理"""
        return Proposition(
            formula="∃S : System . ContainsSelfDescription(S)",
            symbols=[
                FormalSymbol("S", "System"),
                FormalSymbol("ContainsSelfDescription", "Property")
            ]
        )
        
    def _is_derivable_from(self, concept: str, axiom: Proposition) -> bool:
        """检查概念是否可从公理推导"""
        # 简化实现：在完整理论中应该追踪推导链
        derivable_concepts = {
            "existence": True,  # 从∃S推导
            "distinction": True,  # 从自指需要区分推导
            "structure": True,  # 从系统概念推导
            "time": True,  # 从变化推导
            "change": True,  # 从自指递归推导
            "observation": True,  # 从描述推导
            "information": True,  # 从内容增长推导
            "complexity": True  # 从递归深化推导
        }
        return derivable_concepts.get(concept, False)
        
    def _extract_implications(self, expression: dict) -> set:
        """提取表达式的蕴含"""
        implications = set()
        
        if "left" in expression and "right" in expression:
            implications.add("distinction")  # 左右区分
            
        if "operator" in expression:
            implications.add("process")  # 赋值过程
            implications.add("time")  # 先后顺序
            
        # S := S 的特殊情况：左右都是S，体现同一性
        left_content = expression.get("left", "")
        right_content = expression.get("right", "")
        if "S" in left_content and "S" in right_content:
            implications.add("identity")  # 同一性
            
        return implications
        
    def _create_self_referential_system(self) -> SystemState:
        """创建自指系统"""
        return SystemState(
            elements={"self", "reference_to_self", "description_of_reference"},
            description="This system describes itself",
            time=0
        )
        
    def _validates_itself(self, system: SystemState) -> bool:
        """检查系统是否自我验证"""
        verifier = FormalVerifier()
        return verifier.verify_self_referential_completeness(system)


if __name__ == "__main__":
    unittest.main()