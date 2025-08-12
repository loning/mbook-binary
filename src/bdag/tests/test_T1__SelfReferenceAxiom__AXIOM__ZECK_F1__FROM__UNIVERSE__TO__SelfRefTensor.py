#!/usr/bin/env python3
"""
T1 自指完备公理测试
基于zeckendorf库的验证
"""

import unittest
from base_theory_test import BaseTheoryTest


class TestT1SelfReferenceAxiom(BaseTheoryTest):
    """T1 自指完备公理测试类"""
    
    def get_theory_number(self) -> int:
        return 1
    
    def test_zeckendorf_decomposition(self):
        """测试T1的Zeckendorf分解"""
        # T1 = 1 = F1
        decomp = self.zeckendorf_decompose(1)
        expected = [1]
        
        self.assertEqual(decomp, expected, "T1的Zeckendorf分解应该是[1]")
        self.assert_zeckendorf_valid(decomp, "T1分解必须符合Zeckendorf性质")
    
    def test_fibonacci_properties(self):
        """测试T1的Fibonacci性质"""
        # 1是第一个Fibonacci数
        self.assertTrue(self.is_fibonacci_number(1), "1应该是Fibonacci数")
        self.assertEqual(self.fibonacci_index(1), 1, "1应该是F1")
    
    def test_theory_document_parsing(self):
        """测试理论文档解析"""
        theory_node = self.load_theory_node()
        
        # 验证理论一致性
        self.assert_theory_consistency(theory_node, "T1理论文档解析")
        
        # 验证具体属性
        self.assertEqual(theory_node.operation.value, "AXIOM", "T1应该是公理")
        self.assertTrue(theory_node.is_fibonacci_theory, "T1应该是Fibonacci理论")
        self.assertTrue(theory_node.is_single_axiom_system, "T1应该是单公理系统")
        self.assertEqual(len(theory_node.theory_dependencies), 0, "T1公理不应有依赖")
    
    def test_information_content(self):
        """测试信息含量"""
        # T1的φ-bits信息量
        phi_bits = self.phi_bits(1)
        shannon_bits = self.shannon_bits(1)
        
        # log_φ(1) = 0, log_2(1) = 0
        self.assertEqual(phi_bits, 0.0, "log_φ(1) = 0")
        self.assertEqual(shannon_bits, 0.0, "log_2(1) = 0")
        
        # 但作为基础公理，T1有特殊地位
        self.assertEqual(self.get_theory_number(), 1, "T1是理论编号1")
    
    def test_axiom_properties(self):
        """测试公理特有性质"""
        theory_node = self.load_theory_node()
        
        # 公理的特殊性质
        self.assertTrue(theory_node.is_single_axiom_system, "T1是唯一公理")
        self.assertEqual(theory_node.theory_dependencies, [], "公理无依赖")
        
        # 复杂度应该是最低的
        self.assertEqual(theory_node.complexity_level, 1, "T1复杂度为1")
    
    def test_golden_ratio_foundation(self):
        """测试黄金比例基础"""
        # T1作为F1，是黄金比例序列的起点
        # 虽然F1/F0无意义，但F2/F1 = 2/1 = 2应该离φ较远
        f1, f2 = 1, 2
        ratio = self.golden_ratio_test(f1, f2)
        
        self.assertEqual(ratio, 2.0, "F2/F1 = 2")
        
        # 验证后续的收敛
        f2, f3 = 2, 3
        self.assert_golden_ratio_convergence(f3, f2, tolerance=0.5, 
                                           msg="F3/F2应该开始接近φ")


if __name__ == '__main__':
    unittest.main(verbosity=2)