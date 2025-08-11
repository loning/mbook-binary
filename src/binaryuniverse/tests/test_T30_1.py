#!/usr/bin/env python3
"""
T30-1 φ-代数几何基础理论测试
============================

严格基于唯一公理：自指完备的系统必然熵增
使用严格Zeckendorf编码，no-11约束二进制宇宙

Author: 回音如一 (Echo-As-One)
Date: 2025-08-08
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zeckendorf_base import (
    ZeckendorfInt, PhiPolynomial, PhiIdeal, PhiVariety, 
    EntropyValidator, PhiConstant
)
from base_framework import VerificationTest
from no11_number_system import No11Number


class TestT30_1_Foundation(VerificationTest):
    """T30-1基础测试"""
    
    def setUp(self):
        super().setUp()
        self.z_zero = ZeckendorfInt.from_int(0)
        self.z_one = ZeckendorfInt.from_int(1)
        self.z_two = ZeckendorfInt.from_int(2)
        self.z_three = ZeckendorfInt.from_int(3)
    
    def test_zeckendorf_basic_properties(self):
        """测试Zeckendorf基础性质"""
        # 测试唯一性表示
        self.assertEqual(self.z_one.indices, frozenset({2}))   # F_2 = 1
        self.assertEqual(self.z_two.indices, frozenset({3}))   # F_3 = 2  
        self.assertEqual(self.z_three.indices, frozenset({4})) # F_4 = 3
        
        # 验证no-11约束
        for z in [self.z_one, self.z_two, self.z_three]:
            self.assertTrue(z._is_valid_zeckendorf())
    
    def test_zeckendorf_arithmetic(self):
        """测试Zeckendorf算术"""
        # 基础运算
        sum_result = self.z_two + self.z_three
        self.assertEqual(sum_result.to_int(), 5)
        
        # 验证熵增 - 使用明确会增加熵的例子
        z_simple = ZeckendorfInt.from_int(2)     # 单个索引，熵=1.000
        z_complex = ZeckendorfInt.from_int(12)   # 三个索引{2,4,6}，熵=2.000
        self.assertTrue(EntropyValidator.verify_entropy_increase(z_simple, z_complex))


class TestT30_1_PhiPolynomial(VerificationTest):
    """T30-1 φ-多项式测试"""
    
    def setUp(self):
        super().setUp()
        # 简单φ-多项式：x + y
        self.p1 = PhiPolynomial({
            (1, 0): ZeckendorfInt.from_int(1),  # x
            (0, 1): ZeckendorfInt.from_int(1)   # y
        }, 2)
        
        # x^2 + 2
        self.p2 = PhiPolynomial({
            (2, 0): ZeckendorfInt.from_int(1),  # x^2
            (0, 0): ZeckendorfInt.from_int(2)   # 2
        }, 2)
    
    def test_polynomial_construction(self):
        """测试多项式构造"""
        self.assertEqual(self.p1.degree(), 1)
        self.assertEqual(self.p2.degree(), 2)
        
        # 验证系数有效性
        for poly in [self.p1, self.p2]:
            for coeff in poly.monomials.values():
                self.assertTrue(coeff._is_valid_zeckendorf())
    
    def test_polynomial_operations(self):
        """测试多项式运算"""
        # 加法
        sum_poly = self.p1 + self.p2
        self.assertEqual(sum_poly.variables, 2)
        self.assertEqual(sum_poly.degree(), max(self.p1.degree(), self.p2.degree()))
        
        # 验证熵增
        self.assertTrue(EntropyValidator.verify_entropy_increase(self.p1, sum_poly))


class TestT30_1_PhiIdeal(VerificationTest):
    """T30-1 φ-理想测试"""
    
    def setUp(self):
        super().setUp()
        # 构造简单理想生成元
        self.g1 = PhiPolynomial({
            (2, 0): ZeckendorfInt.from_int(1),  # x^2
            (0, 1): ZeckendorfInt.from_int(1)   # y
        }, 2)
        
        self.g2 = PhiPolynomial({
            (1, 1): ZeckendorfInt.from_int(1),  # xy
            (0, 0): ZeckendorfInt.from_int(1)   # 1
        }, 2)
        
        self.ideal = PhiIdeal([self.g1, self.g2])
    
    def test_ideal_construction(self):
        """测试理想构造"""
        self.assertEqual(len(self.ideal.generators), 2)
        
        # 验证生成元在理想中
        for gen in self.ideal.generators:
            self.assertTrue(self.ideal.contains(gen))
    
    def test_polynomial_division_basic(self):
        """测试基础多项式除法"""
        # 简单测试案例
        dividend = PhiPolynomial({
            (3, 0): ZeckendorfInt.from_int(1),  # x^3
        }, 2)
        
        quotient, remainder = self.ideal._polynomial_division(dividend, self.g1)
        
        # 验证除法有效性
        self.assertIsInstance(quotient, PhiPolynomial)
        self.assertIsInstance(remainder, PhiPolynomial)


class TestT30_1_PhiVariety(VerificationTest):
    """T30-1 φ-代数簇测试"""
    
    def setUp(self):
        super().setUp()
        # 简单理想
        g1 = PhiPolynomial({
            (1, 0): ZeckendorfInt.from_int(1),  # x
        }, 2)
        
        self.ideal = PhiIdeal([g1])
        self.variety = PhiVariety(self.ideal, 2)
    
    def test_variety_construction(self):
        """测试代数簇构造"""
        self.assertIsInstance(self.variety.ideal, PhiIdeal)
        self.assertEqual(self.variety.ambient_dimension, 2)
    
    def test_dimension_calculation(self):
        """测试维数计算"""
        dim = self.variety.dimension
        self.assertGreaterEqual(dim, 0)
        self.assertLessEqual(dim, self.variety.ambient_dimension)
        
        # 零理想的维数应该是环境维数
        zero_ideal = PhiIdeal([])
        zero_variety = PhiVariety(zero_ideal, 2)
        self.assertEqual(zero_variety.dimension, 2)
    
    def test_emptiness_check_basic(self):
        """测试簇空性检查（基础）"""
        # 单位理想应该是空簇
        unit_poly = PhiPolynomial({
            (0, 0): ZeckendorfInt.from_int(1)  # 常数1
        }, 2)
        unit_ideal = PhiIdeal([unit_poly])
        unit_variety = PhiVariety(unit_ideal, 2)
        
        self.assertTrue(unit_variety.is_empty())


class TestT30_1_SystemIntegration(VerificationTest):
    """T30-1系统集成测试"""
    
    def test_unique_axiom_verification(self):
        """测试唯一公理验证"""
        # 创建自指多项式
        self_ref_poly = PhiPolynomial({
            (1, 1): ZeckendorfInt.from_int(1),  # xy 
        }, 2)
        
        # 验证自指性
        self.assertTrue(EntropyValidator.verify_self_reference(self_ref_poly))
        
        # 扩展系统验证熵增
        extended = self_ref_poly + PhiPolynomial({
            (1, 0): ZeckendorfInt.from_int(1)   # x
        }, 2)
        
        self.assertTrue(EntropyValidator.verify_entropy_increase(self_ref_poly, extended))
    
    def test_no11_constraint_compatibility(self):
        """测试no-11约束兼容性"""
        no11_num = No11Number(3)
        zeck_num = ZeckendorfInt.from_int(3)
        
        self.assertEqual(no11_num.value, 3)
        self.assertEqual(zeck_num.to_int(), 3)
    
    def test_phi_properties(self):
        """测试φ性质"""
        phi_const = PhiConstant()
        phi = phi_const.phi()
        
        # φ^2 = φ + 1
        self.assertAlmostEqual(phi * phi, phi + 1, places=10)
    
    def test_complete_workflow(self):
        """测试完整工作流程"""
        # 1. 创建多项式
        p1 = PhiPolynomial({
            (2, 0): ZeckendorfInt.from_int(1),
            (0, 1): ZeckendorfInt.from_int(1)
        }, 2)
        
        p2 = PhiPolynomial({
            (1, 1): ZeckendorfInt.from_int(1),
        }, 2)
        
        # 2. 创建理想
        ideal = PhiIdeal([p1, p2])
        
        # 3. 创建代数簇
        variety = PhiVariety(ideal, 2)
        
        # 4. 验证整个系统
        self.assertIsInstance(variety, PhiVariety)
        self.assertGreaterEqual(variety.dimension, 0)
        
        # 5. 验证熵增链
        p1_entropy = EntropyValidator.entropy(p1)
        ideal_entropy = EntropyValidator.entropy(ideal)
        variety_entropy = EntropyValidator.entropy(variety)
        
        self.assertLessEqual(p1_entropy, ideal_entropy)
        self.assertLessEqual(ideal_entropy, variety_entropy)


if __name__ == '__main__':
    unittest.main(verbosity=2)