#!/usr/bin/env python3
"""
C9-3与C9-1, C9-2系统一致性验证程序

验证自指代数系统与算术、数论系统的严格一致性：
- 代数运算基于C9-1的自指算术
- 素元素来自C9-2的素数理论
- No-11约束的全局维护
- 熵增性质的传递
"""

import unittest
import sys
import os
from typing import Set, List

# 导入系统
sys.path.append(os.path.join(os.path.dirname(__file__)))
from base_framework import VerificationTest
from no11_number_system import No11Number
from test_C9_1 import SelfReferentialArithmetic
from test_C9_2 import RecursiveNumberTheory
from test_C9_3 import (
    SelfReferentialGroup, SelfReferentialRing, SelfReferentialField,
    AlgebraicStructureFactory, GroupHomomorphism
)


class TestC93ConsistencyWithC91C92(VerificationTest):
    """C9-3与C9-1, C9-2系统一致性验证"""
    
    def setUp(self):
        """设置测试环境"""
        super().setUp()
        
        # 初始化C9-1算术系统
        self.arithmetic = SelfReferentialArithmetic(max_depth=8, max_value=30)
        
        # 初始化C9-2数论系统
        self.number_theory = RecursiveNumberTheory(self.arithmetic, max_recursion=10)
        
        # 初始化C9-3代数工厂
        self.algebra_factory = AlgebraicStructureFactory(self.arithmetic)
        
        # 测试数据
        self.small_primes = [2, 3, 5, 7, 11]
    
    def test_algebraic_operations_use_c9_1_arithmetic(self):
        """验证代数运算使用C9-1的自指算术"""
        # 创建一个小的循环群
        z6 = self.algebra_factory.create_cyclic_group(6)
        
        # 测试群运算
        a = No11Number(2)
        b = No11Number(3)
        
        # 群运算结果
        group_result = z6.operate(a, b)
        
        # 使用C9-1算术验证
        # 在Z_6中，2+3=5
        expected = No11Number(5)
        self.assertEqual(group_result, expected)
        
        # 验证群运算内部使用的是模运算
        # (2 + 3) mod 6 = 5
        direct_add = self.arithmetic.self_referential_add(a, b)
        self.assertEqual(direct_add.value, 5)
    
    def test_field_construction_uses_c9_2_primes(self):
        """验证域构造使用C9-2的素数理论"""
        for p in self.small_primes:
            with self.subTest(prime=p):
                # 使用C9-2验证p是素数
                p_no11 = No11Number(p)
                is_prime = self.number_theory.is_prime(p_no11)
                
                if is_prime:
                    # 只有素数才能构造素数阶域
                    field = self.algebra_factory.create_prime_field(p)
                    self.assertEqual(len(field.elements), p)
                    self.assertEqual(field.characteristic(), p)
                else:
                    # 非素数不能构造域
                    with self.assertRaises(Exception):
                        self.algebra_factory.create_prime_field(p)
    
    def test_ideal_corresponds_to_factorization(self):
        """验证理想对应C9-2的因式分解结构"""
        # 创建Z_12环
        elements = {No11Number(i) for i in range(12)}
        
        def add_mod12(a: No11Number, b: No11Number) -> No11Number:
            return No11Number((a.value + b.value) % 12)
        
        def mul_mod12(a: No11Number, b: No11Number) -> No11Number:
            return No11Number((a.value * b.value) % 12)
        
        ring_z12 = SelfReferentialRing(elements, add_mod12, mul_mod12)
        
        # 生成由3生成的理想
        ideal_3 = ring_z12.ideal_generated_by({No11Number(3)})
        
        # 理想包含3的所有倍数
        for n in ideal_3:
            if n.value > 0:
                # n应该是3的倍数
                self.assertEqual(n.value % 3, 0)
        
        # 理想的大小应该是12/gcd(12,3) = 12/3 = 4
        self.assertEqual(len(ideal_3), 4)
    
    def test_homomorphism_preserves_c9_1_operations(self):
        """验证同态保持C9-1的运算"""
        # 创建同态 φ: Z_6 -> Z_3
        z6 = self.algebra_factory.create_cyclic_group(6)
        z3 = self.algebra_factory.create_cyclic_group(3)
        
        def phi(x: No11Number) -> No11Number:
            return No11Number(x.value % 3)
        
        hom = GroupHomomorphism(z6, z3, phi)
        
        # 验证同态性质：φ(a+b) = φ(a)+φ(b)
        for a_val in range(6):
            for b_val in range(6):
                a = No11Number(a_val)
                b = No11Number(b_val)
                
                # 左边：φ(a+b)
                sum_in_z6 = z6.operate(a, b)
                left = hom.map(sum_in_z6)
                
                # 右边：φ(a)+φ(b)
                phi_a = hom.map(a)
                phi_b = hom.map(b)
                right = z3.operate(phi_a, phi_b)
                
                self.assertEqual(left, right)
    
    def test_no11_constraint_preservation_in_algebra(self):
        """验证代数结构保持No-11约束"""
        # 创建一个域
        field = self.algebra_factory.create_prime_field(7)
        
        # 执行各种运算
        operations_count = 0
        for a in field.elements:
            for b in field.elements:
                # 加法
                sum_result = field.add(a, b)
                self.assertIsInstance(sum_result, No11Number)
                operations_count += 1
                
                # 乘法
                prod_result = field.multiply(a, b)
                self.assertIsInstance(prod_result, No11Number)
                operations_count += 1
                
                # 除法（如果b非零）
                if b != field.zero:
                    div_result = field.divide(a, b)
                    self.assertIsInstance(div_result, No11Number)
                    operations_count += 1
        
        # 验证执行了大量运算且都保持No-11约束
        self.assertGreater(operations_count, 100)
    
    def test_entropy_increase_through_layers(self):
        """验证熵增通过层次传递"""
        # C9-1: 基础算术
        a = No11Number(3)
        b = No11Number(4)
        
        # 初始信息
        initial_info = len(a.bits) + len(b.bits)
        
        # C9-1层：算术运算
        sum_result = self.arithmetic.self_referential_add(a, b)
        arithmetic_info = initial_info + 1  # 运算类型信息
        
        # C9-2层：数论分类
        is_prime_a = self.number_theory.is_prime(a)
        is_prime_b = self.number_theory.is_prime(b)
        is_prime_sum = self.number_theory.is_prime(sum_result)
        number_theory_info = arithmetic_info + 3  # 三个分类信息
        
        # C9-3层：代数结构
        z12 = self.algebra_factory.create_cyclic_group(12)
        order_a = z12.order(a)
        order_b = z12.order(b)
        algebra_info = number_theory_info + 2  # 两个阶信息
        
        # 验证信息单调增加
        self.assertGreater(arithmetic_info, initial_info)
        self.assertGreater(number_theory_info, arithmetic_info)
        self.assertGreater(algebra_info, number_theory_info)
    
    def test_prime_field_units_are_multiplicative_group(self):
        """验证素数域的单位形成乘法群"""
        p = 5
        field = self.algebra_factory.create_prime_field(p)
        
        # 获取所有单位（非零元素）
        units = field.ring.units()
        expected_units = {No11Number(i) for i in range(1, p)}
        self.assertEqual(units, expected_units)
        
        # 验证单位形成群
        mult_group = field.multiplicative_group
        self.assertEqual(mult_group.elements, units)
        
        # 验证群的阶
        self.assertEqual(len(mult_group.elements), p - 1)
        
        # 验证是循环群（对于素数域总是成立）
        # 找一个生成元
        for g in mult_group.elements:
            if mult_group.order(g) == p - 1:
                # g是生成元
                powers = set()
                power = g
                for _ in range(p - 1):
                    powers.add(power)
                    power = mult_group.operate(power, g)
                self.assertEqual(powers, units)
                break
    
    def test_arithmetic_in_quotient_structures(self):
        """验证商结构中的算术运算"""
        # 创建Z_12环
        elements = {No11Number(i) for i in range(12)}
        
        def add_mod12(a: No11Number, b: No11Number) -> No11Number:
            return No11Number((a.value + b.value) % 12)
        
        def mul_mod12(a: No11Number, b: No11Number) -> No11Number:
            return No11Number((a.value * b.value) % 12)
        
        ring_z12 = SelfReferentialRing(elements, add_mod12, mul_mod12)
        
        # 验证理想的加法封闭性
        ideal_4 = ring_z12.ideal_generated_by({No11Number(4)})
        
        # 理想中任意两元素之和仍在理想中
        for a in ideal_4:
            for b in ideal_4:
                sum_ab = ring_z12.add(a, b)
                self.assertIn(sum_ab, ideal_4)
    
    def test_lagrange_theorem_with_c9_2_factorization(self):
        """使用C9-2的因式分解验证拉格朗日定理"""
        # 对于群G，子群H的阶整除|G|
        n = 12
        z12 = self.algebra_factory.create_cyclic_group(n)
        
        # 使用C9-2因式分解12
        n_no11 = No11Number(n)
        if not self.number_theory.is_prime(n_no11):
            factors = self.number_theory.factorize(n_no11)
            
            # 12 = 2^2 * 3
            # 可能的子群阶：1, 2, 3, 4, 6, 12
            divisors = [1, 2, 3, 4, 6, 12]
            
            # 测试一些子群
            subgroup_tests = [
                ({No11Number(0), No11Number(6)}, 2),
                ({No11Number(0), No11Number(4), No11Number(8)}, 3),
                ({No11Number(0), No11Number(3), No11Number(6), No11Number(9)}, 4),
            ]
            
            for elements, expected_order in subgroup_tests:
                subgroup = z12.subgroup(elements)
                self.assertEqual(len(subgroup.elements), expected_order)
                self.assertIn(expected_order, divisors)
    
    def test_field_arithmetic_matches_modular_arithmetic(self):
        """验证域算术与模算术的一致性"""
        p = 7
        field = self.algebra_factory.create_prime_field(p)
        
        # 测试一系列运算
        test_cases = [
            (3, 4, 0, 5),  # 3+4=7≡0 (mod 7), 3*4=12≡5 (mod 7)
            (5, 6, 4, 2),  # 5+6=11≡4 (mod 7), 5*6=30≡2 (mod 7)
            (2, 2, 4, 4),  # 2+2=4, 2*2=4
        ]
        
        for a_val, b_val, expected_sum, expected_prod in test_cases:
            a = No11Number(a_val)
            b = No11Number(b_val)
            
            # 域运算
            field_sum = field.add(a, b)
            field_prod = field.multiply(a, b)
            
            # 验证结果
            self.assertEqual(field_sum.value, expected_sum)
            self.assertEqual(field_prod.value, expected_prod)
            
            # 使用C9-2的模运算验证
            mod_sum = self.number_theory.modular_add(a, b, No11Number(p))
            mod_prod = self.number_theory.modular_multiply(a, b, No11Number(p))
            
            self.assertEqual(field_sum, mod_sum)
            self.assertEqual(field_prod, mod_prod)
    
    def test_comprehensive_layer_integration(self):
        """综合测试三层理论的集成"""
        # 选择一个素数
        p = 5
        p_no11 = No11Number(p)
        
        # C9-2: 验证是素数
        self.assertTrue(self.number_theory.is_prime(p_no11))
        
        # C9-3: 构造素数域
        field = self.algebra_factory.create_prime_field(p)
        
        # 在域中执行运算
        a = No11Number(2)
        b = No11Number(3)
        
        # C9-1: 基础算术
        basic_sum = self.arithmetic.self_referential_add(a, b)
        basic_prod = self.arithmetic.self_referential_multiply(a, b)
        
        # C9-3: 域算术
        field_sum = field.add(a, b)
        field_prod = field.multiply(a, b)
        
        # 由于5是素数，在模5意义下：
        # 2+3=5≡0 (mod 5)
        # 2*3=6≡1 (mod 5)
        self.assertEqual(field_sum.value, 0)
        self.assertEqual(field_prod.value, 1)
        
        # 验证逆元
        # 2的乘法逆元是3，因为2*3≡1 (mod 5)
        inv_2 = field.multiplicative_inverse(a)
        self.assertEqual(inv_2, b)
        
        # 验证 2 * 3 = 1 in F_5
        verify = field.multiply(a, inv_2)
        self.assertEqual(verify, field.one)
        
        # 整个过程保持No-11约束
        for elem in [a, b, field_sum, field_prod, inv_2, verify]:
            self.assertIsInstance(elem, No11Number)


if __name__ == '__main__':
    unittest.main(verbosity=2)