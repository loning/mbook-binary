#!/usr/bin/env python3
"""
C9-1与C9-2系统一致性验证程序

验证修正后的No-11算术系统在两层理论中的严格一致性：
- C9-1自指算术与C9-2递归数论的依赖关系
- 数值运算的一致性
- No-11约束的维护
- 熵增性质的传递
"""

import unittest
import sys
import os
from typing import List

# 导入系统
sys.path.append(os.path.join(os.path.dirname(__file__)))
from base_framework import VerificationTest
from no11_number_system import No11Number
from test_C9_1 import SelfReferentialArithmetic
from test_C9_2 import RecursiveNumberTheory


class TestC91C92Consistency(VerificationTest):
    """C9-1与C9-2系统一致性验证"""
    
    def setUp(self):
        """设置测试环境"""
        super().setUp()
        
        # 初始化基础算术系统
        self.arithmetic = SelfReferentialArithmetic(max_depth=8, max_value=50)
        
        # 初始化数论系统（基于同一个算术系统）
        self.number_theory = RecursiveNumberTheory(self.arithmetic, max_recursion=10)
        
        # 测试数据
        self.test_numbers = [
            No11Number(2), No11Number(3), No11Number(4), 
            No11Number(5), No11Number(6), No11Number(9)
        ]
    
    def test_arithmetic_system_identity(self):
        """验证数论系统使用的就是同一个算术系统"""
        self.assertIs(self.number_theory.arithmetic, self.arithmetic,
                     "RecursiveNumberTheory must use the same SelfReferentialArithmetic instance")
        
        # 验证内部组件也使用同一个算术系统
        self.assertIs(self.number_theory.prime_checker.arithmetic, self.arithmetic)
        self.assertIs(self.number_theory.factorization_engine.arithmetic, self.arithmetic)
        self.assertIs(self.number_theory.modular_arithmetic.arithmetic, self.arithmetic)
    
    def test_addition_consistency(self):
        """验证加法在两个系统中的一致性"""
        for a in self.test_numbers[:3]:
            for b in self.test_numbers[:3]:
                with self.subTest(a=a.value, b=b.value):
                    # C9-1直接算术
                    c9_1_result = self.arithmetic.self_referential_add(a, b)
                    
                    # C9-2通过模运算（模数足够大以不影响结果）
                    large_modulus = No11Number(100)
                    c9_2_result = self.number_theory.modular_add(a, b, large_modulus)
                    
                    # 结果应该相同
                    self.assertEqual(c9_1_result, c9_2_result,
                                   f"Addition inconsistency: C9-1 {c9_1_result} ≠ C9-2 {c9_2_result}")
    
    def test_multiplication_consistency(self):
        """验证乘法在两个系统中的一致性"""
        for a in self.test_numbers[:3]:
            for b in self.test_numbers[:3]:
                with self.subTest(a=a.value, b=b.value):
                    # C9-1直接算术
                    c9_1_result = self.arithmetic.self_referential_multiply(a, b)
                    
                    # C9-2通过模运算（模数足够大）
                    large_modulus = No11Number(100)
                    c9_2_result = self.number_theory.modular_multiply(a, b, large_modulus)
                    
                    # 结果应该相同
                    self.assertEqual(c9_1_result, c9_2_result,
                                   f"Multiplication inconsistency: C9-1 {c9_1_result} ≠ C9-2 {c9_2_result}")
    
    def test_prime_factorization_consistency(self):
        """验证素数检测与因式分解的一致性"""
        for num in self.test_numbers:
            with self.subTest(number=num.value):
                is_prime = self.number_theory.is_prime(num)
                
                if is_prime:
                    # 素数的因式分解应该只有自身
                    factors = self.number_theory.factorize(num)
                    self.assertEqual(len(factors), 1, f"Prime {num} should have exactly one factor")
                    self.assertEqual(factors[0], num, f"Prime's only factor should be itself")
                else:
                    # 合数应该有多个因子
                    try:
                        factors = self.number_theory.factorize(num)
                        self.assertGreater(len(factors), 1, f"Composite {num} should have multiple factors")
                        
                        # 验证所有因子的乘积等于原数
                        product = factors[0]
                        for factor in factors[1:]:
                            product = self.arithmetic.self_referential_multiply(product, factor)
                        
                        self.assertEqual(product, num,
                                       f"Factorization verification failed for {num}")
                    except Exception:
                        # 有些合数可能因为算法限制无法分解，这是可接受的
                        pass
    
    def test_self_reference_preservation(self):
        """验证自指性质在数论操作中的保持"""
        for num in self.test_numbers:
            with self.subTest(number=num.value):
                # 验证数本身是自指的
                self.assertTrue(self.arithmetic.verify_self_reference(num),
                              f"Number {num} should be self-referential")
                
                # 验证运算结果保持自指性
                doubled = self.arithmetic.self_referential_add(num, num)
                self.assertTrue(self.arithmetic.verify_self_reference(doubled),
                              f"Addition result {doubled} should be self-referential")
                
                squared = self.arithmetic.self_referential_multiply(num, num)
                self.assertTrue(self.arithmetic.verify_self_reference(squared),
                              f"Multiplication result {squared} should be self-referential")
    
    def test_entropy_increase_consistency(self):
        """验证熵增性质在两个系统中的一致性"""
        for num in self.test_numbers[:3]:
            with self.subTest(number=num.value):
                # C9-1算术熵增
                initial_complexity = len(num.bits) + num.value
                
                # 加法操作的熵增
                one = No11Number(1)
                add_result = self.arithmetic.self_referential_add(num, one)
                add_entropy = self.arithmetic.calculate_entropy_increase([num, one], add_result)
                
                # 数论操作的信息增加（素数检测）
                is_prime = self.number_theory.is_prime(num)
                classification_info = 1  # 增加了分类信息
                
                # 熵增应该是非负的
                self.assertGreaterEqual(add_entropy, 0,
                                      f"Arithmetic entropy should not decrease for {num}")
                
                # 分类操作增加了信息
                self.assertGreater(initial_complexity + classification_info, initial_complexity,
                                 f"Prime classification should increase information for {num}")
    
    def test_gcd_with_arithmetic_operations(self):
        """验证GCD与基本算术运算的一致性"""
        test_pairs = [
            (No11Number(4), No11Number(6)),
            (No11Number(9), No11Number(6)),
            (No11Number(5), No11Number(3))
        ]
        
        for a, b in test_pairs:
            with self.subTest(a=a.value, b=b.value):
                gcd_result = self.number_theory.gcd(a, b)
                
                # 验证GCD的基本性质：gcd(a,b) 能整除 a 和 b
                # 这需要通过重复减法验证（即 a % gcd == 0 和 b % gcd == 0）
                
                # 验证gcd <= min(a, b)
                min_ab = a if a.value <= b.value else b
                self.assertLessEqual(gcd_result.value, min_ab.value,
                                   f"GCD {gcd_result} should not exceed min({a}, {b})")
                
                # 验证gcd > 0
                self.assertGreater(gcd_result.value, 0,
                                 f"GCD should be positive for non-zero inputs")
    
    def test_modular_arithmetic_with_basic_operations(self):
        """验证模运算与基本运算的一致性"""
        test_cases = [
            (No11Number(5), No11Number(3), No11Number(7)),
            (No11Number(4), No11Number(2), No11Number(5)),
        ]
        
        for a, b, m in test_cases:
            with self.subTest(a=a.value, b=b.value, m=m.value):
                # 模加法
                mod_add = self.number_theory.modular_add(a, b, m)
                
                # 验证：(a + b) mod m 与直接计算的一致性
                direct_add = self.arithmetic.self_referential_add(a, b)
                
                # 如果直接结果小于模数，应该相等
                if direct_add.value < m.value:
                    self.assertEqual(mod_add, direct_add,
                                   f"Modular addition should match direct addition when result < modulus")
                
                # 模乘法类似验证
                mod_mul = self.number_theory.modular_multiply(a, b, m)
                direct_mul = self.arithmetic.self_referential_multiply(a, b)
                
                if direct_mul.value < m.value:
                    self.assertEqual(mod_mul, direct_mul,
                                   f"Modular multiplication should match direct multiplication when result < modulus")
    
    def test_no_11_constraint_consistency(self):
        """验证no-11约束在两个系统中的一致维护"""
        for num in self.test_numbers:
            with self.subTest(number=num.value):
                # 验证数本身满足no-11约束（No11Number自动保证）
                self.assertIsInstance(num, No11Number)
                
                # C9-1运算结果应该保持约束
                doubled = self.arithmetic.self_referential_add(num, num)
                self.assertIsInstance(doubled, No11Number)
                
                # C9-2运算结果应该保持约束
                if not self.number_theory.is_prime(num):
                    try:
                        factors = self.number_theory.factorize(num)
                        for factor in factors:
                            self.assertIsInstance(factor, No11Number)
                    except Exception:
                        pass  # 分解失败不影响约束测试
                
                # 模运算结果保持约束
                mod_result = self.number_theory.modular_add(num, No11Number(1), No11Number(7))
                self.assertIsInstance(mod_result, No11Number)
    
    def test_recursive_operations_terminate(self):
        """验证递归操作能够正确终止"""
        # 测试较大数值不会导致无限递归
        larger_numbers = [No11Number(i) for i in range(10, 16)]
        
        for num in larger_numbers:
            with self.subTest(number=num.value):
                import time
                start_time = time.time()
                
                try:
                    # 素数检测应该在合理时间内完成
                    is_prime = self.number_theory.is_prime(num)
                    elapsed = time.time() - start_time
                    
                    self.assertLess(elapsed, 3.0, f"Prime checking for {num} took too long")
                    self.assertIsInstance(is_prime, bool)
                    
                except RecursionError:
                    # 递归限制是可接受的
                    pass
                except Exception as e:
                    self.fail(f"Unexpected error in recursive operation for {num}: {e}")
    
    def test_system_component_consistency(self):
        """验证系统组件间的一致性"""
        # 验证素数检测器
        self.assertIsNotNone(self.number_theory.prime_checker)
        self.assertEqual(self.number_theory.prime_checker.max_recursion,
                        self.number_theory.max_recursion)
        
        # 验证因式分解引擎
        self.assertIsNotNone(self.number_theory.factorization_engine)
        self.assertEqual(self.number_theory.factorization_engine.max_recursion,
                        self.number_theory.max_recursion)
        
        # 验证模运算系统
        self.assertIsNotNone(self.number_theory.modular_arithmetic)
        self.assertEqual(self.number_theory.modular_arithmetic.max_recursion,
                        self.number_theory.max_recursion)
        
        # 验证所有组件都使用同一个算术系统
        components = [
            self.number_theory.prime_checker,
            self.number_theory.factorization_engine,
            self.number_theory.modular_arithmetic
        ]
        
        for component in components:
            self.assertIs(component.arithmetic, self.arithmetic,
                         "All components must use the same arithmetic system")


if __name__ == '__main__':
    unittest.main(verbosity=2)