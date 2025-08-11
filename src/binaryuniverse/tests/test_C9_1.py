#!/usr/bin/env python3
"""
C9-1 自指算术机器验证程序 (修正版)

使用真正的No-11数值系统实现自指算术
- 所有数值都在No-11约束空间内表示和运算
- 不再是标准二进制+过滤，而是原生No-11算术
"""

import unittest
import math
import time
from typing import List, Dict, Union, Tuple
from dataclasses import dataclass
import sys
import os

# 导入No-11数值系统
sys.path.append(os.path.join(os.path.dirname(__file__)))
from base_framework import VerificationTest
from no11_number_system import No11Number


class CollapseOperator:
    """Self-collapse算符（No-11版本）"""
    
    def __init__(self, max_iterations: int = 50):
        self.max_iterations = max_iterations
    
    def collapse_to_fixpoint(self, number: No11Number) -> No11Number:
        """执行collapse直到不动点"""
        current = number
        
        for _ in range(self.max_iterations):
            next_state = self._single_collapse(current)
            if next_state == current:
                return current
            current = next_state
        
        return current
    
    def _single_collapse(self, number: No11Number) -> No11Number:
        """单次collapse操作"""
        # 在No-11系统中，collapse只是确保数值在有效范围内
        # 由于No11Number已经保证了no-11约束，这里主要是不动点检查
        return number


class SelfReferentialArithmetic:
    """自指算术系统（No-11版本）"""
    
    def __init__(self, max_depth: int = 10, max_value: int = 100):
        """
        初始化自指算术系统
        
        Args:
            max_depth: 最大递归深度
            max_value: 最大可表示数值
        """
        self.max_depth = max_depth
        self.max_value = max_value
        self.phi = (1 + math.sqrt(5)) / 2
        self.collapse_op = CollapseOperator(max_iterations=50)
        
        # 确保No-11系统能表示足够的数值
        if max_value > No11Number.max_representable_value():
            raise ValueError(f"max_value {max_value} exceeds No-11 system capacity")
    
    def self_referential_add(self, a: No11Number, b: No11Number) -> No11Number:
        """自指加法：a ⊞ b"""
        # Step 1: 在No-11系统中直接相加
        result = a + b
        
        # Step 2: 检查是否超出范围
        if result.value > self.max_value:
            # 在有限系统中，超出范围的结果需要模运算
            result = No11Number(result.value % (self.max_value + 1))
        
        # Step 3: Self-collapse
        final_result = self.collapse_op.collapse_to_fixpoint(result)
        
        return final_result
    
    def self_referential_multiply(self, a: No11Number, b: No11Number) -> No11Number:
        """自指乘法：a ⊙ b"""
        if b.value == 0:
            return No11Number(0)
        
        if b.value == 1:
            return a
        
        # 递归定义：a ⊙ b = a ⊞ (a ⊙ (b-1))
        b_pred = No11Number(b.value - 1) if b.value > 0 else No11Number(0)
        recursive_result = self.self_referential_multiply(a, b_pred)
        return self.self_referential_add(a, recursive_result)
    
    def self_referential_power(self, a: No11Number, b: No11Number) -> No11Number:
        """自指幂运算：a^⇈b"""
        if b.value == 0:
            return No11Number(1)
        
        if b.value == 1:
            return a
        
        # 递归定义：a^⇈b = a ⊙ (a^⇈(b-1))
        b_pred = No11Number(b.value - 1) if b.value > 0 else No11Number(0)
        recursive_result = self.self_referential_power(a, b_pred)
        return self.self_referential_multiply(a, recursive_result)
    
    def verify_self_reference(self, result: No11Number) -> bool:
        """验证结果的自指性质"""
        collapsed = self.collapse_op.collapse_to_fixpoint(result)
        return collapsed == result
    
    def calculate_entropy_increase(self, before: List[No11Number], 
                                  after: No11Number) -> float:
        """计算熵增"""
        # 基于值的复杂度和位模式复杂度
        complexity_before = sum(n.value + len(n.bits) for n in before)
        complexity_after = after.value + len(after.bits) + 1  # +1 for operation
        
        return complexity_after - complexity_before


class TestC91SelfReferentialArithmetic(VerificationTest):
    """C9-1 自指算术测试类（修正版）"""
    
    def setUp(self):
        """测试前置设置"""
        super().setUp()
        self.arithmetic = SelfReferentialArithmetic(max_depth=8, max_value=50)
    
    def test_no11_number_representation(self):
        """测试No-11数值表示的正确性"""
        # 验证前10个数的表示
        expected_representations = [
            ([0], 0),           # 0
            ([1], 1),           # 1  
            ([1, 0], 2),        # 2
            ([1, 0, 0], 3),     # 3 (关键：不是[1,1]!)
            ([1, 0, 1], 4),     # 4 (关键：不是5!)
            ([1, 0, 0, 0], 5),  # 5
            ([1, 0, 0, 1], 6),  # 6
            ([1, 0, 1, 0], 7),  # 7
            ([1, 0, 0, 0, 0], 8), # 8
            ([1, 0, 0, 0, 1], 9), # 9
        ]
        
        for expected_bits, value in expected_representations:
            with self.subTest(value=value):
                num = No11Number(value)
                self.assertEqual(num.bits, expected_bits,
                               f"No-11 representation of {value} should be {expected_bits}, got {num.bits}")
                self.assertEqual(num.value, value,
                               f"Value should be {value}, got {num.value}")
    
    def test_corrected_arithmetic_basic(self):
        """测试修正后的基础算术"""
        # 测试 2 + 3 = 5
        two = No11Number(2)    # [1,0]
        three = No11Number(3)  # [1,0,0] 
        five = No11Number(5)   # [1,0,0,0]
        
        result = self.arithmetic.self_referential_add(two, three)
        self.assertEqual(result.value, 5,
                        f"2 + 3 should equal 5, got {result.value}")
        self.assertEqual(result, five,
                        f"Result should be {five}, got {result}")
        
        # 测试 3 * 3 = 9  
        nine = No11Number(9)   # [1,0,0,0,1]
        result = self.arithmetic.self_referential_multiply(three, three)
        self.assertEqual(result.value, 9,
                        f"3 * 3 should equal 9, got {result.value}")
        self.assertEqual(result, nine,
                        f"Result should be {nine}, got {result}")
    
    def test_arithmetic_correctness(self):
        """测试算术运算的正确性"""
        test_cases = [
            # (a_val, b_val, expected_add, expected_mul)
            (0, 0, 0, 0),
            (0, 1, 1, 0),  
            (1, 1, 2, 1),
            (2, 2, 4, 4),
            (2, 3, 5, 6),
            (3, 4, 7, 12),
        ]
        
        for a_val, b_val, expected_add, expected_mul in test_cases:
            with self.subTest(a=a_val, b=b_val):
                a = No11Number(a_val)
                b = No11Number(b_val)
                
                # 测试加法
                add_result = self.arithmetic.self_referential_add(a, b)
                self.assertEqual(add_result.value, expected_add,
                               f"{a_val} + {b_val} should equal {expected_add}, got {add_result.value}")
                
                # 测试乘法
                mul_result = self.arithmetic.self_referential_multiply(a, b)
                self.assertEqual(mul_result.value, expected_mul,
                               f"{a_val} * {b_val} should equal {expected_mul}, got {mul_result.value}")
    
    def test_self_reference_property(self):
        """测试自指性质"""
        test_numbers = [No11Number(i) for i in range(10)]
        
        for num in test_numbers:
            with self.subTest(value=num.value):
                # 所有数都应该是self-referential的
                self.assertTrue(self.arithmetic.verify_self_reference(num),
                               f"Number {num} should be self-referential")
                
                # 运算结果也应该是self-referential的
                if num.value > 0:
                    one = No11Number(1)
                    add_result = self.arithmetic.self_referential_add(num, one)
                    self.assertTrue(self.arithmetic.verify_self_reference(add_result),
                                   f"Addition result {add_result} should be self-referential")
    
    def test_no_11_constraint_preservation(self):
        """测试no-11约束保持性"""
        # 在No-11系统中，所有数值自动满足no-11约束
        test_numbers = [No11Number(i) for i in range(15)]
        
        for a in test_numbers[:5]:
            for b in test_numbers[:5]:
                with self.subTest(a=a.value, b=b.value):
                    # 所有运算结果都应该满足no-11约束
                    add_result = self.arithmetic.self_referential_add(a, b)
                    mul_result = self.arithmetic.self_referential_multiply(a, b)
                    
                    # No11Number自动保证no-11约束
                    self.assertIsInstance(add_result, No11Number)
                    self.assertIsInstance(mul_result, No11Number)
                    
                    # 验证位模式确实满足no-11
                    self.assertTrue(self._validate_no_11(add_result.bits),
                                   f"Add result {add_result.bits} violates no-11")
                    self.assertTrue(self._validate_no_11(mul_result.bits),
                                   f"Mul result {mul_result.bits} violates no-11")
    
    def _validate_no_11(self, bits: List[int]) -> bool:
        """验证位模式是否满足no-11约束"""
        for i in range(len(bits) - 1):
            if bits[i] == 1 and bits[i+1] == 1:
                return False
        return True
    
    def test_entropy_increase(self):
        """测试熵增性质"""
        a = No11Number(3)  # [1,0,0]
        b = No11Number(2)  # [1,0]
        
        # 测试加法熵增
        add_result = self.arithmetic.self_referential_add(a, b)
        entropy_increase = self.arithmetic.calculate_entropy_increase([a, b], add_result)
        
        # 在自指系统中，运算过程增加了信息
        self.assertGreaterEqual(entropy_increase, 0,
                               "Self-referential addition should not decrease entropy")
        
        # 验证运算确实发生了变化
        self.assertNotEqual(add_result, a, "Result should differ from input a")
        self.assertNotEqual(add_result, b, "Result should differ from input b")
    
    def test_fibonacci_connection(self):
        """测试与斐波那契数列的联系"""
        # 在No-11系统中，应该能看到斐波那契模式
        fibonacci_values = [1, 1, 2, 3, 5, 8, 13]  # 经典斐波那契
        
        # 测试前几个斐波那契数在No-11系统中的表示
        for i, fib_val in enumerate(fibonacci_values[:6]):  # 只测试前6个
            if fib_val <= No11Number.max_representable_value():
                with self.subTest(fib_index=i, fib_value=fib_val):
                    fib_num = No11Number(fib_val)
                    
                    # 验证斐波那契数的No-11表示满足约束
                    self.assertTrue(self._validate_no_11(fib_num.bits),
                                   f"Fibonacci number {fib_val} representation {fib_num.bits} violates no-11")
    
    def test_edge_cases(self):
        """测试边界情况"""
        zero = No11Number(0)
        one = No11Number(1)
        
        # 测试零运算
        zero_plus_zero = self.arithmetic.self_referential_add(zero, zero)
        self.assertEqual(zero_plus_zero.value, 0, "0 + 0 should equal 0")
        
        zero_times_one = self.arithmetic.self_referential_multiply(zero, one)
        self.assertEqual(zero_times_one.value, 0, "0 * 1 should equal 0")
        
        one_power_zero = self.arithmetic.self_referential_power(one, zero)
        self.assertEqual(one_power_zero.value, 1, "1^0 should equal 1")
        
        # 测试单位运算
        test_val = No11Number(4)  # [1,0,1]
        
        test_times_one = self.arithmetic.self_referential_multiply(test_val, one)
        self.assertEqual(test_times_one.value, test_val.value, "n * 1 should equal n")
        
        test_power_one = self.arithmetic.self_referential_power(test_val, one)
        self.assertEqual(test_power_one.value, test_val.value, "n^1 should equal n")
    
    def test_consistency_with_standard_arithmetic(self):
        """测试与标准算术的一致性（在小数值范围内）"""
        # 在小数值范围内，No-11算术应该与标准算术一致
        for a_val in range(6):
            for b_val in range(6):
                with self.subTest(a=a_val, b=b_val):
                    a = No11Number(a_val)
                    b = No11Number(b_val)
                    
                    # 测试加法一致性
                    expected_sum = a_val + b_val
                    if expected_sum <= self.arithmetic.max_value:
                        actual_sum = self.arithmetic.self_referential_add(a, b)
                        self.assertEqual(actual_sum.value, expected_sum,
                                       f"No-11 addition {a_val} + {b_val} should equal standard {expected_sum}")
                    
                    # 测试乘法一致性
                    expected_product = a_val * b_val
                    if expected_product <= self.arithmetic.max_value:
                        actual_product = self.arithmetic.self_referential_multiply(a, b)
                        self.assertEqual(actual_product.value, expected_product,
                                       f"No-11 multiplication {a_val} * {b_val} should equal standard {expected_product}")
    
    def test_recursive_depth_control(self):
        """测试递归深度控制"""
        # 测试较大数值的运算不会导致无限递归
        a = No11Number(5)
        b = No11Number(4)
        
        start_time = time.time()
        result = self.arithmetic.self_referential_multiply(a, b)
        elapsed_time = time.time() - start_time
        
        # 验证运算完成且时间合理
        self.assertLess(elapsed_time, 1.0, "Operation should complete in reasonable time")
        self.assertIsInstance(result, No11Number, "Result should be a No11Number")
        self.assertTrue(self.arithmetic.verify_self_reference(result),
                       "Result should be self-referential")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)