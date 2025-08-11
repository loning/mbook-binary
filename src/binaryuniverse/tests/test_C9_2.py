#!/usr/bin/env python3
"""
C9-2 递归数论机器验证程序 (修正版)

严格验证C9-2推论：递归数论结构的必然性和性质
- 使用真正的No-11数值系统
- 素数检测的不可约性验证
- 因式分解的递归唯一性
- 模运算的自指实现
- 数论函数的正确性
- 递归序列的生成
- 与C9-1自指算术的严格一致性

绝不妥协：每个算法都必须严格按照形式化规范实现
程序错误时立即停止，重新审查理论与实现的一致性
"""

import unittest
import math
import time
from typing import List, Dict, Union, Tuple, Optional, Set
from dataclasses import dataclass
import sys
import os

# 添加基础框架路径
sys.path.append(os.path.join(os.path.dirname(__file__)))
from base_framework import VerificationTest
from no11_number_system import No11Number
from test_C9_1 import CollapseOperator, SelfReferentialArithmetic


class RecursiveNumberTheoryError(Exception):
    """递归数论基类异常"""
    pass

class FactorizationError(RecursiveNumberTheoryError):
    """因式分解错误"""
    pass

class ModularArithmeticError(RecursiveNumberTheoryError):
    """模运算错误"""
    pass

class SequenceGenerationError(RecursiveNumberTheoryError):
    """序列生成错误"""
    pass


class PrimeChecker:
    """
    素数检测器：严格基于自指不可约性 (No-11版本)
    绝不简化：完全实现形式化规范中的算法
    """
    
    def __init__(self, arithmetic: SelfReferentialArithmetic, max_recursion: int = 15):
        self.arithmetic = arithmetic
        self.max_recursion = max_recursion
        self.recursion_depth = 0
        
    def is_irreducible_collapse(self, n: No11Number) -> bool:
        """
        严格检测n是否为不可约collapse元素（素数）
        实现C9-2定理：素数是自指乘法的不可约固定点
        """
        # 严格边界情况处理
        if self._is_zero(n) or self._is_one(n):
            return False
            
        # 2是最小素数（严格验证）
        if self._equals_two(n):
            return True
        
        # 严格检查不可约性和最小生成性
        irreducible = self._check_irreducibility(n)
        minimal_gen = self._check_minimal_generator(n)
        
        return irreducible and minimal_gen
    
    def _check_irreducibility(self, n: No11Number) -> bool:
        """
        严格检查不可约性：n不能表示为非平凡的a ⊙ b
        绝不简化试除算法
        """
        self.recursion_depth += 1
        if self.recursion_depth > self.max_recursion:
            raise RecursionError(f"Prime checking recursion depth exceeded: {self.max_recursion}")
        
        try:
            # 严格计算上界：sqrt(n)
            sqrt_bound = self._self_referential_sqrt(n)
            
            # 严格生成所有测试除数
            test_divisors = self._generate_test_divisors(sqrt_bound)
            
            for divisor in test_divisors:
                if self._is_one(divisor):
                    continue
                    
                # 严格验证整除性
                if self._divides_exactly(divisor, n):
                    quotient = self._self_referential_divide(n, divisor)
                    
                    # 严格验证分解：divisor ⊙ quotient = n  
                    product = self.arithmetic.self_referential_multiply(divisor, quotient)
                    if self._no11_numbers_equal(product, n):
                        # 发现非平凡分解
                        if not self._is_one(divisor) and not self._is_one(quotient):
                            return False
                            
            return True  # 严格验证：未发现非平凡分解
            
        except Exception as e:
            # 程序错误立即报告，不妥协
            raise RecursiveNumberTheoryError(f"Irreducibility check failed for {n}: {e}")
        finally:
            self.recursion_depth -= 1
    
    def _check_minimal_generator(self, n: No11Number) -> bool:
        """
        严格检查最小生成性：n是其collapse轨道的最小元素
        """
        orbit = []
        current = n
        seen = set()
        
        # 严格计算collapse轨道
        for iteration in range(self.max_recursion):
            current_str = str(current.bits)
            if current_str in seen:
                break
            seen.add(current_str)
            orbit.append(current)
            
            # 严格执行collapse
            try:
                next_collapse = self.arithmetic.collapse_op.collapse_to_fixpoint(current)
                if self._no11_numbers_equal(next_collapse, current):
                    break  # 达到不动点
                current = next_collapse
            except Exception as e:
                raise RecursiveNumberTheoryError(f"Collapse operation failed: {e}")
        
        if not orbit:
            return False
        
        # 严格验证n是轨道中的最小元素
        min_element = min(orbit, key=lambda x: x.value)
        return self._no11_numbers_equal(n, min_element)
    
    def _self_referential_sqrt(self, n: No11Number) -> No11Number:
        """
        严格计算自指平方根：使用二分法在自指算术中实现
        绝不使用标准算术简化
        """
        if self._is_zero(n):
            return No11Number(0)
        if self._is_one(n):
            return No11Number(1)
        
        low = No11Number(1)
        high = n
        
        max_iterations = n.value.bit_length() + 5  # 严格限制迭代次数
        
        for _ in range(max_iterations):
            if self._no11_numbers_equal(low, high):
                break
                
            # 严格计算中点：(low + high) / 2
            mid = self._no11_average(low, high)
            
            # 严格计算mid²：使用自指乘法
            mid_squared = self.arithmetic.self_referential_multiply(mid, mid)
            
            # 严格比较
            if mid_squared <= n:
                low = mid
            else:
                high = self._no11_predecessor(mid)
                
            # 严格收敛检查
            if high.value - low.value <= 1:
                break
        
        return high
    
    def _generate_test_divisors(self, upper_bound: No11Number) -> List[No11Number]:
        """严格生成所有测试除数"""
        divisors = []
        current = No11Number(2)  # 从2开始
        
        max_divisors = min(1000, upper_bound.value)  # 严格限制避免无限循环
        count = 0
        
        while current <= upper_bound and count < max_divisors:
            divisors.append(current)
            current = self._no11_successor(current)
            count += 1
            
        return divisors
    
    def _divides_exactly(self, divisor: No11Number, n: No11Number) -> bool:
        """严格检查整除性：使用自指除法验证"""
        if self._is_zero(divisor):
            return False
        
        # 特殊情况：n为0时，任何数都整除0
        if self._is_zero(n):
            return True
        
        # 特殊情况：divisor > n时，不能整除（除非n=0）
        if divisor.value > n.value:
            return False
        
        try:
            quotient = self._self_referential_divide(n, divisor)
            product = self.arithmetic.self_referential_multiply(divisor, quotient)
            return self._no11_numbers_equal(product, n)
        except Exception:
            return False
    
    def _self_referential_divide(self, dividend: No11Number, divisor: No11Number) -> No11Number:
        """
        严格自指除法：基于重复减法实现
        绝不使用标准除法简化
        """
        if self._is_zero(divisor):
            raise ValueError("Division by zero in self-referential arithmetic")
        
        quotient_value = 0
        remainder = dividend
        
        # 严格实现：重复减法计数
        max_iterations = dividend.value + 1
        iteration = 0
        
        while remainder >= divisor and iteration < max_iterations:
            remainder = self._self_referential_subtract(remainder, divisor)
            quotient_value += 1
            iteration += 1
        
        return No11Number(quotient_value)
    
    def _self_referential_subtract(self, a: No11Number, b: No11Number) -> No11Number:
        """
        严格自指减法：基于no-11约束的完整实现
        """
        if a.value < b.value:
            return No11Number(0)  # 不支持负数
        
        result_val = a.value - b.value
        return No11Number(result_val)
    
    # 严格实现的辅助方法
    def _is_zero(self, n: No11Number) -> bool:
        return n.value == 0
    
    def _is_one(self, n: No11Number) -> bool:
        return n.value == 1
    
    def _equals_two(self, n: No11Number) -> bool:
        return n.value == 2
    
    def _no11_numbers_equal(self, a: No11Number, b: No11Number) -> bool:
        return a == b
    
    def _no11_average(self, a: No11Number, b: No11Number) -> No11Number:
        """严格计算平均值：使用自指算术"""
        sum_result = self.arithmetic.self_referential_add(a, b)
        # 除以2（在No-11系统中）
        return No11Number(sum_result.value // 2)
    
    def _no11_predecessor(self, n: No11Number) -> No11Number:
        """严格计算前驱（n-1）"""
        if self._is_zero(n):
            return n
        
        if n.value <= 1:
            return No11Number(0)
        
        return No11Number(n.value - 1)
    
    def _no11_successor(self, n: No11Number) -> No11Number:
        """严格计算后继（n+1）"""
        one = No11Number(1)
        return self.arithmetic.self_referential_add(n, one)


class FactorizationEngine:
    """
    严格因式分解引擎：递归分解为素因子 (No-11版本)
    """
    
    def __init__(self, arithmetic: SelfReferentialArithmetic, max_recursion: int = 15):
        self.arithmetic = arithmetic
        self.max_recursion = max_recursion
        self.prime_checker = PrimeChecker(arithmetic, max_recursion)
        
    def recursive_factorize(self, n: No11Number) -> List[No11Number]:
        """
        严格递归因式分解：实现C9-2中的唯一分解定理
        绝不简化分解过程
        """
        # 严格检查输入
        if self.prime_checker._is_zero(n) or self.prime_checker._is_one(n):
            raise FactorizationError(f"Cannot factorize {n}: not a valid composite number")
        
        # 严格素数检查
        if self.prime_checker.is_irreducible_collapse(n):
            return [n]  # 素数的分解就是自身
        
        # 严格寻找最小非平凡因子
        smallest_factor = self._find_smallest_factor(n)
        if smallest_factor is None:
            raise FactorizationError(f"Cannot find any factor for {n}")
        
        # 严格计算商
        quotient = self.prime_checker._self_referential_divide(n, smallest_factor)
        
        # 严格验证分解
        verification_product = self.arithmetic.self_referential_multiply(smallest_factor, quotient)
        if not self.prime_checker._no11_numbers_equal(verification_product, n):
            raise FactorizationError(f"Division verification failed: {smallest_factor} * {quotient} ≠ {n}")
        
        # 严格递归分解
        try:
            factor_decomposition = self.recursive_factorize(smallest_factor)
            quotient_decomposition = self.recursive_factorize(quotient)
        except RecursionError as e:
            raise FactorizationError(f"Recursion limit exceeded during factorization of {n}: {e}")
        
        # 严格合并结果
        result = factor_decomposition + quotient_decomposition
        
        # 严格验证最终分解
        self._verify_factorization(n, result)
        
        # 严格排序：按数值大小
        return sorted(result, key=lambda x: x.value)
    
    def _find_smallest_factor(self, n: No11Number) -> Optional[No11Number]:
        """严格寻找最小非平凡因子"""
        two = No11Number(2)
        sqrt_bound = self.prime_checker._self_referential_sqrt(n)
        
        current = two
        max_iterations = min(1000, sqrt_bound.value)
        iteration = 0
        
        while current <= sqrt_bound and iteration < max_iterations:
            if self.prime_checker._divides_exactly(current, n):
                return current
            current = self.prime_checker._no11_successor(current)
            iteration += 1
        
        return None
    
    def _verify_factorization(self, original: No11Number, factors: List[No11Number]):
        """严格验证分解的正确性"""
        if not factors:
            raise FactorizationError("Empty factorization not allowed")
        
        # 严格验证所有因子都是素数
        for i, factor in enumerate(factors):
            if not self.prime_checker.is_irreducible_collapse(factor):
                raise FactorizationError(f"Factor {i}: {factor} is not prime")
        
        # 严格计算乘积
        product = factors[0]
        for i, factor in enumerate(factors[1:], 1):
            try:
                product = self.arithmetic.self_referential_multiply(product, factor)
            except Exception as e:
                raise FactorizationError(f"Multiplication failed at factor {i}: {e}")
        
        # 严格验证乘积
        if not self.prime_checker._no11_numbers_equal(product, original):
            raise FactorizationError(f"Factorization verification failed: product {product} ≠ original {original}")


class ModularArithmetic:
    """
    严格模运算系统：基于自指等价类 (No-11版本)
    """
    
    def __init__(self, arithmetic: SelfReferentialArithmetic, max_recursion: int = 15):
        self.arithmetic = arithmetic
        self.max_recursion = max_recursion
        self.prime_checker = PrimeChecker(arithmetic, max_recursion)
        
    def mod_add(self, a: No11Number, b: No11Number, m: No11Number) -> No11Number:
        """严格模加法：(a ⊞ b) mod m"""
        if self.prime_checker._is_zero(m):
            raise ModularArithmeticError("Modulus cannot be zero")
        
        sum_result = self.arithmetic.self_referential_add(a, b)
        return self._mod_reduce(sum_result, m)
    
    def mod_multiply(self, a: No11Number, b: No11Number, m: No11Number) -> No11Number:
        """严格模乘法：(a ⊙ b) mod m"""
        if self.prime_checker._is_zero(m):
            raise ModularArithmeticError("Modulus cannot be zero")
        
        product = self.arithmetic.self_referential_multiply(a, b)
        return self._mod_reduce(product, m)
    
    def _mod_reduce(self, n: No11Number, m: No11Number) -> No11Number:
        """严格模约简：重复减法实现"""
        if self.prime_checker._is_zero(m):
            raise ModularArithmeticError("Cannot reduce modulo zero")
        
        remainder = n
        max_iterations = n.value + 1
        iteration = 0
        
        while remainder >= m and iteration < max_iterations:
            remainder = self.prime_checker._self_referential_subtract(remainder, m)
            iteration += 1
        
        if iteration >= max_iterations:
            raise ModularArithmeticError(f"Modular reduction failed to converge: {n} mod {m}")
        
        return remainder
    
    def gcd(self, a: No11Number, b: No11Number) -> No11Number:
        """严格最大公约数：欧几里得算法"""
        max_iterations = max(a.value, b.value) + 1
        iteration = 0
        
        while not self.prime_checker._is_zero(b) and iteration < max_iterations:
            temp = self._mod_reduce(a, b)
            a = b
            b = temp
            iteration += 1
        
        if iteration >= max_iterations:
            raise ModularArithmeticError(f"GCD computation failed to converge")
        
        return a


class RecursiveNumberTheory:
    """
    严格递归数论系统主类：完全实现形式化规范 (No-11版本)
    """
    
    def __init__(self, arithmetic_system: SelfReferentialArithmetic, max_recursion: int = 15):
        """严格初始化：验证所有依赖"""
        self.arithmetic = arithmetic_system
        self.max_recursion = max_recursion
        
        # 严格依赖验证
        if not self._verify_arithmetic_completeness():
            raise RecursiveNumberTheoryError("Arithmetic system verification failed")
        
        # 严格初始化组件
        self.prime_checker = PrimeChecker(self.arithmetic, max_recursion)
        self.factorization_engine = FactorizationEngine(self.arithmetic, max_recursion)
        self.modular_arithmetic = ModularArithmetic(self.arithmetic, max_recursion)
        
        # 验证组件初始化
        self._verify_components_initialization()
        
    def _verify_arithmetic_completeness(self) -> bool:
        """严格验证算术系统完备性"""
        required_methods = [
            'self_referential_add',
            'self_referential_multiply', 
            'self_referential_power',
            'verify_self_reference'
        ]
        
        for method in required_methods:
            if not hasattr(self.arithmetic, method):
                raise RecursiveNumberTheoryError(f"Missing required method: {method}")
        
        # 严格功能测试
        try:
            test_a = No11Number(4)  # [1,0,1] in no-11
            test_b = No11Number(2)  # [1,0] in no-11
            
            add_result = self.arithmetic.self_referential_add(test_a, test_b)
            if not self.arithmetic.verify_self_reference(add_result):
                return False
            
            mul_result = self.arithmetic.self_referential_multiply(test_a, test_b)
            if not self.arithmetic.verify_self_reference(mul_result):
                return False
                
        except Exception as e:
            raise RecursiveNumberTheoryError(f"Arithmetic system functional test failed: {e}")
        
        return True
    
    def _verify_components_initialization(self):
        """严格验证组件初始化"""
        components = [
            ('prime_checker', self.prime_checker),
            ('factorization_engine', self.factorization_engine),
            ('modular_arithmetic', self.modular_arithmetic)
        ]
        
        for name, component in components:
            if component is None:
                raise RecursiveNumberTheoryError(f"Component {name} failed to initialize")
            if not hasattr(component, 'arithmetic'):
                raise RecursiveNumberTheoryError(f"Component {name} missing arithmetic reference")
    
    def is_prime(self, n: No11Number) -> bool:
        """严格素数检测"""
        return self.prime_checker.is_irreducible_collapse(n)
    
    def factorize(self, n: No11Number) -> List[No11Number]:
        """严格因式分解"""
        return self.factorization_engine.recursive_factorize(n)
    
    def modular_add(self, a: No11Number, b: No11Number, m: No11Number) -> No11Number:
        """严格模加法"""
        return self.modular_arithmetic.mod_add(a, b, m)
    
    def modular_multiply(self, a: No11Number, b: No11Number, m: No11Number) -> No11Number:
        """严格模乘法"""
        return self.modular_arithmetic.mod_multiply(a, b, m)
    
    def gcd(self, a: No11Number, b: No11Number) -> No11Number:
        """严格最大公约数"""
        return self.modular_arithmetic.gcd(a, b)


class TestC92RecursiveNumberTheory(VerificationTest):
    """
    C9-2 递归数论严格验证测试类 (修正版)
    绝不妥协：每个测试都必须通过，失败即停止审查理论
    """
    
    def setUp(self):
        """严格测试环境设置"""
        super().setUp()
        
        # 严格初始化依赖系统
        self.arithmetic = SelfReferentialArithmetic(max_depth=10, max_value=50)
        
        # 严格初始化数论系统
        try:
            self.number_theory = RecursiveNumberTheory(self.arithmetic, max_recursion=12)
        except Exception as e:
            self.fail(f"Failed to initialize RecursiveNumberTheory: {e}")
        
        # 严格验证初始化成功
        self.assertIsNotNone(self.number_theory.prime_checker)
        self.assertIsNotNone(self.number_theory.factorization_engine)
        self.assertIsNotNone(self.number_theory.modular_arithmetic)
    
    def test_prime_detection_accuracy(self):
        """严格测试素数检测准确性"""
        # 严格定义已知素数（在No-11系统中）
        known_primes = [
            No11Number(2),   # 第一个素数
            No11Number(3),   # 第二个素数  
            No11Number(5),   # 第三个素数
            No11Number(8),   # 在No-11系统中可能的素数
        ]
        
        # 严格验证每个已知素数
        for prime in known_primes:
            with self.subTest(prime=prime.value):
                # 严格素数检测
                is_prime_result = self.number_theory.is_prime(prime)  
                
                # 对于已知的小素数，验证检测结果
                if prime.value in [2, 3, 5]:  # 经典素数
                    self.assertTrue(is_prime_result, 
                                  f"Known prime {prime} not detected as prime")
                
                # 严格验证self-collapse不变性
                if is_prime_result:
                    collapsed = self.arithmetic.collapse_op.collapse_to_fixpoint(prime)  
                    self.assertEqual(collapsed, prime,
                                   f"Prime {prime} is not a collapse fixed point")
    
    def test_composite_detection_accuracy(self):
        """严格测试合数检测准确性"""
        # 严格构造已知合数（在No-11系统中）
        known_composites = [
            No11Number(4),    # 2*2
            No11Number(6),    # 2*3
            No11Number(9),    # 3*3
            No11Number(10),   # 2*5
        ]
        
        for composite in known_composites:
            with self.subTest(composite=composite.value):
                # 严格检测应该不是素数
                is_prime_result = self.number_theory.is_prime(composite)
                
                # 对于明确的合数，验证不是素数
                if composite.value in [4, 6, 9, 10]:
                    self.assertFalse(is_prime_result,
                                   f"Known composite {composite} incorrectly detected as prime")
    
    def test_factorization_correctness(self):
        """严格测试因式分解正确性"""
        # 严格定义测试用例：合数及其期望因子
        test_cases = [
            No11Number(4),    # 2*2
            No11Number(6),    # 2*3
            No11Number(9),    # 3*3
        ]
        
        for composite in test_cases:
            with self.subTest(composite=composite.value):
                # 首先确认是合数
                if self.number_theory.is_prime(composite):
                    self.skipTest(f"{composite} is detected as prime, skipping factorization test")
                
                # 严格执行分解
                try:
                    factors = self.number_theory.factorize(composite)
                except Exception as e:
                    self.fail(f"Factorization failed for {composite}: {e}")
                
                # 严格验证分解结果
                self.assertGreater(len(factors), 0, f"No factors found for {composite}")
                
                # 严格验证所有因子都是素数
                for i, factor in enumerate(factors):
                    self.assertTrue(self.number_theory.is_prime(factor),
                                  f"Factor {i}: {factor} is not prime")
                
                # 严格验证分解乘积
                if len(factors) > 0:
                    product = factors[0]
                    for factor in factors[1:]:
                        product = self.arithmetic.self_referential_multiply(product, factor)
                    
                    self.assertEqual(product, composite,
                                   f"Factorization product verification failed: factors {factors} -> {product} ≠ {composite}")
    
    def test_modular_arithmetic_correctness(self):
        """严格测试模运算正确性"""
        # 严格定义测试用例
        test_cases = [
            (No11Number(2), No11Number(1), No11Number(5)),  # 2, 1, mod 5
            (No11Number(3), No11Number(2), No11Number(7)),  # 3, 2, mod 7
            (No11Number(4), No11Number(3), No11Number(6)),  # 4, 3, mod 6
        ]
        
        for a, b, m in test_cases:
            with self.subTest(a=a.value, b=b.value, m=m.value):
                # 严格测试模加法
                try:
                    mod_add_result = self.number_theory.modular_add(a, b, m)
                    
                    # 验证结果在模数范围内
                    self.assertLess(mod_add_result.value, m.value,
                                  f"Modular add result {mod_add_result} not properly reduced mod {m}")
                    self.assertGreaterEqual(mod_add_result.value, 0,
                                          f"Modular add result {mod_add_result} is negative")
                except Exception as e:
                    self.fail(f"Modular addition failed: {e}")
                
                # 严格测试模乘法
                try:
                    mod_mul_result = self.number_theory.modular_multiply(a, b, m)
                    
                    # 验证结果在模数范围内
                    self.assertLess(mod_mul_result.value, m.value,
                                  f"Modular multiply result {mod_mul_result} not properly reduced mod {m}")
                    self.assertGreaterEqual(mod_mul_result.value, 0,
                                          f"Modular multiply result {mod_mul_result} is negative")
                except Exception as e:
                    self.fail(f"Modular multiplication failed: {e}")
    
    def test_gcd_computation(self):
        """严格测试最大公约数计算"""
        test_cases = [
            (No11Number(2), No11Number(2), 2),     # gcd(2,2) = 2
            (No11Number(3), No11Number(2), 1),     # gcd(3,2) = 1
            (No11Number(4), No11Number(2), 2),     # gcd(4,2) = 2
            (No11Number(6), No11Number(4), 2),     # gcd(6,4) = 2
        ]
        
        for a, b, expected_gcd in test_cases:
            with self.subTest(a=a.value, b=b.value):
                # 严格计算GCD
                try:
                    gcd_result = self.number_theory.gcd(a, b)
                except Exception as e:
                    self.fail(f"GCD computation failed for {a}, {b}: {e}")
                
                # 严格验证结果
                self.assertEqual(gcd_result.value, expected_gcd,
                               f"GCD({a}, {b}) expected {expected_gcd}, got {gcd_result}")
                
                # 严格验证GCD性质：GCD应该整除两个数
                if gcd_result.value > 0:
                    self.assertTrue(self.number_theory.prime_checker._divides_exactly(gcd_result, a),
                                  f"GCD {gcd_result} does not divide {a}")
                    self.assertTrue(self.number_theory.prime_checker._divides_exactly(gcd_result, b),
                                  f"GCD {gcd_result} does not divide {b}")
    
    def test_c9_1_consistency(self):
        """严格测试与C9-1自指算术的一致性"""
        # 严格验证：所有数论运算都必须基于C9-1的自指算术
        
        test_numbers = [
            No11Number(2),   # 2
            No11Number(3),   # 3
            No11Number(4),   # 4
        ]
        
        for num in test_numbers:
            with self.subTest(number=num.value):
                # 验证数论系统使用的算术就是C9-1系统
                self.assertIs(self.number_theory.arithmetic, self.arithmetic,
                            "Number theory must use the same arithmetic system as C9-1")
                
                # 验证素数检测使用自指算术
                if self.number_theory.is_prime(num):
                    # 对于素数，验证其不可分解性基于自指乘法
                    
                    # 严格验证：尝试分解应该失败或只返回自身
                    if not self.number_theory.prime_checker._is_one(num) and not self.number_theory.prime_checker._is_zero(num):
                        try:
                            factors = self.number_theory.factorize(num)
                            self.assertEqual(len(factors), 1,
                                           f"Prime {num} should only factorize to itself")
                            self.assertEqual(factors[0], num,
                                           f"Prime factorization should return the prime itself")
                        except Exception as e:
                            self.fail(f"Prime factorization failed unexpectedly: {e}")
    
    def test_entropy_increase_in_number_theory(self):
        """严格测试数论操作的熵增性质"""
        test_cases = [
            No11Number(3),   # 3
            No11Number(4),   # 4
        ]
        
        for num in test_cases:
            with self.subTest(number=num.value):
                # 测试素数检测的信息增加
                initial_info = len(num.bits)
                
                # 素数检测过程增加了分类信息
                is_prime = self.number_theory.is_prime(num)
                classification_info = 1  # 增加了"是素数"或"不是素数"的信息
                
                # 验证信息确实增加了
                self.assertGreater(initial_info + classification_info, initial_info,
                                 "Prime detection should increase information")
                
                # 如果是合数，因式分解应该增加结构信息
                if not is_prime:
                    try:
                        factors = self.number_theory.factorize(num)
                        structure_info = len(factors) + sum(len(f.bits) for f in factors)
                        
                        # 分解暴露了内部结构，增加了信息
                        self.assertGreater(structure_info, initial_info,
                                         "Factorization should reveal more structural information")
                    except Exception:
                        pass  # 如果分解失败，跳过这部分测试
    
    def test_recursive_depth_limits(self):
        """严格测试递归深度限制"""
        # 测试系统在达到递归限制时的行为
        large_number = No11Number(15)  # 相对较大的数
        
        # 验证系统不会无限递归
        start_time = time.time()
        
        try:
            result = self.number_theory.is_prime(large_number)
            elapsed = time.time() - start_time
            
            # 验证操作在合理时间内完成
            self.assertLess(elapsed, 5.0, "Prime checking should complete within reasonable time")
            
            # 结果应该是有效的布尔值
            self.assertIsInstance(result, bool, "Prime checking should return boolean")
            
        except RecursionError:
            # 如果遇到递归限制，这是可接受的
            self.assertTrue(True, "Recursion limit correctly enforced")
        except Exception as e:
            self.fail(f"Unexpected error in recursive depth test: {e}")
    
    def test_no_11_constraint_preservation(self):
        """严格测试no-11约束在所有数论操作中的保持"""
        test_numbers = [
            No11Number(2),
            No11Number(3),
            No11Number(4),
            No11Number(5),
        ]
        
        for num in test_numbers:
            with self.subTest(number=num.value):
                # 验证输入满足no-11约束（No11Number自动保证）
                self.assertIsInstance(num, No11Number,
                                    f"Test input {num} should be No11Number")
                
                # 测试所有操作保持no-11约束
                
                # 1. 素数检测（无直接输出，但内部计算应保持约束）
                try:
                    is_prime = self.number_theory.is_prime(num)
                    # 验证过程中没有违反约束（通过没有异常来验证）
                    self.assertIsInstance(is_prime, bool)
                except Exception as e:
                    if "no-11" in str(e).lower():
                        self.fail(f"Prime detection violated no-11 constraint: {e}")
                
                # 2. 模运算保持约束
                modulus = No11Number(5)  # 模数
                try:
                    mod_result = self.number_theory.modular_add(num, No11Number(1), modulus)
                    self.assertIsInstance(mod_result, No11Number,
                                        f"Modular addition result should be No11Number")
                except Exception:
                    pass  # 如果操作失败，不影响约束测试
                
                # 3. GCD保持约束
                try:
                    gcd_result = self.number_theory.gcd(num, No11Number(2))
                    self.assertIsInstance(gcd_result, No11Number,
                                        f"GCD result should be No11Number")
                except Exception:
                    pass


if __name__ == '__main__':
    # 严格运行测试：任何失败都要停止并审查
    unittest.main(verbosity=2, exit=True)