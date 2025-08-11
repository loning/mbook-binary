"""
Unit tests for D1-8: φ-Representation System
D1-8：φ-表示系统的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
from formal_system import BinaryEncoding
import math


class PhiRepresentationSystem:
    """φ-表示系统实现"""
    
    def __init__(self):
        # 预计算Fibonacci数列
        self.fibonacci = self._generate_fibonacci(50)
        self.phi = (1 + math.sqrt(5)) / 2
        
    def _generate_fibonacci(self, n):
        """生成修改的Fibonacci数列：F1=1, F2=2, ..."""
        F = [1, 2]  # F[0] = F1, F[1] = F2
        for i in range(2, n):
            F.append(F[i-1] + F[i-2])
        return F
        
    def encode_phi(self, binary_string):
        """将满足no-11约束的二进制串编码为正整数"""
        if not binary_string:
            return 0
            
        result = 0
        for i, bit in enumerate(binary_string[::-1]):  # 从右到左
            if bit == '1':
                result += self.fibonacci[i]
        return result
        
    def decode_phi(self, n):
        """将正整数解码为满足no-11约束的二进制串（贪心算法）"""
        if n == 0:
            return ""
        if n == 1:
            return "1"
            
        # 找到最大的k使得F[k] <= n
        k = 0
        while k < len(self.fibonacci) and self.fibonacci[k] <= n:
            k += 1
        k -= 1  # 回退到最后一个有效的k
        
        # 贪心解码
        result = ['0'] * (k + 1)
        remaining = n
        
        for i in range(k, -1, -1):
            if self.fibonacci[i] <= remaining:
                result[i] = '1'
                remaining -= self.fibonacci[i]
                
        # 转换为字符串并去除前导0
        binary = ''.join(result[::-1]).lstrip('0')
        return binary if binary else "0"
        
    def is_valid_no11(self, binary_string):
        """检查二进制串是否满足no-11约束"""
        return '11' not in binary_string
        
    def information_capacity(self):
        """计算φ-表示系统的信息容量"""
        return math.log2(self.phi)


class TestD1_8_PhiRepresentation(VerificationTest):
    """D1-8 φ-表示系统的形式化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        self.phi_system = PhiRepresentationSystem()
        
    def test_fibonacci_generation(self):
        """测试Fibonacci生成验证 - 验证检查点1"""
        # 验证前几个Fibonacci数
        expected = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        actual = self.phi_system.fibonacci[:10]
        
        self.assertEqual(
            actual, expected,
            "First 10 Fibonacci numbers should match expected sequence"
        )
        
        # 验证递归关系
        for i in range(2, 10):
            self.assertEqual(
                self.phi_system.fibonacci[i],
                self.phi_system.fibonacci[i-1] + self.phi_system.fibonacci[i-2],
                f"F[{i}] should equal F[{i-1}] + F[{i-2}]"
            )
            
    def test_bijection_property(self):
        """测试双射性验证 - 验证检查点2"""
        # 测试范围内的所有正整数
        for n in range(1, 100):
            # 解码到二进制串
            binary = self.phi_system.decode_phi(n)
            
            # 验证满足no-11约束
            self.assertTrue(
                self.phi_system.is_valid_no11(binary),
                f"Decoded string '{binary}' for n={n} should satisfy no-11 constraint"
            )
            
            # 编码回整数
            encoded = self.phi_system.encode_phi(binary)
            
            # 验证双射性：decode(encode(n)) = n
            self.assertEqual(
                encoded, n,
                f"Bijection property failed for n={n}: encoded back to {encoded}"
            )
            
    def test_no11_constraint_preservation(self):
        """测试no-11约束保持验证 - 验证检查点3"""
        # 测试一系列数字的解码
        test_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 50, 100]
        
        for n in test_numbers:
            binary = self.phi_system.decode_phi(n)
            
            # 验证解码结果满足no-11约束
            self.assertTrue(
                self.phi_system.is_valid_no11(binary),
                f"Decoded binary '{binary}' for n={n} should not contain '11'"
            )
            
            # 验证不包含连续的1
            self.assertNotIn(
                '11', binary,
                f"Binary representation of {n} should not contain consecutive 1s"
            )
            
    def test_zeckendorf_uniqueness(self):
        """测试Zeckendorf唯一性验证 - 验证检查点4"""
        # 测试特定的例子
        test_cases = [
            (1, "1"),           # F1 = 1
            (2, "10"),          # F2 = 2
            (3, "100"),         # F3 = 3
            (4, "101"),         # F3 + F1 = 3 + 1
            (5, "1000"),        # F4 = 5
            (6, "1001"),        # F4 + F1 = 5 + 1
            (7, "1010"),        # F4 + F2 = 5 + 2
            (8, "10000"),       # F5 = 8
            (9, "10001"),       # F5 + F1 = 8 + 1
            (10, "10010"),      # F5 + F2 = 8 + 2
        ]
        
        for n, expected_binary in test_cases:
            actual_binary = self.phi_system.decode_phi(n)
            
            # 验证解码的唯一性
            self.assertEqual(
                actual_binary, expected_binary,
                f"Zeckendorf representation of {n} should be '{expected_binary}'"
            )
            
            # 验证编码回原值
            encoded = self.phi_system.encode_phi(actual_binary)
            self.assertEqual(
                encoded, n,
                f"Encoding '{actual_binary}' should give {n}"
            )
            
    def test_encoding_examples(self):
        """测试编码示例"""
        # 基于文档中的示例
        test_cases = [
            ("1", 1),
            ("10", 2),
            ("100", 3),
            ("101", 4),
            ("1000", 5),
            ("1001", 6),
            ("1010", 7),
        ]
        
        for binary, expected_value in test_cases:
            actual_value = self.phi_system.encode_phi(binary)
            self.assertEqual(
                actual_value, expected_value,
                f"encode_phi('{binary}') should equal {expected_value}"
            )
            
    def test_order_preserving(self):
        """测试保序性"""
        # 测试数值顺序与字典序的对应
        numbers = list(range(1, 20))
        binary_representations = [self.phi_system.decode_phi(n) for n in numbers]
        
        # 验证数值递增
        for i in range(len(numbers) - 1):
            self.assertLess(
                numbers[i], numbers[i+1],
                "Numbers should be in increasing order"
            )
            
            # 验证对应的二进制表示也保持某种顺序
            val_i = self.phi_system.encode_phi(binary_representations[i])
            val_i1 = self.phi_system.encode_phi(binary_representations[i+1])
            self.assertLess(
                val_i, val_i1,
                "Encoded values should preserve order"
            )
            
    def test_compactness(self):
        """测试紧致性"""
        # 测试表示长度的紧致性
        for n in [10, 20, 50, 100]:
            binary = self.phi_system.decode_phi(n)
            length = len(binary)
            
            # 理论长度界限
            theoretical_bound = math.floor(math.log(n, self.phi_system.phi)) + 1
            
            # 验证实际长度接近理论界限
            self.assertLessEqual(
                abs(length - theoretical_bound), 1,
                f"Length of φ-representation for {n} should be close to theoretical bound"
            )
            
    def test_information_capacity(self):
        """测试信息容量"""
        # 计算φ-表示的信息容量
        capacity = self.phi_system.information_capacity()
        expected_capacity = math.log2(self.phi_system.phi)
        
        self.assertAlmostEqual(
            capacity, expected_capacity,
            places=10,
            msg="Information capacity should be log2(φ)"
        )
        
        # 验证容量约为0.694
        self.assertAlmostEqual(
            capacity, 0.694,
            places=2,
            msg="Information capacity should be approximately 0.694"
        )
        
    def test_arithmetic_operations(self):
        """测试算术运算"""
        # 测试加法的正确性
        a = 5  # 1000
        b = 7  # 1010
        
        # 解码
        binary_a = self.phi_system.decode_phi(a)
        binary_b = self.phi_system.decode_phi(b)
        
        # 通过数值加法
        sum_value = a + b  # 12
        
        # 解码和的φ-表示
        binary_sum = self.phi_system.decode_phi(sum_value)
        
        # 验证结果满足no-11约束
        self.assertTrue(
            self.phi_system.is_valid_no11(binary_sum),
            f"Sum representation '{binary_sum}' should satisfy no-11 constraint"
        )
        
        # 验证编码的正确性
        self.assertEqual(
            self.phi_system.encode_phi(binary_sum),
            sum_value,
            "Encoded sum should equal numerical sum"
        )
        
    def test_golden_ratio_connection(self):
        """测试与黄金比例的关系"""
        # 验证黄金比例的自指性质
        phi = self.phi_system.phi
        
        # φ = 1 + 1/φ
        self.assertAlmostEqual(
            phi, 1 + 1/phi,
            places=10,
            msg="Golden ratio should satisfy self-referential equation"
        )
        
        # 验证Fibonacci数列的渐近行为
        # 对于修改的Fibonacci数列，我们验证比率收敛到φ
        ratios = []
        for i in range(10, 20):
            ratio = self.phi_system.fibonacci[i] / self.phi_system.fibonacci[i-1]
            ratios.append(ratio)
            
        # 验证比率收敛到黄金比例
        for ratio in ratios[-5:]:  # 检查最后几个比率
            self.assertAlmostEqual(
                ratio, phi,
                places=3,
                msg=f"Fibonacci ratio {ratio} should converge to φ ≈ {phi}"
            )


if __name__ == "__main__":
    unittest.main()