"""
Unit tests for L1-6: φ-Representation System Establishment Lemma
L1-6：φ-表示系统建立引理的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
import math


class PhiRepresentationSystem:
    """φ-表示系统实现"""
    
    def __init__(self):
        # 修正的Fibonacci数列 F(1)=1, F(2)=2, F(3)=3, F(4)=5, ...
        self.fib = self._generate_modified_fibonacci(100)
        
    def _generate_modified_fibonacci(self, max_terms):
        """生成修正的Fibonacci数列"""
        # F(1) = 1, F(2) = 2, 然后正常递归
        fib = [0, 1, 2]  # fib[0]未使用，fib[1]=1, fib[2]=2
        
        for i in range(3, max_terms):
            fib.append(fib[-1] + fib[-2])
            
        return fib
        
    def greedy_decomposition(self, n):
        """贪心算法计算φ-表示"""
        if n <= 0:
            return []
            
        result = []
        remaining = n
        
        # 找到不超过n的最大Fibonacci数的索引
        i = len(self.fib) - 1
        while i >= 1:
            if i < len(self.fib) and self.fib[i] <= remaining:
                result.append(i)
                remaining -= self.fib[i]
                i -= 2  # 确保非连续性
            else:
                i -= 1
                
            if remaining == 0:
                break
                
        return sorted(result)
        
    def decode(self, representation):
        """从φ-表示恢复整数"""
        return sum(self.fib[i] for i in representation)
        
    def encode_binary_string(self, binary_str):
        """从二进制串（LSB在前）计算对应整数"""
        total = 0
        for i, bit in enumerate(binary_str, 1):
            if bit == '1':
                if i < len(self.fib):
                    total += self.fib[i]
        return total
        
    def to_binary_string(self, representation, length=None):
        """将φ-表示转换为二进制串"""
        if not representation:
            return '0' if length is None else '0' * length
            
        max_idx = max(representation)
        if length is None:
            length = max_idx
        else:
            length = max(length, max_idx)
            
        binary = ['0'] * length
        for idx in representation:
            if idx <= length:
                binary[idx-1] = '1'
                
        return ''.join(binary)
        
    def verify_non_consecutive(self, representation):
        """验证表示的非连续性"""
        sorted_repr = sorted(representation)
        for i in range(len(sorted_repr) - 1):
            if sorted_repr[i+1] - sorted_repr[i] < 2:
                return False
        return True
        
    def find_all_representations(self, n, allow_consecutive=False):
        """寻找n的所有可能表示（用于验证唯一性）"""
        all_reprs = []
        
        def backtrack(target, max_idx, current):
            if target == 0:
                all_reprs.append(sorted(list(current)))
                return
                
            for i in range(min(max_idx, len(self.fib)-1), 0, -1):
                if self.fib[i] <= target:
                    # 检查约束
                    if allow_consecutive or not current or all(abs(i - j) >= 2 for j in current):
                        current.add(i)
                        backtrack(target - self.fib[i], i-1, current)
                        current.remove(i)
                        
        # 找到最大可能的索引
        max_possible = len(self.fib) - 1
        while max_possible > 0 and self.fib[max_possible] > n:
            max_possible -= 1
            
        backtrack(n, max_possible, set())
        return all_reprs
        
    def add_representations(self, repr1, repr2):
        """φ-表示的加法"""
        # 合并索引
        merged = {}
        for i in repr1:
            merged[i] = merged.get(i, 0) + 1
        for i in repr2:
            merged[i] = merged.get(i, 0) + 1
            
        # 处理重复和违反约束的情况
        result = []
        carry = 0
        
        indices = sorted(set(repr1) | set(repr2) | {1})
        max_idx = max(indices) + 5  # 预留空间处理进位
        
        # 模拟二进制加法但处理Fibonacci进位
        for i in range(1, max_idx):
            count = merged.get(i, 0) + carry
            carry = 0
            
            if count == 0:
                continue
            elif count == 1:
                result.append(i)
            elif count == 2:
                # F(i) + F(i) = F(i+1) + F(i-2)
                if i >= 3:
                    merged[i-2] = merged.get(i-2, 0) + 1
                carry = 1
            else:
                # count >= 3，需要多次应用恒等式
                if i >= 3:
                    merged[i-2] = merged.get(i-2, 0) + count // 2
                carry = count // 2
                if count % 2 == 1:
                    result.append(i)
                    
        # 清理结果，确保非连续性
        return self.greedy_decomposition(self.decode(result))


class TestL1_6_PhiRepresentation(VerificationTest):
    """L1-6 φ-表示系统建立的形式化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        self.phi_system = PhiRepresentationSystem()
        self.golden_ratio = (1 + math.sqrt(5)) / 2
        
    def test_existence_proof(self):
        """测试存在性证明 - 验证检查点1"""
        # 测试前100个正整数都有φ-表示
        for n in range(1, 100):
            repr = self.phi_system.greedy_decomposition(n)
            
            # 验证表示的和等于n
            decoded = self.phi_system.decode(repr)
            self.assertEqual(
                decoded, n,
                f"Decoded value {decoded} != {n} for representation {repr}"
            )
            
            # 验证非连续性
            self.assertTrue(
                self.phi_system.verify_non_consecutive(repr),
                f"Representation {repr} for {n} has consecutive indices"
            )
            
        # 测试特殊值
        special_values = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        for n in special_values:
            repr = self.phi_system.greedy_decomposition(n)
            self.assertEqual(self.phi_system.decode(repr), n)
            
    def test_uniqueness_proof(self):
        """测试唯一性证明 - 验证检查点2"""
        # 对较小的数验证唯一性
        for n in range(1, 50):
            all_reprs = self.phi_system.find_all_representations(n)
            
            self.assertEqual(
                len(all_reprs), 1,
                f"Number {n} has {len(all_reprs)} representations: {all_reprs}"
            )
            
            # 验证贪心算法给出的就是唯一表示
            greedy_repr = sorted(self.phi_system.greedy_decomposition(n))
            unique_repr = sorted(all_reprs[0])
            
            self.assertEqual(
                greedy_repr, unique_repr,
                f"Greedy {greedy_repr} != unique {unique_repr} for n={n}"
            )
            
    def test_bijection_establishment(self):
        """测试双射建立 - 验证检查点3"""
        # 生成所有标准形式的no-11二进制串（去除尾部0）
        no11_strings = []
        seen = set()
        
        for length in range(1, 11):
            for i in range(1, 2**length):  # 从1开始，排除全0
                binary = format(i, f'0{length}b')
                if '11' not in binary:
                    # 转换为标准形式（LSB在前，去除尾部0）
                    binary_rev = binary[::-1].rstrip('0')
                    if binary_rev and binary_rev not in seen:
                        no11_strings.append(binary_rev)
                        seen.add(binary_rev)
                    
        # 映射到整数
        mapped_integers = set()
        string_to_int = {}
        
        for s in no11_strings:
            n = self.phi_system.encode_binary_string(s)
            self.assertNotIn(
                n, mapped_integers,
                f"Not injective: {s} and {string_to_int.get(n)} both map to {n}"
            )
            mapped_integers.add(n)
            string_to_int[n] = s
                
        # 验证映射的整数形成连续序列
        mapped_list = sorted(mapped_integers)
        expected = list(range(1, len(mapped_list) + 1))
        
        self.assertEqual(
            mapped_list, expected,
            f"Mapped integers not consecutive: {mapped_list[:10]}..."
        )
        
    def test_encoding_efficiency(self):
        """测试编码效率 - 验证检查点4"""
        # 测试不同大小的数的编码长度
        test_values = [10, 50, 100, 500, 1000, 5000, 10000]
        
        for n in test_values:
            repr = self.phi_system.greedy_decomposition(n)
            actual_length = max(repr) if repr else 0
            
            # 理论长度 ⌊log_φ(n)⌋ + O(1)
            theoretical_length = math.floor(math.log(n) / math.log(self.golden_ratio))
            
            # 允许常数偏差
            difference = abs(actual_length - theoretical_length)
            self.assertLessEqual(
                difference, 3,
                f"Length for n={n}: actual={actual_length}, theoretical={theoretical_length}"
            )
            
    def test_greedy_algorithm_properties(self):
        """测试贪心算法性质"""
        # 验证贪心选择性质
        for n in range(1, 100):
            repr = self.phi_system.greedy_decomposition(n)
            if repr:
                max_idx = max(repr)
                
                # 验证使用了不超过n的最大Fibonacci数
                self.assertLessEqual(
                    self.phi_system.fib[max_idx], n,
                    f"F({max_idx}) = {self.phi_system.fib[max_idx]} > {n}"
                )
                
                # 验证不能使用更大的Fibonacci数
                if max_idx + 1 < len(self.phi_system.fib):
                    self.assertGreater(
                        self.phi_system.fib[max_idx + 1], n,
                        f"Could use larger F({max_idx + 1})"
                    )
                    
    def test_fibonacci_values(self):
        """测试修正的Fibonacci数列"""
        # 验证前几项
        expected = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        
        for i in range(1, len(expected)):
            self.assertEqual(
                self.phi_system.fib[i], expected[i],
                f"F({i}) = {self.phi_system.fib[i]}, expected {expected[i]}"
            )
            
        # 验证递归关系
        for i in range(3, 20):
            self.assertEqual(
                self.phi_system.fib[i],
                self.phi_system.fib[i-1] + self.phi_system.fib[i-2],
                f"F({i}) != F({i-1}) + F({i-2})"
            )
            
    def test_addition_operation(self):
        """测试加法运算"""
        # 测试简单加法
        test_cases = [
            (5, 8, 13),    # F(4) + F(5) = F(6)
            (3, 5, 8),     # F(3) + F(4) = F(5)
            (10, 14, 24),  # 一般情况：10 + 14 = 24
        ]
        
        for a, b, expected in test_cases:
            repr_a = self.phi_system.greedy_decomposition(a)
            repr_b = self.phi_system.greedy_decomposition(b)
            repr_sum = self.phi_system.add_representations(repr_a, repr_b)
            
            actual_sum = self.phi_system.decode(repr_sum)
            self.assertEqual(
                actual_sum, expected,
                f"{a} + {b} = {actual_sum}, expected {expected}"
            )
            
            # 验证结果仍满足非连续性
            self.assertTrue(
                self.phi_system.verify_non_consecutive(repr_sum),
                f"Sum representation {repr_sum} violates non-consecutive constraint"
            )
            
    def test_binary_string_correspondence(self):
        """测试与二进制串的对应关系"""
        # 测试具体例子
        test_cases = [
            ("1", 1),      # F(1) = 1
            ("01", 2),     # F(2) = 2
            ("001", 3),    # F(3) = 3
            ("101", 4),    # F(1) + F(3) = 1 + 3 = 4
            ("0001", 5),   # F(4) = 5
        ]
        
        for binary_str, expected in test_cases:
            actual = self.phi_system.encode_binary_string(binary_str)
            self.assertEqual(
                actual, expected,
                f"Binary '{binary_str}' encodes to {actual}, expected {expected}"
            )
            
            # 验证逆向转换
            repr = self.phi_system.greedy_decomposition(expected)
            binary_back = self.phi_system.to_binary_string(repr, len(binary_str))
            
            # 移除尾部的0进行比较
            self.assertEqual(
                binary_back.rstrip('0'), binary_str.rstrip('0'),
                f"Inverse conversion failed for {expected}"
            )
            
    def test_completeness(self):
        """测试系统完备性"""
        # 验证连续整数的表示
        representations = []
        for n in range(1, 50):
            repr = self.phi_system.greedy_decomposition(n)
            representations.append(repr)
            
        # 验证表示的单调性（字典序）
        for i in range(len(representations) - 1):
            # 转换为二进制串比较
            bin1 = self.phi_system.to_binary_string(representations[i])
            bin2 = self.phi_system.to_binary_string(representations[i+1])
            
            # 解码值应该递增
            val1 = self.phi_system.decode(representations[i])
            val2 = self.phi_system.decode(representations[i+1])
            
            self.assertLess(
                val1, val2,
                f"Order not preserved: {val1} >= {val2}"
            )
            
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试Fibonacci数本身
        fib_numbers = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        for i, fib in enumerate(fib_numbers, 1):
            repr = self.phi_system.greedy_decomposition(fib)
            
            # Fibonacci数的表示应该只包含一个索引
            self.assertEqual(
                len(repr), 1,
                f"F({i}) = {fib} should have single-element representation"
            )
            
            if i < len(self.phi_system.fib):
                self.assertEqual(
                    repr[0], i,
                    f"F({i}) should be represented by index {i}"
                )
                
        # 测试接近Fibonacci数的值
        for fib in [8, 13, 21]:
            # fib - 1 的表示应该更复杂
            repr_minus_1 = self.phi_system.greedy_decomposition(fib - 1)
            repr_fib = self.phi_system.greedy_decomposition(fib)
            
            self.assertGreater(
                len(repr_minus_1), len(repr_fib),
                f"Representation of {fib-1} should be more complex than {fib}"
            )


if __name__ == "__main__":
    unittest.main()