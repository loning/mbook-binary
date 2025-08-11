"""
Unit tests for T2-6: no-11 Constraint Theorem
T2-6：no-11约束定理的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
import math
from typing import List, Set, Dict, Tuple


class TestT2_6_No11Constraint(VerificationTest):
    """T2-6 no-11约束定理的数学化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        self.golden_ratio = (1 + math.sqrt(5)) / 2
        
    def count_valid_strings_no11(self, n: int) -> int:
        """计算长度为n的不含'11'的二进制串数量"""
        if n == 0:
            return 1
        if n == 1:
            return 2
        
        # 使用动态规划
        prev2, prev1 = 1, 2
        for i in range(2, n+1):
            current = prev1 + prev2
            prev2, prev1 = prev1, current
        
        return prev1
    
    def enumerate_valid_strings(self, n: int) -> List[str]:
        """枚举所有长度为n的不含'11'的二进制串"""
        if n == 0:
            return [""]
        
        valid = []
        for i in range(2**n):
            binary = format(i, f'0{n}b')
            if "11" not in binary:
                valid.append(binary)
        
        return sorted(valid)
    
    def compute_fibonacci_sequence(self, n: int) -> List[int]:
        """计算前n个Fibonacci数"""
        if n == 0:
            return []
        if n == 1:
            return [0]
        
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        
        return fib
    
    def is_valid_phi_representation(self, rep: List[int]) -> bool:
        """检查是否是有效的φ-表示（无相邻的1）"""
        for i in range(len(rep) - 1):
            if rep[i] == 1 and rep[i+1] == 1:
                return False
        return True
    
    def compute_phi_value(self, rep: List[int], fib_sequence: List[int]) -> int:
        """计算φ-表示的值"""
        value = 0
        for i, bit in enumerate(rep):
            if bit == 1:
                value += fib_sequence[i]
        return value
    
    def test_fibonacci_recurrence(self):
        """测试Fibonacci递归 - 验证检查点1"""
        # 计算前20个值
        valid_counts = []
        for n in range(20):
            count = self.count_valid_strings_no11(n)
            valid_counts.append(count)
        
        # 验证递归关系
        for i in range(2, len(valid_counts)):
            expected = valid_counts[i-1] + valid_counts[i-2]
            actual = valid_counts[i]
            self.assertEqual(
                actual, expected,
                f"Recurrence failed at n={i}: {actual} != {expected}"
            )
        
        # 验证与Fibonacci数列的关系
        fibonacci = self.compute_fibonacci_sequence(22)
        for i in range(len(valid_counts)):
            self.assertEqual(
                valid_counts[i], fibonacci[i+2],
                f"Not equal to F_{i+2}: {valid_counts[i]} != {fibonacci[i+2]}"
            )
            
    def test_initial_conditions(self):
        """测试初始条件 - 验证检查点2"""
        # n=0: 空串
        self.assertEqual(
            self.count_valid_strings_no11(0), 1,
            "n=0 should have 1 valid string (empty)"
        )
        
        # n=1: "0", "1"
        self.assertEqual(
            self.count_valid_strings_no11(1), 2,
            "n=1 should have 2 valid strings"
        )
        valid_1 = set(self.enumerate_valid_strings(1))
        self.assertEqual(
            valid_1, {"0", "1"},
            "n=1 strings should be {'0', '1'}"
        )
        
        # n=2: "00", "01", "10" (不含"11")
        self.assertEqual(
            self.count_valid_strings_no11(2), 3,
            "n=2 should have 3 valid strings"
        )
        valid_2 = set(self.enumerate_valid_strings(2))
        self.assertEqual(
            valid_2, {"00", "01", "10"},
            "n=2 strings should exclude '11'"
        )
        
    def test_counting_verification(self):
        """测试计数验证 - 验证检查点3"""
        # 前几项的详细验证
        test_cases = [
            (0, 1, set()),  # 空串算作一个
            (1, 2, {"0", "1"}),
            (2, 3, {"00", "01", "10"}),
            (3, 5, {"000", "001", "010", "100", "101"}),
            (4, 8, {"0000", "0001", "0010", "0100", "0101", 
                    "1000", "1001", "1010"})
        ]
        
        for n, expected_count, expected_set in test_cases:
            # 计数验证
            actual_count = self.count_valid_strings_no11(n)
            self.assertEqual(
                actual_count, expected_count,
                f"Count mismatch at n={n}: {actual_count} != {expected_count}"
            )
            
            # 枚举验证（除了n=0）
            if n > 0:
                actual_set = set(self.enumerate_valid_strings(n))
                self.assertEqual(
                    len(actual_set), expected_count,
                    f"Enumeration count mismatch at n={n}"
                )
                self.assertEqual(
                    actual_set, expected_set,
                    f"Enumeration set mismatch at n={n}"
                )
                
                # 验证每个字符串都不含"11"
                for s in actual_set:
                    self.assertNotIn(
                        "11", s,
                        f"String '{s}' contains forbidden pattern '11'"
                    )
                    
    def test_phi_representation_definition(self):
        """测试φ-表示定义 - 验证检查点4"""
        # 定义修改的Fibonacci序列 (1, 2, 3, 5, 8, ...)
        fib = [1, 2]
        for i in range(2, 20):
            fib.append(fib[i-1] + fib[i-2])
        
        # 测试一些φ-表示
        test_cases = [
            ([1, 0, 0], 1),      # 1*1 = 1
            ([0, 1, 0], 2),      # 1*2 = 2
            ([1, 0, 1], 4),      # 1*1 + 1*3 = 4
            ([0, 0, 0, 1], 5),   # 1*5 = 5
            ([1, 0, 0, 0, 1], 9) # 1*1 + 1*8 = 9
        ]
        
        for rep, expected_value in test_cases:
            # 验证是有效的φ-表示（无相邻的1）
            self.assertTrue(
                self.is_valid_phi_representation(rep),
                f"Representation {rep} should be valid"
            )
            
            # 计算值
            value = self.compute_phi_value(rep, fib)
            self.assertEqual(
                value, expected_value,
                f"Value mismatch for {rep}: {value} != {expected_value}"
            )
        
        # 验证无效表示
        invalid_reps = [[1, 1], [0, 1, 1, 0], [1, 1, 0, 0]]
        for rep in invalid_reps:
            self.assertFalse(
                self.is_valid_phi_representation(rep),
                f"Representation {rep} should be invalid (has adjacent 1s)"
            )
            
    def test_growth_rate_analysis(self):
        """测试增长率分析 - 验证检查点5"""
        # 计算足够多的项来分析增长率
        counts = []
        for n in range(50):
            counts.append(self.count_valid_strings_no11(n))
        
        # 计算连续项的比率
        ratios = []
        for i in range(10, len(counts)-1):
            if counts[i] > 0:
                ratio = counts[i+1] / counts[i]
                ratios.append(ratio)
        
        # 计算平均比率
        avg_ratio = sum(ratios) / len(ratios)
        
        # 验证接近黄金比例
        self.assertAlmostEqual(
            avg_ratio, self.golden_ratio, 3,
            f"Average growth ratio {avg_ratio} should be close to φ"
        )
        
        # 验证生成函数的正确性
        # 生成函数应该是 (1+x)/(1-x-x²)
        # 因为 a_0 = 1, a_1 = 2, 而标准Fibonacci是 f_0 = 0, f_1 = 1
        
        # 选择一个小的x值
        x = 0.2
        # 正确的生成函数
        theoretical = (1 + x) / (1 - x - x*x)
        
        # 计算足够多项的部分和
        partial_sum = sum(counts[i] * (x**i) for i in range(len(counts)))
        
        # 验证收敛
        relative_error = abs(partial_sum - theoretical) / theoretical
        self.assertLess(
            relative_error, 0.01,
            f"Generating function error: {relative_error}"
        )
        
        # 验证递归关系
        # 对于n >= 2: a_n = a_{n-1} + a_{n-2}
        # 这在前面的测试中已经验证过了
        
    def test_fibonacci_correspondence_proof(self):
        """测试Fibonacci对应关系的证明"""
        # 验证归纳法的基础情况
        fib = self.compute_fibonacci_sequence(50)
        
        # 基础情况
        self.assertEqual(self.count_valid_strings_no11(0), fib[2])  # F_2 = 1
        self.assertEqual(self.count_valid_strings_no11(1), fib[3])  # F_3 = 2
        
        # 验证前30项
        for n in range(30):
            count = self.count_valid_strings_no11(n)
            expected = fib[n+2]
            self.assertEqual(
                count, expected,
                f"Failed correspondence at n={n}: {count} != F_{n+2} = {expected}"
            )
            
    def test_combinatorial_generation(self):
        """测试组合生成方法"""
        # 使用递归方法生成有效字符串
        def generate_recursive(n: int, current: str = "") -> Set[str]:
            if len(current) == n:
                return {current}
            
            result = set()
            # 可以添加'0'
            result.update(generate_recursive(n, current + "0"))
            
            # 只有当不会产生"11"时才能添加'1'
            if not current or current[-1] == '0':
                result.update(generate_recursive(n, current + "1"))
                
            return result
        
        # 验证递归生成与枚举方法一致
        for n in range(1, 6):
            recursive_set = generate_recursive(n)
            enum_set = set(self.enumerate_valid_strings(n))
            
            self.assertEqual(
                recursive_set, enum_set,
                f"Generation methods differ at n={n}"
            )
            
    def test_specific_string_patterns(self):
        """测试特定字符串模式"""
        # 验证一些特定模式的计数
        patterns = {
            "all_zeros": lambda n: 1,  # "000...0"总是有效
            "alternating_01": lambda n: 1,  # "010101..."总是有效
            "alternating_10": lambda n: 1,  # "101010..."总是有效
            "single_1_at_end": lambda n: n,  # "000...01"的变体
        }
        
        for n in range(1, 10):
            valid_strings = self.enumerate_valid_strings(n)
            
            # 验证全0字符串
            all_zeros = "0" * n
            self.assertIn(
                all_zeros, valid_strings,
                f"All zeros string should be valid for n={n}"
            )
            
            # 验证交替模式
            if n >= 2:
                alt_01 = "01" * (n // 2) + ("0" if n % 2 else "")
                alt_10 = "10" * (n // 2) + ("1" if n % 2 else "")
                
                self.assertIn(alt_01, valid_strings)
                self.assertIn(alt_10, valid_strings)
                
    def test_asymptotic_behavior(self):
        """测试渐近行为"""
        # 验证a_n ~ φ^(n+2) / √5
        sqrt5 = math.sqrt(5)
        
        # 计算足够大的n
        test_n = [20, 30, 40]
        
        for n in test_n:
            actual = self.count_valid_strings_no11(n)
            fib_n_plus_2 = self.compute_fibonacci_sequence(n+3)[n+2]
            
            # 验证等于F_{n+2}
            self.assertEqual(actual, fib_n_plus_2)
            
            # 验证渐近公式
            asymptotic = (self.golden_ratio ** (n+2)) / sqrt5
            relative_error = abs(actual - asymptotic) / actual
            
            # 随着n增大，相对误差应该减小
            self.assertLess(
                relative_error, 0.01,
                f"Asymptotic formula error too large at n={n}: {relative_error}"
            )
            
    def test_zeckendorf_representation(self):
        """测试Zeckendorf表示的性质"""
        # 生成修改的Fibonacci序列
        fib = [1, 2]
        for i in range(2, 15):
            fib.append(fib[i-1] + fib[i-2])
        
        # 测试一些数的Zeckendorf表示
        def find_zeckendorf(n: int) -> List[int]:
            """贪心算法找Zeckendorf表示"""
            result = []
            remaining = n
            
            # 从大到小尝试Fibonacci数
            for i in range(len(fib)-1, -1, -1):
                if fib[i] <= remaining:
                    result.append(i)
                    remaining -= fib[i]
                    
            return result
        
        # 验证前20个数都有唯一的Zeckendorf表示
        for n in range(1, 21):
            indices = find_zeckendorf(n)
            
            # 验证和等于n
            total = sum(fib[i] for i in indices)
            self.assertEqual(total, n)
            
            # 验证没有相邻的索引（对应no-11约束）
            for i in range(len(indices)-1):
                self.assertGreater(
                    indices[i] - indices[i+1], 1,
                    f"Adjacent indices in Zeckendorf representation of {n}"
                )


if __name__ == "__main__":
    unittest.main()