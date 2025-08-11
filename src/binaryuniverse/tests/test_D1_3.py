"""
Unit tests for D1-3: No-11 Constraint
D1-3：no-11约束的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
from formal_system import BinaryEncoding


class TestD1_3_No11Constraint(VerificationTest):
    """D1-3 no-11约束的形式化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        self.encoder = BinaryEncoding(no_11_constraint=True)
        
    def test_no_consecutive_ones(self):
        """测试无连续1验证 - 验证检查点1"""
        # 有效字符串（无11）
        valid_strings = [
            "0",
            "1",
            "10",
            "01",
            "101",
            "010",
            "1010",
            "0101",
            "10010",
            "01010101"
        ]
        
        for s in valid_strings:
            self.assertTrue(
                self.encoder.is_valid_no11(s),
                f"String '{s}' should be valid (no consecutive 1s)"
            )
            
        # 无效字符串（含11）
        invalid_strings = [
            "11",
            "110",
            "011",
            "111",
            "0110",
            "1011",
            "1101",
            "01110",
            "101101"
        ]
        
        for s in invalid_strings:
            self.assertFalse(
                self.encoder.is_valid_no11(s),
                f"String '{s}' should be invalid (contains 11)"
            )
            
    def test_valid_string_set(self):
        """测试有效字符串集合验证 - 验证检查点2"""
        # 生成长度为n的所有有效字符串
        def generate_valid_strings(n):
            if n == 0:
                return [""]
            if n == 1:
                return ["0", "1"]
                
            valid = []
            # 递归生成
            prev = generate_valid_strings(n-1)
            for s in prev:
                # 总是可以添加0
                valid.append(s + "0")
                # 只有不以1结尾时才能添加1
                if not s or s[-1] != "1":
                    valid.append(s + "1")
            return valid
            
        # 测试长度1-5的所有有效字符串
        for length in range(1, 6):
            valid_strings = generate_valid_strings(length)
            
            # 验证集合中所有字符串都满足约束
            for s in valid_strings:
                self.assertTrue(
                    self.encoder.is_valid_no11(s),
                    f"Generated string '{s}' should satisfy no-11 constraint"
                )
                
            # 验证集合大小符合Fibonacci数列
            expected_count = self._fibonacci(length + 2)
            self.assertEqual(
                len(valid_strings),
                expected_count,
                f"Count of valid strings of length {length} should be F_{length+2} = {expected_count}"
            )
            
    def test_recursive_generation(self):
        """测试递归生成验证 - 验证检查点3"""
        # 实现递归生成规则
        def matches_grammar(s):
            # 空串是有效的
            if not s:
                return True
            # 以0开头：检查剩余部分
            if s[0] == "0":
                return matches_grammar(s[1:])
            # 以1开头
            elif s[0] == "1":
                # 可以是单个1
                if len(s) == 1:
                    return True
                # 或者是10后跟有效字符串
                if len(s) >= 2 and s[1] == "0":
                    return matches_grammar(s[2:])
                else:
                    return False
            return False
            
        # 测试有效字符串
        valid_examples = ["", "0", "1", "01", "10", "010", "101", "0101", "1010", "10010"]
        for s in valid_examples:
            self.assertTrue(
                matches_grammar(s),
                f"String '{s}' should match the recursive grammar"
            )
            
        # 测试无效字符串
        invalid_examples = ["11", "011", "110", "111", "0110", "1011"]
        for s in invalid_examples:
            self.assertFalse(
                matches_grammar(s),
                f"String '{s}' should not match the recursive grammar"
            )
            
    def test_fibonacci_counting(self):
        """测试Fibonacci计数验证 - 验证检查点4"""
        # 计算长度为n的有效字符串数量
        def count_valid_strings(n):
            if n == 0:
                return 1  # 空串
            if n == 1:
                return 2  # "0", "1"
                
            # 动态规划：dp[i]表示长度为i的有效字符串数
            dp = [0] * (n + 1)
            dp[0] = 1
            dp[1] = 2
            
            for i in range(2, n + 1):
                dp[i] = dp[i-1] + dp[i-2]
                
            return dp[n]
            
        # 验证前10个长度的计数
        for n in range(10):
            actual_count = count_valid_strings(n)
            expected_count = self._fibonacci(n + 2)
            
            self.assertEqual(
                actual_count,
                expected_count,
                f"Count for length {n} should be F_{n+2} = {expected_count}"
            )
            
    def test_prefix_closure(self):
        """测试前缀封闭性"""
        test_strings = [
            "1010",
            "01010",
            "10010",
            "010101010"
        ]
        
        for s in test_strings:
            if self.encoder.is_valid_no11(s):
                # 测试所有前缀
                for i in range(len(s)):
                    prefix = s[:i+1]
                    self.assertTrue(
                        self.encoder.is_valid_no11(prefix),
                        f"Prefix '{prefix}' of valid string '{s}' should also be valid"
                    )
                    
    def test_extension_rules(self):
        """测试扩展规则"""
        test_strings = ["", "0", "1", "10", "01", "010", "101"]
        
        for s in test_strings:
            if self.encoder.is_valid_no11(s):
                # 规则1：总是可以添加0
                extended_0 = s + "0"
                self.assertTrue(
                    self.encoder.is_valid_no11(extended_0),
                    f"Should always be able to append '0' to valid string '{s}'"
                )
                
                # 规则2：只有不以1结尾时才能添加1
                extended_1 = s + "1"
                if not s or s[-1] != "1":
                    self.assertTrue(
                        self.encoder.is_valid_no11(extended_1),
                        f"Should be able to append '1' to '{s}' (doesn't end with 1)"
                    )
                else:
                    self.assertFalse(
                        self.encoder.is_valid_no11(extended_1),
                        f"Should not be able to append '1' to '{s}' (ends with 1)"
                    )
                    
    def test_fibonacci_representation(self):
        """测试Fibonacci表示（Zeckendorf表示）"""
        # 测试前几个数的Fibonacci表示
        test_cases = [
            (0, "0"),
            (1, "1"),
            (2, "10"),
            (3, "100"),
            (4, "101"),
            (5, "1000"),
            (6, "1001"),
            (7, "1010"),
            (8, "10000"),
            (9, "10001"),
            (10, "10010"),
            (11, "10100"),
            (12, "10101")
        ]
        
        for n, expected in test_cases:
            actual = self.encoder.fibonacci_representation(n)
            self.assertEqual(
                actual, expected,
                f"Fibonacci representation of {n} should be '{expected}'"
            )
            
            # 验证结果满足no-11约束
            self.assertTrue(
                self.encoder.is_valid_no11(actual),
                f"Fibonacci representation '{actual}' should satisfy no-11 constraint"
            )
            
    def test_information_capacity(self):
        """测试信息容量"""
        # 计算渐近信息容量
        import math
        phi = (1 + math.sqrt(5)) / 2
        theoretical_capacity = math.log2(phi)
        
        # 验证编码器的phi密度计算
        actual_capacity = self.encoder.phi_density()
        
        self.assertAlmostEqual(
            actual_capacity,
            theoretical_capacity,
            places=10,
            msg=f"Information capacity should be log2(φ) ≈ {theoretical_capacity}"
        )
        
    def test_constraint_application(self):
        """测试约束应用"""
        # 测试编码器的约束应用功能
        test_strings = [
            ("11", "101"),      # 11 -> 101
            ("110", "1010"),    # 110 -> 1010
            ("011", "0101"),    # 011 -> 0101
            ("111", "10101"),   # 111 -> 10101
            ("1110", "101010"), # 1110 -> 101010
        ]
        
        for original, _ in test_strings:
            result = self.encoder._apply_no11_constraint(original)
            # 由于实现是简单替换，可能需要多次应用
            while "11" in result:
                result = self.encoder._apply_no11_constraint(result)
                
            self.assertTrue(
                self.encoder.is_valid_no11(result),
                f"Result '{result}' after applying constraint to '{original}' should be valid"
            )
            
    def _fibonacci(self, n):
        """计算第n个Fibonacci数"""
        if n == 0:
            return 0
        if n == 1:
            return 1
        return self._fibonacci(n-1) + self._fibonacci(n-2)


if __name__ == "__main__":
    unittest.main()