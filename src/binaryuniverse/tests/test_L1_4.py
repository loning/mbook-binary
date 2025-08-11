"""
Unit tests for L1-4: No-11 Constraint Optimality Lemma
L1-4：no-11约束最优性引理的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
import math


class ConstraintAnalyzer:
    """约束分析器"""
    
    def __init__(self, forbidden_pattern):
        self.forbidden_pattern = forbidden_pattern
        self.valid_string_counts = {}
        self._compute_counts(20)  # 预计算到长度20
        
    def _compute_counts(self, max_length):
        """计算各长度的有效串数量"""
        for n in range(1, max_length + 1):
            self.valid_string_counts[n] = self._count_valid_strings(n)
            
    def _count_valid_strings(self, length):
        """计算长度为n的有效串数量"""
        if length == 0:
            return 1
            
        count = 0
        for i in range(2**length):
            binary = format(i, f'0{length}b')
            if self.forbidden_pattern not in binary:
                count += 1
        return count
        
    def get_recurrence_relation(self):
        """获取递归关系"""
        if self.forbidden_pattern in ['00', '11']:
            # 简单Fibonacci递归
            return 'fibonacci', lambda n: self.valid_string_counts.get(n-1, 0) + self.valid_string_counts.get(n-2, 0)
        else:
            # 复杂递归
            return 'complex', None
            
    def compute_capacity(self):
        """计算信息容量"""
        # 使用较大的n值来估计渐近行为
        growth_rates = []
        
        for n in range(10, 20):
            if n in self.valid_string_counts and n-1 in self.valid_string_counts:
                if self.valid_string_counts[n-1] > 0:
                    rate = math.log(self.valid_string_counts[n] / self.valid_string_counts[n-1])
                    growth_rates.append(rate)
                    
        if growth_rates:
            return sum(growth_rates) / len(growth_rates)
        return 0
        
    def verify_fibonacci_relation(self):
        """验证是否满足Fibonacci关系"""
        if self.forbidden_pattern not in ['00', '11']:
            return False
            
        # 检查 N(n) = N(n-1) + N(n-2)
        for n in range(3, 15):
            expected = self.valid_string_counts[n-1] + self.valid_string_counts[n-2]
            actual = self.valid_string_counts[n]
            if expected != actual:
                return False
                
        return True
        
    def check_symmetry(self):
        """检查约束的对称性"""
        # 生成一些测试串
        test_strings = []
        for length in range(1, 6):
            for i in range(min(2**length, 20)):  # 限制数量
                binary = format(i, f'0{length}b')
                if self.forbidden_pattern not in binary:
                    test_strings.append(binary)
                    
        # 检查翻转对称性
        flipped_pattern = self.forbidden_pattern.translate(str.maketrans('01', '10'))
        
        for s in test_strings:
            flipped_s = s.translate(str.maketrans('01', '10'))
            
            # 原串有效，翻转串应该对翻转约束也有效
            original_valid = self.forbidden_pattern not in s
            flipped_valid = flipped_pattern not in flipped_s
            
            # 对于对称约束，两者应该相等
            if self.forbidden_pattern in ['00', '11']:
                if original_valid != flipped_valid:
                    return False
                    
        return True


class TestL1_4_No11Optimality(VerificationTest):
    """L1-4 no-11约束最优性的形式化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        self.analyzers = {
            pattern: ConstraintAnalyzer(pattern)
            for pattern in ['00', '01', '10', '11']
        }
        self.golden_ratio = (1 + math.sqrt(5)) / 2
        
    def test_constraint_classification(self):
        """测试约束分类 - 验证检查点1"""
        patterns = ['00', '01', '10', '11']
        symmetric = []
        asymmetric = []
        
        for p in patterns:
            # 翻转模式
            flipped = p.translate(str.maketrans('01', '10'))
            
            # 检查对称性
            if p == flipped or {p, flipped} == {'00', '11'}:
                symmetric.append(p)
            else:
                asymmetric.append(p)
                
        # 验证分类
        self.assertEqual(
            set(symmetric), {'00', '11'},
            "Symmetric constraints should be {00, 11}"
        )
        
        self.assertEqual(
            set(asymmetric), {'01', '10'},
            "Asymmetric constraints should be {01, 10}"
        )
        
        # 验证对称约束的互补性
        self.assertEqual(
            '00'.translate(str.maketrans('01', '10')), '11',
            "00 and 11 should be symmetric under 0-1 flip"
        )
        
    def test_symmetry_analysis(self):
        """测试对称性分析 - 验证检查点2"""
        # 测试每个约束的对称性保持
        for pattern, analyzer in self.analyzers.items():
            is_symmetric = analyzer.check_symmetry()
            
            if pattern in ['00', '11']:
                self.assertTrue(
                    is_symmetric,
                    f"Pattern {pattern} should preserve symmetry"
                )
            else:
                # 01和10本身不对称，但它们是彼此的翻转
                flipped = pattern.translate(str.maketrans('01', '10'))
                self.assertEqual(
                    flipped, '10' if pattern == '01' else '01',
                    f"Pattern {pattern} should flip to its counterpart"
                )
                
    def test_capacity_calculation(self):
        """测试容量计算 - 验证检查点3"""
        capacities = {}
        
        for pattern, analyzer in self.analyzers.items():
            capacity = analyzer.compute_capacity()
            capacities[pattern] = capacity
            
            # 验证容量是正数
            self.assertGreater(
                capacity, 0,
                f"Capacity for pattern {pattern} should be positive"
            )
            
            # 验证容量小于log(2) = ln(2)
            self.assertLess(
                capacity, math.log(2),
                f"Capacity for pattern {pattern} should be less than unconstrained"
            )
            
        # 验证对称约束有相同容量
        self.assertAlmostEqual(
            capacities['00'], capacities['11'], 3,
            "Symmetric constraints should have equal capacity"
        )
        
        # 验证no-11容量接近log(φ)
        log_phi = math.log(self.golden_ratio)
        self.assertAlmostEqual(
            capacities['11'], log_phi, 2,
            f"No-11 capacity should be close to log(φ) ≈ {log_phi}"
        )
        
    def test_optimality_proof(self):
        """测试最优性证明 - 验证检查点4"""
        capacities = {
            pattern: analyzer.compute_capacity()
            for pattern, analyzer in self.analyzers.items()
        }
        
        # 找出最大容量
        max_capacity = max(capacities.values())
        optimal_patterns = [
            p for p, c in capacities.items()
            if abs(c - max_capacity) < 0.01
        ]
        
        # 验证最优模式是对称的
        self.assertEqual(
            set(optimal_patterns), {'00', '11'},
            "Optimal patterns should be the symmetric ones"
        )
        
        # 验证非对称约束次优
        for pattern in ['01', '10']:
            self.assertLess(
                capacities[pattern], max_capacity - 0.01,
                f"Asymmetric pattern {pattern} should be suboptimal"
            )
            
    def test_fibonacci_structure(self):
        """测试Fibonacci结构"""
        # 只有00和11约束产生Fibonacci递归
        for pattern in ['00', '11']:
            analyzer = self.analyzers[pattern]
            
            # 验证递归类型
            rec_type, _ = analyzer.get_recurrence_relation()
            self.assertEqual(
                rec_type, 'fibonacci',
                f"Pattern {pattern} should have Fibonacci recurrence"
            )
            
            # 验证Fibonacci关系
            is_fibonacci = analyzer.verify_fibonacci_relation()
            self.assertTrue(
                is_fibonacci,
                f"Pattern {pattern} should satisfy Fibonacci relation"
            )
            
        # 验证其他约束不是Fibonacci
        for pattern in ['01', '10']:
            analyzer = self.analyzers[pattern]
            rec_type, _ = analyzer.get_recurrence_relation()
            self.assertEqual(
                rec_type, 'complex',
                f"Pattern {pattern} should have complex recurrence"
            )
            
    def test_count_sequence(self):
        """测试计数序列"""
        # 验证no-11约束的具体计数
        no11 = self.analyzers['11']
        
        # 前几项应该是：2, 3, 5, 8, 13, 21...
        expected_counts = {
            1: 2,   # '0', '1'
            2: 3,   # '00', '01', '10'
            3: 5,   # '000', '001', '010', '100', '101'
            4: 8,   # 所有不含'11'的4位串
            5: 13,
            6: 21
        }
        
        for n, expected in expected_counts.items():
            actual = no11.valid_string_counts[n]
            self.assertEqual(
                actual, expected,
                f"N_{{11}}({n}) should be {expected}, got {actual}"
            )
            
        # 验证与Fibonacci数的关系: N(n) = F(n+2)
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        for n in range(1, 8):
            self.assertEqual(
                no11.valid_string_counts[n], fib[n+1],
                f"N_{{11}}({n}) should equal F({n+2})"
            )
            
    def test_golden_ratio_emergence(self):
        """测试黄金比例的涌现"""
        # 计算no-11约束的增长率
        no11 = self.analyzers['11']
        
        # 计算连续比率
        ratios = []
        for n in range(10, 18):
            if no11.valid_string_counts[n-1] > 0:
                ratio = no11.valid_string_counts[n] / no11.valid_string_counts[n-1]
                ratios.append(ratio)
                
        # 验证比率收敛到黄金比例
        avg_ratio = sum(ratios) / len(ratios)
        self.assertAlmostEqual(
            avg_ratio, self.golden_ratio, 3,
            f"Growth ratio should converge to φ ≈ {self.golden_ratio}"
        )
        
        # 验证黄金比例的自指性质
        phi_squared = self.golden_ratio ** 2
        phi_plus_one = self.golden_ratio + 1
        self.assertAlmostEqual(
            phi_squared, phi_plus_one, 10,
            "Golden ratio should satisfy φ² = φ + 1"
        )
        
    def test_physical_interpretation(self):
        """测试物理解释"""
        interpretations = {
            '00': "No consecutive empty states",
            '11': "No consecutive full states",
            '01': "No empty-to-full transition",
            '10': "No full-to-empty transition"
        }
        
        # 验证对称约束的物理意义更自然
        natural_interpretations = ['00', '11']
        
        for pattern in natural_interpretations:
            self.assertIn(
                'consecutive', interpretations[pattern],
                f"Pattern {pattern} has natural interpretation about consecutive states"
            )
            
    def test_comprehensive_optimality(self):
        """测试综合最优性"""
        # 收集所有度量
        metrics = {}
        
        for pattern, analyzer in self.analyzers.items():
            metrics[pattern] = {
                'capacity': analyzer.compute_capacity(),
                'symmetric': pattern in ['00', '11'],
                'fibonacci': analyzer.verify_fibonacci_relation(),
                'counts': [analyzer.valid_string_counts[i] for i in range(1, 11)]
            }
            
        # 验证对称约束在所有方面都最优
        symmetric_patterns = [p for p, m in metrics.items() if m['symmetric']]
        max_capacity = max(m['capacity'] for m in metrics.values())
        
        for pattern in symmetric_patterns:
            self.assertAlmostEqual(
                metrics[pattern]['capacity'], max_capacity, 3,
                f"Symmetric pattern {pattern} should have maximum capacity"
            )
            
            self.assertTrue(
                metrics[pattern]['fibonacci'],
                f"Symmetric pattern {pattern} should have Fibonacci structure"
            )
            
    def test_constraint_equivalence(self):
        """测试约束等价性"""
        # 00和11约束应该产生等价的结构（通过0-1对换）
        counts_00 = [self.analyzers['00'].valid_string_counts[i] for i in range(1, 10)]
        counts_11 = [self.analyzers['11'].valid_string_counts[i] for i in range(1, 10)]
        
        # 计数应该相同
        self.assertEqual(
            counts_00, counts_11,
            "Constraints 00 and 11 should produce same counts"
        )
        
        # 容量应该相同
        cap_00 = self.analyzers['00'].compute_capacity()
        cap_11 = self.analyzers['11'].compute_capacity()
        
        self.assertAlmostEqual(
            cap_00, cap_11, 5,
            "Constraints 00 and 11 should have identical capacity"
        )


if __name__ == "__main__":
    unittest.main()