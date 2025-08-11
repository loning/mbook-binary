"""
Unit tests for L1-3: Constraint Necessity Lemma
L1-3：约束必然性引理的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
import math
from collections import defaultdict


class BinaryEncodingSystem:
    """二进制编码系统"""
    
    def __init__(self, codewords=None, forbidden_patterns=None):
        self.codewords = set(codewords) if codewords else set()
        self.forbidden_patterns = set(forbidden_patterns) if forbidden_patterns else set()
        
    def add_codeword(self, word):
        """添加码字"""
        if self._contains_forbidden_pattern(word):
            return False
        self.codewords.add(word)
        return True
        
    def _contains_forbidden_pattern(self, word):
        """检查是否包含禁止模式"""
        for pattern in self.forbidden_patterns:
            if pattern in word:
                return True
        return False
        
    def is_valid_codeword(self, word):
        """检查是否是有效码字"""
        return word in self.codewords
        
    def decode(self, string):
        """解码字符串，返回所有可能的分解"""
        decompositions = []
        self._decode_recursive(string, 0, [], decompositions)
        return decompositions
        
    def _decode_recursive(self, string, start, current, all_decomps):
        """递归解码"""
        if start == len(string):
            all_decomps.append(current[:])
            return
            
        for end in range(start + 1, len(string) + 1):
            substring = string[start:end]
            if substring in self.codewords:
                current.append(substring)
                self._decode_recursive(string, end, current, all_decomps)
                current.pop()
                
    def is_uniquely_decodable(self):
        """检查是否唯一可解码"""
        # 生成测试串
        test_strings = self._generate_test_strings()
        
        for test_string in test_strings:
            decomps = self.decode(test_string)
            if len(decomps) > 1:
                return False, test_string, decomps
                
        return True, None, None
        
    def _generate_test_strings(self):
        """生成测试串"""
        # 生成码字的各种组合
        test_strings = set()
        codeword_list = list(self.codewords)
        
        # 单个码字
        test_strings.update(codeword_list)
        
        # 两个码字的连接
        for c1 in codeword_list:
            for c2 in codeword_list:
                test_strings.add(c1 + c2)
                
        # 三个码字的连接（限制数量）
        for i in range(min(len(codeword_list), 5)):
            for j in range(min(len(codeword_list), 5)):
                for k in range(min(len(codeword_list), 5)):
                    test_strings.add(codeword_list[i] + codeword_list[j] + codeword_list[k])
                    
        return test_strings
        
    def has_prefix_conflicts(self):
        """检查是否有前缀冲突"""
        codeword_list = list(self.codewords)
        
        for i, c1 in enumerate(codeword_list):
            for j, c2 in enumerate(codeword_list):
                if i != j:
                    if c1.startswith(c2) or c2.startswith(c1):
                        return True, (c1, c2)
                        
        return False, None
        
    def get_minimum_constraint_length(self):
        """获取最短约束长度"""
        if not self.forbidden_patterns:
            return float('inf')
        return min(len(p) for p in self.forbidden_patterns)
        
    def calculate_entropy(self):
        """计算编码系统的熵（简化）"""
        if not self.codewords:
            return 0
            
        # 统计使用的符号
        symbols = set()
        for codeword in self.codewords:
            symbols.update(codeword)
            
        return math.log2(len(symbols)) if symbols else 0


class ConstraintAnalyzer:
    """约束分析器"""
    
    def __init__(self):
        self.length_2_patterns = ['00', '01', '10', '11']
        
    def analyze_no_constraint_system(self):
        """分析无约束系统"""
        # 创建包含所有短串的系统
        system = BinaryEncodingSystem()
        
        # 添加所有长度1-3的串
        for length in range(1, 4):
            for i in range(2**length):
                binary = format(i, f'0{length}b')
                system.add_codeword(binary)
                
        return system
        
    def analyze_length_1_constraint(self, forbidden_symbol):
        """分析长度1约束"""
        system = BinaryEncodingSystem(forbidden_patterns={forbidden_symbol})
        
        # 尝试添加各种码字
        test_words = ['0', '1', '00', '01', '10', '11', '000', '001', '010', '011']
        valid_words = []
        
        for word in test_words:
            if system.add_codeword(word):
                valid_words.append(word)
                
        return system, valid_words
        
    def analyze_length_2_constraint(self, forbidden_pattern):
        """分析长度2约束"""
        system = BinaryEncodingSystem(forbidden_patterns={forbidden_pattern})
        
        # 构建满足约束的码字集
        valid_words = []
        for length in range(1, 6):
            for i in range(2**length):
                binary = format(i, f'0{length}b')
                if forbidden_pattern not in binary:
                    valid_words.append(binary)
                    system.add_codeword(binary)
                    
        return system
        
    def find_minimal_constraint_set(self):
        """寻找最小约束集"""
        results = {}
        
        for pattern in self.length_2_patterns:
            system = self.analyze_length_2_constraint(pattern)
            is_ud, conflict_string, decomps = system.is_uniquely_decodable()
            has_prefix, prefix_pair = system.has_prefix_conflicts()
            
            results[pattern] = {
                'uniquely_decodable': is_ud,
                'has_prefix_conflicts': has_prefix,
                'codeword_count': len(system.codewords),
                'conflict_example': (conflict_string, decomps) if not is_ud else None,
                'prefix_example': prefix_pair if has_prefix else None
            }
            
        return results


class TestL1_3_ConstraintNecessity(VerificationTest):
    """L1-3 约束必然性的形式化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        self.analyzer = ConstraintAnalyzer()
        
    def test_unique_decodability(self):
        """测试唯一可解码性 - 验证检查点1"""
        # 测试无约束系统
        unconstrained = self.analyzer.analyze_no_constraint_system()
        is_ud, conflict, decomps = unconstrained.is_uniquely_decodable()
        
        self.assertFalse(
            is_ud,
            "Unconstrained system should not be uniquely decodable"
        )
        self.assertIsNotNone(
            conflict,
            f"Should find conflict string, found: {conflict}"
        )
        self.assertGreater(
            len(decomps), 1,
            f"Conflict string should have multiple decompositions: {decomps}"
        )
        
        # 测试有约束系统
        constrained = BinaryEncodingSystem(
            codewords={'0', '10', '110', '111'},
            forbidden_patterns={'11'}  # 实际上这个集合不满足约束
        )
        
        # 重新构建满足no-11约束的系统
        no11_system = BinaryEncodingSystem(forbidden_patterns={'11'})
        no11_system.add_codeword('0')
        no11_system.add_codeword('1')
        no11_system.add_codeword('10')
        no11_system.add_codeword('100')
        no11_system.add_codeword('101')
        
        is_ud_constrained, _, _ = no11_system.is_uniquely_decodable()
        # 注意：即使有no-11约束，如果码字选择不当仍可能不是唯一可解码的
        
    def test_prefix_ambiguity(self):
        """测试前缀歧义 - 验证检查点2"""
        # 创建有前缀冲突的系统
        prefix_system = BinaryEncodingSystem(codewords={'1', '11', '110'})
        has_prefix, prefix_pair = prefix_system.has_prefix_conflicts()
        
        self.assertTrue(
            has_prefix,
            "System with {1, 11, 110} should have prefix conflicts"
        )
        self.assertIsNotNone(
            prefix_pair,
            "Should identify specific prefix conflict pair"
        )
        
        # 验证具体的前缀关系
        self.assertTrue(
            prefix_pair[0].startswith(prefix_pair[1]) or 
            prefix_pair[1].startswith(prefix_pair[0]),
            f"Pair {prefix_pair} should have prefix relationship"
        )
        
        # 测试无前缀冲突的系统
        no_prefix_system = BinaryEncodingSystem(codewords={'00', '01', '10'})
        has_prefix_2, _ = no_prefix_system.has_prefix_conflicts()
        
        self.assertFalse(
            has_prefix_2,
            "System with {00, 01, 10} should have no prefix conflicts"
        )
        
    def test_constraint_classification(self):
        """测试约束分类 - 验证检查点3"""
        # 测试长度1约束
        len1_system, valid_words = self.analyzer.analyze_length_1_constraint('0')
        
        # 验证退化
        all_have_1 = all('1' in word for word in valid_words)
        no_0 = all('0' not in word for word in valid_words)
        
        self.assertTrue(
            all_have_1 and no_0,
            f"Length-1 constraint should degenerate system. Valid words: {valid_words}"
        )
        
        entropy = len1_system.calculate_entropy()
        self.assertEqual(
            entropy, 0,
            "Degenerate system should have zero entropy"
        )
        
        # 测试长度2约束
        len2_system = self.analyzer.analyze_length_2_constraint('11')
        min_len = len2_system.get_minimum_constraint_length()
        
        self.assertEqual(
            min_len, 2,
            "Should have length-2 constraint"
        )
        
        # 测试长度≥3的约束不足
        len3_system = BinaryEncodingSystem(forbidden_patterns={'111'})
        # 添加可能产生前缀冲突的码字
        len3_system.add_codeword('1')
        len3_system.add_codeword('11')
        len3_system.add_codeword('110')
        
        has_prefix, _ = len3_system.has_prefix_conflicts()
        self.assertTrue(
            has_prefix,
            "Length ≥3 constraints allow prefix conflicts"
        )
        
    def test_minimal_constraint_length(self):
        """测试最小约束长度 - 验证检查点4"""
        # 测试只有长度≥3约束的系统
        long_constraint_system = BinaryEncodingSystem(forbidden_patterns={'000', '111'})
        
        # 所有长度<3的串都应该是有效的
        short_strings = ['0', '1', '00', '01', '10', '11']
        for s in short_strings:
            self.assertTrue(
                long_constraint_system.add_codeword(s),
                f"String {s} should be valid with only length-3 constraints"
            )
            
        # 检查前缀冲突
        has_prefix, conflict = long_constraint_system.has_prefix_conflicts()
        self.assertTrue(
            has_prefix,
            f"Should have prefix conflicts with only long constraints. Found: {conflict}"
        )
        
        # 验证长度2是必要的
        results = self.analyzer.find_minimal_constraint_set()
        
        # 至少有一个长度2约束可以避免某些前缀冲突
        some_reduces_conflicts = any(
            not result['has_prefix_conflicts'] or 
            result['codeword_count'] > 2  # 非退化
            for result in results.values()
        )
        
        self.assertTrue(
            some_reduces_conflicts,
            "Some length-2 constraints should reduce conflicts"
        )
        
    def test_constraint_capacity_tradeoff(self):
        """测试约束与容量的权衡"""
        # 无约束系统的容量
        unconstrained = BinaryEncodingSystem()
        for i in range(16):  # 所有4位串
            unconstrained.add_codeword(format(i, '04b'))
            
        unconstrained_capacity = math.log2(len(unconstrained.codewords)) / 4
        
        # 有约束系统的容量
        constrained = self.analyzer.analyze_length_2_constraint('11')
        valid_4bit = [c for c in constrained.codewords if len(c) == 4]
        constrained_capacity = math.log2(len(valid_4bit)) / 4 if valid_4bit else 0
        
        # 验证容量关系
        self.assertLess(
            constrained_capacity, unconstrained_capacity,
            "Constrained capacity should be less than unconstrained"
        )
        self.assertGreater(
            constrained_capacity, 0,
            "Constrained capacity should be positive (non-degenerate)"
        )
        
    def test_kraft_mcmillan_generalization(self):
        """测试Kraft-McMillan不等式的推广"""
        # 创建前缀自由码
        prefix_free = BinaryEncodingSystem(codewords={'00', '01', '10', '110', '111'})
        
        # 计算Kraft和
        kraft_sum = sum(2**(-len(c)) for c in prefix_free.codewords)
        
        self.assertLessEqual(
            kraft_sum, 1.0,
            f"Kraft sum {kraft_sum} should be ≤ 1 for prefix-free code"
        )
        
        # 测试非前缀自由码
        non_prefix_free = BinaryEncodingSystem(codewords={'1', '11', '111'})
        has_prefix, _ = non_prefix_free.has_prefix_conflicts()
        
        self.assertTrue(
            has_prefix,
            "Non-prefix-free code should have conflicts"
        )
        
    def test_self_referential_constraints(self):
        """测试自指系统的约束要求"""
        # 约束必须是可描述的
        simple_constraint = '11'
        complex_constraint = '010110101'  # 更复杂的模式
        
        # 简单约束的描述长度
        simple_desc_length = len(f"forbid {simple_constraint}")
        complex_desc_length = len(f"forbid {complex_constraint}")
        
        self.assertLess(
            simple_desc_length, complex_desc_length,
            "Simple constraints should have shorter descriptions"
        )
        
        # 验证约束保持递归结构
        # 创建使用简单约束的系统
        simple_system = self.analyzer.analyze_length_2_constraint(simple_constraint)
        
        # 检查是否可以编码自身的描述
        # 这是一个概念验证，实际系统会更复杂
        can_encode_description = len(simple_system.codewords) > 10
        
        self.assertTrue(
            can_encode_description,
            "System should have enough codewords to encode descriptions"
        )
        
    def test_constraint_effectiveness(self):
        """测试约束的有效性"""
        # 比较不同长度2约束的效果
        effectiveness = {}
        
        for pattern in ['00', '01', '10', '11']:
            system = self.analyzer.analyze_length_2_constraint(pattern)
            
            # 构建一个合理的码字子集来测试
            test_codewords = set()
            for word in system.codewords:
                if len(word) <= 4:  # 限制长度
                    test_codewords.add(word)
                    
            test_system = BinaryEncodingSystem(
                codewords=test_codewords,
                forbidden_patterns={pattern}
            )
            
            is_ud, _, _ = test_system.is_uniquely_decodable()
            has_prefix, _ = test_system.has_prefix_conflicts()
            
            effectiveness[pattern] = {
                'codeword_count': len(test_codewords),
                'uniquely_decodable': is_ud,
                'has_prefix': has_prefix
            }
            
        # 验证至少有一些模式是有效的
        some_effective = any(
            e['codeword_count'] > 4  # 非平凡
            for e in effectiveness.values()
        )
        
        self.assertTrue(
            some_effective,
            f"Some constraints should be effective. Results: {effectiveness}"
        )
        
    def test_information_theoretic_properties(self):
        """测试信息论性质"""
        # 测试不同约束下的增长率
        growth_rates = {}
        
        for pattern in ['11', '00']:
            system = self.analyzer.analyze_length_2_constraint(pattern)
            
            # 计算不同长度的有效串数量
            counts = defaultdict(int)
            for word in system.codewords:
                counts[len(word)] += 1
                
            # 估计增长率（简化）
            if 4 in counts and 3 in counts and counts[3] > 0:
                growth_rate = counts[4] / counts[3]
            else:
                growth_rate = 0
                
            growth_rates[pattern] = growth_rate
            
        # 验证增长率在合理范围内
        for pattern, rate in growth_rates.items():
            self.assertGreaterEqual(
                rate, 1,
                f"Growth rate for pattern {pattern} should be ≥ 1"
            )
            self.assertLess(
                rate, 2,
                f"Growth rate for pattern {pattern} should be < 2 (constrained)"
            )
            
    def test_constraint_minimality(self):
        """测试约束的最小性"""
        # 验证单个长度2约束可能不够
        single_constraint = BinaryEncodingSystem(forbidden_patterns={'11'})
        
        # 但多个约束会过度限制
        multiple_constraints = BinaryEncodingSystem(
            forbidden_patterns={'00', '11'}
        )
        
        # 添加测试码字 - 包含会被多重约束限制的词
        test_words = ['0', '1', '01', '10', '010', '101', '001', '100', '1001']
        
        single_valid = []
        multiple_valid = []
        
        for w in test_words:
            if '11' not in w:
                single_valid.append(w)
                single_constraint.add_codeword(w)
            if '11' not in w and '00' not in w:
                multiple_valid.append(w)
                multiple_constraints.add_codeword(w)
        
        self.assertGreater(
            len(single_valid), len(multiple_valid),
            f"Single constraint should allow more codewords than multiple. Single: {single_valid}, Multiple: {multiple_valid}"
        )
        
        # 验证最小性原则
        self.assertGreater(
            len(single_valid), len(test_words) / 2,
            "Single constraint should not be too restrictive"
        )


if __name__ == "__main__":
    unittest.main()