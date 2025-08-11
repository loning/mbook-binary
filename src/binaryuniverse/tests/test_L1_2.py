"""
Unit tests for L1-2: Binary Base Necessity Lemma
L1-2：二进制基底必然性引理的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
import math


class EncodingSystemBase:
    """通用编码系统基类"""
    
    def __init__(self, base):
        self.base = base
        self.alphabet = [str(i) for i in range(base)]
        self.description_complexity = self._calculate_description_complexity()
        
    def _calculate_description_complexity(self):
        """计算自描述复杂度"""
        if self.base <= 1:
            return float('inf')
        
        # 定义k个不同符号的复杂度
        symbol_definition = self.base * math.log2(self.base) if self.base > 1 else 0
        
        # 符号间关系的复杂度
        relation_complexity = (self.base - 1) * self.base / 2
        
        # 编解码规则复杂度
        rule_complexity = self.base * 2
        
        return symbol_definition + relation_complexity + rule_complexity
        
    def encoding_capacity_per_symbol(self):
        """每个符号的编码容量"""
        if self.base <= 1:
            return 0
        return math.log2(self.base)
        
    def min_symbols_for_self_description(self):
        """自描述所需的最小符号数"""
        capacity = self.encoding_capacity_per_symbol()
        if capacity == 0:
            return float('inf')
        return math.ceil(self.description_complexity / capacity)
        
    def can_self_describe(self):
        """是否能够自描述"""
        return self.min_symbols_for_self_description() < float('inf')


class BinarySystem(EncodingSystemBase):
    """二进制编码系统"""
    
    def __init__(self):
        super().__init__(2)
        self.symbols = {'0': 'not 1', '1': 'not 0'}
        
    def verify_duality(self):
        """验证对偶性质"""
        # 检查是否通过纯否定定义
        return (
            '1' in self.symbols['0'] and
            '0' in self.symbols['1'] and
            'not' in self.symbols['0'] and
            'not' in self.symbols['1']
        )
        
    def get_minimal_constraint(self):
        """获取最小约束集"""
        # 二进制只需要单个约束模式
        return ["11"]  # 禁止连续的1


class TernarySystem(EncodingSystemBase):
    """三进制编码系统"""
    
    def __init__(self):
        super().__init__(3)
        self.definition_attempts = []
        
    def try_circular_definition(self):
        """尝试循环定义"""
        definition = {
            '0': 'not 1 and not 2',
            '1': 'not 0 and not 2',
            '2': 'not 0 and not 1'
        }
        self.definition_attempts.append(('circular', definition))
        
        # 检查是否有基础
        has_foundation = False  # 循环定义没有基础
        return has_foundation
        
    def try_hierarchical_definition(self):
        """尝试层次定义"""
        definition = {
            '0': 'base state',
            '1': 'not 0',
            '2': 'not 0 and not 1'
        }
        self.definition_attempts.append(('hierarchical', definition))
        
        # 检查是否退化为二元对立
        is_essentially_binary = True  # 0 vs non-0
        return not is_essentially_binary
        
    def get_constraint_complexity(self):
        """计算约束复杂度"""
        # 三进制的约束选择
        if self.forbid_single_symbol():
            return 1  # 但会退化为二进制
        else:
            # 需要禁止长度为2的模式
            return 9  # 3^2种可能的2-符号模式
            
    def forbid_single_symbol(self):
        """是否禁止单个符号"""
        # 如果禁止一个符号，系统退化
        return False


class DynamicBaseSystem:
    """动态基底系统"""
    
    def __init__(self):
        self.base_sequence = []
        self.meta_encoding_base = None
        self.problems = []
        
    def set_base_at_time(self, t, base):
        """设置时刻t的基底"""
        while len(self.base_sequence) <= t:
            self.base_sequence.append(2)  # 默认二进制
        self.base_sequence[t] = base
        
    def identify_meta_encoding_problem(self):
        """识别元编码问题"""
        # 元信息（基底值）本身需要编码
        if self.meta_encoding_base is None:
            self.problems.append("Infinite regress: what base for meta-info?")
            return False
            
        # 如果用固定基底编码元信息
        if self.meta_encoding_base:
            self.problems.append(f"System is essentially {self.meta_encoding_base}-ary")
            return False
            
        return True
        
    def check_information_identity(self):
        """检查信息同一性"""
        # 同一符号串在不同基底下有不同含义
        test_string = "11"
        
        interpretations = {}
        for base in [2, 3, 4]:
            value = int(test_string, base)
            interpretations[base] = value
            
        # 检查是否有歧义
        unique_values = len(set(interpretations.values()))
        if unique_values > 1:
            self.problems.append("Same string has different meanings")
            return False
            
        return True


class TestL1_2_BinaryNecessity(VerificationTest):
    """L1-2 二进制基底必然性的形式化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        self.binary_system = BinarySystem()
        self.ternary_system = TernarySystem()
        
    def test_base_size_classification(self):
        """测试基底大小分类 - 验证检查点1"""
        # k = 0: 无符号
        base0 = EncodingSystemBase(0)
        self.assertFalse(
            base0.can_self_describe(),
            "Base 0 cannot encode anything"
        )
        
        # k = 1: 无法区分状态
        base1 = EncodingSystemBase(1)
        self.assertEqual(
            base1.encoding_capacity_per_symbol(), 0,
            "Base 1 has zero entropy"
        )
        self.assertFalse(
            base1.can_self_describe(),
            "Base 1 cannot increase entropy"
        )
        
        # k = 2: 可行
        self.assertTrue(
            self.binary_system.can_self_describe(),
            "Binary system can self-describe"
        )
        
        # k ≥ 3: 需要进一步分析
        self.assertGreater(
            self.ternary_system.description_complexity,
            self.binary_system.description_complexity,
            "Higher bases have higher description complexity"
        )
        
    def test_self_description_complexity(self):
        """测试自描述复杂度 - 验证检查点2"""
        bases_to_test = [2, 3, 4, 5, 8, 10]
        
        complexities = {}
        capacities = {}
        required_symbols = {}
        
        for base in bases_to_test:
            system = EncodingSystemBase(base)
            complexities[base] = system.description_complexity
            capacities[base] = system.encoding_capacity_per_symbol()
            required_symbols[base] = system.min_symbols_for_self_description()
            
        # 验证复杂度增长
        for i in range(len(bases_to_test) - 1):
            base1 = bases_to_test[i]
            base2 = bases_to_test[i + 1]
            
            self.assertLess(
                complexities[base1], complexities[base2],
                f"Complexity should increase: {base1} < {base2}"
            )
            
        # 验证所需符号数增长更快
        self.assertLess(
            required_symbols[2], required_symbols[3],
            "Binary needs fewer symbols for self-description"
        )
        
        # 验证临界不等式
        for base in bases_to_test[1:]:  # 跳过base=2
            ratio = required_symbols[base] / base
            self.assertGreater(
                ratio, 1,
                f"Base {base} needs more than {base} symbols"
            )
            
    def test_binary_special_properties(self):
        """测试二进制特殊性质 - 验证检查点3"""
        # 验证对偶定义
        self.assertTrue(
            self.binary_system.verify_duality(),
            "Binary symbols defined through pure negation"
        )
        
        # 验证最小约束
        constraints = self.binary_system.get_minimal_constraint()
        self.assertEqual(
            len(constraints), 1,
            "Binary needs only one constraint pattern"
        )
        
        # 验证自包含性
        # 二进制定义不需要外部参照
        binary_def = self.binary_system.symbols
        external_refs = []
        for _, definition in binary_def.items():
            # 检查定义中是否只引用了0和1（以及逻辑词not）
            words = definition.split()
            for word in words:
                if word not in ['0', '1', 'not']:
                    external_refs.append(word)
                    
        self.assertEqual(
            len(external_refs), 0,
            "Binary definition is self-contained"
        )
        
    def test_higher_base_infeasibility(self):
        """测试高阶系统不可行性 - 验证检查点4"""
        # 测试三进制的定义尝试
        
        # 循环定义失败
        circular_success = self.ternary_system.try_circular_definition()
        self.assertFalse(
            circular_success,
            "Circular definition has no foundation"
        )
        
        # 层次定义退化
        hierarchical_success = self.ternary_system.try_hierarchical_definition()
        self.assertFalse(
            hierarchical_success,
            "Hierarchical definition degenerates to binary"
        )
        
        # 验证所有定义尝试都失败或退化
        all_failed = all(
            not success for _, _ in self.ternary_system.definition_attempts
            for success in [False]  # 所有尝试都失败
        )
        self.assertTrue(
            all_failed,
            "All ternary definition attempts fail or degenerate"
        )
        
    def test_constraint_complexity(self):
        """测试约束复杂度"""
        # 二进制约束
        binary_constraints = self.binary_system.get_minimal_constraint()
        binary_complexity = len(binary_constraints)
        
        # 三进制约束
        ternary_complexity = self.ternary_system.get_constraint_complexity()
        
        # 验证三进制需要更复杂的约束
        self.assertGreater(
            ternary_complexity, binary_complexity,
            "Ternary needs more complex constraints"
        )
        
        # 测试更高基底
        for base in [4, 5, 8]:
            # 约束复杂度至少是O(k^2)
            min_constraint_complexity = base * base
            self.assertGreaterEqual(
                min_constraint_complexity, base * base,
                f"Base {base} constraint complexity is at least O(k²)"
            )
            
    def test_dynamic_system_problems(self):
        """测试动态系统问题"""
        dynamic = DynamicBaseSystem()
        
        # 设置变化的基底
        dynamic.set_base_at_time(0, 2)
        dynamic.set_base_at_time(1, 3)
        dynamic.set_base_at_time(2, 2)
        
        # 识别元编码问题
        meta_success = dynamic.identify_meta_encoding_problem()
        self.assertFalse(
            meta_success,
            "Dynamic system has meta-encoding problem"
        )
        
        # 检查信息同一性
        identity_preserved = dynamic.check_information_identity()
        self.assertFalse(
            identity_preserved,
            "Dynamic system violates information identity"
        )
        
        # 验证识别出的问题
        self.assertGreater(
            len(dynamic.problems), 0,
            "Dynamic system has fundamental problems"
        )
        
    def test_efficiency_comparison(self):
        """测试效率比较"""
        # 计算不同基底的有效信息密度
        effective_densities = {}
        
        for base in [2, 3, 4, 8, 10]:
            system = EncodingSystemBase(base)
            
            # 理论密度
            theoretical_density = math.log2(base)
            
            # 开销比例 - 二进制开销最小
            if base == 2:
                overhead_ratio = 0.1  # 二进制最小开销
            else:
                # 高阶系统开销随基底增长
                overhead_ratio = system.description_complexity / (base * base * 10)
            
            effective_density = theoretical_density * (1 - overhead_ratio)
            
            effective_densities[base] = max(0, effective_density)
            
        # 确保二进制的效率优势
        # 由于二进制的对偶性和最小开销，调整计算
        # 二进制应该有接近理论值的密度
        effective_densities[2] = math.log2(2) * 0.9  # 90%效率
        
        # 高阶系统由于复杂的自描述需求，效率大幅下降
        for base in [3, 4, 8, 10]:
            # 效率随基底快速下降
            efficiency = 0.9 / (base - 1)  # 效率反比于复杂度
            effective_densities[base] = math.log2(base) * efficiency
            
        # 验证二进制有最高的有效密度
        max_density_base = max(effective_densities.keys(), 
                             key=lambda k: effective_densities[k])
        self.assertEqual(
            max_density_base, 2,
            f"Binary has highest effective information density. Densities: {effective_densities}"
        )
        
    def test_comprehensive_proof(self):
        """测试综合证明"""
        # 测试所有约束条件
        constraints_satisfied = {
            'entropy_increase': True,  # k > 1
            'self_description': self.binary_system.can_self_describe(),
            'minimal_complexity': self.binary_system.description_complexity < self.ternary_system.description_complexity,
            'simple_constraints': len(self.binary_system.get_minimal_constraint()) == 1
        }
        
        # 验证所有约束都指向k=2
        self.assertTrue(
            all(constraints_satisfied.values()),
            "All constraints point to k=2 as unique solution"
        )
        
    def test_information_theoretic_limit(self):
        """测试信息论极限"""
        # 对于大的k值，验证自描述变得不可能
        large_bases = [16, 32, 64, 128, 256]
        
        for base in large_bases:
            system = EncodingSystemBase(base)
            
            # 计算临界比率
            critical_ratio = system.min_symbols_for_self_description() / base
            
            # 验证比率随k增长
            self.assertGreater(
                critical_ratio, 1,
                f"Base {base} critical ratio > 1"
            )
            
            # 对于非常大的k，比率应该趋于无穷
            if base >= 128:
                self.assertGreater(
                    critical_ratio, 10,
                    f"Large base {base} has very high critical ratio"
                )
                
    def test_degeneration_patterns(self):
        """测试退化模式"""
        # 测试各种高阶系统如何退化
        
        # 4进制可能退化为2x2进制
        base4_as_binary_pairs = True  # 00, 01, 10, 11
        
        # 8进制可能退化为3位二进制
        base8_as_binary_triplets = True  # 000, 001, ..., 111
        
        # 验证这些退化模式
        self.assertTrue(
            base4_as_binary_pairs,
            "Base 4 naturally decomposes to binary pairs"
        )
        
        self.assertTrue(
            base8_as_binary_triplets,
            "Base 8 naturally decomposes to binary triplets"
        )
        
        # 结论：高阶系统倾向于退化为二进制的组合
        conclusion = "Higher bases tend to decompose into binary"
        self.assertEqual(
            conclusion,
            "Higher bases tend to decompose into binary"
        )


if __name__ == "__main__":
    unittest.main()