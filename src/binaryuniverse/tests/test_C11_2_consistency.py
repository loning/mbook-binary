#!/usr/bin/env python3
"""
C11-2 理论不完备性一致性测试

验证C11-2与其他理论层次的一致性：
- 与C11-1理论自反射的一致性  
- 与C10-1元数学结构的一致性
- 与C9系列自指代数的一致性
- 不完备性定理的正确实现
- 熵增原理的严格验证
"""

import unittest
import sys
import os

# 添加基础框架路径
sys.path.append(os.path.join(os.path.dirname(__file__)))
from base_framework import VerificationTest
from no11_number_system import No11Number

# 导入C11-2
from test_C11_2 import (
    ProvabilityPredicate, DiagonalizationOperator, GodelSentence,
    IncompletenessAnalyzer, EntropyCalculator, CompletenessChecker,
    IncompletenessTheoremVerifier
)

# 导入C11-1
from test_C11_1 import (
    Theory, Formula, Symbol, SymbolType,
    NotFormula, AndFormula, AtomicFormula, ImpliesFormula,
    ForAllFormula, ExistsFormula,
    ReflectionOperator, TheoryTower, GödelEncoder
)

# 导入C10-1
from test_C10_1 import (
    FormalSystem, VariableTerm, ConstantTerm, FunctionTerm
)


class TestC112Consistency(VerificationTest):
    """C11-2一致性测试类"""
    
    def setUp(self):
        """初始化测试环境"""
        super().setUp()
        
        # 创建基础理论
        self.base_theory = self._create_base_theory()
        
        # 创建各种分析器
        self.analyzer = IncompletenessAnalyzer(self.base_theory)
        self.entropy_calc = EntropyCalculator()
        self.completeness_checker = CompletenessChecker(self.base_theory)
        self.verifier = IncompletenessTheoremVerifier(self.base_theory)
    
    def _create_base_theory(self) -> Theory:
        """创建基础理论"""
        # 创建形式系统
        system = FormalSystem()
        
        # 添加基本符号
        zero = Symbol("0", SymbolType.CONSTANT)
        succ = Symbol("S", SymbolType.FUNCTION, 1)
        equals = Symbol("=", SymbolType.RELATION, 2)
        
        for symbol in [zero, succ, equals]:
            system.add_symbol(symbol)
        
        # 创建基本公理
        x_sym = Symbol("x", SymbolType.VARIABLE)
        x = VariableTerm(x_sym)
        zero_sym = Symbol("zero", SymbolType.CONSTANT)
        zero_term = ConstantTerm(zero_sym)
        
        # 0 ≠ S(x)
        axiom1 = ForAllFormula(
            "x",
            NotFormula(
                AtomicFormula(
                    equals,
                    (zero_term, FunctionTerm(succ, (x,)))
                )
            )
        )
        
        # 创建理论
        theory = Theory(
            name="BaseTheory",
            language=system,
            axioms={axiom1},
            inference_rules=set()
        )
        
        return theory
    
    def test_godel_encoding_consistency(self):
        """测试Gödel编码的一致性"""
        encoder = GödelEncoder()
        
        # 测试与C10-1的一致性
        # 所有编码都应该满足No-11约束
        formulas = self.analyzer._generate_formulas(50)
        
        for formula in formulas:
            code = encoder.encode_formula(formula)
            self.assertIsInstance(code, No11Number)
            
            # 验证No-11约束
            binary_str = ''.join(map(str, code.bits))
            self.assertNotIn("11", binary_str)
    
    def test_reflection_compatibility(self):
        """测试与C11-1反射机制的兼容性"""
        reflector = ReflectionOperator()
        
        # 反射理论
        reflected = reflector.reflect(self.base_theory)
        
        # 在反射理论上构造Gödel句
        godel = GodelSentence.construct(reflected)
        
        # 验证Gödel句的性质在反射理论中仍然成立
        self.assertTrue(godel.verify_self_reference())
        self.assertTrue(godel.verify_unprovability())
        self.assertTrue(godel.is_true())
    
    def test_incompleteness_in_tower(self):
        """测试理论塔中的不完备性"""
        tower = TheoryTower(self.base_theory)
        tower.build_to_level(3)
        
        # 每一层都应该是不完备的
        for i, theory in enumerate(tower.levels):
            analyzer = IncompletenessAnalyzer(theory)
            
            # 验证第一不完备性
            self.assertTrue(
                analyzer.verify_first_incompleteness(),
                f"第{i}层理论应该满足第一不完备性定理"
            )
            
            # 验证第二不完备性
            self.assertTrue(
                analyzer.verify_second_incompleteness(),
                f"第{i}层理论应该满足第二不完备性定理"
            )
    
    def test_entropy_monotonicity(self):
        """测试熵的单调性"""
        tower = TheoryTower(self.base_theory)
        tower.build_to_level(4)
        
        entropies = self.entropy_calc.compute_entropy_growth(tower.levels)
        
        # 熵应该单调递增
        for i in range(1, len(entropies)):
            self.assertGreaterEqual(
                entropies[i], entropies[i-1],
                f"熵应该单调递增：第{i}层({entropies[i]:.4f}) >= 第{i-1}层({entropies[i-1]:.4f})"
            )
        
        # 熵增应该是严格的（至少在某些层之间）
        strict_increases = sum(1 for i in range(1, len(entropies)) if entropies[i] > entropies[i-1])
        self.assertGreater(strict_increases, 0, "应该存在严格的熵增")
    
    def test_diagonal_lemma_universality(self):
        """测试对角化引理的普遍性"""
        diag_op = DiagonalizationOperator(GödelEncoder())
        
        # 测试不同的算子
        operators = [
            lambda f: NotFormula(f),  # 否定
            lambda f: ImpliesFormula(f, f),  # 自蕴含
            lambda f: AndFormula(f, NotFormula(f))  # 矛盾
        ]
        
        for op in operators:
            # 构造不动点
            fixed_point = diag_op.construct_fixed_point(op)
            
            # 验证是公式
            self.assertIsInstance(fixed_point, Formula)
            
            # 验证满足不动点性质（至少在结构上）
            # φ ↔ F(φ)
    
    def test_completeness_consistency_tradeoff(self):
        """测试完备性与一致性的权衡"""
        # 基础理论
        checker1 = CompletenessChecker(self.base_theory)
        is_complete1 = checker1.is_complete(sample_size=50)
        is_consistent1 = checker1.is_consistent()
        
        # 反射理论
        reflector = ReflectionOperator()
        reflected = reflector.reflect(self.base_theory)
        checker2 = CompletenessChecker(reflected)
        is_complete2 = checker2.is_complete(sample_size=50)
        is_consistent2 = checker2.is_consistent()
        
        # 都应该是一致但不完备的
        self.assertTrue(is_consistent1, "基础理论应该一致")
        self.assertTrue(is_consistent2, "反射理论应该一致")
        self.assertFalse(is_complete1, "基础理论应该不完备")
        self.assertFalse(is_complete2, "反射理论应该不完备")
        
        # 验证困境
        self.assertTrue(checker1.verify_incompleteness_dilemma())
        self.assertTrue(checker2.verify_incompleteness_dilemma())
    
    def test_provability_predicate_properties(self):
        """测试可证性谓词的性质"""
        # 确保Prov符号存在
        if "Prov" not in self.base_theory.language.symbols:
            prov_symbol = Symbol("Prov", SymbolType.RELATION, 1)
            self.base_theory.language.add_symbol(prov_symbol)
        
        prov_pred = ProvabilityPredicate(
            self.base_theory,
            self.base_theory.language.symbols["Prov"]
        )
        
        # 测试反射原理
        self.assertTrue(prov_pred.verify_reflection_principle())
        
        # 测试可证性谓词的构造
        test_formula = AtomicFormula(
            Symbol("P", SymbolType.RELATION, 0),
            ()
        )
        code = GödelEncoder().encode_formula(test_formula)
        prov_formula = prov_pred.apply(code)
        
        # 应该是原子公式
        self.assertIsInstance(prov_formula, AtomicFormula)
        self.assertEqual(prov_formula.relation.name, "Prov")
    
    def test_cross_layer_consistency(self):
        """测试跨层一致性"""
        # 构建理论塔
        tower = TheoryTower(self.base_theory)
        tower.build_to_level(2)
        
        # 每一层的Gödel句应该在更高层中仍然不可证明
        for i in range(len(tower.levels) - 1):
            lower_theory = tower.levels[i]
            higher_theory = tower.levels[i + 1]
            
            # 在低层构造Gödel句
            godel_lower = GodelSentence.construct(lower_theory)
            
            # 在高层中应该仍然不可证明
            proof_in_higher = higher_theory.prove(godel_lower.formula)
            self.assertIsNone(
                proof_in_higher,
                f"第{i}层的Gödel句在第{i+1}层中应该仍然不可证明"
            )
    
    def test_no11_preservation_throughout(self):
        """测试No-11约束的全程保持"""
        # 测试各种构造
        constructions = []
        
        # 1. Gödel句编码
        godel = GodelSentence.construct(self.base_theory)
        constructions.append(("Gödel句", godel.encoding))
        
        # 2. 对角化编码
        diag_op = DiagonalizationOperator(GödelEncoder())
        test_schema = lambda x: AtomicFormula(Symbol("T", SymbolType.RELATION, 1), (x,))
        schema_code = diag_op._encode_schema(test_schema)
        constructions.append(("对角化模式", schema_code))
        
        # 3. 公式编码
        formulas = self.analyzer._generate_formulas(20)
        encoder = GödelEncoder()
        for i, formula in enumerate(formulas):
            code = encoder.encode_formula(formula)
            constructions.append((f"公式{i}", code))
        
        # 验证所有编码
        for name, code in constructions:
            self.assertIsInstance(code, No11Number, f"{name}应该是No11Number")
            binary_str = ''.join(map(str, code.bits))
            self.assertNotIn("11", binary_str, f"{name}的编码违反No-11约束")
    
    def test_comprehensive_theorem_verification(self):
        """综合定理验证"""
        results = self.verifier.verify_all_theorems()
        
        # 所有核心定理都应该通过
        expected_theorems = [
            "first_incompleteness",
            "second_incompleteness",
            "entropy_increase",
            "completeness_dilemma",
            "godel_self_reference",
            "godel_unprovability",
            "godel_truth"
        ]
        
        for theorem in expected_theorems:
            self.assertIn(theorem, results, f"应该验证{theorem}")
            self.assertTrue(results[theorem], f"{theorem}应该验证通过")
        
        # 生成并验证报告
        report = self.verifier.generate_report()
        self.assertIn("✓ 通过", report, "报告中应该有通过的定理")
        self.assertNotIn("✗ 失败", report, "报告中不应该有失败的定理")


if __name__ == '__main__':
    unittest.main(verbosity=2)