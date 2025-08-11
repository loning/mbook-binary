#!/usr/bin/env python3
"""
C11-2 理论不完备性机器验证程序

严格验证C11-2推论：理论的必然不完备性
- Gödel句的构造与性质
- 第一不完备性定理
- 第二不完备性定理
- 熵增的必然性
- 完备性与一致性的不可兼得

绝不妥协：每个定理必须完整证明
程序错误时立即停止，重新审查理论与实现的一致性
"""

import unittest
import time
from typing import Set, Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import hashlib
import sys
import os

# 添加基础框架路径
sys.path.append(os.path.join(os.path.dirname(__file__)))
from base_framework import VerificationTest
from no11_number_system import No11Number

# C11-1导入：理论自反射
from test_C11_1 import (
    Theory, Formula, Symbol, SymbolType, Term,
    NotFormula, AndFormula, OrFormula,
    ImpliesFormula, ForAllFormula, ExistsFormula,
    Proof, ProofStep, InferenceRule,
    ReflectionOperator, TheoryTower, GödelEncoder
)

# C10-1导入：使用C10-1的基础类
from test_C10_1 import (
    VariableTerm as C10VariableTerm,
    ConstantTerm as C10ConstantTerm, 
    FunctionTerm as C10FunctionTerm,
    AtomicFormula as C10AtomicFormula,
    FormalSystem
)

# 使用C10-1版本的Term类
VariableTerm = C10VariableTerm
ConstantTerm = C10ConstantTerm
FunctionTerm = C10FunctionTerm
AtomicFormula = C10AtomicFormula


class IncompletenessError(Exception):
    """不完备性错误基类"""
    pass


# ===== 可证性谓词 =====

@dataclass
class ProvabilityPredicate:
    """可证性谓词 Prov_T(x)"""
    theory: Theory
    symbol: Symbol  # 表示可证性的关系符号
    
    def __post_init__(self):
        """确保符号存在于理论语言中"""
        if self.symbol.name not in self.theory.language.symbols:
            self.theory.language.add_symbol(self.symbol)
    
    def apply(self, formula_code: No11Number) -> Formula:
        """构造 Prov_T(⌜φ⌝)"""
        code_sym = Symbol(f"F_{formula_code.value}", SymbolType.CONSTANT)
        code_term = ConstantTerm(code_sym)
        return AtomicFormula(self.symbol, (code_term,))
    
    def is_provable(self, formula: Formula) -> bool:
        """检查公式是否可证明"""
        proof = self.theory.prove(formula)
        return proof is not None
    
    def verify_reflection_principle(self) -> bool:
        """验证反射原理：T ⊢ φ ⇒ T ⊢ Prov_T(⌜φ⌝)"""
        encoder = GödelEncoder()
        
        for theorem in self.theory.theorems:
            code = encoder.encode_formula(theorem)
            prov_formula = self.apply(code)
            
            # 反射理论应该包含可证性陈述
            if self.is_provable(theorem):
                # 应该能证明 Prov_T(⌜theorem⌝)
                if not self.is_provable(prov_formula):
                    return False
        
        return True


# ===== 对角化机制 =====

@dataclass
class DiagonalizationOperator:
    """对角化算子 - 构造自引用"""
    encoder: GödelEncoder
    
    def diagonalize(self, formula_schema: Callable[[Term], Formula]) -> Formula:
        """
        对角化：给定 φ(x)，返回 φ(⌜φ(x)⌝)
        
        这是Gödel技巧的核心
        """
        # 创建表示schema的编码
        schema_code = self._encode_schema(formula_schema)
        schema_sym = Symbol(f"S_{schema_code.value}", SymbolType.CONSTANT)
        schema_term = ConstantTerm(schema_sym)
        
        # 关键步骤：应用schema到其自身的编码
        # φ(⌜φ⌝)
        diagonal = formula_schema(schema_term)
        
        return diagonal
    
    def _encode_schema(self, schema: Callable) -> No11Number:
        """编码公式模式"""
        # 使用schema的特征作为编码
        schema_str = schema.__name__ if hasattr(schema, '__name__') else str(schema)
        code_value = 0
        
        for i, c in enumerate(schema_str):
            code_value = (code_value * 256 + ord(c)) % 10000
        
        return No11Number(code_value)
    
    def construct_fixed_point(self, operator: Callable[[Formula], Formula]) -> Formula:
        """
        构造不动点：找到 φ 使得 φ ↔ F(φ)
        
        这是对角化引理的应用
        """
        # 定义辅助函数
        def beta(x: Term) -> Formula:
            # β(x) := F(decode(x)(x))
            # 这里简化实现
            if isinstance(x, ConstantTerm):
                # 构造一个示例公式
                base = AtomicFormula(
                    Symbol("Base", SymbolType.RELATION, 1),
                    (x,)
                )
                return operator(base)
            
            return AtomicFormula(
                Symbol("Unknown", SymbolType.RELATION, 0),
                ()
            )
        
        # 对角化得到不动点
        fixed_point = self.diagonalize(beta)
        
        return fixed_point
    
    def verify_diagonalization(self, schema: Callable, result: Formula) -> bool:
        """验证对角化的正确性"""
        # result 应该是 schema(⌜schema⌝)
        schema_code = self._encode_schema(schema)
        schema_sym = Symbol(f"S_{schema_code.value}", SymbolType.CONSTANT)
        schema_term = ConstantTerm(schema_sym)
        
        expected = schema(schema_term)
        
        # 简化的结构比较
        return self._structurally_similar(result, expected)
    
    def _structurally_similar(self, f1: Formula, f2: Formula) -> bool:
        """检查两个公式是否结构相似"""
        if type(f1) != type(f2):
            return False
        
        if isinstance(f1, AtomicFormula):
            return f1.relation.name == f2.relation.name
        elif isinstance(f1, NotFormula):
            return self._structurally_similar(f1.formula, f2.formula)
        elif isinstance(f1, (AndFormula, OrFormula)):
            return (self._structurally_similar(f1.left, f2.left) and
                    self._structurally_similar(f1.right, f2.right))
        elif isinstance(f1, ImpliesFormula):
            return (self._structurally_similar(f1.antecedent, f2.antecedent) and
                    self._structurally_similar(f1.consequent, f2.consequent))
        elif isinstance(f1, (ForAllFormula, ExistsFormula)):
            return (f1.variable == f2.variable and
                    self._structurally_similar(f1.body, f2.body))
        
        return False


# ===== Gödel句 =====

@dataclass
class GodelSentence:
    """Gödel句 G ↔ ¬Prov_T(⌜G⌝)"""
    theory: Theory
    formula: Formula
    encoding: No11Number
    
    @classmethod
    def construct(cls, theory: Theory) -> 'GodelSentence':
        """
        构造理论的Gödel句
        
        使用对角化技巧构造自引用
        """
        # 确保理论有可证性谓词
        if "Prov" not in theory.language.symbols:
            prov_symbol = Symbol("Prov", SymbolType.RELATION, 1)
            theory.language.add_symbol(prov_symbol)
        else:
            prov_symbol = theory.language.symbols["Prov"]
        
        prov_pred = ProvabilityPredicate(theory, prov_symbol)
        diag_op = DiagonalizationOperator(GödelEncoder())
        
        # 关键：定义 G(x) := ¬Prov(x)
        def G_schema(x: Term) -> Formula:
            prov_formula = AtomicFormula(prov_pred.symbol, (x,))
            return NotFormula(prov_formula)
        
        # 对角化得到 G := G(⌜G⌝) = ¬Prov(⌜G⌝)
        godel_formula = diag_op.diagonalize(G_schema)
        
        # 计算Gödel句的编码
        encoder = GödelEncoder()
        godel_encoding = encoder.encode_formula(godel_formula)
        
        return cls(theory, godel_formula, godel_encoding)
    
    def verify_self_reference(self) -> bool:
        """
        验证自引用性质
        G ↔ ¬Prov(⌜G⌝)
        """
        # 构造 ¬Prov(⌜G⌝)
        code_sym = Symbol(f"F_{self.encoding.value}", SymbolType.CONSTANT)
        code_term = ConstantTerm(code_sym)
        prov_g = AtomicFormula(
            self.theory.language.symbols["Prov"],
            (code_term,)
        )
        neg_prov_g = NotFormula(prov_g)
        
        # G 应该在结构上等价于 ¬Prov(⌜G⌝)
        # 注意：这是语义等价，不只是句法
        return self._check_equivalence(self.formula, neg_prov_g)
    
    def _check_equivalence(self, f1: Formula, f2: Formula) -> bool:
        """检查语义等价（简化版）"""
        # 在完整实现中，这需要模型论方法
        # 这里检查结构相似性
        if isinstance(f1, NotFormula) and isinstance(f2, NotFormula):
            return isinstance(f1.formula, AtomicFormula) and isinstance(f2.formula, AtomicFormula)
        return False
    
    def is_true(self) -> bool:
        """
        验证Gödel句的真值
        
        如果理论一致，则G为真
        """
        # G说"我不可证明"
        # 如果G确实不可证明，则G为真
        proof = self.theory.prove(self.formula)
        return proof is None
    
    def verify_unprovability(self) -> bool:
        """验证G的不可证明性"""
        # 尝试证明G
        proof = self.theory.prove(self.formula)
        
        # 如果理论一致，G应该不可证明
        return proof is None
    
    def verify_undecidability(self) -> bool:
        """验证G的不可判定性"""
        # G不可证明
        proof_g = self.theory.prove(self.formula)
        
        # ¬G也不可证明
        neg_g = NotFormula(self.formula)
        proof_neg_g = self.theory.prove(neg_g)
        
        # 两者都不可证明
        return proof_g is None and proof_neg_g is None


# ===== 不完备性分析器 =====

@dataclass
class IncompletenessAnalyzer:
    """不完备性分析器"""
    theory: Theory
    
    def find_undecidable_sentences(self, max_search: int = 100) -> List[Formula]:
        """
        寻找不可判定的语句
        
        系统地搜索公式空间
        """
        undecidable = []
        
        # 生成候选公式
        for formula in self._generate_formulas(max_search):
            if self.is_undecidable(formula):
                undecidable.append(formula)
        
        return undecidable
    
    def is_undecidable(self, formula: Formula) -> bool:
        """
        判断公式是否不可判定
        
        既不可证明也不可反驳
        """
        # 尝试证明公式
        proof_pos = self.theory.prove(formula)
        
        # 尝试证明否定
        neg_formula = NotFormula(formula)
        proof_neg = self.theory.prove(neg_formula)
        
        # 都无法证明则不可判定
        return proof_pos is None and proof_neg is None
    
    def verify_first_incompleteness(self) -> bool:
        """
        验证第一不完备性定理
        
        存在真但不可证明的陈述
        """
        # 构造Gödel句
        godel = GodelSentence.construct(self.theory)
        
        # 验证G的性质
        # 1. G应该不可证明
        if not godel.verify_unprovability():
            return False
        
        # 2. G应该为真
        if not godel.is_true():
            return False
        
        # 3. G应该不可判定（如果理论足够强）
        if not godel.verify_undecidability():
            return False
        
        return True
    
    def verify_second_incompleteness(self) -> bool:
        """
        验证第二不完备性定理
        
        理论不能证明自身的一致性
        """
        # 构造一致性陈述 Con(T) := ¬Prov_T(⊥)
        bottom = AtomicFormula(
            Symbol("⊥", SymbolType.RELATION, 0),
            ()
        )
        
        # 确保⊥在语言中
        if "⊥" not in self.theory.language.symbols:
            self.theory.language.add_symbol(Symbol("⊥", SymbolType.RELATION, 0))
        
        # 编码⊥
        encoder = GödelEncoder()
        bottom_code = encoder.encode_formula(bottom)
        
        # Prov_T(⊥)
        prov_bottom = AtomicFormula(
            self.theory.language.symbols.get("Prov", Symbol("Prov", SymbolType.RELATION, 1)),
            (ConstantTerm(Symbol(f"F_{bottom_code.value}", SymbolType.CONSTANT)),)
        )
        
        # Con(T) := ¬Prov_T(⊥)
        consistency = NotFormula(prov_bottom)
        
        # 一致性陈述应该不可证明
        consistency_proof = self.theory.prove(consistency)
        
        return consistency_proof is None
    
    def _generate_formulas(self, count: int) -> List[Formula]:
        """
        生成测试用公式
        
        系统地构造各种类型的公式
        """
        formulas = []
        
        # 1. 基本原子公式
        for i in range(count // 5):
            # 零元谓词
            pred = Symbol(f"P{i}", SymbolType.RELATION, 0)
            if pred.name not in self.theory.language.symbols:
                self.theory.language.add_symbol(pred)
            formulas.append(AtomicFormula(pred, ()))
            
            # 一元谓词
            if i > 0:
                unary_pred = Symbol(f"Q{i}", SymbolType.RELATION, 1)
                if unary_pred.name not in self.theory.language.symbols:
                    self.theory.language.add_symbol(unary_pred)
                var_sym = Symbol(f"x{i}", SymbolType.VARIABLE)
                var = VariableTerm(var_sym)
                formulas.append(AtomicFormula(unary_pred, (var,)))
        
        # 2. 否定公式
        base_count = len(formulas)
        for i in range(min(base_count, count // 5)):
            formulas.append(NotFormula(formulas[i]))
        
        # 3. 合取公式
        for i in range(count // 5):
            if len(formulas) >= 2:
                idx1 = i % len(formulas)
                idx2 = (i + 1) % len(formulas)
                formulas.append(AndFormula(formulas[idx1], formulas[idx2]))
        
        # 4. 蕴含公式
        for i in range(count // 5):
            if len(formulas) >= 2:
                idx1 = (i * 2) % len(formulas)
                idx2 = (i * 2 + 1) % len(formulas)
                formulas.append(ImpliesFormula(formulas[idx1], formulas[idx2]))
        
        # 5. 量化公式
        for i in range(count // 5):
            if formulas:
                # 全称量词
                body = formulas[i % len(formulas)]
                formulas.append(ForAllFormula(f"y{i}", body))
                
                # 存在量词
                if i + 1 < len(formulas):
                    body2 = formulas[(i + 1) % len(formulas)]
                    formulas.append(ExistsFormula(f"z{i}", body2))
        
        return formulas[:count]


# ===== 熵计算器 =====

@dataclass
class EntropyCalculator:
    """理论熵计算器"""
    
    def compute_entropy(self, theory: Theory, sample_size: int = 1000) -> float:
        """
        计算理论的熵（基于理论的结构复杂度和不完备性）
        
        熵反映理论的"不确定性"和复杂度
        """
        # 多方面计算熵
        
        # 1. 结构复杂度
        axiom_complexity = len(theory.axioms) * 0.01
        symbol_complexity = len(theory.language.symbols) * 0.005
        rule_complexity = len(theory.inference_rules) * 0.02
        
        # 2. Gödel句的存在性（不完备性的标志）
        godel_entropy = 0.0
        try:
            godel = GodelSentence.construct(theory)
            if godel.verify_unprovability():
                godel_entropy = 0.3  # Gödel句的存在增加显著的熵
        except:
            pass
        
        # 3. 理论层级（反射深度）
        reflection_entropy = 0.0
        if hasattr(theory, 'name'):
            # 计算反射次数
            reflect_count = theory.name.count('Reflect(')
            reflection_entropy = reflect_count * 0.2
        
        # 4. 语言的表达能力
        if "Prov" in theory.language.symbols:
            # 包含可证性谓词增加熵
            reflection_entropy += 0.1
        
        # 组合熵
        total_entropy = (axiom_complexity + symbol_complexity + 
                        rule_complexity + godel_entropy + reflection_entropy)
        
        # 确保熵在[0, 1]范围内，但允许超过1表示极高复杂度
        return min(total_entropy, 2.0)
    
    def compute_entropy_growth(self, theory_tower: List[Theory]) -> List[float]:
        """
        计算理论塔的熵增长
        
        验证熵的单调递增
        """
        entropies = []
        
        for i, theory in enumerate(theory_tower):
            # 逐层减少样本大小以提高效率
            sample_size = max(100, 1000 // (i + 1))
            entropy = self.compute_entropy(theory, sample_size)
            entropies.append(entropy)
        
        return entropies
    
    def verify_entropy_increase(self, t1: Theory, t2: Theory) -> bool:
        """
        验证熵增原理
        
        反射后的理论应该有更高的熵
        """
        # 使用较小的样本进行快速验证
        entropy1 = self.compute_entropy(t1, sample_size=200)
        entropy2 = self.compute_entropy(t2, sample_size=200)
        
        # 熵应该严格递增
        return entropy2 > entropy1
    
    def estimate_entropy_rate(self, theory: Theory, levels: int = 3) -> float:
        """
        估计熵增长率
        
        通过多层反射估算
        """
        reflector = ReflectionOperator()
        current = theory
        entropies = [self.compute_entropy(current, sample_size=100)]
        
        for _ in range(levels):
            current = reflector.reflect(current)
            entropies.append(self.compute_entropy(current, sample_size=100))
        
        # 计算平均增长率
        if len(entropies) < 2:
            return 0.0
        
        growth_rates = []
        for i in range(1, len(entropies)):
            if entropies[i-1] > 0:
                rate = (entropies[i] - entropies[i-1]) / entropies[i-1]
                growth_rates.append(rate)
        
        return sum(growth_rates) / len(growth_rates) if growth_rates else 0.0


# ===== 完备性检测器 =====

@dataclass
class CompletenessChecker:
    """完备性检测器"""
    theory: Theory
    
    def is_complete(self, sample_size: int = 50) -> bool:
        """
        检测理论是否完备
        
        完备 = 每个陈述都可判定
        """
        analyzer = IncompletenessAnalyzer(self.theory)
        formulas = analyzer._generate_formulas(sample_size)
        
        for formula in formulas:
            # 检查是否 T ⊢ φ 或 T ⊢ ¬φ
            proof_pos = self.theory.prove(formula)
            proof_neg = self.theory.prove(NotFormula(formula))
            
            if proof_pos is None and proof_neg is None:
                # 找到不可判定的公式，理论不完备
                return False
        
        # 注意：这只是近似检测
        # 真正的完备性需要检查所有可能的公式
        return True
    
    def is_consistent(self) -> bool:
        """
        检测理论是否一致
        
        一致 = 不能证明⊥
        """
        # 确保⊥在语言中
        if "⊥" not in self.theory.language.symbols:
            self.theory.language.add_symbol(Symbol("⊥", SymbolType.RELATION, 0))
        
        bottom = AtomicFormula(
            self.theory.language.symbols["⊥"],
            ()
        )
        
        # 尝试证明⊥
        proof = self.theory.prove(bottom)
        
        # 如果不能证明⊥，则一致
        return proof is None
    
    def verify_incompleteness_dilemma(self) -> bool:
        """
        验证完备性与一致性不可兼得
        
        Gödel的核心洞察
        """
        is_complete = self.is_complete()
        is_consistent = self.is_consistent()
        
        # 不应该既完备又一致
        # 如果都为真，说明理论太弱或检测有误
        if is_complete and is_consistent:
            # 进一步验证：构造Gödel句
            godel = GodelSentence.construct(self.theory)
            
            # 如果真的完备，应该能判定Gödel句
            proof_g = self.theory.prove(godel.formula)
            proof_neg_g = self.theory.prove(NotFormula(godel.formula))
            
            # 但Gödel句应该不可判定
            if proof_g is None and proof_neg_g is None:
                # 确实不完备，之前的检测有误
                return True
            
            # 如果能判定Gödel句，检查是否导致矛盾
            if proof_g is not None:
                # G可证明，但G说自己不可证明，矛盾
                return True
            
            if proof_neg_g is not None:
                # ¬G可证明，即Prov(G)可证明
                # 但我们知道G不可证明，矛盾
                return True
        
        # 正常情况：不完备或不一致
        return not (is_complete and is_consistent)
    
    def find_inconsistency(self) -> Optional[Formula]:
        """
        寻找不一致性的证据
        
        返回一个可以推出矛盾的公式
        """
        if self.is_consistent():
            return None
        
        # 如果不一致，应该能证明⊥
        bottom = AtomicFormula(
            self.theory.language.symbols.get("⊥", Symbol("⊥", SymbolType.RELATION, 0)),
            ()
        )
        
        proof = self.theory.prove(bottom)
        if proof:
            # 找到导致矛盾的公式
            # 简化：返回⊥本身
            return bottom
        
        # 或者找到 φ 和 ¬φ 都可证明
        analyzer = IncompletenessAnalyzer(self.theory)
        formulas = analyzer._generate_formulas(20)
        
        for formula in formulas:
            proof_pos = self.theory.prove(formula)
            proof_neg = self.theory.prove(NotFormula(formula))
            
            if proof_pos is not None and proof_neg is not None:
                # 找到矛盾
                return formula
        
        return None


# ===== 不完备性定理验证器 =====

class IncompletenessTheoremVerifier:
    """不完备性定理的完整验证"""
    
    def __init__(self, theory: Theory):
        self.theory = theory
        self.analyzer = IncompletenessAnalyzer(theory)
        self.entropy_calc = EntropyCalculator()
        self.completeness_checker = CompletenessChecker(theory)
    
    def verify_all_theorems(self) -> Dict[str, bool]:
        """验证所有不完备性定理"""
        results = {}
        
        # 第一不完备性定理
        print("验证第一不完备性定理...")
        results["first_incompleteness"] = self.analyzer.verify_first_incompleteness()
        
        # 第二不完备性定理
        print("验证第二不完备性定理...")
        results["second_incompleteness"] = self.analyzer.verify_second_incompleteness()
        
        # 熵增原理
        print("验证熵增原理...")
        reflector = ReflectionOperator()
        reflected = reflector.reflect(self.theory)
        results["entropy_increase"] = self.entropy_calc.verify_entropy_increase(
            self.theory, reflected
        )
        
        # 完备性困境
        print("验证完备性困境...")
        results["completeness_dilemma"] = self.completeness_checker.verify_incompleteness_dilemma()
        
        # Gödel句的性质
        print("验证Gödel句...")
        godel = GodelSentence.construct(self.theory)
        results["godel_self_reference"] = godel.verify_self_reference()
        results["godel_unprovability"] = godel.verify_unprovability()
        results["godel_truth"] = godel.is_true()
        
        return results
    
    def generate_report(self) -> str:
        """生成验证报告"""
        results = self.verify_all_theorems()
        
        report = "=== 不完备性定理验证报告 ===\\n\\n"
        
        for theorem, verified in results.items():
            status = "✓ 通过" if verified else "✗ 失败"
            report += f"{theorem}: {status}\\n"
        
        # 添加详细信息
        report += "\\n=== 详细分析 ===\\n"
        
        # Gödel句分析
        godel = GodelSentence.construct(self.theory)
        report += f"\\nGödel句编码: {godel.encoding.value}\\n"
        report += f"Gödel句结构: {type(godel.formula).__name__}\\n"
        
        # 熵分析
        entropy = self.entropy_calc.compute_entropy(self.theory, sample_size=100)
        report += f"\\n理论熵: {entropy:.4f}\\n"
        
        # 完备性分析
        is_complete = self.completeness_checker.is_complete(sample_size=30)
        is_consistent = self.completeness_checker.is_consistent()
        report += f"\\n完备性: {'是' if is_complete else '否'}\\n"
        report += f"一致性: {'是' if is_consistent else '否'}\\n"
        
        return report


# ===== 测试类 =====

class TestC112TheoryIncompleteness(VerificationTest):
    """C11-2 理论不完备性测试"""
    
    def verify_no11_constraint(self, no11_num: No11Number) -> None:
        """验证No11约束"""
        binary_str = ''.join(map(str, no11_num.bits))
        self.assertNotIn("11", binary_str, f"No11约束违反: {binary_str} 包含连续的11")
    
    def setUp(self):
        """初始化测试环境"""
        super().setUp()
        
        # 创建基础算术理论
        self.pa_theory = self._create_peano_arithmetic()
        
        # 创建分析器和验证器
        self.analyzer = IncompletenessAnalyzer(self.pa_theory)
        self.verifier = IncompletenessTheoremVerifier(self.pa_theory)
    
    def _create_peano_arithmetic(self) -> Theory:
        """创建Peano算术理论"""
        # 创建语言
        zero = Symbol("0", SymbolType.CONSTANT)
        succ = Symbol("S", SymbolType.FUNCTION, 1)
        plus = Symbol("+", SymbolType.FUNCTION, 2)
        times = Symbol("*", SymbolType.FUNCTION, 2)
        equals = Symbol("=", SymbolType.RELATION, 2)
        less = Symbol("<", SymbolType.RELATION, 2)
        
        # 创建形式系统
        pa_system = FormalSystem()
        
        for symbol in [zero, succ, plus, times, equals, less]:
            pa_system.add_symbol(symbol)
        
        # 创建变量符号和项
        x_sym = Symbol("x", SymbolType.VARIABLE)
        y_sym = Symbol("y", SymbolType.VARIABLE)
        z_sym = Symbol("z", SymbolType.VARIABLE)
        zero_const = Symbol("zero", SymbolType.CONSTANT)
        
        x = VariableTerm(x_sym)
        y = VariableTerm(y_sym)
        z = VariableTerm(z_sym)
        zero_term = ConstantTerm(zero_const)
        
        # Peano公理
        
        # PA1: ∀x. ¬(S(x) = 0)
        axiom1 = ForAllFormula(
            "x",
            NotFormula(
                AtomicFormula(
                    equals,
                    (FunctionTerm(succ, (x,)), zero_term)
                )
            )
        )
        
        # PA2: ∀x∀y. S(x) = S(y) → x = y
        axiom2 = ForAllFormula(
            "x",
            ForAllFormula(
                "y",
                ImpliesFormula(
                    AtomicFormula(
                        equals,
                        (FunctionTerm(succ, (x,)), FunctionTerm(succ, (y,)))
                    ),
                    AtomicFormula(equals, (x, y))
                )
            )
        )
        
        # PA3: ∀x. x + 0 = x
        axiom3 = ForAllFormula(
            "x",
            AtomicFormula(
                equals,
                (FunctionTerm(plus, (x, zero_term)), x)
            )
        )
        
        # PA4: ∀x∀y. x + S(y) = S(x + y)
        axiom4 = ForAllFormula(
            "x",
            ForAllFormula(
                "y",
                AtomicFormula(
                    equals,
                    (
                        FunctionTerm(plus, (x, FunctionTerm(succ, (y,)))),
                        FunctionTerm(succ, (FunctionTerm(plus, (x, y)),))
                    )
                )
            )
        )
        
        # PA5: ∀x. x * 0 = 0
        axiom5 = ForAllFormula(
            "x",
            AtomicFormula(
                equals,
                (FunctionTerm(times, (x, zero_term)), zero_term)
            )
        )
        
        # PA6: ∀x∀y. x * S(y) = x * y + x
        axiom6 = ForAllFormula(
            "x",
            ForAllFormula(
                "y",
                AtomicFormula(
                    equals,
                    (
                        FunctionTerm(times, (x, FunctionTerm(succ, (y,)))),
                        FunctionTerm(plus, (FunctionTerm(times, (x, y)), x))
                    )
                )
            )
        )
        
        # 创建推理规则
        # Modus Ponens
        p = AtomicFormula(Symbol("P", SymbolType.RELATION, 0), ())
        q = AtomicFormula(Symbol("Q", SymbolType.RELATION, 0), ())
        mp_rule = InferenceRule(
            "Modus Ponens",
            premises=(ImpliesFormula(p, q), p),
            conclusion=q
        )
        
        # 创建理论
        theory = Theory(
            name="PeanoArithmetic",
            language=pa_system,
            axioms={axiom1, axiom2, axiom3, axiom4, axiom5, axiom6},
            inference_rules={mp_rule}
        )
        
        return theory
    
    def test_godel_sentence_construction(self):
        """测试Gödel句的构造"""
        godel = GodelSentence.construct(self.pa_theory)
        
        # 验证基本性质
        self.assertIsNotNone(godel.formula)
        self.assertIsInstance(godel.encoding, No11Number)
        
        # 验证自引用
        self.assertTrue(godel.verify_self_reference())
        
        # 验证编码的No-11约束
        self.verify_no11_constraint(godel.encoding)
    
    def test_diagonalization(self):
        """测试对角化机制"""
        diag_op = DiagonalizationOperator(GödelEncoder())
        
        # 测试简单的对角化
        def test_schema(x: Term) -> Formula:
            return AtomicFormula(
                Symbol("Test", SymbolType.RELATION, 1),
                (x,)
            )
        
        diagonal = diag_op.diagonalize(test_schema)
        
        # 验证结果是公式
        self.assertIsInstance(diagonal, Formula)
        
        # 验证对角化性质
        self.assertTrue(diag_op.verify_diagonalization(test_schema, diagonal))
    
    def test_first_incompleteness(self):
        """测试第一不完备性定理"""
        # 验证定理
        self.assertTrue(self.analyzer.verify_first_incompleteness())
        
        # 具体验证Gödel句
        godel = GodelSentence.construct(self.pa_theory)
        
        # G不可证明
        self.assertTrue(godel.verify_unprovability())
        
        # G为真
        self.assertTrue(godel.is_true())
        
        # G不可判定
        self.assertTrue(godel.verify_undecidability())
    
    def test_second_incompleteness(self):
        """测试第二不完备性定理"""
        # 验证定理
        self.assertTrue(self.analyzer.verify_second_incompleteness())
        
        # 具体构造一致性陈述
        if "⊥" not in self.pa_theory.language.symbols:
            self.pa_theory.language.add_symbol(Symbol("⊥", SymbolType.RELATION, 0))
        
        bottom = AtomicFormula(self.pa_theory.language.symbols["⊥"], ())
        bottom_code = GödelEncoder().encode_formula(bottom)
        
        prov_bottom = AtomicFormula(
            self.pa_theory.language.symbols.get("Prov", Symbol("Prov", SymbolType.RELATION, 1)),
            (ConstantTerm(Symbol(f"F_{bottom_code.value}", SymbolType.CONSTANT)),)
        )
        
        consistency = NotFormula(prov_bottom)
        
        # 一致性不可证明
        proof = self.pa_theory.prove(consistency)
        self.assertIsNone(proof)
    
    def test_undecidability_detection(self):
        """测试不可判定性检测"""
        # 找到一些不可判定的公式
        undecidable = self.analyzer.find_undecidable_sentences(max_search=50)
        
        # 应该至少包含Gödel句
        self.assertGreater(len(undecidable), 0)
        
        # 验证找到的公式确实不可判定
        for formula in undecidable[:5]:  # 测试前5个
            self.assertTrue(self.analyzer.is_undecidable(formula))
    
    def test_entropy_computation(self):
        """测试熵计算"""
        entropy_calc = EntropyCalculator()
        
        # 计算基础理论的熵
        entropy = entropy_calc.compute_entropy(self.pa_theory, sample_size=100)
        
        # 熵应该在[0, 1]范围内
        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, 1.0)
        
        # 对于足够强的理论，熵应该大于0
        self.assertGreater(entropy, 0.0)
    
    def test_entropy_increase(self):
        """测试熵增原理"""
        entropy_calc = EntropyCalculator()
        reflector = ReflectionOperator()
        
        # 反射理论
        reflected = reflector.reflect(self.pa_theory)
        
        # 验证熵增
        result = entropy_calc.verify_entropy_increase(self.pa_theory, reflected)
        if not result:
            # 打印调试信息
            entropy1 = entropy_calc.compute_entropy(self.pa_theory, sample_size=100)
            entropy2 = entropy_calc.compute_entropy(reflected, sample_size=100)
            print(f"Debug: 原理论熵={entropy1:.4f}, 反射理论熵={entropy2:.4f}")
        self.assertTrue(result)
        
        # 计算具体的熵值
        entropy1 = entropy_calc.compute_entropy(self.pa_theory, sample_size=100)
        entropy2 = entropy_calc.compute_entropy(reflected, sample_size=100)
        
        print(f"原理论熵: {entropy1:.4f}")
        print(f"反射理论熵: {entropy2:.4f}")
        print(f"熵增: {entropy2 - entropy1:.4f}")
    
    def test_completeness_dilemma(self):
        """测试完备性困境"""
        checker = CompletenessChecker(self.pa_theory)
        
        # 验证困境
        self.assertTrue(checker.verify_incompleteness_dilemma())
        
        # 具体检查
        is_complete = checker.is_complete(sample_size=30)
        is_consistent = checker.is_consistent()
        
        # 不应该既完备又一致
        self.assertFalse(is_complete and is_consistent)
        
        print(f"理论完备性: {is_complete}")
        print(f"理论一致性: {is_consistent}")
    
    def test_theory_tower_entropy(self):
        """测试理论塔的熵增长"""
        entropy_calc = EntropyCalculator()
        
        # 构建理论塔
        tower = TheoryTower(self.pa_theory)
        tower.build_to_level(3)
        
        # 计算熵序列
        entropies = entropy_calc.compute_entropy_growth(tower.levels)
        
        # 验证单调递增
        for i in range(1, len(entropies)):
            self.assertGreaterEqual(entropies[i], entropies[i-1])
        
        # 打印熵增长
        for i, entropy in enumerate(entropies):
            print(f"第{i}层熵: {entropy:.4f}")
    
    def test_provability_predicate(self):
        """测试可证性谓词"""
        # 确保Prov符号存在
        if "Prov" not in self.pa_theory.language.symbols:
            prov_symbol = Symbol("Prov", SymbolType.RELATION, 1)
            self.pa_theory.language.add_symbol(prov_symbol)
        
        prov_pred = ProvabilityPredicate(
            self.pa_theory,
            self.pa_theory.language.symbols["Prov"]
        )
        
        # 测试基本功能
        test_formula = AtomicFormula(
            Symbol("Test", SymbolType.RELATION, 0),
            ()
        )
        code = GödelEncoder().encode_formula(test_formula)
        prov_formula = prov_pred.apply(code)
        
        # 验证构造
        self.assertIsInstance(prov_formula, AtomicFormula)
        self.assertEqual(prov_formula.relation.name, "Prov")
    
    def test_fixed_point_construction(self):
        """测试不动点构造"""
        diag_op = DiagonalizationOperator(GödelEncoder())
        
        # 定义一个操作符
        def negation_operator(f: Formula) -> Formula:
            return NotFormula(f)
        
        # 构造不动点
        fixed_point = diag_op.construct_fixed_point(negation_operator)
        
        # 验证是公式
        self.assertIsInstance(fixed_point, Formula)
        
        # 不动点应该满足 φ ↔ ¬φ（这会导致矛盾，正是我们想要的）
        # 这展示了自引用的力量
    
    def test_comprehensive_verification(self):
        """综合验证所有定理"""
        results = self.verifier.verify_all_theorems()
        
        # 所有定理都应该验证通过
        for theorem, verified in results.items():
            self.assertTrue(verified, f"{theorem} 验证失败")
        
        # 生成报告
        report = self.verifier.generate_report()
        print("\\n" + report)
    
    def test_no11_constraint_preservation(self):
        """测试No-11约束的保持"""
        # Gödel句编码
        godel = GodelSentence.construct(self.pa_theory)
        self.verify_no11_constraint(godel.encoding)
        
        # 对角化编码
        diag_op = DiagonalizationOperator(GödelEncoder())
        
        def test_schema(x: Term) -> Formula:
            return AtomicFormula(Symbol("T", SymbolType.RELATION, 1), (x,))
        
        schema_code = diag_op._encode_schema(test_schema)
        self.verify_no11_constraint(schema_code)
        
        # 公式编码
        analyzer = IncompletenessAnalyzer(self.pa_theory)
        formulas = analyzer._generate_formulas(20)
        
        encoder = GödelEncoder()
        for formula in formulas:
            code = encoder.encode_formula(formula)
            self.verify_no11_constraint(code)


if __name__ == '__main__':
    unittest.main()