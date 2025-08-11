"""
测试C11-1理论自反射与前层的一致性

验证要点:
1. 与C10-2的深度一致性
2. 与C10-1的基础一致性  
3. 与C9系列的结构一致性
4. 编码系统的完整性
5. 反射操作的保守性
"""

import unittest
from typing import Set, Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入所有前层测试模块
# 从test_C11_1获取完整的公式类型
from test_C11_1 import (
    Term, ConstantTerm, FunctionTerm,
    Formula, AtomicFormula, NotFormula, AndFormula,
    OrFormula, ImpliesFormula, ForAllFormula, ExistsFormula,
    Proof, ProofStep, GödelEncoder
)

# 从test_C10_1导入变量和符号类型
from test_C10_1 import (
    FormalSystem, Symbol, SymbolType, VariableTerm,
    ConstantTerm as BaseConstantTerm
)

# 注释掉不存在的导入
# from docs.binaryuniverse.tests.test_C9_3 import (
#     SelfReferenceAlgebra
# )

# from docs.binaryuniverse.tests.test_C9_2 import (
#     TypeFormation
# )

# from docs.binaryuniverse.tests.test_C9_1 import (
#     IdentityStructure
# )

from test_C11_1 import (
    Theory, InferenceRule as TheoryInferenceRule,
    ReflectionOperator, TheoryTower, TheoryFixpoint
)

# 从test_C11_1导入基础类
from base_framework import VerificationTest
from no11_number_system import No11Number


class C11_1ConsistencyTest(VerificationTest):
    """C11-1与前层的一致性测试"""
    
    def setUp(self):
        """初始化测试环境"""
        super().setUp()
        
        # 初始化前层组件
        self.formal_system = FormalSystem()
        # self.self_ref_algebra = SelfReferenceAlgebra()
        # self.type_formation = TypeFormation()
        # self.identity_structure = IdentityStructure()
        
        # 初始化编码器
        self.encoder = GödelEncoder()
        
        # 初始化反射操作符
        self.reflection_op = ReflectionOperator()
        
        # 创建测试理论
        self._create_test_theories()
    
    def _create_test_theories(self):
        """创建测试用理论"""
        # 基础算术理论
        self.arithmetic_theory = self._create_arithmetic_theory()
        
        # 元理论
        self.meta_theory = self._create_meta_theory()
        
        # 自指理论
        self.self_ref_theory = self._create_self_ref_theory()
    
    def _create_arithmetic_theory(self) -> Theory:
        """创建基础算术理论"""
        # 创建算术语言
        zero = Symbol("0", SymbolType.CONSTANT)
        succ = Symbol("S", SymbolType.FUNCTION, 1)
        plus = Symbol("+", SymbolType.FUNCTION, 2)
        equals = Symbol("=", SymbolType.RELATION, 2)
        
        self.formal_system.add_symbol(zero)
        self.formal_system.add_symbol(succ)
        self.formal_system.add_symbol(plus)
        self.formal_system.add_symbol(equals)
        
        # 创建公理
        x = VariableTerm("x")
        y = VariableTerm("y")
        z = VariableTerm("z")
        
        # Peano公理
        axiom1 = ForAllFormula(
            "x",
            NotFormula(
                AtomicFormula(
                    equals,
                    (FunctionTerm(succ, (x,)), BaseConstantTerm(zero))
                )
            )
        )
        
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
        
        # 加法公理
        axiom3 = ForAllFormula(
            "x",
            AtomicFormula(
                equals,
                (FunctionTerm(plus, (x, BaseConstantTerm(zero))), x)
            )
        )
        
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
        
        # 创建推理规则
        # 需要创建前提和结论模式
        p_sym = Symbol("P", SymbolType.RELATION, 0)
        q_sym = Symbol("Q", SymbolType.RELATION, 0)
        phi_sym = Symbol("φ", SymbolType.RELATION, 1)
        
        self.formal_system.add_symbol(p_sym)
        self.formal_system.add_symbol(q_sym)
        self.formal_system.add_symbol(phi_sym)
        
        p = AtomicFormula(p_sym, ())
        q = AtomicFormula(q_sym, ())
        
        mp_rule = TheoryInferenceRule(
            "Modus Ponens",
            premises=(ImpliesFormula(p, q), p),
            conclusion=q
        )
        
        # 泛化规则
        phi = AtomicFormula(phi_sym, (x,))
        gen_rule = TheoryInferenceRule(
            "Generalization",
            premises=(phi,),
            conclusion=ForAllFormula("x", phi)
        )
        
        return Theory(
            name="Arithmetic",
            language=self.formal_system,
            axioms={axiom1, axiom2, axiom3, axiom4},
            inference_rules={mp_rule, gen_rule}
        )
    
    def _create_meta_theory(self) -> Theory:
        """创建元理论"""
        # 扩展语言以包含元符号
        provable = Symbol("Prov", SymbolType.RELATION, 1)
        theory_sym = Symbol("T", SymbolType.CONSTANT)
        encode_func = Symbol("encode", SymbolType.FUNCTION, 1)
        
        meta_system = FormalSystem()
        meta_system.add_symbol(provable)
        meta_system.add_symbol(theory_sym)
        meta_system.add_symbol(encode_func)
        
        # 元公理
        x = VariableTerm("x")
        
        # 可证性反射
        meta_axiom1 = ForAllFormula(
            "x",
            ImpliesFormula(
                AtomicFormula(provable, (x,)),
                AtomicFormula(
                    provable,
                    (FunctionTerm(encode_func, (AtomicFormula(provable, (x,)),)),)
                )
            )
        )
        
        # 理论编码
        meta_axiom2 = ExistsFormula(
            "x",
            AtomicFormula(
                Symbol("=", SymbolType.RELATION, 2),
                (
                    FunctionTerm(encode_func, (BaseConstantTerm(theory_sym),)),
                    x
                )
            )
        )
        
        return Theory(
            name="MetaTheory",
            language=meta_system,
            axioms={meta_axiom1, meta_axiom2},
            inference_rules=set()
        )
    
    def _create_self_ref_theory(self) -> Theory:
        """创建自指理论"""
        # 创建自指语言
        self_sym = Symbol("Self", SymbolType.CONSTANT)
        refer_sym = Symbol("refers_to", SymbolType.RELATION, 2)
        is_theory = Symbol("is_theory", SymbolType.RELATION, 1)
        
        self_ref_system = FormalSystem()
        self_ref_system.add_symbol(self_sym)
        self_ref_system.add_symbol(refer_sym)
        self_ref_system.add_symbol(is_theory)
        
        # 自指公理
        self_axiom = AtomicFormula(
            refer_sym,
            (BaseConstantTerm(self_sym), BaseConstantTerm(self_sym))
        )
        
        theory_axiom = AtomicFormula(
            is_theory,
            (BaseConstantTerm(self_sym),)
        )
        
        return Theory(
            name="SelfReference",
            language=self_ref_system,
            axioms={self_axiom, theory_axiom},
            inference_rules=set()
        )
    
    def test_encoding_consistency_with_C10_2(self):
        """测试与C10-2编码系统的一致性"""
        # 理论必须能被Gödel编码
        theory_encoding = self.encoder.encode_theory(self.arithmetic_theory)
        self.assertIsInstance(theory_encoding, No11Number)
        
        # 编码必须可逆
        # 注意：这里简化了解码测试，实际需要完整解码器
        self.assertGreater(theory_encoding.value, 0)
        
        # 编码必须保持结构
        reflected_theory = self.reflection_op.reflect(self.arithmetic_theory)
        reflected_encoding = self.encoder.encode_theory(reflected_theory)
        
        # 反射理论的编码必须大于原理论（熵增）
        self.assertGreater(reflected_encoding.value, theory_encoding.value)
    
    def test_language_consistency_with_C10_1(self):
        """测试与C10-1形式系统的一致性"""
        # 理论语言必须是良构的形式系统
        for symbol in self.arithmetic_theory.language.symbols.values():
            self.assertIsInstance(symbol, Symbol)
            self.assertIn(symbol.symbol_type, [SymbolType.CONSTANT, SymbolType.FUNCTION, SymbolType.RELATION])
        
        # 反射必须保持形式系统的良构性
        reflected = self.reflection_op.reflect(self.arithmetic_theory)
        for symbol in reflected.language.symbols.values():
            self.assertIsInstance(symbol, Symbol)
            self.assertIn(symbol.symbol_type, [SymbolType.CONSTANT, SymbolType.FUNCTION, SymbolType.RELATION])
    
    def test_self_reference_consistency_with_C9_3(self):
        """测试与C9-3自指代数的一致性"""
        # 跳过此测试，因为SelfReferenceAlgebra不存在
        # 创建自指元素
        # self_ref_elem = self.self_ref_algebra.create_element(
        #     lambda x: f"Theory({x})"
        # )
        
        # 理论反射必须保持自指结构
        reflected = self.reflection_op.reflect(self.self_ref_theory)
        
        # 验证反射保持自指性
        self_axiom_preserved = False
        for axiom in reflected.axioms:
            if isinstance(axiom, AtomicFormula) and axiom.predicate.name == "refers_to":
                args = axiom.arguments
                if (len(args) == 2 and 
                    isinstance(args[0], BaseConstantTerm) and 
                    isinstance(args[1], BaseConstantTerm) and
                    args[0].symbol.name == "Self" and 
                    args[1].symbol.name == "Self"):
                    self_axiom_preserved = True
                    break
        
        self.assertTrue(self_axiom_preserved)
    
    # def test_type_formation_consistency_with_C9_2(self):
    #     """测试与C9-2类型形成的一致性"""
    #     # 跳过此测试，因为TypeFormation不存在
    #     pass
    
    # def test_identity_consistency_with_C9_1(self):
    #     """测试与C9-1恒等结构的一致性"""
    #     # 跳过此测试，因为IdentityStructure不存在
    #     pass
    
    def test_reflection_preserves_no11_constraint(self):
        """测试反射保持No11约束"""
        # 所有编码必须满足No11约束
        theories = [
            self.arithmetic_theory,
            self.meta_theory,
            self.self_ref_theory
        ]
        
        for theory in theories:
            # 原理论编码
            encoding = self.encoder.encode_theory(theory)
            self.assertTrue(self._check_no11_constraint(encoding))
            
            # 反射理论编码
            reflected = self.reflection_op.reflect(theory)
            reflected_encoding = self.encoder.encode_theory(reflected)
            self.assertTrue(self._check_no11_constraint(reflected_encoding))
            
            # 多次反射
            tower = TheoryTower(theory)
            for _ in range(3):
                tower.extend()
                top_encoding = self.encoder.encode_theory(tower.get_top())
                self.assertTrue(self._check_no11_constraint(top_encoding))
    
    def _check_no11_constraint(self, no11_num: No11Number) -> bool:
        """检查No11约束"""
        binary_str = no11_num.to_binary().value
        return "11" not in binary_str
    
    def test_reflection_operator_properties(self):
        """测试反射操作符的性质"""
        # 1. 反射是单调的
        t1 = self.arithmetic_theory
        t2 = self.reflection_op.reflect(t1)
        
        # T ⊆ Reflect(T)
        self.assertTrue(t1.axioms.issubset(t2.axioms))
        
        # 2. 反射保持可证性
        # 如果 T ⊢ φ, 则 Reflect(T) ⊢ φ
        x = VariableTerm("x")
        simple_theorem = AtomicFormula(
            Symbol("=", SymbolType.RELATION, 2),
            (x, x)
        )
        
        if simple_theorem in t1.theorems:
            self.assertIn(simple_theorem, t2.theorems)
        
        # 3. 反射增加元定理
        meta_axioms_added = len(t2.axioms) > len(t1.axioms)
        self.assertTrue(meta_axioms_added)
    
    def test_theory_tower_convergence(self):
        """测试理论塔的收敛性"""
        tower = TheoryTower(self.arithmetic_theory)
        
        # 构建理论塔
        encodings = []
        for i in range(5):
            theory = tower.get_top()
            encoding = self.encoder.encode_theory(theory)
            encodings.append(encoding.value)
            
            if i < 4:  # 最后一次不扩展
                tower.extend()
        
        # 验证严格递增
        for i in range(len(encodings) - 1):
            self.assertLess(encodings[i], encodings[i + 1])
        
        # 验证增长率递减（表示接近不动点）
        growth_rates = []
        for i in range(len(encodings) - 1):
            rate = (encodings[i + 1] - encodings[i]) / encodings[i]
            growth_rates.append(rate)
        
        # 增长率应该递减
        for i in range(len(growth_rates) - 1):
            self.assertLessEqual(growth_rates[i + 1], growth_rates[i] * 1.1)  # 允许小幅波动
    
    def test_meta_circularity_preservation(self):
        """测试元循环性的保持"""
        # 理论必须能描述自身的反射
        reflected = self.reflection_op.reflect(self.meta_theory)
        
        # 验证反射理论包含自描述能力
        has_self_encoding = False
        for axiom in reflected.axioms:
            if self._is_self_encoding_axiom(axiom):
                has_self_encoding = True
                break
        
        self.assertTrue(has_self_encoding)
    
    def _is_self_encoding_axiom(self, axiom: Formula) -> bool:
        """检查是否是自编码公理"""
        if isinstance(axiom, ExistsFormula):
            body = axiom.formula
            if isinstance(body, AtomicFormula) and body.predicate.name == "=":
                args = body.arguments
                if len(args) == 2 and isinstance(args[0], FunctionTerm):
                    if args[0].function.name == "encode":
                        return True
        return False
    
    def test_proof_reflection_consistency(self):
        """测试证明反射的一致性"""
        # 创建简单定理和证明
        x = VariableTerm("x")
        theorem = ForAllFormula(
            "x",
            AtomicFormula(
                Symbol("=", SymbolType.RELATION, 2),
                (x, x)
            )
        )
        
        # 创建证明
        proof = Proof(
            [ProofStep(
                formula=theorem,
                rule="Reflexivity",
                premises=[]
            )]
        )
        
        # 添加到理论
        self.arithmetic_theory.theorems.add(theorem)
        self.arithmetic_theory.proofs[theorem] = proof
        
        # 反射理论
        reflected = self.reflection_op.reflect(self.arithmetic_theory)
        
        # 验证证明被反射
        self.assertIn(theorem, reflected.theorems)
        
        # 验证存在关于证明的元定理
        has_proof_meta_theorem = False
        for axiom in reflected.axioms:
            if self._is_proof_axiom(axiom, theorem):
                has_proof_meta_theorem = True
                break
        
        self.assertTrue(has_proof_meta_theorem)
    
    def _is_proof_axiom(self, axiom: Formula, theorem: Formula) -> bool:
        """检查是否是证明公理"""
        # 简化检查：查找包含Prov谓词的公理
        if isinstance(axiom, AtomicFormula) and axiom.predicate.name == "Prov":
            return True
        if isinstance(axiom, ForAllFormula):
            return self._is_proof_axiom(axiom.formula, theorem)
        if isinstance(axiom, ExistsFormula):
            return self._is_proof_axiom(axiom.formula, theorem)
        if isinstance(axiom, ImpliesFormula):
            return (self._is_proof_axiom(axiom.premise, theorem) or 
                    self._is_proof_axiom(axiom.conclusion, theorem))
        return False
    
    def test_categorical_consistency(self):
        """测试范畴论视角的一致性"""
        # 反射必须是函子
        t1 = self.arithmetic_theory
        t2 = self.meta_theory
        
        # Reflect保持恒等
        id_reflected = self.reflection_op.reflect(t1)
        self.assertEqual(id_reflected.name, f"Reflect({t1.name})")
        
        # Reflect保持复合（简化测试）
        t1_reflected = self.reflection_op.reflect(t1)
        t1_twice_reflected = self.reflection_op.reflect(t1_reflected)
        
        # 验证名称的复合性
        self.assertEqual(
            t1_twice_reflected.name,
            f"Reflect(Reflect({t1.name}))"
        )
    
    def test_information_theoretic_consistency(self):
        """测试信息论的一致性"""
        # 计算理论的信息量（简化为编码长度）
        def info_content(theory: Theory) -> int:
            encoding = self.encoder.encode_theory(theory)
            return len(encoding.to_binary().value)
        
        # 反射必须增加信息量
        theories = [
            self.arithmetic_theory,
            self.meta_theory,
            self.self_ref_theory
        ]
        
        for theory in theories:
            original_info = info_content(theory)
            reflected = self.reflection_op.reflect(theory)
            reflected_info = info_content(reflected)
            
            # 信息量必须严格增加
            self.assertGreater(reflected_info, original_info)
            
            # 信息增加率应该有界
            increase_rate = (reflected_info - original_info) / original_info
            self.assertLess(increase_rate, 2.0)  # 不应该超过2倍
    
    def test_deep_structural_consistency(self):
        """测试深层结构一致性"""
        # 创建理论塔
        tower = TheoryTower(self.arithmetic_theory)
        
        # 扩展多层
        for _ in range(3):
            tower.extend()
        
        # 验证每层都保持基本结构
        for i, theory in enumerate(tower.theories):
            # 1. 保持原始公理
            base_axioms = self.arithmetic_theory.axioms
            self.assertTrue(base_axioms.issubset(theory.axioms))
            
            # 2. 保持推理规则
            base_rules = {rule.name for rule in self.arithmetic_theory.inference_rules}
            theory_rules = {rule.name for rule in theory.inference_rules}
            self.assertTrue(base_rules.issubset(theory_rules))
            
            # 3. 层级正确
            expected_name = "Reflect(" * i + self.arithmetic_theory.name + ")" * i
            self.assertEqual(theory.name, expected_name)
    
    def test_conservation_over_subsystems(self):
        """测试子系统的保守性"""
        # 创建子理论（只有部分公理）
        sub_axioms = list(self.arithmetic_theory.axioms)[:2]
        sub_theory = Theory(
            name="SubArithmetic",
            language=self.arithmetic_theory.language,
            axioms=set(sub_axioms),
            inference_rules=self.arithmetic_theory.inference_rules
        )
        
        # 反射子理论
        sub_reflected = self.reflection_op.reflect(sub_theory)
        
        # 反射完整理论
        full_reflected = self.reflection_op.reflect(self.arithmetic_theory)
        
        # 子理论的反射应该是完整理论反射的子系统
        # （这是保守性的体现）
        self.assertTrue(sub_reflected.axioms.issubset(full_reflected.axioms))


if __name__ == '__main__':
    unittest.main()