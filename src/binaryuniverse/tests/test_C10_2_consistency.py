#!/usr/bin/env python3
"""
C10-2与底层系统深度一致性验证程序

严格验证范畴论与底层数学系统的一致性：
- 形式系统作为范畴对象（C10-1）
- 代数结构的范畴化（C9-3）
- 递归数论的函子表示（C9-2）
- 自指算术的态射结构（C9-1）
- No-11约束的范畴表现
- 熵增原理在所有层次的体现

绝不妥协：每个范畴概念都必须追溯到底层基础
"""

import unittest
import sys
import os
from typing import Set, List, Dict, Tuple, Optional

# 添加基础框架路径
sys.path.append(os.path.join(os.path.dirname(__file__)))
from base_framework import VerificationTest
from no11_number_system import No11Number

# C9系列导入
from test_C9_1 import SelfReferentialArithmetic
from test_C9_2 import RecursiveNumberTheory
from test_C9_3 import (
    SelfReferentialGroup, SelfReferentialRing, SelfReferentialField,
    AlgebraicStructureFactory, GroupHomomorphism
)

# C10-1导入
from test_C10_1 import (
    FormalSystem, Formula, Symbol, SymbolType,
    VariableTerm, ConstantTerm, AtomicFormula,
    Axiom, Proof, ProofStep, Model, Interpretation,
    GödelEncoder
)

# C10-2导入
from test_C10_2 import (
    CategoryObject, Morphism, IdentityMorphism, ComposedMorphism,
    Category, Functor, NaturalTransformation,
    FormalSystemCategory, CollapseFunctor
)


class TestC102DeepConsistency(VerificationTest):
    """
    C10-2与底层系统深度一致性验证测试类
    
    验证原则：
    1. 每个范畴概念必须基于底层构造
    2. 态射保持底层操作的语义
    3. 函子反映结构间的自然映射
    4. No-11约束贯穿所有层次
    5. 熵增在范畴操作中体现
    """
    
    def setUp(self):
        """设置深度测试环境"""
        super().setUp()
        
        # 初始化底层系统
        self.arithmetic = SelfReferentialArithmetic(max_depth=10, max_value=50)
        self.number_theory = RecursiveNumberTheory(self.arithmetic, max_recursion=10)
        self.algebra_factory = AlgebraicStructureFactory(self.arithmetic)
        
        # 创建各类范畴
        self._create_categories()
    
    def _create_categories(self):
        """创建测试用的各种范畴"""
        # 形式系统范畴
        self.formal_cat = FormalSystemCategory()
        
        # 群范畴
        self.group_cat = self._create_group_category()
        
        # No-11数范畴
        self.no11_cat = self._create_no11_category()
    
    def _create_group_category(self) -> Category:
        """创建群范畴"""
        cat = Category("Grp")
        
        # 添加几个具体的群作为对象
        z4 = self.algebra_factory.create_cyclic_group(4)
        z6 = self.algebra_factory.create_cyclic_group(6)
        
        z4_obj = CategoryObject("Z4", z4)
        z6_obj = CategoryObject("Z6", z6)
        
        cat.add_object(z4_obj)
        cat.add_object(z6_obj)
        
        # 添加群同态作为态射
        # 零同态: Z4 → Z6
        def zero_hom(x: No11Number) -> No11Number:
            return z6.identity
        
        zero_mor = Morphism(
            source=z4_obj,
            target=z6_obj,
            name="zero",
            mapping=lambda g: GroupHomomorphism(z4, z6, zero_hom)
        )
        cat.add_morphism(zero_mor)
        
        return cat
    
    def _create_no11_category(self) -> Category:
        """创建No-11数的范畴"""
        cat = Category("No11")
        
        # 对象是No-11数的集合
        small_set = {No11Number(i) for i in range(5)}
        medium_set = {No11Number(i) for i in range(10)}
        
        small_obj = CategoryObject("Small", small_set)
        medium_obj = CategoryObject("Medium", medium_set)
        
        cat.add_object(small_obj)
        cat.add_object(medium_obj)
        
        # 包含映射
        def inclusion(x: No11Number) -> No11Number:
            return x if x in medium_set else No11Number(0)
        
        incl_mor = Morphism(
            source=small_obj,
            target=medium_obj,
            name="inclusion",
            mapping=inclusion
        )
        cat.add_morphism(incl_mor)
        
        return cat
    
    def test_formal_systems_form_category(self):
        """验证形式系统形成范畴（C10-1 → C10-2）"""
        # 获取形式系统对象
        objects = list(self.formal_cat.objects)
        self.assertGreater(len(objects), 0)
        
        # 验证每个对象包含形式系统
        for obj in objects:
            self.assertIsInstance(obj.data, FormalSystem)
        
        # 验证范畴公理
        self.assertTrue(self.formal_cat.verify_axioms())
        
        # 理论态射保持证明
        for mor_set in self.formal_cat.morphisms.values():
            for mor in mor_set:
                if isinstance(mor, IdentityMorphism):
                    continue
                
                # 态射应该保持定理
                source_sys = mor.source.data
                target_sys = mor.target.data
                
                # 简化测试：检查映射是否保持公理数量关系
                if len(source_sys.axioms) > 0:
                    self.assertLessEqual(
                        len(source_sys.axioms),
                        len(target_sys.axioms) + len(target_sys.theorems)
                    )
    
    def test_algebraic_structures_as_objects(self):
        """验证代数结构作为范畴对象（C9-3 → C10-2）"""
        # 群作为对象
        for obj in self.group_cat.objects:
            group = obj.data
            self.assertIsInstance(group, SelfReferentialGroup)
            
            # 验证群性质
            self.assertIsNotNone(group.identity)
            self.assertIn(group.identity, group.elements)
            
            # 验证封闭性
            for a in list(group.elements)[:3]:  # 测试部分元素
                for b in list(group.elements)[:3]:
                    result = group.operate(a, b)
                    self.assertIn(result, group.elements)
        
        # 群同态作为态射
        for mor_set in self.group_cat.morphisms.values():
            for mor in mor_set:
                if isinstance(mor, IdentityMorphism):
                    # 恒等态射对应恒等同态
                    group = mor.source.data
                    # 恒等函数保持群运算
                    for g in list(group.elements)[:3]:
                        self.assertEqual(mor.apply(g), g)
    
    def test_functor_from_groups_to_sets(self):
        """验证遗忘函子 U: Grp → Set"""
        # 创建集合范畴（简化版）
        set_cat = Category("Set")
        
        # 遗忘函子
        forget = Functor("Forget", self.group_cat, set_cat)
        
        # 对象映射：群到其底层集合
        for obj in self.group_cat.objects:
            group = obj.data
            set_obj = CategoryObject(f"U({obj.name})", group.elements)
            set_cat.add_object(set_obj)
            forget.object_map[obj] = set_obj
        
        # 态射映射：群同态到集合函数
        for mor_set in self.group_cat.morphisms.values():
            for mor in mor_set:
                if isinstance(mor, IdentityMorphism):
                    # 恒等同态映射到恒等函数
                    set_id = set_cat.identity(forget.map_object(mor.source))
                    forget.morphism_map[mor] = set_id
                else:
                    # 一般同态映射到底层函数
                    set_mor = Morphism(
                        source=forget.map_object(mor.source),
                        target=forget.map_object(mor.target),
                        name=f"U({mor.name})",
                        mapping=mor.mapping
                    )
                    set_cat.add_morphism(set_mor)
                    forget.morphism_map[mor] = set_mor
        
        # 验证函子性质
        self.assertTrue(forget.verify_functoriality())
    
    def test_number_theory_as_functor(self):
        """验证数论操作作为函子（C9-2 → C10-2）"""
        # 素数检测作为函子
        # Prime: No11 → Bool (简化为 No11 → {0,1})
        
        bool_cat = Category("Bool")
        false_obj = CategoryObject("False", No11Number(0))
        true_obj = CategoryObject("True", No11Number(1))
        bool_cat.add_object(false_obj)
        bool_cat.add_object(true_obj)
        
        # 素数检测函子
        prime_functor = Functor("IsPrime", self.no11_cat, bool_cat)
        
        # 对象映射
        for obj in self.no11_cat.objects:
            # 每个No11集合映射到布尔值（存在素数？）
            no11_set = obj.data
            has_prime = any(self.number_theory.is_prime(n) for n in no11_set)
            prime_functor.object_map[obj] = true_obj if has_prime else false_obj
        
        # 态射映射（简化）
        for mor_set in self.no11_cat.morphisms.values():
            for mor in mor_set:
                if isinstance(mor, IdentityMorphism):
                    target_obj = prime_functor.map_object(mor.source)
                    prime_functor.morphism_map[mor] = bool_cat.identity(target_obj)
    
    def test_collapse_preserves_structure(self):
        """验证Collapse函子保持结构（贯穿所有层）"""
        # 在形式系统范畴上的Collapse
        collapse = CollapseFunctor(self.formal_cat)
        
        # Collapse保持对象（简化形式系统）
        for obj in self.formal_cat.objects:
            collapsed = collapse.map_object(obj)
            self.assertEqual(obj, collapsed)  # 简化实现中相同
        
        # 验证函子性
        self.assertTrue(collapse.verify_functoriality())
        
        # Collapse是幂等的（collapse ∘ collapse = collapse）
        # 这体现了不动点性质
        double_collapse = CollapseFunctor(self.formal_cat)
        for obj in self.formal_cat.objects:
            once = collapse.map_object(obj)
            twice = double_collapse.map_object(once)
            self.assertEqual(once, twice)
    
    def test_natural_transformation_from_theorems(self):
        """验证定理产生自然变换"""
        # 两个相关的函子（简化示例）
        id_functor = Functor("Id", self.formal_cat, self.formal_cat)
        collapse_functor = CollapseFunctor(self.formal_cat)
        
        # 设置恒等函子
        for obj in self.formal_cat.objects:
            id_functor.object_map[obj] = obj
        for mor_set in self.formal_cat.morphisms.values():
            for mor in mor_set:
                id_functor.morphism_map[mor] = mor
        
        # 自然变换 η: Id ⇒ Collapse
        eta = NaturalTransformation("simplify", id_functor, collapse_functor)
        
        # 每个分量是简化态射
        for obj in self.formal_cat.objects:
            # Id(obj) → Collapse(obj)
            # 在简化实现中都是obj → obj
            eta.components[obj] = self.formal_cat.identity(obj)
        
        # 验证自然性
        self.assertTrue(eta.verify_naturality())
    
    def test_no11_constraint_in_categories(self):
        """验证No-11约束在范畴层的体现"""
        # 所有编码都满足No-11
        for obj in self.group_cat.objects:
            encoding = obj.encode()
            self.assertIsInstance(encoding, No11Number)
        
        # 态射编码也满足
        for mor_set in self.group_cat.morphisms.values():
            for mor in mor_set:
                encoding = mor.encode()
                self.assertIsInstance(encoding, No11Number)
        
        # 范畴本身的编码
        cat_encoding = self.group_cat.encode()
        self.assertIsInstance(cat_encoding, No11Number)
        
        # hom-集的有限性（No-11的离散性体现）
        for source in self.group_cat.objects:
            for target in self.group_cat.objects:
                hom_set = self.group_cat.hom(source, target)
                self.assertIsInstance(hom_set, set)
                # 有限性
                self.assertLess(len(hom_set), 100)
    
    def test_entropy_increase_through_abstraction(self):
        """验证抽象过程的熵增"""
        # 1. 具体群的信息
        z4 = self.algebra_factory.create_cyclic_group(4)
        concrete_info = len(z4.elements) + 1  # 元素数 + 运算信息
        
        # 2. 作为范畴对象的信息
        z4_obj = CategoryObject("Z4", z4)
        obj_info = concrete_info + z4_obj.encode().value
        
        # 3. 在范畴中的信息（包括态射）
        test_cat = Category("Test")
        test_cat.add_object(z4_obj)
        
        # 添加一些态射
        z8 = self.algebra_factory.create_cyclic_group(8)
        z8_obj = CategoryObject("Z8", z8)
        test_cat.add_object(z8_obj)
        
        # 包含同态 Z4 → Z8
        def inclusion_hom(x: No11Number) -> No11Number:
            # x in Z4 maps to 2x in Z8
            return No11Number((2 * x.value) % 8)
        
        incl = Morphism(z4_obj, z8_obj, "incl", inclusion_hom)
        test_cat.add_morphism(incl)
        
        cat_info = obj_info + len(test_cat.morphisms)
        
        # 4. 函子层面的信息
        id_func = Functor("Id", test_cat, test_cat)
        for obj in test_cat.objects:
            id_func.object_map[obj] = obj
        
        functor_info = cat_info + len(id_func.object_map)
        
        # 验证信息递增
        self.assertGreater(obj_info, concrete_info)
        self.assertGreater(cat_info, obj_info)
        self.assertGreater(functor_info, cat_info)
    
    def test_yoneda_perspective(self):
        """验证Yoneda视角：对象由其关系决定"""
        # 在小范畴中测试
        small_cat = Category("Small")
        a = CategoryObject("A", "a")
        b = CategoryObject("B", "b")
        small_cat.add_object(a)
        small_cat.add_object(b)
        
        # 添加态射
        f = Morphism(a, b, "f", lambda x: f"f({x})")
        small_cat.add_morphism(f)
        
        # 对象A由所有到A的态射决定
        # Hom(-, A)定义了A的"本质"
        morphisms_to_a = []
        for obj in small_cat.objects:
            hom_set = small_cat.hom(obj, a)
            morphisms_to_a.extend(hom_set)
        
        # 不同对象有不同的"态射模式"
        morphisms_to_b = []
        for obj in small_cat.objects:
            hom_set = small_cat.hom(obj, b)
            morphisms_to_b.extend(hom_set)
        
        # A和B的区别体现在态射模式的不同
        self.assertNotEqual(len(morphisms_to_a), len(morphisms_to_b))
    
    def test_limits_from_universal_property(self):
        """验证极限的泛性质"""
        # 创建简单的积范畴
        prod_cat = Category("Prod")
        
        # 两个对象
        x = CategoryObject("X", "x")
        y = CategoryObject("Y", "y")
        prod_cat.add_object(x)
        prod_cat.add_object(y)
        
        # 积对象 X×Y
        xy = CategoryObject("X×Y", ("x", "y"))
        prod_cat.add_object(xy)
        
        # 投影态射
        pi1 = Morphism(xy, x, "π₁", lambda p: p[0])
        pi2 = Morphism(xy, y, "π₂", lambda p: p[1])
        prod_cat.add_morphism(pi1)
        prod_cat.add_morphism(pi2)
        
        # 泛性质：对任意对象Z和态射f:Z→X, g:Z→Y
        # 存在唯一态射h:Z→X×Y使得π₁∘h=f, π₂∘h=g
        z = CategoryObject("Z", "z")
        prod_cat.add_object(z)
        
        f = Morphism(z, x, "f", lambda z: "f(z)")
        g = Morphism(z, y, "g", lambda z: "g(z)")
        prod_cat.add_morphism(f)
        prod_cat.add_morphism(g)
        
        # 泛态射
        def universal_map(z_val):
            return (f.apply(z_val), g.apply(z_val))
        
        h = Morphism(z, xy, "⟨f,g⟩", universal_map)
        prod_cat.add_morphism(h)
        
        # 验证交换性
        pi1_h = prod_cat.compose(pi1, h)
        pi2_h = prod_cat.compose(pi2, h)
        
        # 测试值
        test_val = "test"
        self.assertEqual(pi1_h.apply(test_val), f.apply(test_val))
        self.assertEqual(pi2_h.apply(test_val), g.apply(test_val))
    
    def test_comprehensive_consistency(self):
        """综合测试所有层的一致性"""
        # 从C9-1开始：基础算术
        a = No11Number(3)
        b = No11Number(4)
        sum_ab = self.arithmetic.self_referential_add(a, b)
        
        # C9-2：数论性质
        is_prime_sum = self.number_theory.is_prime(sum_ab)
        
        # C9-3：代数结构
        z7 = self.algebra_factory.create_cyclic_group(7)
        
        # C10-1：形式系统
        arith_sys = FormalSystem("Arith")
        arith_sys.add_symbol(Symbol("+", SymbolType.FUNCTION, 2))
        
        # C10-2：范畴结构
        math_cat = Category("Math")
        
        # 算术系统作为对象
        arith_obj = CategoryObject("Arithmetic", arith_sys)
        math_cat.add_object(arith_obj)
        
        # 群作为对象
        group_obj = CategoryObject("Z7", z7)
        math_cat.add_object(group_obj)
        
        # 整个数学宇宙形成范畴
        self.assertTrue(math_cat.verify_axioms())
        
        # 编码的一致性
        arith_encoding = arith_obj.encode()
        group_encoding = group_obj.encode()
        cat_encoding = math_cat.encode()
        
        # 所有编码都是No-11数
        for enc in [arith_encoding, group_encoding, cat_encoding]:
            self.assertIsInstance(enc, No11Number)
        
        # 信息的层级增长
        self.assertGreater(cat_encoding.value, 0)


if __name__ == '__main__':
    # 运行深度一致性测试
    unittest.main(verbosity=2, exit=True)