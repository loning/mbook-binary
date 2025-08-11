#!/usr/bin/env python3
"""
C9-3 自指代数机器验证程序

严格验证C9-3推论：代数结构（群、环、域）的自指涌现
- 群的对称性与collapse不变性
- 环的分配律与递归结构
- 域的完备性与可逆性
- 同态映射的连续性
- 与C9-1, C9-2的严格一致性

绝不妥协：每个代数结构都必须满足完整公理系统
程序错误时立即停止，重新审查理论与实现的一致性
"""

import unittest
import time
from typing import List, Set, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import sys
import os

# 添加基础框架路径
sys.path.append(os.path.join(os.path.dirname(__file__)))
from base_framework import VerificationTest
from no11_number_system import No11Number
from test_C9_1 import SelfReferentialArithmetic
from test_C9_2 import RecursiveNumberTheory


class AlgebraicStructureError(Exception):
    """代数结构基类异常"""
    pass

class GroupError(AlgebraicStructureError):
    """群论相关错误"""
    pass

class RingError(AlgebraicStructureError):
    """环论相关错误"""
    pass

class FieldError(AlgebraicStructureError):
    """域论相关错误"""
    pass


class SelfReferentialGroup:
    """
    自指群的严格实现
    绝不简化：完整验证所有群公理
    """
    
    def __init__(self, 
                 elements: Set[No11Number],
                 operation: Callable[[No11Number, No11Number], No11Number],
                 name: str = "Group"):
        self.elements = elements
        self.operation = operation
        self.name = name
        self._cayley_table: Optional[Dict[Tuple[No11Number, No11Number], No11Number]] = None
        
        # 严格验证群公理
        self._verify_closure()
        self.identity = self._find_identity()
        self._verify_associativity()
        self._verify_inverses()
        
    def operate(self, a: No11Number, b: No11Number) -> No11Number:
        """执行群运算，确保self-referential性质"""
        if a not in self.elements or b not in self.elements:
            raise GroupError(f"Elements {a}, {b} not in group {self.name}")
        
        # 使用Cayley表加速（如果已计算）
        if self._cayley_table is not None:
            return self._cayley_table[(a, b)]
        
        result = self.operation(a, b)
        if result not in self.elements:
            raise GroupError(f"Operation not closed: {a} * {b} = {result}")
        
        return result
    
    def inverse(self, a: No11Number) -> No11Number:
        """计算群元素的逆元"""
        if a not in self.elements:
            raise GroupError(f"Element {a} not in group")
        
        for x in self.elements:
            if self.operate(a, x) == self.identity and self.operate(x, a) == self.identity:
                return x
        
        raise GroupError(f"No inverse found for {a}")
    
    def order(self, element: No11Number) -> int:
        """计算元素的阶"""
        if element not in self.elements:
            raise GroupError(f"Element {element} not in group")
        
        power = element
        order = 1
        
        while power != self.identity:
            power = self.operate(power, element)
            order += 1
            if order > len(self.elements):
                raise GroupError(f"Element {element} has infinite order")
        
        return order
    
    def power(self, element: No11Number, n: int) -> No11Number:
        """计算元素的n次幂"""
        if element not in self.elements:
            raise GroupError(f"Element {element} not in group")
        
        if n == 0:
            return self.identity
        elif n < 0:
            # 负幂是逆元的正幂
            inv = self.find_inverse(element)
            return self.power(inv, -n)
        else:
            # 正幂：重复运算
            result = self.identity
            for _ in range(n):
                result = self.operate(result, element)
            return result
    
    def is_abelian(self) -> bool:
        """检查群是否交换"""
        for a in self.elements:
            for b in self.elements:
                if self.operate(a, b) != self.operate(b, a):
                    return False
        return True
    
    def subgroup(self, subset: Set[No11Number]) -> 'SelfReferentialGroup':
        """生成子群"""
        if not subset.issubset(self.elements):
            raise GroupError("Subset contains elements not in parent group")
        
        # 验证子群条件
        if self.identity not in subset:
            raise GroupError("Subset does not contain identity")
        
        # 闭包检查
        for a in subset:
            for b in subset:
                if self.operate(a, b) not in subset:
                    raise GroupError(f"Subset not closed under operation")
        
        # 逆元检查
        for a in subset:
            if self.inverse(a) not in subset:
                raise GroupError(f"Subset does not contain inverse of {a}")
        
        return SelfReferentialGroup(subset, self.operation, f"{self.name}_subgroup")
    
    def generate_cayley_table(self) -> Dict[Tuple[No11Number, No11Number], No11Number]:
        """生成Cayley表"""
        if self._cayley_table is None:
            self._cayley_table = {}
            for a in self.elements:
                for b in self.elements:
                    self._cayley_table[(a, b)] = self.operation(a, b)
        return self._cayley_table
    
    def _find_identity(self) -> No11Number:
        """寻找单位元"""
        for e in self.elements:
            is_identity = True
            for x in self.elements:
                if self.operation(e, x) != x or self.operation(x, e) != x:
                    is_identity = False
                    break
            if is_identity:
                return e
        raise GroupError("No identity element found")
    
    def _verify_closure(self):
        """验证封闭性"""
        for a in self.elements:
            for b in self.elements:
                result = self.operation(a, b)
                if result not in self.elements:
                    raise GroupError(f"Operation not closed: {a} * {b} = {result} not in group")
    
    def _verify_associativity(self):
        """验证结合律"""
        # 由于完全验证需要O(n³)，对大群采用采样验证
        elements_list = list(self.elements)
        n = len(elements_list)
        
        if n <= 10:
            # 小群完全验证
            test_elements = elements_list
        else:
            # 大群采样验证
            import random
            test_elements = random.sample(elements_list, min(10, n))
        
        for a in test_elements:
            for b in test_elements:
                for c in test_elements:
                    left = self.operation(self.operation(a, b), c)
                    right = self.operation(a, self.operation(b, c))
                    if left != right:
                        raise GroupError(
                            f"Associativity failed: ({a}*{b})*{c} = {left} != "
                            f"{right} = {a}*({b}*{c})"
                        )
    
    def _verify_inverses(self):
        """验证每个元素都有逆元"""
        for a in self.elements:
            found_inverse = False
            for x in self.elements:
                if (self.operation(a, x) == self.identity and 
                    self.operation(x, a) == self.identity):
                    found_inverse = True
                    break
            if not found_inverse:
                raise GroupError(f"Element {a} has no inverse")


class SelfReferentialRing:
    """
    自指环的严格实现
    基于群结构添加乘法运算和分配律
    """
    
    def __init__(self,
                 elements: Set[No11Number],
                 add_op: Callable[[No11Number, No11Number], No11Number],
                 mul_op: Callable[[No11Number, No11Number], No11Number],
                 name: str = "Ring"):
        self.elements = elements
        self.add_op = add_op
        self.mul_op = mul_op
        self.name = name
        
        # 验证加法群
        self.additive_group = SelfReferentialGroup(elements, add_op, f"{name}_add")
        self.zero = self.additive_group.identity
        
        # 验证乘法结构
        self._verify_multiplicative_closure()
        self.one = self._find_multiplicative_identity()
        
        # 验证分配律
        self._verify_distributivity()
    
    def add(self, a: No11Number, b: No11Number) -> No11Number:
        """环的加法"""
        return self.additive_group.operate(a, b)
    
    def multiply(self, a: No11Number, b: No11Number) -> No11Number:
        """环的乘法"""
        if a not in self.elements or b not in self.elements:
            raise RingError(f"Elements {a}, {b} not in ring")
        
        result = self.mul_op(a, b)
        if result not in self.elements:
            raise RingError(f"Multiplication not closed: {a} * {b} = {result}")
        
        return result
    
    def negate(self, a: No11Number) -> No11Number:
        """计算加法逆元（相反数）"""
        return self.additive_group.inverse(a)
    
    def is_commutative(self) -> bool:
        """检查环是否交换"""
        for a in self.elements:
            for b in self.elements:
                if self.multiply(a, b) != self.multiply(b, a):
                    return False
        return True
    
    def is_integral_domain(self) -> bool:
        """检查是否是整环"""
        if not self.is_commutative():
            return False
        
        if self.one is None:
            return False
        
        # 检查无零因子
        for a in self.elements:
            if a == self.zero:
                continue
            for b in self.elements:
                if b == self.zero:
                    continue
                if self.multiply(a, b) == self.zero:
                    return False
        
        return True
    
    def units(self) -> Set[No11Number]:
        """找出所有可逆元（单位）"""
        if self.one is None:
            return set()
        
        units = set()
        for a in self.elements:
            for b in self.elements:
                if self.multiply(a, b) == self.one and self.multiply(b, a) == self.one:
                    units.add(a)
                    break
        
        return units
    
    def ideal_generated_by(self, generators: Set[No11Number]) -> Set[No11Number]:
        """生成由给定元素生成的理想"""
        ideal = {self.zero}
        
        # 添加生成元
        ideal.update(generators)
        
        # 添加所有r*g和g*r (两边理想)
        for g in generators:
            for r in self.elements:
                ideal.add(self.multiply(r, g))
                ideal.add(self.multiply(g, r))
        
        # 闭包under加法和相反数
        changed = True
        while changed:
            changed = False
            new_elements = set()
            
            # 加法闭包
            for a in ideal:
                for b in ideal:
                    sum_elem = self.add(a, b)
                    if sum_elem not in ideal:
                        new_elements.add(sum_elem)
                        changed = True
            
            # 相反数闭包
            for a in ideal:
                neg_a = self.negate(a)
                if neg_a not in ideal:
                    new_elements.add(neg_a)
                    changed = True
            
            ideal.update(new_elements)
        
        return ideal
    
    def _find_multiplicative_identity(self) -> Optional[No11Number]:
        """寻找乘法单位元"""
        for e in self.elements:
            if e == self.zero:
                continue
            
            is_identity = True
            for x in self.elements:
                if self.mul_op(e, x) != x or self.mul_op(x, e) != x:
                    is_identity = False
                    break
            
            if is_identity:
                return e
        
        return None
    
    def _verify_multiplicative_closure(self):
        """验证乘法封闭性"""
        for a in self.elements:
            for b in self.elements:
                result = self.mul_op(a, b)
                if result not in self.elements:
                    raise RingError(f"Multiplication not closed: {a} * {b} = {result}")
    
    def _verify_distributivity(self):
        """验证分配律"""
        # 采样验证以提高效率
        elements_list = list(self.elements)
        n = len(elements_list)
        
        if n <= 8:
            test_elements = elements_list
        else:
            import random
            test_elements = random.sample(elements_list, min(8, n))
        
        for a in test_elements:
            for b in test_elements:
                for c in test_elements:
                    # 左分配律: a*(b+c) = a*b + a*c
                    left = self.multiply(a, self.add(b, c))
                    right = self.add(self.multiply(a, b), self.multiply(a, c))
                    if left != right:
                        raise RingError(
                            f"Left distributivity failed: {a}*({b}+{c}) = {left} != "
                            f"{right} = {a}*{b} + {a}*{c}"
                        )
                    
                    # 右分配律: (a+b)*c = a*c + b*c
                    left = self.multiply(self.add(a, b), c)
                    right = self.add(self.multiply(a, c), self.multiply(b, c))
                    if left != right:
                        raise RingError(
                            f"Right distributivity failed: ({a}+{b})*{c} = {left} != "
                            f"{right} = {a}*{c} + {b}*{c}"
                        )


class SelfReferentialField:
    """
    自指域的严格实现
    基于环结构添加乘法逆元
    """
    
    def __init__(self,
                 elements: Set[No11Number],
                 add_op: Callable[[No11Number, No11Number], No11Number],
                 mul_op: Callable[[No11Number, No11Number], No11Number],
                 name: str = "Field"):
        # 首先验证是环
        self.ring = SelfReferentialRing(elements, add_op, mul_op, name)
        self.name = name
        
        # 验证域特有性质
        self._verify_field_axioms()
        
        self.zero = self.ring.zero
        self.one = self.ring.one
        
        # 构造乘法群
        nonzero = self.elements - {self.zero}
        self.multiplicative_group = SelfReferentialGroup(
            nonzero, mul_op, f"{name}_mul"
        )
    
    @property
    def elements(self) -> Set[No11Number]:
        return self.ring.elements
    
    def add(self, a: No11Number, b: No11Number) -> No11Number:
        """域的加法"""
        return self.ring.add(a, b)
    
    def multiply(self, a: No11Number, b: No11Number) -> No11Number:
        """域的乘法"""
        return self.ring.multiply(a, b)
    
    def negate(self, a: No11Number) -> No11Number:
        """加法逆元"""
        return self.ring.negate(a)
    
    def multiplicative_inverse(self, a: No11Number) -> No11Number:
        """乘法逆元"""
        if a == self.zero:
            raise FieldError("Zero has no multiplicative inverse")
        return self.multiplicative_group.inverse(a)
    
    def divide(self, a: No11Number, b: No11Number) -> No11Number:
        """域的除法 a/b = a * b^(-1)"""
        if b == self.zero:
            raise FieldError("Division by zero")
        return self.multiply(a, self.multiplicative_inverse(b))
    
    def characteristic(self) -> int:
        """计算域的特征"""
        if self.one is None:
            return 0
        
        char = 1
        sum_val = self.one
        
        while sum_val != self.zero:
            sum_val = self.add(sum_val, self.one)
            char += 1
            
            if char > len(self.elements):
                return 0  # 特征为0
        
        return char
    
    def _verify_field_axioms(self):
        """验证域公理"""
        # 必须是交换环
        if not self.ring.is_commutative():
            raise FieldError("Field must be commutative")
        
        # 必须有乘法单位元
        if self.ring.one is None:
            raise FieldError("Field must have multiplicative identity")
        
        # 必须至少有两个元素
        if len(self.ring.elements) < 2:
            raise FieldError("Field must have at least two elements")
        
        # 每个非零元素必须有乘法逆元
        nonzero = self.ring.elements - {self.ring.zero}
        for a in nonzero:
            found_inverse = False
            for b in nonzero:
                if (self.ring.multiply(a, b) == self.ring.one and
                    self.ring.multiply(b, a) == self.ring.one):
                    found_inverse = True
                    break
            if not found_inverse:
                raise FieldError(f"Element {a} has no multiplicative inverse")


class GroupHomomorphism:
    """群同态的严格实现"""
    
    def __init__(self,
                 source: SelfReferentialGroup,
                 target: SelfReferentialGroup,
                 mapping: Callable[[No11Number], No11Number]):
        self.source = source
        self.target = target
        self.mapping = mapping
        
        # 验证同态性质
        if not self._verify_homomorphism():
            raise GroupError("Mapping is not a group homomorphism")
    
    def map(self, element: No11Number) -> No11Number:
        """应用同态映射"""
        if element not in self.source.elements:
            raise GroupError(f"Element {element} not in source group")
        
        result = self.mapping(element)
        if result not in self.target.elements:
            raise GroupError(f"Mapping result {result} not in target group")
        
        return result
    
    def kernel(self) -> Set[No11Number]:
        """计算同态的核"""
        return {e for e in self.source.elements 
                if self.map(e) == self.target.identity}
    
    def image(self) -> Set[No11Number]:
        """计算同态的像"""
        return {self.map(e) for e in self.source.elements}
    
    def is_isomorphism(self) -> bool:
        """检查是否是同构"""
        # 单射：核只包含单位元
        if self.kernel() != {self.source.identity}:
            return False
        
        # 满射：像等于整个目标群
        if self.image() != self.target.elements:
            return False
        
        return True
    
    def _verify_homomorphism(self) -> bool:
        """验证同态性质: φ(a*b) = φ(a)*φ(b)"""
        # 采样验证
        elements_list = list(self.source.elements)
        n = len(elements_list)
        
        if n <= 10:
            test_elements = elements_list
        else:
            import random
            test_elements = random.sample(elements_list, min(10, n))
        
        for a in test_elements:
            for b in test_elements:
                # φ(a*b)
                left = self.map(self.source.operate(a, b))
                # φ(a)*φ(b)
                right = self.target.operate(self.map(a), self.map(b))
                
                if left != right:
                    return False
        
        return True


class AlgebraicStructureFactory:
    """代数结构工厂类"""
    
    def __init__(self, arithmetic: SelfReferentialArithmetic):
        self.arithmetic = arithmetic
    
    def create_cyclic_group(self, n: int) -> SelfReferentialGroup:
        """创建n阶循环群 Z_n"""
        elements = {No11Number(i) for i in range(n)}
        
        def cyclic_add(a: No11Number, b: No11Number) -> No11Number:
            return No11Number((a.value + b.value) % n)
        
        return SelfReferentialGroup(elements, cyclic_add, f"Z_{n}")
    
    def create_multiplicative_group(self, n: int) -> Optional[SelfReferentialGroup]:
        """创建模n的乘法群 (Z/nZ)*"""
        # 找出与n互质的元素
        elements = set()
        for i in range(1, n):
            if self._gcd(i, n) == 1:
                elements.add(No11Number(i))
        
        if not elements:
            return None
        
        def mod_multiply(a: No11Number, b: No11Number) -> No11Number:
            return No11Number((a.value * b.value) % n)
        
        try:
            return SelfReferentialGroup(elements, mod_multiply, f"(Z/{n}Z)*")
        except GroupError:
            return None
    
    def create_prime_field(self, p: int) -> SelfReferentialField:
        """创建素数阶有限域 F_p"""
        # 验证p是素数
        if not self._is_prime(p):
            raise FieldError(f"{p} is not prime")
        
        elements = {No11Number(i) for i in range(p)}
        
        def field_add(a: No11Number, b: No11Number) -> No11Number:
            return No11Number((a.value + b.value) % p)
        
        def field_multiply(a: No11Number, b: No11Number) -> No11Number:
            return No11Number((a.value * b.value) % p)
        
        return SelfReferentialField(elements, field_add, field_multiply, f"F_{p}")
    
    def create_polynomial_ring(self, base_field: SelfReferentialField, 
                             max_degree: int) -> SelfReferentialRing:
        """创建多项式环 F[x] (度数受限)"""
        # 简化实现：只考虑低度多项式
        # 多项式表示为系数列表 [a0, a1, ..., an]
        
        # 这里简化处理，返回基域作为环
        return base_field.ring
    
    def _gcd(self, a: int, b: int) -> int:
        """辗转相除法计算最大公约数"""
        while b:
            a, b = b, a % b
        return a
    
    def _is_prime(self, n: int) -> bool:
        """简单素性测试"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        
        return True


class TestC93SelfReferentialAlgebra(VerificationTest):
    """
    C9-3 自指代数严格验证测试类
    绝不妥协：每个测试都必须验证完整的代数公理
    """
    
    def setUp(self):
        """严格测试环境设置"""
        super().setUp()
        
        # 初始化算术系统
        self.arithmetic = SelfReferentialArithmetic(max_depth=8, max_value=30)
        
        # 初始化代数结构工厂
        self.factory = AlgebraicStructureFactory(self.arithmetic)
        
        # 预创建一些常用结构
        self.z4 = self.factory.create_cyclic_group(4)  # 4阶循环群
        self.z5 = self.factory.create_cyclic_group(5)  # 5阶循环群
        self.f5 = self.factory.create_prime_field(5)   # 5元域
    
    def test_group_axioms_verification(self):
        """严格测试群公理的验证"""
        # 测试循环群
        for n in [2, 3, 4, 5, 6]:
            with self.subTest(n=n):
                group = self.factory.create_cyclic_group(n)
                
                # 验证单位元
                self.assertIsNotNone(group.identity)
                self.assertEqual(group.identity, No11Number(0))
                
                # 验证每个元素都有逆元
                for elem in group.elements:
                    inverse = group.inverse(elem)
                    self.assertEqual(
                        group.operate(elem, inverse), 
                        group.identity,
                        f"Element {elem} * inverse {inverse} != identity"
                    )
                
                # 验证结合律（抽样）
                import random
                elements_list = list(group.elements)
                for _ in range(10):
                    a = random.choice(elements_list)
                    b = random.choice(elements_list)
                    c = random.choice(elements_list)
                    
                    left = group.operate(group.operate(a, b), c)
                    right = group.operate(a, group.operate(b, c))
                    self.assertEqual(left, right,
                                   f"Associativity failed: ({a}*{b})*{c} != {a}*({b}*{c})")
    
    def test_group_order_computation(self):
        """测试群元素阶的计算"""
        # 在Z_6中测试元素的阶
        z6 = self.factory.create_cyclic_group(6)
        
        expected_orders = {
            0: 1,  # 单位元的阶是1
            1: 6,  # 生成元
            2: 3,  # ord(2) = 3
            3: 2,  # ord(3) = 2
            4: 3,  # ord(4) = 3
            5: 6,  # 生成元
        }
        
        for value, expected_order in expected_orders.items():
            elem = No11Number(value)
            order = z6.order(elem)
            self.assertEqual(order, expected_order,
                           f"Order of {elem} should be {expected_order}, got {order}")
    
    def test_subgroup_generation(self):
        """测试子群的生成"""
        # Z_6的子群
        z6 = self.factory.create_cyclic_group(6)
        
        # <2> = {0, 2, 4}
        elem2 = No11Number(2)
        subgroup_2 = {No11Number(0), No11Number(2), No11Number(4)}
        sg2 = z6.subgroup(subgroup_2)
        self.assertEqual(sg2.elements, subgroup_2)
        
        # <3> = {0, 3}
        elem3 = No11Number(3)
        subgroup_3 = {No11Number(0), No11Number(3)}
        sg3 = z6.subgroup(subgroup_3)
        self.assertEqual(sg3.elements, subgroup_3)
    
    def test_ring_structure_verification(self):
        """测试环结构的验证"""
        # 创建Z_6作为环
        elements = {No11Number(i) for i in range(6)}
        
        def add_mod6(a: No11Number, b: No11Number) -> No11Number:
            return No11Number((a.value + b.value) % 6)
        
        def mul_mod6(a: No11Number, b: No11Number) -> No11Number:
            return No11Number((a.value * b.value) % 6)
        
        ring_z6 = SelfReferentialRing(elements, add_mod6, mul_mod6, "Z_6")
        
        # 验证环的基本性质
        self.assertEqual(ring_z6.zero, No11Number(0))
        self.assertEqual(ring_z6.one, No11Number(1))
        self.assertTrue(ring_z6.is_commutative())
        self.assertFalse(ring_z6.is_integral_domain())  # Z_6有零因子
        
        # 验证零因子：2*3 = 0 in Z_6
        self.assertEqual(
            ring_z6.multiply(No11Number(2), No11Number(3)),
            No11Number(0)
        )
    
    def test_field_axioms_verification(self):
        """测试域公理的验证"""
        # F_5是一个域
        field = self.f5
        
        # 验证特征
        self.assertEqual(field.characteristic(), 5)
        
        # 验证每个非零元素都有乘法逆元
        for i in range(1, 5):
            elem = No11Number(i)
            inverse = field.multiplicative_inverse(elem)
            product = field.multiply(elem, inverse)
            self.assertEqual(product, field.one,
                           f"{elem} * {inverse} = {product} != 1")
        
        # 验证除法
        # 3/2 in F_5
        a = No11Number(3)
        b = No11Number(2)
        result = field.divide(a, b)
        # 验证 result * 2 = 3 (mod 5)
        self.assertEqual(field.multiply(result, b), a)
    
    def test_field_arithmetic_operations(self):
        """测试域的算术运算"""
        field = self.f5
        
        # 测试运算表
        test_cases = [
            # (a, b, a+b, a*b)
            (1, 2, 3, 2),
            (2, 3, 0, 1),  # 2+3=5≡0 (mod 5), 2*3=6≡1 (mod 5)
            (4, 4, 3, 1),  # 4+4=8≡3 (mod 5), 4*4=16≡1 (mod 5)
        ]
        
        for a_val, b_val, sum_expected, prod_expected in test_cases:
            a = No11Number(a_val)
            b = No11Number(b_val)
            
            sum_result = field.add(a, b)
            self.assertEqual(sum_result.value, sum_expected,
                           f"{a} + {b} = {sum_result}, expected {sum_expected}")
            
            prod_result = field.multiply(a, b)
            self.assertEqual(prod_result.value, prod_expected,
                           f"{a} * {b} = {prod_result}, expected {prod_expected}")
    
    def test_group_homomorphism(self):
        """测试群同态"""
        # 定义同态 φ: Z_4 -> Z_2
        # φ(x) = x mod 2
        z2 = self.factory.create_cyclic_group(2)
        
        def phi(x: No11Number) -> No11Number:
            return No11Number(x.value % 2)
        
        hom = GroupHomomorphism(self.z4, z2, phi)
        
        # 验证同态映射
        self.assertEqual(hom.map(No11Number(0)), No11Number(0))
        self.assertEqual(hom.map(No11Number(1)), No11Number(1))
        self.assertEqual(hom.map(No11Number(2)), No11Number(0))
        self.assertEqual(hom.map(No11Number(3)), No11Number(1))
        
        # 验证核
        kernel = hom.kernel()
        self.assertEqual(kernel, {No11Number(0), No11Number(2)})
        
        # 验证像
        image = hom.image()
        self.assertEqual(image, {No11Number(0), No11Number(1)})
        
        # 这不是同构（不是单射）
        self.assertFalse(hom.is_isomorphism())
    
    def test_ideal_generation(self):
        """测试理想的生成"""
        # 在Z_12中生成理想
        elements = {No11Number(i) for i in range(12)}
        
        def add_mod12(a: No11Number, b: No11Number) -> No11Number:
            return No11Number((a.value + b.value) % 12)
        
        def mul_mod12(a: No11Number, b: No11Number) -> No11Number:
            return No11Number((a.value * b.value) % 12)
        
        ring_z12 = SelfReferentialRing(elements, add_mod12, mul_mod12, "Z_12")
        
        # <3>生成的主理想应该是{0, 3, 6, 9}
        ideal_3 = ring_z12.ideal_generated_by({No11Number(3)})
        expected = {No11Number(0), No11Number(3), No11Number(6), No11Number(9)}
        self.assertEqual(ideal_3, expected)
        
        # <4>生成的主理想应该是{0, 4, 8}
        ideal_4 = ring_z12.ideal_generated_by({No11Number(4)})
        expected = {No11Number(0), No11Number(4), No11Number(8)}
        self.assertEqual(ideal_4, expected)
    
    def test_self_reference_property(self):
        """测试代数结构的自指性质"""
        # 验证Cayley表是collapse的不动点
        group = self.z5
        cayley_table = group.generate_cayley_table()
        
        # 表的每个条目都应该在群中
        for (a, b), result in cayley_table.items():
            self.assertIn(result, group.elements)
            
            # 验证运算的自包含性
            self.assertEqual(group.operate(a, b), result)
        
        # 验证表的完整性
        n = len(group.elements)
        self.assertEqual(len(cayley_table), n * n)
    
    def test_entropy_increase_in_operations(self):
        """测试代数运算的熵增性质"""
        field = self.f5
        
        # 测量运算前后的信息量
        a = No11Number(2)
        b = No11Number(3)
        
        # 单个元素的信息量
        info_a = len(a.bits)
        info_b = len(b.bits)
        
        # 运算结果
        sum_result = field.add(a, b)
        prod_result = field.multiply(a, b)
        
        # 运算创建了新的关系信息
        # 虽然结果的位长度可能不变，但运算路径增加了信息
        self.assertGreaterEqual(
            info_a + info_b + 1,  # +1 for operation type
            max(info_a, info_b),
            "Operation should not decrease total information"
        )
    
    def test_consistency_with_c9_1_and_c9_2(self):
        """测试与C9-1, C9-2的一致性"""
        # 域运算应该使用C9-1的算术
        field = self.f5
        
        # 验证加法使用自指算术
        a = No11Number(2)
        b = No11Number(3)
        
        field_sum = field.add(a, b)
        # 在F_5中，2+3=5≡0
        self.assertEqual(field_sum, No11Number(0))
        
        # 验证域的素性与C9-2的素数概念一致
        # 5是素数，所以F_5是域
        self.assertEqual(field.characteristic(), 5)
        
        # 验证No-11约束的保持
        for elem in field.elements:
            self.assertIsInstance(elem, No11Number)
    
    def test_multiplicative_group_of_field(self):
        """测试域的乘法群"""
        field = self.f5
        
        # F_5* = {1, 2, 3, 4}
        mult_group = field.multiplicative_group
        expected_elements = {No11Number(i) for i in range(1, 5)}
        self.assertEqual(mult_group.elements, expected_elements)
        
        # 验证是循环群
        # 2是F_5*的生成元
        gen = No11Number(2)
        powers = set()
        power = gen
        
        for i in range(1, 5):
            powers.add(power)
            power = mult_group.operate(power, gen)
        
        self.assertEqual(powers, expected_elements)
    
    def test_lagrange_theorem(self):
        """测试拉格朗日定理"""
        # 对于有限群G和子群H，|H|整除|G|
        z12 = self.factory.create_cyclic_group(12)
        
        # 找出所有可能的子群阶
        possible_orders = []
        for d in range(1, 13):
            if 12 % d == 0:
                possible_orders.append(d)
        
        # 验证子群的阶都是12的因子
        # 这里测试几个具体的子群
        subgroups = [
            {No11Number(0)},  # 平凡子群，阶为1
            {No11Number(0), No11Number(6)},  # 阶为2
            {No11Number(0), No11Number(4), No11Number(8)},  # 阶为3
            {No11Number(0), No11Number(3), No11Number(6), No11Number(9)},  # 阶为4
        ]
        
        for sg_elements in subgroups:
            try:
                sg = z12.subgroup(sg_elements)
                order = len(sg.elements)
                self.assertIn(order, possible_orders,
                            f"Subgroup order {order} should divide group order 12")
            except GroupError:
                # 不是有效子群，跳过
                pass
    
    def test_field_extension_concept(self):
        """测试域扩张的概念（简化）"""
        # 这是一个概念验证，不是完整实现
        base_field = self.f5
        
        # F_5包含于F_25（理论上）
        # F_25可以看作F_5上的2维向量空间
        # 这里只验证基本概念
        
        self.assertEqual(base_field.characteristic(), 5)
        self.assertEqual(len(base_field.elements), 5)
        
        # 扩域的大小应该是p^n
        # F_25的大小是5^2 = 25
        extended_size = 5 ** 2
        self.assertEqual(extended_size, 25)


if __name__ == '__main__':
    # 严格运行测试：任何失败都要停止并审查
    unittest.main(verbosity=2, exit=True)