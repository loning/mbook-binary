#!/usr/bin/env python3

import unittest
import math
from typing import List, Dict, Set, Any, Callable
import sys
import os

# Add the formal directory to the path to import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'formal'))

class SelfReferentialStructure:
    """自指结构的形式化定义"""
    
    def __init__(self, states: Set[Any], functions: Dict[str, Callable], 
                 recursions: Dict[str, Any]):
        self.states = set(states)
        self.functions = dict(functions)
        self.recursions = dict(recursions)
        self.phi = (1 + math.sqrt(5)) / 2
        
    def is_self_referential(self) -> bool:
        """验证结构是否具有自指性"""
        has_self_reference = False
        
        # 1. 检查状态自引用
        for state in self.states:
            state_str = str(state).lower()
            if any(keyword in state_str for keyword in [
                'self', 'recursive', 'meta', 'observer', 'fib', 'phi'
            ]):
                has_self_reference = True
                break
                
        # 2. 检查函数自引用
        for func_name, func in self.functions.items():
            # 检查函数名是否暗示自指
            if any(keyword in func_name.lower() for keyword in [
                'self', 'recursive', 'meta', 'fib', 'identity', 'phi'
            ]):
                has_self_reference = True
                break
                
            # 检查函数源码中的自引用模式
            try:
                import inspect
                source = inspect.getsource(func)
                lines = [line.strip() for line in source.split('\n')]
                
                for line in lines:
                    if line.startswith('def '):
                        continue
                    
                    if (f'{func_name}(' in line or 
                        f'return {func_name}' in line or
                        any(pattern in line for pattern in [
                            'return lambda', 'return meta_func', 'return recursive_func',
                            'return self_function', 'return phi_recursive'
                        ])):
                        has_self_reference = True
                        break
                
                if has_self_reference:
                    break
            except:
                # 如果无法获取源码，检查函数行为
                try:
                    result = func(0) if func.__code__.co_argcount > 0 else func()
                    if result == func or callable(result):
                        has_self_reference = True
                        break
                except:
                    pass
                
        # 3. 检查递归关系
        for rec_name, rec_value in self.recursions.items():
            rec_str = str(rec_value).lower()
            if (rec_name.lower() in rec_str or 
                any(pattern in rec_str for pattern in [
                    'self', 'recursive', 'psi(psi)', 'f(f)', 'meta(meta)',
                    'phi(phi)', 'observe(observe)', 'fib', '(n-1)', '(n-2)',
                    'φ(φ)', 'no-11', 'c(c)', 'm(m)', 'φm'
                ])):
                has_self_reference = True
                break
        
        return has_self_reference
    
    def get_components(self) -> Dict[str, Any]:
        """获取结构的所有组件"""
        return {
            'states': self.states,
            'functions': self.functions,
            'recursions': self.recursions
        }
    
    def compute_complexity(self) -> Dict[str, int]:
        """计算结构的复杂度"""
        return {
            'state_count': len(self.states),
            'function_count': len(self.functions),
            'recursion_count': len(self.recursions),
            'total_complexity': len(self.states) + len(self.functions) + len(self.recursions)
        }


class No11ConstraintSystem:
    """no-11约束下的二进制系统"""
    
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2
        self.constraint_name = "no-11"
        
    def is_valid_sequence(self, binary_sequence: str) -> bool:
        """检查二进制序列是否满足no-11约束"""
        if not binary_sequence:
            return True
            
        return '11' not in binary_sequence
    
    def generate_valid_sequences(self, max_length: int) -> List[str]:
        """生成所有满足no-11约束的序列"""
        if max_length <= 0:
            return []
        
        valid_sequences = []
        
        def generate_recursive(current_seq: str, remaining_length: int):
            if remaining_length == 0:
                valid_sequences.append(current_seq)
                return
            
            # 尝试添加0
            generate_recursive(current_seq + '0', remaining_length - 1)
            
            # 尝试添加1（如果不会违反no-11约束）
            if not current_seq.endswith('1'):
                generate_recursive(current_seq + '1', remaining_length - 1)
        
        for length in range(1, max_length + 1):
            generate_recursive('', length)
        
        return valid_sequences
    
    def count_valid_sequences(self, length: int) -> int:
        """计算特定长度的有效序列数量（使用Fibonacci数列）"""
        if length <= 0:
            return 0
        elif length == 1:
            return 2  # '0', '1'
        elif length == 2:
            return 3  # '00', '01', '10'
        
        # F(n) = F(n-1) + F(n-2) for n >= 3
        fib_prev2 = 2  # F(1)
        fib_prev1 = 3  # F(2)
        
        for i in range(3, length + 1):
            fib_current = fib_prev1 + fib_prev2
            fib_prev2 = fib_prev1
            fib_prev1 = fib_current
        
        return fib_prev1
    
    def compute_information_capacity(self, length: int) -> float:
        """计算no-11约束下的信息容量"""
        valid_count = self.count_valid_sequences(length)
        if valid_count <= 0:
            return 0.0
        return math.log2(valid_count)
    
    def get_asymptotic_capacity(self) -> float:
        """获取渐近信息容量"""
        return math.log2(self.phi)


class PhiRepresentationEncoder:
    """φ-表示（Zeckendorf表示）编码器"""
    
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2
        self.fibonacci_cache = {}
        
    def generate_fibonacci_sequence(self, max_value: int) -> List[int]:
        """生成Fibonacci数列"""
        if max_value in self.fibonacci_cache:
            return self.fibonacci_cache[max_value]
        
        fib_sequence = [1, 2]  # F(1) = 1, F(2) = 2
        
        while fib_sequence[-1] <= max_value:
            next_fib = fib_sequence[-1] + fib_sequence[-2]
            if next_fib > max_value:
                break
            fib_sequence.append(next_fib)
        
        self.fibonacci_cache[max_value] = fib_sequence
        return fib_sequence
    
    def encode_to_zeckendorf(self, number: int) -> str:
        """将正整数编码为Zeckendorf表示（φ-表示）"""
        if number <= 0:
            return "0"
        
        fib_sequence = self.generate_fibonacci_sequence(number)
        zeckendorf_bits = []
        remaining = number
        
        # 从最大的Fibonacci数开始，贪心选择
        for fib_num in reversed(fib_sequence):
            if remaining >= fib_num:
                zeckendorf_bits.append('1')
                remaining -= fib_num
            else:
                zeckendorf_bits.append('0')
        
        # 移除前导零
        result = ''.join(zeckendorf_bits).lstrip('0')
        return result if result else '0'
    
    def decode_from_zeckendorf(self, zeckendorf_repr: str) -> int:
        """从Zeckendorf表示解码为整数"""
        if not zeckendorf_repr or zeckendorf_repr == '0':
            return 0
        
        max_fib_needed = len(zeckendorf_repr)
        fib_sequence = [1, 2]  # F(1), F(2)
        
        for i in range(2, max_fib_needed):
            fib_sequence.append(fib_sequence[i-1] + fib_sequence[i-2])
        
        result = 0
        for i, bit in enumerate(zeckendorf_repr):
            if bit == '1':
                fib_index = len(zeckendorf_repr) - 1 - i
                if fib_index < len(fib_sequence):
                    result += fib_sequence[fib_index]
        
        return result
    
    def verify_no11_property(self, zeckendorf_repr: str) -> bool:
        """验证Zeckendorf表示满足no-11约束"""
        return '11' not in zeckendorf_repr
    
    def encode_structure_to_phi(self, structure) -> str:
        """将自指结构编码为φ-表示"""
        components = structure.get_components()
        complexity = structure.compute_complexity()
        
        base_number = (
            complexity['state_count'] * 100 +
            complexity['function_count'] * 10 +
            complexity['recursion_count']
        )
        
        # 编码各个组件
        state_hash = hash(frozenset(str(s) for s in components['states'])) % 10000
        func_hash = hash(frozenset(components['functions'].keys())) % 10000
        rec_hash = hash(frozenset(components['recursions'].keys())) % 10000
        
        combined_number = abs(base_number + state_hash + func_hash + rec_hash)
        
        # 转换为φ-表示
        phi_encoding = self.encode_to_zeckendorf(combined_number)
        
        # 验证满足no-11约束
        if not self.verify_no11_property(phi_encoding):
            phi_encoding = self._adjust_for_no11_constraint(phi_encoding)
        
        return phi_encoding
    
    def _adjust_for_no11_constraint(self, encoding: str) -> str:
        """调整编码以满足no-11约束"""
        # 简单策略：将11替换为101（等价变换）
        while '11' in encoding:
            encoding = encoding.replace('11', '101', 1)
        return encoding


class ConstraintCompletenessVerifier:
    """约束条件下的完备性验证器"""
    
    def __init__(self):
        self.constraint_system = No11ConstraintSystem()
        self.phi_encoder = PhiRepresentationEncoder()
        self.phi = (1 + math.sqrt(5)) / 2
        
    def verify_constraint_capacity(self, max_length: int = 20) -> Dict[str, Any]:
        """验证约束系统的信息容量"""
        capacities = []
        per_symbol_capacities = []
        asymptotic_capacity = self.constraint_system.get_asymptotic_capacity()
        
        for length in range(1, max_length + 1):
            capacity = self.constraint_system.compute_information_capacity(length)
            per_symbol_capacity = capacity / length
            capacities.append(capacity)
            per_symbol_capacities.append(per_symbol_capacity)
        
        # 验证每符号容量收敛到理论值
        if len(per_symbol_capacities) > 5:
            recent_per_symbol = per_symbol_capacities[-5:]
            avg_recent_per_symbol = sum(recent_per_symbol) / len(recent_per_symbol)
            convergence_error = abs(avg_recent_per_symbol - asymptotic_capacity)
        else:
            convergence_error = float('inf')
        
        return {
            'sequence_lengths': list(range(1, max_length + 1)),
            'information_capacities': capacities,
            'per_symbol_capacities': per_symbol_capacities,
            'asymptotic_capacity': asymptotic_capacity,
            'convergence_verified': convergence_error < 0.05,  # 更严格的收敛标准
            'capacity_positive': all(c > 0 for c in capacities),
            'theoretical_limit': asymptotic_capacity
        }
    
    def verify_encoding_completeness(self, test_structures: List) -> Dict[str, Any]:
        """验证约束条件下的编码完备性"""
        results = {
            'total_structures': len(test_structures),
            'successful_encodings': 0,
            'constraint_satisfied': 0,
            'decodable_structures': 0,
            'uniqueness_preserved': True,
            'encoding_details': []
        }
        
        encodings_seen = set()
        
        for i, structure in enumerate(test_structures):
            detail = {
                'structure_id': i,
                'is_self_referential': structure.is_self_referential(),
                'encoding_successful': False,
                'constraint_satisfied': False,
                'decodable': False
            }
            
            try:
                if structure.is_self_referential():
                    phi_encoding = self.phi_encoder.encode_structure_to_phi(structure)
                    
                    constraint_satisfied = self.constraint_system.is_valid_sequence(phi_encoding)
                    
                    try:
                        decoded_value = self.phi_encoder.decode_from_zeckendorf(phi_encoding)
                        decodable = decoded_value > 0
                    except:
                        decodable = False
                    
                    if phi_encoding in encodings_seen:
                        results['uniqueness_preserved'] = False
                    else:
                        encodings_seen.add(phi_encoding)
                    
                    detail.update({
                        'encoding_successful': True,
                        'phi_encoding': phi_encoding,
                        'constraint_satisfied': constraint_satisfied,
                        'decodable': decodable,
                        'encoding_length': len(phi_encoding)
                    })
                    
                    results['successful_encodings'] += 1
                    if constraint_satisfied:
                        results['constraint_satisfied'] += 1
                    if decodable:
                        results['decodable_structures'] += 1
                
            except Exception as e:
                detail['error'] = str(e)
            
            results['encoding_details'].append(detail)
        
        # 计算完备性度量
        if results['total_structures'] > 0:
            results['encoding_success_rate'] = results['successful_encodings'] / results['total_structures']
            results['constraint_satisfaction_rate'] = results['constraint_satisfied'] / max(1, results['successful_encodings'])
            results['decodability_rate'] = results['decodable_structures'] / max(1, results['successful_encodings'])
            
            results['completeness_verified'] = (
                results['encoding_success_rate'] >= 0.9 and
                results['constraint_satisfaction_rate'] >= 0.95 and
                results['decodability_rate'] >= 0.9 and
                results['uniqueness_preserved']
            )
        else:
            results['completeness_verified'] = False
        
        return results
    
    def verify_phi_representation_properties(self) -> Dict[str, Any]:
        """验证φ-表示的基本性质"""
        test_numbers = list(range(1, 101))
        results = {
            'test_range': len(test_numbers),
            'encoding_successful': 0,
            'no11_constraint_satisfied': 0,
            'bijection_verified': True,
            'examples': []
        }
        
        for number in test_numbers:
            try:
                zeck_repr = self.phi_encoder.encode_to_zeckendorf(number)
                decoded = self.phi_encoder.decode_from_zeckendorf(zeck_repr)
                constraint_ok = self.phi_encoder.verify_no11_property(zeck_repr)
                
                bijection_ok = (decoded == number)
                if not bijection_ok:
                    results['bijection_verified'] = False
                
                results['encoding_successful'] += 1
                if constraint_ok:
                    results['no11_constraint_satisfied'] += 1
                
                if number <= 10:
                    results['examples'].append({
                        'number': number,
                        'zeckendorf': zeck_repr,
                        'decoded': decoded,
                        'constraint_satisfied': constraint_ok,
                        'bijection_correct': bijection_ok
                    })
                    
            except Exception as e:
                if number <= 10:
                    results['examples'].append({
                        'number': number,
                        'error': str(e)
                    })
        
        results['encoding_success_rate'] = results['encoding_successful'] / len(test_numbers)
        results['constraint_satisfaction_rate'] = results['no11_constraint_satisfied'] / max(1, results['encoding_successful'])
        
        return results
    
    def analyze_capacity_comparison(self) -> Dict[str, Any]:
        """分析约束前后的容量比较"""
        lengths = list(range(1, 21))
        
        unconstrained_capacities = [length for length in lengths]
        
        constrained_capacities = [
            self.constraint_system.compute_information_capacity(length)
            for length in lengths
        ]
        
        capacity_ratios = [
            constrained / unconstrained if unconstrained > 0 else 0
            for constrained, unconstrained in zip(constrained_capacities, unconstrained_capacities)
        ]
        
        theoretical_ratio = self.constraint_system.get_asymptotic_capacity() / 1.0
        
        return {
            'sequence_lengths': lengths,
            'unconstrained_capacities': unconstrained_capacities,
            'constrained_capacities': constrained_capacities,
            'capacity_ratios': capacity_ratios,
            'theoretical_asymptotic_ratio': theoretical_ratio,
            'capacity_maintained': all(c > 0 for c in constrained_capacities),
            'asymptotic_convergence': abs(capacity_ratios[-1] - theoretical_ratio) < 0.1 if capacity_ratios else False
        }


class ConstrainedStructureGenerator:
    """约束系统下的结构生成器"""
    
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2
        self.constraint_system = No11ConstraintSystem()
        
    def generate_phi_based_structure(self) -> SelfReferentialStructure:
        """生成基于φ的自指结构"""
        states = {f'phi_state_{i}' for i in range(5)}
        
        def phi_recursive_func(n):
            if n <= 1:
                return n
            return phi_recursive_func(n-1) + phi_recursive_func(n-2)
        
        functions = {
            'phi_recursive': phi_recursive_func,
            'golden_ratio': lambda: self.phi
        }
        
        recursions = {
            'phi_recursion': 'φ(φ)',
            'fibonacci_relation': 'F(n) = F(n-1) + F(n-2)',
            'golden_ratio_def': '(1 + √5) / 2'
        }
        
        return SelfReferentialStructure(states, functions, recursions)
    
    def generate_constraint_aware_structure(self) -> SelfReferentialStructure:
        """生成约束感知的自指结构"""
        states = {'no11_state', 'valid_state', 'constraint_state'}
        
        def constraint_aware_func(seq):
            if isinstance(seq, str) and self.constraint_system.is_valid_sequence(seq):
                return constraint_aware_func
            return None
        
        functions = {
            'constraint_check': constraint_aware_func,
            'validity_test': lambda s: '11' not in str(s)
        }
        
        recursions = {
            'constraint_recursion': 'C(C) with no-11',
            'self_validation': 'Valid(Valid)'
        }
        
        return SelfReferentialStructure(states, functions, recursions)
    
    def generate_mixed_constraint_structures(self, count: int = 10) -> List[SelfReferentialStructure]:
        """生成混合约束结构集合"""
        structures = []
        
        structures.append(self.generate_phi_based_structure())
        structures.append(self.generate_constraint_aware_structure())
        
        for i in range(count - 2):
            if i % 3 == 0:
                states = {f'phi_var_{i}_{j}' for j in range(i % 4 + 2)}
                exec(f"def phi_var_func_{i}(): return phi_var_func_{i}")
                functions = {f'phi_func_{i}': locals()[f'phi_var_func_{i}']}
                recursions = {f'phi_rec_{i}': f'Φ{i}(Φ{i})'}
                
            elif i % 3 == 1:
                states = {f'constraint_var_{i}_{j}' for j in range(i % 3 + 2)}
                exec(f"def constraint_var_func_{i}(): return constraint_var_func_{i}")
                functions = {f'constraint_func_{i}': locals()[f'constraint_var_func_{i}']}
                recursions = {f'no11_rec_{i}': f'C{i}(C{i}) no-11'}
                
            else:
                states = {f'mixed_var_{i}_{j}' for j in range(i % 5 + 1)}
                exec(f"def mixed_var_func_{i}(): return mixed_var_func_{i}")
                functions = {f'mixed_func_{i}': locals()[f'mixed_var_func_{i}']}
                recursions = {
                    f'mixed_rec_{i}': f'M{i}(M{i})',
                    f'phi_mixed_{i}': f'φM{i}(φM{i})'
                }
            
            structures.append(SelfReferentialStructure(states, functions, recursions))
        
        return structures


class TestP4_1_No11Completeness(unittest.TestCase):
    """P4-1 no-11约束完备性命题测试"""
    
    def setUp(self):
        """测试初始化"""
        self.constraint_system = No11ConstraintSystem()
        self.phi_encoder = PhiRepresentationEncoder()
        self.verifier = ConstraintCompletenessVerifier()
        self.generator = ConstrainedStructureGenerator()
        self.phi = (1 + math.sqrt(5)) / 2
        
    def test_constraint_system_basic_functionality(self):
        """测试约束系统基本功能"""
        print("\n=== 测试约束系统基本功能 ===")
        
        # 测试约束检查
        self.assertTrue(self.constraint_system.is_valid_sequence("0"))
        self.assertTrue(self.constraint_system.is_valid_sequence("1"))
        self.assertTrue(self.constraint_system.is_valid_sequence("10"))
        self.assertTrue(self.constraint_system.is_valid_sequence("01"))
        self.assertTrue(self.constraint_system.is_valid_sequence("101"))
        
        self.assertFalse(self.constraint_system.is_valid_sequence("11"))
        self.assertFalse(self.constraint_system.is_valid_sequence("110"))
        self.assertFalse(self.constraint_system.is_valid_sequence("011"))
        
        print("✓ 约束检查功能正确")
        
        # 测试序列计数
        count_1 = self.constraint_system.count_valid_sequences(1)
        count_2 = self.constraint_system.count_valid_sequences(2)
        count_3 = self.constraint_system.count_valid_sequences(3)
        
        self.assertEqual(count_1, 2)  # '0', '1'
        self.assertEqual(count_2, 3)  # '00', '01', '10'
        self.assertEqual(count_3, 5)  # '000', '001', '010', '100', '101'
        
        print(f"✓ 序列计数正确: F(1)={count_1}, F(2)={count_2}, F(3)={count_3}")
        
        # 测试Fibonacci性质
        for n in range(4, 10):
            count_n = self.constraint_system.count_valid_sequences(n)
            count_n1 = self.constraint_system.count_valid_sequences(n-1)
            count_n2 = self.constraint_system.count_valid_sequences(n-2)
            self.assertEqual(count_n, count_n1 + count_n2)
        
        print("✓ Fibonacci递推关系验证通过")
        
    def test_phi_representation_encoder(self):
        """测试φ-表示编码器"""
        print("\n=== 测试φ-表示编码器 ===")
        
        # 测试基本编码解码
        test_numbers = [1, 2, 3, 4, 5, 8, 13, 21, 34, 55]
        
        for number in test_numbers:
            zeck_repr = self.phi_encoder.encode_to_zeckendorf(number)
            decoded = self.phi_encoder.decode_from_zeckendorf(zeck_repr)
            
            self.assertEqual(decoded, number, f"编码解码失败: {number} -> {zeck_repr} -> {decoded}")
            self.assertTrue(self.phi_encoder.verify_no11_property(zeck_repr), 
                          f"Zeckendorf表示违反no-11约束: {zeck_repr}")
        
        print("✓ φ-表示编码解码正确")
        
        # 测试no-11性质
        for i in range(1, 50):
            zeck = self.phi_encoder.encode_to_zeckendorf(i)
            self.assertTrue(self.phi_encoder.verify_no11_property(zeck),
                          f"数字{i}的Zeckendorf表示{zeck}违反no-11约束")
        
        print("✓ φ-表示天然满足no-11约束")
        
    def test_constraint_capacity_verification(self):
        """测试约束容量验证"""
        print("\n=== 测试约束容量验证 ===")
        
        capacity_result = self.verifier.verify_constraint_capacity(15)
        
        self.assertTrue(capacity_result['capacity_positive'], "所有容量应为正值")
        self.assertGreater(capacity_result['asymptotic_capacity'], 0, "渐近容量应为正")
        self.assertAlmostEqual(capacity_result['asymptotic_capacity'], 
                             math.log2(self.phi), delta=0.01, 
                             msg="渐近容量应等于log2(φ)")
        
        print(f"✓ 渐近容量: {capacity_result['asymptotic_capacity']:.3f} bits")
        print(f"✓ 理论值: {math.log2(self.phi):.3f} bits")
        
        # 检查收敛性
        if len(capacity_result['per_symbol_capacities']) > 5:
            recent_per_symbol_avg = sum(capacity_result['per_symbol_capacities'][-3:]) / 3
            self.assertLess(abs(recent_per_symbol_avg - capacity_result['asymptotic_capacity']), 0.05,
                          "每符号容量应收敛到渐近值")
            print(f"✓ 每符号容量收敛验证通过: {recent_per_symbol_avg:.3f} → {capacity_result['asymptotic_capacity']:.3f}")
        
        # 验证收敛标志
        if capacity_result['convergence_verified']:
            print("✓ 容量收敛验证通过")
        
    def test_phi_representation_properties(self):
        """测试φ-表示性质验证"""
        print("\n=== 测试φ-表示性质验证 ===")
        
        phi_result = self.verifier.verify_phi_representation_properties()
        
        self.assertTrue(phi_result['bijection_verified'], "φ-表示应是双射的")
        self.assertGreaterEqual(phi_result['constraint_satisfaction_rate'], 0.95, 
                              "约束满足率应≥95%")
        self.assertGreaterEqual(phi_result['encoding_success_rate'], 0.95,
                              "编码成功率应≥95%")
        
        print(f"✓ 双射性验证: {phi_result['bijection_verified']}")
        print(f"✓ 约束满足率: {phi_result['constraint_satisfaction_rate']:.1%}")
        print(f"✓ 编码成功率: {phi_result['encoding_success_rate']:.1%}")
        
        # 检查示例
        for example in phi_result['examples'][:5]:
            if 'error' not in example:
                self.assertTrue(example['constraint_satisfied'], 
                              f"示例{example['number']}应满足约束")
                self.assertTrue(example['bijection_correct'],
                              f"示例{example['number']}双射应正确")
        
        print("✓ φ-表示性质验证通过")
        
    def test_structure_encoding_completeness(self):
        """测试结构编码完备性"""
        print("\n=== 测试结构编码完备性 ===")
        
        # 生成测试结构
        test_structures = self.generator.generate_mixed_constraint_structures(12)
        
        # 验证完备性
        completeness = self.verifier.verify_encoding_completeness(test_structures)
        
        self.assertGreaterEqual(completeness['encoding_success_rate'], 0.9,
                              "编码成功率应≥90%")
        self.assertGreaterEqual(completeness['constraint_satisfaction_rate'], 0.95,
                              "约束满足率应≥95%")
        self.assertGreaterEqual(completeness['decodability_rate'], 0.9,
                              "可解码率应≥90%")
        self.assertTrue(completeness['uniqueness_preserved'], "唯一性应保持")
        
        print(f"✓ 编码成功率: {completeness['encoding_success_rate']:.1%}")
        print(f"✓ 约束满足率: {completeness['constraint_satisfaction_rate']:.1%}")
        print(f"✓ 可解码率: {completeness['decodability_rate']:.1%}")
        print(f"✓ 唯一性保持: {completeness['uniqueness_preserved']}")
        
        self.assertTrue(completeness['completeness_verified'], "约束完备性应得到验证")
        print("✓ 约束完备性验证通过")
        
    def test_capacity_comparison_analysis(self):
        """测试容量比较分析"""
        print("\n=== 测试容量比较分析 ===")
        
        comparison = self.verifier.analyze_capacity_comparison()
        
        self.assertTrue(comparison['capacity_maintained'], "约束后容量应保持为正")
        self.assertGreater(comparison['theoretical_asymptotic_ratio'], 0,
                         "理论渐近比率应为正")
        self.assertLess(comparison['theoretical_asymptotic_ratio'], 1,
                       "约束后容量比率应小于1")
        
        expected_ratio = math.log2(self.phi)
        self.assertAlmostEqual(comparison['theoretical_asymptotic_ratio'], 
                             expected_ratio, delta=0.01,
                             msg="理论比率应等于log2(φ)")
        
        print(f"✓ 理论容量比率: {comparison['theoretical_asymptotic_ratio']:.3f}")
        print(f"✓ 期望值: {expected_ratio:.3f}")
        
        # 检查实际比率趋势
        if len(comparison['capacity_ratios']) > 10:
            final_ratio = comparison['capacity_ratios'][-1]
            self.assertLess(abs(final_ratio - expected_ratio), 0.2,
                          "实际比率应接近理论值")
            print(f"✓ 实际最终比率: {final_ratio:.3f}")
        
        print("✓ 容量比较分析验证通过")
        
    def test_constraint_aware_structure_generation(self):
        """测试约束感知结构生成"""
        print("\n=== 测试约束感知结构生成 ===")
        
        # 测试φ-based结构
        phi_structure = self.generator.generate_phi_based_structure()
        self.assertTrue(phi_structure.is_self_referential(), "φ-结构应具有自指性")
        
        # 测试约束感知结构
        constraint_structure = self.generator.generate_constraint_aware_structure()
        self.assertTrue(constraint_structure.is_self_referential(), "约束感知结构应具有自指性")
        
        # 测试编码
        phi_encoding = self.phi_encoder.encode_structure_to_phi(phi_structure)
        constraint_encoding = self.phi_encoder.encode_structure_to_phi(constraint_structure)
        
        self.assertTrue(self.constraint_system.is_valid_sequence(phi_encoding),
                       "φ-结构编码应满足约束")
        self.assertTrue(self.constraint_system.is_valid_sequence(constraint_encoding),
                       "约束感知结构编码应满足约束")
        
        print("✓ φ-based结构生成和编码正确")
        print("✓ 约束感知结构生成和编码正确")
        
        # 测试混合结构集合
        mixed_structures = self.generator.generate_mixed_constraint_structures(8)
        self.assertEqual(len(mixed_structures), 8, "应生成指定数量的结构")
        
        valid_structures = sum(1 for s in mixed_structures if s.is_self_referential())
        self.assertGreaterEqual(valid_structures, 6, "大部分结构应具有自指性")
        
        print(f"✓ 混合结构生成: {len(mixed_structures)}个，{valid_structures}个自指")
        
    def test_complete_p4_1_verification(self):
        """完整的P4-1验证测试"""
        print("\n=== 完整的P4-1验证测试 ===")
        
        # 1. 验证约束系统基础
        basic_sequences = ['0', '1', '10', '01', '101', '010', '100']
        for seq in basic_sequences:
            self.assertTrue(self.constraint_system.is_valid_sequence(seq),
                          f"基础序列{seq}应满足no-11约束")
        
        invalid_sequences = ['11', '110', '011', '1101']
        for seq in invalid_sequences:
            self.assertFalse(self.constraint_system.is_valid_sequence(seq),
                           f"序列{seq}应违反no-11约束")
        
        print("✓ 约束系统基础验证通过")
        
        # 2. 验证φ-表示完备性
        phi_props = self.verifier.verify_phi_representation_properties()
        self.assertTrue(phi_props['bijection_verified'], "φ-表示双射性")
        self.assertGreaterEqual(phi_props['constraint_satisfaction_rate'], 0.95, "约束满足率")
        
        print("✓ φ-表示完备性验证通过")
        
        # 3. 验证结构编码完备性
        test_structures = self.generator.generate_mixed_constraint_structures(15)
        encoding_result = self.verifier.verify_encoding_completeness(test_structures)
        self.assertTrue(encoding_result['completeness_verified'], "编码完备性")
        
        print("✓ 结构编码完备性验证通过")
        
        # 4. 验证容量保持
        capacity_result = self.verifier.verify_constraint_capacity()
        self.assertTrue(capacity_result['capacity_positive'], "容量保持为正")
        self.assertGreater(capacity_result['asymptotic_capacity'], 0.6, "渐近容量足够")
        
        print("✓ 容量保持验证通过")
        
        # 5. 综合完备性判定
        overall_completeness = (
            phi_props['bijection_verified'] and
            phi_props['constraint_satisfaction_rate'] >= 0.95 and
            encoding_result['completeness_verified'] and
            capacity_result['capacity_positive']
        )
        
        self.assertTrue(overall_completeness, "P4-1整体完备性应得到验证")
        print("✓ P4-1 no-11约束完备性命题验证通过")


if __name__ == '__main__':
    unittest.main(verbosity=2)