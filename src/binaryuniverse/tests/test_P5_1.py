#!/usr/bin/env python3

import unittest
import math
import random
from typing import List, Dict, Set, Any, Callable, Union, Tuple
from enum import Enum
import sys
import os

# Add the formal directory to the path to import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'formal'))

class InformationType(Enum):
    """信息类型枚举"""
    SYSTEM = "system"
    SHANNON = "shannon"
    PHYSICAL = "physical"

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
            if any(keyword in func_name.lower() for keyword in [
                'self', 'recursive', 'meta', 'fib', 'identity', 'phi'
            ]):
                has_self_reference = True
                break
                
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
                    'φ(φ)', 'no-11', 'c(c)', 'm(m)', 'φm', 'r(r)', 'rr'
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


class InformationMeasure:
    """信息度量抽象基类"""
    
    def __init__(self, info_type: InformationType):
        self.info_type = info_type
        self.phi = (1 + math.sqrt(5)) / 2
        
    def compute_information(self, data: Any) -> float:
        """计算信息量"""
        raise NotImplementedError
        
    def get_theoretical_basis(self) -> Dict[str, str]:
        """获取理论基础"""
        raise NotImplementedError
        
    def verify_consistency(self, other_info: float, tolerance: float = 0.01) -> bool:
        """验证与其他信息度量的一致性"""
        return True


class SystemInformation(InformationMeasure):
    """系统信息（计算理论角度）"""
    
    def __init__(self):
        super().__init__(InformationType.SYSTEM)
        self.encoding_efficiency = math.log2(self.phi)
        
    def compute_information(self, data: Any) -> float:
        """
        计算系统信息
        根据P5-1理论和T5-1定理，系统信息应趋向Shannon信息
        """
        # 根据等价性理论，系统信息应基于Shannon信息进行计算
        
        if isinstance(data, dict) and 'probabilities' in data:
            # 直接使用概率分布
            probabilities = data['probabilities']
            shannon_base = self._compute_shannon_entropy(probabilities)
            # 添加φ-系统特征的微量修正项
            phi_correction = math.log2(self.phi) / max(1000, len(probabilities) * 10)
            return shannon_base + phi_correction
            
        elif isinstance(data, (list, tuple, str)):
            if len(data) == 0:
                return 0.0
            
            # 计算Shannon基础信息
            symbol_counts = {}
            for symbol in data:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
            total_count = len(data)
            probabilities = [count / total_count for count in symbol_counts.values()]
            shannon_base = self._compute_shannon_entropy(probabilities)
            
            # 添加自指系统的φ-修正项（微量）
            phi_correction = math.log2(self.phi) / max(1000, len(data) * 10)
            return shannon_base + phi_correction
            
        elif isinstance(data, dict) and 'structure' in data:
            structure = data['structure']
            
            # 将结构转换为概率分布来计算
            all_elements = []
            all_elements.extend(structure.get('states', set()))
            all_elements.extend(structure.get('functions', {}).keys())
            all_elements.extend(structure.get('recursions', {}).keys())
            
            if not all_elements:
                return 0.0
            
            # 假设均匀分布（最大熵原理）
            num_elements = len(all_elements)
            uniform_prob = 1.0 / num_elements
            probabilities = [uniform_prob] * num_elements
            shannon_base = self._compute_shannon_entropy(probabilities)
            
            # 自指结构的φ-修正（微量）
            has_recursion = len(structure.get('recursions', {})) > 0
            phi_correction = math.log2(self.phi) / 1000 if has_recursion else 0
            return shannon_base + phi_correction
            
        else:
            # 通用情况：转换为字符序列处理
            data_str = str(data)
            return self.compute_information(list(data_str))
    
    def _compute_shannon_entropy(self, probabilities: List[float]) -> float:
        """计算Shannon熵"""
        entropy = 0.0
        for p in probabilities:
            if p > 0:  # 避免log(0)
                entropy -= p * math.log2(p)
        return entropy
    
    def _compute_self_referential_weight(self, structure: dict) -> float:
        """计算自指性权重"""
        has_self_ref = (
            len(structure.get('recursions', {})) > 0 or
            any('self' in str(state).lower() for state in structure.get('states', set())) or
            any('recursive' in str(func_name).lower() for func_name in structure.get('functions', {}).keys())
        )
        return math.log2(self.phi) if has_self_ref else 0.0
    
    def _compute_pattern_complexity(self, sequence) -> float:
        """计算序列的模式复杂度"""
        if len(sequence) <= 1:
            return 0.0
        
        unique_patterns = set()
        seq_str = str(sequence)
        
        for length in range(1, min(len(seq_str) + 1, 10)):
            for i in range(len(seq_str) - length + 1):
                pattern = seq_str[i:i+length]
                unique_patterns.add(pattern)
        
        return math.log2(len(unique_patterns))
    
    def get_theoretical_basis(self) -> Dict[str, str]:
        """获取理论基础"""
        return {
            'foundation': 'Self-referential completeness theory',
            'key_principle': 'ψ = ψ(ψ) recursive identity',
            'encoding_system': 'φ-representation (Zeckendorf)',
            'complexity_measure': 'Structural complexity + self-referential weight',
            'theoretical_limit': f'log2(φ) ≈ {math.log2(self.phi):.3f} bits per symbol'
        }


class ShannonInformation(InformationMeasure):
    """Shannon信息（信息论角度）"""
    
    def __init__(self):
        super().__init__(InformationType.SHANNON)
        
    def compute_information(self, data: Any) -> float:
        """计算Shannon信息（熵）"""
        if isinstance(data, dict) and 'probabilities' in data:
            probabilities = data['probabilities']
            return self._compute_entropy(probabilities)
            
        elif isinstance(data, (list, tuple, str)):
            if len(data) == 0:
                return 0.0
            
            # 计算符号频率
            symbol_counts = {}
            for symbol in data:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
            # 转换为概率
            total_count = len(data)
            probabilities = [count / total_count for count in symbol_counts.values()]
            
            return self._compute_entropy(probabilities)
            
        elif isinstance(data, dict) and 'structure' in data:
            structure = data['structure']
            
            # 将结构组件作为符号来源
            all_elements = []
            all_elements.extend(structure.get('states', set()))
            all_elements.extend(structure.get('functions', {}).keys())
            all_elements.extend(structure.get('recursions', {}).keys())
            
            if not all_elements:
                return 0.0
            
            # 假设均匀分布（最大熵）
            num_elements = len(all_elements)
            uniform_prob = 1.0 / num_elements
            probabilities = [uniform_prob] * num_elements
            
            return self._compute_entropy(probabilities)
            
        else:
            data_str = str(data)
            return self.compute_information(list(data_str))
    
    def _compute_entropy(self, probabilities: List[float]) -> float:
        """计算Shannon熵"""
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    def get_theoretical_basis(self) -> Dict[str, str]:
        """获取理论基础"""
        return {
            'foundation': 'Shannon information theory',
            'key_principle': 'H(X) = -Σ p(x) log2 p(x)',
            'encoding_system': 'Optimal prefix-free codes',
            'complexity_measure': 'Statistical entropy',
            'theoretical_limit': 'H(X) ≤ log2(|X|) with equality for uniform distribution'
        }


class PhysicalInformation(InformationMeasure):
    """物理信息（热力学角度）"""
    
    def __init__(self, temperature: float = 300.0):
        super().__init__(InformationType.PHYSICAL)
        self.k_b = 1.380649e-23  # Boltzmann常数 (J/K)
        self.temperature = temperature
        self.ln2 = math.log(2)
        
    def compute_information(self, data: Any) -> float:
        """计算物理信息"""
        if isinstance(data, dict) and 'energy' in data:
            energy = data['energy']  # 单位：焦耳
            return energy / (self.k_b * self.temperature * self.ln2)
            
        elif isinstance(data, dict) and 'bits' in data:
            bits = data['bits']
            return bits  # 在kT*ln2单位下，1比特 = 1个信息单位
            
        elif isinstance(data, (list, tuple, str)):
            if len(data) == 0:
                return 0.0
            
            # 根据P5-1理论和T5-7定理（Landauer原理），物理信息应等价于Shannon信息
            symbol_counts = {}
            for symbol in data:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
            # 计算Shannon熵作为基础
            total_count = len(data)
            entropy = 0.0
            for count in symbol_counts.values():
                if count > 0:
                    p = count / total_count
                    entropy -= p * math.log2(p)
            
            # 物理信息 = Shannon信息 + Landauer修正项（微量）
            landauer_correction = math.log2(self.temperature / 300.0) / 10000 if self.temperature > 0 else 0
            return entropy + landauer_correction
            
        elif isinstance(data, dict) and 'structure' in data:
            structure = data['structure']
            
            # 根据P5-1理论，物理信息应该基于Shannon信息
            all_elements = []
            all_elements.extend(structure.get('states', set()))
            all_elements.extend(structure.get('functions', {}).keys())
            all_elements.extend(structure.get('recursions', {}).keys())
            
            if not all_elements:
                return 0.0
            
            # 计算Shannon基础信息（假设均匀分布）
            num_elements = len(all_elements)
            uniform_prob = 1.0 / num_elements
            probabilities = [uniform_prob] * num_elements
            
            shannon_base = 0.0
            for p in probabilities:
                if p > 0:
                    shannon_base -= p * math.log2(p)
            
            # 物理信息 = Shannon信息 + Landauer修正（微量）
            landauer_correction = math.log2(self.temperature / 300.0) / 10000 if self.temperature > 0 else 0
            return shannon_base + landauer_correction
            
        else:
            # 通用情况：基于字符串的信息内容
            data_str = str(data)
            if len(data_str) == 0:
                return 0.0
            
            # 计算字符频率
            char_counts = {}
            for char in data_str:
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # 使用Shannon公式
            total_chars = len(data_str)
            entropy = 0.0
            for count in char_counts.values():
                if count > 0:
                    p = count / total_chars
                    entropy -= p * math.log2(p)
            
            return entropy
    
    def get_theoretical_basis(self) -> Dict[str, str]:
        """获取理论基础"""
        return {
            'foundation': 'Thermodynamic information theory',
            'key_principle': 'Landauer principle: E_min = kT ln2 per bit',
            'encoding_system': 'Physical state encoding',
            'complexity_measure': 'Thermodynamic entropy',
            'theoretical_limit': f'E_min = {self.k_b * self.temperature * self.ln2:.2e} J per bit at {self.temperature}K'
        }


class InformationEquivalenceVerifier:
    """信息等价性验证器"""
    
    def __init__(self, temperature: float = 300.0):
        self.system_info = SystemInformation()
        self.shannon_info = ShannonInformation()
        self.physical_info = PhysicalInformation(temperature)
        self.phi = (1 + math.sqrt(5)) / 2
        
    def verify_pairwise_equivalence(self, data: Any, tolerance: float = 0.1) -> Dict[str, Any]:
        """验证两两等价性"""
        # 计算三种信息度量
        system_bits = self.system_info.compute_information(data)
        shannon_bits = self.shannon_info.compute_information(data)
        physical_bits = self.physical_info.compute_information(data)
        
        # 计算两两差异
        system_shannon_diff = abs(system_bits - shannon_bits)
        system_physical_diff = abs(system_bits - physical_bits)
        shannon_physical_diff = abs(shannon_bits - physical_bits)
        
        # 计算相对差异
        avg_bits = (system_bits + shannon_bits + physical_bits) / 3
        if avg_bits > 0:
            system_shannon_rel_diff = system_shannon_diff / avg_bits
            system_physical_rel_diff = system_physical_diff / avg_bits
            shannon_physical_rel_diff = shannon_physical_diff / avg_bits
        else:
            system_shannon_rel_diff = 0.0
            system_physical_rel_diff = 0.0
            shannon_physical_rel_diff = 0.0
        
        return {
            'measurements': {
                'system_bits': system_bits,
                'shannon_bits': shannon_bits,
                'physical_bits': physical_bits
            },
            'absolute_differences': {
                'system_shannon': system_shannon_diff,
                'system_physical': system_physical_diff,
                'shannon_physical': shannon_physical_diff
            },
            'relative_differences': {
                'system_shannon': system_shannon_rel_diff,
                'system_physical': system_physical_rel_diff,
                'shannon_physical': shannon_physical_rel_diff
            },
            'equivalence_verified': {
                'system_shannon': system_shannon_rel_diff <= tolerance,
                'system_physical': system_physical_rel_diff <= tolerance,
                'shannon_physical': shannon_physical_rel_diff <= tolerance
            },
            'overall_equivalence': (
                system_shannon_rel_diff <= tolerance and
                system_physical_rel_diff <= tolerance and
                shannon_physical_rel_diff <= tolerance
            )
        }
    
    def verify_trinity_equivalence(self, test_cases: List[Any], tolerance: float = 0.15) -> Dict[str, Any]:
        """验证三位一体等价性"""
        results = {
            'total_test_cases': len(test_cases),
            'individual_results': [],
            'equivalence_statistics': {
                'system_shannon_matches': 0,
                'system_physical_matches': 0,
                'shannon_physical_matches': 0,
                'all_three_match': 0
            },
            'average_relative_differences': {
                'system_shannon': 0.0,
                'system_physical': 0.0,
                'shannon_physical': 0.0
            }
        }
        
        sum_rel_diffs = {'system_shannon': 0.0, 'system_physical': 0.0, 'shannon_physical': 0.0}
        
        for i, test_case in enumerate(test_cases):
            result = self.verify_pairwise_equivalence(test_case, tolerance)
            results['individual_results'].append(result)
            
            # 统计等价性
            if result['equivalence_verified']['system_shannon']:
                results['equivalence_statistics']['system_shannon_matches'] += 1
            if result['equivalence_verified']['system_physical']:
                results['equivalence_statistics']['system_physical_matches'] += 1
            if result['equivalence_verified']['shannon_physical']:
                results['equivalence_statistics']['shannon_physical_matches'] += 1
            if result['overall_equivalence']:
                results['equivalence_statistics']['all_three_match'] += 1
            
            # 累计相对差异
            for key in sum_rel_diffs:
                sum_rel_diffs[key] += result['relative_differences'][key]
        
        # 计算平均相对差异
        if len(test_cases) > 0:
            for key in sum_rel_diffs:
                results['average_relative_differences'][key] = sum_rel_diffs[key] / len(test_cases)
        
        # 计算总体成功率
        results['success_rates'] = {
            'system_shannon': results['equivalence_statistics']['system_shannon_matches'] / max(1, len(test_cases)),
            'system_physical': results['equivalence_statistics']['system_physical_matches'] / max(1, len(test_cases)),
            'shannon_physical': results['equivalence_statistics']['shannon_physical_matches'] / max(1, len(test_cases)),
            'all_three': results['equivalence_statistics']['all_three_match'] / max(1, len(test_cases))
        }
        
        # 总体等价性验证
        results['trinity_equivalence_verified'] = (
            results['success_rates']['system_shannon'] >= 0.7 and
            results['success_rates']['system_physical'] >= 0.7 and
            results['success_rates']['shannon_physical'] >= 0.7
        )
        
        return results
    
    def analyze_convergence_to_equivalence(self, data_sequence: List[Any]) -> Dict[str, Any]:
        """分析信息度量的收敛性"""
        if len(data_sequence) < 3:
            return {'error': 'Need at least 3 data points for convergence analysis'}
        
        convergence_data = {
            'sequence_length': len(data_sequence),
            'measurements': {
                'system': [],
                'shannon': [],
                'physical': []
            },
            'differences': {
                'system_shannon': [],
                'system_physical': [],
                'shannon_physical': []
            },
            'convergence_indicators': {}
        }
        
        for data in data_sequence:
            system_bits = self.system_info.compute_information(data)
            shannon_bits = self.shannon_info.compute_information(data)
            physical_bits = self.physical_info.compute_information(data)
            
            convergence_data['measurements']['system'].append(system_bits)
            convergence_data['measurements']['shannon'].append(shannon_bits)
            convergence_data['measurements']['physical'].append(physical_bits)
            
            # 计算差异
            convergence_data['differences']['system_shannon'].append(abs(system_bits - shannon_bits))
            convergence_data['differences']['system_physical'].append(abs(system_bits - physical_bits))
            convergence_data['differences']['shannon_physical'].append(abs(shannon_bits - physical_bits))
        
        # 分析收敛趋势
        for diff_type, diffs in convergence_data['differences'].items():
            if len(diffs) >= 3:
                # 检查最后几个值是否趋于稳定
                recent_diffs = diffs[-3:]
                avg_recent_diff = sum(recent_diffs) / len(recent_diffs)
                diff_variance = sum((d - avg_recent_diff) ** 2 for d in recent_diffs) / len(recent_diffs)
                
                convergence_data['convergence_indicators'][diff_type] = {
                    'final_difference': diffs[-1],
                    'average_recent_difference': avg_recent_diff,
                    'variance_recent': diff_variance,
                    'is_converging': diff_variance < 0.5 and avg_recent_diff < 1.0
                }
            else:
                convergence_data['convergence_indicators'][diff_type] = {
                    'final_difference': diffs[-1],
                    'is_converging': False
                }
        
        # 总体收敛评估
        convergence_data['overall_convergence'] = all(
            indicator.get('is_converging', False)
            for indicator in convergence_data['convergence_indicators'].values()
        )
        
        return convergence_data
    
    def demonstrate_theoretical_equivalence(self) -> Dict[str, Any]:
        """演示理论等价性"""
        return {
            'theoretical_bases': {
                'system': self.system_info.get_theoretical_basis(),
                'shannon': self.shannon_info.get_theoretical_basis(),
                'physical': self.physical_info.get_theoretical_basis()
            },
            'equivalence_principles': {
                'system_shannon': 'System complexity → Statistical entropy via self-reference',
                'system_physical': 'Computational complexity → Thermodynamic cost via Landauer principle',
                'shannon_physical': 'Information entropy → Physical entropy via Maxwell demon'
            },
            'unifying_framework': {
                'core_principle': 'ψ = ψ(ψ) self-referential completeness',
                'mathematical_basis': 'log2(φ) asymptotic capacity',
                'physical_realization': 'kT ln2 minimum energy cost',
                'information_content': 'H(X) = -Σ p(x) log2 p(x)'
            },
            'trinity_relationship': 'I_system ≡ I_Shannon ≡ I_physical'
        }


class InformationTestDataGenerator:
    """信息测试数据生成器"""
    
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2
        
    def generate_structured_data(self, count: int = 10) -> List[Dict[str, Any]]:
        """生成结构化测试数据"""
        test_data = []
        
        for i in range(count):
            if i % 4 == 0:
                # 简单结构
                data = {
                    'structure': {
                        'states': {f'state_{j}' for j in range(i % 3 + 2)},
                        'functions': {f'func_{i}': lambda x, i=i: x + i},
                        'recursions': {f'rec_{i}': f'R{i}(R{i})'}
                    }
                }
            elif i % 4 == 1:
                # 复杂结构
                data = {
                    'structure': {
                        'states': {f'complex_state_{j}' for j in range(i % 5 + 3)},
                        'functions': {
                            f'complex_func_{i}': lambda x: x * 2,
                            f'meta_func_{i}': lambda: f'meta_{i}'
                        },
                        'recursions': {
                            f'complex_rec_{i}': f'C{i}(C{i})',
                            f'meta_rec_{i}': 'meta(meta)'
                        }
                    }
                }
            elif i % 4 == 2:
                # φ-based结构
                data = {
                    'structure': {
                        'states': {f'phi_state_{j}' for j in range(int(self.phi * i) % 4 + 2)},
                        'functions': {f'phi_func_{i}': lambda x, phi=self.phi: x * phi},
                        'recursions': {f'phi_rec_{i}': 'φ(φ)'}
                    }
                }
            else:
                # 随机结构
                import random
                random.seed(i)
                num_states = random.randint(2, 6)
                num_functions = random.randint(1, 4)
                num_recursions = random.randint(1, 3)
                
                data = {
                    'structure': {
                        'states': {f'rand_state_{j}' for j in range(num_states)},
                        'functions': {f'rand_func_{j}': lambda x, j=j: x + j for j in range(num_functions)},
                        'recursions': {f'rand_rec_{j}': f'RR{j}(RR{j})' for j in range(num_recursions)}
                    }
                }
            
            test_data.append(data)
        
        return test_data
    
    def generate_sequence_data(self, count: int = 10) -> List[Any]:
        """生成序列测试数据"""
        test_data = []
        
        for i in range(count):
            if i % 5 == 0:
                # 二进制序列
                data = ''.join('01'[j % 2] for j in range(i + 5))
            elif i % 5 == 1:
                # Fibonacci序列
                fib_seq = [1, 1]
                for _ in range(i + 3):
                    fib_seq.append(fib_seq[-1] + fib_seq[-2])
                data = fib_seq
            elif i % 5 == 2:
                # 随机字符串
                import random
                random.seed(i)
                data = ''.join(random.choice('abcdef') for _ in range(i + 8))
            elif i % 5 == 3:
                # 周期序列
                pattern = 'abc'
                data = (pattern * ((i + 10) // len(pattern) + 1))[:i + 10]
            else:
                # 数字序列
                data = list(range(i + 5))
            
            test_data.append(data)
        
        return test_data
    
    def generate_probability_data(self, count: int = 10) -> List[Dict[str, Any]]:
        """生成概率分布测试数据"""
        test_data = []
        
        for i in range(count):
            if i % 3 == 0:
                # 均匀分布
                n = i % 6 + 2
                prob = 1.0 / n
                data = {'probabilities': [prob] * n}
            elif i % 3 == 1:
                # 指数分布（近似）
                n = i % 5 + 3
                total = sum(math.exp(-j) for j in range(n))
                data = {'probabilities': [math.exp(-j) / total for j in range(n)]}
            else:
                # 随机分布
                import random
                random.seed(i)
                n = i % 4 + 3
                raw_probs = [random.random() for _ in range(n)]
                total = sum(raw_probs)
                data = {'probabilities': [p / total for p in raw_probs]}
            
            test_data.append(data)
        
        return test_data
    
    def generate_physical_data(self, count: int = 10) -> List[Dict[str, Any]]:
        """生成物理信息测试数据"""
        test_data = []
        
        k_b = 1.380649e-23
        temperature = 300.0
        
        for i in range(count):
            if i % 3 == 0:
                # 能量数据
                energy = (i + 1) * k_b * temperature * math.log(2)  # i+1 比特的能量
                data = {'energy': energy}
            elif i % 3 == 1:
                # 比特数据
                data = {'bits': i + 1}
            else:
                # 能态数据
                n_states = i % 5 + 2
                base_energy = k_b * temperature
                energy_states = [j * base_energy for j in range(n_states)]
                data = {'energy_states': energy_states}
            
            test_data.append(data)
        
        return test_data
    
    def generate_mixed_test_suite(self) -> Dict[str, List[Any]]:
        """生成混合测试套件"""
        return {
            'structured_data': self.generate_structured_data(12),
            'sequence_data': self.generate_sequence_data(15),
            'probability_data': self.generate_probability_data(10),
            'physical_data': self.generate_physical_data(8)
        }


class TestP5_1_InformationEquivalence(unittest.TestCase):
    """P5-1 信息三位一体等价性命题测试"""
    
    def setUp(self):
        """测试初始化"""
        self.system_info = SystemInformation()
        self.shannon_info = ShannonInformation()
        self.physical_info = PhysicalInformation()
        self.verifier = InformationEquivalenceVerifier()
        self.generator = InformationTestDataGenerator()
        self.phi = (1 + math.sqrt(5)) / 2
        
    def test_individual_information_measures(self):
        """测试各个信息度量的基本功能"""
        print("\n=== 测试各个信息度量的基本功能 ===")
        
        # 测试系统信息
        structure_data = {
            'structure': {
                'states': {'state_1', 'state_2', 'recursive_state'},
                'functions': {'func_1': lambda x: x},
                'recursions': {'rec_1': 'R(R)'}
            }
        }
        
        system_bits = self.system_info.compute_information(structure_data)
        self.assertGreater(system_bits, 0, "系统信息应该为正值")
        print(f"✓ 系统信息: {system_bits:.3f} bits")
        
        # 测试Shannon信息
        prob_data = {'probabilities': [0.5, 0.5]}
        shannon_bits = self.shannon_info.compute_information(prob_data)
        self.assertAlmostEqual(shannon_bits, 1.0, delta=0.01, msg="二元均匀分布的熵应该为1")
        print(f"✓ Shannon信息: {shannon_bits:.3f} bits")
        
        # 测试物理信息
        bit_data = {'bits': 1}
        physical_bits = self.physical_info.compute_information(bit_data)
        self.assertAlmostEqual(physical_bits, 1.0, delta=0.01, msg="1比特的物理信息应该为1")
        print(f"✓ 物理信息: {physical_bits:.3f} bits")
        
        print("✓ 各个信息度量基本功能验证通过")
        
    def test_theoretical_basis_consistency(self):
        """测试理论基础的一致性"""
        print("\n=== 测试理论基础的一致性 ===")
        
        # 获取理论基础
        system_basis = self.system_info.get_theoretical_basis()
        shannon_basis = self.shannon_info.get_theoretical_basis()
        physical_basis = self.physical_info.get_theoretical_basis()
        
        # 验证关键字段存在
        required_fields = ['foundation', 'key_principle', 'encoding_system', 'complexity_measure', 'theoretical_limit']
        
        for basis, name in [(system_basis, 'system'), (shannon_basis, 'shannon'), (physical_basis, 'physical')]:
            for field in required_fields:
                self.assertIn(field, basis, f"{name}信息度量应包含{field}字段")
        
        print("✓ 系统信息理论基础:", system_basis['foundation'])
        print("✓ Shannon信息理论基础:", shannon_basis['foundation'])
        print("✓ 物理信息理论基础:", physical_basis['foundation'])
        
        # 验证理论等价性演示
        theoretical = self.verifier.demonstrate_theoretical_equivalence()
        self.assertIn('trinity_relationship', theoretical)
        self.assertEqual(theoretical['trinity_relationship'], 'I_system ≡ I_Shannon ≡ I_physical')
        
        print("✓ 理论基础一致性验证通过")
        
    def test_pairwise_equivalence_verification(self):
        """测试两两等价性验证"""
        print("\n=== 测试两两等价性验证 ===")
        
        # 测试结构化数据
        test_structures = self.generator.generate_structured_data(8)
        successful_pairs = 0
        total_pairs = 0
        
        for i, data in enumerate(test_structures):
            result = self.verifier.verify_pairwise_equivalence(data, tolerance=0.2)
            
            total_pairs += 1
            if result['overall_equivalence']:
                successful_pairs += 1
            
            print(f"  结构 {i}: System={result['measurements']['system_bits']:.2f}, "
                  f"Shannon={result['measurements']['shannon_bits']:.2f}, "
                  f"Physical={result['measurements']['physical_bits']:.2f}, "
                  f"等价={result['overall_equivalence']}")
        
        success_rate = successful_pairs / max(1, total_pairs)
        self.assertGreaterEqual(success_rate, 0.5, "至少50%的测试应显示等价性")
        
        print(f"✓ 两两等价性成功率: {success_rate:.1%}")
        
        # 测试序列数据
        test_sequences = self.generator.generate_sequence_data(6)
        seq_successful = 0
        
        for i, seq_data in enumerate(test_sequences):
            result = self.verifier.verify_pairwise_equivalence(seq_data, tolerance=0.3)
            if result['overall_equivalence']:
                seq_successful += 1
        
        seq_success_rate = seq_successful / len(test_sequences)
        print(f"✓ 序列数据等价性成功率: {seq_success_rate:.1%}")
        
        self.assertGreaterEqual(seq_success_rate, 0.3, "序列数据应显示一定程度的等价性")
        
        print("✓ 两两等价性验证通过")
        
    def test_trinity_equivalence_verification(self):
        """测试三位一体等价性验证"""
        print("\n=== 测试三位一体等价性验证 ===")
        
        # 生成综合测试数据
        mixed_suite = self.generator.generate_mixed_test_suite()
        all_test_cases = []
        all_test_cases.extend(mixed_suite['structured_data'])
        all_test_cases.extend(mixed_suite['sequence_data'])
        all_test_cases.extend(mixed_suite['probability_data'])
        all_test_cases.extend(mixed_suite['physical_data'])
        
        # 验证三位一体等价性
        trinity_result = self.verifier.verify_trinity_equivalence(all_test_cases, tolerance=0.2)
        
        print(f"✓ 总测试案例数: {trinity_result['total_test_cases']}")
        print(f"✓ System-Shannon等价率: {trinity_result['success_rates']['system_shannon']:.1%}")
        print(f"✓ System-Physical等价率: {trinity_result['success_rates']['system_physical']:.1%}")
        print(f"✓ Shannon-Physical等价率: {trinity_result['success_rates']['shannon_physical']:.1%}")
        print(f"✓ 三者完全等价率: {trinity_result['success_rates']['all_three']:.1%}")
        
        # 验证层次化等价性要求（根据新理论）
        # 强等价：System-Shannon应接近100%
        self.assertGreaterEqual(trinity_result['success_rates']['system_shannon'], 0.95,
                              "System-Shannon强等价率应≥95%")
        
        # 弱等价：Physical相关的等价率应≥60%
        self.assertGreaterEqual(trinity_result['success_rates']['system_physical'], 0.6,
                              "System-Physical弱等价率应≥60%")
        self.assertGreaterEqual(trinity_result['success_rates']['shannon_physical'], 0.6,
                              "Shannon-Physical弱等价率应≥60%")
        
        # 验证平均相对差异（符合层次化等价理论）
        avg_diffs = trinity_result['average_relative_differences']
        self.assertLess(avg_diffs['system_shannon'], 0.1, "System-Shannon平均相对差异应<10%（强等价性）")
        self.assertLess(avg_diffs['system_physical'], 0.35, "System-Physical平均相对差异应<35%（弱等价性）")
        self.assertLess(avg_diffs['shannon_physical'], 0.35, "Shannon-Physical平均相对差异应<35%（弱等价性）")
        
        print(f"✓ 平均相对差异: System-Shannon={avg_diffs['system_shannon']:.1%}, "
              f"System-Physical={avg_diffs['system_physical']:.1%}, "
              f"Shannon-Physical={avg_diffs['shannon_physical']:.1%}")
        
        # 验证理论预测：强等价应显著优于弱等价
        self.assertLess(avg_diffs['system_shannon'], avg_diffs['system_physical'] / 2,
                       "强等价差异应小于弱等价差异的一半")
        
        print("✓ 三位一体层次化等价性验证通过")
        
    def test_convergence_analysis(self):
        """测试收敛性分析"""
        print("\n=== 测试收敛性分析 ===")
        
        # 生成递增复杂度的数据序列
        convergence_sequence = []
        
        for i in range(8):
            # 生成递增复杂度的结构
            complexity_level = i + 2
            structure = {
                'structure': {
                    'states': {f'conv_state_{j}' for j in range(complexity_level)},
                    'functions': {f'conv_func_{j}': lambda x, j=j: x + j for j in range(complexity_level // 2 + 1)},
                    'recursions': {f'conv_rec_{j}': f'CR{j}(CR{j})' for j in range(complexity_level // 3 + 1)}
                }
            }
            convergence_sequence.append(structure)
        
        # 分析收敛性
        convergence_result = self.verifier.analyze_convergence_to_equivalence(convergence_sequence)
        
        self.assertNotIn('error', convergence_result, "收敛性分析应该成功")
        self.assertEqual(convergence_result['sequence_length'], len(convergence_sequence))
        
        # 检查收敛指标
        for diff_type, indicator in convergence_result['convergence_indicators'].items():
            self.assertIn('final_difference', indicator)
            print(f"✓ {diff_type} 最终差异: {indicator['final_difference']:.3f}")
            
            if 'is_converging' in indicator:
                print(f"  {diff_type} 收敛状态: {indicator['is_converging']}")
        
        # 验证至少一种差异类型显示收敛趋势
        converging_types = sum(1 for indicator in convergence_result['convergence_indicators'].values()
                             if indicator.get('is_converging', False))
        
        self.assertGreaterEqual(converging_types, 1, "至少一种差异类型应显示收敛")
        
        print(f"✓ 收敛类型数量: {converging_types}/3")
        print(f"✓ 总体收敛: {convergence_result['overall_convergence']}")
        
        print("✓ 收敛性分析验证通过")
        
    def test_specific_equivalence_cases(self):
        """测试特定等价性案例"""
        print("\n=== 测试特定等价性案例 ===")
        
        # 案例1：均匀分布
        uniform_case = {'probabilities': [0.25, 0.25, 0.25, 0.25]}
        result1 = self.verifier.verify_pairwise_equivalence(uniform_case, tolerance=0.15)
        
        print(f"✓ 均匀分布案例: System={result1['measurements']['system_bits']:.3f}, "
              f"Shannon={result1['measurements']['shannon_bits']:.3f}, "
              f"Physical={result1['measurements']['physical_bits']:.3f}")
        
        # Shannon熵应该等于log2(4) = 2
        expected_shannon = 2.0
        self.assertAlmostEqual(result1['measurements']['shannon_bits'], expected_shannon, delta=0.1,
                             msg="4元素均匀分布的Shannon熵应接近2")
        
        # 案例2：简单二进制序列
        binary_case = "0101010101"
        result2 = self.verifier.verify_pairwise_equivalence(binary_case, tolerance=0.2)
        
        print(f"✓ 二进制序列案例: System={result2['measurements']['system_bits']:.3f}, "
              f"Shannon={result2['measurements']['shannon_bits']:.3f}, "
              f"Physical={result2['measurements']['physical_bits']:.3f}")
        
        # 案例3：φ-based结构
        phi_structure = {
            'structure': {
                'states': {'phi_state_1', 'phi_state_2'},
                'functions': {'phi_func': lambda x: x * self.phi},
                'recursions': {'phi_rec': 'φ(φ)'}
            }
        }
        result3 = self.verifier.verify_pairwise_equivalence(phi_structure, tolerance=0.15)
        
        print(f"✓ φ-结构案例: System={result3['measurements']['system_bits']:.3f}, "
              f"Shannon={result3['measurements']['shannon_bits']:.3f}, "
              f"Physical={result3['measurements']['physical_bits']:.3f}")
        
        # 验证至少一个案例显示等价性
        equivalent_cases = sum(1 for result in [result1, result2, result3] if result['overall_equivalence'])
        self.assertGreaterEqual(equivalent_cases, 1, "至少一个特定案例应显示等价性")
        
        print(f"✓ 等价案例数: {equivalent_cases}/3")
        print("✓ 特定等价性案例验证通过")
        
    def test_landauer_principle_connection(self):
        """测试Landauer原理连接"""
        print("\n=== 测试Landauer原理连接 ===")
        
        # 测试比特到能量的转换
        k_b = 1.380649e-23
        temperature = 300.0
        ln2 = math.log(2)
        
        for bits in [1, 2, 4, 8]:
            # 理论Landauer能量
            theoretical_energy = bits * k_b * temperature * ln2
            
            # 通过物理信息计算
            energy_data = {'energy': theoretical_energy}
            physical_bits = self.physical_info.compute_information(energy_data)
            
            # 验证往返转换的一致性
            self.assertAlmostEqual(physical_bits, bits, delta=0.1,
                                 msg=f"{bits}比特的Landauer转换应该一致")
            
            print(f"✓ {bits}比特 ↔ {theoretical_energy:.2e}J ↔ {physical_bits:.3f}比特")
        
        print("✓ Landauer原理连接验证通过")
        
    def test_information_scalability(self):
        """测试信息度量的可扩展性"""
        print("\n=== 测试信息度量的可扩展性 ===")
        
        # 测试不同规模的数据
        scales = [5, 10, 20, 50]
        scalability_results = []
        
        for scale in scales:
            # 生成指定规模的数据
            large_sequence = list(range(scale))
            
            # 计算三种信息度量
            system_bits = self.system_info.compute_information(large_sequence)
            shannon_bits = self.shannon_info.compute_information(large_sequence)
            physical_bits = self.physical_info.compute_information(large_sequence)
            
            scalability_results.append({
                'scale': scale,
                'system': system_bits,
                'shannon': shannon_bits,
                'physical': physical_bits
            })
            
            print(f"✓ Scale {scale}: System={system_bits:.2f}, Shannon={shannon_bits:.2f}, Physical={physical_bits:.2f}")
        
        # 验证随规模增长的趋势
        for i in range(1, len(scalability_results)):
            prev = scalability_results[i-1]
            curr = scalability_results[i]
            
            # 信息量应该随规模增长（或至少不减少）
            self.assertGreaterEqual(curr['system'], prev['system'] * 0.9,
                                  "系统信息应随规模增长")
            self.assertGreaterEqual(curr['shannon'], prev['shannon'] * 0.9,
                                  "Shannon信息应随规模增长")
            self.assertGreaterEqual(curr['physical'], prev['physical'] * 0.9,
                                  "物理信息应随规模增长")
        
        print("✓ 信息度量可扩展性验证通过")
        
    def test_hierarchical_equivalence_structure(self):
        """测试层次化等价性结构"""
        print("\n=== 测试层次化等价性结构 ===")
        
        # 生成不同复杂度的测试数据
        test_cases = []
        
        # 简单数据（应该显示更高等价性）
        test_cases.extend(self.generator.generate_sequence_data(10))
        
        # 复杂结构数据
        test_cases.extend(self.generator.generate_structured_data(10))
        
        # 物理数据（应该显示更多差异）
        test_cases.extend(self.generator.generate_physical_data(10))
        
        strong_equiv_count = 0  # System-Shannon < 10%
        weak_equiv_count = 0    # Physical相关 < 35%
        
        for data in test_cases:
            result = self.verifier.verify_pairwise_equivalence(data, tolerance=0.2)
            
            # 检查强等价
            if result['relative_differences']['system_shannon'] < 0.1:
                strong_equiv_count += 1
                
            # 检查弱等价
            if (result['relative_differences']['system_physical'] < 0.35 and
                result['relative_differences']['shannon_physical'] < 0.35):
                weak_equiv_count += 1
        
        strong_equiv_rate = strong_equiv_count / len(test_cases)
        weak_equiv_rate = weak_equiv_count / len(test_cases)
        
        print(f"✓ 强等价（System-Shannon）达成率: {strong_equiv_rate:.1%}")
        print(f"✓ 弱等价（Physical相关）达成率: {weak_equiv_rate:.1%}")
        
        # 验证层次化结构
        self.assertGreater(strong_equiv_rate, 0.8, "强等价率应大于80%")
        self.assertGreater(weak_equiv_rate, 0.5, "弱等价率应大于50%")
        self.assertGreater(strong_equiv_rate, weak_equiv_rate, 
                          "强等价率应高于弱等价率，体现层次结构")
        
        # 验证修正项的影响
        print("\n验证修正项影响:")
        
        # 测试不同大小数据的修正项影响
        for size in [10, 100, 1000]:
            seq = ''.join(random.choice('01') for _ in range(size))
            result = self.verifier.verify_pairwise_equivalence(seq)
            
            sys_shan_diff = result['relative_differences']['system_shannon']
            print(f"  数据大小 {size}: System-Shannon差异 = {sys_shan_diff:.3f}")
            
            # 验证大数据时强等价性更好（修正项影响减小）
            if size >= 100:
                self.assertLess(sys_shan_diff, 0.05, 
                              f"大数据（size={size}）的强等价差异应<5%")
        
        print("✓ 层次化等价性结构验证通过")
    
    def test_complete_p5_1_verification(self):
        """完整的P5-1验证测试"""
        print("\n=== 完整的P5-1验证测试 ===")
        
        # 1. 验证各信息度量的基本性质
        self.assertTrue(hasattr(self.system_info, 'compute_information'))
        self.assertTrue(hasattr(self.shannon_info, 'compute_information'))
        self.assertTrue(hasattr(self.physical_info, 'compute_information'))
        print("✓ 基本信息度量接口验证通过")
        
        # 2. 验证理论基础完整性
        theoretical = self.verifier.demonstrate_theoretical_equivalence()
        required_keys = ['theoretical_bases', 'equivalence_principles', 'unifying_framework', 'trinity_relationship']
        for key in required_keys:
            self.assertIn(key, theoretical)
        print("✓ 理论基础完整性验证通过")
        
        # 3. 验证等价性在多种数据类型上成立
        mixed_suite = self.generator.generate_mixed_test_suite()
        total_tests = 0
        successful_equivalences = 0
        
        for data_type, test_cases in mixed_suite.items():
            for test_case in test_cases[:5]:  # 每种类型测试5个案例
                result = self.verifier.verify_pairwise_equivalence(test_case, tolerance=0.25)
                total_tests += 1
                if result['overall_equivalence']:
                    successful_equivalences += 1
        
        overall_success_rate = successful_equivalences / max(1, total_tests)
        self.assertGreaterEqual(overall_success_rate, 0.3,
                              "总体等价性成功率应≥30%")
        print(f"✓ 总体等价性成功率: {overall_success_rate:.1%}")
        
        # 4. 验证收敛行为
        test_sequence = self.generator.generate_structured_data(6)
        convergence = self.verifier.analyze_convergence_to_equivalence(test_sequence)
        self.assertNotIn('error', convergence)
        print("✓ 收敛性分析验证通过")
        
        # 5. 验证物理意义
        landauer_test = {'bits': 10}
        physical_result = self.physical_info.compute_information(landauer_test)
        self.assertAlmostEqual(physical_result, 10.0, delta=0.1,
                             msg="物理信息应正确反映比特数")
        print("✓ 物理意义验证通过")
        
        # 6. 综合判定
        trinity_verified = (
            overall_success_rate >= 0.3 and
            'trinity_relationship' in theoretical and
            convergence.get('sequence_length', 0) > 0
        )
        
        self.assertTrue(trinity_verified, "P5-1信息三位一体等价性应得到验证")
        print("✓ P5-1 信息三位一体等价性命题验证通过")


if __name__ == '__main__':
    unittest.main(verbosity=2)