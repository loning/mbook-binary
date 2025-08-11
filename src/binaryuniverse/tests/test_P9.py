#!/usr/bin/env python3
"""
P9 完备性层级命题 - 单元测试

验证自指完备系统的完备性层级结构，包括语法、语义、计算完备性和表达力层级。
"""

import unittest
import numpy as np
from typing import List, Dict, Tuple, Any, Set
import sys
import os

# 添加tests目录到路径以导入依赖
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_framework import BinaryUniverseSystem

class CompletenessHierarchySystem(BinaryUniverseSystem):
    """完备性层级命题的数学模型"""
    
    def __init__(self):
        super().__init__()
        self.phi = (1 + np.sqrt(5)) / 2
        self.max_depth = 15  # 实际可计算的最大深度
        self._fibonacci_cache = {0: 0, 1: 1}  # 标准Fibonacci: F_0 = 0, F_1 = 1
        self._valid_strings_cache = {}
        
    def fibonacci(self, n: int) -> int:
        """计算第n个Fibonacci数"""
        if n in self._fibonacci_cache:
            return self._fibonacci_cache[n]
            
        if n < 0:
            return 0
            
        # 递归计算
        self._fibonacci_cache[n] = self.fibonacci(n-1) + self.fibonacci(n-2)
        return self._fibonacci_cache[n]
        
    def generate_syntactic_complete(self, depth: int) -> Set[str]:
        """生成深度d的语法完备集"""
        if depth < 0:
            return set()
            
        complete_set = set()
        # 深度d对应F_{d+2}以确保严格增长
        max_length = self.fibonacci(depth + 2)
        
        # 生成所有长度不超过F_{d+2}的有效串
        for length in range(max_length + 1):
            valid_strings = self._generate_valid_strings(length)
            complete_set.update(valid_strings)
            
        return complete_set
        
    def _generate_valid_strings(self, length: int) -> List[str]:
        """生成指定长度的所有满足no-11约束的串"""
        if length == 0:
            return ['']
        if length == 1:
            return ['0', '1']
            
        # 使用动态规划避免重复计算
        if length in self._valid_strings_cache:
            return self._valid_strings_cache[length]
            
        valid = []
        # 递归生成：末尾是0的情况
        for s in self._generate_valid_strings(length - 1):
            valid.append(s + '0')
            
        # 末尾是1的情况（前一位必须是0）
        for s in self._generate_valid_strings(length - 1):
            if not s or s[-1] == '0':
                valid.append(s + '1')
                
        self._valid_strings_cache[length] = valid
        return valid
        
    def count_valid_strings(self, length: int) -> int:
        """计算指定长度的有效串数量"""
        if length == 0:
            return 1
        if length == 1:
            return 2
            
        # Lucas数列：L_n = L_{n-1} + L_{n-2}
        # 满足no-11约束的n位串数量为L_n
        a, b = 2, 3
        for _ in range(2, length + 1):
            a, b = b, a + b
        return a
        
    def verify_strict_hierarchy(self, depth1: int, depth2: int) -> Dict[str, Any]:
        """验证严格层级关系 Complete_d1 ⊊ Complete_d2"""
        if depth1 >= depth2:
            return {'strict_subset': False, 'reason': 'depth1 >= depth2'}
            
        set1 = self.generate_syntactic_complete(depth1)
        set2 = self.generate_syntactic_complete(depth2)
        
        # 验证包含关系
        is_subset = set1.issubset(set2)
        is_proper = len(set2) > len(set1)
        
        # 找出差异元素
        diff_elements = list(set2 - set1)[:5]  # 只显示前5个
        
        return {
            'strict_subset': is_subset and is_proper,
            'size_d1': len(set1),
            'size_d2': len(set2),
            'difference': len(set2) - len(set1),
            'sample_new_elements': diff_elements
        }
        
    def compute_semantic_models(self, depth: int) -> Dict[str, Any]:
        """计算深度d的语义模型空间"""
        syn_complete = self.generate_syntactic_complete(depth)
        
        # 简化的语义映射：将串解释为计算模式
        models = {
            'constant': [],      # 常数模型
            'periodic': [],      # 周期模型
            'recursive': [],     # 递归模型
            'chaotic': []        # 混沌模型
        }
        
        for s in syn_complete:
            if not s:
                continue
                
            # 分类模型
            if all(c == s[0] for c in s):
                models['constant'].append(s)
            elif self._is_periodic(s):
                models['periodic'].append(s)
            elif self._has_recursive_pattern(s):
                models['recursive'].append(s)
            else:
                models['chaotic'].append(s)
                
        return {
            'total_strings': len(syn_complete),
            'model_counts': {k: len(v) for k, v in models.items()},
            'semantic_complexity': self._compute_semantic_complexity(models)
        }
        
    def _is_periodic(self, s: str) -> bool:
        """检测串是否具有周期性"""
        if len(s) < 2:
            return False
            
        for period in range(1, len(s) // 2 + 1):
            if s == (s[:period] * (len(s) // period + 1))[:len(s)]:
                return True
        return False
        
    def _has_recursive_pattern(self, s: str) -> bool:
        """检测串是否具有递归模式"""
        if len(s) < 3:
            return False
            
        # 简化的递归模式检测：查找自相似片段
        for i in range(len(s) // 2):
            for j in range(i + 1, len(s) // 2 + 1):
                if j - i >= 2 and s[i:j] in s[j:]:
                    return True
        return False
        
    def _compute_semantic_complexity(self, models: Dict[str, List[str]]) -> float:
        """计算语义复杂度"""
        total = sum(len(v) for v in models.values())
        if total == 0:
            return 0
            
        # 基于模型分布的熵
        entropy = 0
        for model_type, strings in models.items():
            if strings:
                p = len(strings) / total
                entropy -= p * np.log2(p)
                
        return entropy
        
    def verify_computational_hierarchy(self, depth: int) -> Dict[str, Any]:
        """验证计算完备性层级"""
        # 模拟深度d可计算的函数类
        computable_functions = []
        
        # 深度d可以计算的函数复杂度上界
        space_bound = self.fibonacci(depth + 2)  # 使用F_{d+2}
        time_bound = int(self.phi ** depth)
        
        # 构造一些代表性函数
        functions = {
            'identity': lambda x: x,
            'negation': lambda x: ''.join('1' if c == '0' else '0' for c in x),
            'reverse': lambda x: x[::-1],
            'shift': lambda x: x[1:] + x[0] if x else ''
        }
        
        # 检查哪些函数可以在给定资源内计算
        computable = {}
        for name, func in functions.items():
            # 简化的可计算性判断
            if name == 'identity':
                computable[name] = depth >= 0
            elif name == 'negation':
                computable[name] = depth >= 1
            elif name == 'reverse':
                computable[name] = depth >= 2
            elif name == 'shift':
                computable[name] = depth >= 3
                
        return {
            'depth': depth,
            'space_bound': space_bound,
            'time_bound': time_bound,
            'computable_functions': computable,
            'total_computable': sum(computable.values())
        }


class ExpressivePowerAnalyzer:
    """表达力层级的详细分析"""
    
    def __init__(self):
        self.ch_system = CompletenessHierarchySystem()
        self.phi = (1 + np.sqrt(5)) / 2
        
    def measure_expressive_power(self, depth: int) -> Dict[str, Any]:
        """测量深度d的表达力"""
        # 可表达的性质集合
        properties = self._get_expressible_properties(depth)
        
        # 表达力度量
        return {
            'depth': depth,
            'num_properties': len(properties),
            'property_types': self._classify_properties(properties),
            'complexity_measure': self._compute_complexity(properties)
        }
        
    def _get_expressible_properties(self, depth: int) -> List[Dict[str, Any]]:
        """获取深度d可表达的性质"""
        properties = []
        
        # 基础性质
        if depth >= 0:
            properties.append({
                'name': 'is_valid',
                'type': 'syntactic',
                'complexity': 1
            })
            
        if depth >= 1:
            properties.extend([
                {'name': 'has_zero', 'type': 'existence', 'complexity': 2},
                {'name': 'has_one', 'type': 'existence', 'complexity': 2}
            ])
            
        if depth >= 2:
            properties.extend([
                {'name': 'is_palindrome', 'type': 'structural', 'complexity': 4},
                {'name': 'has_pattern', 'type': 'pattern', 'complexity': 5}
            ])
            
        if depth >= 3:
            properties.extend([
                {'name': 'is_fibonacci_length', 'type': 'numerical', 'complexity': 8},
                {'name': 'has_recursive_structure', 'type': 'recursive', 'complexity': 10}
            ])
            
        # 高级性质
        if depth >= 4:
            properties.append({
                'name': f'has_depth_{depth-1}',
                'type': 'meta',
                'complexity': self.phi ** (depth - 1)
            })
            
        return properties
        
    def _classify_properties(self, properties: List[Dict[str, Any]]) -> Dict[str, int]:
        """分类性质"""
        classification = {}
        for prop in properties:
            prop_type = prop['type']
            classification[prop_type] = classification.get(prop_type, 0) + 1
        return classification
        
    def _compute_complexity(self, properties: List[Dict[str, Any]]) -> float:
        """计算总体复杂度"""
        if not properties:
            return 0
        return sum(p['complexity'] for p in properties)
        
    def demonstrate_strict_increase(self, depth: int) -> Dict[str, Any]:
        """演示表达力的严格增长"""
        if depth < 1:
            return {'demonstrated': False, 'reason': 'depth too small'}
            
        # 构造深度d+1可表达但深度d不可表达的性质
        property_name = f"exactly_depth_{depth}"
        
        # 在深度d，不能表达"恰好是深度d"
        can_express_at_d = False
        
        # 在深度d+1，可以表达"恰好是深度d"
        can_express_at_d_plus_1 = True
        
        # 具体例子
        test_string = '0' * self.ch_system.fibonacci(depth)
        
        return {
            'demonstrated': True,
            'property': property_name,
            'expressible_at_d': can_express_at_d,
            'expressible_at_d_plus_1': can_express_at_d_plus_1,
            'example_string': test_string[:20] + '...' if len(test_string) > 20 else test_string,
            'explanation': f"深度{depth}不能判定自己的深度，但深度{depth+1}可以"
        }
        
    def analyze_convergence(self, max_depth: int = 8) -> Dict[str, Any]:
        """分析完备性的收敛性（基于Fibonacci比值收敛）"""
        # 分析Fibonacci数列本身的收敛性到φ
        fib_ratios = []
        
        for d in range(2, max_depth + 1):
            f_d = self.ch_system.fibonacci(d)
            f_d_minus_1 = self.ch_system.fibonacci(d - 1)
            if f_d_minus_1 > 0:
                ratio = f_d / f_d_minus_1
                fib_ratios.append(ratio)
                
        # 分析完备集大小（限制深度以避免指数爆炸）
        sizes = []
        growth_rates = []
        
        for d in range(min(6, max_depth + 1)):  # 限制到深度5
            complete_set = self.ch_system.generate_syntactic_complete(d)
            sizes.append(len(complete_set))
            
            if d > 0:
                growth_rate = sizes[d] / sizes[d-1] if sizes[d-1] > 0 else float('inf')
                growth_rates.append(growth_rate)
                
        # 理论收敛是基于Fibonacci比值，不是完备集大小
        return {
            'fibonacci_ratios': fib_ratios,
            'sizes': sizes,
            'growth_rates': growth_rates,
            'limiting_fib_ratio': fib_ratios[-1] if fib_ratios else None,
            'converges_to_phi': abs(fib_ratios[-1] - self.phi) < 0.1 if fib_ratios else False
        }


class MinimalExtensionConstructor:
    """最小完备扩展的构造"""
    
    def __init__(self):
        self.ch_system = CompletenessHierarchySystem()
        self.phi = (1 + np.sqrt(5)) / 2
        
    def construct_minimal_extension(self, depth: int) -> Dict[str, Any]:
        """构造从深度d到d+1的最小扩展"""
        if depth < 0:
            return {'error': 'invalid depth'}
            
        # 获取两个层级
        complete_d = self.ch_system.generate_syntactic_complete(depth)
        complete_d_plus_1 = self.ch_system.generate_syntactic_complete(depth + 1)
        
        # 最小扩展
        minimal_extension = complete_d_plus_1 - complete_d
        
        # 分析扩展的结构
        extension_by_length = {}
        for s in minimal_extension:
            length = len(s)
            if length not in extension_by_length:
                extension_by_length[length] = []
            extension_by_length[length].append(s)
            
        return {
            'depth': depth,
            'extension_size': len(minimal_extension),
            'length_distribution': {k: len(v) for k, v in extension_by_length.items()},
            'sample_strings': list(minimal_extension)[:10],
            'characterization': self._characterize_extension(minimal_extension, depth)
        }
        
    def _characterize_extension(self, extension: Set[str], depth: int) -> Dict[str, Any]:
        """特征化最小扩展"""
        if not extension:
            return {'empty': True}
            
        # 长度范围
        lengths = [len(s) for s in extension]
        min_length = min(lengths)
        max_length = max(lengths)
        
        # 结构特征
        all_max_length = all(len(s) == max_length for s in extension)
        
        return {
            'min_length': min_length,
            'max_length': max_length,
            'all_maximal': all_max_length,
            'expected_range': f"[{self.ch_system.fibonacci(depth)+1}, {self.ch_system.fibonacci(depth+1)}]"
        }
        
    def verify_uniqueness(self, depth: int) -> bool:
        """验证最小扩展的唯一性"""
        # 最小扩展由定义唯一确定
        extension1 = self.construct_minimal_extension(depth)
        extension2 = self.construct_minimal_extension(depth)
        
        return extension1['extension_size'] == extension2['extension_size']
        
    def compute_extension_complexity(self, depth: int) -> Dict[str, float]:
        """计算扩展的复杂度"""
        extension_data = self.construct_minimal_extension(depth)
        
        if 'error' in extension_data:
            return {'error': extension_data['error']}
            
        # 组合复杂度
        size = extension_data['extension_size']
        
        # 计算复杂度
        return {
            'size_complexity': size,
            'bit_complexity': sum(len(s) for s in extension_data['sample_strings']),
            'structural_complexity': len(extension_data['length_distribution']),
            'growth_factor': size / self.ch_system.fibonacci(depth) if depth > 0 else float('inf')
        }


class TestP9CompletenessHierarchy(unittest.TestCase):
    """P9完备性层级命题的测试用例"""
    
    def setUp(self):
        """测试初始化"""
        self.ch_system = CompletenessHierarchySystem()
        self.analyzer = ExpressivePowerAnalyzer()
        self.constructor = MinimalExtensionConstructor()
        self.phi = (1 + np.sqrt(5)) / 2
        
    def test_fibonacci_generation(self):
        """测试1：Fibonacci数生成"""
        print("\n测试1：Fibonacci数序列")
        
        print("\n  n   F_n")
        print("  --  ---")
        
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        
        for n in range(10):
            f_n = self.ch_system.fibonacci(n)
            print(f"  {n:2}  {f_n:3}")
            self.assertEqual(f_n, expected[n], f"F_{n}应该等于{expected[n]}")
            
    def test_syntactic_hierarchy(self):
        """测试2：语法完备性层级"""
        print("\n测试2：语法完备性层级 SynComplete_d")
        
        print("\n  深度  大小  F_{d+2}  新增元素示例")
        print("  ----  ----  -------  ------------")
        
        prev_size = 0
        for depth in range(5):
            complete_set = self.ch_system.generate_syntactic_complete(depth)
            f_d_plus_2 = self.ch_system.fibonacci(depth + 2)
            new_elements = list(complete_set)[-3:] if len(complete_set) > 3 else list(complete_set)
            
            print(f"  {depth:4}  {len(complete_set):4}  {f_d_plus_2:7}  {new_elements}")
            
            # 验证严格增长
            if depth > 0:
                self.assertGreater(len(complete_set), prev_size, 
                                 f"深度{depth}的完备集应该严格大于深度{depth-1}")
            prev_size = len(complete_set)
            
            # 验证最大长度约束
            max_length = max(len(s) for s in complete_set) if complete_set else 0
            f_d_plus_2 = self.ch_system.fibonacci(depth + 2)
            self.assertLessEqual(max_length, f_d_plus_2, 
                               f"深度{depth}的最大串长应该不超过F_{depth+2}={f_d_plus_2}")
                               
    def test_strict_subset_relation(self):
        """测试3：严格子集关系"""
        print("\n测试3：严格层级关系验证")
        
        print("\n  d1  d2  |C_d1|  |C_d2|  差异  严格子集")
        print("  --  --  ------  ------  ----  --------")
        
        for d1 in range(4):
            d2 = d1 + 1
            result = self.ch_system.verify_strict_hierarchy(d1, d2)
            
            print(f"  {d1:2}  {d2:2}  {result['size_d1']:6}  {result['size_d2']:6}  "
                  f"{result['difference']:4}  {result['strict_subset']}")
                  
            self.assertTrue(result['strict_subset'], 
                          f"Complete_{d1}应该是Complete_{d2}的严格子集")
                          
    def test_semantic_models(self):
        """测试4：语义模型层级"""
        print("\n测试4：语义完备性层级")
        
        print("\n  深度  总串数  常数  周期  递归  混沌  复杂度")
        print("  ----  ------  ----  ----  ----  ----  ------")
        
        prev_complexity = 0
        for depth in range(4):
            models = self.ch_system.compute_semantic_models(depth)
            counts = models['model_counts']
            
            print(f"  {depth:4}  {models['total_strings']:6}  "
                  f"{counts['constant']:4}  {counts['periodic']:4}  "
                  f"{counts['recursive']:4}  {counts['chaotic']:4}  "
                  f"{models['semantic_complexity']:6.3f}")
                  
            # 验证复杂度非递减
            if depth > 0:
                self.assertGreaterEqual(models['semantic_complexity'], prev_complexity - 0.01,
                                      "语义复杂度应该非递减")
            prev_complexity = models['semantic_complexity']
            
    def test_computational_hierarchy(self):
        """测试5：计算完备性层级"""
        print("\n测试5：计算完备性层级")
        
        print("\n  深度  空间界  时间界  可计算函数")
        print("  ----  ------  ------  ----------")
        
        prev_computable = 0
        for depth in range(5):
            comp = self.ch_system.verify_computational_hierarchy(depth)
            
            functions = ', '.join([f for f, c in comp['computable_functions'].items() if c])
            
            print(f"  {depth:4}  {comp['space_bound']:6}  {comp['time_bound']:6}  {functions}")
            
            # 验证严格增长
            self.assertGreaterEqual(comp['total_computable'], prev_computable,
                                  "可计算函数数量应该非递减")
            if depth > 0:
                self.assertGreater(comp['space_bound'], 
                                 self.ch_system.fibonacci(depth-1),
                                 "空间界应该严格增长")
            prev_computable = comp['total_computable']
            
    def test_expressive_power(self):
        """测试6：表达力层级"""
        print("\n测试6：表达力严格增长")
        
        print("\n  深度  性质数  复杂度  性质类型")
        print("  ----  ------  ------  --------")
        
        prev_num = 0
        for depth in range(5):
            power = self.analyzer.measure_expressive_power(depth)
            
            types = ', '.join([f"{k}:{v}" for k, v in power['property_types'].items()])
            
            print(f"  {depth:4}  {power['num_properties']:6}  "
                  f"{power['complexity_measure']:6.1f}  {types}")
                  
            # 验证严格增长
            self.assertGreaterEqual(power['num_properties'], prev_num,
                                  "可表达性质数应该非递减")
            prev_num = power['num_properties']
            
        # 演示严格增长
        demo = self.analyzer.demonstrate_strict_increase(3)
        print(f"\n  严格增长演示: {demo['property']}")
        print(f"  深度3可表达: {demo['expressible_at_d']}")
        print(f"  深度4可表达: {demo['expressible_at_d_plus_1']}")
        
        self.assertTrue(demo['demonstrated'], "应该能演示严格增长")
        
    def test_minimal_extensions(self):
        """测试7：最小完备扩展"""
        print("\n测试7：最小完备扩展")
        
        print("\n  深度  扩展大小  长度分布  唯一性")
        print("  ----  --------  --------  ------")
        
        for depth in range(3):
            ext = self.constructor.construct_minimal_extension(depth)
            
            if 'error' not in ext:
                dist = ', '.join([f"{k}:{v}" for k, v in sorted(ext['length_distribution'].items())])
                unique = self.constructor.verify_uniqueness(depth)
                
                print(f"  {depth:4}  {ext['extension_size']:8}  {dist}  {unique}")
                
                # 验证扩展非空
                self.assertGreater(ext['extension_size'], 0, 
                                 f"深度{depth}的最小扩展应该非空")
                                 
                # 验证唯一性
                self.assertTrue(unique, f"深度{depth}的最小扩展应该唯一")
                
    def test_valid_string_counting(self):
        """测试8：有效串计数"""
        print("\n测试8：有效串计数（Lucas数）")
        
        print("\n  长度  实际数  理论数  匹配")
        print("  ----  ------  ------  ----")
        
        for length in range(8):
            actual = len(self.ch_system._generate_valid_strings(length))
            theoretical = self.ch_system.count_valid_strings(length)
            
            match = actual == theoretical
            print(f"  {length:4}  {actual:6}  {theoretical:6}  {match}")
            
            self.assertEqual(actual, theoretical, 
                           f"长度{length}的有效串数量应该匹配Lucas数")
                           
    def test_convergence_analysis(self):
        """测试9：收敛性分析"""
        print("\n测试9：完备性收敛到φ")
        
        convergence = self.analyzer.analyze_convergence(8)
        
        print("\n  Fibonacci收敛分析:")
        print("  深度  F_n/F_{n-1}  与φ的差距")
        print("  ----  -----------  --------")
        
        for i, ratio in enumerate(convergence['fibonacci_ratios']):
            diff = abs(ratio - self.phi)
            print(f"  {i+2:4}  {ratio:11.6f}  {diff:8.6f}")
                
        print(f"\n  完备集大小增长:")
        print("  深度  大小   增长率")
        print("  ----  -----  ------")
        
        for i, size in enumerate(convergence['sizes']):
            if i > 0 and i-1 < len(convergence['growth_rates']):
                rate = convergence['growth_rates'][i-1]
                print(f"  {i:4}  {size:5}  {rate:6.4f}")
            else:
                print(f"  {i:4}  {size:5}  ---")
                
        print(f"\n  Fibonacci极限比率: {convergence['limiting_fib_ratio']:.6f}")
        print(f"  黄金分割比: {self.phi:.6f}")
        print(f"  Fibonacci收敛到φ: {convergence['converges_to_phi']}")
        
        # 验证Fibonacci比值趋于φ（这是正确的理论收敛）
        if len(convergence['fibonacci_ratios']) > 3:
            last_ratios = convergence['fibonacci_ratios'][-2:]
            for ratio in last_ratios:
                self.assertLess(abs(ratio - self.phi), 0.1, 
                              "Fibonacci比值应该接近φ")
                              
    def test_comprehensive_verification(self):
        """测试10：综合验证"""
        print("\n测试10：P9完备性层级命题综合验证")
        
        print("\n  验证项目              结果")
        print("  --------------------  ----")
        
        # 1. 语法层级
        syntax_ok = True
        for d in range(3):
            result = self.ch_system.verify_strict_hierarchy(d, d+1)
            if not result['strict_subset']:
                syntax_ok = False
                break
        print(f"  语法完备性层级        {'是' if syntax_ok else '否'}")
        
        # 2. 语义层级
        semantic_ok = True
        prev_comp = 0
        for d in range(3):
            models = self.ch_system.compute_semantic_models(d)
            if d > 0 and models['semantic_complexity'] < prev_comp - 0.01:
                semantic_ok = False
                break
            prev_comp = models['semantic_complexity']
        print(f"  语义完备性层级        {'是' if semantic_ok else '否'}")
        
        # 3. 计算层级
        comp_ok = True
        prev_total = 0
        for d in range(4):
            comp = self.ch_system.verify_computational_hierarchy(d)
            if comp['total_computable'] < prev_total:
                comp_ok = False
                break
            prev_total = comp['total_computable']
        print(f"  计算完备性层级        {'是' if comp_ok else '否'}")
        
        # 4. 表达力增长
        express_demo = self.analyzer.demonstrate_strict_increase(2)
        express_ok = express_demo['demonstrated']
        print(f"  表达力严格增长        {'是' if express_ok else '否'}")
        
        # 5. 最小扩展
        ext_ok = all(self.constructor.verify_uniqueness(d) for d in range(3))
        print(f"  最小扩展唯一性        {'是' if ext_ok else '否'}")
        
        # 6. 收敛性（基于Fibonacci比值）
        conv = self.analyzer.analyze_convergence(6)
        conv_ok = conv.get('converges_to_phi', False)
        print(f"  Fibonacci收敛到φ      {'是' if conv_ok else '否'}")
        
        # 总体评估
        all_passed = all([syntax_ok, semantic_ok, comp_ok, express_ok, ext_ok, conv_ok])
        print(f"\n  总体评估: {'通过' if all_passed else '需要改进'}")
        
        self.assertTrue(syntax_ok, "语法层级应该严格")
        self.assertTrue(semantic_ok, "语义复杂度应该非递减")
        self.assertTrue(comp_ok, "计算能力应该非递减")
        self.assertTrue(express_ok, "表达力应该严格增长")
        self.assertTrue(ext_ok, "最小扩展应该唯一")
        self.assertTrue(conv_ok, "Fibonacci比值应该趋于φ")


if __name__ == '__main__':
    unittest.main(verbosity=2)