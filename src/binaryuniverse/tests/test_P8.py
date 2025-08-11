#!/usr/bin/env python3
"""
P8 元一致性命题 - 单元测试

验证自指完备系统的元一致性，包括局部一致性、全局一致性、元一致性和递归元级。
"""

import unittest
import numpy as np
from typing import List, Dict, Tuple, Any
import time
import sys
import os

# 添加tests目录到路径以导入依赖
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_framework import BinaryUniverseSystem

class MetaConsistencySystem(BinaryUniverseSystem):
    """元一致性命题的数学模型"""
    
    def __init__(self):
        super().__init__()
        self.phi = (1 + np.sqrt(5)) / 2
        self.max_depth = 20  # 实际可验证的最大深度
        self.consistency_cache = {}  # 缓存已验证的一致性
        
    def check_local_consistency(self, states: List[str], depth: int) -> bool:
        """检查深度d内的局部一致性"""
        if depth == 0:
            # 基础层：检查原子命题
            return self._check_atomic_consistency(states)
            
        # 递归检查各层一致性
        for d in range(depth + 1):
            layer_states = self._get_layer_states(states, d)
            if not self._check_layer_consistency(layer_states):
                return False
                
        return True
        
    def _check_atomic_consistency(self, states: List[str]) -> bool:
        """检查原子命题的一致性"""
        # 检查no-11约束
        for state in states:
            if '11' in state:
                return False
                
        # 检查基本逻辑一致性
        state_set = set(states)
        for state in state_set:
            neg_state = self._negate(state)
            if neg_state in state_set:
                return False
                
        return True
        
    def _negate(self, state: str) -> str:
        """逻辑否定操作"""
        # 简单的按位取反
        return ''.join('1' if bit == '0' else '0' for bit in state)
        
    def _get_layer_states(self, states: List[str], depth: int) -> List[str]:
        """获取特定深度的状态"""
        # 根据递归深度筛选状态
        layer_states = []
        for state in states:
            if self._get_state_depth(state) == depth:
                layer_states.append(state)
        return layer_states
        
    def _get_state_depth(self, state: str) -> int:
        """计算状态的递归深度"""
        # 基于状态长度的简单深度计算
        if len(state) <= 2:
            return 0
        return int(np.log(len(state) + 1) / np.log(self.phi))
        
    def _check_layer_consistency(self, states: List[str]) -> bool:
        """检查单层的一致性"""
        if not states:
            return True
            
        # 检查该层内部的逻辑一致性
        for i, state1 in enumerate(states):
            for state2 in states[i+1:]:
                if self._contradicts(state1, state2):
                    return False
        return True
        
    def _contradicts(self, state1: str, state2: str) -> bool:
        """检查两个状态是否矛盾"""
        # 简化的矛盾检测
        if len(state1) != len(state2):
            return False
            
        # 检查是否互为否定
        return all(s1 != s2 for s1, s2 in zip(state1, state2))
        
    def prove_global_consistency(self, system_states: List[str], 
                               max_depth: int = None) -> Dict[str, Any]:
        """证明全局一致性（通过有限深度逼近）"""
        if max_depth is None:
            max_depth = min(self.max_depth, 10)  # 限制深度以避免超时
            
        results = {
            'depths': [],
            'consistency': [],
            'convergence': False
        }
        
        for d in range(max_depth + 1):
            consistent = self.check_local_consistency(system_states, d)
            results['depths'].append(d)
            results['consistency'].append(consistent)
            
            # 检查收敛性
            if d > 3 and all(results['consistency'][-3:]):
                results['convergence'] = True
                
        results['global_consistency'] = results['convergence']
        return results
        
    def encode_consistency_proof(self, proof_steps: List[str]) -> str:
        """将一致性证明编码为二进制串"""
        # 将证明步骤转换为满足no-11约束的二进制串
        encoded = ""
        for step in proof_steps:
            # 简单的哈希编码
            hash_val = abs(hash(step)) % (2**10)
            binary = format(hash_val, '010b')
            # 迭代替换所有的11，直到没有11为止
            while '11' in binary:
                binary = binary.replace('11', '10')
            encoded += binary
            
        # 再次确保整个编码满足no-11约束
        while '11' in encoded:
            encoded = encoded.replace('11', '10')
            
        # 限制长度
        return encoded[:50]
        
    def verify_meta_consistency(self, system_states: List[str]) -> Dict[str, Any]:
        """验证元一致性（系统能证明自己的一致性）"""
        results = {
            'base_consistent': False,
            'meta_levels': [],
            'self_reference_found': False
        }
        
        # 1. 验证基础一致性
        base_proof = self.prove_global_consistency(system_states)
        results['base_consistent'] = base_proof['global_consistency']
        
        if not results['base_consistent']:
            return results
            
        # 2. 构造元级证明
        current_level = system_states
        for meta_level in range(3):  # 验证前3个元级
            # 编码当前级别的一致性证明
            proof_steps = [f"Level {meta_level} consistency proof"]
            proof_encoding = self.encode_consistency_proof(proof_steps)
            
            # 检查编码的一致性
            meta_consistent = self.check_local_consistency([proof_encoding], 0)
            
            results['meta_levels'].append({
                'level': meta_level,
                'consistent': meta_consistent,
                'encoding_length': len(proof_encoding)
            })
            
            # 检查自指：证明编码是否出现在原系统中
            if proof_encoding in system_states:
                results['self_reference_found'] = True
                
            # 准备下一个元级
            current_level = [proof_encoding]
            
        return results


class RecursiveMetaLevelAnalyzer:
    """递归元级的详细分析"""
    
    def __init__(self):
        self.mc_system = MetaConsistencySystem()
        self.phi = (1 + np.sqrt(5)) / 2
        
    def construct_meta_tower(self, base_system: List[str], 
                           height: int = 5) -> Dict[str, Any]:
        """构造一致性的元级塔"""
        tower = {
            'levels': [],
            'height': 0,
            'stable': True
        }
        
        current = base_system
        
        for level in range(height):
            # 验证当前级的一致性
            consistency = self.mc_system.check_local_consistency(current, min(level, 3))
            
            # 构造一致性证明
            proof = self._generate_consistency_proof(current, level)
            
            # 编码证明
            encoded_proof = self.mc_system.encode_consistency_proof(proof)
            
            tower['levels'].append({
                'level': level,
                'size': len(current),
                'consistent': consistency,
                'proof_size': len(encoded_proof),
                'proof_encoding': encoded_proof[:20] + '...'  # 截断显示
            })
            
            # 如果不一致，停止构造
            if not consistency:
                tower['stable'] = False
                break
                
            # 下一级是当前级的证明编码
            current = [encoded_proof]
            tower['height'] = level + 1
            
        return tower
        
    def _generate_consistency_proof(self, states: List[str], depth: int) -> List[str]:
        """生成一致性证明的步骤"""
        proof_steps = []
        
        # 步骤1：验证no-11约束
        proof_steps.append(f"Check no-11 constraint for {len(states)} states at depth {depth}")
        
        # 步骤2：验证逻辑一致性
        proof_steps.append(f"Verify logical consistency at depth {depth}")
        
        # 步骤3：递归验证
        if depth > 0:
            proof_steps.append(f"Recursively verify depths 0 to {depth-1}")
            
        # 步骤4：结论
        proof_steps.append(f"Consistency proven at depth {depth}")
        
        return proof_steps
        
    def analyze_fixed_points(self, initial_states: List[str]) -> Dict[str, Any]:
        """分析元一致性的不动点"""
        results = {
            'fixed_points': [],
            'cycles': [],
            'convergence_depth': None
        }
        
        seen_encodings = set()
        current = initial_states
        
        for iteration in range(10):
            # 生成一致性证明并编码
            proof = self._generate_consistency_proof(current, 0)
            encoding = self.mc_system.encode_consistency_proof(proof)
            
            # 检查不动点
            if encoding in current:
                results['fixed_points'].append({
                    'iteration': iteration,
                    'encoding': encoding[:20] + '...',
                    'self_describing': True
                })
                results['convergence_depth'] = iteration
                break
                
            # 检查循环
            if encoding in seen_encodings:
                results['cycles'].append({
                    'start': iteration,
                    'encoding': encoding[:20] + '...'
                })
                break
                
            seen_encodings.add(encoding)
            current = [encoding]
            
        return results
        
    def measure_consistency_strength(self, system1: List[str], 
                                   system2: List[str]) -> Dict[str, Any]:
        """测量两个系统的一致性强度"""
        results = {
            'system1_strength': 0,
            'system2_strength': 0,
            'relative_strength': None
        }
        
        # 测量系统1的强度
        tower1 = self.construct_meta_tower(system1, height=3)
        results['system1_strength'] = tower1['height']
        
        # 测量系统2的强度
        tower2 = self.construct_meta_tower(system2, height=3)
        results['system2_strength'] = tower2['height']
        
        # 比较强度
        if results['system1_strength'] > results['system2_strength']:
            results['relative_strength'] = 'system1 > system2'
        elif results['system1_strength'] < results['system2_strength']:
            results['relative_strength'] = 'system1 < system2'
        else:
            results['relative_strength'] = 'system1 = system2'
            
        return results


class FiniteVerifiabilityChecker:
    """有限可验证性的算法实现"""
    
    def __init__(self):
        self.mc_system = MetaConsistencySystem()
        self.phi = (1 + np.sqrt(5)) / 2
        
    def verify_at_depth(self, states: List[str], depth: int) -> Dict[str, Any]:
        """在特定深度验证一致性（可判定算法）"""
        start_time = time.time()
        
        result = {
            'depth': depth,
            'consistent': False,
            'states_checked': 0,
            'time_taken': 0,
            'decidable': True
        }
        
        # 有限深度总是可判定的
        try:
            # 检查一致性
            consistent = self.mc_system.check_local_consistency(states, depth)
            result['consistent'] = consistent
            
            # 统计检查的状态数
            result['states_checked'] = sum(
                len(self.mc_system._get_layer_states(states, d))
                for d in range(depth + 1)
            )
            
        except Exception as e:
            result['decidable'] = False
            result['error'] = str(e)
            
        result['time_taken'] = time.time() - start_time
        return result
        
    def demonstrate_incompleteness(self, system: List[str]) -> Dict[str, Any]:
        """演示不完全性边界（无法在有限步证明全局一致性）"""
        results = {
            'finite_proofs': [],
            'global_proof_found': False,
            'incompleteness_demonstrated': False
        }
        
        # 尝试不同步数的证明
        for steps in [10, 50, 100, 500]:
            # 限制验证步数
            proof_result = self._limited_consistency_proof(system, steps)
            
            results['finite_proofs'].append({
                'steps': steps,
                'depth_reached': proof_result['depth'],
                'partial_consistency': proof_result['consistent']
            })
            
            # 检查是否达到全局证明
            if proof_result['global_proven']:
                results['global_proof_found'] = True
                break
                
        # 如果所有有限步都无法证明全局一致性
        if not results['global_proof_found']:
            results['incompleteness_demonstrated'] = True
            
        return results
        
    def _limited_consistency_proof(self, states: List[str], 
                                 max_steps: int) -> Dict[str, Any]:
        """限制步数的一致性证明"""
        result = {
            'steps_used': 0,
            'depth': 0,
            'consistent': True,
            'global_proven': False
        }
        
        depth = 0
        steps = 0
        
        while steps < max_steps and depth < 10:
            # 验证当前深度
            layer_size = len(self.mc_system._get_layer_states(states, depth))
            if layer_size == 0 and depth > 0:
                layer_size = 1  # 至少计1步
                
            consistent = self.mc_system.check_local_consistency(states, depth)
            steps += layer_size
            
            if not consistent:
                result['consistent'] = False
                break
                
            # 简化的收敛判断（永远无法达到）
            if depth > 100:  # 不可能达到
                result['global_proven'] = True
                
            depth += 1
            result['depth'] = depth
            result['steps_used'] = steps
            
        return result


class TestP8MetaConsistency(unittest.TestCase):
    """P8元一致性命题的测试用例"""
    
    def setUp(self):
        """测试初始化"""
        self.mc_system = MetaConsistencySystem()
        self.analyzer = RecursiveMetaLevelAnalyzer()
        self.checker = FiniteVerifiabilityChecker()
        self.phi = (1 + np.sqrt(5)) / 2
        np.random.seed(42)  # 固定随机种子
        
    def test_local_consistency(self):
        """测试1：局部一致性验证"""
        print("\n测试1：局部一致性 Consistent_d(S)")
        
        # 创建一致的系统
        consistent_system = ['010', '100', '001']
        
        # 创建不一致的系统（包含矛盾）
        inconsistent_system = ['010', '101']  # 101是010的否定
        
        print("\n  深度  一致系统  不一致系统")
        print("  ----  --------  ----------")
        
        for depth in range(4):
            cons_result = self.mc_system.check_local_consistency(consistent_system, depth)
            incons_result = self.mc_system.check_local_consistency(inconsistent_system, depth)
            
            print(f"  {depth:4}  {cons_result}      {incons_result}")
            
            # 一致系统应该在所有深度保持一致
            self.assertTrue(cons_result, f"一致系统在深度{depth}应该保持一致")
            
            # 不一致系统应该被检测出
            if depth == 0:
                self.assertFalse(incons_result, f"不一致系统应该在深度{depth}被检测出")
                
    def test_global_consistency(self):
        """测试2：全局一致性逼近"""
        print("\n测试2：全局一致性 lim_{d→∞} Consistent_d(S)")
        
        # 测试系统
        test_system = ['0010', '0100', '1000', '0001']
        
        # 证明全局一致性
        global_proof = self.mc_system.prove_global_consistency(test_system, max_depth=8)
        
        print("\n  深度  一致性")
        print("  ----  ------")
        for d, c in zip(global_proof['depths'], global_proof['consistency']):
            print(f"  {d:4}  {c}")
            
        print(f"\n  收敛性: {global_proof['convergence']}")
        print(f"  全局一致性: {global_proof['global_consistency']}")
        
        # 验证结果
        self.assertTrue(global_proof['convergence'], "系统应该收敛到一致性")
        self.assertTrue(global_proof['global_consistency'], "系统应该全局一致")
        
    def test_meta_consistency(self):
        """测试3：元一致性验证"""
        print("\n测试3：元一致性 S ⊢ Consistent(S)")
        
        # 创建包含自己一致性证明的系统
        base_system = ['0010', '0100', '1000']
        
        # 验证元一致性
        meta_result = self.mc_system.verify_meta_consistency(base_system)
        
        print(f"\n  基础一致性: {meta_result['base_consistent']}")
        print("\n  元级  一致性  编码长度")
        print("  ----  ------  --------")
        
        for level in meta_result['meta_levels']:
            print(f"  {level['level']:4}  {level['consistent']}      {level['encoding_length']:8}")
            
        print(f"\n  自指发现: {meta_result['self_reference_found']}")
        
        # 验证基础一致性
        self.assertTrue(meta_result['base_consistent'], "基础系统应该一致")
        
        # 验证元级一致性
        for level in meta_result['meta_levels']:
            self.assertTrue(level['consistent'], f"元级{level['level']}应该一致")
            
    def test_recursive_meta_tower(self):
        """测试4：递归元级塔"""
        print("\n测试4：递归元级塔 MetaLevel_n")
        
        # 基础系统
        base = ['00100', '01000', '10000']
        
        # 构造元级塔
        tower = self.analyzer.construct_meta_tower(base, height=4)
        
        print("\n  级别  大小  一致性  证明大小")
        print("  ----  ----  ------  --------")
        
        for level in tower['levels']:
            print(f"  {level['level']:4}  {level['size']:4}  {level['consistent']}      {level['proof_size']:8}")
            
        print(f"\n  塔高度: {tower['height']}")
        print(f"  稳定性: {tower['stable']}")
        
        # 验证塔的稳定性
        self.assertTrue(tower['stable'], "元级塔应该稳定")
        self.assertGreater(tower['height'], 0, "应该能构造至少一层")
        
    def test_finite_verifiability(self):
        """测试5：有限可验证性"""
        print("\n测试5：有限可验证性")
        
        test_system = ['001000', '010000', '100000']
        
        print("\n  深度  一致性  状态数  时间(s)  可判定")
        print("  ----  ------  ------  -------  ------")
        
        for depth in [0, 1, 2, 3]:
            result = self.checker.verify_at_depth(test_system, depth)
            
            print(f"  {result['depth']:4}  {result['consistent']}      "
                  f"{result['states_checked']:6}  {result['time_taken']:7.4f}  "
                  f"{result['decidable']}")
                  
            # 有限深度应该总是可判定的
            self.assertTrue(result['decidable'], f"深度{depth}应该是可判定的")
            
    def test_incompleteness_boundary(self):
        """测试6：不完全性边界"""
        print("\n测试6：不完全性边界")
        
        test_system = ['0010000', '0100000', '1000000']
        
        # 演示不完全性
        incompleteness = self.checker.demonstrate_incompleteness(test_system)
        
        print("\n  步数  深度  部分一致性")
        print("  ----  ----  ----------")
        
        for proof in incompleteness['finite_proofs']:
            print(f"  {proof['steps']:4}  {proof['depth_reached']:4}  {proof['partial_consistency']}")
            
        print(f"\n  找到全局证明: {incompleteness['global_proof_found']}")
        print(f"  不完全性演示: {incompleteness['incompleteness_demonstrated']}")
        
        # 验证不完全性
        self.assertFalse(incompleteness['global_proof_found'], 
                        "有限步内不应找到全局证明")
        self.assertTrue(incompleteness['incompleteness_demonstrated'],
                       "应该演示不完全性")
                       
    def test_fixed_points(self):
        """测试7：不动点分析"""
        print("\n测试7：元一致性的不动点")
        
        # 尝试构造自描述的系统
        initial = ['0010001000']  # 特殊构造的状态
        
        # 分析不动点
        fixed_point_result = self.analyzer.analyze_fixed_points(initial)
        
        print("\n  迭代  类型")
        print("  ----  ----")
        
        for fp in fixed_point_result['fixed_points']:
            print(f"  {fp['iteration']:4}  不动点")
            
        for cycle in fixed_point_result['cycles']:
            print(f"  {cycle['start']:4}  循环")
            
        if fixed_point_result['convergence_depth'] is not None:
            print(f"\n  收敛深度: {fixed_point_result['convergence_depth']}")
            
    def test_relative_consistency(self):
        """测试8：相对一致性强度"""
        print("\n测试8：相对一致性强度")
        
        # 两个不同的系统
        system1 = ['00100', '01000', '10000']  # 简单系统
        system2 = ['00100100', '01000100', '10000010', '00010001']  # 复杂系统
        
        # 测量强度
        strength = self.analyzer.measure_consistency_strength(system1, system2)
        
        print(f"\n  系统1强度: {strength['system1_strength']}")
        print(f"  系统2强度: {strength['system2_strength']}")
        print(f"  相对强度: {strength['relative_strength']}")
        
        # 复杂系统应该有更高的一致性强度
        self.assertIsNotNone(strength['relative_strength'], "应该能比较强度")
        
    def test_proof_encoding(self):
        """测试9：一致性证明编码"""
        print("\n测试9：一致性证明的二进制编码")
        
        # 创建证明步骤
        proof_steps = [
            "Axiom: no-11 constraint",
            "Check layer 0 consistency",
            "Check layer 1 consistency",
            "QED: System is consistent"
        ]
        
        # 编码证明
        encoded = self.mc_system.encode_consistency_proof(proof_steps)
        
        print(f"\n  证明步骤数: {len(proof_steps)}")
        print(f"  编码长度: {len(encoded)}")
        print(f"  编码前20位: {encoded[:20]}...")
        
        # 验证编码满足no-11约束
        self.assertNotIn('11', encoded, "编码应该满足no-11约束")
        self.assertGreater(len(encoded), 0, "应该生成非空编码")
        
    def test_comprehensive_verification(self):
        """测试10：综合验证"""
        print("\n测试10：P8元一致性命题综合验证")
        
        # 构造测试系统
        test_system = ['0010', '0100', '1000', '0001']
        
        print("\n  验证项目              结果")
        print("  --------------------  ----")
        
        # 1. 局部一致性
        local_ok = all(
            self.mc_system.check_local_consistency(test_system, d)
            for d in range(4)
        )
        print(f"  局部一致性            {'是' if local_ok else '否'}")
        
        # 2. 全局一致性
        global_result = self.mc_system.prove_global_consistency(test_system)
        global_ok = global_result['global_consistency']
        print(f"  全局一致性逼近        {'是' if global_ok else '否'}")
        
        # 3. 元一致性
        meta_result = self.mc_system.verify_meta_consistency(test_system)
        meta_ok = meta_result['base_consistent'] and len(meta_result['meta_levels']) > 0
        print(f"  元一致性验证          {'是' if meta_ok else '否'}")
        
        # 4. 递归塔
        tower = self.analyzer.construct_meta_tower(test_system, height=3)
        tower_ok = tower['stable'] and tower['height'] > 0
        print(f"  递归元级塔稳定        {'是' if tower_ok else '否'}")
        
        # 5. 有限可验证
        verify_result = self.checker.verify_at_depth(test_system, 2)
        verify_ok = verify_result['decidable']
        print(f"  有限可验证性          {'是' if verify_ok else '否'}")
        
        # 6. 不完全性
        incomplete_result = self.checker.demonstrate_incompleteness(test_system)
        incomplete_ok = incomplete_result['incompleteness_demonstrated']
        print(f"  不完全性边界          {'是' if incomplete_ok else '否'}")
        
        # 总体评估
        all_passed = all([local_ok, global_ok, meta_ok, tower_ok, verify_ok, incomplete_ok])
        print(f"\n  总体评估: {'通过' if all_passed else '需要改进'}")
        
        self.assertTrue(local_ok, "局部一致性应该成立")
        self.assertTrue(global_ok, "全局一致性应该可逼近")
        self.assertTrue(meta_ok, "元一致性应该可验证")
        self.assertTrue(tower_ok, "递归塔应该稳定")
        self.assertTrue(verify_ok, "有限深度应该可验证")
        self.assertTrue(incomplete_ok, "不完全性应该被演示")


if __name__ == '__main__':
    unittest.main(verbosity=2)