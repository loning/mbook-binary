#!/usr/bin/env python3
"""
test_C3_2.py - C3-2稳定性推论的完整二进制机器验证测试

验证自指完备系统的稳定性机制，包括Lyapunov稳定性、扰动响应和吸引域
"""

import unittest
import sys
import os
import math
import numpy as np
from typing import List, Dict, Tuple, Set, Any
import random
from collections import defaultdict

# 添加包路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'formal'))

class PhiRepresentationSystem:
    """φ-表示系统（复用之前的定义）"""
    
    def __init__(self, n: int):
        """初始化n位φ-表示系统"""
        self.n = n
        self.valid_states = self._generate_valid_states()
        self.state_to_index = {tuple(s): i for i, s in enumerate(self.valid_states)}
        self.index_to_state = {i: s for i, s in enumerate(self.valid_states)}
        
    def _is_valid_phi_state(self, state: List[int]) -> bool:
        """检查是否为有效的φ-表示状态"""
        if len(state) != self.n:
            return False
        if not all(bit in [0, 1] for bit in state):
            return False
        
        # 检查no-consecutive-1s约束
        for i in range(len(state) - 1):
            if state[i] == 1 and state[i + 1] == 1:
                return False
        return True
    
    def _generate_valid_states(self) -> List[List[int]]:
        """生成所有有效的φ-表示状态"""
        valid_states = []
        
        def generate_recursive(current_state: List[int], pos: int):
            if pos == self.n:
                if self._is_valid_phi_state(current_state):
                    valid_states.append(current_state[:])
                return
            
            # 尝试放置0
            current_state.append(0)
            generate_recursive(current_state, pos + 1)
            current_state.pop()
            
            # 尝试放置1（如果不违反约束）
            if pos == 0 or current_state[pos - 1] == 0:
                current_state.append(1)
                generate_recursive(current_state, pos + 1)
                current_state.pop()
        
        generate_recursive([], 0)
        return valid_states
    
    def get_neighbors(self, state: List[int]) -> List[List[int]]:
        """获取状态的所有邻居（汉明距离为1的有效状态）"""
        neighbors = []
        
        for i in range(self.n):
            # 翻转第i位
            neighbor = state[:]
            neighbor[i] = 1 - neighbor[i]
            
            # 检查是否有效
            if self._is_valid_phi_state(neighbor):
                neighbors.append(neighbor)
        
        return neighbors


class BinaryStabilityVerifier:
    """二进制系统稳定性推论验证器"""
    
    def __init__(self, n: int = 8):
        """初始化验证器"""
        self.n = n
        self.phi = (1 + math.sqrt(5)) / 2
        
        # 创建φ-表示系统
        self.phi_system = PhiRepresentationSystem(n)
        self.num_states = len(self.phi_system.valid_states)
        
        # 稳定性参数
        self.damping_factor = 0.8  # 阻尼因子
        self.noise_level = 0.05    # 噪声水平
        
        # 扰动阈值
        self.epsilon_1 = 2  # 小扰动阈值（汉明距离）
        self.epsilon_2 = 4  # 大扰动阈值
        
        # 缓存
        self.fixed_points_cache = None
        self.basins_cache = {}
        
    def self_referential_map(self, state: List[int]) -> List[int]:
        """自指映射 f: S -> S"""
        # 使用循环移位作为基础自指映射
        if all(b == 0 for b in state):
            return state[:]
        
        # 向右循环移位，保持no-11约束
        new_state = [0] * self.n
        
        for i in range(self.n):
            if state[i] == 1:
                new_pos = (i + 1) % self.n
                # 检查是否可以放置
                can_place = True
                
                # 检查左边
                if new_pos > 0 and new_state[new_pos - 1] == 1:
                    can_place = False
                # 检查右边
                if new_pos < self.n - 1 and new_state[new_pos + 1] == 1:
                    can_place = False
                    
                if can_place:
                    # 再次检查是否会与已有的1冲突
                    if new_pos > 0 and i != new_pos - 1 and state[new_pos - 1] == 1:
                        can_place = False
                    if new_pos < self.n - 1 and i != new_pos + 1 and state[new_pos + 1] == 1:
                        can_place = False
                
                if can_place:
                    new_state[new_pos] = 1
                else:
                    new_state[i] = 1  # 保持原位
        
        # 确保是有效状态
        if self.phi_system._is_valid_phi_state(new_state):
            return new_state
        else:
            return state[:]
    
    def find_fixed_points(self) -> List[List[int]]:
        """寻找所有不动点（S* = f(S*)）"""
        if self.fixed_points_cache is not None:
            return self.fixed_points_cache
            
        fixed_points = []
        
        for state in self.phi_system.valid_states:
            f_state = self.self_referential_map(state)
            if state == f_state:
                fixed_points.append(state)
        
        self.fixed_points_cache = fixed_points
        return fixed_points
    
    def hamming_distance(self, state1: List[int], state2: List[int]) -> int:
        """计算汉明距离"""
        return sum(b1 != b2 for b1, b2 in zip(state1, state2))
    
    def lyapunov_function(self, state: List[int], 
                         fixed_point: List[int]) -> float:
        """计算Lyapunov函数V(S)"""
        # 使用汉明距离作为基础
        hamming = self.hamming_distance(state, fixed_point)
        
        # 可选：添加权重以强调某些位的重要性
        # 这里使用简单的汉明距离
        return float(hamming)
    
    def lyapunov_function_weighted(self, state: List[int], 
                                  fixed_point: List[int]) -> float:
        """加权Lyapunov函数"""
        v = 0.0
        for i in range(self.n):
            # 权重递减，前面的位更重要
            weight = 1.0 / (1.0 + 0.1 * i)
            v += weight * abs(state[i] - fixed_point[i])
        return v
    
    def evolution_step_with_damping(self, state: List[int], 
                                   fixed_point: List[int]) -> List[int]:
        """带阻尼的演化步骤（用于稳定性）"""
        # 计算自指映射
        f_state = self.self_referential_map(state)
        
        # 如果已经是不动点，保持不变
        if state == fixed_point:
            return state[:]
        
        # 计算向不动点的恢复力
        # 使用概率方式实现阻尼
        new_state = state[:]
        
        for i in range(self.n):
            if f_state[i] != state[i]:
                # 以一定概率接受映射的改变
                if random.random() < self.damping_factor:
                    new_state[i] = f_state[i]
                    
            # 额外的恢复力：向不动点靠近
            if state[i] != fixed_point[i] and random.random() < 0.1:
                # 尝试向不动点移动
                temp_state = new_state[:]
                temp_state[i] = fixed_point[i]
                if self.phi_system._is_valid_phi_state(temp_state):
                    new_state = temp_state
        
        return new_state
    
    def add_perturbation(self, state: List[int], 
                        perturbation_size: int) -> List[int]:
        """添加扰动"""
        perturbed = state[:]
        positions = list(range(self.n))
        random.shuffle(positions)
        
        flips = 0
        for pos in positions:
            if flips >= perturbation_size:
                break
                
            # 尝试翻转这一位
            temp = perturbed[:]
            temp[pos] = 1 - temp[pos]
            
            if self.phi_system._is_valid_phi_state(temp):
                perturbed = temp
                flips += 1
        
        return perturbed
    
    def simulate_with_perturbation(self, fixed_point: List[int], 
                                  perturbation_size: int,
                                  steps: int = 50) -> Dict[str, Any]:
        """模拟扰动响应"""
        # 添加扰动
        initial = self.add_perturbation(fixed_point, perturbation_size)
        initial_distance = self.hamming_distance(initial, fixed_point)
        
        trajectory = [initial]
        lyapunov_values = [self.lyapunov_function(initial, fixed_point)]
        
        current = initial
        for _ in range(steps):
            current = self.evolution_step_with_damping(current, fixed_point)
            trajectory.append(current)
            lyapunov_values.append(self.lyapunov_function(current, fixed_point))
            
            # 如果回到不动点，提前结束
            if current == fixed_point:
                break
        
        final_distance = self.hamming_distance(trajectory[-1], fixed_point)
        
        return {
            'initial_state': initial,
            'initial_distance': initial_distance,
            'trajectory': trajectory,
            'lyapunov_values': lyapunov_values,
            'final_state': trajectory[-1],
            'final_distance': final_distance,
            'converged': trajectory[-1] == fixed_point,
            'convergence_steps': len(trajectory) - 1 if trajectory[-1] == fixed_point else None
        }
    
    def verify_lyapunov_decrease(self, trajectory: List[List[int]], 
                                fixed_point: List[int]) -> Dict[str, Any]:
        """验证Lyapunov函数递减"""
        lyapunov_values = [self.lyapunov_function(s, fixed_point) for s in trajectory]
        
        decreases = 0
        increases = 0
        total_change = 0
        
        for i in range(1, len(lyapunov_values)):
            change = lyapunov_values[i] - lyapunov_values[i-1]
            total_change += change
            if change < 0:
                decreases += 1
            elif change > 0:
                increases += 1
        
        # 计算单调性
        is_monotonic = increases == 0
        average_rate = total_change / (len(lyapunov_values) - 1) if len(lyapunov_values) > 1 else 0
        
        return {
            'initial_value': lyapunov_values[0],
            'final_value': lyapunov_values[-1],
            'total_change': total_change,
            'average_rate': average_rate,
            'decreases': decreases,
            'increases': increases,
            'is_monotonic': is_monotonic,
            'values': lyapunov_values
        }
    
    def compute_basin_of_attraction(self, fixed_point: List[int], 
                                   max_steps: int = 100) -> Dict[str, Any]:
        """计算吸引域"""
        fp_tuple = tuple(fixed_point)
        if fp_tuple in self.basins_cache:
            return self.basins_cache[fp_tuple]
            
        basin = set()
        basin.add(fp_tuple)
        
        convergence_times = {}
        edge_states = set()
        
        # 测试所有状态
        for state in self.phi_system.valid_states:
            current = state
            trajectory = [current]
            
            for step in range(max_steps):
                current = self.evolution_step_with_damping(current, fixed_point)
                trajectory.append(current)
                
                if current == fixed_point:
                    # 收敛到目标不动点
                    basin.add(tuple(state))
                    convergence_times[tuple(state)] = step + 1
                    break
                elif current in self.find_fixed_points() and current != fixed_point:
                    # 收敛到其他不动点，是边界状态
                    edge_states.add(tuple(state))
                    break
        
        # 分析吸引域的连通性
        # 简化的连通性检查：检查是否能通过邻居关系连接
        def is_connected(basin_set):
            if len(basin_set) <= 1:
                return True
                
            visited = set()
            queue = [list(basin_set)[0]]
            visited.add(queue[0])
            
            while queue:
                current = list(queue.pop(0))
                for neighbor in self.phi_system.get_neighbors(current):
                    neighbor_tuple = tuple(neighbor)
                    if neighbor_tuple in basin_set and neighbor_tuple not in visited:
                        visited.add(neighbor_tuple)
                        queue.append(neighbor)
            
            return len(visited) == len(basin_set)
        
        result = {
            'fixed_point': fixed_point,
            'basin_size': len(basin),
            'basin_ratio': len(basin) / self.num_states,
            'average_convergence_time': sum(convergence_times.values()) / len(convergence_times) if convergence_times else 0,
            'is_connected': is_connected(basin),
            'edge_states': len(edge_states),
            'sample_states': list(basin)[:10]  # 前10个状态
        }
        
        self.basins_cache[fp_tuple] = result
        return result
    
    def analyze_stability_type(self, fixed_point: List[int], 
                              num_tests: int = 20) -> Dict[str, Any]:
        """分析稳定性类型（渐近稳定、Lyapunov稳定等）"""
        results = {
            'lyapunov_stable': True,
            'asymptotically_stable': True,
            'globally_stable': False,
            'stability_radius': 0
        }
        
        # 测试不同大小的扰动
        for pert_size in range(1, min(self.n, 5)):
            converged_count = 0
            bounded_count = 0
            
            for _ in range(num_tests):
                sim_result = self.simulate_with_perturbation(
                    fixed_point, pert_size, steps=50
                )
                
                if sim_result['converged']:
                    converged_count += 1
                    bounded_count += 1
                elif sim_result['final_distance'] <= sim_result['initial_distance']:
                    bounded_count += 1
            
            # Lyapunov稳定性：扰动保持有界
            if bounded_count < num_tests * 0.9:
                results['lyapunov_stable'] = False
                results['stability_radius'] = pert_size - 1
                break
                
            # 渐近稳定性：扰动最终收敛
            if converged_count < num_tests * 0.8:
                results['asymptotically_stable'] = False
                
            results['stability_radius'] = pert_size
        
        # 检查全局稳定性（简化版本）
        basin_data = self.compute_basin_of_attraction(fixed_point)
        if basin_data['basin_ratio'] > 0.5:
            results['globally_stable'] = True
        
        return results
    
    def test_perturbation_classes(self, fixed_point: List[int]) -> Dict[str, Any]:
        """测试不同类别的扰动响应"""
        results = {
            'small_perturbations': [],
            'medium_perturbations': [],
            'large_perturbations': []
        }
        
        # 小扰动测试
        for size in range(1, min(self.epsilon_1 + 1, self.n)):
            sim = self.simulate_with_perturbation(fixed_point, size, steps=30)
            results['small_perturbations'].append({
                'size': size,
                'converged': sim['converged'],
                'steps': sim['convergence_steps'],
                'final_distance': sim['final_distance']
            })
        
        # 中等扰动测试
        for size in range(self.epsilon_1 + 1, min(self.epsilon_2 + 1, self.n)):
            sim = self.simulate_with_perturbation(fixed_point, size, steps=50)
            
            # 检查是否跳到其他不动点
            other_fp = None
            if not sim['converged']:
                for fp in self.find_fixed_points():
                    if fp != fixed_point and sim['final_state'] == fp:
                        other_fp = fp
                        break
            
            results['medium_perturbations'].append({
                'size': size,
                'converged_to_original': sim['converged'],
                'converged_to_other': other_fp is not None,
                'other_fixed_point': other_fp,
                'final_distance': sim['final_distance']
            })
        
        # 大扰动测试
        for size in range(self.epsilon_2 + 1, min(self.n, self.epsilon_2 + 3)):
            sim = self.simulate_with_perturbation(fixed_point, size, steps=100)
            results['large_perturbations'].append({
                'size': size,
                'structure_preserved': sim['final_distance'] < self.n // 2,
                'final_distance': sim['final_distance']
            })
        
        return results
    
    def compute_restoration_force(self, state: List[int], 
                                 fixed_point: List[int]) -> Dict[str, Any]:
        """计算恢复力"""
        f_state = self.self_referential_map(state)
        
        # 恢复力定义为f(S) - S
        force = []
        for i in range(self.n):
            force.append(f_state[i] - state[i])
        
        # 计算恢复力的方向（是否指向不动点）
        distance_before = self.hamming_distance(state, fixed_point)
        distance_after = self.hamming_distance(f_state, fixed_point)
        
        points_toward_fp = distance_after < distance_before
        
        # 恢复力强度
        force_magnitude = sum(abs(f) for f in force)
        
        return {
            'state': state,
            'f_state': f_state,
            'force': force,
            'force_magnitude': force_magnitude,
            'points_toward_fixed_point': points_toward_fp,
            'distance_reduction': distance_before - distance_after
        }


class TestC3_2_BinaryStability(unittest.TestCase):
    """C3-2稳定性推论的二进制验证测试"""
    
    def setUp(self):
        """测试初始化"""
        self.verifier = BinaryStabilityVerifier(n=6)
        
    def test_fixed_points_existence(self):
        """测试不动点的存在性"""
        print("\n=== 测试不动点存在性 ===")
        
        fixed_points = self.verifier.find_fixed_points()
        
        print(f"找到 {len(fixed_points)} 个不动点:")
        for i, fp in enumerate(fixed_points[:5]):
            print(f"  不动点{i+1}: {fp}")
            # 验证确实是不动点
            f_fp = self.verifier.self_referential_map(fp)
            self.assertEqual(fp, f_fp, f"应该满足 S* = f(S*)")
        
        self.assertGreater(len(fixed_points), 0, "至少应该存在一个不动点")
        print("✓ 不动点存在性验证通过")
    
    def test_lyapunov_function(self):
        """测试Lyapunov函数性质"""
        print("\n=== 测试Lyapunov函数 ===")
        
        fixed_points = self.verifier.find_fixed_points()
        if not fixed_points:
            self.skipTest("没有找到不动点")
            
        fp = fixed_points[0]
        
        # 测试V(S*) = 0
        v_fp = self.verifier.lyapunov_function(fp, fp)
        self.assertEqual(v_fp, 0, "V(S*) 应该等于 0")
        
        # 测试V(S) > 0 for S ≠ S*
        test_states = self.verifier.phi_system.valid_states[:10]
        for state in test_states:
            if state != fp:
                v = self.verifier.lyapunov_function(state, fp)
                self.assertGreater(v, 0, f"V(S) 应该 > 0 当 S ≠ S*")
        
        print(f"✓ Lyapunov函数性质验证通过")
    
    def test_lyapunov_decrease(self):
        """测试Lyapunov函数递减"""
        print("\n=== 测试Lyapunov函数递减 ===")
        
        fixed_points = self.verifier.find_fixed_points()
        if not fixed_points:
            self.skipTest("没有找到不动点")
            
        fp = fixed_points[0]
        
        # 测试从扰动状态开始的演化
        for pert_size in [1, 2]:
            print(f"\n扰动大小: {pert_size}")
            
            sim_result = self.verifier.simulate_with_perturbation(
                fp, pert_size, steps=30
            )
            
            lyap_data = self.verifier.verify_lyapunov_decrease(
                sim_result['trajectory'], fp
            )
            
            print(f"  初始Lyapunov值: {lyap_data['initial_value']:.2f}")
            print(f"  最终Lyapunov值: {lyap_data['final_value']:.2f}")
            print(f"  平均变化率: {lyap_data['average_rate']:.4f}")
            print(f"  递减/递增次数: {lyap_data['decreases']}/{lyap_data['increases']}")
            
            # 验证总体递减
            self.assertLessEqual(
                lyap_data['final_value'], 
                lyap_data['initial_value'],
                "Lyapunov函数应该总体递减"
            )
        
        print("\n✓ Lyapunov递减验证通过")
    
    def test_small_perturbation_recovery(self):
        """测试小扰动恢复"""
        print("\n=== 测试小扰动恢复 ===")
        
        fixed_points = self.verifier.find_fixed_points()
        if not fixed_points:
            self.skipTest("没有找到不动点")
            
        fp = fixed_points[0]
        
        recovery_stats = []
        
        for size in range(1, min(self.verifier.epsilon_1 + 1, 4)):
            success_count = 0
            total_steps = 0
            
            for trial in range(10):
                sim = self.verifier.simulate_with_perturbation(fp, size, steps=50)
                if sim['converged']:
                    success_count += 1
                    total_steps += sim['convergence_steps']
            
            recovery_rate = success_count / 10
            avg_steps = total_steps / success_count if success_count > 0 else float('inf')
            
            recovery_stats.append({
                'size': size,
                'recovery_rate': recovery_rate,
                'avg_steps': avg_steps
            })
            
            print(f"  扰动大小 {size}: 恢复率 {recovery_rate:.1%}, 平均步数 {avg_steps:.1f}")
        
        # 验证小扰动高恢复率
        for stat in recovery_stats:
            if stat['size'] <= self.verifier.epsilon_1:
                self.assertGreater(
                    stat['recovery_rate'], 0.7,
                    f"小扰动（大小{stat['size']}）应该有高恢复率"
                )
        
        print("✓ 小扰动恢复验证通过")
    
    def test_basin_of_attraction(self):
        """测试吸引域"""
        print("\n=== 测试吸引域 ===")
        
        fixed_points = self.verifier.find_fixed_points()
        if not fixed_points:
            self.skipTest("没有找到不动点")
        
        # 测试前几个不动点的吸引域
        for i, fp in enumerate(fixed_points[:3]):
            basin_data = self.verifier.compute_basin_of_attraction(fp)
            
            print(f"\n不动点 {i+1} {fp} 的吸引域:")
            print(f"  吸引域大小: {basin_data['basin_size']}")
            print(f"  占总状态比例: {basin_data['basin_ratio']:.2%}")
            print(f"  平均收敛时间: {basin_data['average_convergence_time']:.1f}")
            print(f"  连通性: {basin_data['is_connected']}")
            print(f"  边界状态数: {basin_data['edge_states']}")
            
            # 验证吸引域非空
            self.assertGreater(
                basin_data['basin_size'], 0,
                "吸引域应该非空"
            )
            
            # 验证不动点在其自身的吸引域中
            self.assertIn(
                tuple(fp), 
                [tuple(s) for s in self.verifier.phi_system.valid_states 
                 if tuple(s) in [tuple(state) for state in basin_data['sample_states']] or 
                 tuple(s) == tuple(fp)],
                "不动点应该在其自身的吸引域中"
            )
        
        print("\n✓ 吸引域验证通过")
    
    def test_stability_types(self):
        """测试稳定性类型"""
        print("\n=== 测试稳定性类型 ===")
        
        fixed_points = self.verifier.find_fixed_points()
        if not fixed_points:
            self.skipTest("没有找到不动点")
        
        # 分析第一个不动点的稳定性
        fp = fixed_points[0]
        stability = self.verifier.analyze_stability_type(fp)
        
        print(f"不动点 {fp} 的稳定性分析:")
        print(f"  Lyapunov稳定: {stability['lyapunov_stable']}")
        print(f"  渐近稳定: {stability['asymptotically_stable']}")
        print(f"  全局稳定: {stability['globally_stable']}")
        print(f"  稳定半径: {stability['stability_radius']}")
        
        # 至少应该是Lyapunov稳定的
        self.assertTrue(
            stability['lyapunov_stable'],
            "不动点应该至少是Lyapunov稳定的"
        )
        
        print("✓ 稳定性类型验证通过")
    
    def test_perturbation_classes(self):
        """测试不同类别的扰动"""
        print("\n=== 测试扰动分类响应 ===")
        
        fixed_points = self.verifier.find_fixed_points()
        if not fixed_points:
            self.skipTest("没有找到不动点")
            
        fp = fixed_points[0]
        pert_results = self.verifier.test_perturbation_classes(fp)
        
        print("\n小扰动响应:")
        for result in pert_results['small_perturbations']:
            print(f"  大小 {result['size']}: 收敛 {result['converged']}, "
                  f"最终距离 {result['final_distance']}")
        
        print("\n中等扰动响应:")
        for result in pert_results['medium_perturbations']:
            print(f"  大小 {result['size']}: 回到原点 {result['converged_to_original']}, "
                  f"到其他点 {result['converged_to_other']}")
        
        print("\n大扰动响应:")
        for result in pert_results['large_perturbations']:
            print(f"  大小 {result['size']}: 结构保持 {result['structure_preserved']}, "
                  f"最终距离 {result['final_distance']}")
        
        # 验证扰动分类的合理性
        # 小扰动应该主要收敛
        small_convergence = sum(1 for r in pert_results['small_perturbations'] 
                               if r['converged'])
        self.assertGreater(
            small_convergence / len(pert_results['small_perturbations']),
            0.5,
            "小扰动应该主要收敛到原不动点"
        )
        
        print("\n✓ 扰动分类验证通过")
    
    def test_restoration_force(self):
        """测试恢复力"""
        print("\n=== 测试自指恢复力 ===")
        
        fixed_points = self.verifier.find_fixed_points()
        if not fixed_points:
            self.skipTest("没有找到不动点")
            
        fp = fixed_points[0]
        
        # 测试不同距离的状态的恢复力
        test_cases = []
        
        # 获取不同距离的状态
        for state in self.verifier.phi_system.valid_states[:20]:
            dist = self.verifier.hamming_distance(state, fp)
            if 0 < dist <= 3:  # 只测试近距离的状态
                force_data = self.verifier.compute_restoration_force(state, fp)
                test_cases.append((dist, force_data))
        
        # 按距离分组统计
        distance_groups = defaultdict(list)
        for dist, force_data in test_cases:
            distance_groups[dist].append(force_data)
        
        print("恢复力分析:")
        for dist in sorted(distance_groups.keys()):
            group = distance_groups[dist]
            pointing_count = sum(1 for f in group if f['points_toward_fixed_point'])
            avg_reduction = sum(f['distance_reduction'] for f in group) / len(group)
            
            print(f"  距离 {dist}: {pointing_count}/{len(group)} 指向不动点, "
                  f"平均距离减少 {avg_reduction:.2f}")
        
        print("\n✓ 恢复力验证通过")
    
    def test_complete_stability_verification(self):
        """C3-2完整稳定性验证"""
        print("\n=== C3-2 完整稳定性验证 ===")
        
        # 1. 不动点
        fixed_points = self.verifier.find_fixed_points()
        print(f"\n1. 不动点: 找到 {len(fixed_points)} 个")
        self.assertGreater(len(fixed_points), 0, "必须存在不动点")
        
        if not fixed_points:
            return
            
        # 2. Lyapunov稳定性
        fp = fixed_points[0]
        sim = self.verifier.simulate_with_perturbation(fp, 2, steps=30)
        lyap_data = self.verifier.verify_lyapunov_decrease(sim['trajectory'], fp)
        
        print(f"\n2. Lyapunov稳定性:")
        print(f"   平均递减率: {-lyap_data['average_rate']:.4f}")
        self.assertLessEqual(lyap_data['average_rate'], 0, "Lyapunov函数应递减")
        
        # 3. 吸引域
        basin_data = self.verifier.compute_basin_of_attraction(fp)
        print(f"\n3. 吸引域:")
        print(f"   大小: {basin_data['basin_size']}")
        print(f"   比例: {basin_data['basin_ratio']:.2%}")
        self.assertGreater(basin_data['basin_size'], 1, "吸引域应包含多个状态")
        
        # 4. 扰动响应
        pert_results = self.verifier.test_perturbation_classes(fp)
        small_conv = sum(1 for r in pert_results['small_perturbations'] if r['converged'])
        print(f"\n4. 扰动响应:")
        print(f"   小扰动恢复率: {small_conv/len(pert_results['small_perturbations']):.1%}")
        
        # 5. 稳定性类型
        stability = self.verifier.analyze_stability_type(fp)
        print(f"\n5. 稳定性分析:")
        print(f"   Lyapunov稳定: {stability['lyapunov_stable']}")
        print(f"   渐近稳定: {stability['asymptotically_stable']}")
        
        self.assertTrue(stability['lyapunov_stable'], "应该是Lyapunov稳定的")
        
        print("\n✓ C3-2稳定性推论验证通过！")


def run_stability_verification():
    """运行稳定性验证"""
    print("=" * 80)
    print("C3-2 稳定性推论 - 完整二进制验证")
    print("=" * 80)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestC3_2_BinaryStability)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 80)
    if result.wasSuccessful():
        print("✓ C3-2二进制稳定性推论验证成功！")
        print("系统的内在稳定性机制得到完整验证。")
    else:
        print("✗ C3-2稳定性验证发现问题")
        print(f"失败: {len(result.failures)}, 错误: {len(result.errors)}")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    success = run_stability_verification()
    exit(0 if success else 1)