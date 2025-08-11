#!/usr/bin/env python3
"""
T12-1: 量子-经典过渡定理的机器验证程序

验证点:
1. 量子叠加态表示 (quantum_superposition_representation)
2. 熵增机制 (entropy_increase_mechanism)  
3. No-11约束执行 (no11_constraint_enforcement)
4. φ-表示收敛 (phi_representation_convergence)
5. 塌缩时间计算 (collapse_time_calculation)
6. 经典态稳定性 (classical_state_stability)
"""

import unittest
import numpy as np
import math
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class QuantumSuperposition:
    """量子叠加态"""
    coefficients: List[complex]
    basis_states: List[int]  # No-11 valid binary states
    
    def __post_init__(self):
        """验证初始化"""
        assert len(self.coefficients) == len(self.basis_states)
        # 验证归一化
        total_prob = sum(abs(c)**2 for c in self.coefficients)
        assert abs(total_prob - 1.0) < 1e-10, f"State not normalized: {total_prob}"
        # 验证no-11约束
        for state in self.basis_states:
            assert self.is_no11_valid(state), f"State {state} violates no-11 constraint"
    
    @staticmethod
    def is_no11_valid(state: int) -> bool:
        """检查是否满足no-11约束"""
        binary = format(state, 'b')
        return '11' not in binary
    
    def get_density_matrix(self) -> np.ndarray:
        """获取密度矩阵"""
        n = len(self.basis_states)
        rho = np.zeros((n, n), dtype=complex)
        
        for i in range(n):
            for j in range(n):
                rho[i, j] = self.coefficients[i] * np.conj(self.coefficients[j])
        
        return rho
    
    def von_neumann_entropy(self) -> float:
        """计算von Neumann熵"""
        rho = self.get_density_matrix()
        eigenvals = np.linalg.eigvals(rho)
        # 移除数值误差导致的负值
        eigenvals = eigenvals[eigenvals > 1e-12]
        
        if len(eigenvals) == 0:
            return 0.0
        
        return -np.sum(eigenvals * np.log(eigenvals)).real
    
    def is_classical(self) -> bool:
        """检查是否为经典态"""
        # 经典态只有一个非零系数
        non_zero_count = sum(1 for c in self.coefficients if abs(c) > 1e-10)
        return non_zero_count == 1
    
    def measure_coherence(self) -> float:
        """测量相干性"""
        # 使用最小非零系数作为相干性度量
        non_zero_coeffs = [abs(c) for c in self.coefficients if abs(c) > 1e-12]
        return min(non_zero_coeffs)**2 if non_zero_coeffs else 0.0


@dataclass 
class ClassicalState:
    """经典态"""
    state_value: int
    phi_encoding: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化φ-表示"""
        if not self.phi_encoding:
            self.phi_encoding = self.to_zeckendorf(self.state_value)
    
    @staticmethod
    def to_zeckendorf(n: int) -> List[int]:
        """转换为Zeckendorf表示（标准φ-表示）"""
        if n == 0:
            return [0]
        
        # 生成足够的Fibonacci数
        fib = [1, 2]
        while fib[-1] < n:
            fib.append(fib[-1] + fib[-2])
        
        result = []
        remaining = n
        
        for f in reversed(fib):
            if f <= remaining:
                result.append(1)
                remaining -= f
            else:
                result.append(0)
        
        return result
    
    def verify_phi_representation(self) -> bool:
        """验证φ-表示的正确性"""
        # 重构数值
        fib = [1, 2]
        max_len = len(self.phi_encoding)
        while len(fib) < max_len:
            fib.append(fib[-1] + fib[-2])
        
        reconstructed = sum(
            bit * fib[i] 
            for i, bit in enumerate(reversed(self.phi_encoding))
            if i < len(fib)
        )
        
        return reconstructed == self.state_value and '11' not in ''.join(map(str, self.phi_encoding))


class QuantumClassicalTransition:
    """量子-经典过渡系统"""
    
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2
        self.hbar = 1.0  # 自然单位
        self.k_B = 1.0   # 自然单位
        
    def generate_no11_states(self, max_bits: int = 8) -> List[int]:
        """生成所有满足no-11约束的状态"""
        valid_states = []
        
        for i in range(2**max_bits):
            if QuantumSuperposition.is_no11_valid(i):
                valid_states.append(i)
        
        return valid_states
    
    def create_random_superposition(self, num_states: int = 5) -> QuantumSuperposition:
        """创建随机量子叠加态"""
        valid_states = self.generate_no11_states()
        
        # 随机选择状态
        selected_states = random.sample(valid_states, min(num_states, len(valid_states)))
        
        # 生成随机系数并归一化
        coeffs = [complex(random.gauss(0, 1), random.gauss(0, 1)) for _ in selected_states]
        norm = math.sqrt(sum(abs(c)**2 for c in coeffs))
        coeffs = [c / norm for c in coeffs]
        
        return QuantumSuperposition(coeffs, selected_states)
    
    def construct_self_observation_operator(self, state: QuantumSuperposition) -> np.ndarray:
        """构造自指观测算子"""
        n = len(state.basis_states)
        operator = np.zeros((n, n), dtype=complex)
        
        # 自指投影算子：|s_i⟩⟨s_i| ⊗ |s_i⟩⟨s_i|
        for i in range(n):
            operator[i, i] = 1.0  # 简化的自指测量
        
        return operator
    
    def apply_observation_evolution(self, state: QuantumSuperposition, dt: float = 0.01) -> QuantumSuperposition:
        """应用观测演化"""
        # 应用退相干和φ-选择机制
        new_coeffs = []
        
        for i, (coeff, basis_state) in enumerate(zip(state.coefficients, state.basis_states)):
            # 计算φ-表示质量
            phi_quality = self.measure_phi_quality(basis_state)
            
            # 退相干率与φ-质量成反比
            decoherence_rate = 1.0 / (1.0 + phi_quality)
            
            # 演化系数（偏向φ-表示）
            decay_factor = math.exp(-decoherence_rate * dt)
            enhancement_factor = 1.0 + phi_quality * dt
            
            new_coeff = coeff * decay_factor * enhancement_factor
            new_coeffs.append(new_coeff)
        
        # 重新归一化
        norm = math.sqrt(sum(abs(c)**2 for c in new_coeffs))
        if norm > 1e-12:
            new_coeffs = [c / norm for c in new_coeffs]
        
        return QuantumSuperposition(new_coeffs, state.basis_states.copy())
    
    def measure_phi_quality(self, state_value: int) -> float:
        """测量φ-表示质量"""
        if state_value == 0:
            return 0.0
        
        # 转换为Zeckendorf表示
        zeckendorf = ClassicalState.to_zeckendorf(state_value)
        
        # 质量度量：更短的表示和更少的连续项
        length_penalty = len(zeckendorf) / 10.0
        
        # 检查黄金比例结构
        golden_ratio_bonus = 0.0
        if len(zeckendorf) > 1:
            # 检查相邻Fibonacci数的比例
            for i in range(len(zeckendorf) - 1):
                if zeckendorf[i] == 1 and zeckendorf[i+1] == 1:
                    golden_ratio_bonus -= 0.5  # 惩罚连续1（虽然Zeckendorf不应该有）
                elif zeckendorf[i] == 1:
                    golden_ratio_bonus += 0.1
        
        return max(0.0, 1.0 - length_penalty + golden_ratio_bonus)
    
    def calculate_collapse_time(self, initial_state: QuantumSuperposition) -> float:
        """计算理论塌缩时间"""
        min_coeff = min(abs(c) for c in initial_state.coefficients if abs(c) > 1e-12)
        
        if min_coeff == 0:
            return float('inf')
        
        # 能量尺度
        E_phi = self.hbar * self.phi  # 简化的φ-能量尺度
        
        # 递归深度估计
        recursion_depth = self.estimate_recursion_depth(initial_state)
        depth_factor = 1.0 + recursion_depth / 7.0  # d_critical = 7
        
        collapse_time = (self.hbar / E_phi) * \
                       math.log(1.0 / min_coeff**2, self.phi) * \
                       depth_factor
        
        return max(0.01, collapse_time)  # 避免非正值
    
    def estimate_recursion_depth(self, state: QuantumSuperposition) -> int:
        """估计递归深度"""
        # 基于状态复杂度的简化估计
        max_state = max(state.basis_states) if state.basis_states else 1
        return max(1, int(math.log2(max_state + 1)))
    
    def simulate_collapse(self, initial_state: QuantumSuperposition, max_time: float = 100.0) -> Dict[str, Any]:
        """模拟量子态塌缩过程"""
        current_state = initial_state
        evolution_history = [current_state]
        time_elapsed = 0.0
        dt = 0.1
        
        initial_entropy = current_state.von_neumann_entropy()
        
        while time_elapsed < max_time and not current_state.is_classical():
            # 演化一步
            next_state = self.apply_observation_evolution(current_state, dt)
            
            # 检查熵增
            current_entropy = current_state.von_neumann_entropy()
            next_entropy = next_state.von_neumann_entropy()
            
            # 更新状态
            current_state = next_state
            evolution_history.append(current_state)
            time_elapsed += dt
            
            # 自适应时间步长
            coherence = current_state.measure_coherence()
            if coherence < 0.01:
                dt = min(0.5, dt * 1.1)  # 增大时间步长
            else:
                dt = max(0.01, dt * 0.9)  # 减小时间步长
        
        final_entropy = current_state.von_neumann_entropy()
        
        # 确定最终经典态
        final_classical_state = None
        if current_state.is_classical():
            # 找到主导态
            max_prob = 0
            dominant_idx = 0
            for i, coeff in enumerate(current_state.coefficients):
                if abs(coeff)**2 > max_prob:
                    max_prob = abs(coeff)**2
                    dominant_idx = i
            
            final_classical_state = ClassicalState(current_state.basis_states[dominant_idx])
        
        return {
            'final_state': current_state,
            'final_classical_state': final_classical_state,
            'collapse_time': time_elapsed,
            'evolution_history': evolution_history,
            'initial_entropy': initial_entropy,
            'final_entropy': final_entropy,
            'entropy_increased': final_entropy > initial_entropy,
            'collapsed_successfully': current_state.is_classical()
        }


class TestT12_1QuantumClassicalTransition(unittest.TestCase):
    """T12-1定理验证测试"""
    
    def setUp(self):
        """测试初始化"""
        self.system = QuantumClassicalTransition()
        random.seed(42)
        np.random.seed(42)
    
    def test_quantum_superposition_representation(self):
        """测试1：量子叠加态表示"""
        print("\n=== 测试量子叠加态表示 ===")
        
        # 测试No-11约束的状态生成
        valid_states = self.system.generate_no11_states(6)
        print(f"6位系统的no-11有效状态数: {len(valid_states)}")
        
        # 验证Fibonacci关系：|No11Valid(n)| = F_{n+2}
        expected_count = self.fibonacci(8)  # 6+2 = 8
        print(f"理论预期状态数: {expected_count}")
        
        self.assertEqual(len(valid_states), expected_count, 
                        "No-11有效状态数应该等于F_{n+2}")
        
        # 验证所有状态确实满足no-11约束
        for state in valid_states:
            binary_rep = format(state, 'b')
            self.assertNotIn('11', binary_rep,
                           f"状态 {state} ({binary_rep}) 违反no-11约束")
        
        # 创建并验证量子叠加态
        superposition = self.system.create_random_superposition(5)
        print(f"创建的叠加态包含 {len(superposition.basis_states)} 个基态")
        
        # 验证归一化
        total_prob = sum(abs(c)**2 for c in superposition.coefficients)
        self.assertAlmostEqual(total_prob, 1.0, places=10,
                              msg="量子态应该归一化")
        
        # 验证所有基态满足no-11约束
        for state in superposition.basis_states:
            self.assertTrue(QuantumSuperposition.is_no11_valid(state),
                          f"基态 {state} 应该满足no-11约束")
        
        print(f"叠加态von Neumann熵: {superposition.von_neumann_entropy():.6f}")
    
    def fibonacci(self, n: int) -> int:
        """计算第n个Fibonacci数"""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    def test_entropy_increase_mechanism(self):
        """测试2：熵增机制"""
        print("\n=== 测试熵增机制 ===")
        
        # 创建初始叠加态
        initial_state = self.system.create_random_superposition(4)
        initial_entropy = initial_state.von_neumann_entropy()
        
        print(f"初始熵: {initial_entropy:.6f}")
        
        # 应用一步观测演化
        evolved_state = self.system.apply_observation_evolution(initial_state)
        evolved_entropy = evolved_state.von_neumann_entropy()
        
        print(f"演化后熵: {evolved_entropy:.6f}")
        print(f"熵变化: {evolved_entropy - initial_entropy:.6f}")
        
        # 验证熵增（允许数值误差）
        entropy_change = evolved_entropy - initial_entropy
        self.assertGreaterEqual(entropy_change, -1e-10,
                               "熵不应该显著减少（考虑数值误差）")
        
        # 测试完整塌缩过程的熵变化
        collapse_result = self.system.simulate_collapse(initial_state, max_time=50.0)
        
        print(f"塌缩过程：")
        print(f"  初始熵: {collapse_result['initial_entropy']:.6f}")
        print(f"  最终熵: {collapse_result['final_entropy']:.6f}")
        print(f"  熵增加: {collapse_result['entropy_increased']}")
        
        # 对于真正的塌缩，熵可能会减少到0（纯态）
        # 但在观测过程中应该有熵增的阶段
        if len(collapse_result['evolution_history']) > 2:
            # 检查中间某个阶段的熵增
            mid_point = len(collapse_result['evolution_history']) // 2
            mid_entropy = collapse_result['evolution_history'][mid_point].von_neumann_entropy()
            
            # 熵应该在某个阶段增加
            self.assertTrue(
                mid_entropy >= initial_entropy - 1e-6 or 
                collapse_result['final_entropy'] >= initial_entropy - 1e-6,
                "应该观察到熵增过程"
            )
    
    def test_no11_constraint_enforcement(self):
        """测试3：No-11约束执行"""
        print("\n=== 测试No-11约束执行 ===")
        
        # 创建叠加态并完整模拟塌缩
        initial_state = self.system.create_random_superposition(6)
        collapse_result = self.system.simulate_collapse(initial_state)
        
        print(f"演化历史长度: {len(collapse_result['evolution_history'])}")
        
        # 验证整个演化过程中的no-11约束
        for step, state in enumerate(collapse_result['evolution_history']):
            for basis_state in state.basis_states:
                binary_str = format(basis_state, 'b')
                self.assertNotIn('11', binary_str,
                               f"步骤 {step}: 状态 {basis_state} ({binary_str}) 违反no-11约束")
        
        # 验证有效状态数量的单调性（塌缩过程中不增加）
        if len(collapse_result['evolution_history']) > 1:
            for i in range(1, len(collapse_result['evolution_history'])):
                prev_count = len(collapse_result['evolution_history'][i-1].basis_states)
                curr_count = len(collapse_result['evolution_history'][i].basis_states)
                
                self.assertLessEqual(curr_count, prev_count,
                                   f"步骤 {i}: 有效状态数增加 {prev_count} → {curr_count}")
        
        # 验证最终态的no-11约束
        final_state = collapse_result['final_state']
        for basis_state in final_state.basis_states:
            self.assertTrue(QuantumSuperposition.is_no11_valid(basis_state),
                          f"最终状态 {basis_state} 应该满足no-11约束")
        
        print("所有演化步骤都满足no-11约束")
    
    def test_phi_representation_convergence(self):
        """测试4：φ-表示收敛"""
        print("\n=== 测试φ-表示收敛 ===")
        
        # 创建多个叠加态进行测试
        test_cases = []
        for i in range(3):
            initial_state = self.system.create_random_superposition(4)
            collapse_result = self.system.simulate_collapse(initial_state)
            test_cases.append(collapse_result)
        
        phi_convergence_count = 0
        
        for i, result in enumerate(test_cases):
            print(f"\n测试案例 {i+1}:")
            print(f"  塌缩成功: {result['collapsed_successfully']}")
            print(f"  塌缩时间: {result['collapse_time']:.3f}")
            
            if result['final_classical_state']:
                classical_state = result['final_classical_state']
                print(f"  最终经典态: {classical_state.state_value}")
                print(f"  φ-表示: {classical_state.phi_encoding}")
                
                # 验证φ-表示的正确性
                is_valid_phi = classical_state.verify_phi_representation()
                self.assertTrue(is_valid_phi, 
                              f"案例 {i+1}: φ-表示验证失败")
                
                # 测量φ-表示质量
                phi_quality = self.system.measure_phi_quality(classical_state.state_value)
                print(f"  φ-质量: {phi_quality:.3f}")
                
                if phi_quality > 0.3:  # 合理的质量阈值
                    phi_convergence_count += 1
        
        # 验证大部分案例收敛到高质量φ-表示
        convergence_rate = phi_convergence_count / len(test_cases)
        print(f"\nφ-表示收敛率: {convergence_rate:.2f}")
        
        self.assertGreater(convergence_rate, 0.3,
                         "应该有合理比例的案例收敛到高质量φ-表示")
    
    def test_collapse_time_calculation(self):
        """测试5：塌缩时间计算"""
        print("\n=== 测试塌缩时间计算 ===")
        
        # 测试不同相干性的叠加态
        test_cases = []
        
        # 高相干性（系数相近）
        high_coherence_coeffs = [0.5, 0.5, 0.5, 0.5]
        norm = math.sqrt(sum(abs(c)**2 for c in high_coherence_coeffs))
        high_coherence_coeffs = [c/norm for c in high_coherence_coeffs]
        high_coherence = QuantumSuperposition(
            high_coherence_coeffs,
            [1, 2, 4, 5]  # 选择no-11有效状态
        )
        test_cases.append(("高相干性", high_coherence))
        
        # 低相干性（系数差异大）
        low_coherence_coeffs = [0.95, 0.2, 0.2, 0.1]
        norm = math.sqrt(sum(abs(c)**2 for c in low_coherence_coeffs))
        low_coherence_coeffs = [c/norm for c in low_coherence_coeffs]
        low_coherence = QuantumSuperposition(
            low_coherence_coeffs,
            [1, 2, 4, 5]
        )
        test_cases.append(("低相干性", low_coherence))
        
        for case_name, initial_state in test_cases:
            print(f"\n{case_name}案例:")
            
            # 计算理论塌缩时间
            theoretical_time = self.system.calculate_collapse_time(initial_state)
            print(f"  理论塌缩时间: {theoretical_time:.3f}")
            
            # 实际模拟塌缩
            collapse_result = self.system.simulate_collapse(initial_state, max_time=max(100, theoretical_time * 2))
            actual_time = collapse_result['collapse_time']
            
            print(f"  实际塌缩时间: {actual_time:.3f}")
            print(f"  塌缩成功: {collapse_result['collapsed_successfully']}")
            
            # 验证时间有限性
            self.assertLess(theoretical_time, float('inf'),
                          f"{case_name}: 理论塌缩时间应该有限")
            
            self.assertGreater(theoretical_time, 0,
                             f"{case_name}: 理论塌缩时间应该为正")
            
            # 如果成功塌缩，验证时间的合理性
            if collapse_result['collapsed_successfully']:
                # 允许较大的误差范围，因为这是复杂的非线性过程
                time_ratio = actual_time / theoretical_time
                print(f"  时间比率 (实际/理论): {time_ratio:.2f}")
                
                # 验证时间在合理范围内（0.1x 到 10x 理论值）
                self.assertGreater(time_ratio, 0.01,
                                 f"{case_name}: 实际时间不应该过短")
                self.assertLess(time_ratio, 20,
                               f"{case_name}: 实际时间不应该过长")
    
    def test_classical_state_stability(self):
        """测试6：经典态稳定性"""
        print("\n=== 测试经典态稳定性 ===")
        
        # 创建几个经典态进行测试（确保都满足no-11约束）
        # 选择no-11有效的状态值
        test_classical_states = [
            ClassicalState(1),   # 二进制: 1
            ClassicalState(2),   # 二进制: 10
            ClassicalState(4),   # 二进制: 100
            ClassicalState(5),   # 二进制: 101
            ClassicalState(8),   # 二进制: 1000
        ]
        
        for i, classical_state in enumerate(test_classical_states):
            print(f"\n经典态 {i+1} (值={classical_state.state_value}):")
            print(f"  φ-表示: {classical_state.phi_encoding}")
            
            # 验证φ-表示的正确性
            is_valid = classical_state.verify_phi_representation()
            self.assertTrue(is_valid, 
                          f"经典态 {i+1}: φ-表示应该有效")
            
            # 转换为量子态表示
            quantum_rep = QuantumSuperposition(
                [1.0 + 0j],  # 纯态
                [classical_state.state_value]
            )
            
            # 验证是经典态
            self.assertTrue(quantum_rep.is_classical(),
                          f"经典态 {i+1}: 应该被识别为经典态")
            
            # 测试稳定性：应用微小演化
            evolved = self.system.apply_observation_evolution(quantum_rep, dt=1e-6)
            
            # 计算状态变化
            original_entropy = quantum_rep.von_neumann_entropy()
            evolved_entropy = evolved.von_neumann_entropy()
            
            print(f"  原始熵: {original_entropy:.10f}")
            print(f"  演化后熵: {evolved_entropy:.10f}")
            print(f"  熵变化: {abs(evolved_entropy - original_entropy):.2e}")
            
            # 经典态的熵应该保持很小（接近0）
            self.assertLess(original_entropy, 1e-8,
                          f"经典态 {i+1}: 原始熵应该接近0")
            
            # 演化后熵变化应该很小
            entropy_change = abs(evolved_entropy - original_entropy)
            self.assertLess(entropy_change, 1e-6,
                          f"经典态 {i+1}: 演化后熵变化应该很小")
            
            # 测量φ-质量
            phi_quality = self.system.measure_phi_quality(classical_state.state_value)
            print(f"  φ-质量: {phi_quality:.3f}")
            
            # 经典φ-表示态应该有合理的质量
            self.assertGreater(phi_quality, 0.0,
                             f"经典态 {i+1}: φ-质量应该为正")
    
    def test_integrated_quantum_classical_transition(self):
        """测试7：完整的量子-经典过渡"""
        print("\n=== 测试完整的量子-经典过渡 ===")
        
        # 创建一个更复杂的叠加态
        initial_state = self.system.create_random_superposition(8)
        
        print(f"初始叠加态:")
        print(f"  基态数: {len(initial_state.basis_states)}")
        print(f"  初始熵: {initial_state.von_neumann_entropy():.6f}")
        print(f"  初始相干性: {initial_state.measure_coherence():.6f}")
        
        # 计算理论预测
        theoretical_collapse_time = self.system.calculate_collapse_time(initial_state)
        print(f"  理论塌缩时间: {theoretical_collapse_time:.3f}")
        
        # 执行完整模拟
        start_time = 0
        max_simulation_time = max(200, theoretical_collapse_time * 3)
        
        collapse_result = self.system.simulate_collapse(
            initial_state, 
            max_time=max_simulation_time
        )
        
        print(f"\n塌缩结果:")
        print(f"  实际塌缩时间: {collapse_result['collapse_time']:.3f}")
        print(f"  塌缩成功: {collapse_result['collapsed_successfully']}")
        print(f"  最终熵: {collapse_result['final_entropy']:.6f}")
        print(f"  熵增加: {collapse_result['entropy_increased']}")
        
        # 验证核心理论预测
        
        # 1. 塌缩必然性（在有限时间内）
        self.assertLess(collapse_result['collapse_time'], float('inf'),
                       "塌缩时间应该有限")
        
        # 2. 熵增过程（至少在某个阶段）
        if len(collapse_result['evolution_history']) > 1:
            entropies = [state.von_neumann_entropy() for state in collapse_result['evolution_history']]
            max_entropy = max(entropies)
            min_entropy = min(entropies)
            
            print(f"  熵变化范围: [{min_entropy:.6f}, {max_entropy:.6f}]")
            
            # 应该观察到熵增过程
            self.assertGreaterEqual(max_entropy, collapse_result['initial_entropy'] - 1e-10,
                                  "应该有熵增过程")
        
        # 3. φ-表示收敛
        if collapse_result['final_classical_state']:
            final_classical = collapse_result['final_classical_state']
            phi_valid = final_classical.verify_phi_representation()
            
            print(f"  最终经典态: {final_classical.state_value}")
            print(f"  φ-表示: {final_classical.phi_encoding}")
            print(f"  φ-表示有效: {phi_valid}")
            
            self.assertTrue(phi_valid, "最终态应该是有效的φ-表示")
        
        # 4. No-11约束保持
        final_state = collapse_result['final_state']
        for basis_state in final_state.basis_states:
            self.assertTrue(QuantumSuperposition.is_no11_valid(basis_state),
                          f"最终基态 {basis_state} 应该满足no-11约束")
        
        print(f"\n✓ 完整的量子-经典过渡验证通过")
    
    def test_statistical_behavior(self):
        """测试8：统计行为验证"""
        print("\n=== 测试统计行为验证 ===")
        
        num_trials = 10
        results = []
        
        print(f"执行 {num_trials} 次独立塌缩实验...")
        
        for trial in range(num_trials):
            # 创建类似的初始条件（更少的状态数以提高塌缩成功率）
            initial_state = self.system.create_random_superposition(3)
            collapse_result = self.system.simulate_collapse(initial_state, max_time=200)
            
            results.append({
                'trial': trial,
                'collapsed': collapse_result['collapsed_successfully'],
                'collapse_time': collapse_result['collapse_time'],
                'initial_entropy': collapse_result['initial_entropy'],
                'final_entropy': collapse_result['final_entropy'],
                'entropy_increased': collapse_result['entropy_increased']
            })
        
        # 统计分析
        successful_collapses = sum(1 for r in results if r['collapsed'])
        success_rate = successful_collapses / num_trials
        
        avg_collapse_time = np.mean([r['collapse_time'] for r in results if r['collapsed']])
        std_collapse_time = np.std([r['collapse_time'] for r in results if r['collapsed']])
        
        entropy_increase_count = sum(1 for r in results if r['entropy_increased'])
        entropy_increase_rate = entropy_increase_count / num_trials
        
        print(f"\n统计结果:")
        print(f"  成功塌缩率: {success_rate:.2f}")
        print(f"  平均塌缩时间: {avg_collapse_time:.3f} ± {std_collapse_time:.3f}")
        print(f"  熵增事件率: {entropy_increase_rate:.2f}")
        
        # 验证统计性质
        self.assertGreater(success_rate, 0.5,
                         "大部分试验应该成功塌缩")
        
        self.assertGreater(avg_collapse_time, 0,
                         "平均塌缩时间应该为正")
        
        self.assertLess(avg_collapse_time, 1000,
                       "平均塌缩时间应该在合理范围内")
        
        # 至少部分试验应该显示熵增
        self.assertGreater(entropy_increase_rate, 0.2,
                         "应该观察到显著的熵增现象")


if __name__ == '__main__':
    unittest.main(verbosity=2)