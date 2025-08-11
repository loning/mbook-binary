#!/usr/bin/env python3
"""
C4-1: 量子系统的经典化推论的机器验证程序

验证点：
1. 量子态演化的正确性
2. 退相干率的φ-标度关系
3. 熵增过程验证
4. 经典极限收敛性
5. φ-基稳定性
6. 不可逆性证明
"""

import unittest
import numpy as np
import torch
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


class QuantumClassicalizationSystem:
    """量子经典化系统实现"""
    
    def __init__(self, dimension: int, coupling_strength: float = 0.1):
        self.dimension = dimension
        self.coupling_strength = coupling_strength
        self.phi = (1 + np.sqrt(5)) / 2
        
        # 计算退相干率矩阵
        self.decoherence_rates = self._calculate_decoherence_rates()
        
    def _calculate_decoherence_rates(self) -> np.ndarray:
        """计算退相干率矩阵"""
        Gamma = np.zeros((self.dimension, self.dimension))
        
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i != j:
                    # φ-标度的退相干率
                    Gamma[i, j] = self.coupling_strength * (abs(i - j) ** (1 / self.phi))
                    
        return Gamma
    
    def create_phi_basis_state(self, n: int) -> np.ndarray:
        """创建φ-基态"""
        state = np.zeros(self.dimension, dtype=complex)
        if n < self.dimension:
            state[n] = 1.0
        return state
    
    def create_superposition_state(self, coefficients: List[complex], 
                                 basis_indices: List[int]) -> np.ndarray:
        """创建量子叠加态"""
        state = np.zeros(self.dimension, dtype=complex)
        
        for coeff, idx in zip(coefficients, basis_indices):
            if idx < self.dimension:
                state[idx] = coeff
        
        # 归一化
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm
            
        return state
    
    def state_to_density_matrix(self, state: np.ndarray) -> np.ndarray:
        """将纯态转换为密度矩阵"""
        return np.outer(state, state.conj())
    
    def quantum_evolution(self, rho: np.ndarray, time: float) -> np.ndarray:
        """量子态演化（纯退相干）"""
        # 复制密度矩阵
        rho_evolved = rho.copy()
        
        # 应用退相干：非对角元素指数衰减
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i != j:
                    rho_evolved[i, j] = rho[i, j] * np.exp(-self.decoherence_rates[i, j] * time)
        
        return rho_evolved
    
    def von_neumann_entropy(self, rho: np.ndarray) -> float:
        """计算von Neumann熵"""
        # 计算本征值
        eigenvalues = np.linalg.eigvalsh(rho)
        
        # 过滤掉数值误差导致的负值和零值
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        
        # 计算熵
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        
        return float(entropy)
    
    def is_classical(self, rho: np.ndarray, tolerance: float = 1e-10) -> bool:
        """检查密度矩阵是否为经典态（对角）"""
        off_diagonal_norm = 0
        
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i != j:
                    off_diagonal_norm += abs(rho[i, j]) ** 2
        
        return np.sqrt(off_diagonal_norm) < tolerance
    
    def measure_classicality(self, rho: np.ndarray) -> float:
        """测量态的经典性（0=纯量子，1=完全经典）"""
        diag_norm = np.sum(np.abs(np.diag(rho)) ** 2)
        total_norm = np.sum(np.abs(rho) ** 2)
        
        if total_norm > 0:
            return diag_norm / total_norm
        return 1.0
    
    def decoherence_timescale(self, system_size: int) -> float:
        """计算退相干时间尺度"""
        tau_0 = 1.0  # 微观时间尺度
        return tau_0 * (self.phi ** (-np.log(system_size)))
    
    def simulate_classicalization(self, initial_state: np.ndarray, 
                                max_time: float, dt: float = 0.1) -> List[dict]:
        """模拟经典化过程"""
        # 转换为密度矩阵
        if len(initial_state.shape) == 1:
            rho = self.state_to_density_matrix(initial_state)
        else:
            rho = initial_state
        
        trajectory = []
        t = 0
        
        while t <= max_time:
            # 演化密度矩阵
            rho_t = self.quantum_evolution(rho, t)
            
            # 计算熵和经典性
            entropy = self.von_neumann_entropy(rho_t)
            classicality = self.measure_classicality(rho_t)
            
            trajectory.append({
                'time': t,
                'density_matrix': rho_t.copy(),
                'entropy': entropy,
                'classicality': classicality,
                'is_classical': self.is_classical(rho_t)
            })
            
            # 如果已经经典化，可以提前结束
            if self.is_classical(rho_t):
                break
            
            t += dt
        
        return trajectory


class TestC4_1QuantumClassicalization(unittest.TestCase):
    """C4-1定理验证测试"""
    
    def setUp(self):
        """测试初始化"""
        self.system = QuantumClassicalizationSystem(dimension=8, coupling_strength=0.1)
        
    def test_quantum_state_evolution(self):
        """测试1：量子态演化的正确性"""
        print("\n=== 测试量子态演化 ===")
        
        # 创建叠加态
        coeffs = [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
        indices = [0, 2, 4]  # 避免连续11编码
        
        state = self.system.create_superposition_state(coeffs, indices)
        rho_initial = self.system.state_to_density_matrix(state)
        
        # 测试不同时间点的演化
        times = [0, 1, 5, 10, 20]
        previous_entropy = -1
        
        for t in times:
            rho_t = self.system.quantum_evolution(rho_initial, t)
            
            # 验证密度矩阵性质
            # 1. 厄米性
            self.assertTrue(np.allclose(rho_t, rho_t.conj().T), 
                          f"密度矩阵在t={t}时不是厄米的")
            
            # 2. 迹为1
            trace = np.trace(rho_t)
            self.assertAlmostEqual(trace.real, 1.0, places=10,
                                 msg=f"密度矩阵在t={t}时迹不为1: {trace}")
            
            # 3. 正定性
            eigenvalues = np.linalg.eigvalsh(rho_t)
            self.assertTrue(np.all(eigenvalues >= -1e-10),
                          f"密度矩阵在t={t}时有负本征值: {eigenvalues}")
            
            # 4. 熵增
            entropy = self.system.von_neumann_entropy(rho_t)
            if previous_entropy >= 0:
                self.assertGreaterEqual(entropy, previous_entropy - 1e-10,
                                      f"熵在t={t}时减少了")
            previous_entropy = entropy
            
            print(f"t={t}: 熵={entropy:.6f}, 经典性={self.system.measure_classicality(rho_t):.6f}")
    
    def test_decoherence_rate_scaling(self):
        """测试2：退相干率的φ-标度关系"""
        print("\n=== 测试退相干率标度 ===")
        
        dimensions = [4, 8, 16, 32]
        max_rates = []
        
        for dim in dimensions:
            system = QuantumClassicalizationSystem(dimension=dim, coupling_strength=1.0)
            rates = system.decoherence_rates
            
            # 检查对角元素为0
            for i in range(dim):
                self.assertEqual(rates[i, i], 0, "对角退相干率应该为0")
            
            # 检查φ-标度
            for i in range(dim):
                for j in range(dim):
                    if i != j:
                        expected_rate = abs(i - j) ** (1 / system.phi)
                        self.assertAlmostEqual(rates[i, j], expected_rate, places=6,
                                             msg=f"退相干率[{i},{j}]不符合φ-标度")
            
            # 记录最大退相干率
            max_rate = rates[0, dim-1]
            max_rates.append(max_rate)
            print(f"维度={dim}: 最大退相干率={max_rate:.6f}")
        
        # 验证不同维度间的标度关系
        for i in range(len(dimensions) - 1):
            ratio = max_rates[i+1] / max_rates[i]
            expected_ratio = ((dimensions[i+1] - 1) / (dimensions[i] - 1)) ** (1 / self.system.phi)
            self.assertAlmostEqual(ratio, expected_ratio, places=3,
                                 msg=f"维度间标度关系不正确: {ratio} vs {expected_ratio}")
    
    def test_entropy_increase_verification(self):
        """测试3：熵增过程验证"""
        print("\n=== 测试熵增过程 ===")
        
        # 测试不同初始态
        test_cases = [
            # 纯态叠加
            ([1/np.sqrt(2), 1/np.sqrt(2)], [0, 2]),
            ([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], [0, 2, 4]),
            # 非均匀叠加
            ([0.8, 0.6], [1, 5]),  # 会自动归一化
        ]
        
        for coeffs, indices in test_cases:
            state = self.system.create_superposition_state(coeffs, indices)
            trajectory = self.system.simulate_classicalization(state, max_time=50, dt=0.5)
            
            # 验证熵的单调性
            for i in range(1, len(trajectory)):
                self.assertGreaterEqual(trajectory[i]['entropy'], 
                                      trajectory[i-1]['entropy'] - 1e-10,
                                      "熵应该单调增加")
            
            initial_entropy = trajectory[0]['entropy']
            final_entropy = trajectory[-1]['entropy']
            
            print(f"初态系数={coeffs}, 基={indices}")
            print(f"  初始熵={initial_entropy:.6f}, 最终熵={final_entropy:.6f}")
            print(f"  熵增={final_entropy - initial_entropy:.6f}")
    
    def test_classical_limit_convergence(self):
        """测试4：经典极限收敛性"""
        print("\n=== 测试经典极限收敛 ===")
        
        # 使用更强的耦合强度来加速经典化
        strong_system = QuantumClassicalizationSystem(dimension=8, coupling_strength=0.5)
        
        # 创建量子叠加态
        state = strong_system.create_superposition_state(
            [0.5, 0.5, 0.5, 0.5], [0, 2, 4, 6]
        )
        
        # 模拟长时间演化
        trajectory = strong_system.simulate_classicalization(state, max_time=100, dt=0.1)
        
        # 找到达到经典态的时间
        classical_time = None
        for point in trajectory:
            if point['is_classical']:
                classical_time = point['time']
                break
        
        self.assertIsNotNone(classical_time, "系统应该达到经典极限")
        
        # 验证经典态的性质
        final_state = trajectory[-1]['density_matrix']
        
        # 1. 应该是对角的
        self.assertTrue(strong_system.is_classical(final_state),
                      "最终态应该是经典的（对角的）")
        
        # 2. 对角元素应该是原始态的模方
        original_rho = strong_system.state_to_density_matrix(state)
        for i in range(strong_system.dimension):
            self.assertAlmostEqual(final_state[i, i].real, abs(original_rho[i, i]), 
                                 places=6,
                                 msg=f"对角元素{i}不正确")
        
        print(f"达到经典极限的时间: {classical_time}")
        print(f"最终经典性: {trajectory[-1]['classicality']:.6f}")
    
    def test_phi_basis_stability(self):
        """测试5：φ-基稳定性"""
        print("\n=== 测试φ-基稳定性 ===")
        
        # 测试φ-基态在演化下的稳定性
        stable_states = []
        
        for n in [0, 1, 2, 4, 5, 8]:  # 避免连续11的索引
            # 创建φ-基态
            state = self.system.create_phi_basis_state(n)
            rho_initial = self.system.state_to_density_matrix(state)
            
            # 演化很长时间
            rho_evolved = self.system.quantum_evolution(rho_initial, 100)
            
            # 检查是否保持不变
            difference = np.linalg.norm(rho_evolved - rho_initial, 'fro')
            is_stable = difference < 1e-10
            stable_states.append((n, is_stable, difference))
            
            self.assertTrue(is_stable, 
                          f"φ-基态|{n}>应该在演化下稳定，但差异为{difference}")
            
            print(f"φ-基态|{n}>: 稳定性={is_stable}, 差异={difference:.2e}")
    
    def test_irreversibility_proof(self):
        """测试6：不可逆性证明"""
        print("\n=== 测试不可逆性 ===")
        
        # 创建纯态（熵=0）
        state = self.system.create_superposition_state([0.6, 0.8], [1, 4])
        rho_initial = self.system.state_to_density_matrix(state)
        
        initial_entropy = self.system.von_neumann_entropy(rho_initial)
        self.assertAlmostEqual(initial_entropy, 0.0, places=10,
                             msg="纯态的初始熵应该为0")
        
        # 演化到经典态
        trajectory = self.system.simulate_classicalization(state, max_time=50)
        final_entropy = trajectory[-1]['entropy']
        
        # 验证熵增
        self.assertGreater(final_entropy, 0.01,
                         "经典化后熵应该显著增加")
        
        # 验证过程的不可逆性
        # 尝试"反演"退相干（这在物理上是不可能的）
        # 这里我们通过检查熵的单调性来证明
        
        entropy_differences = []
        for i in range(1, len(trajectory)):
            diff = trajectory[i]['entropy'] - trajectory[i-1]['entropy']
            entropy_differences.append(diff)
        
        # 所有熵变应该非负
        self.assertTrue(all(d >= -1e-10 for d in entropy_differences),
                      "熵应该单调不减，证明过程不可逆")
        
        print(f"初始熵: {initial_entropy:.6f}")
        print(f"最终熵: {final_entropy:.6f}")
        print(f"总熵增: {final_entropy - initial_entropy:.6f}")
        print(f"最大单步熵增: {max(entropy_differences):.6f}")
    
    def test_decoherence_timescale(self):
        """测试7：退相干时间尺度"""
        print("\n=== 测试退相干时间尺度 ===")
        
        system_sizes = [2, 4, 8, 16, 32, 64]
        timescales = []
        
        for N in system_sizes:
            tau_D = self.system.decoherence_timescale(N)
            timescales.append(tau_D)
            print(f"系统规模 N={N}: 退相干时间 τ_D={tau_D:.6e}")
        
        # 验证标度关系
        for i in range(len(system_sizes) - 1):
            N1, N2 = system_sizes[i], system_sizes[i+1]
            tau1, tau2 = timescales[i], timescales[i+1]
            
            # 验证 τ_D(N) = τ_0 * φ^(-ln(N))
            ratio_actual = tau2 / tau1
            ratio_expected = (self.system.phi ** (-np.log(N2))) / (self.system.phi ** (-np.log(N1)))
            
            self.assertAlmostEqual(ratio_actual, ratio_expected, places=6,
                                 msg=f"时间尺度比值不符合理论: {ratio_actual} vs {ratio_expected}")
        
        # 验证大系统退相干更快
        for i in range(len(timescales) - 1):
            self.assertLess(timescales[i+1], timescales[i],
                          "更大的系统应该有更短的退相干时间")
    
    def test_visualization(self):
        """测试8：可视化经典化过程"""
        print("\n=== 生成可视化 ===")
        
        # 创建量子叠加态
        state = self.system.create_superposition_state(
            [0.5, 0.5, 0.5, 0.5], [0, 2, 4, 6]
        )
        
        # 模拟经典化
        trajectory = self.system.simulate_classicalization(state, max_time=30, dt=0.1)
        
        # 提取数据
        times = [p['time'] for p in trajectory]
        entropies = [p['entropy'] for p in trajectory]
        classicalities = [p['classicality'] for p in trajectory]
        
        # 绘图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 熵演化
        ax1.plot(times, entropies, 'b-', linewidth=2)
        ax1.set_ylabel('von Neumann熵', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('量子系统的经典化过程', fontsize=14)
        
        # 经典性演化
        ax2.plot(times, classicalities, 'r-', linewidth=2)
        ax2.set_xlabel('时间', fontsize=12)
        ax2.set_ylabel('经典性度量', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig('quantum_classicalization_C4_1.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 密度矩阵演化可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        time_indices = [0, len(trajectory)//5, len(trajectory)//3, 
                       len(trajectory)//2, 3*len(trajectory)//4, -1]
        
        for idx, (ax, t_idx) in enumerate(zip(axes.flat, time_indices)):
            rho = trajectory[t_idx]['density_matrix']
            t = trajectory[t_idx]['time']
            
            # 绘制密度矩阵的模
            im = ax.imshow(np.abs(rho), cmap='hot', aspect='equal')
            ax.set_title(f't = {t:.1f}', fontsize=12)
            ax.set_xlabel('j')
            ax.set_ylabel('i')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.suptitle('密度矩阵演化：从量子到经典', fontsize=16)
        plt.tight_layout()
        plt.savefig('density_matrix_evolution_C4_1.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("可视化已保存为:")
        print("  - quantum_classicalization_C4_1.png")
        print("  - density_matrix_evolution_C4_1.png")


if __name__ == '__main__':
    unittest.main(verbosity=2)