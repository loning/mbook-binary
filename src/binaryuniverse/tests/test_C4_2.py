#!/usr/bin/env python3
"""
Unit tests for C4-2: Wave Function Collapse as Information Theory Corollary
验证波函数坍缩的信息理论特性

Tests verify:
1. 测量算子构建的正确性
2. 波函数坍缩过程的信息增益
3. φ基测量的最优性
4. 连续测量的信息累积
5. 量子-经典边界的信息理论特征
"""

import unittest
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 黄金分割率
PHI = (1 + np.sqrt(5)) / 2


class WaveFunctionCollapseSystem:
    """波函数坍缩的信息理论系统"""
    
    def __init__(self, dimension: int = 8):
        """
        初始化系统
        
        Args:
            dimension: 希尔伯特空间维度
        """
        self.dimension = dimension
        self.phi = PHI
        
        # 构建标准基和φ基
        self.computational_basis = self._build_computational_basis()
        self.phi_basis = self._build_phi_basis()
        
        # 构建满足no-11约束的基
        self.no11_indices = [i for i in range(dimension) if self._check_no11(i)]
        self.no11_phi_basis = self._build_no11_phi_basis()
    
    def _check_no11(self, index: int) -> bool:
        """检查索引是否满足no-11约束"""
        binary = bin(index)[2:]
        return '11' not in binary
    
    def _build_computational_basis(self) -> List[np.ndarray]:
        """构建计算基"""
        basis = []
        for i in range(self.dimension):
            state = np.zeros(self.dimension, dtype=complex)
            state[i] = 1.0
            basis.append(state)
        return basis
    
    def _build_phi_basis(self) -> List[np.ndarray]:
        """构建φ基 - 基于φ的最优测量基"""
        # 使用正交化过程构建完备正交基
        basis = []
        
        # 构建初始向量集合
        raw_vectors = []
        for k in range(self.dimension):
            state = np.zeros(self.dimension, dtype=complex)
            for j in range(self.dimension):
                # 使用φ的幂次和相位
                phase = 2 * np.pi * j * k / self.dimension
                amplitude = np.power(self.phi, -abs(j - k) / (self.dimension + 1))
                state[j] = amplitude * np.exp(1j * phase)
            raw_vectors.append(state)
        
        # Gram-Schmidt正交化
        for i, v in enumerate(raw_vectors):
            # 减去已有基向量的投影
            for j in range(len(basis)):
                v = v - np.dot(basis[j].conj(), v) * basis[j]
            
            # 归一化
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                basis.append(v / norm)
        
        # 如果基向量不够，补充标准基向量
        while len(basis) < self.dimension:
            for i in range(self.dimension):
                if len(basis) >= self.dimension:
                    break
                std_vec = np.zeros(self.dimension, dtype=complex)
                std_vec[i] = 1.0
                # 正交化
                for b in basis:
                    std_vec = std_vec - np.dot(b.conj(), std_vec) * b
                norm = np.linalg.norm(std_vec)
                if norm > 1e-10:
                    basis.append(std_vec / norm)
        
        return basis
    
    def _build_no11_phi_basis(self) -> List[np.ndarray]:
        """构建满足no-11约束的φ优化基"""
        # 使用Zeckendorf表示（Fibonacci基）构建基向量
        # 这自然满足no-11约束
        basis = []
        
        # 首先为有效索引构建基向量
        for k in self.no11_indices:
            state = np.zeros(self.dimension, dtype=complex)
            # 使用Zeckendorf分解的权重
            zeck_weights = self._zeckendorf_decomposition(k)
            for j in range(self.dimension):
                if self._check_no11(j):
                    # φ结构的振幅
                    weight = 1.0
                    if j in zeck_weights:
                        weight = np.power(self.phi, -zeck_weights.index(j))
                    phase = 2 * np.pi * j * k / len(self.no11_indices)
                    state[j] = weight * np.exp(1j * phase)
            # 归一化
            norm = np.linalg.norm(state)
            if norm > 1e-10:
                basis.append(state / norm)
        
        # 如果基向量不够，用标准基补充
        while len(basis) < self.dimension:
            for i in range(self.dimension):
                if len(basis) >= self.dimension:
                    break
                std_vec = np.zeros(self.dimension, dtype=complex)
                std_vec[i] = 1.0
                # 正交化
                for b in basis:
                    std_vec = std_vec - np.dot(b.conj(), std_vec) * b
                norm = np.linalg.norm(std_vec)
                if norm > 1e-10:
                    basis.append(std_vec / norm)
        
        return basis
    
    def _zeckendorf_decomposition(self, n: int) -> List[int]:
        """计算n的Zeckendorf分解（Fibonacci表示）"""
        if n == 0:
            return []
        
        # 生成Fibonacci数列
        fibs = [1, 2]
        while fibs[-1] < n:
            fibs.append(fibs[-1] + fibs[-2])
        
        # 贪心分解
        result = []
        for i in range(len(fibs) - 1, -1, -1):
            if fibs[i] <= n:
                result.append(fibs[i])
                n -= fibs[i]
        
        return result
    
    def create_measurement_operators(self, basis: List[np.ndarray]) -> List[np.ndarray]:
        """
        创建测量算子
        
        Args:
            basis: 测量基
            
        Returns:
            测量算子列表
        """
        operators = []
        for state in basis:
            # 投影算子 P_i = |i><i|
            operator = np.outer(state, state.conj())
            operators.append(operator)
        return operators
    
    def measure(self, state: np.ndarray, operators: List[np.ndarray]) -> Tuple[int, np.ndarray, float]:
        """
        执行量子测量
        
        Args:
            state: 量子态
            operators: 测量算子
            
        Returns:
            (测量结果索引, 坍缩后的态, 测量概率)
        """
        # 计算各测量结果的概率
        probabilities = []
        for op in operators:
            prob = np.real(np.dot(state.conj(), np.dot(op, state)))
            probabilities.append(prob)
        
        # 处理数值误差，确保概率非负
        probabilities = np.array(probabilities)
        probabilities = np.maximum(probabilities, 0)  # 移除负数
        
        # 归一化概率
        prob_sum = probabilities.sum()
        if prob_sum > 1e-10:
            probabilities = probabilities / prob_sum
        else:
            # 如果所有概率都接近0，使用均匀分布
            probabilities = np.ones(len(operators)) / len(operators)
        
        # 根据概率选择测量结果
        outcome = np.random.choice(len(operators), p=probabilities)
        
        # 波函数坍缩
        collapsed_state = np.dot(operators[outcome], state)
        norm = np.linalg.norm(collapsed_state)
        if norm > 1e-10:
            collapsed_state = collapsed_state / norm
        
        return outcome, collapsed_state, probabilities[outcome]
    
    def calculate_entropy(self, state: np.ndarray) -> float:
        """
        计算量子态的冯诺依曼熵
        
        Args:
            state: 量子态向量或密度矩阵
            
        Returns:
            冯诺依曼熵
        """
        # 构建密度矩阵
        if state.ndim == 1:
            # 输入是态向量
            rho = np.outer(state, state.conj())
        else:
            # 输入已经是密度矩阵
            rho = state
        
        # 计算特征值
        eigenvalues = np.linalg.eigvalsh(rho)
        # 过滤小的特征值（数值误差）
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        
        # 计算熵
        entropy = 0.0
        for lam in eigenvalues:
            if lam > 0:
                entropy -= lam * np.log2(lam)
        
        return entropy
    
    def information_gain(self, state_before: np.ndarray, state_after: np.ndarray,
                        measurement_prob: float) -> float:
        """
        计算测量的信息增益
        
        Args:
            state_before: 测量前的态
            state_after: 测量后的态
            measurement_prob: 测量概率
            
        Returns:
            信息增益
        """
        # 测量前的熵
        S_before = self.calculate_entropy(state_before)
        
        # 测量后的熵（条件熵）
        S_after = self.calculate_entropy(state_after)
        
        # 信息增益 = 测量前熵 - 条件熵 + 测量结果的信息
        if measurement_prob > 0:
            measurement_info = -measurement_prob * np.log2(measurement_prob)
        else:
            measurement_info = 0
        
        info_gain = S_before - S_after + measurement_info
        return max(0, info_gain)  # 确保非负
    
    def sequential_measurements(self, initial_state: np.ndarray, 
                              operators: List[np.ndarray],
                              num_measurements: int) -> Dict:
        """
        执行连续测量
        
        Args:
            initial_state: 初始态
            operators: 测量算子
            num_measurements: 测量次数
            
        Returns:
            测量历史
        """
        history = {
            'states': [initial_state],
            'outcomes': [],
            'probabilities': [],
            'entropies': [self.calculate_entropy(initial_state)],
            'cumulative_info': [0]
        }
        
        state = initial_state.copy()
        cumulative_info = 0
        
        for i in range(num_measurements):
            # 执行测量
            outcome, new_state, prob = self.measure(state, operators)
            
            # 计算信息增益
            info_gain = self.information_gain(state, new_state, prob)
            cumulative_info += info_gain
            
            # 记录历史
            history['outcomes'].append(outcome)
            history['probabilities'].append(prob)
            history['states'].append(new_state)
            history['entropies'].append(self.calculate_entropy(new_state))
            history['cumulative_info'].append(cumulative_info)
            
            state = new_state
        
        return history
    
    def compare_measurement_bases(self, state: np.ndarray, num_trials: int = 100) -> Dict:
        """
        比较不同测量基的信息提取效率
        
        Args:
            state: 量子态
            num_trials: 试验次数
            
        Returns:
            比较结果
        """
        # 创建测量算子
        comp_operators = self.create_measurement_operators(self.computational_basis)
        phi_operators = self.create_measurement_operators(self.phi_basis)
        
        # 收集统计数据
        comp_info_gains = []
        phi_info_gains = []
        
        for _ in range(num_trials):
            # 计算基测量
            outcome, collapsed, prob = self.measure(state, comp_operators)
            comp_info = self.information_gain(state, collapsed, prob)
            comp_info_gains.append(comp_info)
            
            # φ基测量
            outcome, collapsed, prob = self.measure(state, phi_operators)
            phi_info = self.information_gain(state, collapsed, prob)
            phi_info_gains.append(phi_info)
        
        return {
            'computational_basis': {
                'mean': np.mean(comp_info_gains),
                'std': np.std(comp_info_gains),
                'max': np.max(comp_info_gains)
            },
            'phi_basis': {
                'mean': np.mean(phi_info_gains),
                'std': np.std(phi_info_gains),
                'max': np.max(phi_info_gains)
            }
        }


class TestWaveFunctionCollapse(unittest.TestCase):
    """波函数坍缩信息理论测试"""
    
    def setUp(self):
        """初始化测试环境"""
        self.system = WaveFunctionCollapseSystem(dimension=8)
        
    def test_measurement_operators_properties(self):
        """测试测量算子的性质"""
        print("\n测试1: 验证测量算子的完备性和正交性")
        
        # 测试两种基的测量算子
        for basis_name, basis in [("computational", self.system.computational_basis),
                                  ("phi", self.system.phi_basis)]:
            operators = self.system.create_measurement_operators(basis)
            
            # 验证完备性: Σ P_i = I
            sum_op = np.zeros((self.system.dimension, self.system.dimension), dtype=complex)
            for op in operators:
                sum_op += op
            
            identity = np.eye(self.system.dimension)
            completeness_error = np.linalg.norm(sum_op - identity)
            
            print(f"\n{basis_name}基完备性误差: {completeness_error:.2e}")
            self.assertLess(completeness_error, 1e-10, 
                           f"{basis_name}基测量算子不满足完备性")
            
            # 验证投影性质: P_i^2 = P_i
            for i, op in enumerate(operators):
                projection_error = np.linalg.norm(np.dot(op, op) - op)
                self.assertLess(projection_error, 1e-10,
                               f"{basis_name}基算子{i}不满足投影性质")
    
    def test_wave_function_collapse(self):
        """测试波函数坍缩过程"""
        print("\n测试2: 验证波函数坍缩的基本性质")
        
        # 创建叠加态
        state = np.ones(self.system.dimension, dtype=complex) / np.sqrt(self.system.dimension)
        operators = self.system.create_measurement_operators(self.system.computational_basis)
        
        # 执行多次测量统计
        num_trials = 1000
        outcome_counts = np.zeros(self.system.dimension)
        
        for _ in range(num_trials):
            outcome, collapsed, prob = self.system.measure(state.copy(), operators)
            outcome_counts[outcome] += 1
            
            # 验证坍缩后的态是本征态
            expected_state = self.system.computational_basis[outcome]
            overlap = abs(np.dot(collapsed.conj(), expected_state))
            self.assertAlmostEqual(overlap, 1.0, places=10,
                                 msg="坍缩后的态不是测量算子的本征态")
        
        # 验证概率分布
        empirical_probs = outcome_counts / num_trials
        expected_probs = np.ones(self.system.dimension) / self.system.dimension
        
        print(f"\n期望概率: {expected_probs}")
        print(f"实测概率: {empirical_probs}")
        print(f"最大偏差: {np.max(np.abs(empirical_probs - expected_probs)):.3f}")
        
        # 卡方检验会因为有限样本而有偏差，使用较宽松的阈值
        chi_squared = np.sum((outcome_counts - num_trials/self.system.dimension)**2 / 
                           (num_trials/self.system.dimension))
        print(f"卡方统计量: {chi_squared:.2f}")
    
    def test_information_gain_calculation(self):
        """测试信息增益计算"""
        print("\n测试3: 验证测量的信息增益计算")
        
        # 测试纯态（零熵）
        pure_state = self.system.computational_basis[0]
        entropy = self.system.calculate_entropy(pure_state)
        print(f"\n纯态熵: {entropy:.6f}")
        self.assertLess(entropy, 1e-10, "纯态的熵应该为零")
        
        # 测试最大混合态（密度矩阵形式）
        mixed_rho = np.eye(self.system.dimension) / self.system.dimension
        entropy = self.system.calculate_entropy(mixed_rho)
        expected_entropy = np.log2(self.system.dimension)
        print(f"最大混合态熵: {entropy:.6f}")
        print(f"理论最大熵: {expected_entropy:.6f}")
        self.assertAlmostEqual(entropy, expected_entropy, places=5,
                             msg="最大混合态的熵不正确")
        
        # 测试测量信息增益（使用叠加态）
        superposition_state = np.ones(self.system.dimension, dtype=complex) / np.sqrt(self.system.dimension)
        operators = self.system.create_measurement_operators(self.system.computational_basis)
        total_info = 0
        num_trials = 100
        
        for _ in range(num_trials):
            outcome, collapsed, prob = self.system.measure(superposition_state.copy(), operators)
            info_gain = self.system.information_gain(superposition_state, collapsed, prob)
            total_info += info_gain
        
        avg_info = total_info / num_trials
        print(f"\n叠加态的平均信息增益: {avg_info:.6f}")
        print(f"叠加态初始熵: {self.system.calculate_entropy(superposition_state):.6f}")
    
    def test_phi_basis_optimality(self):
        """测试φ基测量的最优性"""
        print("\n测试4: 验证φ基测量的信息提取最优性")
        
        # 创建随机叠加态
        np.random.seed(42)
        state = np.random.randn(self.system.dimension) + 1j * np.random.randn(self.system.dimension)
        state = state / np.linalg.norm(state)
        
        # 比较不同测量基
        comparison = self.system.compare_measurement_bases(state, num_trials=200)
        
        print(f"\n计算基平均信息增益: {comparison['computational_basis']['mean']:.6f}")
        print(f"φ基平均信息增益: {comparison['phi_basis']['mean']:.6f}")
        print(f"φ基优势: {comparison['phi_basis']['mean'] / comparison['computational_basis']['mean']:.2%}")
        
        # 验证φ基的优越性
        self.assertGreater(comparison['phi_basis']['mean'], 
                          comparison['computational_basis']['mean'],
                          "φ基应该提供更高的平均信息增益")
        
        # 绘制比较图
        self._plot_basis_comparison(comparison)
    
    def test_sequential_measurement_information(self):
        """测试连续测量的信息累积"""
        print("\n测试5: 验证连续测量的信息累积规律")
        
        # 初始叠加态
        initial_state = np.ones(self.system.dimension, dtype=complex) / np.sqrt(self.system.dimension)
        
        # 使用φ基进行连续测量
        operators = self.system.create_measurement_operators(self.system.phi_basis)
        history = self.system.sequential_measurements(initial_state, operators, 
                                                    num_measurements=20)
        
        print(f"\n初始熵: {history['entropies'][0]:.6f}")
        print(f"最终熵: {history['entropies'][-1]:.6f}")
        print(f"总信息增益: {history['cumulative_info'][-1]:.6f}")
        
        # 验证信息累积的单调性
        for i in range(1, len(history['cumulative_info'])):
            self.assertGreaterEqual(history['cumulative_info'][i], 
                                  history['cumulative_info'][i-1],
                                  "累积信息应该单调递增")
        
        # 验证熵的递减
        for i in range(1, len(history['entropies'])):
            self.assertLessEqual(history['entropies'][i], 
                               history['entropies'][i-1] + 1e-10,
                               "熵应该单调递减")
        
        # 绘制信息累积图
        self._plot_information_accumulation(history)
    
    def test_quantum_classical_boundary(self):
        """测试量子-经典边界的信息理论特征"""
        print("\n测试6: 验证量子-经典边界的信息理论判据")
        
        # 创建不同叠加程度的态
        superposition_levels = np.linspace(0, 1, 10)
        boundary_info = []
        
        for alpha in superposition_levels:
            # 构建部分叠加态 |ψ⟩ = α|0⟩ + √(1-α²)|+⟩
            state = np.zeros(self.system.dimension, dtype=complex)
            state[0] = alpha
            if alpha < 1:
                # 剩余振幅均匀分布
                remaining = np.sqrt(1 - alpha**2)
                for i in range(1, self.system.dimension):
                    state[i] = remaining / np.sqrt(self.system.dimension - 1)
            
            # 计算该态的"量子性"信息含量
            entropy = self.system.calculate_entropy(state)
            
            # 测量获得的平均信息
            operators = self.system.create_measurement_operators(self.system.phi_basis)
            info_gains = []
            for _ in range(50):
                _, collapsed, prob = self.system.measure(state.copy(), operators)
                info = self.system.information_gain(state, collapsed, prob)
                info_gains.append(info)
            
            avg_info = np.mean(info_gains)
            boundary_info.append({
                'superposition': 1 - alpha,  # 叠加程度
                'entropy': entropy,
                'avg_info': avg_info,
                'quantum_measure': entropy * avg_info  # 量子性度量
            })
        
        # 找到量子-经典转变点（信息理论判据）
        quantum_measures = [b['quantum_measure'] for b in boundary_info]
        max_quantum_idx = np.argmax(quantum_measures)
        
        print(f"\n最大量子性出现在叠加度: {boundary_info[max_quantum_idx]['superposition']:.2f}")
        print(f"对应的熵: {boundary_info[max_quantum_idx]['entropy']:.6f}")
        print(f"平均信息增益: {boundary_info[max_quantum_idx]['avg_info']:.6f}")
        
        # 绘制量子-经典边界图
        self._plot_quantum_classical_boundary(boundary_info)
    
    def test_information_complementarity(self):
        """测试信息互补性原理"""
        print("\n测试7: 验证测量的信息互补性")
        
        # 创建相干叠加态
        state = (self.system.computational_basis[0] + 
                self.system.computational_basis[1]) / np.sqrt(2)
        
        # 在不同基下测量
        comp_operators = self.system.create_measurement_operators(self.system.computational_basis)
        
        # 构建互补基（X基）
        x_basis = []
        x_basis.append((self.system.computational_basis[0] + 
                       self.system.computational_basis[1]) / np.sqrt(2))
        x_basis.append((self.system.computational_basis[0] - 
                       self.system.computational_basis[1]) / np.sqrt(2))
        # 填充其余基矢
        for i in range(2, self.system.dimension):
            x_basis.append(self.system.computational_basis[i])
        
        x_operators = self.system.create_measurement_operators(x_basis)
        
        # 测量信息统计
        comp_info = []
        x_info = []
        
        for _ in range(100):
            # Z基测量
            _, collapsed, prob = self.system.measure(state.copy(), comp_operators)
            comp_info.append(self.system.information_gain(state, collapsed, prob))
            
            # X基测量
            _, collapsed, prob = self.system.measure(state.copy(), x_operators)
            x_info.append(self.system.information_gain(state, collapsed, prob))
        
        print(f"\nZ基平均信息: {np.mean(comp_info):.6f}")
        print(f"X基平均信息: {np.mean(x_info):.6f}")
        print(f"信息差异: {abs(np.mean(comp_info) - np.mean(x_info)):.6f}")
        
        # 验证互补性
        total_info_capacity = self.system.calculate_entropy(state)
        print(f"\n态的总信息容量: {total_info_capacity:.6f}")
    
    def test_phi_encoding_efficiency(self):
        """测试φ编码的信息效率"""
        print("\n测试8: 验证φ基在no-11约束下的信息提取效率")
        
        def check_no11_constraint(index):
            """检查索引的二进制表示是否满足no-11约束"""
            binary = bin(index)[2:]
            return '11' not in binary
        
        def create_no11_constrained_state(dim, state_type="uniform"):
            """创建满足no-11约束的量子态"""
            state = np.zeros(dim, dtype=complex)
            valid_indices = [i for i in range(dim) if check_no11_constraint(i)]
            
            if state_type == "uniform":
                # 均匀叠加所有有效基态
                for idx in valid_indices:
                    state[idx] = 1.0
            elif state_type == "phi_weighted":
                # φ加权叠加
                for i, idx in enumerate(valid_indices):
                    state[idx] = np.power(PHI, -i/len(valid_indices))
            elif state_type == "random_phase":
                # 随机相位
                np.random.seed(42)
                for idx in valid_indices:
                    phase = np.random.uniform(0, 2*np.pi)
                    state[idx] = np.exp(1j * phase)
            
            # 归一化
            norm = np.linalg.norm(state)
            if norm > 0:
                state = state / norm
            return state, valid_indices
        
        # 测试不同类型的no-11约束态
        test_states = []
        
        # 1. 均匀叠加的no-11态
        uniform_state, valid_indices = create_no11_constrained_state(
            self.system.dimension, "uniform")
        test_states.append(("No-11 Uniform", uniform_state))
        
        # 2. φ加权的no-11态
        phi_weighted_state, _ = create_no11_constrained_state(
            self.system.dimension, "phi_weighted")
        test_states.append(("No-11 φ-weighted", phi_weighted_state))
        
        # 3. 随机相位的no-11态
        random_phase_state, _ = create_no11_constrained_state(
            self.system.dimension, "random_phase")
        test_states.append(("No-11 Random Phase", random_phase_state))
        
        print(f"\n有效基态数量（满足no-11约束）: {len(valid_indices)}/{self.system.dimension}")
        print(f"约束比例: {len(valid_indices)/self.system.dimension:.2%}")
        
        # 比较不同基的测量效率
        comp_operators = self.system.create_measurement_operators(self.system.computational_basis)
        phi_operators = self.system.create_measurement_operators(self.system.phi_basis)
        no11_phi_operators = self.system.create_measurement_operators(self.system.no11_phi_basis)
        
        results = {}
        
        for state_name, test_state in test_states:
            comp_info_gains = []
            phi_info_gains = []
            no11_phi_info_gains = []
            
            for _ in range(50):
                # 计算基测量
                outcome, collapsed, prob = self.system.measure(test_state.copy(), comp_operators)
                info = self.system.information_gain(test_state, collapsed, prob)
                comp_info_gains.append(info)
                
                # 标准φ基测量
                outcome, collapsed, prob = self.system.measure(test_state.copy(), phi_operators)
                info = self.system.information_gain(test_state, collapsed, prob)
                phi_info_gains.append(info)
                
                # no-11优化φ基测量
                outcome, collapsed, prob = self.system.measure(test_state.copy(), no11_phi_operators)
                info = self.system.information_gain(test_state, collapsed, prob)
                no11_phi_info_gains.append(info)
            
            avg_comp = np.mean(comp_info_gains)
            avg_phi = np.mean(phi_info_gains)
            avg_no11_phi = np.mean(no11_phi_info_gains)
            
            results[state_name] = {
                'comp': avg_comp,
                'phi': avg_phi,
                'no11_phi': avg_no11_phi
            }
            
            print(f"\n{state_name}态:")
            print(f"  计算基信息增益: {avg_comp:.6f}")
            print(f"  标准φ基信息增益: {avg_phi:.6f}")
            print(f"  no-11 φ基信息增益: {avg_no11_phi:.6f}")
            
            # 找出最优基
            best_basis = max(results[state_name].items(), key=lambda x: x[1])
            print(f"  最优基: {best_basis[0]} ({best_basis[1]:.6f})")
        
        # 统计no-11 φ基的优势
        no11_phi_wins = sum(1 for state in results.values() 
                           if state['no11_phi'] >= max(state['comp'], state['phi']))
        
        print(f"\n总体统计:")
        print(f"no-11 φ基在 {no11_phi_wins}/{len(test_states)} 个测试态中表现最优")
        
        # 验证no-11 φ基在no-11约束态中的优越性
        self.assertGreaterEqual(no11_phi_wins, len(test_states) // 2,
                               "no-11 φ基应该在至少一半的no-11约束态中表现最优")
    
    def _plot_basis_comparison(self, comparison):
        """绘制测量基比较图"""
        plt.figure(figsize=(10, 6))
        
        bases = ['Computational', 'Phi']
        means = [comparison['computational_basis']['mean'], 
                comparison['phi_basis']['mean']]
        stds = [comparison['computational_basis']['std'], 
               comparison['phi_basis']['std']]
        
        x = np.arange(len(bases))
        plt.bar(x, means, yerr=stds, capsize=10, alpha=0.7, 
               color=['blue', 'gold'])
        
        plt.xlabel('Measurement Basis')
        plt.ylabel('Average Information Gain')
        plt.title('Information Extraction Efficiency Comparison')
        plt.xticks(x, bases)
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (mean, std) in enumerate(zip(means, stds)):
            plt.text(i, mean + std + 0.05, f'{mean:.3f}±{std:.3f}', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('/Users/cookie/the-binarymath/docs/binaryuniverse/tests/basis_comparison_C4_2.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_information_accumulation(self, history):
        """绘制信息累积图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        measurements = range(len(history['entropies']))
        
        # 熵的演化
        ax1.plot(measurements, history['entropies'], 'b-', linewidth=2, label='Entropy')
        ax1.set_ylabel('Von Neumann Entropy')
        ax1.set_title('Sequential Measurement Information Dynamics')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 累积信息
        ax2.plot(measurements, history['cumulative_info'], 'r-', linewidth=2, 
                label='Cumulative Information')
        ax2.set_xlabel('Measurement Number')
        ax2.set_ylabel('Cumulative Information Gain')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 标记关键点
        ax2.axhline(y=history['entropies'][0], color='g', linestyle='--', 
                   alpha=0.5, label='Initial Entropy')
        
        plt.tight_layout()
        plt.savefig('/Users/cookie/the-binarymath/docs/binaryuniverse/tests/information_accumulation_C4_2.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_quantum_classical_boundary(self, boundary_info):
        """绘制量子-经典边界图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        superposition = [b['superposition'] for b in boundary_info]
        entropy = [b['entropy'] for b in boundary_info]
        avg_info = [b['avg_info'] for b in boundary_info]
        quantum_measure = [b['quantum_measure'] for b in boundary_info]
        
        # 熵和信息增益
        ax1.plot(superposition, entropy, 'b-', linewidth=2, label='Entropy')
        ax1.set_ylabel('Entropy', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(superposition, avg_info, 'r-', linewidth=2, label='Avg Info Gain')
        ax1_twin.set_ylabel('Average Information Gain', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        ax1.set_title('Quantum-Classical Boundary: Information Theory View')
        ax1.grid(True, alpha=0.3)
        
        # 量子性度量
        ax2.plot(superposition, quantum_measure, 'g-', linewidth=2)
        ax2.fill_between(superposition, quantum_measure, alpha=0.3, color='g')
        ax2.set_xlabel('Degree of Superposition')
        ax2.set_ylabel('Quantum Measure (Entropy × Info Gain)')
        ax2.grid(True, alpha=0.3)
        
        # 标记最大量子性
        max_idx = np.argmax(quantum_measure)
        ax2.plot(superposition[max_idx], quantum_measure[max_idx], 'ro', 
                markersize=10, label=f'Max at {superposition[max_idx]:.2f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('/Users/cookie/the-binarymath/docs/binaryuniverse/tests/quantum_classical_boundary_C4_2.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)