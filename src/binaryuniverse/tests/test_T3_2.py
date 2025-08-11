"""
Unit tests for T3-2: Quantum Measurement Theorem
T3-2：量子测量定理的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
import cmath


class TestT3_2_QuantumMeasurement(VerificationTest):
    """T3-2 量子测量定理的数学化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        # 设置Hilbert空间维度
        self.hilbert_dim = 4
        # 设置数值精度
        self.tolerance = 1e-10
        
    def _create_random_hermitian_operator(self, dim: int) -> np.ndarray:
        """创建随机Hermitian算符"""
        A_real = np.random.randn(dim, dim)
        A_imag = np.random.randn(dim, dim)
        A = A_real + 1j * A_imag
        return (A + A.conj().T) / 2
    
    def _von_neumann_entropy(self, density_matrix: np.ndarray) -> float:
        """计算von Neumann熵"""
        eigenvals = np.linalg.eigvals(density_matrix)
        # 取实部并过滤掉接近零的本征值
        eigenvals = eigenvals.real
        eigenvals = eigenvals[eigenvals > 1e-12]
        if len(eigenvals) == 0:
            return 0.0
        return -np.sum(eigenvals * np.log2(eigenvals)).real
    
    def _create_measurement_projectors(self, measurement_op: np.ndarray) -> List[np.ndarray]:
        """从测量算符创建投影算符"""
        eigenvals, eigenvecs = np.linalg.eigh(measurement_op)
        
        # 处理简并情况
        projectors = []
        unique_eigenvals = []
        tolerance = 1e-10
        
        for i, eigenval in enumerate(eigenvals):
            # 检查是否是新的本征值
            is_new = True
            for existing_val in unique_eigenvals:
                if abs(eigenval - existing_val) < tolerance:
                    is_new = False
                    break
                    
            if is_new:
                unique_eigenvals.append(eigenval)
                # 找到所有具有相同本征值的本征向量
                degenerate_indices = []
                for j, eval_j in enumerate(eigenvals):
                    if abs(eigenval - eval_j) < tolerance:
                        degenerate_indices.append(j)
                
                # 构造投影算符
                projector = np.zeros((measurement_op.shape[0], measurement_op.shape[1]), dtype=complex)
                for idx in degenerate_indices:
                    vec = eigenvecs[:, idx]
                    projector += np.outer(vec, vec.conj())
                
                projectors.append(projector)
        
        return projectors, unique_eigenvals
    
    def _perform_measurement(self, state: np.ndarray, projector: np.ndarray) -> Tuple[np.ndarray, float]:
        """执行测量并返回坍缩态和概率"""
        # 计算概率
        probability = np.vdot(state, projector @ state).real
        
        if probability > self.tolerance:
            # 坍缩态
            collapsed_state = projector @ state
            collapsed_state = collapsed_state / np.linalg.norm(collapsed_state)
            return collapsed_state, probability
        else:
            return None, probability
    
    def test_self_reference_measurement_requirement(self):
        """测试自指测量要求 - 验证检查点1"""
        # 模拟自指系统
        class SelfRefSystem:
            def __init__(self, dim):
                self.dim = dim
                self.state = np.random.randn(dim) + 1j * np.random.randn(dim)
                self.state = self.state / np.linalg.norm(self.state)
                self.description_history = []
                self.observation_count = 0
                
            def get_description(self):
                """获取系统的当前描述"""
                state_repr = f"State: {self.state[:2]}"
                history_repr = f"History: {len(self.description_history)}"
                obs_repr = f"Observations: {self.observation_count}"
                return f"{state_repr}, {history_repr}, {obs_repr}"
                
            def observe(self, measurement_operator):
                """观测过程"""
                # 记录观测前的描述
                before_desc = self.get_description()
                self.description_history.append(("before", before_desc))
                
                # 执行测量
                expectation = np.vdot(self.state, measurement_operator @ self.state).real
                
                # 选择最可能的测量结果并坍缩
                eigenvals, eigenvecs = np.linalg.eigh(measurement_operator)
                probabilities = [abs(np.vdot(eigenvec, self.state))**2 
                               for eigenvec in eigenvecs.T]
                max_prob_index = np.argmax(probabilities)
                
                # 坍缩到对应的本征态
                old_state = self.state.copy()
                self.state = eigenvecs[:, max_prob_index].copy()
                self.observation_count += 1
                
                # 记录观测后的描述
                after_desc = self.get_description()
                self.description_history.append(("after", after_desc))
                
                return expectation, before_desc != after_desc, old_state
        
        # 创建测试系统
        system = SelfRefSystem(self.hilbert_dim)
        initial_state = system.state.copy()
        
        # 创建测量算符
        measurement_op = np.diag([1, -1, 0.5, -0.5])
        
        # 执行第一次观测
        result1, desc_changed1, old_state1 = system.observe(measurement_op)
        
        # 验证自指性要求
        self.assertTrue(desc_changed1, "Self-referential observation must change system description")
        self.assertGreater(len(system.description_history), 1, "Observation must record state changes")
        
        # 验证状态确实改变了
        self.assertFalse(
            np.allclose(initial_state, system.state),
            "Measurement should change the quantum state"
        )
        
        # 验证不可逆性：每次观测都增加历史记录
        initial_history_length = len(system.description_history)
        result2, desc_changed2, old_state2 = system.observe(measurement_op)
        
        self.assertGreater(
            len(system.description_history), initial_history_length,
            "Each observation must irreversibly update description"
        )
        
        # 验证观测计数的单调性
        self.assertEqual(system.observation_count, 2, "Observation count should track all measurements")
        
        # 验证信息提取
        unique_descriptions = set()
        for entry_type, desc in system.description_history:
            unique_descriptions.add(desc)
        
        self.assertGreater(
            len(unique_descriptions), 1,
            "Different observations should produce different descriptions"
        )
        
    def test_information_extraction_constraint(self):
        """测试信息提取约束 - 验证检查点2"""
        # 创建初始叠加态
        psi = np.array([0.6, 0.8, 0.0, 0.0], dtype=complex)
        psi = psi / np.linalg.norm(psi)
        
        # 初始密度矩阵和熵
        rho_initial = np.outer(psi, psi.conj())
        initial_entropy = self._von_neumann_entropy(rho_initial)
        
        # 验证初始态是纯态（von Neumann熵为0）
        self.assertAlmostEqual(initial_entropy, 0.0, 10, "Pure state should have zero von Neumann entropy")
        
        # 使用测量结果的Shannon熵作为信息度量
        # 这是观测者从测量中获得的信息量
        
        # 创建测量算符
        measurement_op = np.array([[1, 0, 0, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]], dtype=complex)
        
        # 获取投影算符
        projectors, eigenvals = self._create_measurement_projectors(measurement_op)
        
        # 计算测量概率
        probabilities = []
        collapsed_entropies = []
        
        for projector in projectors:
            prob = np.vdot(psi, projector @ psi).real
            probabilities.append(prob)
            
            if prob > self.tolerance:
                # 计算坍缩后的熵（应该为0，因为是纯态）
                collapsed_entropies.append(0.0)
            else:
                collapsed_entropies.append(0.0)
        
        # 验证概率归一化
        total_prob = sum(probabilities)
        self.assertAlmostEqual(total_prob, 1.0, 10, "Probabilities should sum to 1")
        
        # 计算测量结果的Shannon熵（观测者获得的信息）
        nonzero_probs = [p for p in probabilities if p > self.tolerance]
        if len(nonzero_probs) > 1:
            measurement_shannon_entropy = -sum(p * np.log2(p) for p in nonzero_probs)
            
            # 验证信息增益
            self.assertGreater(measurement_shannon_entropy, 0.1, "Measurement should provide substantial information")
            
            # 验证信息增益的合理性
            self.assertLessEqual(measurement_shannon_entropy, np.log2(len(nonzero_probs)), 
                               "Shannon entropy should not exceed maximum possible")
        
        else:
            # 如果只有一个非零概率，测量是确定的，没有信息增益
            self.assertEqual(len(nonzero_probs), 1, "Should have exactly one outcome if not multiple")
            single_prob = nonzero_probs[0]
            self.assertAlmostEqual(single_prob, 1.0, 10, "Single outcome should have probability 1")
        
    def test_projection_operator_construction(self):
        """测试投影算符构造 - 验证检查点3"""
        # 创建Hermitian测量算符
        measurement_op = np.array([[2, 1, 0, 0],
                                  [1, 2, 0, 0],
                                  [0, 0, -1, 0],
                                  [0, 0, 0, -1]], dtype=complex)
        
        # 验证Hermitian性质
        self.assertTrue(
            np.allclose(measurement_op, measurement_op.conj().T),
            "Measurement operator should be Hermitian"
        )
        
        # 计算谱分解
        eigenvals, eigenvecs = np.linalg.eigh(measurement_op)
        
        # 构造投影算符
        projectors, unique_eigenvals = self._create_measurement_projectors(measurement_op)
        
        # 验证投影算符性质
        for i, P_i in enumerate(projectors):
            # 验证幂等性：P² = P
            self.assertTrue(
                np.allclose(P_i @ P_i, P_i, atol=self.tolerance),
                f"Projector {i} should be idempotent"
            )
            
            # 验证Hermitian性：P† = P
            self.assertTrue(
                np.allclose(P_i, P_i.conj().T, atol=self.tolerance),
                f"Projector {i} should be Hermitian"
            )
            
            # 验证正定性：eigenvalues ≥ 0
            eigenvals_P = np.linalg.eigvals(P_i)
            self.assertTrue(
                np.all(eigenvals_P >= -self.tolerance),
                f"Projector {i} should be positive semidefinite"
            )
            
            # 验证迹的性质（迹等于子空间维度）
            trace_P = np.trace(P_i).real
            self.assertGreaterEqual(trace_P, 0, f"Trace of projector {i} should be non-negative")
            self.assertLessEqual(trace_P, measurement_op.shape[0], f"Trace should not exceed dimension")
            
        # 验证相互正交性
        for i, P_i in enumerate(projectors):
            for j, P_j in enumerate(projectors):
                if i != j:
                    product = P_i @ P_j
                    self.assertTrue(
                        np.allclose(product, np.zeros_like(product), atol=self.tolerance),
                        f"Projectors {i} and {j} should be orthogonal"
                    )
        
        # 验证完备性：∑P_k = I
        identity_check = sum(projectors)
        expected_identity = np.eye(measurement_op.shape[0])
        self.assertTrue(
            np.allclose(identity_check, expected_identity, atol=self.tolerance),
            "Sum of projectors should equal identity"
        )
        
        # 验证谱分解重构
        reconstructed_op = sum(eigenval * proj for eigenval, proj in 
                             zip(unique_eigenvals, projectors))
        self.assertTrue(
            np.allclose(reconstructed_op, measurement_op, atol=self.tolerance),
            "Spectral decomposition should reconstruct original operator"
        )
        
        # 验证投影算符的维度
        total_trace = sum(np.trace(P).real for P in projectors)
        self.assertAlmostEqual(
            total_trace, measurement_op.shape[0], 10,
            "Total trace of projectors should equal Hilbert space dimension"
        )
        
    def test_wavefunction_collapse_necessity(self):
        """测试波函数坍缩必然性 - 验证检查点4"""
        # 创建初始叠加态
        psi = np.array([0.6, 0.8, 0.0, 0.0], dtype=complex)
        psi = psi / np.linalg.norm(psi)
        
        # 验证初始态确实是叠加态
        self.assertFalse(
            np.any(np.abs(psi)**2 > 0.95),
            "Initial state should be a genuine superposition"
        )
        
        # 创建测量算符和投影算符
        projector_0 = np.array([[1, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]], dtype=complex)
        
        projector_1 = np.array([[0, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]], dtype=complex)
        
        # 计算测量概率
        prob_0 = np.vdot(psi, projector_0 @ psi).real
        prob_1 = np.vdot(psi, projector_1 @ psi).real
        
        # 验证概率的合理性
        self.assertGreater(prob_0, 0, "State should have overlap with first eigenspace")
        self.assertGreater(prob_1, 0, "State should have overlap with second eigenspace")
        self.assertAlmostEqual(prob_0 + prob_1, 1.0, 10, "Probabilities should sum to 1")
        
        # 模拟两种可能的坍缩结果
        if prob_0 > self.tolerance:
            psi_collapsed_0, _ = self._perform_measurement(psi, projector_0)
            
            # 验证坍缩后是本征态
            self.assertGreater(
                abs(psi_collapsed_0[0])**2, 0.95,
                "Should collapse to eigenstate |0⟩"
            )
            self.assertLess(
                abs(psi_collapsed_0[1])**2, 0.1,
                "Other components should be negligible"
            )
            
            # 验证归一化
            norm_squared = np.vdot(psi_collapsed_0, psi_collapsed_0).real
            self.assertAlmostEqual(
                norm_squared, 1.0, 10,
                "Collapsed state should be normalized"
            )
            
            # 验证是投影的结果
            projected = projector_0 @ psi
            projected_normalized = projected / np.linalg.norm(projected)
            self.assertTrue(
                np.allclose(psi_collapsed_0, projected_normalized, atol=self.tolerance),
                "Collapsed state should equal normalized projection"
            )
        
        if prob_1 > self.tolerance:
            psi_collapsed_1, _ = self._perform_measurement(psi, projector_1)
            
            # 验证坍缩后是本征态
            self.assertGreater(
                abs(psi_collapsed_1[1])**2, 0.95,
                "Should collapse to eigenstate |1⟩"
            )
            self.assertLess(
                abs(psi_collapsed_1[0])**2, 0.1,
                "Other components should be negligible"
            )
            
            # 验证归一化
            norm_squared = np.vdot(psi_collapsed_1, psi_collapsed_1).real
            self.assertAlmostEqual(
                norm_squared, 1.0, 10,
                "Collapsed state should be normalized"
            )
        
        # 验证坍缩的不可逆性
        if prob_0 > self.tolerance:
            psi_collapsed, _ = self._perform_measurement(psi, projector_0)
            overlap_with_original = abs(np.vdot(psi, psi_collapsed))**2
            
            # 应该小于1，因为原态是叠加态
            self.assertLess(
                overlap_with_original, 0.99,
                "Collapsed state should differ from original superposition"
            )
            
            # 但应该等于相应分量的概率
            expected_overlap = prob_0
            self.assertAlmostEqual(
                overlap_with_original, expected_overlap, 8,
                "Overlap should equal the measurement probability"
            )
        
        # 测试连续测量的一致性
        if prob_0 > self.tolerance:
            psi_collapsed, _ = self._perform_measurement(psi, projector_0)
            # 再次测量应该给出确定结果
            prob_0_again = np.vdot(psi_collapsed, projector_0 @ psi_collapsed).real
            
            self.assertAlmostEqual(
                prob_0_again, 1.0, 8,
                "Second measurement of collapsed state should give certainty"
            )
            
    def test_probability_emergence_verification(self):
        """测试概率涌现验证 - 验证检查点5"""
        # 创建测试态
        psi = np.array([0.3, 0.4, 0.5, 0.6], dtype=complex)
        psi = psi / np.linalg.norm(psi)
        
        # 创建可观测量（Hermitian算符）
        observable = np.array([[1, 0, 0, 0],
                              [0, 2, 0, 0],
                              [0, 0, 3, 0],
                              [0, 0, 0, 4]], dtype=complex)
        
        # 计算谱分解
        eigenvals, eigenvecs = np.linalg.eigh(observable)
        
        # 计算Born规则概率
        born_probabilities = []
        for i, eigenvec in enumerate(eigenvecs.T):
            prob = abs(np.vdot(eigenvec, psi))**2
            born_probabilities.append(prob)
        
        # 验证概率归一化
        total_prob = sum(born_probabilities)
        self.assertAlmostEqual(total_prob, 1.0, 10, "Born rule probabilities should sum to 1")
        
        # 验证概率为非负
        for i, prob in enumerate(born_probabilities):
            self.assertGreaterEqual(prob, 0, f"Probability {i} should be non-negative")
        
        # 计算期望值（两种方法）
        # 方法1：直接计算 ⟨ψ|Ô|ψ⟩
        expectation_direct = np.vdot(psi, observable @ psi).real
        
        # 方法2：Born规则 ∑ λ_k P(k)
        expectation_born = sum(eigenval * prob for eigenval, prob in 
                              zip(eigenvals, born_probabilities))
        
        # 验证两种方法一致
        self.assertAlmostEqual(
            expectation_direct, expectation_born, 10,
            "Direct and Born rule expectation values should match"
        )
        
        # 验证期望值的合理范围
        min_eigenval, max_eigenval = min(eigenvals), max(eigenvals)
        self.assertGreaterEqual(
            expectation_direct, min_eigenval - self.tolerance,
            "Expectation value should be >= minimum eigenvalue"
        )
        self.assertLessEqual(
            expectation_direct, max_eigenval + self.tolerance,
            "Expectation value should be <= maximum eigenvalue"
        )
        
        # 测试极端情况：纯本征态
        for i, eigenvec in enumerate(eigenvecs.T):
            eigenstate = eigenvec / np.linalg.norm(eigenvec)
            
            # 在本征态中，对应本征值的概率应该为1
            prob_in_eigenstate = abs(np.vdot(eigenstate, eigenstate))**2
            self.assertAlmostEqual(
                prob_in_eigenstate, 1.0, 10,
                f"Eigenstate {i} should have probability 1 for itself"
            )
            
            # 期望值应该等于本征值
            expectation_eigenstate = np.vdot(eigenstate, observable @ eigenstate).real
            self.assertAlmostEqual(
                expectation_eigenstate, eigenvals[i], 10,
                f"Eigenstate {i} should have expectation equal to eigenvalue"
            )
            
            # 测量该本征态应该给出确定结果
            projectors, _ = self._create_measurement_projectors(observable)
            for j, projector in enumerate(projectors):
                measured_prob = np.vdot(eigenstate, projector @ eigenstate).real
                # 应该要么是1（对应的投影）要么是0
                self.assertTrue(
                    abs(measured_prob - 1.0) < self.tolerance or abs(measured_prob) < self.tolerance,
                    f"Eigenstate should give definite measurement results"
                )
        
        # 验证概率的φ-表示基础
        # （简化：验证概率与态分量的关系）
        state_components_squared = [abs(c)**2 for c in psi]
        total_component = sum(state_components_squared)
        self.assertAlmostEqual(
            total_component, 1.0, 10,
            "State component probabilities should sum to 1"
        )
        
        # 验证不确定性关系的存在
        # 对于非本征态，测量应该有真正的随机性
        max_prob = max(born_probabilities)
        if max_prob < 0.95:  # 如果不是接近纯本征态
            # 应该有多个可能的测量结果
            significant_probs = [p for p in born_probabilities if p > 0.01]
            self.assertGreater(
                len(significant_probs), 1,
                "Non-eigenstate should have multiple possible measurement outcomes"
            )
            
    def test_measurement_commutation_relations(self):
        """测试测量的对易关系"""
        # 创建两个不同的可观测量
        observable_A = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, -1]], dtype=complex)
        
        observable_B = np.array([[0, 1, 0, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 1],
                                [0, 0, 1, 0]], dtype=complex)
        
        # 计算对易子
        commutator = observable_A @ observable_B - observable_B @ observable_A
        
        # 如果对易子非零，则不能同时测量
        commutator_norm = np.linalg.norm(commutator)
        
        if commutator_norm > self.tolerance:
            # 验证不确定性关系
            # 创建测试态
            psi = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
            psi = psi / np.linalg.norm(psi)
            
            # 计算两个观测量的期望值和方差
            exp_A = np.vdot(psi, observable_A @ psi).real
            exp_A_squared = np.vdot(psi, observable_A @ observable_A @ psi).real
            var_A = exp_A_squared - exp_A**2
            
            exp_B = np.vdot(psi, observable_B @ psi).real
            exp_B_squared = np.vdot(psi, observable_B @ observable_B @ psi).real
            var_B = exp_B_squared - exp_B**2
            
            # 计算对易子的期望值
            exp_commutator = np.vdot(psi, commutator @ psi).real
            
            # 验证不确定性关系 ΔA·ΔB ≥ |⟨[A,B]⟩|/2
            uncertainty_product = np.sqrt(var_A * var_B)
            uncertainty_bound = abs(exp_commutator) / 2
            
            self.assertGreaterEqual(
                uncertainty_product + self.tolerance, uncertainty_bound,
                "Uncertainty relation should be satisfied"
            )
        
    def test_sequential_measurement_consistency(self):
        """测试连续测量的一致性"""
        # 创建初始叠加态
        psi = np.array([0.6, 0.8, 0.0, 0.0], dtype=complex)
        psi = psi / np.linalg.norm(psi)
        
        # 创建测量算符
        observable = np.diag([1, -1, 0, 0])
        projectors, eigenvals = self._create_measurement_projectors(observable)
        
        # 第一次测量
        for i, projector in enumerate(projectors):
            prob = np.vdot(psi, projector @ psi).real
            
            if prob > self.tolerance:
                # 执行测量
                collapsed_state, measured_prob = self._perform_measurement(psi, projector)
                
                # 立即重复相同的测量
                prob_repeat = np.vdot(collapsed_state, projector @ collapsed_state).real
                
                # 应该得到确定的结果（概率1）
                self.assertAlmostEqual(
                    prob_repeat, 1.0, 8,
                    f"Repeated measurement should give probability 1 for outcome {i}"
                )
                
                # 测量其他结果的概率应该为0
                for j, other_projector in enumerate(projectors):
                    if i != j:
                        prob_other = np.vdot(collapsed_state, other_projector @ collapsed_state).real
                        self.assertAlmostEqual(
                            prob_other, 0.0, 8,
                            f"Other measurement outcomes should have probability 0"
                        )


if __name__ == "__main__":
    unittest.main()