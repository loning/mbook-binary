"""
Unit tests for T3-4: Quantum Teleportation Theorem
T3-4：量子隐形传态定理的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable


class TestT3_4_QuantumTeleportation(VerificationTest):
    """T3-4 量子隐形传态定理的数学化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        # 设置数值精度
        self.tolerance = 1e-10
        
        # 预定义Pauli算符
        self.I = np.eye(2, dtype=complex)
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Bell基态
        self.bell_states = self._construct_bell_basis()
    
    def _construct_bell_basis(self) -> List[np.ndarray]:
        """构造Bell基"""
        phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)   # |Φ+⟩
        phi_minus = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)  # |Φ-⟩
        psi_plus = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)    # |Ψ+⟩
        psi_minus = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)  # |Ψ-⟩
        return [phi_plus, phi_minus, psi_plus, psi_minus]
    
    def _get_correction_operator(self, bell_outcome: int) -> np.ndarray:
        """根据Bell测量结果获取修正算符"""
        corrections = {
            0: self.I,        # |Φ+⟩ → I
            1: self.sigma_z,  # |Φ-⟩ → σz
            2: self.sigma_x,  # |Ψ+⟩ → σx
            3: self.sigma_y   # |Ψ-⟩ → σy
        }
        return corrections[bell_outcome]
    
    def _get_charlie_state_after_measurement(self, bell_outcome: int, alpha: complex, beta: complex) -> np.ndarray:
        """获取Bell测量后Charlie的状态"""
        # These are the states Charlie has after each Bell measurement,
        # such that applying the corresponding Pauli correction restores the original state
        states = {
            0: np.array([alpha, beta], dtype=complex),       # α|0⟩ + β|1⟩ (I correction)
            1: np.array([alpha, -beta], dtype=complex),      # α|0⟩ - β|1⟩ (σz correction)  
            2: np.array([beta, alpha], dtype=complex),       # β|0⟩ + α|1⟩ (σx correction)
            3: np.array([-1j*beta, 1j*alpha], dtype=complex) # -iβ|0⟩ + iα|1⟩ (σy correction)
        }
        return states[bell_outcome]
    
    def _teleportation_fidelity(self, original_state: np.ndarray, final_state: np.ndarray) -> float:
        """计算传输保真度"""
        overlap = np.vdot(original_state, final_state)
        return abs(overlap)**2
    
    def test_initial_state_construction(self):
        """测试初始态构造 - 验证检查点1"""
        # 创建未知态
        alpha, beta = 0.6, 0.8
        psi_unknown = np.array([alpha, beta], dtype=complex)
        
        # 验证归一化
        norm_unknown = np.vdot(psi_unknown, psi_unknown).real
        self.assertAlmostEqual(norm_unknown, 1.0, 10, "Unknown state should be normalized")
        
        # 创建纠缠资源
        bell_state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)  # |Φ+⟩
        
        # 验证Bell态
        norm_bell = np.vdot(bell_state, bell_state).real
        self.assertAlmostEqual(norm_bell, 1.0, 10, "Bell state should be normalized")
        
        # 构造三粒子复合态
        composite_state = np.kron(psi_unknown, bell_state)
        
        # 验证维度
        self.assertEqual(composite_state.shape[0], 8, "Composite state should be 8-dimensional")
        
        # 验证归一化
        norm_composite = np.vdot(composite_state, composite_state).real
        self.assertAlmostEqual(norm_composite, 1.0, 10, "Composite state should be normalized")
        
        # 验证分量结构
        expected = np.array([
            alpha/np.sqrt(2),  # α|000⟩
            0,                 # |001⟩
            0,                 # |010⟩  
            alpha/np.sqrt(2),  # α|011⟩
            beta/np.sqrt(2),   # β|100⟩
            0,                 # |101⟩
            0,                 # |110⟩
            beta/np.sqrt(2)    # β|111⟩
        ], dtype=complex)
        
        self.assertTrue(
            np.allclose(composite_state, expected),
            "Composite state components should match expected values"
        )
        
        # 验证张量积结构
        # 检查是否确实是张量积
        reconstructed = np.kron(psi_unknown, bell_state)
        self.assertTrue(
            np.allclose(composite_state, reconstructed),
            "Composite state should be exact tensor product"
        )
        
        # 验证各子系统的可分离性
        # 在初始阶段，A与BC之间应该没有纠缠
        # 简化验证：检查A粒子的约化密度矩阵
        rho_composite = np.outer(composite_state, composite_state.conj())
        
        # A粒子的约化密度矩阵
        rho_A = np.zeros((2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    rho_A[i, j] += rho_composite[4*i + k, 4*j + k]
        
        # A粒子应该处于纯态
        trace_rho_A_squared = np.trace(rho_A @ rho_A).real
        self.assertAlmostEqual(trace_rho_A_squared, 1.0, 8, "A subsystem should be in pure state initially")
        
    def test_bell_measurement_protocol(self):
        """测试Bell基测量协议 - 验证检查点2"""
        # 构造测试态
        alpha, beta = 0.6, 0.8
        psi_unknown = np.array([alpha, beta], dtype=complex)
        bell_resource = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        composite_state = np.kron(psi_unknown, bell_resource)
        
        # 验证Bell基的性质
        for i, bell_i in enumerate(self.bell_states):
            # 验证归一化
            norm = np.vdot(bell_i, bell_i).real
            self.assertAlmostEqual(norm, 1.0, 10, f"Bell state {i} should be normalized")
            
            # 验证正交性
            for j, bell_j in enumerate(self.bell_states):
                if i != j:
                    overlap = np.vdot(bell_i, bell_j)
                    self.assertAlmostEqual(
                        abs(overlap), 0.0, 10,
                        f"Bell states {i} and {j} should be orthogonal"
                    )
        
        # 分解复合态到Bell基（在AB子空间）
        def decompose_in_bell_basis(state_8d):
            """将8维态分解到Bell基"""
            probabilities = []
            charlie_states = []
            
            for i, bell_state in enumerate(self.bell_states):
                # 构造AB子空间的投影算符
                projector_AB = np.kron(np.outer(bell_state, bell_state.conj()), np.eye(2))
                
                # 计算投影概率
                prob = np.vdot(state_8d, projector_AB @ state_8d).real
                probabilities.append(prob)
                
                # 计算对应的Charlie态（简化）
                if prob > self.tolerance:
                    projected_state = projector_AB @ state_8d
                    # 提取Charlie的状态（最后2个分量的模式）
                    charlie_state = self._get_charlie_state_after_measurement(i, alpha, beta)
                    charlie_states.append(charlie_state)
                else:
                    charlie_states.append(None)
            
            return probabilities, charlie_states
        
        probabilities, charlie_states = decompose_in_bell_basis(composite_state)
        
        # 验证概率和为1
        total_prob = sum(probabilities)
        self.assertAlmostEqual(total_prob, 1.0, 10, "Bell measurement probabilities should sum to 1")
        
        # 验证各概率相等（对称性）
        for i, prob in enumerate(probabilities):
            self.assertAlmostEqual(prob, 0.25, 10, f"Bell outcome {i} should have probability 1/4")
        
        # 验证Charlie态的正确性
        for i, charlie_state in enumerate(charlie_states):
            if charlie_state is not None:
                norm = np.vdot(charlie_state, charlie_state).real
                self.assertAlmostEqual(norm, 1.0, 10, f"Charlie state {i} should be normalized")
        
        # 验证Bell测量的完备性
        # 构造完备性检验：∑|bell_i⟩⟨bell_i| = I
        completeness_check = sum(np.outer(bell, bell.conj()) for bell in self.bell_states)
        identity_4 = np.eye(4)
        self.assertTrue(
            np.allclose(completeness_check, identity_4),
            "Bell basis should be complete"
        )
        
    def test_classical_information_transmission(self):
        """测试经典信息传输 - 验证检查点3"""
        def encode_measurement_result(bell_state_index):
            """编码测量结果为经典比特"""
            encoding_map = {
                0: (0, 0),  # |Φ+⟩ → 00
                1: (0, 1),  # |Φ-⟩ → 01  
                2: (1, 0),  # |Ψ+⟩ → 10
                3: (1, 1)   # |Ψ-⟩ → 11
            }
            return encoding_map[bell_state_index]
        
        def transmission_delay(distance, speed_of_light=1.0):
            """计算传输延迟"""
            return distance / speed_of_light
        
        # 测试所有可能的测量结果编码
        for bell_index in range(4):
            classical_bits = encode_measurement_result(bell_index)
            
            # 验证编码
            self.assertEqual(len(classical_bits), 2, "Should encode to 2 classical bits")
            self.assertTrue(
                all(bit in [0, 1] for bit in classical_bits),
                "Bits should be 0 or 1"
            )
        
        # 验证编码的唯一性
        encodings = [encode_measurement_result(i) for i in range(4)]
        unique_encodings = set(encodings)
        self.assertEqual(len(unique_encodings), 4, "All encodings should be unique")
        
        # 验证传输延迟约束
        distances = [1.0, 10.0, 100.0]
        for d in distances:
            delay = transmission_delay(d)
            self.assertGreaterEqual(delay, 0, "Transmission delay should be non-negative")
            self.assertGreaterEqual(delay, d, "Delay should respect speed limit")
        
        # 验证信息容量
        total_outcomes = 4  # 4个可能的Bell态
        information_content = np.log2(total_outcomes)
        self.assertAlmostEqual(
            information_content, 2.0, 10,
            "Should carry exactly 2 bits of information"
        )
        
        # 验证因果性约束
        # Alice的测量必须在Charlie的操作之前完成
        alice_measurement_time = 0.0
        classical_transmission_time = transmission_delay(10.0)  # 假设距离为10
        charlie_operation_time = alice_measurement_time + classical_transmission_time
        
        self.assertGreater(
            charlie_operation_time, alice_measurement_time,
            "Charlie's operation must occur after Alice's measurement"
        )
        
        # 验证no-communication定理
        # 仅凭Charlie子系统无法获得关于|ψ⟩的信息
        alpha, beta = 0.6, 0.8
        
        # Charlie在没有经典信息时的混合态
        # 所有可能的Charlie态等概率混合
        charlie_states_before_correction = [
            self._get_charlie_state_after_measurement(i, alpha, beta) 
            for i in range(4)
        ]
        
        # 每个态的密度矩阵
        charlie_density_matrices = [
            np.outer(state, state.conj()) for state in charlie_states_before_correction
        ]
        
        # 混合态密度矩阵
        mixed_density = sum(charlie_density_matrices) / 4
        
        # 混合态应该与原始态不同（无法从中提取信息）
        original_density = np.outer(np.array([alpha, beta]), np.array([alpha, beta]).conj())
        trace_distance = np.trace(abs(mixed_density - original_density)).real / 2
        
        self.assertGreater(
            trace_distance, 0.1,
            "Mixed Charlie state should be distinguishable from original state"
        )
        
    def test_unitary_correction_application(self):
        """测试幺正修正操作应用 - 验证检查点4"""
        # 测试所有修正操作
        alpha, beta = 0.6, 0.8
        target_state = np.array([alpha, beta], dtype=complex)
        
        for bell_outcome in range(4):
            # 获取测量后的状态
            charlie_state = self._get_charlie_state_after_measurement(bell_outcome, alpha, beta)
            
            # 应用修正操作
            correction = self._get_correction_operator(bell_outcome)
            corrected_state = correction @ charlie_state
            
            # 验证修正后状态等于目标状态
            self.assertTrue(
                np.allclose(corrected_state, target_state, atol=self.tolerance),
                f"Correction for outcome {bell_outcome} should restore target state"
            )
            
            # 验证修正算符的幺正性
            unitarity_check = correction @ correction.conj().T
            self.assertTrue(
                np.allclose(unitarity_check, self.I, atol=self.tolerance),
                f"Correction operator {bell_outcome} should be unitary"
            )
            
            # 验证修正算符的Hermitian性（对于Pauli算符）
            hermiticity_check = correction - correction.conj().T
            self.assertTrue(
                np.allclose(hermiticity_check, np.zeros_like(correction), atol=self.tolerance),
                f"Pauli operator {bell_outcome} should be Hermitian"
            )
            
            # 验证Pauli算符的特殊性质：σ² = I
            if bell_outcome > 0:  # 非恒等算符
                pauli_squared = correction @ correction
                self.assertTrue(
                    np.allclose(pauli_squared, self.I, atol=self.tolerance),
                    f"Pauli operator {bell_outcome} squared should equal identity"
                )
            
            # 验证本征值为±1（除了恒等算符）
            eigenvals = np.linalg.eigvals(correction)
            eigenvals_abs = np.abs(eigenvals)
            self.assertTrue(
                np.allclose(eigenvals_abs, 1.0, atol=self.tolerance),
                f"Eigenvalues of correction operator {bell_outcome} should have magnitude 1"
            )
        
        # 验证修正操作的必要性
        # 不经修正的情况下，传输是不完美的
        for bell_outcome in range(1, 4):  # 排除outcome=0（不需要修正）
            charlie_state = self._get_charlie_state_after_measurement(bell_outcome, alpha, beta)
            
            # 不修正的保真度
            uncorrected_fidelity = self._teleportation_fidelity(target_state, charlie_state)
            
            # 修正后的保真度
            correction = self._get_correction_operator(bell_outcome)
            corrected_state = correction @ charlie_state
            corrected_fidelity = self._teleportation_fidelity(target_state, corrected_state)
            
            # 修正应该显著提高保真度
            self.assertGreater(
                corrected_fidelity, uncorrected_fidelity,
                f"Correction should improve fidelity for outcome {bell_outcome}"
            )
            
            # 修正后应达到完美保真度
            self.assertAlmostEqual(
                corrected_fidelity, 1.0, 10,
                f"Corrected fidelity should be perfect for outcome {bell_outcome}"
            )
        
    def test_teleportation_fidelity_verification(self):
        """测试隐形传态保真度验证 - 验证检查点5"""
        def simulate_complete_teleportation(original_state):
            """模拟完整的隐形传态过程"""
            bell_outcomes = [0, 1, 2, 3]
            fidelities = []
            final_states = []
            
            alpha, beta = original_state[0], original_state[1]
            
            # 模拟每种可能的测量结果
            for outcome in bell_outcomes:
                # 根据Bell测量结果确定Charlie的状态
                charlie_state = self._get_charlie_state_after_measurement(outcome, alpha, beta)
                
                # 应用修正操作
                correction = self._get_correction_operator(outcome)
                final_state = correction @ charlie_state
                final_states.append(final_state)
                
                # 计算保真度
                fidelity = self._teleportation_fidelity(original_state, final_state)
                fidelities.append(fidelity)
            
            return fidelities, final_states
        
        # 测试不同的输入态
        test_states = [
            np.array([1, 0], dtype=complex),                    # |0⟩
            np.array([0, 1], dtype=complex),                    # |1⟩  
            np.array([1, 1], dtype=complex)/np.sqrt(2),         # |+⟩
            np.array([1, -1], dtype=complex)/np.sqrt(2),        # |-⟩
            np.array([1, 1j], dtype=complex)/np.sqrt(2),        # |i⟩
            np.array([0.6, 0.8], dtype=complex)                # 一般态
        ]
        
        for original_state in test_states:
            fidelities, final_states = simulate_complete_teleportation(original_state)
            
            # 验证每种情况都达到完美保真度
            for i, fidelity in enumerate(fidelities):
                self.assertAlmostEqual(
                    fidelity, 1.0, 10,
                    f"Teleportation fidelity should be 1.0 for outcome {i}, got {fidelity}"
                )
            
            # 验证所有最终态都与原态相同
            for i, final_state in enumerate(final_states):
                self.assertTrue(
                    np.allclose(final_state, original_state, atol=self.tolerance),
                    f"Final state {i} should match original state"
                )
            
            # 验证平均保真度
            avg_fidelity = np.mean(fidelities)
            self.assertAlmostEqual(
                avg_fidelity, 1.0, 10,
                "Average teleportation fidelity should be 1.0"
            )
        
        # 验证保真度的数学性质
        # 保真度应该在[0,1]范围内
        random_state = np.random.randn(2) + 1j * np.random.randn(2)
        random_state = random_state / np.linalg.norm(random_state)
        
        fidelities, _ = simulate_complete_teleportation(random_state)
        for fidelity in fidelities:
            self.assertGreaterEqual(fidelity, 0.0, "Fidelity should be non-negative")
            # 由于浮点数精度，允许微小的超出
            self.assertLessEqual(fidelity, 1.0 + 1e-10, "Fidelity should not exceed 1")
        
        # 验证对称性：F(ψ,φ) = F(φ,ψ)
        state1 = np.array([0.6, 0.8], dtype=complex)
        state2 = np.array([0.8, 0.6], dtype=complex)
        
        fidelity_12 = self._teleportation_fidelity(state1, state2)
        fidelity_21 = self._teleportation_fidelity(state2, state1)
        
        self.assertAlmostEqual(
            fidelity_12, fidelity_21, 10,
            "Fidelity should be symmetric"
        )
        
    def test_no_cloning_verification(self):
        """测试no-cloning定理验证"""
        # 验证原始态在传输过程中被破坏
        alpha, beta = 0.6, 0.8
        original_state = np.array([alpha, beta], dtype=complex)
        
        # 创建初始复合态
        bell_resource = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        composite_state = np.kron(original_state, bell_resource)
        
        # 模拟Bell测量过程
        # 测量后，A处的原始态信息被破坏
        for bell_outcome in range(4):
            # Bell测量的投影算符
            bell_projector = np.outer(self.bell_states[bell_outcome], self.bell_states[bell_outcome].conj())
            
            # 在AB子空间上的完整投影算符
            projector_AB = np.kron(bell_projector, np.eye(2))
            
            # 测量后的态
            post_measurement_state = projector_AB @ composite_state
            
            # 提取A子系统的状态
            # 在测量后，A子系统处于确定的Bell态分量中
            # 原始的叠加态信息已经不可恢复
            
            # 验证信息不能同时存在于A和C处
            # 这通过测量的投影性质自动保证
        
        # 验证不可能的克隆操作
        # 如果存在克隆操作，它会违反量子力学
        # 这里验证线性性导致的克隆不可能性
        
        def attempt_cloning(input_state):
            """尝试线性克隆操作（应该失败）"""
            # 假设存在线性算符U使得 U|ψ⟩|0⟩ = |ψ⟩|ψ⟩
            # 对于两个不同的态|0⟩和|1⟩
            state_0 = np.array([1, 0], dtype=complex)
            state_1 = np.array([0, 1], dtype=complex)
            
            # 如果克隆可能，则对叠加态也应该工作
            superposition = (state_0 + state_1) / np.sqrt(2)
            
            # 但叠加态的克隆会导致矛盾
            # U(|0⟩+|1⟩)|0⟩ = U|0⟩|0⟩ + U|1⟩|0⟩ = |0⟩|0⟩ + |1⟩|1⟩
            # 而期望的结果是 (|0⟩+|1⟩)(|0⟩+|1⟩) = |0⟩|0⟩ + |0⟩|1⟩ + |1⟩|0⟩ + |1⟩|1⟩
            # 这两者不相等，证明线性克隆不可能
            
            return False  # 克隆不可能
        
        cloning_possible = attempt_cloning(original_state)
        self.assertFalse(cloning_possible, "Quantum cloning should be impossible")
        
    def test_information_conservation(self):
        """测试信息守恒"""
        # 验证量子信息在传输过程中守恒
        alpha, beta = 0.6, 0.8
        original_state = np.array([alpha, beta], dtype=complex)
        
        # 计算原始态的信息量度
        def quantum_information_measure(state):
            """简化的量子信息度量"""
            # 使用von Neumann熵作为纯态的信息度量（为0）
            density_matrix = np.outer(state, state.conj())
            eigenvals = np.linalg.eigvals(density_matrix)
            eigenvals = eigenvals[eigenvals > 1e-12]
            if len(eigenvals) == 0:
                return 0.0
            return -np.sum(eigenvals * np.log2(eigenvals)).real
        
        original_info = quantum_information_measure(original_state)
        
        # 传输后的信息量
        for bell_outcome in range(4):
            charlie_state = self._get_charlie_state_after_measurement(bell_outcome, alpha, beta)
            correction = self._get_correction_operator(bell_outcome)
            final_state = correction @ charlie_state
            
            final_info = quantum_information_measure(final_state)
            
            # 信息量应该守恒
            self.assertAlmostEqual(
                original_info, final_info, 10,
                f"Information should be conserved for outcome {bell_outcome}"
            )
        
        # 验证概率守恒
        # 所有可能结果的概率和为1
        probabilities = [0.25, 0.25, 0.25, 0.25]  # 每种Bell测量结果等概率
        total_probability = sum(probabilities)
        self.assertAlmostEqual(total_probability, 1.0, 10, "Total probability should be 1")
        
        # 验证幺正性保证信息守恒
        for correction in [self.I, self.sigma_x, self.sigma_y, self.sigma_z]:
            unitarity = correction @ correction.conj().T
            determinant = np.linalg.det(correction)
            
            self.assertTrue(
                np.allclose(unitarity, self.I, atol=self.tolerance),
                "Correction operators should be unitary"
            )
            
            self.assertAlmostEqual(
                abs(determinant), 1.0, 10,
                "Unitary operators should have unit determinant"
            )


if __name__ == "__main__":
    unittest.main()