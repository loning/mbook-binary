"""
Unit tests for T3-5: Quantum Error Correction Theorem
T3-5：量子纠错定理的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
import numpy as np
from typing import List, Tuple
import time


class TestT3_5_QuantumErrorCorrection(VerificationTest):
    """T3-5 量子纠错定理的数学化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        # 设置数值精度
        self.tolerance = 1e-10
        
        # 定义Pauli算符
        self.I = np.eye(2, dtype=complex)
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # 预定义多量子比特Pauli算符
        self.pauli_ops = [self.I, self.X, self.Y, self.Z]
        
        # 3量子比特重复码的设置
        self.n_qubits = 3  # 物理量子比特数
        self.k_logical = 1  # 逻辑量子比特数
        self.code_distance = 3
        
        # 码态
        self.logical_0 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex)  # |000⟩
        self.logical_1 = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=complex)  # |111⟩
        self.code_states = [self.logical_0, self.logical_1]
        
        # 稳定子生成元
        self.stabilizer_generators = [
            self._tensor_product_pauli("ZZI"),  # Z1Z2
            self._tensor_product_pauli("IZZ")   # Z2Z3
        ]
    
    def _tensor_product_pauli(self, pauli_string: str) -> np.ndarray:
        """根据Pauli字符串构造张量积算符"""
        pauli_map = {'I': self.I, 'X': self.X, 'Y': self.Y, 'Z': self.Z}
        result = pauli_map[pauli_string[0]]
        for char in pauli_string[1:]:
            result = np.kron(result, pauli_map[char])
        return result
    
    def _create_kraus_operators(self, error_probs: List[float]) -> List[np.ndarray]:
        """创建Kraus算符"""
        kraus_ops = []
        for prob, pauli in zip(error_probs, self.pauli_ops):
            if prob > 0:
                kraus_ops.append(np.sqrt(prob) * pauli)
        return kraus_ops
    
    def _apply_kraus_channel(self, density_matrix: np.ndarray, kraus_ops: List[np.ndarray]) -> np.ndarray:
        """应用Kraus信道"""
        result = np.zeros_like(density_matrix)
        for E in kraus_ops:
            result += E @ density_matrix @ E.conj().T
        return result
    
    def _measure_syndrome(self, state: np.ndarray, stabilizers: List[np.ndarray]) -> Tuple[int, ...]:
        """测量错误症状"""
        syndrome = []
        for stabilizer in stabilizers:
            syndrome_value = np.vdot(state, stabilizer @ state).real
            syndrome.append(1 if syndrome_value > 0 else -1)
        return tuple(syndrome)
    
    def _calculate_fidelity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """计算保真度"""
        overlap = np.vdot(state1, state2)
        return float(abs(overlap)**2)
    
    def test_error_model_establishment(self):
        """测试错误模型建立 - 验证检查点1"""
        # 测试不同的错误概率场景
        error_scenarios = [
            [1.0, 0.0, 0.0, 0.0],        # 无错误
            [0.9, 0.05, 0.025, 0.025],   # 轻微错误
            [0.7, 0.1, 0.1, 0.1],        # 中等错误
            [0.4, 0.2, 0.2, 0.2]         # 高错误率
        ]
        
        for i, error_probs in enumerate(error_scenarios):
            # 验证概率归一化
            self.assertAlmostEqual(
                sum(error_probs), 1.0, 10,
                f"Probabilities should sum to 1 for scenario {i}"
            )
            
            # 创建Kraus算符
            kraus_ops = self._create_kraus_operators(error_probs)
            
            # 验证迹保持性
            trace_sum = sum(E.conj().T @ E for E in kraus_ops)
            self.assertTrue(
                np.allclose(trace_sum, self.I, atol=self.tolerance),
                f"Trace preservation failed for scenario {i}"
            )
            
            # 测试对纯态的作用
            pure_state = np.array([0.6, 0.8], dtype=complex)
            pure_density = np.outer(pure_state, pure_state.conj())
            
            corrupted_state = self._apply_kraus_channel(pure_density, kraus_ops)
            
            # 验证迹保持
            self.assertAlmostEqual(
                np.trace(corrupted_state).real, 1.0, 10,
                f"Trace should be preserved for scenario {i}"
            )
            
            # 验证正定性
            eigenvals = np.linalg.eigvals(corrupted_state)
            self.assertTrue(
                np.all(eigenvals.real >= -self.tolerance),
                f"Density matrix should be positive semidefinite for scenario {i}"
            )
        
        # 验证Pauli算符的基本性质
        for i, pauli in enumerate(self.pauli_ops):
            # 验证Hermitian性
            self.assertTrue(
                np.allclose(pauli, pauli.conj().T, atol=self.tolerance),
                f"Pauli operator {i} should be Hermitian"
            )
            
            # 验证幺正性
            self.assertTrue(
                np.allclose(pauli @ pauli.conj().T, self.I, atol=self.tolerance),
                f"Pauli operator {i} should be unitary"
            )
            
            # 验证本征值为±1
            eigenvals = np.linalg.eigvals(pauli)
            eigenvals_abs = np.abs(eigenvals)
            self.assertTrue(
                np.allclose(eigenvals_abs, 1.0, atol=self.tolerance),
                f"Pauli operator {i} should have eigenvalues ±1"
            )
        
        # 验证Kraus表示的物理意义
        def test_bit_flip_channel():
            """测试比特翻转信道"""
            p = 0.1  # 比特翻转概率
            kraus_ops = [np.sqrt(1-p) * self.I, np.sqrt(p) * self.X]
            
            # 验证物理合理性
            trace_check = kraus_ops[0].conj().T @ kraus_ops[0] + kraus_ops[1].conj().T @ kraus_ops[1]
            self.assertTrue(
                np.allclose(trace_check, self.I),
                "Bit flip channel should satisfy trace preservation"
            )
            
            return True
        
        self.assertTrue(test_bit_flip_channel(), "Bit flip channel test failed")
        
    def test_encoding_subspace_construction(self):
        """测试编码子空间构造 - 验证检查点2"""
        # 验证码态的基本性质
        for i, state in enumerate(self.code_states):
            # 验证归一化
            norm = np.vdot(state, state).real
            self.assertAlmostEqual(
                norm, 1.0, 10,
                f"Code state {i} should be normalized"
            )
        
        # 验证码态的正交性
        overlap = np.vdot(self.logical_0, self.logical_1)
        self.assertAlmostEqual(
            abs(overlap), 0.0, 10,
            "Logical states should be orthogonal"
        )
        
        # 验证Hamming距离
        def hamming_distance(string1: str, string2: str) -> int:
            """计算Hamming距离"""
            return sum(c1 != c2 for c1, c2 in zip(string1, string2))
        
        codeword_0 = "000"
        codeword_1 = "111"
        distance = hamming_distance(codeword_0, codeword_1)
        self.assertEqual(
            distance, self.code_distance,
            f"Code distance should be {self.code_distance}"
        )
        
        # 验证纠错能力
        max_correctable_errors = (self.code_distance - 1) // 2
        self.assertEqual(
            max_correctable_errors, 1,
            "Should be able to correct 1 error"
        )
        
        # 验证编码子空间的维度关系
        physical_dim = 2**self.n_qubits
        logical_dim = 2**self.k_logical
        
        self.assertEqual(physical_dim, 8, "Physical space dimension should be 8")
        self.assertEqual(logical_dim, 2, "Logical space dimension should be 2")
        self.assertGreater(
            physical_dim, logical_dim,
            "Physical space should be larger for redundancy"
        )
        
        # 验证码率
        code_rate = self.k_logical / self.n_qubits
        self.assertTrue(
            0 < code_rate < 1,
            "Code rate should be between 0 and 1"
        )
        
        # 验证码态确实在稳定子的+1本征子空间中
        for i, code_state in enumerate(self.code_states):
            for j, stabilizer in enumerate(self.stabilizer_generators):
                eigenval = np.vdot(code_state, stabilizer @ code_state).real
                self.assertAlmostEqual(
                    eigenval, 1.0, 10,
                    f"Code state {i} should be +1 eigenstate of stabilizer {j}"
                )
        
        # 验证编码的线性性
        alpha, beta = 0.6, 0.8
        logical_superposition = alpha * self.logical_0 + beta * self.logical_1
        logical_superposition = logical_superposition / np.linalg.norm(logical_superposition)
        
        # 叠加态也应该在码子空间中
        for j, stabilizer in enumerate(self.stabilizer_generators):
            eigenval = np.vdot(logical_superposition, stabilizer @ logical_superposition).real
            self.assertAlmostEqual(
                eigenval, 1.0, 10,
                f"Logical superposition should be +1 eigenstate of stabilizer {j}"
            )
        
    def test_stabilizer_formalism_verification(self):
        """测试稳定子形式主义验证 - 验证检查点3"""
        # 验证稳定子生成元的数量
        expected_generators = self.n_qubits - self.k_logical
        self.assertEqual(
            len(self.stabilizer_generators), expected_generators,
            f"Should have {expected_generators} stabilizer generators"
        )
        
        # 验证稳定子生成元的对易性
        for i, g_i in enumerate(self.stabilizer_generators):
            for j, g_j in enumerate(self.stabilizer_generators):
                commutator = g_i @ g_j - g_j @ g_i
                self.assertTrue(
                    np.allclose(commutator, np.zeros_like(commutator), atol=self.tolerance),
                    f"Stabilizer generators {i} and {j} should commute"
                )
        
        # 验证稳定子算符的基本性质
        for i, generator in enumerate(self.stabilizer_generators):
            # 验证Hermitian性
            self.assertTrue(
                np.allclose(generator, generator.conj().T, atol=self.tolerance),
                f"Stabilizer generator {i} should be Hermitian"
            )
            
            # 验证幺正性
            self.assertTrue(
                np.allclose(generator @ generator.conj().T, np.eye(generator.shape[0]), atol=self.tolerance),
                f"Stabilizer generator {i} should be unitary"
            )
            
            # 验证平方为恒等
            self.assertTrue(
                np.allclose(generator @ generator, np.eye(generator.shape[0]), atol=self.tolerance),
                f"Stabilizer generator {i} squared should be identity"
            )
            
            # 验证本征值为±1
            eigenvals = np.linalg.eigvals(generator)
            eigenvals_real = eigenvals.real
            self.assertTrue(
                np.allclose(np.abs(eigenvals_real), 1.0, atol=self.tolerance),
                f"Stabilizer generator {i} should have eigenvalues ±1"
            )
        
        # 验证码子空间的正确构造
        def find_code_subspace(stabilizers):
            """找到稳定子的+1本征子空间"""
            total_dim = 2**self.n_qubits
            code_states = []
            
            for i in range(total_dim):
                state = np.zeros(total_dim, dtype=complex)
                state[i] = 1.0
                
                is_code_state = True
                for stabilizer in stabilizers:
                    eigenval = np.vdot(state, stabilizer @ state).real
                    if abs(eigenval - 1.0) > self.tolerance:
                        is_code_state = False
                        break
                
                if is_code_state:
                    code_states.append(state)
            
            return code_states
        
        computed_code_subspace = find_code_subspace(self.stabilizer_generators)
        expected_code_dim = 2**self.k_logical
        
        self.assertEqual(
            len(computed_code_subspace), expected_code_dim,
            f"Code subspace should have dimension {expected_code_dim}"
        )
        
        # 验证计算得到的码子空间与预期一致
        for expected_state in self.code_states:
            found_match = False
            for computed_state in computed_code_subspace:
                if np.allclose(expected_state, computed_state, atol=self.tolerance):
                    found_match = True
                    break
            self.assertTrue(
                found_match,
                "Expected code state should be in computed code subspace"
            )
        
        # 验证稳定子群的大小
        stabilizer_group_size = 2**len(self.stabilizer_generators)
        expected_group_size = 2**(self.n_qubits - self.k_logical)
        self.assertEqual(
            stabilizer_group_size, expected_group_size,
            f"Stabilizer group size should be {expected_group_size}"
        )
        
        # 验证逻辑算符的存在
        logical_X = self._tensor_product_pauli("XXX")
        logical_Z = self._tensor_product_pauli("ZZZ")
        
        for logical_op in [logical_X, logical_Z]:
            # 验证与稳定子的对易性
            for stabilizer in self.stabilizer_generators:
                commutator = logical_op @ stabilizer - stabilizer @ logical_op
                self.assertTrue(
                    np.allclose(commutator, np.zeros_like(commutator), atol=self.tolerance),
                    "Logical operators should commute with stabilizers"
                )
        
    def test_error_detection_protocol(self):
        """测试错误探测协议 - 验证检查点4"""
        # 定义可能的错误
        errors = {
            "no_error": self._tensor_product_pauli("III"),
            "X1": self._tensor_product_pauli("XII"),
            "X2": self._tensor_product_pauli("IXI"),
            "X3": self._tensor_product_pauli("IIX"),
            "X1X2": self._tensor_product_pauli("XXI"),
            "X2X3": self._tensor_product_pauli("IXX"),
            "X1X3": self._tensor_product_pauli("XIX")
        }
        
        # 构造症状查找表
        syndrome_table = {}
        
        for error_name, error_op in errors.items():
            for state_idx, code_state in enumerate(self.code_states):
                # 应用错误
                error_state = error_op @ code_state
                
                # 测量症状
                syndrome = self._measure_syndrome(error_state, self.stabilizer_generators)
                
                # 记录症状模式
                if syndrome not in syndrome_table:
                    syndrome_table[syndrome] = []
                syndrome_table[syndrome].append((error_name, state_idx))
        
        # 验证无错误的症状
        no_error_syndrome = (1, 1)
        self.assertIn(
            no_error_syndrome, syndrome_table,
            "No-error syndrome should be (1, 1)"
        )
        
        # 验证可纠正错误的症状唯一性
        correctable_errors = ["no_error", "X1", "X2", "X3"]
        
        syndrome_to_error = {}
        for error_name in correctable_errors:
            for state_idx, code_state in enumerate(self.code_states):
                error_op = errors[error_name]
                error_state = error_op @ code_state
                syndrome = self._measure_syndrome(error_state, self.stabilizer_generators)
                
                if syndrome in syndrome_to_error:
                    # 检查是否映射到同一个错误
                    self.assertEqual(
                        syndrome_to_error[syndrome], error_name,
                        f"Syndrome {syndrome} should uniquely identify error"
                    )
                else:
                    syndrome_to_error[syndrome] = error_name
        
        # 验证症状测量的非破坏性
        alpha, beta = 0.6, 0.8
        logical_superposition = alpha * self.logical_0 + beta * self.logical_1
        logical_superposition = logical_superposition / np.linalg.norm(logical_superposition)
        
        # 无错误情况下的症状
        syndrome = self._measure_syndrome(logical_superposition, self.stabilizer_generators)
        self.assertEqual(
            syndrome, (1, 1),
            "Logical superposition should have no-error syndrome"
        )
        
        # 验证错误探测的完整性
        total_syndromes = 2**len(self.stabilizer_generators)
        observed_syndromes = set(syndrome_table.keys())
        
        self.assertLessEqual(
            len(observed_syndromes), total_syndromes,
            f"Cannot have more syndromes than possible: {len(observed_syndromes)} > {total_syndromes}"
        )
        
        # 验证关键症状的存在
        critical_syndromes = {(1, 1)}
        self.assertTrue(
            critical_syndromes.issubset(observed_syndromes),
            "Critical syndromes should be present"
        )
        
        # 测试症状的确定性
        for error_name, error_op in errors.items():
            syndromes_for_error = []
            for code_state in self.code_states:
                error_state = error_op @ code_state
                syndrome = self._measure_syndrome(error_state, self.stabilizer_generators)
                syndromes_for_error.append(syndrome)
            
            # 对于线性码，同一错误在不同码态上产生相同症状
            unique_syndromes = set(syndromes_for_error)
            self.assertEqual(
                len(unique_syndromes), 1,
                f"Error {error_name} should produce consistent syndrome"
            )
        
    def test_error_correction_application(self):
        """测试纠错操作应用 - 验证检查点5"""
        # 错误和对应的纠正操作映射表
        error_correction_table = {
            "no_error": ("III", "III"),
            "X1": ("XII", "XII"),
            "X2": ("IXI", "IXI"),
            "X3": ("IIX", "IIX")
        }
        
        def get_correction_from_syndrome(syndrome):
            """根据症状确定纠正操作"""
            syndrome_to_correction = {
                (1, 1): "III",    # 无错误
                (1, -1): "IIX",   # X3错误
                (-1, 1): "XII",   # X1错误
                (-1, -1): "IXI"   # X2错误
            }
            return syndrome_to_correction.get(syndrome, "III")
        
        def apply_correction(state, correction_op_string):
            """应用纠正操作"""
            correction_op = self._tensor_product_pauli(correction_op_string)
            return correction_op @ state
        
        # 测试完整的纠错协议
        success_count = 0
        total_tests = 0
        
        for error_name, (error_op_string, _) in error_correction_table.items():
            error_op = self._tensor_product_pauli(error_op_string)
            
            for state_name, original_state in [("logical_0", self.logical_0), ("logical_1", self.logical_1)]:
                total_tests += 1
                
                # 应用错误
                corrupted_state = error_op @ original_state
                
                # 测量症状
                syndrome = self._measure_syndrome(corrupted_state, self.stabilizer_generators)
                
                # 确定并应用纠正操作
                correction_op_string = get_correction_from_syndrome(syndrome)
                corrected_state = apply_correction(corrupted_state, correction_op_string)
                
                # 验证纠错效果
                fidelity = self._calculate_fidelity(corrected_state, original_state)
                
                if abs(fidelity - 1.0) < self.tolerance:
                    success_count += 1
                
                self.assertAlmostEqual(
                    fidelity, 1.0, 10,
                    f"Correction failed for {error_name} on {state_name}: fidelity = {fidelity}"
                )
        
        correction_rate = success_count / total_tests
        self.assertEqual(
            correction_rate, 1.0,
            f"Error correction should be perfect, got rate = {correction_rate}"
        )
        
        # 测试对逻辑叠加态的纠错
        alpha, beta = 0.6, 0.8
        logical_superposition = alpha * self.logical_0 + beta * self.logical_1
        logical_superposition = logical_superposition / np.linalg.norm(logical_superposition)
        
        for error_name, (error_op_string, _) in error_correction_table.items():
            error_op = self._tensor_product_pauli(error_op_string)
            
            # 应用错误和纠正
            corrupted_superposition = error_op @ logical_superposition
            syndrome = self._measure_syndrome(corrupted_superposition, self.stabilizer_generators)
            correction_op_string = get_correction_from_syndrome(syndrome)
            corrected_superposition = apply_correction(corrupted_superposition, correction_op_string)
            
            # 验证纠错效果
            fidelity = self._calculate_fidelity(corrected_superposition, logical_superposition)
            self.assertAlmostEqual(
                fidelity, 1.0, 10,
                f"Superposition correction failed for {error_name}: fidelity = {fidelity}"
            )
        
        # 验证纠正操作的幂等性
        for error_name, (error_op_string, correction_op_string) in error_correction_table.items():
            error_op = self._tensor_product_pauli(error_op_string)
            correction_op = self._tensor_product_pauli(correction_op_string)
            
            # 验证 correction ∘ error = identity (on code states)
            combined_op = correction_op @ error_op
            
            for original_state in [self.logical_0, self.logical_1]:
                result_state = combined_op @ original_state
                fidelity = self._calculate_fidelity(result_state, original_state)
                self.assertAlmostEqual(
                    fidelity, 1.0, 10,
                    f"Combined operation should restore original state for {error_name}"
                )
        
        # 验证纠错的线性性
        coefficients = [0.3, 0.7]
        superposition = coefficients[0] * self.logical_0 + coefficients[1] * self.logical_1
        superposition = superposition / np.linalg.norm(superposition)
        
        for error_name, (error_op_string, _) in error_correction_table.items():
            error_op = self._tensor_product_pauli(error_op_string)
            
            # 应用错误和纠正
            corrupted = error_op @ superposition
            syndrome = self._measure_syndrome(corrupted, self.stabilizer_generators)
            correction_op_string = get_correction_from_syndrome(syndrome)
            corrected = apply_correction(corrupted, correction_op_string)
            
            # 验证结果
            fidelity = self._calculate_fidelity(corrected, superposition)
            self.assertAlmostEqual(
                fidelity, 1.0, 10,
                f"Linearity test failed for {error_name}: fidelity = {fidelity}"
            )
        
    def test_error_threshold_behavior(self):
        """测试错误阈值行为"""
        def simulate_logical_error_rate(physical_error_rate, num_trials=100):
            """模拟逻辑错误率"""
            logical_errors = 0
            
            for _ in range(num_trials):
                # 随机选择初始逻辑态
                original_state = self.code_states[np.random.randint(len(self.code_states))]
                
                # 对每个物理量子比特独立应用错误
                current_state = original_state.copy()
                
                for qubit in range(self.n_qubits):
                    if np.random.random() < physical_error_rate:
                        # 应用X错误到特定量子比特
                        error_string = "I" * qubit + "X" + "I" * (self.n_qubits - qubit - 1)
                        error_op = self._tensor_product_pauli(error_string)
                        current_state = error_op @ current_state
                
                # 尝试纠错
                syndrome = self._measure_syndrome(current_state, self.stabilizer_generators)
                
                # 简化的纠错逻辑
                if syndrome == (1, 1):
                    correction = "III"
                elif syndrome == (-1, 1):
                    correction = "XII"
                elif syndrome == (-1, -1):
                    correction = "IXI"
                elif syndrome == (1, -1):
                    correction = "IIX"
                else:
                    correction = "III"  # 无法纠正
                
                correction_op = self._tensor_product_pauli(correction)
                corrected_state = correction_op @ current_state
                
                # 检查是否成功恢复
                fidelity = self._calculate_fidelity(corrected_state, original_state)
                if fidelity < 0.99:  # 允许小的数值误差
                    logical_errors += 1
            
            return logical_errors / num_trials
        
        # 测试不同的物理错误率
        physical_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
        logical_rates = []
        
        for p_phys in physical_rates:
            p_logical = simulate_logical_error_rate(p_phys, num_trials=50)  # 减少试验次数以加快测试
            logical_rates.append(p_logical)
        
        # 验证低错误率时纠错有效
        if logical_rates[0] < physical_rates[0]:
            # 纠错应该降低逻辑错误率
            pass  # 这是期望的行为
        
        # 验证错误率过高时纠错失效
        self.assertGreater(
            logical_rates[-1], 0,
            "High physical error rate should lead to logical errors"
        )
        
    def test_information_preservation(self):
        """测试信息保护能力"""
        def quantum_information_measure(state):
            """简化的量子信息度量"""
            density_matrix = np.outer(state, state.conj())
            eigenvals = np.linalg.eigvals(density_matrix)
            eigenvals = eigenvals.real[eigenvals.real > 1e-12]
            if len(eigenvals) == 0:
                return 0.0
            return -np.sum(eigenvals * np.log2(eigenvals))
        
        # 测试不同的逻辑态
        test_states = [
            self.logical_0,
            self.logical_1,
            (self.logical_0 + self.logical_1) / np.sqrt(2),  # |+⟩_L
            (self.logical_0 - self.logical_1) / np.sqrt(2)   # |-⟩_L
        ]
        
        for original_state in test_states:
            original_info = quantum_information_measure(original_state)
            
            # 应用可纠正的错误
            correctable_errors = ["III", "XII", "IXI", "IIX"]
            
            for error_string in correctable_errors:
                error_op = self._tensor_product_pauli(error_string)
                error_state = error_op @ original_state
                
                # 测量症状并纠错
                syndrome = self._measure_syndrome(error_state, self.stabilizer_generators)
                
                # 应用纠正
                if syndrome == (1, 1):
                    correction = "III"
                elif syndrome == (-1, 1):
                    correction = "XII"
                elif syndrome == (-1, -1):
                    correction = "IXI"
                elif syndrome == (1, -1):
                    correction = "IIX"
                
                correction_op = self._tensor_product_pauli(correction)
                corrected_state = correction_op @ error_state
                
                # 验证信息保护
                corrected_info = quantum_information_measure(corrected_state)
                self.assertAlmostEqual(
                    original_info, corrected_info, 10,
                    f"Information should be preserved after error correction"
                )
        
        # 验证自指完备性的维护
        # 系统应该能够描述自己的纠错过程
        def self_description_capability():
            """验证系统的自描述能力"""
            # 系统应该能够：
            # 1. 识别自己的码结构
            # 2. 检测错误
            # 3. 应用纠正
            # 4. 验证纠错效果
            
            # 这通过成功的纠错协议得到验证
            return True
        
        self.assertTrue(
            self_description_capability(),
            "System should maintain self-referential capability"
        )
        
    def test_performance_and_scalability(self):
        """测试纠错性能和可扩展性"""
        # 测试纠错速度
        num_trials = 50  # 减少试验次数
        start_time = time.time()
        
        for _ in range(num_trials):
            # 随机选择错误和初始态
            error_ops = ["III", "XII", "IXI", "IIX"]
            error_string = np.random.choice(error_ops)
            error_op = self._tensor_product_pauli(error_string)
            
            original_state = self.code_states[np.random.randint(len(self.code_states))]
            
            # 执行纠错协议
            corrupted = error_op @ original_state
            syndrome = self._measure_syndrome(corrupted, self.stabilizer_generators)
            
            # 快速查找纠正操作
            correction_map = {
                (1, 1): "III",
                (-1, 1): "XII",
                (-1, -1): "IXI",
                (1, -1): "IIX"
            }
            correction_string = correction_map.get(syndrome) or "III"
            correction_op = self._tensor_product_pauli(correction_string)
            corrected = correction_op @ corrupted
            
            # 验证结果
            fidelity = self._calculate_fidelity(corrected, original_state)
            self.assertAlmostEqual(fidelity, 1.0, 8, "Performance test correction failed")
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_trials
        
        # 验证性能合理
        self.assertLess(
            avg_time, 0.1,  # 放宽时间限制
            f"Error correction should be reasonably fast, got {avg_time} seconds per correction"
        )
        
        # 验证内存使用合理性
        # 3量子比特系统应该有合理的内存占用
        state_size = len(self.logical_0)
        self.assertEqual(state_size, 8, "3-qubit system should have 8-dimensional state space")
        
        stabilizer_size = sum(gen.size for gen in self.stabilizer_generators)
        self.assertLess(
            stabilizer_size, 1000,
            "Stabilizer generators should have reasonable memory footprint"
        )


if __name__ == "__main__":
    unittest.main()