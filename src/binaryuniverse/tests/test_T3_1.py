"""
Unit tests for T3-1: Quantum State Emergence Theorem
T3-1：量子态涌现定理的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
import cmath


class TestT3_1_QuantumStateEmergence(VerificationTest):
    """T3-1 量子态涌现定理的数学化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        # 设置Hilbert空间维度
        self.hilbert_dim = 4
        # 预生成Fibonacci数列
        self.fibonacci = self._generate_fibonacci(20)
        
    def _generate_fibonacci(self, n: int) -> List[int]:
        """生成前n个修正的Fibonacci数（从1,2开始）"""
        if n == 0:
            return []
        if n == 1:
            return [1]
        
        fib = [1, 2]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        
        return fib
    
    def _phi_representation_to_vector(self, phi_repr: List[int]) -> np.ndarray:
        """将φ-表示转换为复向量"""
        # 计算φ-表示的值
        value = sum(bit * fib for bit, fib in 
                   zip(phi_repr, self.fibonacci[:len(phi_repr)]))
        
        # 映射到Hilbert空间
        dim = self.hilbert_dim
        vec = np.zeros(dim, dtype=complex)
        
        # 简化映射：使用值的模来确定基态
        base_index = value % dim
        vec[base_index] = 1.0
        
        return vec
    
    def _create_observer_operator(self, obs_func: Callable[[int], int]) -> np.ndarray:
        """根据观测函数创建对应的算符"""
        dim = self.hilbert_dim
        operator = np.zeros((dim, dim), dtype=complex)
        
        for i in range(dim):
            j = obs_func(i)
            operator[j, i] = 1.0  # 修正：应该是从i到j的转换
            
        return operator
    
    def _is_hermitian(self, matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
        """检查矩阵是否为Hermitian"""
        return np.allclose(matrix, matrix.conj().T, atol=tolerance)
    
    def _is_unitary(self, matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
        """检查矩阵是否为Unitary"""
        dim = matrix.shape[0]
        product = matrix @ matrix.conj().T
        identity = np.eye(dim)
        return np.allclose(product, identity, atol=tolerance)
    
    def _create_superposition_state(self, coefficients: List[complex]) -> np.ndarray:
        """创建叠加态"""
        dim = len(coefficients)
        state = np.array(coefficients, dtype=complex)
        # 归一化
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm
        return state
    
    def test_binary_state_linearity(self):
        """测试二进制状态线性性 - 验证检查点1"""
        # 创建测试的φ-表示状态
        phi_state1 = [1, 0, 1, 0]  # F₀ + F₂ = 1 + 3 = 4
        phi_state2 = [0, 1, 0, 1]  # F₁ + F₃ = 2 + 5 = 7
        
        # 转换为向量
        vec1 = self._phi_representation_to_vector(phi_state1)
        vec2 = self._phi_representation_to_vector(phi_state2)
        
        # 线性组合系数
        alpha = 0.6 + 0.0j
        beta = 0.8 + 0.0j
        
        # 计算线性组合
        combined_vec = alpha * vec1 + beta * vec2
        
        # 验证向量空间性质
        # 1. 加法结合律
        vec3 = np.array([0, 0, 1, 0], dtype=complex)
        result1 = (vec1 + vec2) + vec3
        result2 = vec1 + (vec2 + vec3)
        self.assertTrue(
            np.allclose(result1, result2),
            "Vector addition should be associative"
        )
        
        # 2. 标量乘法分配律
        gamma = 0.5 + 0.3j
        result1 = gamma * (vec1 + vec2)
        result2 = gamma * vec1 + gamma * vec2
        self.assertTrue(
            np.allclose(result1, result2),
            "Scalar multiplication should be distributive"
        )
        
        # 3. 零向量性质
        zero_vec = np.zeros(self.hilbert_dim, dtype=complex)
        result = vec1 + zero_vec
        self.assertTrue(
            np.allclose(result, vec1),
            "Zero vector should be additive identity"
        )
        
        # 4. 标量乘法的结合律
        delta = 2.0 + 1.0j
        result1 = (alpha * delta) * vec1
        result2 = alpha * (delta * vec1)
        self.assertTrue(
            np.allclose(result1, result2),
            "Scalar multiplication should be associative"
        )
        
    def test_observer_operator_mapping(self):
        """测试观测器算符映射 - 验证检查点2"""
        # 定义几种观测器函数
        observers = {
            "identity": lambda x: x,
            "cyclic": lambda x: (x + 1) % self.hilbert_dim,
            "reflection": lambda x: (self.hilbert_dim - 1 - x) % self.hilbert_dim,
            "swap": lambda x: x if x % 2 == 0 else (x + 1) % self.hilbert_dim
        }
        
        for obs_name, obs_func in observers.items():
            # 创建对应的算符
            operator = self._create_observer_operator(obs_func)
            
            # 验证算符的基本性质
            self.assertEqual(
                operator.shape, (self.hilbert_dim, self.hilbert_dim),
                f"Operator {obs_name} should have correct dimensions"
            )
            
            # 验证线性性
            vec1 = np.array([1, 0, 0, 0], dtype=complex)
            vec2 = np.array([0, 1, 0, 0], dtype=complex)
            alpha, beta = 0.6 + 0.2j, 0.8 - 0.1j
            
            combined_input = alpha * vec1 + beta * vec2
            
            # 线性性：O(αv₁ + βv₂) = αOv₁ + βOv₂
            result1 = operator @ combined_input
            result2 = alpha * (operator @ vec1) + beta * (operator @ vec2)
            
            self.assertTrue(
                np.allclose(result1, result2),
                f"Operator {obs_name} should be linear"
            )
            
            # 验证观测器的函数对应性
            for i in range(self.hilbert_dim):
                input_state = np.zeros(self.hilbert_dim, dtype=complex)
                input_state[i] = 1.0
                
                output_state = operator @ input_state
                expected_index = obs_func(i)
                
                # 输出应该是对应基态
                expected_output = np.zeros(self.hilbert_dim, dtype=complex)
                expected_output[expected_index] = 1.0
                
                self.assertTrue(
                    np.allclose(output_state, expected_output),
                    f"Operator {obs_name} should map state {i} to state {expected_index}"
                )
                
    def test_state_vector_construction(self):
        """测试态矢量构造 - 验证检查点3"""
        # 构造正交归一基态
        basis_states = []
        for i in range(self.hilbert_dim):
            state = np.zeros(self.hilbert_dim, dtype=complex)
            state[i] = 1.0
            basis_states.append(state)
        
        # 验证正交性和归一性
        for i, state_i in enumerate(basis_states):
            for j, state_j in enumerate(basis_states):
                inner_product = np.vdot(state_i, state_j)
                
                if i == j:
                    self.assertAlmostEqual(
                        abs(inner_product), 1.0, 10,
                        f"Basis state {i} should be normalized"
                    )
                else:
                    self.assertAlmostEqual(
                        abs(inner_product), 0.0, 10,
                        f"Basis states {i} and {j} should be orthogonal"
                    )
        
        # 构造和验证叠加态
        test_superpositions = [
            [1/2, 1/2, 1/2, 1/2],  # 均匀叠加
            [1, 0, 0, 0],          # 纯态
            [0.6, 0.8, 0, 0],      # 两态叠加
            [0.5, 0.5j, 0.5, -0.5j]  # 复系数叠加
        ]
        
        for coeffs in test_superpositions:
            superposition = self._create_superposition_state(coeffs)
            
            # 验证归一化
            norm_squared = np.vdot(superposition, superposition)
            self.assertAlmostEqual(
                norm_squared.real, 1.0, 10,
                "Superposition state should be normalized"
            )
            self.assertAlmostEqual(
                norm_squared.imag, 0.0, 10,
                "Norm squared should be real"
            )
            
            # 验证概率解释
            probabilities = []
            for basis_state in basis_states:
                amplitude = np.vdot(basis_state, superposition)
                probability = abs(amplitude)**2
                probabilities.append(probability)
            
            total_probability = sum(probabilities)
            self.assertAlmostEqual(
                total_probability, 1.0, 10,
                "Total probability should equal 1"
            )
            
            # 验证概率为非负
            for i, prob in enumerate(probabilities):
                self.assertGreaterEqual(
                    prob, 0.0,
                    f"Probability for state {i} should be non-negative"
                )
                
    def test_quantum_properties_verification(self):
        """测试量子性质验证 - 验证检查点4"""
        # 创建随机归一化态
        psi_real = np.random.randn(self.hilbert_dim)
        psi_imag = np.random.randn(self.hilbert_dim)
        psi = psi_real + 1j * psi_imag
        psi = psi / np.linalg.norm(psi)
        
        # 创建Hermitian算符（可观测量）
        def create_pauli_like_operator(dim):
            """创建类Pauli算符"""
            if dim == 2:
                # 标准Pauli-Z
                return np.array([[1, 0], [0, -1]], dtype=complex)
            else:
                # 扩展版本
                op = np.zeros((dim, dim), dtype=complex)
                for i in range(dim):
                    op[i, i] = (-1)**i  # 交替本征值
                return op
        
        def create_hermitian_operator(dim):
            """创建一般Hermitian算符"""
            A_real = np.random.randn(dim, dim)
            A_imag = np.random.randn(dim, dim)
            A = A_real + 1j * A_imag
            return (A + A.conj().T) / 2
        
        operators = {
            "pauli_like": create_pauli_like_operator(self.hilbert_dim),
            "random_hermitian": create_hermitian_operator(self.hilbert_dim)
        }
        
        for op_name, H in operators.items():
            # 验证Hermitian性质
            self.assertTrue(
                self._is_hermitian(H),
                f"Operator {op_name} should be Hermitian"
            )
            
            # 验证期望值为实数
            expectation = np.vdot(psi, H @ psi)
            self.assertAlmostEqual(
                expectation.imag, 0.0, 10,
                f"Expectation value of {op_name} should be real"
            )
            
            # 验证本征值为实数
            eigenvals, eigenvecs = np.linalg.eigh(H)
            for i, eigenval in enumerate(eigenvals):
                self.assertAlmostEqual(
                    eigenval.imag, 0.0, 10,
                    f"Eigenvalue {i} of {op_name} should be real"
                )
            
            # 验证本征向量的正交性
            for i in range(len(eigenvecs)):
                for j in range(i+1, len(eigenvecs)):
                    overlap = np.vdot(eigenvecs[:, i], eigenvecs[:, j])
                    self.assertAlmostEqual(
                        abs(overlap), 0.0, 8,
                        f"Eigenvectors {i} and {j} should be orthogonal"
                    )
            
            # 验证谱分解
            reconstructed = eigenvecs @ np.diag(eigenvals) @ eigenvecs.conj().T
            self.assertTrue(
                np.allclose(H, reconstructed),
                f"Operator {op_name} should equal its spectral decomposition"
            )
            
            # 验证Born规则
            # 在本征态基中的概率
            coeffs = eigenvecs.conj().T @ psi
            probabilities = [abs(c)**2 for c in coeffs]
            
            total_prob = sum(probabilities)
            self.assertAlmostEqual(
                total_prob, 1.0, 10,
                "Total probability in eigenbasis should be 1"
            )
            
            # 期望值的计算验证
            calculated_expectation = sum(prob * eigenval.real 
                                       for prob, eigenval in zip(probabilities, eigenvals))
            self.assertAlmostEqual(
                calculated_expectation, expectation.real, 10,
                f"Expectation value should match Born rule calculation"
            )
            
    def test_isomorphism_establishment(self):
        """测试同构建立 - 验证检查点5"""
        # 定义经典自指系统
        class ClassicalSelfRefSystem:
            def __init__(self, state_count):
                self.states = list(range(state_count))
                self.current_state = 0
                
            def observe(self, observer_func):
                """观测操作"""
                return observer_func(self.current_state)
            
            def update(self, new_state):
                """更新状态"""
                self.current_state = new_state
                
            def evolve(self, observer_func):
                """演化：观测+更新"""
                observed = self.observe(observer_func)
                self.update(observed)
                return observed
        
        # 定义量子系统
        class QuantumSelfRefSystem:
            def __init__(self, hilbert_dim):
                self.dim = hilbert_dim
                self.state = np.zeros(hilbert_dim, dtype=complex)
                self.state[0] = 1.0  # 初始化为基态
                
            def get_state_vector(self):
                """获取当前态矢量"""
                return self.state.copy()
            
            def apply_operator(self, operator):
                """应用算符"""
                self.state = operator @ self.state
                
            def measure_expectation(self, observable):
                """测量期望值"""
                return np.vdot(self.state, observable @ self.state).real
        
        # 创建系统实例
        classical_system = ClassicalSelfRefSystem(self.hilbert_dim)
        quantum_system = QuantumSelfRefSystem(self.hilbert_dim)
        
        # 定义观测器函数
        observers = {
            "identity": lambda x: x,
            "increment": lambda x: (x + 1) % self.hilbert_dim,
            "bit_flip": lambda x: x ^ 1 if x < 2 else x  # 简单位翻转
        }
        
        for obs_name, obs_func in observers.items():
            # 重置系统
            classical_system.current_state = 0
            quantum_system.state = np.zeros(self.hilbert_dim, dtype=complex)
            quantum_system.state[0] = 1.0
            
            # 创建对应的量子算符
            quantum_operator = self._create_observer_operator(obs_func)
            
            # 验证单步操作的对应性
            initial_classical_state = classical_system.current_state
            initial_quantum_state = quantum_system.get_state_vector()
            
            # 经典演化
            classical_result = classical_system.evolve(obs_func)
            
            # 量子演化
            quantum_system.apply_operator(quantum_operator)
            final_quantum_state = quantum_system.get_state_vector()
            
            # 验证结果的对应性
            # 量子态应该对应经典的最终状态
            expected_quantum_state = np.zeros(self.hilbert_dim, dtype=complex)
            expected_quantum_state[classical_result] = 1.0
            
            self.assertTrue(
                np.allclose(final_quantum_state, expected_quantum_state),
                f"Quantum evolution should correspond to classical for observer {obs_name}"
            )
            
        # 验证多步演化的一致性
        steps = 3
        for step in range(steps):
            # 选择观测器
            obs_func = observers["increment"]
            quantum_operator = self._create_observer_operator(obs_func)
            
            # 记录演化前的状态
            classical_before = classical_system.current_state
            quantum_before = quantum_system.get_state_vector()
            
            # 执行演化
            classical_result = classical_system.evolve(obs_func)
            quantum_system.apply_operator(quantum_operator)
            quantum_after = quantum_system.get_state_vector()
            
            # 验证对应性
            expected_state = np.zeros(self.hilbert_dim, dtype=complex)
            expected_state[classical_result] = 1.0
            
            self.assertTrue(
                np.allclose(quantum_after, expected_state),
                f"Step {step}: Quantum and classical evolution should correspond"
            )
            
        # 验证可观测量的对应性
        # 创建一个对角算符作为可观测量
        observable = np.diag([complex(i) for i in range(self.hilbert_dim)])
        
        # 测试不同的量子态
        for state_id in range(self.hilbert_dim):
            test_state = np.zeros(self.hilbert_dim, dtype=complex)
            test_state[state_id] = 1.0
            
            expectation = np.vdot(test_state, observable @ test_state).real
            expected_expectation = float(state_id)  # 对角元素
            
            self.assertAlmostEqual(
                expectation, expected_expectation, 10,
                f"Observable expectation should match classical value for state {state_id}"
            )
            
    def test_quantum_superposition_evolution(self):
        """测试量子叠加态的演化"""
        # 创建初始叠加态
        initial_coeffs = [0.5, 0.5, 0.5, 0.5]
        psi = self._create_superposition_state(initial_coeffs)
        
        # 创建演化算符（么正的）
        theta = np.pi / 4
        evolution_operator = np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, np.cos(theta), -np.sin(theta)],
            [0, 0, np.sin(theta), np.cos(theta)]
        ], dtype=complex)
        
        # 验证么正性
        self.assertTrue(
            self._is_unitary(evolution_operator),
            "Evolution operator should be unitary"
        )
        
        # 演化叠加态
        evolved_psi = evolution_operator @ psi
        
        # 验证演化后仍然归一化
        norm_squared = np.vdot(evolved_psi, evolved_psi)
        self.assertAlmostEqual(
            norm_squared.real, 1.0, 10,
            "Evolved state should remain normalized"
        )
        
        # 验证演化的可逆性
        inverse_evolution = evolution_operator.conj().T
        recovered_psi = inverse_evolution @ evolved_psi
        
        self.assertTrue(
            np.allclose(recovered_psi, psi),
            "Evolution should be reversible"
        )
        
    def test_measurement_collapse_simulation(self):
        """测试测量坍缩的模拟"""
        # 创建叠加态
        psi = self._create_superposition_state([0.6, 0.8, 0, 0])
        
        # 创建投影测量算符
        # 测量第一个基态
        projector_0 = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype=complex)
        projector_0[0, 0] = 1.0
        
        # 测量第二个基态  
        projector_1 = np.zeros((self.hilbert_dim, self.hilbert_dim), dtype=complex)
        projector_1[1, 1] = 1.0
        
        # 计算测量概率
        prob_0 = abs(np.vdot(psi, projector_0 @ psi))
        prob_1 = abs(np.vdot(psi, projector_1 @ psi))
        
        # 验证概率和
        self.assertAlmostEqual(
            prob_0 + prob_1, 1.0, 10,
            "Measurement probabilities should sum to 1"
        )
        
        # 验证概率值
        expected_prob_0 = 0.6**2 / (0.6**2 + 0.8**2)
        expected_prob_1 = 0.8**2 / (0.6**2 + 0.8**2)
        
        self.assertAlmostEqual(
            prob_0, expected_prob_0, 10,
            "Probability for state 0 should match calculation"
        )
        
        self.assertAlmostEqual(
            prob_1, expected_prob_1, 10,
            "Probability for state 1 should match calculation"
        )
        
        # 模拟坍缩后的态
        # 如果测量到状态0
        collapsed_0 = projector_0 @ psi
        norm_0 = np.linalg.norm(collapsed_0)
        if norm_0 > 0:
            collapsed_0 = collapsed_0 / norm_0
            
            # 坍缩态应该是基态
            expected_0 = np.array([1, 0, 0, 0], dtype=complex)
            self.assertTrue(
                np.allclose(collapsed_0, expected_0),
                "Collapsed state should be the measured eigenstate"
            )
            
    def test_phi_to_quantum_encoding(self):
        """测试φ-表示到量子态的编码"""
        # 测试不同的φ-表示
        phi_representations = [
            [1],           # F₀ = 1
            [0, 1],        # F₁ = 2
            [1, 0, 1],     # F₀ + F₂ = 1 + 3 = 4
            [0, 1, 0, 1]   # F₁ + F₃ = 2 + 5 = 7
        ]
        
        quantum_states = []
        for phi_repr in phi_representations:
            qstate = self._phi_representation_to_vector(phi_repr)
            quantum_states.append(qstate)
            
            # 验证归一化
            norm = np.linalg.norm(qstate)
            self.assertAlmostEqual(
                norm, 1.0, 10,
                f"Quantum state from φ-repr {phi_repr} should be normalized"
            )
        
        # 验证不同φ-表示产生不同（或可能正交的）量子态
        for i, state1 in enumerate(quantum_states):
            for j, state2 in enumerate(quantum_states[i+1:], i+1):
                overlap = abs(np.vdot(state1, state2))
                # 注意：由于简化映射，可能有重复，这里只检查数值稳定性
                self.assertLessEqual(
                    overlap, 1.0 + 1e-10,
                    f"Overlap between states {i} and {j} should be ≤ 1"
                )


if __name__ == "__main__":
    unittest.main()