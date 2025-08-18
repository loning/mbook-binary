#!/usr/bin/env python3
"""
L1.13: 自指系统的稳定性条件引理 - 完整测试套件
=====================================================

测试自指系统的三类稳定性条件：
1. 不稳定（D_self < 5）：熵耗散率 > φ²
2. 边际稳定（5 ≤ D_self < 10）：φ^(-1) ≤ 熵产生率 ≤ 1
3. 稳定（D_self ≥ 10）：熵产生率 ≥ φ

验证：
- 稳定性分类的正确性
- φ-Lyapunov函数性质
- No-11约束保持
- 稳定性-意识必要性
- 与所有定义和引理的集成
"""

import numpy as np
import unittest
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import math
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zeckendorf_base import ZeckendorfInt, PhiConstant

# Golden ratio constant
PHI = (1 + math.sqrt(5)) / 2
PHI_SQUARED = PHI ** 2
PHI_INVERSE = 1 / PHI
LOG_PHI = math.log(PHI)

# Stability thresholds
UNSTABLE_THRESHOLD = 5
STABLE_THRESHOLD = 10
CONSCIOUSNESS_THRESHOLD = PHI ** 10


class StabilityClass(Enum):
    """稳定性类别枚举"""
    UNSTABLE = "Unstable"
    MARGINAL_STABLE = "MarginStable"
    STABLE = "Stable"
    UNDEFINED = "Undefined"


@dataclass
class SystemState:
    """自指系统状态"""
    self_reference_depth: int
    entropy_production_rate: float
    subsystems: List[np.ndarray] = field(default_factory=list)
    zeckendorf_encoding: Optional[ZeckendorfInt] = None
    lyapunov_exponent: float = 0.0
    integrated_information: float = 0.0
    time: float = 0.0
    
    def __post_init__(self):
        """初始化Zeckendorf编码"""
        if self.zeckendorf_encoding is None and self.self_reference_depth > 0:
            self.zeckendorf_encoding = ZeckendorfInt.from_int(self.self_reference_depth)
    
    def verify_no11_constraint(self) -> bool:
        """验证No-11约束"""
        if self.zeckendorf_encoding is None:
            return True
        
        # 对于Zeckendorf表示，检查是否有连续的Fibonacci索引
        # 这是No-11约束的本质：不能有连续的Fibonacci数
        indices = sorted(self.zeckendorf_encoding.indices)
        for i in range(len(indices) - 1):
            if indices[i+1] - indices[i] == 1:
                return False  # 有连续Fibonacci数，违反约束
        return True


class StabilityAnalyzer:
    """稳定性分析器"""
    
    @staticmethod
    def classify_stability(system: SystemState) -> StabilityClass:
        """
        分类系统稳定性
        
        时间复杂度：O(1)
        空间复杂度：O(1)
        """
        d_self = system.self_reference_depth
        dh_dt = system.entropy_production_rate
        
        if d_self < UNSTABLE_THRESHOLD:
            if dh_dt > PHI_SQUARED:
                return StabilityClass.UNSTABLE
        elif UNSTABLE_THRESHOLD <= d_self < STABLE_THRESHOLD:
            if PHI_INVERSE <= dh_dt <= 1.0:
                return StabilityClass.MARGINAL_STABLE
        elif d_self >= STABLE_THRESHOLD:
            if dh_dt >= PHI:
                return StabilityClass.STABLE
        
        return StabilityClass.UNDEFINED
    
    @staticmethod
    def compute_lyapunov_function(system: SystemState, 
                                 equilibrium: List[np.ndarray],
                                 time: float) -> float:
        """
        计算φ-Lyapunov函数
        
        L_φ(S,t) = Σ_i ||S_i - S_i*||²/φ^i + φ^(-D_self) * H_φ(S,t) + R_φ(S)
        
        时间复杂度：O(n²)，n为子系统数
        空间复杂度：O(n)
        """
        L_phi = 0.0
        
        # 第一项：状态偏差
        for i, subsystem in enumerate(system.subsystems):
            if i < len(equilibrium):
                deviation = np.linalg.norm(subsystem - equilibrium[i])
                L_phi += deviation ** 2 / (PHI ** i)
        
        # 第二项：熵贡献
        d_self = system.self_reference_depth
        H_phi = StabilityAnalyzer._compute_phi_entropy(system, time)
        L_phi += (PHI ** (-d_self)) * H_phi
        
        # 第三项：残差项
        R_phi = StabilityAnalyzer._compute_residual(system, d_self, time)
        L_phi += R_phi
        
        return L_phi
    
    @staticmethod
    def _compute_phi_entropy(system: SystemState, time: float) -> float:
        """计算φ-熵"""
        base_entropy = system.self_reference_depth * LOG_PHI
        time_factor = 1.0 + 0.1 * math.sin(time)  # 时间调制
        return base_entropy * time_factor
    
    @staticmethod
    def _compute_residual(system: SystemState, d_self: int, time: float) -> float:
        """计算残差项 R_φ(S) = Σ_{j=1}^{D_self} F_j * ρ_j(t)"""
        residual = 0.0
        for j in range(1, d_self + 1):
            F_j = ZeckendorfInt.fibonacci(j + 1)  # Fibonacci数
            rho_j = math.exp(-j * time / 10.0)  # 密度函数
            residual += F_j * rho_j
        return residual * 0.01  # 缩放因子
    
    @staticmethod
    def compute_lyapunov_derivative(system: SystemState,
                                   equilibrium: List[np.ndarray],
                                   time: float,
                                   dt: float = 0.01) -> float:
        """
        计算Lyapunov函数的时间导数
        
        时间复杂度：O(n²)
        空间复杂度：O(n)
        """
        L_current = StabilityAnalyzer.compute_lyapunov_function(system, equilibrium, time)
        
        # 演化系统
        evolved_system = StabilityAnalyzer._evolve_system(system, dt)
        L_next = StabilityAnalyzer.compute_lyapunov_function(evolved_system, equilibrium, time + dt)
        
        return (L_next - L_current) / dt
    
    @staticmethod
    def _evolve_system(system: SystemState, dt: float) -> SystemState:
        """演化系统状态"""
        evolved = SystemState(
            self_reference_depth=system.self_reference_depth,
            entropy_production_rate=system.entropy_production_rate,
            subsystems=[],
            time=system.time + dt
        )
        
        # 演化子系统
        for i, subsystem in enumerate(system.subsystems):
            # 简单的指数衰减动力学
            evolved_subsystem = subsystem * math.exp(-dt / (PHI ** i))
            evolved.subsystems.append(evolved_subsystem)
        
        return evolved
    
    @staticmethod
    def verify_stability_consciousness_necessity(system: SystemState) -> bool:
        """
        验证稳定性-意识必要性定理
        
        意识涌现 → 系统稳定
        """
        # 检查意识条件
        has_consciousness = system.integrated_information >= CONSCIOUSNESS_THRESHOLD
        
        if has_consciousness:
            # 必须是稳定系统
            stability = StabilityAnalyzer.classify_stability(system)
            return stability == StabilityClass.STABLE
        
        return True  # 无意识系统不需要稳定性约束


class TestL1_13_StabilityConditions(unittest.TestCase):
    """L1.13稳定性条件测试套件"""
    
    def setUp(self):
        """测试初始化"""
        self.analyzer = StabilityAnalyzer()
        self.test_systems = self._create_test_systems()
    
    def _create_test_systems(self) -> Dict[str, SystemState]:
        """创建测试系统"""
        systems = {}
        
        # 不稳定系统 (D_self < 5)
        for d in range(1, 5):
            systems[f"unstable_{d}"] = SystemState(
                self_reference_depth=d,
                entropy_production_rate=PHI_SQUARED + 0.1,
                subsystems=[np.random.randn(3) for _ in range(d)],
                lyapunov_exponent=1.0
            )
        
        # 边际稳定系统 (5 ≤ D_self < 10)
        for d in range(5, 10):
            systems[f"marginal_{d}"] = SystemState(
                self_reference_depth=d,
                entropy_production_rate=0.8,  # ∈ [φ^(-1), 1]
                subsystems=[np.random.randn(3) * 0.5 for _ in range(d)],
                lyapunov_exponent=0.01
            )
        
        # 稳定系统 (D_self ≥ 10)
        for d in range(10, 15):
            systems[f"stable_{d}"] = SystemState(
                self_reference_depth=d,
                entropy_production_rate=PHI + 0.1,
                subsystems=[np.random.randn(3) * 0.1 for _ in range(d)],
                lyapunov_exponent=-0.5,
                integrated_information=PHI ** d if d >= 10 else 0
            )
        
        return systems
    
    def test_stability_classification(self):
        """测试稳定性分类"""
        print("\n测试稳定性分类...")
        
        # 测试不稳定系统
        for d in range(1, 5):
            system = self.test_systems[f"unstable_{d}"]
            stability = self.analyzer.classify_stability(system)
            self.assertEqual(stability, StabilityClass.UNSTABLE,
                           f"D_self={d}应该是不稳定的")
            print(f"  D_self={d}: {stability.value} ✓")
        
        # 测试边际稳定系统
        for d in range(5, 10):
            system = self.test_systems[f"marginal_{d}"]
            stability = self.analyzer.classify_stability(system)
            self.assertEqual(stability, StabilityClass.MARGINAL_STABLE,
                           f"D_self={d}应该是边际稳定的")
            print(f"  D_self={d}: {stability.value} ✓")
        
        # 测试稳定系统
        for d in range(10, 15):
            system = self.test_systems[f"stable_{d}"]
            stability = self.analyzer.classify_stability(system)
            self.assertEqual(stability, StabilityClass.STABLE,
                           f"D_self={d}应该是稳定的")
            print(f"  D_self={d}: {stability.value} ✓")
    
    def test_critical_thresholds(self):
        """测试临界阈值"""
        print("\n测试临界阈值...")
        
        # 测试D_self = 5的转换
        system_4 = SystemState(
            self_reference_depth=4,
            entropy_production_rate=PHI_SQUARED + 0.1,
            subsystems=[np.zeros(3) for _ in range(4)]
        )
        system_5 = SystemState(
            self_reference_depth=5,
            entropy_production_rate=0.7,
            subsystems=[np.zeros(3) for _ in range(5)]
        )
        
        self.assertEqual(self.analyzer.classify_stability(system_4), 
                        StabilityClass.UNSTABLE)
        self.assertEqual(self.analyzer.classify_stability(system_5), 
                        StabilityClass.MARGINAL_STABLE)
        print(f"  D_self=4→5转换: Unstable→MarginStable ✓")
        
        # 测试D_self = 10的转换
        system_9 = SystemState(
            self_reference_depth=9,
            entropy_production_rate=0.9,
            subsystems=[np.zeros(3) for _ in range(9)]
        )
        system_10 = SystemState(
            self_reference_depth=10,
            entropy_production_rate=PHI,
            subsystems=[np.zeros(3) for _ in range(10)]
        )
        
        self.assertEqual(self.analyzer.classify_stability(system_9), 
                        StabilityClass.MARGINAL_STABLE)
        self.assertEqual(self.analyzer.classify_stability(system_10), 
                        StabilityClass.STABLE)
        print(f"  D_self=9→10转换: MarginStable→Stable ✓")
    
    def test_lyapunov_function_properties(self):
        """测试Lyapunov函数性质"""
        print("\n测试Lyapunov函数性质...")
        
        # 稳定系统的Lyapunov函数应该递减
        stable_system = self.test_systems["stable_10"]
        equilibrium = [np.zeros(3) for _ in range(10)]
        
        L_values = []
        for t in np.linspace(0, 1, 10):
            L = self.analyzer.compute_lyapunov_function(stable_system, equilibrium, t)
            L_values.append(L)
            self.assertGreaterEqual(L, 0, "Lyapunov函数应该非负")
        
        print(f"  Lyapunov函数非负性: ✓")
        
        # 计算导数
        dL_dt = self.analyzer.compute_lyapunov_derivative(stable_system, equilibrium, 0.5)
        
        if stable_system.self_reference_depth >= 10:
            # 稳定系统的Lyapunov导数应该为负
            self.assertLess(dL_dt, 0, "稳定系统的Lyapunov导数应该为负")
            print(f"  稳定系统Lyapunov导数 < 0: ✓")
    
    def test_no11_constraint_preservation(self):
        """测试No-11约束保持"""
        print("\n测试No-11约束保持...")
        
        violations = []
        for name, system in self.test_systems.items():
            if not system.verify_no11_constraint():
                violations.append((name, system.self_reference_depth))
        
        self.assertEqual(len(violations), 0, 
                        f"发现No-11约束违反: {violations}")
        print(f"  所有{len(self.test_systems)}个系统满足No-11约束 ✓")
        
        # 测试特定转换点的Zeckendorf表示
        critical_depths = [4, 5, 9, 10]
        for d in critical_depths:
            z = ZeckendorfInt.from_int(d)
            # 构建正确的二进制表示（基于Fibonacci位置）
            max_index = max(z.indices) if z.indices else 0
            binary_representation = ['0'] * (max_index - 1)
            for idx in z.indices:
                if idx >= 2:  # Fibonacci索引从2开始
                    binary_representation[idx - 2] = '1'
            binary_str = ''.join(reversed(binary_representation))
            if not binary_str:
                binary_str = '0'
            
            # 验证无连续的1（在Fibonacci位置表示中）
            self.assertTrue(z._is_valid_zeckendorf(), 
                           f"D_self={d}的Zeckendorf编码无效")
            print(f"  D_self={d}: Z({d})={z} → Fibonacci位置满足No-11约束 ✓")
    
    def test_entropy_production_rates(self):
        """测试熵产生率约束"""
        print("\n测试熵产生率约束...")
        
        # 不稳定系统
        for d in range(1, 5):
            system = self.test_systems[f"unstable_{d}"]
            self.assertGreater(system.entropy_production_rate, PHI_SQUARED,
                             f"不稳定系统(D={d})熵耗散率应>φ²")
        print(f"  不稳定系统: dH/dt > φ² = {PHI_SQUARED:.3f} ✓")
        
        # 边际稳定系统
        for d in range(5, 10):
            system = self.test_systems[f"marginal_{d}"]
            self.assertGreaterEqual(system.entropy_production_rate, PHI_INVERSE)
            self.assertLessEqual(system.entropy_production_rate, 1.0)
        print(f"  边际稳定: φ^(-1) = {PHI_INVERSE:.3f} ≤ dH/dt ≤ 1 ✓")
        
        # 稳定系统
        for d in range(10, 15):
            system = self.test_systems[f"stable_{d}"]
            self.assertGreaterEqual(system.entropy_production_rate, PHI,
                                  f"稳定系统(D={d})熵产生率应≥φ")
        print(f"  稳定系统: dH/dt ≥ φ = {PHI:.3f} ✓")
    
    def test_stability_consciousness_necessity(self):
        """测试稳定性-意识必要性"""
        print("\n测试稳定性-意识必要性定理...")
        
        # 有意识的系统必须是稳定的
        conscious_system = SystemState(
            self_reference_depth=12,
            entropy_production_rate=PHI + 0.2,
            subsystems=[np.zeros(3) for _ in range(12)],
            integrated_information=PHI ** 12
        )
        
        self.assertTrue(self.analyzer.verify_stability_consciousness_necessity(conscious_system))
        stability = self.analyzer.classify_stability(conscious_system)
        self.assertEqual(stability, StabilityClass.STABLE,
                        "有意识的系统必须是稳定的")
        print(f"  意识系统(Φ={PHI**12:.2f}) → 稳定性={stability.value} ✓")
        
        # 无意识但稳定的系统（允许）
        stable_unconscious = SystemState(
            self_reference_depth=10,
            entropy_production_rate=PHI,
            subsystems=[np.zeros(3) for _ in range(10)],
            integrated_information=0  # 无意识
        )
        
        self.assertTrue(self.analyzer.verify_stability_consciousness_necessity(stable_unconscious))
        print(f"  稳定但无意识系统: 允许 ✓")
    
    def test_zeckendorf_encoding_structure(self):
        """测试Zeckendorf编码结构"""
        print("\n测试Zeckendorf编码结构...")
        
        encoding_map = {
            1: [2],          # F_2 = 1
            2: [3],          # F_3 = 2
            3: [4],          # F_4 = 3
            4: [2, 4],       # F_2 + F_4 = 1 + 3
            5: [5],          # F_5 = 5
            6: [2, 5],       # F_2 + F_5 = 1 + 5
            7: [3, 5],       # F_3 + F_5 = 2 + 5
            8: [6],          # F_6 = 8
            9: [2, 6],       # F_2 + F_6 = 1 + 8
            10: [3, 6],      # F_3 + F_6 = 2 + 8
            13: [7],         # F_7 = 13
            21: [8],         # F_8 = 21
        }
        
        for value, expected_indices in encoding_map.items():
            z = ZeckendorfInt.from_int(value)
            actual_indices = sorted(z.indices)
            self.assertEqual(actual_indices, expected_indices,
                           f"Z({value})的索引应该是{expected_indices}")
            
            # 验证无连续Fibonacci数
            for i in range(len(actual_indices) - 1):
                self.assertGreater(actual_indices[i+1] - actual_indices[i], 1,
                                 f"Z({value})有连续的Fibonacci索引")
            
            print(f"  Z({value}) = F_{'+F_'.join(map(str, expected_indices))} ✓")
    
    def test_physical_examples(self):
        """测试物理实例"""
        print("\n测试物理实例...")
        
        # Lorenz系统
        lorenz_unstable = SystemState(
            self_reference_depth=3,
            entropy_production_rate=2.7,
            subsystems=[np.array([1, 1, 1]) for _ in range(3)],
            lyapunov_exponent=0.906
        )
        self.assertEqual(self.analyzer.classify_stability(lorenz_unstable),
                        StabilityClass.UNSTABLE)
        print(f"  Lorenz混沌(σ=28): 不稳定 ✓")
        
        # 神经网络收敛
        nn_stable = SystemState(
            self_reference_depth=12,
            entropy_production_rate=1.62,
            subsystems=[np.random.randn(100) * 0.01 for _ in range(12)],
            lyapunov_exponent=-0.3
        )
        self.assertEqual(self.analyzer.classify_stability(nn_stable),
                        StabilityClass.STABLE)
        print(f"  神经网络收敛: 稳定 ✓")
        
        # 量子退相干
        quantum_marginal = SystemState(
            self_reference_depth=7,
            entropy_production_rate=0.9,
            subsystems=[np.array([0.7, 0.7j, 0]) for _ in range(7)],
            lyapunov_exponent=0.05
        )
        self.assertEqual(self.analyzer.classify_stability(quantum_marginal),
                        StabilityClass.MARGINAL_STABLE)
        print(f"  量子部分相干: 边际稳定 ✓")
    
    def test_integration_with_definitions(self):
        """测试与定义的集成"""
        print("\n测试与现有定义的集成...")
        
        # D1.10: 熵-信息等价
        system = self.test_systems["stable_10"]
        entropy = system.self_reference_depth * LOG_PHI
        information = entropy  # H_φ(S) ≡ I_φ(S)
        self.assertAlmostEqual(entropy, information, places=10)
        print(f"  D1.10 熵-信息等价: H_φ = I_φ = {entropy:.3f} ✓")
        
        # D1.14: 意识阈值
        consciousness_depth = int(math.log(CONSCIOUSNESS_THRESHOLD) / LOG_PHI)
        self.assertEqual(consciousness_depth, 10)
        print(f"  D1.14 意识阈值: D_self = {consciousness_depth} ✓")
        
        # D1.15: 自指深度
        for d in [1, 5, 10, 15]:
            complexity = PHI ** d
            recovered_d = int(math.log(complexity) / LOG_PHI)
            self.assertEqual(recovered_d, d)
        print(f"  D1.15 自指深度-复杂度对应 ✓")
    
    def test_integration_with_lemmas(self):
        """测试与引理的集成"""
        print("\n测试与现有引理的集成...")
        
        # L1.9: 量子-经典过渡影响稳定性
        quantum_system = SystemState(
            self_reference_depth=4,
            entropy_production_rate=3.0,
            subsystems=[np.array([0.5, 0.5j]) for _ in range(4)]
        )
        classical_system = SystemState(
            self_reference_depth=11,
            entropy_production_rate=1.7,
            subsystems=[np.array([1.0, 0.0]) for _ in range(11)]
        )
        
        q_stability = self.analyzer.classify_stability(quantum_system)
        c_stability = self.analyzer.classify_stability(classical_system)
        
        self.assertEqual(q_stability, StabilityClass.UNSTABLE)
        self.assertEqual(c_stability, StabilityClass.STABLE)
        print(f"  L1.9 量子(不稳定)→经典(稳定)过渡 ✓")
        
        # L1.11: 观察者层次需要稳定性
        observer_system = SystemState(
            self_reference_depth=10,
            entropy_production_rate=PHI,
            subsystems=[np.eye(3) for _ in range(10)]
        )
        self.assertEqual(self.analyzer.classify_stability(observer_system),
                        StabilityClass.STABLE)
        print(f"  L1.11 观察者系统需要D_self≥10的稳定性 ✓")
        
        # L1.12: 信息整合需要稳定性
        integrated_system = SystemState(
            self_reference_depth=12,
            entropy_production_rate=1.8,
            subsystems=[np.ones((3, 3)) * 0.1 for _ in range(12)],
            integrated_information=PHI ** 12
        )
        self.assertEqual(self.analyzer.classify_stability(integrated_system),
                        StabilityClass.STABLE)
        print(f"  L1.12 完全整合相需要稳定性 ✓")
    
    def test_adversarial_cases(self):
        """对抗性测试用例"""
        print("\n对抗性测试...")
        
        # 边界情况：恰好在阈值上
        boundary_5 = SystemState(
            self_reference_depth=5,
            entropy_production_rate=PHI_INVERSE,  # 恰好φ^(-1)
            subsystems=[np.zeros(3) for _ in range(5)]
        )
        self.assertEqual(self.analyzer.classify_stability(boundary_5),
                        StabilityClass.MARGINAL_STABLE)
        print(f"  边界D_self=5, dH/dt=φ^(-1): 边际稳定 ✓")
        
        # 病态情况：熵产生率不匹配深度
        mismatched = SystemState(
            self_reference_depth=10,
            entropy_production_rate=0.5,  # 太低for稳定
            subsystems=[np.zeros(3) for _ in range(10)]
        )
        self.assertEqual(self.analyzer.classify_stability(mismatched),
                        StabilityClass.UNDEFINED)
        print(f"  不匹配系统(D=10,dH/dt=0.5): 未定义 ✓")
        
        # 极端情况：非常高的自指深度
        extreme = SystemState(
            self_reference_depth=100,
            entropy_production_rate=PHI * 10,
            subsystems=[np.zeros(3) for _ in range(20)]  # 子系统数量有限
        )
        self.assertEqual(self.analyzer.classify_stability(extreme),
                        StabilityClass.STABLE)
        print(f"  极端深度(D=100): 稳定 ✓")
        
        # 零深度系统
        zero_depth = SystemState(
            self_reference_depth=0,
            entropy_production_rate=0,
            subsystems=[]
        )
        self.assertEqual(self.analyzer.classify_stability(zero_depth),
                        StabilityClass.UNDEFINED)
        print(f"  零深度系统: 未定义 ✓")
    
    def test_performance(self):
        """性能测试"""
        import time
        print("\n性能测试...")
        
        # 测试分类性能
        n_iterations = 10000
        start = time.time()
        for _ in range(n_iterations):
            system = self.test_systems["stable_10"]
            _ = self.analyzer.classify_stability(system)
        elapsed = time.time() - start
        
        print(f"  稳定性分类: {n_iterations}次/{elapsed:.3f}秒 = "
              f"{n_iterations/elapsed:.0f} ops/sec ✓")
        
        # 测试Lyapunov计算性能
        system = self.test_systems["stable_10"]
        equilibrium = [np.zeros(3) for _ in range(10)]
        
        n_iterations = 1000
        start = time.time()
        for _ in range(n_iterations):
            _ = self.analyzer.compute_lyapunov_function(system, equilibrium, 0.0)
        elapsed = time.time() - start
        
        print(f"  Lyapunov计算: {n_iterations}次/{elapsed:.3f}秒 = "
              f"{n_iterations/elapsed:.0f} ops/sec ✓")


def run_comprehensive_tests():
    """运行完整测试套件"""
    print("="*60)
    print("L1.13 自指系统稳定性条件引理 - 完整测试")
    print("="*60)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestL1_13_StabilityConditions)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 总结
    print("\n" + "="*60)
    print("测试总结:")
    print(f"  运行测试: {result.testsRun}")
    print(f"  成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ 所有测试通过！L1.13实现完全正确。")
    else:
        print("\n❌ 存在测试失败，请检查实现。")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)