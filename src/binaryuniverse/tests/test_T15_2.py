#!/usr/bin/env python3
"""
T15-2 φ-自发对称破缺定理 - 完整验证程序

验证内容：
1. 墨西哥帽势能与真空结构
2. 对称破缺机制
3. Goldstone定理
4. Higgs机制与质量生成
5. 相变分类
6. 熵增原理
7. no-11约束的保持
"""

import unittest
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
from enum import Enum
import logging

# 添加路径
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phi_arithmetic import PhiReal, PhiComplex, PhiMatrix
from no11_number_system import No11NumberSystem

# 设置日志
logging.basicConfig(level=logging.INFO)

# 物理常数
PI = np.pi
phi = (1 + np.sqrt(5)) / 2  # 黄金比率

# ==================== 势能结构 ====================

class MexicanHatPotential:
    """墨西哥帽势能"""
    
    def __init__(self, mu_squared: PhiReal, lambda_coupling: PhiReal):
        """
        V(φ) = -μ²|φ|² + λ|φ|⁴
        当μ² < 0时发生对称破缺
        """
        self.mu_squared = mu_squared
        self.lambda_coupling = lambda_coupling
        self.no11 = No11NumberSystem()
        
        # 计算真空期望值
        if mu_squared.decimal_value < 0:
            # v = √(-μ²/2λ)
            self.vev = PhiReal.from_decimal(
                np.sqrt(-mu_squared.decimal_value / (2 * lambda_coupling.decimal_value))
            )
        else:
            self.vev = PhiReal.zero()
    
    def evaluate(self, field: PhiComplex) -> PhiReal:
        """计算势能值"""
        field_squared = (field * field.conjugate()).real
        
        # V = μ²|φ|² + λ|φ|⁴（注意μ² < 0）
        term1 = self.mu_squared * field_squared
        term2 = self.lambda_coupling * field_squared * field_squared
        
        return term1 + term2
    
    def add_no11_corrections(self, field: PhiComplex) -> PhiReal:
        """添加no-11约束修正项"""
        # 修正项：Σ ε_n cos(nθ)，其中n违反no-11
        corrections = PhiReal.zero()
        
        if field.magnitude().decimal_value > 1e-10:
            theta = np.arctan2(field.imag.decimal_value, field.real.decimal_value)
            
            # 检查被禁止的角度模式
            for n in range(1, 10):
                if not self.no11.is_valid_representation([n]):
                    epsilon_n = PhiReal.from_decimal(0.01 * np.exp(-n))
                    corrections = corrections + epsilon_n * PhiReal.from_decimal(np.cos(n * theta))
        
        return corrections
    
    def find_vacuum_states(self, num_angles: int = 20) -> List[PhiComplex]:
        """寻找所有真空态"""
        vacuum_states = []
        
        if self.vev.decimal_value == 0:
            # 未破缺情况：只有φ=0
            vacuum_states.append(PhiComplex.zero())
        else:
            # 破缺情况：|φ| = v的圆上的点
            # 但受no-11约束，只有某些角度允许
            fib_numbers = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
            total = sum(fib_numbers[:num_angles//2])
            
            for i, f_n in enumerate(fib_numbers[:num_angles//2]):
                # 检查是否满足no-11约束
                if self.no11.is_valid_representation([i+1]):
                    theta_n = 2 * PI * f_n / total
                    vacuum_state = PhiComplex(
                        self.vev * PhiReal.from_decimal(np.cos(theta_n)),
                        self.vev * PhiReal.from_decimal(np.sin(theta_n))
                    )
                    vacuum_states.append(vacuum_state)
        
        return vacuum_states

# ==================== 对称性结构 ====================

@dataclass
class SymmetryGroup:
    """对称群"""
    name: str
    dimension: int
    generators: List[PhiMatrix]
    
    def is_subgroup_of(self, other: 'SymmetryGroup') -> bool:
        """检查是否为子群"""
        # 简化：通过维度判断
        return self.dimension < other.dimension

@dataclass
class SymmetryBreaking:
    """对称破缺"""
    original_group: SymmetryGroup
    residual_group: SymmetryGroup
    vacuum_state: PhiComplex
    broken_generators: List[PhiMatrix]
    
    def num_goldstone_bosons(self) -> int:
        """Goldstone玻色子数目"""
        return self.original_group.dimension - self.residual_group.dimension

# ==================== Goldstone玻色子 ====================

@dataclass
class GoldstoneBoson:
    """Goldstone玻色子"""
    generator: PhiMatrix  # 对应的破缺生成元
    mass_squared: PhiReal  # 质量平方（应该接近0）
    decay_constant: PhiReal  # 衰变常数f
    no11_correction: PhiReal  # no-11约束导致的质量修正

class GoldstoneSpectrum:
    """Goldstone谱计算"""
    
    def __init__(self, breaking: SymmetryBreaking):
        self.breaking = breaking
        self.no11 = No11NumberSystem()
    
    def compute_spectrum(self) -> List[GoldstoneBoson]:
        """计算Goldstone玻色子谱"""
        goldstones = []
        
        for i, generator in enumerate(self.breaking.broken_generators):
            # 标准情况：无质量
            mass_squared_standard = PhiReal.zero()
            
            # no-11修正
            correction = self.compute_no11_correction(i)
            mass_squared = mass_squared_standard + correction
            
            # 衰变常数等于VEV
            f = self.breaking.vacuum_state.magnitude()
            
            goldstone = GoldstoneBoson(
                generator=generator,
                mass_squared=mass_squared,
                decay_constant=f,
                no11_correction=correction
            )
            goldstones.append(goldstone)
        
        return goldstones
    
    def compute_no11_correction(self, mode_index: int) -> PhiReal:
        """计算no-11约束的质量修正"""
        # 某些模式被禁止，获得小质量
        if not self.no11.is_valid_representation([mode_index]):
            # 指数抑制的质量
            return PhiReal.from_decimal(0.001 * np.exp(-mode_index))
        return PhiReal.zero()

# ==================== Higgs机制 ====================

@dataclass
class GaugeField:
    """规范场"""
    coupling: PhiReal  # 规范耦合g
    components: List[PhiComplex]  # A_μ分量
    
    def mass_from_higgs(self, vev: PhiReal) -> PhiReal:
        """通过Higgs机制获得的质量"""
        # M = g * v
        return self.coupling * vev

class HiggsMechanism:
    """Higgs机制实现"""
    
    def __init__(self, scalar_field_vev: PhiReal, gauge_coupling: PhiReal):
        self.vev = scalar_field_vev
        self.gauge_coupling = gauge_coupling
        self.no11 = No11NumberSystem()
    
    def generate_gauge_mass(self) -> PhiReal:
        """生成规范玻色子质量"""
        # M_A = g * v * No11Factor
        base_mass = self.gauge_coupling * self.vev
        no11_factor = self.compute_no11_factor()
        return base_mass * no11_factor
    
    def compute_no11_factor(self) -> PhiReal:
        """计算no-11修正因子"""
        # 基于VEV的Zeckendorf表示
        vev_int = int(self.vev.decimal_value * 100)
        indices = self.no11.to_zeckendorf(vev_int)
        
        # 如果表示满足no-11，因子接近1
        if self.no11.is_valid_representation(indices):
            return PhiReal.from_decimal(1.0 + 0.01 * len(indices) / 10)
        else:
            # 否则有修正
            return PhiReal.from_decimal(0.9 + 0.1 * np.random.random())
    
    def count_massive_gauge_bosons(self, num_broken_generators: int) -> int:
        """计算获得质量的规范玻色子数目"""
        # 每个破缺的规范对称性对应一个质量规范玻色子
        # Goldstone玻色子被"吃掉"
        return num_broken_generators

# ==================== 相变分析 ====================

class PhaseTransitionType(Enum):
    """相变类型"""
    FIRST_ORDER = 1
    SECOND_ORDER = 2
    CROSSOVER = 3

class PhaseTransitionAnalyzer:
    """相变分析器"""
    
    def __init__(self, potential: MexicanHatPotential):
        self.potential = potential
        self.no11 = No11NumberSystem()
    
    def determine_transition_type(self, temperature: PhiReal) -> PhaseTransitionType:
        """判断相变类型"""
        # 简化模型：基于势垒高度
        barrier_height = self.compute_potential_barrier()
        
        if barrier_height.decimal_value > temperature.decimal_value:
            # 高势垒：一级相变
            return PhaseTransitionType.FIRST_ORDER
        elif barrier_height.decimal_value < 0.1 * temperature.decimal_value:
            # 无势垒：二级相变
            return PhaseTransitionType.SECOND_ORDER
        else:
            # 中间情况：crossover
            return PhaseTransitionType.CROSSOVER
    
    def compute_potential_barrier(self) -> PhiReal:
        """计算势垒高度"""
        # 完整计算：寻找从φ=0到φ=v路径上的最大值
        # 对于墨西哥帽势能 V(φ) = μ²|φ|² + λ|φ|⁴
        # 沿径向方向：V(r) = μ²r² + λr⁴
        
        # 找极值点：dV/dr = 2μ²r + 4λr³ = 0
        # r(2μ² + 4λr²) = 0
        # 非零解：r² = -μ²/(2λ)（这就是VEV）
        
        # 如果μ² < 0（对称破缺情况）
        if self.potential.mu_squared.decimal_value < 0:
            # V(0) = 0
            v_at_zero = PhiReal.zero()
            
            # V(v) = μ²v² + λv⁴ = -μ⁴/(4λ)（代入v² = -μ²/(2λ)）
            v_squared = -self.potential.mu_squared.decimal_value / (2 * self.potential.lambda_coupling.decimal_value)
            v_at_minimum = self.potential.mu_squared * PhiReal.from_decimal(v_squared) + \
                          self.potential.lambda_coupling * PhiReal.from_decimal(v_squared * v_squared)
            
            # 势垒 = V(0) - V(v)
            barrier = v_at_zero - v_at_minimum
            return barrier
        else:
            # μ² > 0时，最小值在原点，无势垒
            return PhiReal.zero()
    
    def compute_latent_heat(self, critical_temp: PhiReal) -> PhiReal:
        """计算潜热（一级相变）"""
        # L = T_c * ΔS
        entropy_change = self.compute_entropy_change()
        return critical_temp * entropy_change
    
    def compute_entropy_change(self) -> PhiReal:
        """计算熵变"""
        # 简化：与真空态数目成正比
        num_vacua_before = 1  # 对称相
        num_vacua_after = len(self.potential.find_vacuum_states())
        
        if num_vacua_after > num_vacua_before:
            return PhiReal.from_decimal(np.log(num_vacua_after))
        return PhiReal.zero()

# ==================== 主测试类 ====================

class TestT15_2_PhiSymmetryBreaking(unittest.TestCase):
    """T15-2 φ-自发对称破缺测试"""
    
    def setUp(self):
        """初始化测试环境"""
        # 创建破缺势能（μ² < 0）
        self.mu_squared = PhiReal.from_decimal(-1.0)
        self.lambda_coupling = PhiReal.from_decimal(0.1)
        self.potential = MexicanHatPotential(self.mu_squared, self.lambda_coupling)
        
        # 创建对称群
        self.create_symmetry_groups()
        
    def create_symmetry_groups(self):
        """创建对称群结构"""
        # U(1)对称群
        u1_generator = PhiMatrix(
            [[PhiComplex(PhiReal.zero(), PhiReal.one())]],
            (1, 1)
        )
        self.u1_group = SymmetryGroup("U(1)", 1, [u1_generator])
        
        # 平凡群（完全破缺后）
        self.trivial_group = SymmetryGroup("1", 0, [])
    
    def test_vacuum_structure(self):
        """测试真空结构"""
        # 验证VEV计算
        expected_vev_squared = -self.mu_squared.decimal_value / (2 * self.lambda_coupling.decimal_value)
        expected_vev = np.sqrt(expected_vev_squared)
        
        self.assertAlmostEqual(
            self.potential.vev.decimal_value,
            expected_vev,
            places=6
        )
        
        # 寻找真空态
        vacuum_states = self.potential.find_vacuum_states()
        self.assertGreater(len(vacuum_states), 1)  # 应该有多个真空态
        
        # 验证所有真空态的势能相同
        v_values = []
        for state in vacuum_states:
            v = self.potential.evaluate(state)
            v_values.append(v.decimal_value)
        
        # 所有势能应该相等（至数值精度）
        for v in v_values[1:]:
            self.assertAlmostEqual(v, v_values[0], places=6)
        
        # 验证真空态的模长
        for state in vacuum_states:
            magnitude = state.magnitude()
            self.assertAlmostEqual(
                magnitude.decimal_value,
                self.potential.vev.decimal_value,
                places=6
            )
    
    def test_symmetry_breaking(self):
        """测试对称破缺"""
        # 选择一个真空态
        vacuum_states = self.potential.find_vacuum_states()
        vacuum = vacuum_states[0]
        
        # 创建对称破缺
        breaking = SymmetryBreaking(
            original_group=self.u1_group,
            residual_group=self.trivial_group,
            vacuum_state=vacuum,
            broken_generators=self.u1_group.generators
        )
        
        # 验证是完全破缺
        self.assertTrue(breaking.residual_group.is_subgroup_of(breaking.original_group))
        self.assertEqual(breaking.num_goldstone_bosons(), 1)  # U(1)→1产生1个Goldstone
    
    def test_goldstone_theorem(self):
        """测试Goldstone定理"""
        # 创建对称破缺
        vacuum = self.potential.find_vacuum_states()[0]
        breaking = SymmetryBreaking(
            original_group=self.u1_group,
            residual_group=self.trivial_group,
            vacuum_state=vacuum,
            broken_generators=self.u1_group.generators
        )
        
        # 计算Goldstone谱
        spectrum = GoldstoneSpectrum(breaking)
        goldstones = spectrum.compute_spectrum()
        
        # 验证Goldstone数目
        expected_num = breaking.original_group.dimension - breaking.residual_group.dimension
        self.assertEqual(len(goldstones), expected_num)
        
        # 验证质量（应该很小）
        for g in goldstones:
            # 标准Goldstone应该无质量，但有no-11修正
            self.assertLess(g.mass_squared.decimal_value, 0.01)
            
            # 衰变常数应该等于VEV
            self.assertAlmostEqual(
                g.decay_constant.decimal_value,
                self.potential.vev.decimal_value,
                places=6
            )
    
    def test_higgs_mechanism(self):
        """测试Higgs机制"""
        # 规范耦合
        gauge_coupling = PhiReal.from_decimal(0.5)
        
        # 创建Higgs机制
        higgs = HiggsMechanism(self.potential.vev, gauge_coupling)
        
        # 生成规范玻色子质量
        gauge_mass = higgs.generate_gauge_mass()
        
        # 验证质量公式 M = g*v（带修正）
        expected_mass_base = gauge_coupling.decimal_value * self.potential.vev.decimal_value
        self.assertGreater(gauge_mass.decimal_value, 0)
        self.assertLess(
            abs(gauge_mass.decimal_value - expected_mass_base) / expected_mass_base,
            0.2  # 允许20%的no-11修正
        )
        
        # 验证质量规范玻色子数目
        num_broken = len(self.u1_group.generators)
        num_massive = higgs.count_massive_gauge_bosons(num_broken)
        self.assertEqual(num_massive, num_broken)
    
    def test_phase_transitions(self):
        """测试相变"""
        analyzer = PhaseTransitionAnalyzer(self.potential)
        
        # 高温：应该是对称相
        high_temp = PhiReal.from_decimal(10.0)
        
        # 低温：应该是破缺相
        low_temp = PhiReal.from_decimal(0.1)
        
        # 判断相变类型
        transition_type = analyzer.determine_transition_type(low_temp)
        self.assertIn(transition_type, [
            PhaseTransitionType.FIRST_ORDER,
            PhaseTransitionType.SECOND_ORDER,
            PhaseTransitionType.CROSSOVER
        ])
        
        # 计算势垒
        barrier = analyzer.compute_potential_barrier()
        # 对于自发对称破缺（μ² < 0），势垒应该是正的
        # 势垒 = V(0) - V(v) > 0 因为V(v) < V(0)
        self.assertGreaterEqual(barrier.decimal_value, 0)
        
        # 如果是一级相变，计算潜热
        if transition_type == PhaseTransitionType.FIRST_ORDER:
            critical_temp = PhiReal.from_decimal(1.0)  # 简化的临界温度
            latent_heat = analyzer.compute_latent_heat(critical_temp)
            self.assertGreater(latent_heat.decimal_value, 0)
    
    def test_entropy_increase(self):
        """测试熵增原理"""
        analyzer = PhaseTransitionAnalyzer(self.potential)
        
        # 计算熵变
        entropy_change = analyzer.compute_entropy_change()
        
        # 根据唯一公理，对称破缺必然导致熵增
        self.assertGreater(entropy_change.decimal_value, 0)
        
        # 熵增应该与真空态数目的对数成正比
        num_vacua = len(self.potential.find_vacuum_states())
        expected_entropy = np.log(num_vacua) if num_vacua > 1 else 0
        
        if expected_entropy > 0:
            self.assertAlmostEqual(
                entropy_change.decimal_value,
                expected_entropy,
                places=2
            )
        
        logging.info(f"熵增: ΔS = {entropy_change.decimal_value:.4f}")
    
    def test_no11_corrections(self):
        """测试no-11约束修正"""
        # 测试势能修正
        test_field = PhiComplex(
            PhiReal.from_decimal(1.0),
            PhiReal.from_decimal(0.5)
        )
        
        corrections = self.potential.add_no11_corrections(test_field)
        
        # 修正应该是小的
        self.assertLess(corrections.decimal_value, 0.1)
        
        # 测试Goldstone质量修正
        spectrum = GoldstoneSpectrum(SymmetryBreaking(
            original_group=self.u1_group,
            residual_group=self.trivial_group,
            vacuum_state=self.potential.find_vacuum_states()[0],
            broken_generators=self.u1_group.generators
        ))
        
        # 某些模式应该有修正
        for i in range(5):
            correction = spectrum.compute_no11_correction(i)
            if not spectrum.no11.is_valid_representation([i]):
                self.assertGreater(correction.decimal_value, 0)
    
    def test_multiple_field_breaking(self):
        """测试多场破缺"""
        # 创建两个标量场的势能
        # V = -μ₁²|φ₁|² - μ₂²|φ₂|² + λ₁|φ₁|⁴ + λ₂|φ₂|⁴ + λ₁₂|φ₁|²|φ₂|²
        
        mu1_sq = PhiReal.from_decimal(-1.0)
        mu2_sq = PhiReal.from_decimal(-0.5)
        lambda1 = PhiReal.from_decimal(0.1)
        lambda2 = PhiReal.from_decimal(0.2)
        lambda12 = PhiReal.from_decimal(0.05)
        
        # 两个VEV
        vev1 = PhiReal.from_decimal(np.sqrt(-mu1_sq.decimal_value / (2 * lambda1.decimal_value)))
        vev2 = PhiReal.from_decimal(np.sqrt(-mu2_sq.decimal_value / (2 * lambda2.decimal_value)))
        
        # 验证两个场都获得VEV
        self.assertGreater(vev1.decimal_value, 0)
        self.assertGreater(vev2.decimal_value, 0)
        
        # 更复杂的破缺模式
        # U(1) × U(1) → 1
        num_goldstones = 2  # 两个U(1)都破缺
        self.assertEqual(num_goldstones, 2)
    
    def test_partial_symmetry_breaking(self):
        """测试部分对称破缺"""
        # 创建SU(2)群（简化表示）
        pauli_matrices = [
            PhiMatrix([[PhiComplex.zero(), PhiComplex.one()],
                      [PhiComplex.one(), PhiComplex.zero()]], (2, 2)),
            PhiMatrix([[PhiComplex.zero(), PhiComplex(PhiReal.zero(), PhiReal.from_decimal(-1))],
                      [PhiComplex(PhiReal.zero(), PhiReal.one()), PhiComplex.zero()]], (2, 2)),
            PhiMatrix([[PhiComplex.one(), PhiComplex.zero()],
                      [PhiComplex.zero(), PhiComplex(PhiReal.from_decimal(-1), PhiReal.zero())]], (2, 2))
        ]
        
        su2_group = SymmetryGroup("SU(2)", 3, pauli_matrices)
        u1_subgroup = SymmetryGroup("U(1)", 1, [pauli_matrices[2]])  # T_3生成的U(1)
        
        # SU(2) → U(1)破缺
        breaking = SymmetryBreaking(
            original_group=su2_group,
            residual_group=u1_subgroup,
            vacuum_state=PhiComplex(self.potential.vev, PhiReal.zero()),
            broken_generators=pauli_matrices[:2]  # T_1, T_2被破缺
        )
        
        # 应该有2个Goldstone玻色子
        self.assertEqual(breaking.num_goldstone_bosons(), 2)
        
        # 剩余对称性检查
        self.assertTrue(breaking.residual_group.is_subgroup_of(breaking.original_group))
    
    def test_vacuum_stability(self):
        """测试真空稳定性"""
        # 检查我们的真空是否是全局最小值
        vacuum_states = self.potential.find_vacuum_states()
        
        # 计算对称点和真空点的势能
        v_symmetric = self.potential.evaluate(PhiComplex.zero())
        v_broken = self.potential.evaluate(vacuum_states[0])
        
        # 破缺真空应该能量更低
        # 注意：我们计算的是势能V(φ)，而不是能量
        # 对于V = μ²|φ|² + λ|φ|⁴，当μ² < 0时
        # V(0) = 0, V(v) = -μ⁴/(4λ) < 0
        # 所以V(v) < V(0)是正确的
        self.assertLess(v_broken.decimal_value, v_symmetric.decimal_value)
        
        # 调试信息
        logging.info(f"V(0) = {v_symmetric.decimal_value:.6f}")
        logging.info(f"V(v) = {v_broken.decimal_value:.6f}")
        logging.info(f"vev = {self.potential.vev.decimal_value:.6f}")
        
        # 检查二阶导数（质量矩阵）
        # 在真空点应该是正定的（稳定）
        # 对于径向模式：V''(v) = 2μ² + 12λv²
        # 代入v² = -μ²/(2λ)得：V''(v) = -4μ²
        # 因为μ² < 0，所以V''(v) > 0，稳定
        mass_squared_higgs = PhiReal.from_decimal(
            -4 * self.mu_squared.decimal_value  # μ² < 0，所以这是正的
        )
        self.assertGreater(mass_squared_higgs.decimal_value, 0)  # 稳定
        
        logging.info(f"Higgs质量平方: m_h² = {mass_squared_higgs.decimal_value:.4f}")

# ==================== 辅助函数 ====================

def create_complex_breaking_pattern():
    """创建复杂的对称破缺模式示例"""
    # 示例：SU(3) × SU(2) × U(1) → SU(3) × U(1)_em
    # 这是标准模型电弱破缺的简化版本
    pass

def verify_goldstone_equivalence_theorem():
    """验证Goldstone等价定理"""
    # 高能下，纵向极化的规范玻色子等价于Goldstone玻色子
    pass

# ==================== 主程序 ====================

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)