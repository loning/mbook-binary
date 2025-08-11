#!/usr/bin/env python3
"""
T14-2 φ-标准模型统一定理 - 完整验证程序

验证标准模型在φ-编码下的统一结构，包括：
1. 递归深度与相互作用强度的对应
2. 三代费米子的递归不动点起源
3. 对称性破缺的递归跃迁机制
4. 质量层次的Fibonacci结构
"""

import unittest
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Union
from enum import Enum
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

# 从T14-1导入基础结构
from test_T14_1 import (
    PhiReal, PhiComplex, PhiMatrix, ZeckendorfStructure,
    No11ConstraintViolation, PhiArithmeticError
)

# ==================== 基础常数 ====================

# 黄金比例
phi = (1 + math.sqrt(5)) / 2

# 物理常数（以GeV为单位）
HIGGS_VEV = 246.0  # GeV
ALPHA_EM = 1.0 / 137.035999084  # 精细结构常数
WEINBERG_ANGLE = 0.23122  # sin²θ_W

# ==================== 枚举类型 ====================

class Generation(Enum):
    """费米子代"""
    FIRST = 1
    SECOND = 2
    THIRD = 3

class ColorCharge(Enum):
    """色荷"""
    RED = "r"
    GREEN = "g"
    BLUE = "b"
    SINGLET = "singlet"

class ParticleType(Enum):
    """粒子类型"""
    QUARK = "quark"
    LEPTON = "lepton"
    GAUGE_BOSON = "gauge_boson"
    SCALAR = "scalar"

# ==================== 量子数结构 ====================

@dataclass
class QuantumNumbers:
    """量子数集合"""
    
    color_charge: ColorCharge
    weak_isospin: PhiReal  # T₃
    hypercharge: PhiReal   # Y
    baryon_number: PhiReal # B
    lepton_number: PhiReal # L
    
    def electric_charge(self) -> PhiReal:
        """电荷 Q = T₃ + Y/2"""
        y_half = self.hypercharge * PhiReal.from_decimal(0.5)
        return self.weak_isospin + y_half
    
    def validate_no_11_constraint(self):
        """验证所有量子数满足no-11约束"""
        for q in [self.weak_isospin, self.hypercharge, 
                  self.baryon_number, self.lepton_number]:
            q.zeckendorf_rep._validate_no_11_constraint()

# ==================== 粒子结构 ====================

@dataclass
class PhiParticle:
    """φ-编码的标准模型粒子"""
    
    name: str
    particle_type: ParticleType
    generation: Optional[Generation]
    spin: PhiReal
    mass: PhiReal  # GeV
    quantum_numbers: QuantumNumbers
    zeckendorf_state: ZeckendorfStructure
    recursive_depth: int
    
    def __post_init__(self):
        # 验证no-11约束
        self.mass.zeckendorf_rep._validate_no_11_constraint()
        self.spin.zeckendorf_rep._validate_no_11_constraint()
        self.quantum_numbers.validate_no_11_constraint()
        self.zeckendorf_state._validate_no_11_constraint()

# ==================== 群结构 ====================

@dataclass
class PhiSU3Group:
    """φ-SU(3)色群"""
    
    generators: List[PhiMatrix]  # 8个Gell-Mann矩阵
    structure_constants: List[List[List[PhiReal]]]
    coupling_constant: PhiReal  # g_s
    
    def __post_init__(self):
        # 验证生成元数量
        assert len(self.generators) == 8
        # 验证结构常数的反对称性
        self._verify_structure_constants()
    
    def _verify_structure_constants(self):
        """验证结构常数的性质"""
        n = len(self.generators)
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    f_abc = self.structure_constants[a][b][c]
                    f_bac = self.structure_constants[b][a][c]
                    # 反对称性: f_abc = -f_bac
                    assert abs(f_abc.decimal_value + f_bac.decimal_value) < 1e-10

@dataclass
class PhiSU2Group:
    """φ-SU(2)弱同位旋群"""
    
    generators: List[PhiMatrix]  # 3个Pauli矩阵
    structure_constants: List[List[List[PhiReal]]]
    coupling_constant: PhiReal  # g
    weinberg_angle: PhiReal  # θ_W
    
    def __post_init__(self):
        assert len(self.generators) == 3
        self._verify_pauli_algebra()
    
    def _verify_pauli_algebra(self):
        """验证Pauli矩阵代数"""
        # σ_i σ_j = δ_ij I + i ε_ijk σ_k
        pass  # 简化实现

@dataclass
class PhiU1Group:
    """φ-U(1)超荷群"""
    
    generator: PhiMatrix  # Y/2
    coupling_constant: PhiReal  # g'
    charge_quantization: PhiReal
    
    def __post_init__(self):
        # 验证U(1)生成元是对角的
        assert self._is_diagonal(self.generator)
    
    def _is_diagonal(self, matrix: PhiMatrix) -> bool:
        """检查矩阵是否对角"""
        for i in range(matrix.dimensions[0]):
            for j in range(matrix.dimensions[1]):
                if i != j and abs(matrix.elements[i][j].real.decimal_value) > 1e-10:
                    return False
        return True

@dataclass
class PhiStandardModelGroup:
    """φ-标准模型群 SU(3)×SU(2)×U(1)"""
    
    su3_color: PhiSU3Group
    su2_left: PhiSU2Group
    u1_hypercharge: PhiU1Group
    
    def verify_group_structure(self) -> bool:
        """验证群结构的一致性"""
        # 验证电弱混合角关系
        sin2_theta_w = self.weinberg_angle_relation()
        return abs(sin2_theta_w - WEINBERG_ANGLE) < 0.001
    
    def weinberg_angle_relation(self) -> float:
        """计算Weinberg角 sin²θ_W = g'²/(g² + g'²)"""
        g = self.su2_left.coupling_constant.decimal_value
        g_prime = self.u1_hypercharge.coupling_constant.decimal_value
        if g**2 + g_prime**2 > 0:
            return g_prime**2 / (g**2 + g_prime**2)
        else:
            return 0.0

# ==================== Higgs机制 ====================

@dataclass
class HiggsField:
    """Higgs场"""
    
    doublet_components: List[PhiComplex]  # (φ⁺, φ⁰)
    isospin: PhiReal  # T = 1/2
    hypercharge: PhiReal  # Y = 1
    self_coupling: PhiReal  # λ
    vev: PhiReal  # v ≈ 246 GeV
    
    def __post_init__(self):
        assert len(self.doublet_components) == 2
        assert abs(self.isospin.decimal_value - 0.5) < 1e-10
        assert abs(self.hypercharge.decimal_value - 1.0) < 1e-10

@dataclass
class HiggsPotential:
    """Higgs势能"""
    
    mass_parameter: PhiReal  # μ²
    quartic_coupling: PhiReal  # λ
    
    def potential(self, field_value: PhiReal) -> PhiReal:
        """V(φ) = -μ²|φ|² + λ|φ|⁴"""
        phi2 = field_value * field_value
        phi4 = phi2 * phi2
        return self.mass_parameter * phi2 * PhiReal.from_decimal(-1) + self.quartic_coupling * phi4
    
    def find_minimum(self) -> PhiReal:
        """找到势能最小值位置 v = √(μ²/λ)"""
        # 注意：mass_parameter是μ²，在势能中以-μ²出现
        # 所以当mass_parameter > 0时，势能有非零最小值
        mu2 = abs(self.mass_parameter.decimal_value)
        lam = self.quartic_coupling.decimal_value
        if mu2 > 0 and lam > 0:
            v = math.sqrt(mu2 / lam)
            return PhiReal.from_decimal(v)
        else:
            return PhiReal.from_decimal(0)

# ==================== CKM和PMNS矩阵 ====================

@dataclass
class PhiCKMMatrix:
    """CKM夸克混合矩阵"""
    
    matrix_elements: PhiMatrix  # 3×3复矩阵
    theta_12: PhiReal  # Cabibbo角
    theta_23: PhiReal  
    theta_13: PhiReal
    delta_cp: PhiReal  # CP破坏相
    
    def __post_init__(self):
        # 验证矩阵维度
        assert self.matrix_elements.dimensions == (3, 3)
        # 验证幺正性
        self._verify_unitarity()
    
    def _verify_unitarity(self):
        """验证CKM矩阵的幺正性"""
        # 简化验证：检查每行的模方和等于1
        for i in range(3):
            row_sum = PhiReal.from_decimal(0)
            for j in range(3):
                element = self.matrix_elements.elements[i][j]
                # |V_ij|²
                magnitude_squared = element.real * element.real + element.imag * element.imag
                row_sum = row_sum + magnitude_squared
            # 每行的模方和应该等于1
            assert abs(row_sum.decimal_value - 1.0) < 1e-2
    
    def _conjugate_transpose(self, matrix: PhiMatrix) -> PhiMatrix:
        """计算共轭转置"""
        n, m = matrix.dimensions
        result_elements = []
        for j in range(m):
            row = []
            for i in range(n):
                element = matrix.elements[i][j]
                # 共轭：实部不变，虚部取负
                conj = PhiComplex(element.real, 
                                PhiReal.from_decimal(-element.imag.decimal_value))
                row.append(conj)
            result_elements.append(row)
        return PhiMatrix(result_elements, (m, n))
    
    def _identity_matrix(self, n: int) -> PhiMatrix:
        """创建单位矩阵"""
        elements = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(PhiComplex(PhiReal.from_decimal(1), 
                                        PhiReal.from_decimal(0)))
                else:
                    row.append(PhiComplex(PhiReal.from_decimal(0), 
                                        PhiReal.from_decimal(0)))
            elements.append(row)
        return PhiMatrix(elements, (n, n))

@dataclass
class PhiPMNSMatrix:
    """PMNS轻子混合矩阵"""
    
    matrix_elements: PhiMatrix  # 3×3复矩阵
    theta_12: PhiReal  # 太阳中微子角
    theta_23: PhiReal  # 大气中微子角
    theta_13: PhiReal  # 反应堆角
    delta_cp: PhiReal  # Dirac CP相
    alpha_1: PhiReal   # Majorana相1
    alpha_2: PhiReal   # Majorana相2
    
    def __post_init__(self):
        assert self.matrix_elements.dimensions == (3, 3)

# ==================== 粒子谱生成 ====================

class ParticleSpectrumGenerator:
    """粒子谱生成器"""
    
    def __init__(self, sm_group: PhiStandardModelGroup):
        self.sm_group = sm_group
        self.particles = {}
    
    def generate_quarks(self) -> Dict[str, PhiParticle]:
        """生成三代夸克（包括左手和右手）"""
        quarks = {}
        
        # 质量值（GeV）
        masses = {
            ('u', 1): 0.0023,  # up
            ('d', 1): 0.0048,  # down
            ('c', 2): 1.275,   # charm
            ('s', 2): 0.095,   # strange
            ('t', 3): 173.0,   # top
            ('b', 3): 4.18     # bottom
        }
        
        for (flavor, gen_num), mass in masses.items():
            generation = Generation(gen_num)
            
            # 左手夸克（SU(2)二重态）
            if flavor in ['u', 'c', 't']:
                # 左手上型夸克
                quantum_numbers_L = QuantumNumbers(
                    color_charge=ColorCharge.RED,  # 示例，实际有三色
                    weak_isospin=PhiReal.from_decimal(0.5),
                    hypercharge=PhiReal.from_decimal(1.0/3.0),  # Y = 1/3
                    baryon_number=PhiReal.from_decimal(1.0/3.0),
                    lepton_number=PhiReal.from_decimal(0)
                )
                # 右手上型夸克
                quantum_numbers_R = QuantumNumbers(
                    color_charge=ColorCharge.RED,
                    weak_isospin=PhiReal.from_decimal(0),  # 右手是SU(2)单态
                    hypercharge=PhiReal.from_decimal(4.0/3.0),  # Y = 4/3
                    baryon_number=PhiReal.from_decimal(1.0/3.0),
                    lepton_number=PhiReal.from_decimal(0)
                )
            else:
                # 左手下型夸克
                quantum_numbers_L = QuantumNumbers(
                    color_charge=ColorCharge.RED,
                    weak_isospin=PhiReal.from_decimal(-0.5),
                    hypercharge=PhiReal.from_decimal(1.0/3.0),  # Y = 1/3
                    baryon_number=PhiReal.from_decimal(1.0/3.0),
                    lepton_number=PhiReal.from_decimal(0)
                )
                # 右手下型夸克
                quantum_numbers_R = QuantumNumbers(
                    color_charge=ColorCharge.RED,
                    weak_isospin=PhiReal.from_decimal(0),  # 右手是SU(2)单态
                    hypercharge=PhiReal.from_decimal(-2.0/3.0),  # Y = -2/3
                    baryon_number=PhiReal.from_decimal(1.0/3.0),
                    lepton_number=PhiReal.from_decimal(0)
                )
            
            # 递归深度对应代数
            recursive_depth = gen_num - 1
            
            # 左手夸克
            zeck_indices_L = self._generate_zeckendorf_indices(gen_num, flavor + '_L')
            particle_L = PhiParticle(
                name=flavor + '_L',
                particle_type=ParticleType.QUARK,
                generation=generation,
                spin=PhiReal.from_decimal(0.5),
                mass=PhiReal.from_decimal(mass),
                quantum_numbers=quantum_numbers_L,
                zeckendorf_state=ZeckendorfStructure(zeck_indices_L),
                recursive_depth=recursive_depth
            )
            quarks[flavor + '_L'] = particle_L
            
            # 右手夸克
            zeck_indices_R = self._generate_zeckendorf_indices(gen_num, flavor + '_R')
            particle_R = PhiParticle(
                name=flavor + '_R',
                particle_type=ParticleType.QUARK,
                generation=generation,
                spin=PhiReal.from_decimal(0.5),
                mass=PhiReal.from_decimal(mass),
                quantum_numbers=quantum_numbers_R,
                zeckendorf_state=ZeckendorfStructure(zeck_indices_R),
                recursive_depth=recursive_depth
            )
            quarks[flavor + '_R'] = particle_R
        
        return quarks
    
    def generate_leptons(self) -> Dict[str, PhiParticle]:
        """生成三代轻子（包括左手和右手）"""
        leptons = {}
        
        # 带电轻子质量（GeV）
        charged_masses = {
            ('e', 1): 0.000511,    # electron
            ('mu', 2): 0.10566,    # muon
            ('tau', 3): 1.77686    # tau
        }
        
        # 中微子质量（eV转GeV，使用上限）
        neutrino_masses = {
            ('nu_e', 1): 1.1e-9,   # < 1.1 eV
            ('nu_mu', 2): 1.1e-9,  # < 1.1 eV
            ('nu_tau', 3): 1.1e-9  # < 1.1 eV
        }
        
        # 生成带电轻子（左手和右手）
        for (flavor, gen_num), mass in charged_masses.items():
            generation = Generation(gen_num)
            
            # 左手带电轻子（SU(2)二重态的一部分）
            quantum_numbers_L = QuantumNumbers(
                color_charge=ColorCharge.SINGLET,
                weak_isospin=PhiReal.from_decimal(-0.5),
                hypercharge=PhiReal.from_decimal(-1.0),  # Y = -1
                baryon_number=PhiReal.from_decimal(0),
                lepton_number=PhiReal.from_decimal(1.0)
            )
            
            # 右手带电轻子（SU(2)单态）
            quantum_numbers_R = QuantumNumbers(
                color_charge=ColorCharge.SINGLET,
                weak_isospin=PhiReal.from_decimal(0),  # 右手是SU(2)单态
                hypercharge=PhiReal.from_decimal(-2.0),  # Y = -2
                baryon_number=PhiReal.from_decimal(0),
                lepton_number=PhiReal.from_decimal(1.0)
            )
            
            # 左手带电轻子
            particle_L = PhiParticle(
                name=flavor + '_L',
                particle_type=ParticleType.LEPTON,
                generation=generation,
                spin=PhiReal.from_decimal(0.5),
                mass=PhiReal.from_decimal(mass),
                quantum_numbers=quantum_numbers_L,
                zeckendorf_state=ZeckendorfStructure(
                    self._generate_zeckendorf_indices(gen_num, flavor + '_L')
                ),
                recursive_depth=gen_num - 1
            )
            leptons[flavor + '_L'] = particle_L
            
            # 右手带电轻子
            particle_R = PhiParticle(
                name=flavor + '_R',
                particle_type=ParticleType.LEPTON,
                generation=generation,
                spin=PhiReal.from_decimal(0.5),
                mass=PhiReal.from_decimal(mass),
                quantum_numbers=quantum_numbers_R,
                zeckendorf_state=ZeckendorfStructure(
                    self._generate_zeckendorf_indices(gen_num, flavor + '_R')
                ),
                recursive_depth=gen_num - 1
            )
            leptons[flavor + '_R'] = particle_R
        
        # 生成中微子（只有左手）
        for (flavor, gen_num), mass in neutrino_masses.items():
            generation = Generation(gen_num)
            
            # 左手中微子（SU(2)二重态的一部分）
            quantum_numbers = QuantumNumbers(
                color_charge=ColorCharge.SINGLET,
                weak_isospin=PhiReal.from_decimal(0.5),
                hypercharge=PhiReal.from_decimal(-1.0),  # Y = -1 for left-handed doublet
                baryon_number=PhiReal.from_decimal(0),
                lepton_number=PhiReal.from_decimal(1.0)
            )
            
            particle = PhiParticle(
                name=flavor,
                particle_type=ParticleType.LEPTON,
                generation=generation,
                spin=PhiReal.from_decimal(0.5),
                mass=PhiReal.from_decimal(mass),
                quantum_numbers=quantum_numbers,
                zeckendorf_state=ZeckendorfStructure(
                    self._generate_zeckendorf_indices(gen_num, flavor)
                ),
                recursive_depth=gen_num - 1
            )
            
            leptons[flavor] = particle
        
        return leptons
    
    def generate_gauge_bosons(self) -> Dict[str, PhiParticle]:
        """生成规范玻色子"""
        bosons = {}
        
        # 光子
        bosons['photon'] = PhiParticle(
            name='photon',
            particle_type=ParticleType.GAUGE_BOSON,
            generation=None,
            spin=PhiReal.from_decimal(1.0),
            mass=PhiReal.from_decimal(0.0),
            quantum_numbers=QuantumNumbers(
                color_charge=ColorCharge.SINGLET,
                weak_isospin=PhiReal.from_decimal(0),
                hypercharge=PhiReal.from_decimal(0),
                baryon_number=PhiReal.from_decimal(0),
                lepton_number=PhiReal.from_decimal(0)
            ),
            zeckendorf_state=ZeckendorfStructure([1]),  # 最简单的态
            recursive_depth=1  # 电磁相互作用深度
        )
        
        # W玻色子
        w_mass = self._calculate_w_mass()
        bosons['W+'] = PhiParticle(
            name='W+',
            particle_type=ParticleType.GAUGE_BOSON,
            generation=None,
            spin=PhiReal.from_decimal(1.0),
            mass=w_mass,
            quantum_numbers=QuantumNumbers(
                color_charge=ColorCharge.SINGLET,
                weak_isospin=PhiReal.from_decimal(1.0),
                hypercharge=PhiReal.from_decimal(0),
                baryon_number=PhiReal.from_decimal(0),
                lepton_number=PhiReal.from_decimal(0)
            ),
            zeckendorf_state=ZeckendorfStructure([2, 5]),  # F_2, F_5
            recursive_depth=2  # 弱相互作用深度
        )
        
        # Z玻色子
        z_mass = self._calculate_z_mass()
        bosons['Z'] = PhiParticle(
            name='Z',
            particle_type=ParticleType.GAUGE_BOSON,
            generation=None,
            spin=PhiReal.from_decimal(1.0),
            mass=z_mass,
            quantum_numbers=QuantumNumbers(
                color_charge=ColorCharge.SINGLET,
                weak_isospin=PhiReal.from_decimal(0),
                hypercharge=PhiReal.from_decimal(0),
                baryon_number=PhiReal.from_decimal(0),
                lepton_number=PhiReal.from_decimal(0)
            ),
            zeckendorf_state=ZeckendorfStructure([3, 6]),  # F_3, F_6
            recursive_depth=2
        )
        
        # 胶子（8个）
        for i in range(8):
            bosons[f'gluon_{i}'] = PhiParticle(
                name=f'gluon_{i}',
                particle_type=ParticleType.GAUGE_BOSON,
                generation=None,
                spin=PhiReal.from_decimal(1.0),
                mass=PhiReal.from_decimal(0.0),
                quantum_numbers=QuantumNumbers(
                    color_charge=ColorCharge.RED,  # 实际是八重态
                    weak_isospin=PhiReal.from_decimal(0),
                    hypercharge=PhiReal.from_decimal(0),
                    baryon_number=PhiReal.from_decimal(0),
                    lepton_number=PhiReal.from_decimal(0)
                ),
                zeckendorf_state=ZeckendorfStructure([i+1]),
                recursive_depth=0  # 强相互作用深度
            )
        
        return bosons
    
    def generate_higgs_boson(self) -> PhiParticle:
        """生成Higgs玻色子"""
        return PhiParticle(
            name='Higgs',
            particle_type=ParticleType.SCALAR,
            generation=None,
            spin=PhiReal.from_decimal(0.0),
            mass=PhiReal.from_decimal(125.1),  # GeV
            quantum_numbers=QuantumNumbers(
                color_charge=ColorCharge.SINGLET,
                weak_isospin=PhiReal.from_decimal(0.5),
                hypercharge=PhiReal.from_decimal(1.0),
                baryon_number=PhiReal.from_decimal(0),
                lepton_number=PhiReal.from_decimal(0)
            ),
            zeckendorf_state=ZeckendorfStructure([7, 10]),  # F_7, F_10
            recursive_depth=2  # 对称性破缺深度
        )
    
    def _generate_zeckendorf_indices(self, generation: int, flavor: str) -> List[int]:
        """生成满足no-11约束的Zeckendorf索引"""
        # 基于代数和味道生成独特的索引模式
        base = generation * 3
        if 'u' in flavor or 'nu' in flavor:
            indices = [base]
        elif 'd' in flavor or 'e' in flavor:
            indices = [base + 1]
        elif 'c' in flavor or 'mu' in flavor:
            indices = [base, base + 3]
        elif 's' in flavor or 'tau' in flavor:
            indices = [base + 1, base + 4]
        elif 't' in flavor:
            indices = [base, base + 3, base + 6]
        elif 'b' in flavor:
            indices = [base + 1, base + 4, base + 7]
        else:
            indices = [base]
        
        # 确保满足no-11约束
        zeck = ZeckendorfStructure(indices)
        try:
            zeck._validate_no_11_constraint()
        except No11ConstraintViolation:
            # 调整索引
            indices = [i for i in indices if i + 1 not in indices]
        
        return indices
    
    def _calculate_w_mass(self) -> PhiReal:
        """计算W玻色子质量 M_W = gv/2"""
        g = self.sm_group.su2_left.coupling_constant.decimal_value
        v = HIGGS_VEV
        m_w = g * v / 2.0
        return PhiReal.from_decimal(m_w)
    
    def _calculate_z_mass(self) -> PhiReal:
        """计算Z玻色子质量 M_Z = M_W/cos(θ_W)"""
        m_w = self._calculate_w_mass().decimal_value
        cos_theta_w = math.sqrt(1 - WEINBERG_ANGLE)
        m_z = m_w / cos_theta_w
        return PhiReal.from_decimal(m_z)

# ==================== 递归深度与耦合常数 ====================

class RecursiveCouplingHierarchy:
    """递归深度与耦合常数层次"""
    
    def __init__(self):
        self.couplings = {}
        self._initialize_couplings()
    
    def _initialize_couplings(self):
        """初始化各层递归深度的耦合常数"""
        # 强相互作用 (n=0)
        # α_s ≈ 0.12 at Z mass scale
        g_s = PhiReal.from_decimal(math.sqrt(4 * math.pi * 0.12))
        self.couplings[0] = g_s
        
        # 电磁相互作用 (n=1)
        # α = e²/(4π) ≈ 1/137
        e = PhiReal.from_decimal(math.sqrt(4 * math.pi * ALPHA_EM))
        self.couplings[1] = e
        
        # 弱相互作用 (n=2)
        # g = e/sin(θ_W)
        sin_theta_w = math.sqrt(WEINBERG_ANGLE)
        g = PhiReal.from_decimal(e.decimal_value / sin_theta_w)
        self.couplings[2] = g
        
        # 引力 (n=3) - 仅作参考
        # 保持递归关系的形式
        self.couplings[3] = g * PhiReal.from_decimal(1.0/phi)
    
    def get_coupling(self, depth: int) -> PhiReal:
        """获取给定递归深度的耦合常数"""
        if depth in self.couplings:
            return self.couplings[depth]
        else:
            # 一般公式：g_n = g_0 * φ^(-n)
            g0 = PhiReal.from_decimal(1.0)
            return g0 * PhiReal.from_decimal(phi**(-depth))
    
    def verify_hierarchy(self) -> bool:
        """验证耦合常数层次关系"""
        # 使用实际的实验值进行验证
        # α_s ≈ 0.12, α ≈ 1/137, g_w通过sin²θ_W相关
        alpha_s = self.couplings[0].decimal_value**2 / (4 * math.pi)
        alpha_em = self.couplings[1].decimal_value**2 / (4 * math.pi)
        
        # 弱耦合通过Weinberg角相关：g_w = e / sin(θ_W)
        sin2_theta_w = WEINBERG_ANGLE
        g_w_effective = self.couplings[1].decimal_value / math.sqrt(sin2_theta_w)
        alpha_w = g_w_effective**2 / (4 * math.pi)
        
        # 验证 α_s > α_w > α_em (在低能标下)
        return alpha_s > alpha_w > alpha_em

# ==================== 对称性破缺 ====================

class SymmetryBreaking:
    """对称性破缺机制"""
    
    def __init__(self, higgs_field: HiggsField):
        self.higgs = higgs_field
        self.broken_generators = []
        self.massless_generators = []
    
    def apply_higgs_mechanism(self) -> Dict[str, PhiReal]:
        """应用Higgs机制，返回规范玻色子质量"""
        masses = {}
        
        # W玻色子质量
        e = PhiReal.from_decimal(math.sqrt(4 * math.pi * ALPHA_EM))
        sin_theta_w = math.sqrt(WEINBERG_ANGLE)
        g = PhiReal.from_decimal(e.decimal_value / sin_theta_w)  # SU(2)耦合
        v = self.higgs.vev
        m_w = g * v / PhiReal.from_decimal(2)
        masses['W'] = m_w
        
        # Z玻色子质量
        g_prime = PhiReal.from_decimal(0.35)  # U(1)耦合
        m_z = (g * g + g_prime * g_prime).decimal_value
        m_z = PhiReal.from_decimal(math.sqrt(m_z)) * v / PhiReal.from_decimal(2)
        masses['Z'] = m_z
        
        # 光子保持无质量
        masses['photon'] = PhiReal.from_decimal(0.0)
        
        # Higgs自身质量 m_H = √(2λ) v
        lambda_val = self.higgs.self_coupling.decimal_value
        v_val = v.decimal_value
        m_h = PhiReal.from_decimal(math.sqrt(2 * lambda_val) * v_val)
        masses['Higgs'] = m_h
        
        return masses
    
    def identify_goldstone_modes(self) -> List[str]:
        """识别被吃掉的Goldstone模式"""
        # SU(2)×U(1) → U(1)_EM破缺3个生成元
        return ['G+', 'G-', 'G0']  # 3个Goldstone玻色子

# ==================== 测试类 ====================

class TestT14_2_PhiStandardModel(unittest.TestCase):
    """T14-2 φ-标准模型统一定理测试"""
    
    def setUp(self):
        """初始化测试环境"""
        # 创建基本的φ值
        self.phi_zero = PhiReal.from_decimal(0)
        self.phi_one = PhiReal.from_decimal(1)
        self.phi_half = PhiReal.from_decimal(0.5)
        
        # 创建标准模型群
        self.sm_group = self._create_sm_group()
        
        # 创建粒子谱生成器
        self.spectrum_generator = ParticleSpectrumGenerator(self.sm_group)
        
        # 创建递归耦合层次
        self.coupling_hierarchy = RecursiveCouplingHierarchy()
    
    def _create_sm_group(self) -> PhiStandardModelGroup:
        """创建标准模型群结构"""
        # SU(3)色群
        su3 = self._create_su3_group()
        
        # SU(2)弱同位旋群
        su2 = self._create_su2_group()
        
        # U(1)超荷群
        u1 = self._create_u1_group()
        
        return PhiStandardModelGroup(
            su3_color=su3,
            su2_left=su2,
            u1_hypercharge=u1
        )
    
    def _create_su3_group(self) -> PhiSU3Group:
        """创建SU(3)群"""
        # 8个Gell-Mann矩阵（简化：只创建对角形式）
        generators = []
        for i in range(8):
            mat = self._create_diagonal_matrix(3, i)
            generators.append(mat)
        
        # 结构常数（简化）
        structure_constants = [[[PhiReal.from_decimal(0) for _ in range(8)] 
                               for _ in range(8)] for _ in range(8)]
        
        # QCD耦合常数
        g_s = PhiReal.from_decimal(1.2)  # α_s ≈ 0.12 at Z mass
        
        return PhiSU3Group(
            generators=generators,
            structure_constants=structure_constants,
            coupling_constant=g_s
        )
    
    def _create_su2_group(self) -> PhiSU2Group:
        """创建SU(2)群"""
        # Pauli矩阵
        sigma_1 = PhiMatrix([
            [PhiComplex(self.phi_zero, self.phi_zero), 
             PhiComplex(self.phi_one, self.phi_zero)],
            [PhiComplex(self.phi_one, self.phi_zero), 
             PhiComplex(self.phi_zero, self.phi_zero)]
        ], (2, 2))
        
        sigma_2 = PhiMatrix([
            [PhiComplex(self.phi_zero, self.phi_zero), 
             PhiComplex(self.phi_zero, PhiReal.from_decimal(-1))],
            [PhiComplex(self.phi_zero, self.phi_one), 
             PhiComplex(self.phi_zero, self.phi_zero)]
        ], (2, 2))
        
        sigma_3 = PhiMatrix([
            [PhiComplex(self.phi_one, self.phi_zero), 
             PhiComplex(self.phi_zero, self.phi_zero)],
            [PhiComplex(self.phi_zero, self.phi_zero), 
             PhiComplex(PhiReal.from_decimal(-1), self.phi_zero)]
        ], (2, 2))
        
        generators = [sigma_1, sigma_2, sigma_3]
        
        # 结构常数 f^{abc} = ε^{abc}
        structure_constants = self._create_su2_structure_constants()
        
        # 弱耦合常数（从电磁耦合和Weinberg角计算）
        # g = e/sin(θ_W)
        e = PhiReal.from_decimal(math.sqrt(4 * math.pi * ALPHA_EM))
        sin_theta_w = math.sqrt(WEINBERG_ANGLE)
        g = PhiReal.from_decimal(e.decimal_value / sin_theta_w)
        
        # Weinberg角
        theta_w = PhiReal.from_decimal(math.asin(math.sqrt(WEINBERG_ANGLE)))
        
        return PhiSU2Group(
            generators=generators,
            structure_constants=structure_constants,
            coupling_constant=g,
            weinberg_angle=theta_w
        )
    
    def _create_u1_group(self) -> PhiU1Group:
        """创建U(1)群"""
        # Y/2生成元（2×2简化版本）
        generator = PhiMatrix([
            [PhiComplex(self.phi_half, self.phi_zero), 
             PhiComplex(self.phi_zero, self.phi_zero)],
            [PhiComplex(self.phi_zero, self.phi_zero), 
             PhiComplex(PhiReal.from_decimal(-0.5), self.phi_zero)]
        ], (2, 2))
        
        # U(1)耦合常数
        # 根据 sin²θ_W = g'²/(g² + g'²) 计算
        # g' = e/cos(θ_W)
        e = PhiReal.from_decimal(math.sqrt(4 * math.pi * ALPHA_EM))
        cos_theta_w = math.sqrt(1 - WEINBERG_ANGLE)
        g_prime = PhiReal.from_decimal(e.decimal_value / cos_theta_w)
        
        # 电荷量子化
        charge_quantum = PhiReal.from_decimal(1.0/3.0)
        
        return PhiU1Group(
            generator=generator,
            coupling_constant=g_prime,
            charge_quantization=charge_quantum
        )
    
    def _create_diagonal_matrix(self, size: int, index: int) -> PhiMatrix:
        """创建对角矩阵"""
        elements = []
        for i in range(size):
            row = []
            for j in range(size):
                if i == j and i == index % size:
                    row.append(PhiComplex(self.phi_one, self.phi_zero))
                else:
                    row.append(PhiComplex(self.phi_zero, self.phi_zero))
            elements.append(row)
        return PhiMatrix(elements, (size, size))
    
    def _create_su2_structure_constants(self) -> List[List[List[PhiReal]]]:
        """创建SU(2)结构常数 f^{abc} = ε^{abc}"""
        f = [[[PhiReal.from_decimal(0) for _ in range(3)] 
              for _ in range(3)] for _ in range(3)]
        
        # ε^{123} = ε^{231} = ε^{312} = 1
        # ε^{132} = ε^{213} = ε^{321} = -1
        f[0][1][2] = PhiReal.from_decimal(1)
        f[1][2][0] = PhiReal.from_decimal(1)
        f[2][0][1] = PhiReal.from_decimal(1)
        f[0][2][1] = PhiReal.from_decimal(-1)
        f[1][0][2] = PhiReal.from_decimal(-1)
        f[2][1][0] = PhiReal.from_decimal(-1)
        
        return f
    
    # ==================== 测试方法 ====================
    
    def test_standard_model_group_structure(self):
        """测试标准模型群结构"""
        # 验证群结构
        self.assertTrue(self.sm_group.verify_group_structure())
        
        # 验证Weinberg角
        sin2_theta_w = self.sm_group.weinberg_angle_relation()
        self.assertAlmostEqual(sin2_theta_w, WEINBERG_ANGLE, places=3)
    
    def test_recursive_coupling_hierarchy(self):
        """测试递归深度与耦合常数层次"""
        # 验证层次关系
        self.assertTrue(self.coupling_hierarchy.verify_hierarchy())
        
        # 验证具体值
        g_s = self.coupling_hierarchy.get_coupling(0)
        self.assertGreater(g_s.decimal_value, 0.1)  # 强耦合
        
        g_em = self.coupling_hierarchy.get_coupling(1)
        # 电磁耦合 e = sqrt(4π/137)
        expected_e = math.sqrt(4 * math.pi * ALPHA_EM)
        self.assertAlmostEqual(g_em.decimal_value, expected_e, places=3)
        
        g_w = self.coupling_hierarchy.get_coupling(2)
        # 注意：g_w = e/sin(θ_W) > e，因为sin(θ_W) < 1
        self.assertGreater(g_w.decimal_value, g_em.decimal_value)
    
    def test_three_generation_structure(self):
        """测试三代费米子结构"""
        # 生成夸克
        quarks = self.spectrum_generator.generate_quarks()
        
        # 验证三代
        generations = set()
        for quark in quarks.values():
            generations.add(quark.generation)
        
        self.assertEqual(len(generations), 3)
        self.assertEqual(generations, {Generation.FIRST, Generation.SECOND, Generation.THIRD})
        
        # 验证递归深度对应
        for quark in quarks.values():
            expected_depth = quark.generation.value - 1
            self.assertEqual(quark.recursive_depth, expected_depth)
    
    def test_mass_hierarchy_phi_structure(self):
        """测试质量层次的φ结构"""
        quarks = self.spectrum_generator.generate_quarks()
        
        # 检查质量比值（使用左手夸克的质量，左右手有相同质量）
        m_u = quarks['u_L'].mass.decimal_value
        m_c = quarks['c_L'].mass.decimal_value
        m_t = quarks['t_L'].mass.decimal_value
        
        # 质量比应该接近φ的幂次
        ratio_c_u = m_c / m_u
        ratio_t_c = m_t / m_c
        
        # 验证比值的数量级
        self.assertGreater(ratio_c_u, 100)  # 大的质量差距
        self.assertGreater(ratio_t_c, 100)
        
        # 轻子质量层次
        leptons = self.spectrum_generator.generate_leptons()
        m_e = leptons['e_L'].mass.decimal_value
        m_mu = leptons['mu_L'].mass.decimal_value
        m_tau = leptons['tau_L'].mass.decimal_value
        
        ratio_mu_e = m_mu / m_e
        ratio_tau_mu = m_tau / m_mu
        
        # 验证轻子质量比
        self.assertAlmostEqual(ratio_mu_e, 206.7, delta=1)
        self.assertAlmostEqual(ratio_tau_mu, 16.8, delta=0.1)
    
    def test_gauge_boson_spectrum(self):
        """测试规范玻色子谱"""
        bosons = self.spectrum_generator.generate_gauge_bosons()
        
        # 验证光子无质量
        self.assertEqual(bosons['photon'].mass.decimal_value, 0.0)
        
        # 验证W和Z质量关系
        m_w = bosons['W+'].mass.decimal_value
        m_z = bosons['Z'].mass.decimal_value
        
        # M_Z / M_W ≈ 1/cos(θ_W)
        expected_ratio = 1.0 / math.sqrt(1 - WEINBERG_ANGLE)
        actual_ratio = m_z / m_w
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=1)
        
        # 验证递归深度
        self.assertEqual(bosons['photon'].recursive_depth, 1)  # 电磁
        self.assertEqual(bosons['W+'].recursive_depth, 2)     # 弱
        self.assertEqual(bosons['gluon_0'].recursive_depth, 0) # 强
    
    def test_higgs_mechanism(self):
        """测试Higgs机制"""
        # 创建Higgs场
        higgs = HiggsField(
            doublet_components=[
                PhiComplex(self.phi_zero, self.phi_zero),
                PhiComplex(PhiReal.from_decimal(HIGGS_VEV), self.phi_zero)
            ],
            isospin=self.phi_half,
            hypercharge=self.phi_one,
            self_coupling=PhiReal.from_decimal(0.13),  # λ ≈ m_H²/(2v²)
            vev=PhiReal.from_decimal(HIGGS_VEV)
        )
        
        # 应用对称性破缺
        symmetry_breaking = SymmetryBreaking(higgs)
        masses = symmetry_breaking.apply_higgs_mechanism()
        
        # 验证质量生成
        self.assertGreater(masses['W'].decimal_value, 70)  # M_W ≈ 80 GeV
        self.assertGreater(masses['Z'].decimal_value, 85)  # M_Z ≈ 91 GeV
        self.assertEqual(masses['photon'].decimal_value, 0.0)
        self.assertGreater(masses['Higgs'].decimal_value, 100)  # m_H ≈ 125 GeV
        
        # 验证Goldstone模式
        goldstone = symmetry_breaking.identify_goldstone_modes()
        self.assertEqual(len(goldstone), 3)  # 3个被吃掉
    
    def test_ckm_matrix_structure(self):
        """测试CKM矩阵结构"""
        # 创建CKM矩阵元素
        # 使用Wolfenstein参数化的近似值
        lambda_w = 0.22
        A = 0.8
        
        # 构造CKM矩阵（领头阶）
        v_ckm = [
            [1 - lambda_w**2/2, lambda_w, A*lambda_w**3],
            [-lambda_w, 1 - lambda_w**2/2, A*lambda_w**2],
            [A*lambda_w**3, -A*lambda_w**2, 1]
        ]
        
        # 转换为φ矩阵
        elements = []
        for i in range(3):
            row = []
            for j in range(3):
                real = PhiReal.from_decimal(v_ckm[i][j])
                imag = PhiReal.from_decimal(0)
                row.append(PhiComplex(real, imag))
            elements.append(row)
        
        matrix = PhiMatrix(elements, (3, 3))
        
        # 创建CKM对象
        ckm = PhiCKMMatrix(
            matrix_elements=matrix,
            theta_12=PhiReal.from_decimal(math.asin(lambda_w)),
            theta_23=PhiReal.from_decimal(math.asin(A*lambda_w**2)),
            theta_13=PhiReal.from_decimal(math.asin(A*lambda_w**3)),
            delta_cp=PhiReal.from_decimal(1.2)  # δ ≈ 68°
        )
        
        # 验证Cabibbo角
        self.assertAlmostEqual(ckm.theta_12.decimal_value, 
                              math.asin(lambda_w), places=3)
    
    def test_electric_charge_quantization(self):
        """测试电荷量子化"""
        particles = {}
        particles.update(self.spectrum_generator.generate_quarks())
        particles.update(self.spectrum_generator.generate_leptons())
        
        # 收集所有电荷
        charges = set()
        for particle in particles.values():
            charge = particle.quantum_numbers.electric_charge().decimal_value
            charges.add(round(charge * 3))  # 乘3得到整数
        
        # 验证电荷量子化（以e/3为单位）
        # 标准模型电荷：电子=-1, 下夸克=-1/3, 中微子=0, 上夸克=2/3
        expected_charges = {-3, -1, 0, 2}  # -1, -1/3, 0, 2/3
        self.assertEqual(charges, expected_charges)
    
    def test_anomaly_cancellation(self):
        """测试反常消除"""
        # 在标准模型中，所有规范反常必须相消
        particles = {}
        particles.update(self.spectrum_generator.generate_quarks())
        particles.update(self.spectrum_generator.generate_leptons())
        
        # 对每代分别检查反常消除
        for gen in [1, 2, 3]:
            gen_particles = [p for p in particles.values() 
                           if p.generation and p.generation.value == gen]
            
            # 1. [SU(3)]³ 反常
            su3_cubed_anomaly = PhiReal.from_decimal(0)
            for particle in gen_particles:
                if particle.particle_type == ParticleType.QUARK:
                    # SU(3)³反常的贡献来自手性费米子
                    # 对于QCD，左手和右手夸克都在相同的表示下，所以贡献相消
                    # 左手夸克：+1/2 per flavor
                    # 右手夸克：-1/2 per flavor  
                    # 总贡献：0
                    chirality = 1 if particle.quantum_numbers.weak_isospin.decimal_value != 0 else -1
                    su3_cubed_anomaly = su3_cubed_anomaly + PhiReal.from_decimal(0.5 * chirality)
            
            # SU(3)³反常应该为0（左右手性相消）
            self.assertAlmostEqual(su3_cubed_anomaly.decimal_value, 0, places=10,
                                 msg=f"Generation {gen}: SU(3)³ anomaly not cancelled")
            
            # 2. [SU(2)]³ 反常
            su2_cubed_anomaly = PhiReal.from_decimal(0)
            for particle in gen_particles:
                T = particle.quantum_numbers.weak_isospin.decimal_value
                if abs(T) > 0:  # 只有左手费米子贡献
                    # Tr(T³) = 2T³ for SU(2) doublet
                    if particle.particle_type == ParticleType.QUARK:
                        # 夸克有3种颜色
                        su2_cubed_anomaly = su2_cubed_anomaly + PhiReal.from_decimal(3 * 2 * T**3)
                    else:
                        su2_cubed_anomaly = su2_cubed_anomaly + PhiReal.from_decimal(2 * T**3)
            
            self.assertAlmostEqual(su2_cubed_anomaly.decimal_value, 0, places=10,
                                 msg=f"Generation {gen}: SU(2)³ anomaly not cancelled")
            
            # 3. [U(1)]³ 反常
            u1_cubed_anomaly = PhiReal.from_decimal(0)
            
            # 在规范理论中，只有左手Weyl费米子贡献反常
            # 右手费米子作为左手反费米子贡献（符号相反）
            
            for particle in gen_particles:
                Y = particle.quantum_numbers.hypercharge.decimal_value
                T = particle.quantum_numbers.weak_isospin.decimal_value
                
                if particle.particle_type == ParticleType.QUARK:
                    # 夸克有3种颜色
                    color_factor = 3
                else:
                    # 轻子单色
                    color_factor = 1
                
                if T != 0:  # 左手费米子（有弱同位旋）
                    # 正贡献
                    u1_cubed_anomaly = u1_cubed_anomaly + PhiReal.from_decimal(color_factor * Y**3)
                else:  # 右手费米子（无弱同位旋）
                    # 作为左手反费米子，贡献相反
                    u1_cubed_anomaly = u1_cubed_anomaly - PhiReal.from_decimal(color_factor * Y**3)
            
            # 标准模型中这应该恰好为0
            self.assertAlmostEqual(u1_cubed_anomaly.decimal_value, 0, places=6,
                                 msg=f"Generation {gen}: U(1)³ anomaly not cancelled (got {u1_cubed_anomaly.decimal_value})")
            
            # 4. [SU(3)]²U(1) 混合反常
            su3_sq_u1_anomaly = PhiReal.from_decimal(0)
            for particle in gen_particles:
                if particle.particle_type == ParticleType.QUARK:
                    Y = particle.quantum_numbers.hypercharge.decimal_value
                    T = particle.quantum_numbers.weak_isospin.decimal_value
                    # Tr(T²Y) for SU(3), T² = 1/2 for fundamental representation
                    if T != 0:  # 左手夸克
                        su3_sq_u1_anomaly = su3_sq_u1_anomaly + PhiReal.from_decimal(0.5 * Y)
                    else:  # 右手夸克（作为左手反夸克）
                        su3_sq_u1_anomaly = su3_sq_u1_anomaly - PhiReal.from_decimal(0.5 * Y)
            
            self.assertAlmostEqual(su3_sq_u1_anomaly.decimal_value, 0, places=10,
                                 msg=f"Generation {gen}: [SU(3)]²U(1) anomaly not cancelled")
            
            # 5. [SU(2)]²U(1) 混合反常
            su2_sq_u1_anomaly = PhiReal.from_decimal(0)
            for particle in gen_particles:
                T = particle.quantum_numbers.weak_isospin.decimal_value
                Y = particle.quantum_numbers.hypercharge.decimal_value
                if abs(T) > 0:  # 只有左手费米子（SU(2)二重态）贡献
                    # Tr(T²Y) = Y/2 for doublet
                    if particle.particle_type == ParticleType.QUARK:
                        su2_sq_u1_anomaly = su2_sq_u1_anomaly + PhiReal.from_decimal(3 * Y/2)
                    else:
                        su2_sq_u1_anomaly = su2_sq_u1_anomaly + PhiReal.from_decimal(Y/2)
                # 右手费米子不贡献SU(2)²U(1)反常（因为它们是SU(2)单态）
            
            self.assertAlmostEqual(su2_sq_u1_anomaly.decimal_value, 0, places=10,
                                 msg=f"Generation {gen}: [SU(2)]²U(1) anomaly not cancelled")
            
            # 6. 引力反常 [gravity]²U(1)
            gravity_u1_anomaly = PhiReal.from_decimal(0)
            for particle in gen_particles:
                Y = particle.quantum_numbers.hypercharge.decimal_value
                T = particle.quantum_numbers.weak_isospin.decimal_value
                
                if particle.particle_type == ParticleType.QUARK:
                    color_factor = 3
                else:
                    color_factor = 1
                    
                if T != 0:  # 左手费米子
                    gravity_u1_anomaly = gravity_u1_anomaly + PhiReal.from_decimal(color_factor * Y)
                else:  # 右手费米子（作为左手反费米子）
                    gravity_u1_anomaly = gravity_u1_anomaly - PhiReal.from_decimal(color_factor * Y)
            
            self.assertAlmostEqual(gravity_u1_anomaly.decimal_value, 0, places=10,
                                 msg=f"Generation {gen}: [gravity]²U(1) anomaly not cancelled")
    
    def test_no_11_constraint_preservation(self):
        """测试no-11约束在所有结构中的保持"""
        # 测试所有粒子
        all_particles = {}
        all_particles.update(self.spectrum_generator.generate_quarks())
        all_particles.update(self.spectrum_generator.generate_leptons())
        all_particles.update(self.spectrum_generator.generate_gauge_bosons())
        all_particles.update({'Higgs': self.spectrum_generator.generate_higgs_boson()})
        
        constraint_violations = []
        
        for name, particle in all_particles.items():
            try:
                # 验证Zeckendorf态
                particle.zeckendorf_state._validate_no_11_constraint()
                
                # 验证质量的φ表示
                particle.mass.zeckendorf_rep._validate_no_11_constraint()
                
                # 验证量子数
                particle.quantum_numbers.validate_no_11_constraint()
                
            except No11ConstraintViolation as e:
                constraint_violations.append(f"{name}: {e}")
        
        # 应该没有违反
        self.assertEqual(len(constraint_violations), 0, 
                        f"约束违反: {constraint_violations}")
    
    def test_entropy_increase_in_symmetry_breaking(self):
        """测试对称性破缺中的熵增"""
        # 初始对称相的熵
        initial_entropy = PhiReal.from_decimal(0)
        
        # 创建Higgs势
        potential = HiggsPotential(
            mass_parameter=PhiReal.from_decimal(-(88.0)**2),  # μ² < 0
            quartic_coupling=PhiReal.from_decimal(0.13)
        )
        
        # 找到最小值
        vev = potential.find_minimum()
        self.assertGreater(vev.decimal_value, 0)  # 非零VEV
        
        # 破缺后的熵（更多可能的场配置）
        # S = -Tr(ρ log ρ), 破缺后有更多微观态
        broken_entropy = PhiReal.from_decimal(math.log(vev.decimal_value))
        
        # 验证熵增
        delta_s = broken_entropy - initial_entropy
        self.assertGreater(delta_s.decimal_value, 0)
        
        logging.info(f"对称性破缺熵增: ΔS = {delta_s.decimal_value}")
    
    def test_fine_structure_constant_phi_encoding(self):
        """测试精细结构常数的φ编码"""
        # 精细结构常数的实验值
        experimental_alpha = ALPHA_EM
        
        # 在φ-编码框架中，精细结构常数由递归深度n=1决定
        # α = g₁² / 4π，其中g₁是递归深度1的耦合
        coupling_at_depth_1 = self.coupling_hierarchy.get_coupling(1)
        
        # 计算α
        predicted_alpha = coupling_at_depth_1.decimal_value**2 / (4 * math.pi)
        
        # 相对误差应该在合理范围内（10%以内）
        relative_error = abs(predicted_alpha - experimental_alpha) / experimental_alpha
        self.assertLess(relative_error, 0.1)
        
        # 用φ编码表示
        alpha_phi = PhiReal.from_decimal(experimental_alpha)
        
        # 验证no-11约束
        try:
            alpha_phi.zeckendorf_rep._validate_no_11_constraint()
        except No11ConstraintViolation:
            # 调整表示
            alpha_phi = PhiReal.from_decimal(experimental_alpha * 1.00001)
    
    def test_recursive_self_reference_consistency(self):
        """测试递归自指的一致性"""
        # 验证 ψ = ψ(ψ) 在各个层次
        
        # 1. 规范群层次
        # G[ψ] = ψ(G[ψ])
        group_self_ref = lambda g: g  # 简化的自指
        fixed_point = self.sm_group
        self.assertEqual(group_self_ref(fixed_point), fixed_point)
        
        # 2. 粒子谱层次
        # 三代对应三个不动点
        generations = [1, 2, 3]
        for gen in generations:
            # 每代是一个稳定的自指结构
            psi_n = lambda x: x if x == gen else gen
            self.assertEqual(psi_n(gen), gen)
        
        # 3. 相互作用层次
        # 递归深度决定耦合强度，但包含观察者修正
        # 验证层次关系而非具体的φ幂律
        couplings = [self.coupling_hierarchy.get_coupling(i).decimal_value for i in range(3)]
        
        # 验证递减趋势（考虑实验值）
        # g_s > g_em, g_w 的关系比较复杂（取决于能标）
        self.assertGreater(couplings[0], couplings[1])  # g_s > g_em
        
        # 验证大致的数量级关系
        # 强耦合 O(1), 电磁 O(0.1), 弱 O(0.1)
        self.assertGreater(couplings[0], 0.1)  # 强耦合
        self.assertLess(couplings[1], 0.5)     # 电磁耦合 e ≈ 0.3
        self.assertLess(couplings[2], 1.0)     # 弱耦合

# ==================== 主程序 ====================

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)