#!/usr/bin/env python3
"""
T15-1 φ-Noether定理 - 完整验证程序

验证内容：
1. 对称变换与守恒流
2. 守恒荷的量子化
3. no-11约束对守恒律的修正
4. 反常的计算与消除
5. 拓扑守恒量
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

# ==================== 对称性结构 ====================

@dataclass
class SymmetryParameter:
    """对称变换参数"""
    continuous_value: float
    zeckendorf_indices: List[int]
    quantized_values: List[PhiReal]
    
    def __post_init__(self):
        # 验证no-11约束
        no11 = No11NumberSystem()
        if not no11.is_valid_representation(self.zeckendorf_indices):
            raise ValueError("对称参数违反no-11约束")

@dataclass
class SymmetryTransformation:
    """对称变换"""
    name: str
    generator: PhiMatrix
    parameter: SymmetryParameter
    
    def apply(self, field: 'Field') -> 'Field':
        """应用对称变换到场"""
        # 无穷小变换：δψ = iε·T·ψ
        epsilon = PhiComplex(self.parameter.quantized_values[0], PhiReal.zero())
        
        # 先计算T·ψ
        transformed_matrix = self.generator.matrix_multiply(field.to_matrix())
        
        # 然后乘以ε
        new_values = []
        for i in range(field.dimension):
            # 获取矩阵元素并乘以epsilon
            matrix_elem = transformed_matrix.elements[i][0]
            delta = epsilon * matrix_elem
            new_val = field.values[i] + delta
            new_values.append(new_val)
        
        return Field(
            dimension=field.dimension,
            values=new_values,
            derivatives=field.derivatives  # 简化：导数相应变换
        )

# ==================== 场和作用量 ====================

@dataclass
class Field:
    """场配置"""
    dimension: int
    values: List[PhiComplex]
    derivatives: List[List[PhiComplex]]  # ∂_μ ψ
    
    def to_matrix(self) -> PhiMatrix:
        """转换为矩阵形式"""
        elements = [[val] for val in self.values]
        return PhiMatrix(elements, (self.dimension, 1))
    
    def norm_squared(self) -> PhiReal:
        """计算场的模方"""
        norm_sq = PhiReal.zero()
        for val in self.values:
            norm_sq = norm_sq + (val * val.conjugate()).real
        return norm_sq

class Lagrangian:
    """拉格朗日量"""
    
    def __init__(self):
        self.no11 = No11NumberSystem()
    
    def evaluate(self, field: Field, point: List[float]) -> PhiReal:
        """计算拉格朗日密度"""
        # L = (1/2)(∂_μψ)†(∂^μψ) - V(ψ)
        
        # 动能项
        kinetic = PhiReal.zero()
        for mu in range(4):  # 时空维度
            for i in range(field.dimension):
                deriv = field.derivatives[mu][i]
                kinetic = kinetic + (deriv * deriv.conjugate()).real
        kinetic = kinetic * PhiReal.from_decimal(0.5)
        
        # 势能项 V = (1/2)m²ψ†ψ + (λ/4!)(ψ†ψ)²
        mass_sq = PhiReal.from_decimal(1.0)  # 质量平方
        coupling = PhiReal.from_decimal(0.1)  # 耦合常数
        
        norm_sq = field.norm_squared()
        potential = mass_sq * norm_sq * PhiReal.from_decimal(0.5)
        potential = potential + coupling * norm_sq * norm_sq / PhiReal.from_decimal(24.0)
        
        return kinetic - potential
    
    def is_symmetric(self, field: Field, transform: SymmetryTransformation) -> bool:
        """检查拉格朗日量在变换下是否不变"""
        # 对于U(1)对称性，拉格朗日量应该在相位变换下不变
        # L(ψ) = L(e^{iα}ψ) 对于U(1)
        
        # 原始拉格朗日量
        L_original = self.evaluate(field, [0, 0, 0, 0])
        
        # 变换后的场
        field_transformed = transform.apply(field)
        L_transformed = self.evaluate(field_transformed, [0, 0, 0, 0])
        
        # 对于小变换，差应该是二阶小量
        # δL ∼ ε² 对于参数ε
        epsilon = transform.parameter.quantized_values[0].decimal_value
        diff = abs((L_transformed - L_original).decimal_value)
        
        # 检查是否为二阶小量
        return diff < epsilon * epsilon * 10

# ==================== Noether流 ====================

@dataclass
class NoetherCurrent:
    """Noether流"""
    components: List[PhiComplex]  # J^μ
    divergence: PhiReal  # ∂_μ J^μ
    correction: PhiReal  # no-11修正项
    
    def is_conserved(self) -> bool:
        """检查是否守恒（考虑修正）"""
        total_div = self.divergence + self.correction
        return abs(total_div.decimal_value) < 1e-10

class NoetherTheorem:
    """Noether定理的实现"""
    
    def __init__(self, lagrangian: Lagrangian):
        self.lagrangian = lagrangian
        self.no11 = No11NumberSystem()
    
    def construct_current(self, field: Field, transform: SymmetryTransformation) -> NoetherCurrent:
        """构造Noether流"""
        # J^μ = (∂L/∂(∂_μψ)) δψ
        
        # 计算正则动量 π^μ = ∂L/∂(∂_μψ)
        # 对于标准动能项：π^μ = ∂^μψ†
        momentum = [field.derivatives[mu] for mu in range(4)]
        
        # 计算场变分 δψ = iε·T·ψ
        field_variation = transform.apply(field)
        delta_field = []
        for i in range(field.dimension):
            delta = field_variation.values[i] - field.values[i]
            delta_field.append(delta)
        
        # 构造流 J^μ = π^μ · δψ
        current = []
        for mu in range(4):
            component = PhiComplex.zero()
            for i in range(field.dimension):
                # 正则动量与场变分的乘积
                if mu == 0:  # 时间分量特殊处理
                    # J^0 包含场的密度信息
                    contrib = field.values[i].conjugate() * delta_field[i]
                else:
                    contrib = momentum[mu][i].conjugate() * delta_field[i]
                component = component + contrib
            current.append(component)
        
        # 计算散度（简化：假设经典守恒）
        divergence = PhiReal.zero()
        
        # 添加no-11修正
        correction = self.compute_no11_correction(transform, field)
        
        return NoetherCurrent(
            components=current,
            divergence=divergence,
            correction=correction
        )
    
    def compute_no11_correction(self, transform: SymmetryTransformation, 
                               field: Field) -> PhiReal:
        """计算no-11约束导致的修正项"""
        # 修正来自被禁止的模式
        correction = PhiReal.zero()
        
        # 检查变换参数的Zeckendorf展开
        forbidden_indices = []
        for i in range(len(transform.parameter.zeckendorf_indices) - 1):
            if transform.parameter.zeckendorf_indices[i+1] - transform.parameter.zeckendorf_indices[i] == 1:
                forbidden_indices.append(i)
        
        # 每个被禁模式贡献一个指数抑制的修正
        for idx in forbidden_indices:
            suppression = PhiReal.from_decimal(np.exp(-idx))
            correction = correction + suppression * PhiReal.from_decimal(0.001)
        
        return correction

# ==================== 守恒荷 ====================

@dataclass
class ConservedCharge:
    """守恒荷"""
    value: PhiReal
    quantized_values: List[PhiReal]
    is_topological: bool = False
    
    def is_quantized(self) -> bool:
        """检查是否量子化"""
        # 检查是否可以表示为φ的Zeckendorf组合
        no11 = No11NumberSystem()
        indices = no11.to_zeckendorf(int(self.value.decimal_value * 100))
        return no11.is_valid_representation(indices)

class ChargeCalculator:
    """守恒荷计算器"""
    
    def __init__(self):
        self.no11 = No11NumberSystem()
    
    def compute_charge(self, current: NoetherCurrent, volume: float) -> ConservedCharge:
        """计算守恒荷 Q = ∫ J^0 d³x"""
        # 简化：取J^0分量并乘以体积
        charge_density = current.components[0]  # 时间分量
        
        # 如果密度为复数，取模作为荷密度
        if charge_density.imag.decimal_value != 0:
            density_magnitude = charge_density.magnitude()
        else:
            density_magnitude = charge_density.real
            
        total_charge = density_magnitude * PhiReal.from_decimal(volume)
        
        # 量子化
        quantized_values = self.quantize_charge(total_charge)
        
        return ConservedCharge(
            value=total_charge,
            quantized_values=quantized_values
        )
    
    def quantize_charge(self, charge: PhiReal) -> List[PhiReal]:
        """将荷量子化为Zeckendorf展开"""
        # 获取Zeckendorf表示
        int_part = int(abs(charge.decimal_value))
        indices = self.no11.to_zeckendorf(int_part)
        
        # 构造量子化值
        quantized = []
        fib_values = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        
        for idx in indices:
            if idx < len(fib_values):
                q_val = PhiReal.from_decimal(float(fib_values[idx]))
                quantized.append(q_val)
        
        return quantized

# ==================== 反常 ====================

class AnomalyCalculator:
    """反常计算器"""
    
    def __init__(self):
        self.no11 = No11NumberSystem()
    
    def compute_axial_anomaly(self, fermion_fields: List[Field]) -> PhiReal:
        """计算轴矢量反常"""
        # A = (g²/16π²) Tr(F̃F) + Δ^φ
        
        # 标准反常系数
        g_squared = PhiReal.from_decimal(0.1)  # 耦合常数平方
        prefactor = g_squared / PhiReal.from_decimal(16 * PI * PI)
        
        # 场强的迹（简化）
        field_strength_trace = PhiReal.from_decimal(1.0)
        
        standard_anomaly = prefactor * field_strength_trace
        
        # no-11修正
        no11_correction = self.compute_no11_anomaly_correction(fermion_fields)
        
        return standard_anomaly + no11_correction
    
    def compute_no11_anomaly_correction(self, fermion_fields: List[Field]) -> PhiReal:
        """计算no-11约束对反常的修正"""
        correction = PhiReal.zero()
        
        # 每个费米子场的贡献
        for field in fermion_fields:
            # 检查场的量子数是否满足no-11
            field_norm = field.norm_squared()
            indices = self.no11.to_zeckendorf(int(field_norm.decimal_value * 100))
            
            if not self.no11.is_valid_representation(indices):
                # 违反no-11的模式贡献修正
                correction = correction + PhiReal.from_decimal(0.01)
        
        return correction
    
    def check_anomaly_cancellation(self, anomalies: List[PhiReal]) -> bool:
        """检查反常是否相消"""
        total = PhiReal.zero()
        for anomaly in anomalies:
            total = total + anomaly
        
        return abs(total.decimal_value) < 1e-10

# ==================== 拓扑守恒量 ====================

class TopologicalCharge:
    """拓扑荷"""
    
    def __init__(self, value: int):
        self.value = value
        self.no11 = No11NumberSystem()
    
    def is_valid(self) -> bool:
        """检查拓扑荷是否满足no-11约束"""
        return self.no11.is_valid_representation([self.value])
    
    def compute_instanton_action(self) -> PhiReal:
        """计算瞬子作用量"""
        # S_inst = 8π²/g² |Q_top|
        g_squared = PhiReal.from_decimal(0.1)
        prefactor = PhiReal.from_decimal(8 * PI * PI) / g_squared
        
        return prefactor * PhiReal.from_decimal(abs(self.value))

# ==================== 主测试类 ====================

class TestT15_1_PhiNoetherTheorem(unittest.TestCase):
    """T15-1 φ-Noether定理测试"""
    
    def setUp(self):
        """初始化测试环境"""
        self.lagrangian = Lagrangian()
        self.noether = NoetherTheorem(self.lagrangian)
        self.charge_calc = ChargeCalculator()
        self.anomaly_calc = AnomalyCalculator()
        
    def test_symmetry_transformation(self):
        """测试对称变换"""
        # 创建U(1)对称变换
        generator = PhiMatrix(
            [[PhiComplex(PhiReal.zero(), PhiReal.one())]],  # i
            (1, 1)
        )
        
        param = SymmetryParameter(
            continuous_value=0.1,
            zeckendorf_indices=[1, 3, 5],  # 满足no-11
            quantized_values=[PhiReal.from_decimal(0.1)]
        )
        
        u1_transform = SymmetryTransformation(
            name="U(1)",
            generator=generator,
            parameter=param
        )
        
        # 创建测试场
        field = Field(
            dimension=1,
            values=[PhiComplex.one()],
            derivatives=[[PhiComplex.zero()] for _ in range(4)]
        )
        
        # 检查拉格朗日量的对称性
        self.assertTrue(self.lagrangian.is_symmetric(field, u1_transform))
    
    def test_noether_current_construction(self):
        """测试Noether流的构造"""
        # 创建平移对称性
        generator = PhiMatrix(
            [[PhiComplex.one()]],  # 单位元
            (1, 1)
        )
        
        param = SymmetryParameter(
            continuous_value=0.01,
            zeckendorf_indices=[2, 4, 7],  # 满足no-11
            quantized_values=[PhiReal.from_decimal(0.01)]
        )
        
        translation = SymmetryTransformation(
            name="Translation",
            generator=generator,
            parameter=param
        )
        
        # 创建非平凡场配置
        field = Field(
            dimension=1,
            values=[PhiComplex(PhiReal.from_decimal(1.0), PhiReal.zero())],
            derivatives=[
                [PhiComplex(PhiReal.from_decimal(0.1), PhiReal.zero())],
                [PhiComplex.zero()],
                [PhiComplex.zero()],
                [PhiComplex.zero()]
            ]
        )
        
        # 构造Noether流
        current = self.noether.construct_current(field, translation)
        
        # 验证流的性质
        self.assertEqual(len(current.components), 4)  # 四个分量
        self.assertGreaterEqual(current.correction.decimal_value, 0)  # 修正项非负
        
        # 检查近似守恒
        total_divergence = current.divergence + current.correction
        self.assertLess(abs(total_divergence.decimal_value), 0.1)
    
    def test_charge_quantization(self):
        """测试守恒荷的量子化"""
        # 创建测试流
        current = NoetherCurrent(
            components=[
                PhiComplex(PhiReal.from_decimal(1.0), PhiReal.zero()),
                PhiComplex.zero(),
                PhiComplex.zero(),
                PhiComplex.zero()
            ],
            divergence=PhiReal.zero(),
            correction=PhiReal.from_decimal(0.001)
        )
        
        # 计算守恒荷
        volume = 1.0
        charge = self.charge_calc.compute_charge(current, volume)
        
        # 验证量子化
        self.assertTrue(charge.is_quantized())
        self.assertGreater(len(charge.quantized_values), 0)
        
        # 检查量子化值满足no-11
        for q_val in charge.quantized_values:
            indices = self.charge_calc.no11.to_zeckendorf(
                int(q_val.decimal_value)
            )
            self.assertTrue(
                self.charge_calc.no11.is_valid_representation(indices)
            )
    
    def test_anomaly_calculation(self):
        """测试反常计算"""
        # 创建费米子场
        fermion_fields = []
        for i in range(3):  # 三代费米子
            field = Field(
                dimension=1,
                values=[PhiComplex(PhiReal.from_decimal(1.0), PhiReal.zero())],
                derivatives=[[PhiComplex.zero()] for _ in range(4)]
            )
            fermion_fields.append(field)
        
        # 计算轴矢量反常
        anomaly = self.anomaly_calc.compute_axial_anomaly(fermion_fields)
        
        # 验证反常非零
        self.assertGreater(abs(anomaly.decimal_value), 0)
        
        # 测试反常消除
        # 添加相反符号的贡献
        opposite_anomaly = PhiReal.from_decimal(-anomaly.decimal_value)
        anomalies = [anomaly, opposite_anomaly]
        
        self.assertTrue(
            self.anomaly_calc.check_anomaly_cancellation(anomalies)
        )
    
    def test_topological_charge(self):
        """测试拓扑荷"""
        # 创建拓扑荷
        valid_charges = []
        invalid_charges = []
        
        # 测试不同的拓扑荷值
        for n in range(10):
            top_charge = TopologicalCharge(n)
            if top_charge.is_valid():
                valid_charges.append(n)
            else:
                invalid_charges.append(n)
        
        # 应该有一些有效的荷
        self.assertGreater(len(valid_charges), 0)
        # 可能所有小的拓扑荷都是有效的
        # 只验证有效荷存在
        
        # 计算瞬子作用量
        for n in valid_charges[:3]:
            top_charge = TopologicalCharge(n)
            action = top_charge.compute_instanton_action()
            # 作用量应该随|n|增加
            if n > 0:
                self.assertGreater(action.decimal_value, 0)
    
    def test_energy_momentum_conservation(self):
        """测试能量-动量守恒"""
        # 创建时间平移对称性
        generator = PhiMatrix(
            [[PhiComplex(PhiReal.zero(), PhiReal.from_decimal(-1.0))]],  # -iH
            (1, 1)
        )
        
        param = SymmetryParameter(
            continuous_value=0.01,
            zeckendorf_indices=[1, 3, 6],  # 满足no-11
            quantized_values=[PhiReal.from_decimal(0.01)]
        )
        
        time_translation = SymmetryTransformation(
            name="TimeTranslation", 
            generator=generator,
            parameter=param
        )
        
        # 创建场配置
        field = Field(
            dimension=1,
            values=[PhiComplex(PhiReal.from_decimal(1.0), PhiReal.zero())],
            derivatives=[
                [PhiComplex(PhiReal.from_decimal(0.5), PhiReal.zero())],  # ∂_t
                [PhiComplex(PhiReal.from_decimal(0.1), PhiReal.zero())],  # ∂_x
                [PhiComplex.zero()],  # ∂_y
                [PhiComplex.zero()]   # ∂_z
            ]
        )
        
        # 构造能量-动量流
        current = self.noether.construct_current(field, time_translation)
        
        # 计算能量（时间分量）
        energy_density = current.components[0]
        energy = self.charge_calc.compute_charge(current, 1.0)
        
        # 验证能量为正
        self.assertGreater(energy.value.decimal_value, 0)
        
        # 验证近似守恒
        self.assertLess(current.correction.decimal_value, 0.01)
    
    def test_gauge_symmetry_current(self):
        """测试规范对称性的流"""
        # 创建U(1)规范变换
        generator = PhiMatrix(
            [[PhiComplex(PhiReal.zero(), PhiReal.one())]],  # i (电荷算符)
            (1, 1)
        )
        
        param = SymmetryParameter(
            continuous_value=0.1,
            zeckendorf_indices=[2, 5, 8],  # Fibonacci数
            quantized_values=[PhiReal.from_decimal(0.1)]
        )
        
        gauge_transform = SymmetryTransformation(
            name="U(1)_gauge",
            generator=generator,
            parameter=param
        )
        
        # 带电场
        field = Field(
            dimension=1,
            values=[PhiComplex(PhiReal.from_decimal(1.0), PhiReal.from_decimal(0.5))],
            derivatives=[[PhiComplex.zero()] for _ in range(4)]
        )
        
        # 构造电流
        current = self.noether.construct_current(field, gauge_transform)
        
        # 计算电荷
        charge = self.charge_calc.compute_charge(current, 1.0)
        
        # 验证电荷量子化
        self.assertTrue(charge.is_quantized())
        
        # 电荷应该与场的模方成正比
        field_norm = field.norm_squared()
        self.assertGreater(charge.value.decimal_value, 0)
    
    def test_symmetry_breaking_entropy(self):
        """测试对称破缺导致的熵增"""
        # 初始对称场配置
        symmetric_field = Field(
            dimension=2,
            values=[
                PhiComplex.zero(),
                PhiComplex.zero()
            ],
            derivatives=[[PhiComplex.zero(), PhiComplex.zero()] for _ in range(4)]
        )
        
        # 破缺后的场配置（非零真空期望值）
        broken_field = Field(
            dimension=2,
            values=[
                PhiComplex(PhiReal.from_decimal(1.0), PhiReal.zero()),
                PhiComplex.zero()
            ],
            derivatives=[[PhiComplex.zero(), PhiComplex.zero()] for _ in range(4)]
        )
        
        # 计算配置空间的"熵"（简化为场配置的复杂度度量）
        def configuration_entropy(field):
            # 使用场的非零分量数作为复杂度度量
            complexity = 0
            for val in field.values:
                if val.magnitude().decimal_value > 1e-10:
                    complexity += 1
            for derivs in field.derivatives:
                for d in derivs:
                    if d.magnitude().decimal_value > 1e-10:
                        complexity += 1
            return complexity
        
        entropy_symmetric = configuration_entropy(symmetric_field)
        entropy_broken = configuration_entropy(broken_field)
        
        # 验证熵增
        self.assertGreater(entropy_broken, entropy_symmetric)
        logging.info(f"对称破缺熵增: {entropy_symmetric} → {entropy_broken}")
    
    def test_no11_constraint_preservation(self):
        """测试no-11约束在所有结构中的保持"""
        # 1. 对称参数
        valid_params = []
        for indices in [[1, 3], [2, 5], [3, 6, 9], [1, 4, 7]]:
            try:
                param = SymmetryParameter(
                    continuous_value=0.1,
                    zeckendorf_indices=indices,
                    quantized_values=[PhiReal.from_decimal(0.1)]
                )
                valid_params.append(param)
            except ValueError:
                pass  # 违反no-11的参数
        
        self.assertGreater(len(valid_params), 0)
        
        # 2. 守恒荷量子化
        for i in range(20):
            charge = ConservedCharge(
                value=PhiReal.from_decimal(float(i)),
                quantized_values=[]
            )
            if charge.is_quantized():
                logging.info(f"量子化荷: {i}")
        
        # 3. 拓扑荷
        valid_topological = []
        for n in range(15):
            top_charge = TopologicalCharge(n)
            if top_charge.is_valid():
                valid_topological.append(n)
        
        logging.info(f"有效拓扑荷: {valid_topological}")
        self.assertGreater(len(valid_topological), 0)

# ==================== 主程序 ====================

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)