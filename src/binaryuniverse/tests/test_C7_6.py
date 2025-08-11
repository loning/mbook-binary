#!/usr/bin/env python3
"""
C7-6: 能量-信息等价推论 - 完整测试程序

验证能量与信息通过观察者自指结构的等价关系，包括：
1. 基本能量-信息转换
2. 观察者热力学代价
3. φ修正的Landauer界限
4. Maxwell妖的能量界限
5. Zeckendorf约束验证
"""

import unittest
import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

# 导入基础类
try:
    from test_C17_1 import ObserverSystem
except ImportError:
    # 最小实现
    class ObserverSystem:
        def __init__(self, dimension: int):
            self.phi = (1 + np.sqrt(5)) / 2
            self.dim = dimension
            self.state = np.zeros(dimension)
            self.state[0] = 1


class EnergyInformationEquivalence:
    """能量-信息等价系统"""
    
    def __init__(self, temperature=300.0):
        self.phi = (1 + np.sqrt(5)) / 2
        self.k_B = 1.380649e-23  # 玻尔兹曼常数 (J/K)
        self.T_observer = temperature
        self.log2_phi = np.log2(self.phi)
        
    def energy_to_information(self, energy):
        """将能量转换为等价信息量(比特)"""
        if energy <= 0:
            return 0
        return (energy * self.phi) / (self.k_B * self.T_observer * self.log2_phi)
    
    def information_to_energy(self, bits):
        """将信息量转换为等价能量(焦耳)"""
        if bits <= 0:
            return 0.0
        return (bits * self.k_B * self.T_observer * self.log2_phi) / self.phi
    
    def landauer_limit_corrected(self):
        """修正的Landauer极限"""
        return self.phi**2 * self.k_B * self.T_observer * np.log(2)
    
    def observation_cost(self, bits):
        """观察者获取信息的热力学代价"""
        return self.phi**2 * bits * self.k_B * self.T_observer * np.log(2)
    
    def maxwell_demon_cost(self, bits_acquired):
        """Maxwell妖获取信息的最小代价"""
        return self.phi**2 * bits_acquired * self.k_B * self.T_observer * np.log(2)
    
    def computation_cost(self, irreversible_ops):
        """不可逆计算的热力学代价"""
        return self.phi * irreversible_ops * self.k_B * self.T_observer * np.log(2)
    
    def storage_energy(self, bits):
        """信息存储的最小能量需求"""
        return bits * self.k_B * self.T_observer / self.log2_phi
    
    def observer_temperature(self, observed_energy, information_bits):
        """从能量-信息平衡计算观察者温度"""
        if information_bits == 0:
            return float('inf')
        # 从等价关系 E * φ = I * k_B * T * log2(φ) 推导
        return (observed_energy * self.phi) / (information_bits * self.k_B * self.log2_phi)
    
    def verify_equivalence(self, energy, information, tolerance=1e-10):
        """验证能量-信息等价关系"""
        if energy == 0 and information == 0:
            return True
            
        left_side = energy * self.phi
        right_side = information * self.k_B * self.T_observer * self.log2_phi
        
        if max(abs(left_side), abs(right_side)) == 0:
            return left_side == right_side
            
        relative_error = abs(left_side - right_side) / max(abs(left_side), abs(right_side))
        return relative_error < tolerance
    
    def zeckendorf_entropy(self, n_bits):
        """计算n比特系统的Zeckendorf熵"""
        F_n_plus_2 = self._fibonacci(n_bits + 2)
        if F_n_plus_2 <= 0:
            return 0.0
        return self.k_B * np.log(F_n_plus_2)
    
    def _fibonacci(self, n):
        """计算第n个Fibonacci数"""
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        elif n == 2:
            return 1
        
        # 使用Binet公式
        phi_n = self.phi ** n
        psi_n = ((-1/self.phi) ** n)
        return int(round((phi_n - psi_n) / np.sqrt(5)))
    
    def information_density(self):
        """Zeckendorf约束下的信息密度"""
        return self.log2_phi
    
    def quantum_measurement_cost(self, hbar_omega, hilbert_dim):
        """量子测量的最小能量代价"""
        hbar = 1.054571817e-34  # 约化普朗克常数
        return self.phi * hbar_omega * self.log2_phi / np.log2(hilbert_dim)
    
    def biological_efficiency(self, body_temperature=310):
        """生物信息处理的理论效率"""
        return self.phi**2 * self.k_B * body_temperature * np.log(2)
    
    def hawking_temperature(self, black_hole_mass):
        """黑洞的Hawking温度（简化模型）"""
        c = 299792458  # 光速
        G = 6.67430e-11  # 引力常数
        hbar = 1.054571817e-34
        
        # Hawking温度公式
        return (hbar * c**3) / (8 * np.pi * self.k_B * G * black_hole_mass)
    
    def _enforce_no11(self, state):
        """强制no-11约束"""
        result = np.array(state, dtype=int)
        for i in range(1, len(result)):
            if result[i-1] == 1 and result[i] == 1:
                result[i] = 0
        return result
    
    def verify_no11_constraint(self, bit_string):
        """验证no-11约束"""
        for i in range(1, len(bit_string)):
            if bit_string[i-1] == 1 and bit_string[i] == 1:
                return False
        return True


class TestEnergyInformationEquivalence(unittest.TestCase):
    """C7-6 能量-信息等价测试套件"""
    
    def setUp(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.k_B = 1.380649e-23
        self.room_temp = 300.0  # 室温
        self.eq_system = EnergyInformationEquivalence(self.room_temp)
        
    def test_basic_energy_information_conversion(self):
        """测试基本能量-信息转换"""
        # 测试能量到信息的转换
        energy = 1e-20  # 焦耳
        info_bits = self.eq_system.energy_to_information(energy)
        
        # 验证结果为正数
        self.assertGreater(info_bits, 0, "信息量应为正数")
        
        # 反向转换
        recovered_energy = self.eq_system.information_to_energy(info_bits)
        
        # 验证双向转换精度
        relative_error = abs(energy - recovered_energy) / energy
        self.assertLess(relative_error, 1e-10, "双向转换精度应在10^-10内")
        
    def test_equivalence_relation(self):
        """测试能量-信息等价关系"""
        test_cases = [
            (1e-21, 1),       # 极小能量
            (1e-18, 500),     # 典型能量 (实际约563比特)
            (1e-15, 500000),  # 较大能量
        ]
        
        for energy, expected_bits_order in test_cases:
            info_bits = self.eq_system.energy_to_information(energy)
            
            # 验证等价关系
            is_equivalent = self.eq_system.verify_equivalence(energy, info_bits)
            self.assertTrue(is_equivalent, f"能量{energy}J与信息{info_bits}bit应该等价")
            
            # 验证信息量级
            self.assertAlmostEqual(np.log10(info_bits), np.log10(expected_bits_order), delta=1.5,
                                 msg="信息量级应在预期范围内")
    
    def test_landauer_limit_correction(self):
        """测试φ修正的Landauer界限"""
        standard_landauer = self.k_B * self.room_temp * np.log(2)
        corrected_landauer = self.eq_system.landauer_limit_corrected()
        
        # 验证修正因子
        correction_factor = corrected_landauer / standard_landauer
        expected_factor = self.phi ** 2
        
        self.assertAlmostEqual(correction_factor, expected_factor, places=10,
                              msg="Landauer界限的修正因子应为φ²")
        
        # 验证修正后的值更大
        self.assertGreater(corrected_landauer, standard_landauer,
                          "φ修正的Landauer界限应大于标准值")
    
    def test_observation_cost(self):
        """测试观察者的热力学代价"""
        for bits in [1, 10, 100, 1000]:
            cost = self.eq_system.observation_cost(bits)
            
            # 验证代价与比特数成正比
            expected_cost = self.phi**2 * bits * self.k_B * self.room_temp * np.log(2)
            self.assertAlmostEqual(cost, expected_cost, places=15,
                                  msg=f"观察{bits}比特的代价计算错误")
            
            # 验证代价为正
            self.assertGreater(cost, 0, "观察代价应为正数")
            
            # 验证线性关系
            if bits > 1:
                cost_per_bit = cost / bits
                single_bit_cost = self.eq_system.observation_cost(1)
                self.assertAlmostEqual(cost_per_bit, single_bit_cost, places=15,
                                      msg="观察代价应与比特数呈线性关系")
    
    def test_maxwell_demon_bound(self):
        """测试Maxwell妖的热力学界限"""
        for acquired_bits in [1, 5, 20, 100]:
            demon_cost = self.eq_system.maxwell_demon_cost(acquired_bits)
            
            # 验证界限公式
            expected_bound = self.phi**2 * acquired_bits * self.k_B * self.room_temp * np.log(2)
            self.assertAlmostEqual(demon_cost, expected_bound, places=15,
                                  msg="Maxwell妖的代价界限计算错误")
            
            # 验证代价不小于标准Landauer界限
            standard_cost = acquired_bits * self.k_B * self.room_temp * np.log(2)
            self.assertGreaterEqual(demon_cost, standard_cost,
                                   "Maxwell妖的代价应不小于标准Landauer界限")
    
    def test_computation_cost(self):
        """测试计算的热力学代价"""
        for ops in [1, 10, 100, 1000]:
            comp_cost = self.eq_system.computation_cost(ops)
            
            # 验证计算代价公式
            expected_cost = self.phi * ops * self.k_B * self.room_temp * np.log(2)
            self.assertAlmostEqual(comp_cost, expected_cost, places=15,
                                  msg="不可逆计算代价计算错误")
            
            # 验证代价与操作数成正比
            if ops > 1:
                cost_per_op = comp_cost / ops
                single_op_cost = self.eq_system.computation_cost(1)
                self.assertAlmostEqual(cost_per_op, single_op_cost, places=15,
                                      msg="计算代价应与操作数呈线性关系")
    
    def test_information_storage_energy(self):
        """测试信息存储的能量需求"""
        for bits in [1, 8, 64, 512]:
            storage_energy = self.eq_system.storage_energy(bits)
            
            # 验证存储能量公式
            expected_energy = bits * self.k_B * self.room_temp / self.eq_system.log2_phi
            self.assertAlmostEqual(storage_energy, expected_energy, places=15,
                                  msg="信息存储能量计算错误")
            
            # 验证存储能量为正
            self.assertGreater(storage_energy, 0, "存储能量应为正数")
            
            # 验证与比特数的线性关系
            energy_per_bit = storage_energy / bits
            self.assertAlmostEqual(energy_per_bit, 
                                  self.k_B * self.room_temp / self.eq_system.log2_phi,
                                  places=15, msg="存储能量应与比特数呈线性关系")
    
    def test_observer_temperature_calculation(self):
        """测试观察者温度计算"""
        # 已知能量和信息量计算温度
        energy = 1e-19  # 焦耳
        info_bits = 100
        
        calculated_temp = self.eq_system.observer_temperature(energy, info_bits)
        
        # 验证温度为正数
        self.assertGreater(calculated_temp, 0, "观察者温度应为正数")
        
        # 验证温度计算的一致性
        # 使用计算出的温度创建新系统，验证能量-信息等价
        test_system = EnergyInformationEquivalence(calculated_temp)
        is_equivalent = test_system.verify_equivalence(energy, info_bits, tolerance=1e-6)
        self.assertTrue(is_equivalent, f"计算出的观察者温度{calculated_temp:.2f}K应满足能量-信息等价")
        
        # 另一种验证：直接检查等价公式
        left_side = energy * self.phi
        right_side = info_bits * self.k_B * calculated_temp * test_system.log2_phi
        relative_error = abs(left_side - right_side) / max(abs(left_side), abs(right_side))
        self.assertLess(relative_error, 1e-6, f"直接等价验证相对误差应小于1e-6，实际为{relative_error:.2e}")
    
    def test_zeckendorf_entropy(self):
        """测试Zeckendorf约束下的熵计算"""
        for n in [1, 2, 3, 5, 8, 13]:  # Fibonacci数
            entropy = self.eq_system.zeckendorf_entropy(n)
            
            # 验证熵为正数
            self.assertGreater(entropy, 0, f"{n}比特系统的熵应为正数")
            
            # 验证熵的增长趋势
            if n > 1:
                prev_entropy = self.eq_system.zeckendorf_entropy(n-1)
                self.assertGreater(entropy, prev_entropy, "熵应随比特数增长")
            
            # 验证Fibonacci熵的特殊性质
            F_n_plus_2 = self.eq_system._fibonacci(n + 2)
            expected_entropy = self.k_B * np.log(F_n_plus_2)
            self.assertAlmostEqual(entropy, expected_entropy, places=15,
                                  msg="Zeckendorf熵计算错误")
    
    def test_information_density(self):
        """测试信息密度"""
        density = self.eq_system.information_density()
        
        # 验证密度等于log2(φ)
        expected_density = np.log2(self.phi)
        self.assertAlmostEqual(density, expected_density, places=15,
                              msg="信息密度应等于log2(φ)")
        
        # 验证密度在0和1之间
        self.assertGreater(density, 0, "信息密度应大于0")
        self.assertLess(density, 1, "信息密度应小于1")
        
        # 验证具体数值
        self.assertAlmostEqual(density, 0.694241914, places=6,
                              msg="信息密度应约为0.694")
    
    def test_fibonacci_calculation(self):
        """测试Fibonacci数计算"""
        known_fibonacci = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        
        for i, expected in enumerate(known_fibonacci):
            calculated = self.eq_system._fibonacci(i)
            self.assertEqual(calculated, expected,
                           f"第{i}个Fibonacci数应为{expected}，实际计算为{calculated}")
    
    def test_no11_constraint_verification(self):
        """测试no-11约束验证"""
        # 合法的no-11序列
        valid_sequences = [
            [1, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 1],
        ]
        
        for seq in valid_sequences:
            self.assertTrue(self.eq_system.verify_no11_constraint(seq),
                           f"序列{seq}应满足no-11约束")
        
        # 非法的序列（包含连续1）
        invalid_sequences = [
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
        ]
        
        for seq in invalid_sequences:
            self.assertFalse(self.eq_system.verify_no11_constraint(seq),
                            f"序列{seq}不应满足no-11约束")
    
    def test_biological_efficiency(self):
        """测试生物效率理论值"""
        body_temp = 310  # 体温(K)
        bio_efficiency = self.eq_system.biological_efficiency(body_temp)
        
        # 验证生物效率公式
        expected = self.phi**2 * self.k_B * body_temp * np.log(2)
        self.assertAlmostEqual(bio_efficiency, expected, places=15,
                              msg="生物效率计算错误")
        
        # 验证数量级
        # 在体温下，每比特约10^-20焦耳
        expected_order = 1e-20
        self.assertAlmostEqual(np.log10(bio_efficiency), np.log10(expected_order), delta=0.5,
                              msg="生物效率数量级应在10^-20焦耳/比特量级")
    
    def test_quantum_measurement_cost(self):
        """测试量子测量成本"""
        # 典型参数
        hbar = 1.054571817e-34
        frequency = 1e9  # 1 GHz
        hbar_omega = hbar * 2 * np.pi * frequency
        hilbert_dim = 2  # 二能级系统
        
        measurement_cost = self.eq_system.quantum_measurement_cost(hbar_omega, hilbert_dim)
        
        # 验证测量成本为正
        self.assertGreater(measurement_cost, 0, "量子测量成本应为正数")
        
        # 验证成本公式
        expected_cost = self.phi * hbar_omega * self.eq_system.log2_phi / np.log2(hilbert_dim)
        self.assertAlmostEqual(measurement_cost, expected_cost, places=15,
                              msg="量子测量成本计算错误")
        
        # 验证维度依赖性
        higher_dim_cost = self.eq_system.quantum_measurement_cost(hbar_omega, 4)
        self.assertLess(higher_dim_cost, measurement_cost,
                       "更高维度系统的测量成本应更小")
    
    def test_temperature_dependence(self):
        """测试温度依赖性"""
        temperatures = [100, 300, 500, 1000]  # 不同温度
        energy = 1e-19  # 固定能量
        
        info_bits = []
        for T in temperatures:
            system = EnergyInformationEquivalence(T)
            bits = system.energy_to_information(energy)
            info_bits.append(bits)
        
        # 验证温度越高，等价信息量越少（反比关系）
        for i in range(1, len(temperatures)):
            self.assertLess(info_bits[i], info_bits[i-1],
                           f"温度{temperatures[i]}K下的等价信息应少于{temperatures[i-1]}K")
    
    def test_energy_conservation(self):
        """测试能量守恒"""
        # 测试多个能量-信息转换过程的总能量
        initial_energies = [1e-21, 2e-21, 3e-21, 4e-21]
        total_initial_energy = sum(initial_energies)
        
        # 转换为信息再转换回能量
        total_recovered_energy = 0
        for energy in initial_energies:
            bits = self.eq_system.energy_to_information(energy)
            recovered = self.eq_system.information_to_energy(bits)
            total_recovered_energy += recovered
        
        # 验证总能量守恒
        relative_error = abs(total_initial_energy - total_recovered_energy) / total_initial_energy
        self.assertLess(relative_error, 1e-10, "能量转换过程应保持总能量守恒")
    
    def test_entropy_increase_compatibility(self):
        """测试与熵增原理的兼容性"""
        # 模拟观察过程
        initial_energy = 1e-19
        observed_bits = self.eq_system.energy_to_information(initial_energy)
        observation_cost = self.eq_system.observation_cost(observed_bits)
        
        # 计算熵变
        initial_entropy = 0  # 假设初态为确定态
        final_entropy = self.eq_system.zeckendorf_entropy(int(observed_bits))
        
        # 验证总熵增加
        min_entropy_increase = self.k_B * self.eq_system.log2_phi
        entropy_increase = final_entropy - initial_entropy
        
        self.assertGreaterEqual(entropy_increase, min_entropy_increase * 0.9,
                               "观察过程应满足最小熵增原理")
    
    def test_physical_constants_precision(self):
        """测试物理常数精度"""
        # 验证玻尔兹曼常数
        self.assertAlmostEqual(self.eq_system.k_B, 1.380649e-23, places=23,
                              msg="玻尔兹曼常数精度")
        
        # 验证黄金比率
        expected_phi = (1 + np.sqrt(5)) / 2
        self.assertAlmostEqual(self.eq_system.phi, expected_phi, places=15,
                              msg="黄金比率精度")
        
        # 验证log2(φ)
        expected_log2_phi = np.log2(expected_phi)
        self.assertAlmostEqual(self.eq_system.log2_phi, expected_log2_phi, places=15,
                              msg="log2(φ)精度")


if __name__ == '__main__':
    unittest.main(verbosity=2)