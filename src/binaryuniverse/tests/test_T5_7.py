"""
测试T5-7：Landauer原理定理

验证：
1. 比特擦除的最小能量 E >= k_B * T * ln(2) * n_bits
2. 描述不可擦除性（熵增原理）
3. φ-表示的能效优势
4. 可逆计算的零能量可能性
5. 温度依赖性
6. no-11约束的局部性影响
"""

import unittest
import numpy as np
import math
from typing import List, Set, Dict, Tuple, Optional
import random

# 物理常数
k_B = 1.380649e-23  # Boltzmann常数 (J/K)
ln2 = math.log(2)   # ln(2)

class ThermodynamicsCalculator:
    """热力学计算器"""
    
    def __init__(self):
        self.k_B = k_B
        self.phi = (1 + math.sqrt(5)) / 2
        self.log_phi = math.log2(self.phi)
    
    def bit_erasure_energy(self, n_bits: int, T: float) -> float:
        """计算擦除n个比特所需的最小能量（焦耳）"""
        return self.k_B * T * ln2 * n_bits
    
    def bit_erasure_energy_eV(self, n_bits: int, T: float) -> float:
        """计算擦除能量（电子伏特）"""
        joules = self.bit_erasure_energy(n_bits, T)
        return joules / 1.602176634e-19  # 转换为eV
    
    def entropy_change_bits(self, states_before: int, states_after: int) -> float:
        """计算熵变（比特为单位）"""
        if states_before == 0 or states_after == 0:
            return float('inf')
        return math.log2(states_after / states_before)
    
    def minimum_work(self, entropy_change_bits: float, T: float) -> float:
        """根据熵变计算最小功（焦耳）"""
        return self.k_B * T * ln2 * entropy_change_bits


class InformationEraser:
    """信息擦除器"""
    
    def __init__(self):
        self.calc = ThermodynamicsCalculator()
        self.erasure_history = []
    
    def erase_bits(self, state: str, positions: List[int], T: float = 300) -> Tuple[str, float]:
        """
        擦除指定位置的比特（设为0）
        返回：(新状态, 能量代价)
        """
        new_state = list(state)
        erased_count = 0
        
        for pos in positions:
            if 0 <= pos < len(state) and state[pos] == '1':
                new_state[pos] = '0'
                erased_count += 1
        
        energy = self.calc.bit_erasure_energy(erased_count, T)
        
        # 记录历史
        self.erasure_history.append({
            'operation': 'bit_erase',
            'positions': positions,
            'bits_erased': erased_count,
            'energy': energy,
            'temperature': T
        })
        
        return ''.join(new_state), energy
    
    def reset_to_zero(self, state: str, T: float = 300) -> Tuple[str, float]:
        """将所有比特重置为0"""
        ones_count = state.count('1')
        new_state = '0' * len(state)
        energy = self.calc.bit_erasure_energy(ones_count, T)
        
        self.erasure_history.append({
            'operation': 'reset',
            'bits_erased': ones_count,
            'energy': energy,
            'temperature': T
        })
        
        return new_state, energy
    
    def reversible_transform(self, state: str) -> Tuple[str, float]:
        """可逆变换（理论上零能量）"""
        # 简单的可逆操作：位翻转
        new_state = ''.join('1' if b == '0' else '0' for b in state)
        
        self.erasure_history.append({
            'operation': 'reversible',
            'energy': 0.0
        })
        
        return new_state, 0.0


class PhiRepresentationEraser:
    """φ-表示擦除器"""
    
    def __init__(self):
        self.calc = ThermodynamicsCalculator()
        self.phi = (1 + math.sqrt(5)) / 2
    
    def erase_with_constraint(self, state: str, positions: List[int], T: float = 300) -> Tuple[str, float]:
        """
        考虑no-11约束的擦除
        擦除可能影响相邻位
        """
        new_state = list(state)
        total_erased = 0
        
        # 先执行请求的擦除
        for pos in positions:
            if 0 <= pos < len(state) and new_state[pos] == '1':
                new_state[pos] = '0'
                total_erased += 1
        
        # 然后修复所有no-11违反
        i = 0
        while i < len(new_state) - 1:
            if new_state[i] == '1' and new_state[i+1] == '1':
                # 发现11模式，擦除第二个1
                new_state[i+1] = '0'
                total_erased += 1
                # 不增加i，因为可能还有连续的1
            else:
                i += 1
        
        energy = self.calc.bit_erasure_energy(total_erased, T)
        return ''.join(new_state), energy
    
    def compare_efficiency(self, n_bits: int, T: float = 300) -> Dict[str, float]:
        """比较标准表示和φ-表示的擦除效率"""
        # 标准表示：n比特
        standard_energy = self.calc.bit_erasure_energy(n_bits, T)
        
        # φ-表示：更多比特（因为编码扩展）
        phi_bits = int(n_bits / self.calc.log_phi)
        phi_energy = self.calc.bit_erasure_energy(phi_bits, T)
        
        return {
            'standard_bits': n_bits,
            'phi_bits': phi_bits,
            'standard_energy': standard_energy,
            'phi_energy': phi_energy,
            'energy_ratio': phi_energy / standard_energy if standard_energy > 0 else 0,
            'bits_ratio': phi_bits / n_bits if n_bits > 0 else 0
        }


class SelfReferentialSystem:
    """自指完备系统（用于测试描述擦除）"""
    
    def __init__(self):
        self.state = ""
        self.descriptions: Set[str] = set()
        self.calc = ThermodynamicsCalculator()
    
    def add_description(self, desc: str):
        """添加描述"""
        self.descriptions.add(desc)
    
    def attempt_erase_description(self, desc: str) -> bool:
        """
        尝试擦除描述
        根据熵增原理，这应该失败
        """
        if desc in self.descriptions:
            # 不能真正擦除（违反熵增）
            return False
        return True
    
    def transform_description(self, desc_from: str, desc_to: str) -> float:
        """
        描述转换（保持|D|不变）
        理论上可以零能量完成
        """
        if desc_from in self.descriptions:
            self.descriptions.remove(desc_from)
            self.descriptions.add(desc_to)
            return 0.0  # 可逆转换
        return float('inf')  # 无效转换
    
    def compute_entropy(self) -> float:
        """计算系统熵 H = log|D|"""
        if len(self.descriptions) == 0:
            return 0.0
        return math.log2(len(self.descriptions))


class ComputationSimulator:
    """计算模拟器"""
    
    def __init__(self):
        self.calc = ThermodynamicsCalculator()
    
    def irreversible_and_gate(self, a: str, b: str, T: float = 300) -> Tuple[str, float]:
        """
        不可逆AND门
        擦除一个输入比特的信息
        """
        result = '1' if a == '1' and b == '1' else '0'
        # AND门擦除约1比特信息（平均）
        energy = self.calc.bit_erasure_energy(1, T)
        return result, energy
    
    def reversible_cnot_gate(self, control: str, target: str) -> Tuple[str, str, float]:
        """
        可逆CNOT门
        不擦除信息，理论上零能量
        """
        new_target = '1' if control == '1' and target == '0' else \
                     '0' if control == '1' and target == '1' else target
        return control, new_target, 0.0
    
    def computation_with_garbage(self, input_bits: int, garbage_bits: int, T: float = 300) -> float:
        """
        带垃圾比特的计算
        返回擦除垃圾比特的能量代价
        """
        return self.calc.bit_erasure_energy(garbage_bits, T)


class TestT5_7LandauerPrinciple(unittest.TestCase):
    """T5-7 Landauer原理定理测试"""
    
    def setUp(self):
        """初始化测试环境"""
        self.calc = ThermodynamicsCalculator()
        self.eraser = InformationEraser()
        self.phi_eraser = PhiRepresentationEraser()
        self.system = SelfReferentialSystem()
        self.computer = ComputationSimulator()
    
    def test_minimum_erasure_energy(self):
        """测试1：比特擦除的最小能量"""
        print("\n测试1：Landauer极限验证")
        
        T = 300  # 室温
        
        print(f"  温度: {T}K")
        print(f"  k_B*T*ln(2) = {self.calc.bit_erasure_energy(1, T):.3e} J")
        print(f"            = {self.calc.bit_erasure_energy_eV(1, T):.3e} eV")
        
        print("\n  比特数  能量(J)       能量(eV)")
        print("  ------  -----------  -----------")
        
        for n_bits in [1, 10, 100, 1000]:
            energy_J = self.calc.bit_erasure_energy(n_bits, T)
            energy_eV = self.calc.bit_erasure_energy_eV(n_bits, T)
            
            print(f"  {n_bits:6}  {energy_J:.3e}  {energy_eV:.3e}")
        
        # 验证线性关系
        energy_1 = self.calc.bit_erasure_energy(1, T)
        energy_100 = self.calc.bit_erasure_energy(100, T)
        
        self.assertAlmostEqual(energy_100, 100 * energy_1, places=10,
                             msg="能量应该与比特数成正比")
    
    def test_description_non_erasability(self):
        """测试2：描述不可擦除性"""
        print("\n测试2：自指系统描述擦除约束")
        
        # 初始化系统
        for i in range(5):
            self.system.add_description(f"desc_{i}")
        
        initial_entropy = self.system.compute_entropy()
        initial_count = len(self.system.descriptions)
        
        print(f"  初始描述数: {initial_count}")
        print(f"  初始系统熵: {initial_entropy:.3f} bits")
        
        # 尝试擦除描述
        print("\n  尝试擦除描述...")
        can_erase = self.system.attempt_erase_description("desc_0")
        
        print(f"  可以擦除? {can_erase}")
        print(f"  原因: 违反熵增原理（|D|不能减少）")
        
        # 描述转换
        print("\n  描述转换测试...")
        energy = self.system.transform_description("desc_1", "desc_1_transformed")
        
        final_count = len(self.system.descriptions)
        final_entropy = self.system.compute_entropy()
        
        print(f"  转换能量: {energy} J（理论零能量）")
        print(f"  最终描述数: {final_count}")
        print(f"  最终系统熵: {final_entropy:.3f} bits")
        
        self.assertEqual(final_count, initial_count,
                        "描述数应该保持不变")
        self.assertEqual(energy, 0.0,
                        "可逆转换应该零能量")
    
    def test_phi_representation_efficiency(self):
        """测试3：φ-表示的能效优势"""
        print("\n测试3：φ-表示能效分析")
        
        T = 300
        
        print("  信息量  标准比特  φ-比特  标准能量(J)  φ-能量(J)  能量比")
        print("  ------  --------  ------  -----------  ---------  ------")
        
        for info_bits in [10, 50, 100, 500]:
            result = self.phi_eraser.compare_efficiency(info_bits, T)
            
            print(f"  {info_bits:6}  {result['standard_bits']:8}  "
                  f"{result['phi_bits']:6}  {result['standard_energy']:.3e}  "
                  f"{result['phi_energy']:.3e}  {result['energy_ratio']:.3f}")
        
        # 验证比率
        expected_ratio = 1 / self.calc.log_phi
        actual_ratio = result['bits_ratio']
        
        print(f"\n  理论比特比: {expected_ratio:.3f}")
        print(f"  实际比特比: {actual_ratio:.3f}")
        
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=2,
                             msg="比特比应该接近1/log2(φ)")
    
    def test_reversible_computation(self):
        """测试4：可逆计算的零能量"""
        print("\n测试4：可逆vs不可逆计算")
        
        T = 300
        
        # 不可逆AND门
        print("  不可逆AND门:")
        result_and, energy_and = self.computer.irreversible_and_gate('1', '1', T)
        print(f"    输入: (1,1) -> 输出: {result_and}")
        print(f"    能量: {energy_and:.3e} J")
        
        # 可逆CNOT门
        print("\n  可逆CNOT门:")
        c, t, energy_cnot = self.computer.reversible_cnot_gate('1', '0')
        print(f"    输入: (1,0) -> 输出: ({c},{t})")
        print(f"    能量: {energy_cnot} J")
        
        # 带垃圾比特的计算
        print("\n  带垃圾比特的计算:")
        garbage_bits = 20
        energy_garbage = self.computer.computation_with_garbage(10, garbage_bits, T)
        print(f"    输入比特: 10")
        print(f"    垃圾比特: {garbage_bits}")
        print(f"    擦除能量: {energy_garbage:.3e} J")
        
        self.assertGreater(energy_and, 0, "不可逆门应该消耗能量")
        self.assertEqual(energy_cnot, 0, "可逆门理论上零能量")
    
    def test_temperature_dependence(self):
        """测试5：温度依赖性"""
        print("\n测试5：擦除能量的温度依赖")
        
        temperatures = [1, 10, 100, 300, 1000]  # K
        n_bits = 100
        
        print("  温度(K)  单比特能量(J)    100比特能量(J)")
        print("  -------  --------------  ---------------")
        
        energies = []
        for T in temperatures:
            e1 = self.calc.bit_erasure_energy(1, T)
            e100 = self.calc.bit_erasure_energy(n_bits, T)
            energies.append(e100)
            
            print(f"  {T:7}  {e1:.3e}      {e100:.3e}")
        
        # 验证线性关系
        for i in range(len(temperatures)):
            expected = k_B * temperatures[i] * ln2 * n_bits
            self.assertAlmostEqual(energies[i], expected, places=20,
                                 msg=f"能量应该与温度成正比")
    
    def test_local_constraint_effects(self):
        """测试6：no-11约束的局部性影响"""
        print("\n测试6：约束擦除的局部效应")
        
        # 测试不同的擦除场景
        test_cases = [
            ("0110101", [2], "预期简单擦除"),
            ("0111101", [2], "可能触发连锁"),
            ("1111111", [3], "密集1区域"),
        ]
        
        T = 300
        
        print("  初始状态  擦除位  最终状态  擦除数  能量(J)")
        print("  --------  ------  --------  ------  --------")
        
        for initial, positions, desc in test_cases:
            final, energy = self.phi_eraser.erase_with_constraint(initial, positions, T)
            
            bits_changed = sum(1 for i in range(len(initial)) 
                             if initial[i] != final[i])
            
            print(f"  {initial}  {positions[0]:6}  {final}  "
                  f"{bits_changed:6}  {energy:.3e}")
            
            # 验证no-11约束
            self.assertNotIn('11', final, "结果不应包含11")
    
    def test_erasure_history(self):
        """测试7：擦除操作历史追踪"""
        print("\n测试7：信息擦除历史")
        
        # 执行一系列操作
        state = "10110101"
        T = 300
        
        print("  操作序列:")
        
        # 操作1：擦除特定位
        state, e1 = self.eraser.erase_bits(state, [1, 3, 5], T)
        print(f"    1. 擦除位[1,3,5]: {state}, 能量={e1:.3e}J")
        
        # 操作2：可逆变换
        state, e2 = self.eraser.reversible_transform(state)
        print(f"    2. 可逆变换: {state}, 能量={e2}J")
        
        # 操作3：重置
        state, e3 = self.eraser.reset_to_zero(state, T)
        print(f"    3. 重置为0: {state}, 能量={e3:.3e}J")
        
        # 分析历史
        total_energy = sum(h['energy'] for h in self.eraser.erasure_history)
        total_bits = sum(h.get('bits_erased', 0) for h in self.eraser.erasure_history)
        
        print(f"\n  总结:")
        print(f"    操作次数: {len(self.eraser.erasure_history)}")
        print(f"    总擦除比特: {total_bits}")
        print(f"    总能量消耗: {total_energy:.3e}J")
        
        # 验证能量守恒
        expected_energy = self.calc.bit_erasure_energy(total_bits, T)
        self.assertAlmostEqual(total_energy, expected_energy, places=20,
                             msg="总能量应该等于擦除比特数的Landauer极限")


if __name__ == '__main__':
    unittest.main()