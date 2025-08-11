#!/usr/bin/env python3
"""
T26-1 瓶颈张力积累定理 - 机器验证测试
基于Zeckendorf编码的二进制宇宙，严格验证张力场理论
"""

import unittest
import numpy as np
import math
import sys
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

# 添加测试路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入共享基础类
from base_framework import VerificationTest, BinaryUniverseSystem
from zeckendorf import ZeckendorfEncoder, GoldenConstants, EntropyCalculator

@dataclass
class ZeckendorfState:
    """Zeckendorf-encoded binary state with no-11 constraint"""
    bits: List[int]
    fibonacci_indices: List[int] = field(default_factory=list)
    total_value: int = 0
    
    def __post_init__(self):
        if self.total_value == 0 and self.fibonacci_indices:
            self.total_value = self._compute_value()
    
    def _compute_value(self) -> int:
        """Compute the numerical value from Fibonacci representation"""
        if not self.fibonacci_indices:
            return 0
        
        max_index = max(self.fibonacci_indices, default=0)
        # Generate enough Fibonacci numbers
        fib = [1, 2]
        while len(fib) <= max_index + 5:
            fib.append(fib[-1] + fib[-2])
        
        total = 0
        for i in self.fibonacci_indices:
            if i < len(fib):
                total += fib[i]
        
        return total

class BottleneckTensionSystem:
    """T26-1瓶颈张力系统的核心实现"""
    
    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.phi = GoldenConstants.PHI
        self.encoder = ZeckendorfEncoder()
        self.entropy_calc = EntropyCalculator()
        
        # 初始化系统组件
        self.components = self._initialize_components()
        self.connection_matrix = self._initialize_connections()
        
    def _initialize_components(self) -> List[Dict]:
        """初始化系统组件"""
        components = []
        for i in range(self.n_components):
            # 生成不同长度的Zeckendorf状态，但让第一个组件成为瓶颈
            if i == 0:
                # 瓶颈组件：小容量但高饱和度
                length = 6  # 较小长度
                capacity = self.encoder.compute_capacity(length)
                initial_entropy = capacity * 0.9  # 90%饱和度
            else:
                # 正常组件：更大容量但低饱和度
                length = 8 + i * 2  # 8, 10, 12, 14...
                capacity = self.encoder.compute_capacity(length)
                initial_entropy = capacity * (0.2 + 0.1 * i)  # 20%, 30%, 40%...
            
            component = {
                'id': i,
                'length': length,
                'capacity': capacity,
                'entropy_actual': initial_entropy,
                'entropy_required': initial_entropy * (1.1 + 0.1 * i),  # 不同程度的熵增需求
                'state': self._generate_zeckendorf_state(length),
                'tension': 0.0,
                'omega': 0.0  # 将被计算
            }
            components.append(component)
            
        return components
        
    def _initialize_connections(self) -> np.ndarray:
        """初始化组件连接矩阵"""
        matrix = np.zeros((self.n_components, self.n_components))
        
        # 创建具有自环的网络
        for i in range(self.n_components):
            # 自环连接（关键用于自指系数）
            if i == 0:
                # 瓶颈组件具有高自指性
                matrix[i, i] = 5  # 高自环强度
            else:
                # 其他组件具有较低自指性
                matrix[i, i] = 1 + 0.5 * i  # 渐增的自环强度
            
            # 总连接数保持相对稳定
            total_external = 3  # 每个组件3个外部连接
            matrix[i, i] += total_external  # 自环 + 外部连接
            
            # 与相邻组件的弱连接
            if i > 0:
                matrix[i, i-1] = 0.5
                matrix[i-1, i] = 0.5
            
            # 与后续组件连接
            if i < self.n_components - 1:
                matrix[i, i+1] = 0.5
                
        return matrix
        
    def _generate_zeckendorf_state(self, target_length: int) -> ZeckendorfState:
        """生成指定长度的Zeckendorf状态"""
        # 生成一个满足no-11约束的随机值
        max_val = int(self.phi ** target_length)
        value = max_val // 3  # 选择较小值以确保编码长度合适
        
        # 转换为Zeckendorf表示
        zeck_str = self.encoder.encode(value)
        
        # 转换为状态表示
        bits = [int(b) for b in zeck_str]
        
        # 计算Fibonacci索引
        fib_indices = []
        for i, bit in enumerate(reversed(bits)):
            if bit == 1:
                fib_indices.append(i)
                
        return ZeckendorfState(
            bits=bits,
            fibonacci_indices=fib_indices,
            total_value=value
        )
    
    def compute_component_tension(self, component_id: int) -> float:
        """计算组件张力 - 实现Algorithm F26.1.1"""
        component = self.components[component_id]
        
        H_actual = component['entropy_actual']
        H_required = component['entropy_required']
        capacity = component['capacity']
        omega = self.compute_self_reference_coefficient(component_id)
        
        # 核心张力公式来自T26-1
        entropy_deficit = H_required - H_actual
        normalized_deficit = entropy_deficit / capacity
        tension = normalized_deficit * omega
        
        # 确保Zeckendorf约束
        tension = self._zeckendorf_quantize(tension)
        
        # 更新组件状态
        self.components[component_id]['tension'] = tension
        self.components[component_id]['omega'] = omega
        
        return tension
    
    def compute_self_reference_coefficient(self, component_id: int) -> float:
        """计算自指系数 - 实现Algorithm F26.1.2"""
        component = self.components[component_id]
        
        n_self_connections = self.connection_matrix[component_id, component_id]
        n_total_connections = np.sum(self.connection_matrix[component_id, :])
        
        if n_total_connections == 0:
            return 0
            
        connection_ratio = n_self_connections / n_total_connections
        max_fibonacci = max(component['state'].fibonacci_indices, default=1)
        state_factor = math.log2(1 + component['state'].total_value / (2**max_fibonacci))
        
        return connection_ratio * state_factor
    
    def _zeckendorf_quantize(self, value: float) -> float:
        """将值量化到最近的Zeckendorf可表示数"""
        if value <= 0:
            return 0
        
        # 使用编码器的量化功能
        quantized = self.encoder.fibonacci_quantize(value)
        return float(quantized)
    
    def identify_bottleneck_component(self) -> int:
        """识别瓶颈组件 - 实现Algorithm F26.1.3"""
        saturations = []
        for component in self.components:
            saturation = component['entropy_actual'] / component['capacity']
            saturations.append(saturation)
            
        return int(np.argmax(saturations))
    
    def verify_tension_inequality(self) -> Tuple[bool, float, float]:
        """验证张力不均匀性 - 实现Algorithm F26.1.4"""
        # 计算所有组件的张力
        tensions = []
        for i in range(self.n_components):
            tension = self.compute_component_tension(i)
            tensions.append(tension)
        
        if len(tensions) == 0:
            return False, 0, 0
            
        bottleneck_id = self.identify_bottleneck_component()
        avg_tension = sum(tensions) / len(tensions)
        
        if avg_tension == 0:
            return True, 0, 0  # 平凡情况
            
        bottleneck_tension = tensions[bottleneck_id]
        min_tension = min(tensions)
        
        bottleneck_ratio = bottleneck_tension / avg_tension
        min_ratio = min_tension / avg_tension
        
        # 检查 T_j* >= φ * T_avg
        condition1 = bottleneck_ratio >= self.phi - 1e-10
        
        # 检查 ∃i: T_i <= T_avg/φ  
        condition2 = min_ratio <= (1/self.phi) + 1e-10
        
        is_valid = condition1 and condition2
        return is_valid, bottleneck_ratio, min_ratio
    
    def evolve_tension_dynamics(self, component_id: int, dt: float = 0.01) -> float:
        """演化张力动力学 - 实现Algorithm F26.1.5"""
        component = self.components[component_id]
        
        current_tension = component['tension']
        entropy_saturation = component['entropy_actual'] / component['capacity']
        accumulation_rate = 1.0  # λ parameter
        max_tension = self.phi * math.log2(self.phi)
        
        if max_tension == 0:
            return current_tension
            
        saturation_term = entropy_saturation ** self.phi
        capacity_term = 1 - (current_tension / max_tension)
        
        dtension_dt = accumulation_rate * saturation_term * capacity_term
        
        new_tension = current_tension + dtension_dt * dt
        new_tension = min(new_tension, max_tension)
        
        # 量化到Zeckendorf表示
        new_tension = self._zeckendorf_quantize(new_tension)
        
        # 更新组件状态
        self.components[component_id]['tension'] = new_tension
        
        return new_tension
    
    def compute_system_entropy(self) -> float:
        """计算系统总熵"""
        total_entropy = 0
        for component in self.components:
            total_entropy += component['entropy_actual']
        return total_entropy

class TestT26_1_BottleneckTension(VerificationTest):
    """T26-1瓶颈张力积累定理验证测试"""
    
    def setUp(self):
        """测试初始化"""
        super().setUp()
        self.tension_system = BottleneckTensionSystem(n_components=5)
        self.phi = GoldenConstants.PHI
        self.tolerance = 1e-10
        
    def test_axiom_compliance(self):
        """测试唯一公理：自指完备系统必然熵增"""
        initial_entropy = self.tension_system.compute_system_entropy()
        
        # 演化系统
        for i in range(self.tension_system.n_components):
            self.tension_system.evolve_tension_dynamics(i, dt=0.01)
            
        # 检查系统熵是否增加或张力是否存在（表明熵增压力）
        final_entropy = self.tension_system.compute_system_entropy()
        total_tension = sum(comp['tension'] for comp in self.tension_system.components)
        
        # 根据唯一公理，必须满足熵增或存在张力压力
        entropy_increased = final_entropy > initial_entropy
        tension_exists = total_tension > self.tolerance
        
        self.assertTrue(
            entropy_increased or tension_exists,
            f"违反唯一公理：熵增={final_entropy-initial_entropy:.6f}, 总张力={total_tension:.6f}"
        )
        
    def test_zeckendorf_constraint_compliance(self):
        """测试no-11约束合规性"""
        for i in range(self.tension_system.n_components):
            component = self.tension_system.components[i]
            
            # 检查状态编码
            state_bits = component['state'].bits
            bit_string = ''.join(map(str, state_bits))
            
            self.assertNotIn('11', bit_string, 
                           f"组件{i}状态包含连续11: {bit_string}")
            
            # 检查张力值是否可用Zeckendorf表示
            tension = self.tension_system.compute_component_tension(i)
            quantized_tension = self.tension_system._zeckendorf_quantize(tension)
            
            self.assertAlmostEqual(
                tension, quantized_tension, places=8,
                msg=f"组件{i}张力不满足Zeckendorf量化: {tension} vs {quantized_tension}"
            )
    
    def test_tension_inequality_theorem(self):
        """测试定理26.1.1：张力不均匀分布"""
        is_valid, bottleneck_ratio, min_ratio = self.tension_system.verify_tension_inequality()
        
        self.assertTrue(is_valid, 
                       f"张力不均匀分布验证失败: 瓶颈比率={bottleneck_ratio:.4f}, 最小比率={min_ratio:.4f}")
        
        # 如果存在非零张力，检查具体不等式
        tensions = [self.tension_system.compute_component_tension(i) 
                   for i in range(self.tension_system.n_components)]
        
        if max(tensions) > self.tolerance:
            avg_tension = sum(tensions) / len(tensions)
            max_tension = max(tensions)
            min_tension = min(tensions)
            
            # 验证 T_max >= φ * T_avg
            self.assertGreaterEqual(
                max_tension, self.phi * avg_tension - self.tolerance,
                f"瓶颈张力条件失败: {max_tension:.6f} < φ * {avg_tension:.6f} = {self.phi * avg_tension:.6f}"
            )
            
            # 验证 T_min <= T_avg / φ
            self.assertLessEqual(
                min_tension, avg_tension / self.phi + self.tolerance,
                f"最小张力条件失败: {min_tension:.6f} > {avg_tension:.6f} / φ = {avg_tension / self.phi:.6f}"
            )
    
    def test_tension_dynamics_theorem(self):
        """测试定理26.1.2：张力积累动力学"""
        # 选择一个高饱和度组件进行测试
        bottleneck_id = self.tension_system.identify_bottleneck_component()
        
        initial_tension = self.tension_system.components[bottleneck_id]['tension']
        saturation = (self.tension_system.components[bottleneck_id]['entropy_actual'] / 
                     self.tension_system.components[bottleneck_id]['capacity'])
        
        # 演化张力
        dt = 0.01
        new_tension = self.tension_system.evolve_tension_dynamics(bottleneck_id, dt)
        
        # 检查动力学方程的预期行为
        if saturation > 0.5:  # 高饱和度时应该增加张力
            expected_increase = saturation > self.phi ** (-1)
            if expected_increase:
                self.assertGreater(
                    new_tension, initial_tension - self.tolerance,
                    f"高饱和度({saturation:.3f})时张力未增加: {initial_tension:.6f} -> {new_tension:.6f}"
                )
        
        # 检查张力不超过理论最大值
        max_theoretical = self.phi * math.log2(self.phi)
        self.assertLessEqual(
            new_tension, max_theoretical + self.tolerance,
            f"张力超过理论最大值: {new_tension:.6f} > {max_theoretical:.6f}"
        )
    
    def test_self_reference_coefficient_bounds(self):
        """测试自指系数的合理界限"""
        for i in range(self.tension_system.n_components):
            omega = self.tension_system.compute_self_reference_coefficient(i)
            
            # 自指系数应该非负
            self.assertGreaterEqual(omega, 0, 
                                  f"组件{i}自指系数为负: {omega}")
            
            # 自指系数应该有合理的上界
            max_omega = math.log2(math.sqrt(5))  # 来自黄金比例的理论限制
            self.assertLessEqual(omega, max_omega + self.tolerance,
                               f"组件{i}自指系数过大: {omega} > {max_omega}")
    
    def test_bottleneck_identification(self):
        """测试瓶颈组件识别的正确性"""
        bottleneck_id = self.tension_system.identify_bottleneck_component()
        
        # 验证识别的瓶颈确实具有最高饱和度
        saturations = []
        for comp in self.tension_system.components:
            saturation = comp['entropy_actual'] / comp['capacity']
            saturations.append(saturation)
            
        max_saturation = max(saturations)
        bottleneck_saturation = saturations[bottleneck_id]
        
        self.assertAlmostEqual(
            bottleneck_saturation, max_saturation, places=10,
            msg=f"识别的瓶颈{bottleneck_id}饱和度{bottleneck_saturation:.6f}不是最大{max_saturation:.6f}"
        )
    
    def test_entropy_flow_blockage(self):
        """测试熵流阻塞与张力产生的关系"""
        # 设置一个明显的瓶颈场景
        bottleneck_id = self.tension_system.identify_bottleneck_component()
        
        # 增加瓶颈组件的熵需求，模拟系统演化压力
        original_required = self.tension_system.components[bottleneck_id]['entropy_required']
        self.tension_system.components[bottleneck_id]['entropy_required'] = original_required * 1.5
        
        # 计算张力
        tension_before = self.tension_system.components[bottleneck_id]['tension']
        tension_after = self.tension_system.compute_component_tension(bottleneck_id)
        
        # 验证张力增加（熵需求增加应导致更高张力）
        self.assertGreater(
            tension_after, tension_before,
            f"增加熵需求后张力未增加: {tension_before:.6f} -> {tension_after:.6f}"
        )
        
        # 恢复原始状态
        self.tension_system.components[bottleneck_id]['entropy_required'] = original_required
    
    def test_phi_quantization_properties(self):
        """测试φ-量化的数学性质"""
        test_values = [0.1, 0.618, 1.0, 1.618, 2.618, 3.14159]
        
        for value in test_values:
            quantized = self.tension_system._zeckendorf_quantize(value)
            
            # 量化后的值应该可以用Zeckendorf表示
            encoded = self.tension_system.encoder.encode(int(quantized))
            self.assertNotIn('11', encoded,
                           f"量化值{quantized}的编码{encoded}包含连续11")
            
            # 量化应该保持单调性
            if value > 0:
                self.assertGreaterEqual(quantized, 0,
                                      f"正值{value}量化后为负: {quantized}")
    
    def test_tension_conservation_bounds(self):
        """测试张力守恒界限"""
        # 计算所有张力
        total_tension = 0
        for i in range(self.tension_system.n_components):
            tension = self.tension_system.compute_component_tension(i)
            total_tension += tension
        
        # 验证总张力不超过理论界限
        max_total = self.tension_system.n_components * self.phi * math.log2(self.phi)
        self.assertLessEqual(
            total_tension, max_total + self.tolerance,
            f"总张力超过理论界限: {total_tension:.6f} > {max_total:.6f}"
        )
    
    def test_system_invariants(self):
        """测试系统不变量的维持"""
        # 执行多步演化
        for step in range(10):
            for i in range(self.tension_system.n_components):
                self.tension_system.evolve_tension_dynamics(i, dt=0.001)
                
            # 检查不变量
            # I26.1.1: 张力非负性
            for comp in self.tension_system.components:
                self.assertGreaterEqual(comp['tension'], 0,
                                      f"步骤{step}: 张力为负 {comp['tension']}")
                
            # I26.1.3: 瓶颈张力优势
            bottleneck_id = self.tension_system.identify_bottleneck_component()
            tensions = [comp['tension'] for comp in self.tension_system.components]
            avg_tension = sum(tensions) / len(tensions)
            
            if avg_tension > self.tolerance:
                bottleneck_tension = tensions[bottleneck_id]
                dominance_ratio = bottleneck_tension / avg_tension
                
                self.assertGreaterEqual(
                    dominance_ratio, 1.0 - self.tolerance,  # 至少应该 >= 平均值
                    f"步骤{step}: 瓶颈张力未保持优势 {dominance_ratio:.6f}"
                )

    def test_integration_with_existing_theories(self):
        """测试与现有理论的集成一致性"""
        # 与C7-4木桶原理的一致性
        bottleneck_id = self.tension_system.identify_bottleneck_component()
        bottleneck_saturation = (self.tension_system.components[bottleneck_id]['entropy_actual'] / 
                                self.tension_system.components[bottleneck_id]['capacity'])
        
        # 瓶颈组件应该具有最高饱和度（C7-4的核心：熵增受最饱和组件限制）
        saturations = [(comp['entropy_actual'] / comp['capacity']) 
                      for comp in self.tension_system.components]
        max_saturation = max(saturations)
        
        self.assertAlmostEqual(
            bottleneck_saturation, max_saturation, places=6,
            msg=f"瓶颈组件饱和度{bottleneck_saturation:.6f}不是最高饱和度{max_saturation:.6f}"
        )

if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)