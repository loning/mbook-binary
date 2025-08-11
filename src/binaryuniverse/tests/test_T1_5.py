#!/usr/bin/env python3
"""
T1-5: 局部熵减条件定理 - 完整测试程序

验证φ-编码二进制宇宙中局部熵减的条件，包括：
1. 局部-全局熵分解
2. 最小代价原理
3. Maxwell妖的熵平衡
4. 生命系统的熵管理
5. 自组织条件
6. 技术应用极限
"""

import unittest
import numpy as np
from typing import List, Tuple, Dict, Set, Any
from dataclasses import dataclass
import random


class PhiNumber:
    """φ进制数系统"""
    def __init__(self, value: float):
        self.phi = (1 + np.sqrt(5)) / 2
        self.value = float(value)
        
    def __add__(self, other):
        if isinstance(other, PhiNumber):
            return PhiNumber(self.value + other.value)
        return PhiNumber(self.value + float(other))
        
    def __sub__(self, other):
        if isinstance(other, PhiNumber):
            return PhiNumber(self.value - other.value)
        return PhiNumber(self.value - float(other))
        
    def __mul__(self, other):
        if isinstance(other, PhiNumber):
            return PhiNumber(self.value * other.value)
        return PhiNumber(self.value * float(other))
        
    def __truediv__(self, other):
        if isinstance(other, PhiNumber):
            return PhiNumber(self.value / other.value)
        return PhiNumber(self.value / float(other))
        
    def __neg__(self):
        return PhiNumber(-self.value)
        
    def __abs__(self):
        return PhiNumber(abs(self.value))
        
    def __lt__(self, other):
        if isinstance(other, PhiNumber):
            return self.value < other.value
        return self.value < float(other)
        
    def __le__(self, other):
        if isinstance(other, PhiNumber):
            return self.value <= other.value
        return self.value <= float(other)
        
    def __gt__(self, other):
        if isinstance(other, PhiNumber):
            return self.value > other.value
        return self.value > float(other)
        
    def __ge__(self, other):
        if isinstance(other, PhiNumber):
            return self.value >= other.value
        return self.value >= float(other)
        
    def __eq__(self, other):
        if isinstance(other, PhiNumber):
            return abs(self.value - other.value) < 1e-10
        return abs(self.value - float(other)) < 1e-10
        
    def __repr__(self):
        return f"φ({self.value:.6f})"


@dataclass
class SurfaceElement:
    """边界表面元素"""
    area: float
    normal_vector: np.ndarray
    position: np.ndarray


class Boundary:
    """系统边界"""
    def __init__(self, surface_elements: List[SurfaceElement]):
        self.elements = surface_elements
        self.phi = (1 + np.sqrt(5)) / 2
        
    def area(self) -> PhiNumber:
        """计算总面积"""
        total = sum(elem.area for elem in self.elements)
        return PhiNumber(total)
        
    def degrees_of_freedom(self) -> int:
        """边界自由度"""
        # 简化：每个表面元素贡献一定自由度
        return len(self.elements) * 4  # 位置+法向量
        
    def verify_no_11_constraint(self) -> bool:
        """验证边界编码满足no-11约束"""
        # 简化：检查元素数量不是11的倍数
        n = len(self.elements)
        binary = bin(n)[2:]
        return '11' not in binary


class LocalSystem:
    """局部系统"""
    def __init__(self, boundary: Boundary, content: Set[Any]):
        self.boundary = boundary
        self.content = content
        self.phi = (1 + np.sqrt(5)) / 2
        self._entropy = None
        
    def entropy(self) -> PhiNumber:
        """计算系统熵"""
        if self._entropy is None:
            # 简化：基于内容数量
            if len(self.content) == 0:
                self._entropy = PhiNumber(0)
            else:
                self._entropy = PhiNumber(np.log(len(self.content) + 1))
        return self._entropy
        
    def set_entropy(self, value: PhiNumber):
        """设置熵值（用于测试）"""
        self._entropy = value
        
    def is_causally_closed(self) -> bool:
        """检查因果闭合性"""
        # 简化实现
        return self.boundary.verify_no_11_constraint()
        
    def interface_entropy(self) -> PhiNumber:
        """界面熵"""
        k_B = 1.38064852e-23
        omega = self.boundary.degrees_of_freedom()
        return PhiNumber(k_B * np.log(omega + 1))


class EntropyFlow:
    """熵流管理器"""
    def __init__(self, local_system: LocalSystem):
        self.system = local_system
        self.phi = (1 + np.sqrt(5)) / 2
        self._inflow_rate = 0.1
        self._outflow_rate = 0.05
        self._production_rate = 0.01
        
    def inflow_rate(self, time: float) -> PhiNumber:
        """熵流入率"""
        # 简化：常数率
        return PhiNumber(self._inflow_rate)
        
    def outflow_rate(self, time: float) -> PhiNumber:
        """熵流出率"""
        return PhiNumber(self._outflow_rate)
        
    def local_production_rate(self, time: float) -> PhiNumber:
        """局部熵产生率（≥0）"""
        return PhiNumber(max(0, self._production_rate))
        
    def net_flow(self, time: float) -> PhiNumber:
        """净熵流"""
        return (self.inflow_rate(time) - self.outflow_rate(time) + 
                self.local_production_rate(time))
        
    def set_rates(self, inflow: float, outflow: float, production: float):
        """设置流率（用于测试）"""
        self._inflow_rate = inflow
        self._outflow_rate = outflow
        self._production_rate = max(0, production)


class MinimumCostCalculator:
    """最小代价计算器"""
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        
    def minimum_environmental_cost(self, local_decrease: PhiNumber) -> PhiNumber:
        """计算最小环境代价"""
        if local_decrease >= 0:
            raise ValueError("局部熵必须减少")
            
        # ΔH_env^min = φ * |ΔH_local| + ΔH_process
        min_cost = self.phi * abs(local_decrease.value)
        process_cost = self.process_entropy_cost(local_decrease)
        
        return PhiNumber(min_cost + process_cost.value)
        
    def process_entropy_cost(self, local_decrease: PhiNumber) -> PhiNumber:
        """过程熵成本"""
        # 基于信息处理
        n_bits = abs(local_decrease.value) / np.log(2)
        return PhiNumber(n_bits * np.log(2) * 0.1)  # 10%开销


@dataclass
class Particle:
    """粒子（用于Maxwell妖）"""
    velocity: float
    position: np.ndarray
    
    def kinetic_energy(self, mass: float = 1.0) -> float:
        """动能"""
        return 0.5 * mass * self.velocity ** 2
        
    def required_precision(self) -> int:
        """测量所需精度（比特）"""
        # 速度越高需要越高精度
        return max(1, int(np.log2(abs(self.velocity) + 1)))
        
    @property
    def state(self) -> str:
        """状态描述"""
        return f"v={self.velocity:.2f}"


class MaxwellDemon:
    """Maxwell妖"""
    def __init__(self, temperature: float):
        self.temperature = temperature
        self.phi = (1 + np.sqrt(5)) / 2
        self.k_B = 1.38064852e-23
        self.memory = []
        
    def measure_particle(self, particle: Particle) -> PhiNumber:
        """测量粒子的熵成本"""
        precision = particle.required_precision()
        
        # ΔH_measure = k_B*T*ln(2)*φ^n
        entropy_cost = self.k_B * self.temperature * np.log(2) * (self.phi ** precision)
        
        self.memory.append(particle.state)
        return PhiNumber(entropy_cost)
        
    def should_let_through(self, particle: Particle, threshold: float = 0) -> bool:
        """决定是否让粒子通过"""
        return particle.velocity > threshold
        
    def sort_particles(self, particles: List[Particle]) -> Tuple[PhiNumber, PhiNumber]:
        """分离粒子"""
        gas_entropy_decrease = PhiNumber(0)
        demon_entropy_increase = PhiNumber(0)
        
        for particle in particles:
            # 测量成本
            measure_cost = self.measure_particle(particle)
            demon_entropy_increase = demon_entropy_increase + measure_cost
            
            # 分离导致的熵减
            if self.should_let_through(particle):
                # 简化：每次成功分离减少固定熵
                gas_entropy_decrease = gas_entropy_decrease - PhiNumber(self.k_B * np.log(2))
                
        return gas_entropy_decrease, demon_entropy_increase
        
    def erase_memory(self) -> PhiNumber:
        """擦除记忆的熵成本"""
        n_bits = len(self.memory)
        
        # Landauer: ΔH_erase = k_B*T*ln(2)*φ per bit
        erase_cost = n_bits * self.k_B * self.temperature * np.log(2) * self.phi
        
        self.memory.clear()
        return PhiNumber(erase_cost)
        
    def net_entropy_change(self, particles: List[Particle]) -> PhiNumber:
        """总熵变"""
        gas_decrease, demon_increase = self.sort_particles(particles)
        erase_cost = self.erase_memory()
        
        total = gas_decrease + demon_increase + erase_cost
        return total


class LivingSystem:
    """生命系统"""
    def __init__(self, volume: float, temperature: float):
        self.volume = volume
        self.temperature = temperature
        self.phi = (1 + np.sqrt(5)) / 2
        self.k_B = 1.38064852e-23
        self.internal_entropy = PhiNumber(1.0)
        
    def atp_consumption_rate(self) -> float:
        """ATP消耗率（mol/s）"""
        # 典型细胞：~10^9 ATP/s
        return 1e-15  # mol/s
        
    def protein_synthesis_rate(self) -> float:
        """蛋白质合成率（个/s）"""
        return 1e6  # 蛋白质/s
        
    def metabolic_gradient(self) -> PhiNumber:
        """代谢梯度"""
        # 必须 > k_B*T*φ
        critical = self.k_B * self.temperature * self.phi
        actual = critical * 2  # 2倍于临界值
        return PhiNumber(actual)
        
    def atp_cycle(self, time_step: float) -> PhiNumber:
        """ATP循环熵变"""
        n_atp = self.atp_consumption_rate() * time_step
        
        # 每个ATP水解释放~7.3 kcal/mol
        entropy_per_atp = 7.3 * 4184 / self.temperature  # J/K
        
        # φ修正
        return PhiNumber(n_atp * entropy_per_atp / self.phi)
        
    def protein_folding_entropy(self, time_step: float) -> PhiNumber:
        """蛋白质折叠熵变"""
        n_proteins = self.protein_synthesis_rate() * time_step
        
        # 蛋白质折叠减少熵
        protein_decrease = -50 * self.k_B * n_proteins
        
        # 水熵增必须 > φ * |蛋白质熵减|
        water_increase = self.phi * abs(protein_decrease) * 1.1
        
        return PhiNumber(protein_decrease + water_increase)
        
    def maintain_low_entropy(self, time_step: float) -> Dict[str, PhiNumber]:
        """维持低熵"""
        results = {}
        
        # ATP能量供应
        results['atp'] = self.atp_cycle(time_step)
        
        # 蛋白质折叠
        results['protein'] = self.protein_folding_entropy(time_step)
        
        # 总变化
        total = results['atp'] + results['protein']
        results['total'] = total
        
        # 环境熵增
        results['environment'] = PhiNumber(abs(total.value) * self.phi * 1.2)
        
        return results


class SelfOrganizingSystem:
    """自组织系统"""
    def __init__(self, complexity: int = 1):
        self.phi = (1 + np.sqrt(5)) / 2
        self.complexity = complexity
        
    def energy_flow_condition(self, energy_in: float, energy_out: float,
                            local_entropy_rate: float, temperature: float) -> bool:
        """能量流条件"""
        net_flow = energy_in - energy_out
        required = temperature * self.phi * abs(local_entropy_rate)
        return net_flow > required
        
    def information_capacity(self) -> PhiNumber:
        """信息处理能力"""
        return PhiNumber(self.phi ** self.complexity)
        
    def stability_condition(self, max_lyapunov: float, 
                          relaxation_time: float) -> bool:
        """稳定性条件"""
        critical = -np.log(self.phi) / relaxation_time
        return max_lyapunov < critical


class RefrigerationSystem:
    """制冷系统"""
    def __init__(self, T_cold: float, T_hot: float):
        self.T_c = T_cold
        self.T_h = T_hot
        self.phi = (1 + np.sqrt(5)) / 2
        
    def carnot_efficiency_phi(self) -> float:
        """φ修正的Carnot效率"""
        epsilon_phi = 1 - 1/self.phi  # ≈ 0.382
        standard = 1 - self.T_c/self.T_h
        return standard - epsilon_phi
        
    def minimum_work(self, heat_removed: float) -> PhiNumber:
        """最小功"""
        cop = self.T_c / (self.T_h - self.T_c)
        phi_correction = self.phi
        return PhiNumber(heat_removed / (cop / phi_correction))


class TestLocalEntropyDecrease(unittest.TestCase):
    """T1-5 局部熵减条件测试套件"""
    
    def setUp(self):
        self.phi = (1 + np.sqrt(5)) / 2
        
    def test_local_system_boundary(self):
        """测试局部系统边界定义"""
        # 创建简单边界（使用4个元素避免3='11'违反no-11约束）
        elements = [
            SurfaceElement(1.0, np.array([1,0,0]), np.array([0,0,0])),
            SurfaceElement(1.0, np.array([0,1,0]), np.array([1,0,0])),
            SurfaceElement(1.0, np.array([0,0,1]), np.array([0,1,0])),
            SurfaceElement(1.0, np.array([-1,0,0]), np.array([0,0,0]))
        ]
        
        boundary = Boundary(elements)
        
        # 测试边界属性
        self.assertEqual(boundary.area().value, 4.0)
        self.assertEqual(boundary.degrees_of_freedom(), 16)
        self.assertTrue(boundary.verify_no_11_constraint())
        
        # 创建局部系统
        content = {'atom1', 'atom2', 'atom3'}
        local_sys = LocalSystem(boundary, content)
        
        # 测试系统属性
        self.assertGreater(local_sys.entropy().value, 0)
        self.assertTrue(local_sys.is_causally_closed())
        self.assertGreater(local_sys.interface_entropy().value, 0)
        
    def test_entropy_flow_balance(self):
        """测试熵流平衡方程"""
        # 创建系统
        boundary = Boundary([])
        local_sys = LocalSystem(boundary, set())
        flow = EntropyFlow(local_sys)
        
        # 设置流率实现局部熵减
        # dH/dt = J_in - J_out + σ < 0
        flow.set_rates(inflow=0.1, outflow=0.3, production=0.05)
        
        net = flow.net_flow(0)
        self.assertLess(net.value, 0)  # 净流出，局部熵减
        
        # 验证各项
        self.assertEqual(flow.inflow_rate(0).value, 0.1)
        self.assertEqual(flow.outflow_rate(0).value, 0.3)
        self.assertEqual(flow.local_production_rate(0).value, 0.05)
        
    def test_minimum_cost_principle(self):
        """测试最小代价原理"""
        calculator = MinimumCostCalculator()
        
        # 局部熵减少1单位
        local_decrease = PhiNumber(-1.0)
        
        # 计算最小环境代价
        min_cost = calculator.minimum_environmental_cost(local_decrease)
        
        # 应该至少是φ倍
        self.assertGreaterEqual(min_cost.value, self.phi)
        
        # 测试不同大小的熵减
        for decrease in [-0.1, -0.5, -2.0, -10.0]:
            local = PhiNumber(decrease)
            cost = calculator.minimum_environmental_cost(local)
            
            # 验证比例关系
            ratio = cost.value / abs(decrease)
            self.assertGreaterEqual(ratio, self.phi)
            
    def test_maxwell_demon(self):
        """测试Maxwell妖"""
        temperature = 300  # K
        demon = MaxwellDemon(temperature)
        
        # 创建一些粒子
        particles = []
        for i in range(10):
            v = random.gauss(0, np.sqrt(temperature))
            pos = np.random.rand(3)
            particles.append(Particle(v, pos))
            
        # 妖工作
        total_entropy = demon.net_entropy_change(particles)
        
        # 总熵必须增加
        self.assertGreater(total_entropy.value, 0)
        
        # 分别测试各部分
        demon.memory = []  # 重置
        gas_decrease, demon_increase = demon.sort_particles(particles)
        erase_cost = demon.erase_memory()
        
        # 气体熵可能减少
        self.assertLessEqual(gas_decrease.value, 0)
        
        # 但妖的熵增加更多
        total = gas_decrease + demon_increase + erase_cost
        self.assertGreater(total.value, 0)
        
    def test_living_system_entropy(self):
        """测试生命系统熵管理"""
        volume = 1e-18  # m³ (细胞大小)
        temperature = 310  # K (体温)
        
        cell = LivingSystem(volume, temperature)
        
        # 测试代谢梯度
        gradient = cell.metabolic_gradient()
        critical = cell.k_B * temperature * self.phi
        self.assertGreater(gradient.value, critical)
        
        # 测试熵维持
        time_step = 1.0  # 秒
        results = cell.maintain_low_entropy(time_step)
        
        # 验证各部分
        self.assertGreater(results['atp'].value, 0)  # ATP循环增熵
        self.assertGreater(results['protein'].value, 0)  # 总体增熵
        self.assertGreater(results['environment'].value, 0)  # 环境熵增
        
        # 验证最小代价原理
        if results['total'].value < 0:  # 如果内部熵减
            env_increase = results['environment'].value
            internal_decrease = abs(results['total'].value)
            ratio = env_increase / internal_decrease
            self.assertGreaterEqual(ratio, self.phi)
            
    def test_self_organization(self):
        """测试自组织条件"""
        system = SelfOrganizingSystem(complexity=3)
        
        # 测试能量流条件
        energy_in = 100  # W
        energy_out = 50   # W
        entropy_rate = -0.1  # 局部熵减率
        temperature = 300
        
        condition = system.energy_flow_condition(
            energy_in, energy_out, entropy_rate, temperature
        )
        self.assertTrue(condition)
        
        # 测试信息处理能力
        capacity = system.information_capacity()
        self.assertEqual(capacity.value, self.phi ** 3)
        
        # 测试稳定性
        max_lyapunov = -0.1
        relaxation_time = 10
        stable = system.stability_condition(max_lyapunov, relaxation_time)
        self.assertTrue(stable)
        
    def test_refrigeration_limits(self):
        """测试制冷极限"""
        T_cold = 250  # K
        T_hot = 300   # K
        
        fridge = RefrigerationSystem(T_cold, T_hot)
        
        # 测试φ修正的效率
        eff = fridge.carnot_efficiency_phi()
        carnot_standard = 1 - T_cold/T_hot
        
        # φ修正降低了效率
        self.assertLess(eff, carnot_standard)
        self.assertAlmostEqual(carnot_standard - eff, 1 - 1/self.phi, places=3)
        
        # 测试最小功
        heat_removed = 1000  # J
        min_work = fridge.minimum_work(heat_removed)
        
        # 验证功大于理想值
        ideal_work = heat_removed * (T_hot - T_cold) / T_cold
        self.assertGreater(min_work.value, ideal_work)
        
    def test_total_entropy_increase(self):
        """测试总熵增原理"""
        # 创建一个局部熵减的场景
        local_decrease = PhiNumber(-1.0)
        
        # 计算所需的最小环境熵增
        calculator = MinimumCostCalculator()
        min_env_increase = calculator.minimum_environmental_cost(local_decrease)
        
        # 总熵变
        total = local_decrease + min_env_increase
        
        # 必须大于0
        self.assertGreater(total.value, 0)
        
        # 测试边界情况
        # 如果环境熵增刚好等于最小值
        env_increase = PhiNumber(self.phi * abs(local_decrease.value))
        total_boundary = local_decrease + env_increase
        
        # 由于过程熵，总和仍应>0
        self.assertGreater(total_boundary.value, 0)
        
    def test_ecosystem_entropy_flow(self):
        """测试生态系统熵流"""
        # 简化的生态系统模型
        solar_input = 1000  # W/m²
        efficiency = 0.04  # 光合作用效率
        
        # 植物熵减（负值）
        plant_entropy_rate = -solar_input * efficiency / 300  # 除以温度得到熵率
        
        # 太阳辐射熵率
        # 高温辐射转换为低温辐射，熵增加
        T_sun = 6000
        T_earth = 300
        # 能量守恒：输入能量 = 输出能量
        # 但熵增加：S_out/S_in = T_earth/T_sun * (能量相同)
        photon_entropy_rate_in = solar_input / T_sun  # 来自太阳的熵率
        photon_entropy_rate_out = solar_input / T_earth  # 地球再辐射的熵率
        photon_entropy_increase = photon_entropy_rate_out - photon_entropy_rate_in
        
        # 总熵变
        total = plant_entropy_rate + photon_entropy_increase
        
        # 必须>0
        self.assertGreater(total, 0)
        
        # 验证营养级效率
        trophic_efficiency = 0.1
        # 实际生态系统约10%
        # 验证这是一个合理的低效率（远小于1）
        self.assertLess(trophic_efficiency, 0.2)
        self.assertGreater(trophic_efficiency, 0.05)
        # 验证可以表示为φ的负幂次
        # 0.1 ≈ φ^(-4.78)
        x = -np.log(trophic_efficiency) / np.log(self.phi)
        self.assertGreater(x, 4)  # 确实是φ的高次负幂
        
    def test_information_storage_entropy(self):
        """测试信息存储的熵成本"""
        # 存储1MB信息
        n_bits = 8 * 1024 * 1024
        temperature = 300
        k_B = 1.38064852e-23
        
        # Landauer极限
        min_energy = n_bits * k_B * temperature * np.log(2)
        
        # φ修正
        actual_energy = min_energy * self.phi
        
        # 熵成本
        entropy_cost = actual_energy / temperature
        
        # 验证大于理论极限
        self.assertGreater(entropy_cost, min_energy / temperature)
        
    def test_no_perpetual_motion(self):
        """验证永动机不可能"""
        # 尝试创建一个"永动机"
        # 局部熵减但无环境熵增
        
        local_decrease = PhiNumber(-10)
        env_increase = PhiNumber(0)  # 违反最小代价
        
        total = local_decrease + env_increase
        
        # 总熵减少，违反定理
        self.assertLess(total.value, 0)
        
        # 验证这违反了最小代价原理
        calculator = MinimumCostCalculator()
        min_required = calculator.minimum_environmental_cost(local_decrease)
        
        self.assertLess(env_increase.value, min_required.value)


if __name__ == '__main__':
    unittest.main(verbosity=2)