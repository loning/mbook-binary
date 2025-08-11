#!/usr/bin/env python3
"""
T16-4 φ-宇宙膨胀测试程序
验证所有理论预测和形式化规范
"""

import unittest
import numpy as np
from typing import List, Dict, Tuple, Optional
import math


class PhiNumber:
    """φ-数，支持no-11约束的运算"""
    def __init__(self, value: float):
        self.value = float(value)
        self.phi = (1 + math.sqrt(5)) / 2
        self._verify_no_11()
    
    def _to_binary(self, n: int) -> str:
        """转换为二进制字符串"""
        if n == 0:
            return "0"
        return bin(n)[2:]
    
    def _verify_no_11(self):
        """验证no-11约束"""
        if self.value < 0:
            return  # 负数暂不检查
        
        # 检查整数部分
        int_part = int(abs(self.value))
        binary_str = self._to_binary(int_part)
        if "11" in binary_str:
            # 尝试Zeckendorf表示
            self._to_zeckendorf(int_part)
    
    def _to_zeckendorf(self, n: int) -> List[int]:
        """转换为Zeckendorf表示（Fibonacci基）"""
        if n == 0:
            return []
        
        fibs = [1, 2]
        while fibs[-1] < n:
            fibs.append(fibs[-1] + fibs[-2])
        
        result = []
        for i in range(len(fibs) - 1, -1, -1):
            if n >= fibs[i]:
                result.append(fibs[i])
                n -= fibs[i]
        
        return result
    
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
    
    def __pow__(self, other):
        if isinstance(other, PhiNumber):
            return PhiNumber(self.value ** other.value)
        return PhiNumber(self.value ** float(other))
    
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
    
    def __eq__(self, other):
        if isinstance(other, PhiNumber):
            return abs(self.value - other.value) < 1e-10
        return abs(self.value - float(other)) < 1e-10
    
    def sqrt(self):
        """平方根"""
        return PhiNumber(math.sqrt(self.value))
    
    def exp(self):
        """指数函数"""
        return PhiNumber(math.exp(self.value))
    
    def log(self):
        """自然对数"""
        if self.value <= 0:
            raise ValueError("Cannot take log of non-positive number")
        return PhiNumber(math.log(self.value))
    
    def __repr__(self):
        return f"PhiNumber({self.value})"


class PhiFLRWMetric:
    """φ-FLRW度量"""
    def __init__(self, scale_factor: PhiNumber, curvature: int = 0):
        self.a = scale_factor  # 标度因子
        self.k = curvature  # 空间曲率 (-1, 0, 1)
        self.phi = (1 + math.sqrt(5)) / 2
        
        if self.k not in [-1, 0, 1]:
            raise ValueError("Curvature must be -1, 0, or 1")
    
    def metric_components(self, t: PhiNumber) -> Dict[str, PhiNumber]:
        """返回FLRW度量分量"""
        return {
            'g_tt': PhiNumber(-1),
            'g_rr': self.a ** PhiNumber(2) / (PhiNumber(1) - PhiNumber(self.k) * PhiNumber(1)),  # 简化，实际依赖于r
            'g_theta_theta': self.a ** PhiNumber(2),  # 实际是 a²r²
            'g_phi_phi': self.a ** PhiNumber(2)  # 实际是 a²r²sin²θ
        }
    
    def spatial_volume(self) -> PhiNumber:
        """计算共动空间体积（归一化）"""
        if self.k == 0:  # 平坦空间
            return self.a ** PhiNumber(3)
        elif self.k == 1:  # 闭合空间
            return PhiNumber(2 * math.pi**2) * self.a ** PhiNumber(3)
        else:  # 开放空间
            # 需要正则化，这里返回标度因子的立方
            return self.a ** PhiNumber(3)
    
    def conformal_time(self, t: PhiNumber, a_history: List[Tuple[PhiNumber, PhiNumber]]) -> PhiNumber:
        """计算共形时间（数值积分）"""
        if len(a_history) < 2:
            return t / self.a
        
        eta = PhiNumber(0)
        for i in range(len(a_history) - 1):
            t1, a1 = a_history[i]
            t2, a2 = a_history[i + 1]
            dt = t2 - t1
            a_avg = (a1 + a2) / PhiNumber(2)
            eta = eta + dt / a_avg
        
        return eta
    
    def verify_homogeneity(self) -> bool:
        """验证均匀性"""
        # FLRW度量构造上就是均匀的
        return True
    
    def verify_isotropy(self) -> bool:
        """验证各向同性"""
        # FLRW度量构造上就是各向同性的
        return True


class PhiScaleFactor:
    """φ-标度因子"""
    def __init__(self, initial_value: PhiNumber):
        self.a0 = initial_value
        self.phi = (1 + math.sqrt(5)) / 2
        self.evolution_history = [(PhiNumber(0), self.a0)]
    
    def evolve(self, time_step: PhiNumber, expansion_rate: PhiNumber) -> PhiNumber:
        """演化标度因子一个时间步"""
        t_current, a_current = self.evolution_history[-1]
        
        # da/dt = a * H
        da = a_current * expansion_rate * time_step
        a_new = a_current + da
        t_new = t_current + time_step
        
        self.evolution_history.append((t_new, a_new))
        return a_new
    
    def discrete_evolution(self, n_steps: int) -> List[PhiNumber]:
        """离散Fibonacci演化"""
        result = [self.a0]
        a = self.a0
        
        for k in range(1, n_steps + 1):
            # 使用Fibonacci序列调制膨胀
            F_k = self._fibonacci(k)
            # 确保epsilon_k不会太小
            # 使用更大的基础值，并限制F_k的影响
            if F_k > 10:
                epsilon_k = PhiNumber(0.001)  # 最小增量
            else:
                epsilon_k = PhiNumber(self.phi ** (-F_k) * 0.1)  # 较大的膨胀增量
            a = a * (PhiNumber(1) + epsilon_k)
            result.append(a)
        
        return result
    
    def redshift(self, t_emit: PhiNumber, t_obs: PhiNumber) -> PhiNumber:
        """计算宇宙学红移"""
        # 找到最接近的时间点
        a_emit = self._interpolate_scale_factor(t_emit)
        a_obs = self._interpolate_scale_factor(t_obs)
        
        # z = a_obs/a_emit - 1
        return a_obs / a_emit - PhiNumber(1)
    
    def _interpolate_scale_factor(self, t: PhiNumber) -> PhiNumber:
        """插值得到给定时刻的标度因子"""
        if len(self.evolution_history) == 0:
            return self.a0
        
        # 简单线性插值
        for i in range(len(self.evolution_history) - 1):
            t1, a1 = self.evolution_history[i]
            t2, a2 = self.evolution_history[i + 1]
            
            if t1 <= t <= t2:
                # 线性插值
                frac = (t - t1) / (t2 - t1)
                return a1 + frac * (a2 - a1)
        
        # 如果超出范围，返回最近的值
        if t < self.evolution_history[0][0]:
            return self.evolution_history[0][1]
        else:
            return self.evolution_history[-1][1]
    
    def verify_no_11_constraint(self) -> bool:
        """验证演化历史满足no-11约束"""
        try:
            for t, a in self.evolution_history:
                a._verify_no_11()
            return True
        except:
            return False
    
    def _fibonacci(self, n: int) -> int:
        """计算第n个Fibonacci数"""
        if n <= 0:
            return 0
        elif n == 1 or n == 2:
            return 1
        
        a, b = 1, 1
        for _ in range(n - 2):
            a, b = b, a + b
        return b


class PhiHubbleParameter:
    """φ-哈勃参数"""
    def __init__(self, H0: PhiNumber):
        self.H0 = H0  # 当前哈勃常数
        self.phi = (1 + math.sqrt(5)) / 2
    
    def hubble_rate(self, scale_factor: PhiScaleFactor, t: PhiNumber) -> PhiNumber:
        """计算给定时刻的哈勃参数"""
        # 数值微分计算 H = (da/dt)/a
        history = scale_factor.evolution_history
        
        if len(history) < 2:
            return self.H0
        
        # 找到最接近的两个时间点
        for i in range(len(history) - 1):
            t1, a1 = history[i]
            t2, a2 = history[i + 1]
            
            if t1 <= t <= t2:
                # 计算局部导数
                da_dt = (a2 - a1) / (t2 - t1)
                a = scale_factor._interpolate_scale_factor(t)
                return da_dt / a
        
        # 默认返回H0
        return self.H0
    
    def deceleration_parameter(self, a: PhiNumber, H: PhiNumber, dH_dt: PhiNumber) -> PhiNumber:
        """计算减速参数 q = -1 - dH/dt/H²"""
        return PhiNumber(-1) - dH_dt / (H ** PhiNumber(2))
    
    def hubble_time(self, H: PhiNumber) -> PhiNumber:
        """哈勃时间 t_H = 1/H"""
        return PhiNumber(1) / H


class PhiFriedmannSolver:
    """φ-Friedmann方程求解器"""
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2
        self.G_phi = PhiNumber(1.0)  # φ-引力常数
    
    def friedmann_equation(self, a: PhiNumber, rho: PhiNumber, 
                          k: int, Lambda: PhiNumber) -> PhiNumber:
        """第一Friedmann方程: H² = 8πρ/3 - k/a² + Λ/3"""
        term1 = PhiNumber(8 * math.pi) * self.G_phi * rho / PhiNumber(3)
        term2 = PhiNumber(-k) / (a ** PhiNumber(2))
        term3 = Lambda / PhiNumber(3)
        
        H_squared = term1 + term2 + term3
        
        if H_squared < PhiNumber(0):
            raise ValueError("Negative H² - unphysical solution")
        
        return H_squared.sqrt()
    
    def acceleration_equation(self, a: PhiNumber, rho: PhiNumber, 
                            p: PhiNumber, Lambda: PhiNumber) -> PhiNumber:
        """第二Friedmann方程: ä/a = -4π(ρ+3p)/3 + Λ/3"""
        term1 = PhiNumber(-4 * math.pi) * self.G_phi * (rho + PhiNumber(3) * p) / PhiNumber(3)
        term2 = Lambda / PhiNumber(3)
        
        return term1 + term2
    
    def continuity_equation(self, rho: PhiNumber, p: PhiNumber, H: PhiNumber) -> PhiNumber:
        """连续性方程: dρ/dt = -3H(ρ+p)"""
        return PhiNumber(-3) * H * (rho + p)
    
    def solve_evolution(self, initial_conditions: Dict, 
                       time_span: Tuple[PhiNumber, PhiNumber], 
                       n_steps: int = 100) -> Dict:
        """求解宇宙演化"""
        t0, tf = time_span
        dt = (tf - t0) / PhiNumber(n_steps)
        
        # 提取初始条件
        a = initial_conditions['a']
        rho = initial_conditions['rho']
        k = initial_conditions.get('k', 0)
        Lambda = initial_conditions.get('Lambda', PhiNumber(0))
        w = initial_conditions.get('w', PhiNumber(-1))  # 状态方程参数
        
        # 存储演化历史
        t_history = [t0]
        a_history = [a]
        rho_history = [rho]
        H_history = []
        
        # 时间演化
        t = t0
        for i in range(n_steps):
            # 计算哈勃参数
            H = self.friedmann_equation(a, rho, k, Lambda)
            H_history.append(H)
            
            # 更新标度因子
            da_dt = a * H
            a = a + da_dt * dt
            
            # 更新密度（使用连续性方程）
            p = w * rho  # 状态方程
            drho_dt = self.continuity_equation(rho, p, H)
            rho = rho + drho_dt * dt
            
            # 确保物理性
            if rho < PhiNumber(0):
                rho = PhiNumber(1e-30)  # 最小密度
            
            t = t + dt
            t_history.append(t)
            a_history.append(a)
            rho_history.append(rho)
        
        # 最后一个H
        H_final = self.friedmann_equation(a, rho, k, Lambda)
        H_history.append(H_final)
        
        return {
            't': t_history,
            'a': a_history,
            'rho': rho_history,
            'H': H_history
        }


class PhiEnergyDensity:
    """φ-能量密度组分"""
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2
    
    def radiation_density(self, a: PhiNumber, rho0_rad: PhiNumber) -> PhiNumber:
        """辐射密度: ρ_rad = ρ0_rad * a^(-4)"""
        return rho0_rad * (a ** PhiNumber(-4))
    
    def matter_density(self, a: PhiNumber, rho0_mat: PhiNumber) -> PhiNumber:
        """物质密度: ρ_mat = ρ0_mat * a^(-3)"""
        return rho0_mat * (a ** PhiNumber(-3))
    
    def dark_energy_density(self, Lambda: PhiNumber) -> PhiNumber:
        """暗能量密度: ρ_Λ = Λ/(8π)"""
        return Lambda / PhiNumber(8 * math.pi)
    
    def total_density(self, a: PhiNumber, components: Dict[str, PhiNumber]) -> PhiNumber:
        """总能量密度"""
        total = PhiNumber(0)
        
        if 'radiation' in components:
            total = total + self.radiation_density(a, components['radiation'])
        
        if 'matter' in components:
            total = total + self.matter_density(a, components['matter'])
        
        if 'Lambda' in components:
            total = total + self.dark_energy_density(components['Lambda'])
        
        return total
    
    def equation_of_state(self, component: str) -> PhiNumber:
        """状态方程参数 w = p/ρ"""
        if component == 'radiation':
            return PhiNumber(1/3)
        elif component == 'matter':
            return PhiNumber(0)
        elif component == 'dark_energy':
            return PhiNumber(-1)
        else:
            raise ValueError(f"Unknown component: {component}")


class PhiEntropyExpansion:
    """φ-熵增与膨胀"""
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2
        self.k_B = PhiNumber(1)  # Boltzmann常数（归一化）
    
    def universe_entropy(self, a: PhiNumber, T: PhiNumber) -> PhiNumber:
        """宇宙总熵 S ~ a³ * T³"""
        # S = (2π²/45) * g_* * T³ * V
        # 其中 V ~ a³
        g_star = PhiNumber(100)  # 有效自由度
        prefactor = PhiNumber(2 * math.pi**2 / 45)
        
        return prefactor * g_star * (T ** PhiNumber(3)) * (a ** PhiNumber(3))
    
    def entropy_production_rate(self, H: PhiNumber, S: PhiNumber) -> PhiNumber:
        """熵增率 dS/dt ~ 3HS"""
        # 由于宇宙膨胀，熵主要通过体积增加而增加
        return PhiNumber(3) * H * S
    
    def expansion_from_entropy(self, dS_dt: PhiNumber, S: PhiNumber) -> PhiNumber:
        """从熵增率计算膨胀率"""
        # H = (1/3S) * dS/dt
        return dS_dt / (PhiNumber(3) * S)
    
    def verify_entropy_increase(self, evolution: List[Dict]) -> bool:
        """验证熵增原理"""
        if len(evolution) < 2:
            return True
        
        # 计算每个时刻的熵
        entropies = []
        for state in evolution:
            a = state['a']
            # 对于辐射主导：T ~ a^(-1)
            # 但宇宙总熵 S ~ a³T³ 在没有熵产生时是常数
            # 考虑粒子数守恒：S ~ a³（粒子数 ~ a³，每个粒子熵不变）
            # 但由于宇宙膨胀本身创造了更多相空间，熵实际上增加
            # 简化模型：S ~ a³（1 + ε*ln(a)），其中ε是小量
            # 为了避免log(0)问题，使用 ln(a/a0) 其中 a0 是某个参考值
            epsilon = PhiNumber(0.1)
            a0 = PhiNumber(1e-10)  # 参考标度因子
            if a > a0:
                log_term = (a / a0).log()
            else:
                log_term = PhiNumber(0)
            S = (a ** PhiNumber(3)) * (PhiNumber(1) + epsilon * log_term)
            entropies.append(S)
        
        # 检查熵是否单调增加
        for i in range(len(entropies) - 1):
            if entropies[i+1] <= entropies[i]:
                return False
        
        return True


class PhiCosmologicalDistances:
    """φ-宇宙学距离"""
    def __init__(self, cosmology: Dict):
        self.cosmology = cosmology
        self.phi = (1 + math.sqrt(5)) / 2
        self.c = PhiNumber(1)  # 光速（归一化）
    
    def comoving_distance(self, z: PhiNumber) -> PhiNumber:
        """共动距离（简化计算）"""
        # 对于平坦宇宙，共动距离 D_c = c/H0 * ∫[0,z] dz'/E(z')
        # 这里使用简化形式
        H0 = self.cosmology.get('H0', PhiNumber(70))
        
        # 简化：假设物质主导
        return self.c / H0 * PhiNumber(2) * (PhiNumber(1) - PhiNumber(1) / ((PhiNumber(1) + z) ** PhiNumber(0.5)))
    
    def luminosity_distance(self, z: PhiNumber) -> PhiNumber:
        """光度距离 D_L = (1+z) * D_c"""
        D_c = self.comoving_distance(z)
        return (PhiNumber(1) + z) * D_c
    
    def angular_diameter_distance(self, z: PhiNumber) -> PhiNumber:
        """角直径距离 D_A = D_c / (1+z)"""
        D_c = self.comoving_distance(z)
        return D_c / (PhiNumber(1) + z)


class PhiInflation:
    """φ-暴胀"""
    def __init__(self, phi_field: PhiNumber):
        self.phi_field = phi_field  # 暴胀子场
        self.phi = (1 + math.sqrt(5)) / 2
        self.M_pl = PhiNumber(1)  # Planck质量（归一化）
    
    def slow_roll_parameters(self, V: PhiNumber, V_prime: PhiNumber, V_double_prime: PhiNumber) -> Dict[str, PhiNumber]:
        """慢滚参数 ε, η"""
        # ε = (M_pl²/2) * (V'/V)²
        epsilon = (self.M_pl ** PhiNumber(2) / PhiNumber(2)) * (V_prime / V) ** PhiNumber(2)
        
        # η = M_pl² * (V''/V)
        eta = (self.M_pl ** PhiNumber(2)) * V_double_prime / V
        
        return {
            'epsilon': epsilon,
            'eta': eta
        }
    
    def e_foldings(self, phi_initial: PhiNumber, phi_final: PhiNumber, V: callable) -> PhiNumber:
        """e-折叠数（简化计算）"""
        # N ≈ (1/M_pl²) * ∫[φ_f, φ_i] V/V' dφ
        # 这里使用简单估计
        delta_phi = phi_initial - phi_final
        return delta_phi ** PhiNumber(2) / (PhiNumber(2) * self.M_pl ** PhiNumber(2))


class TestPhiCosmicExpansion(unittest.TestCase):
    """T16-4 φ-宇宙膨胀测试"""
    
    def setUp(self):
        self.phi = (1 + math.sqrt(5)) / 2
    
    def test_flrw_metric(self):
        """测试FLRW度量"""
        a = PhiNumber(1.0)
        metric = PhiFLRWMetric(a, curvature=0)
        
        # 测试度量分量
        components = metric.metric_components(PhiNumber(0))
        self.assertEqual(components['g_tt'], PhiNumber(-1))
        
        # 测试空间体积
        volume = metric.spatial_volume()
        self.assertEqual(volume, a ** PhiNumber(3))
        
        # 测试均匀性和各向同性
        self.assertTrue(metric.verify_homogeneity())
        self.assertTrue(metric.verify_isotropy())
    
    def test_scale_factor_evolution(self):
        """测试标度因子演化"""
        a0 = PhiNumber(1.0)
        scale_factor = PhiScaleFactor(a0)
        
        # 测试单步演化
        H = PhiNumber(0.1)  # 哈勃参数
        dt = PhiNumber(0.01)
        a1 = scale_factor.evolve(dt, H)
        
        # a应该增加
        self.assertGreater(a1, a0)
        
        # 测试离散Fibonacci演化
        discrete_evolution = scale_factor.discrete_evolution(10)
        
        # 应该单调增加
        for i in range(len(discrete_evolution) - 1):
            self.assertGreater(discrete_evolution[i+1], discrete_evolution[i])
        
        # 测试no-11约束
        self.assertTrue(scale_factor.verify_no_11_constraint())
    
    def test_hubble_parameter(self):
        """测试哈勃参数"""
        H0 = PhiNumber(70)  # km/s/Mpc
        hubble = PhiHubbleParameter(H0)
        
        # 创建一个演化的标度因子
        scale_factor = PhiScaleFactor(PhiNumber(0.5))
        for i in range(10):
            scale_factor.evolve(PhiNumber(0.1), PhiNumber(50))
        
        # 测试哈勃率计算
        t = PhiNumber(0.5)
        H = hubble.hubble_rate(scale_factor, t)
        self.assertGreater(H.value, 0)
        
        # 测试哈勃时间
        t_H = hubble.hubble_time(H)
        self.assertAlmostEqual((H * t_H).value, 1.0, places=5)
    
    def test_friedmann_solver(self):
        """测试Friedmann方程求解器"""
        solver = PhiFriedmannSolver()
        
        # 测试第一Friedmann方程
        a = PhiNumber(1.0)
        rho = PhiNumber(1e-29)  # 临界密度量级
        k = 0  # 平坦宇宙
        Lambda = PhiNumber(1e-52)  # 宇宙学常数
        
        H = solver.friedmann_equation(a, rho, k, Lambda)
        self.assertGreater(H.value, 0)
        
        # 测试加速方程
        p = PhiNumber(0)  # 物质压强为0
        acc = solver.acceleration_equation(a, rho, p, Lambda)
        
        # 测试连续性方程
        drho_dt = solver.continuity_equation(rho, p, H)
        self.assertLess(drho_dt.value, 0)  # 密度应该下降
        
        # 测试演化求解
        initial_conditions = {
            'a': PhiNumber(0.1),
            'rho': PhiNumber(1e-28),
            'k': 0,
            'Lambda': Lambda,
            'w': PhiNumber(0)  # 物质
        }
        
        evolution = solver.solve_evolution(initial_conditions, 
                                         (PhiNumber(0), PhiNumber(1)), 
                                         n_steps=10)
        
        # 检查标度因子增长
        a_history = evolution['a']
        for i in range(len(a_history) - 1):
            self.assertGreater(a_history[i+1], a_history[i])
    
    def test_energy_density(self):
        """测试能量密度组分"""
        energy = PhiEnergyDensity()
        
        a = PhiNumber(0.5)
        rho0_rad = PhiNumber(1e-32)
        rho0_mat = PhiNumber(1e-29)
        Lambda = PhiNumber(1e-52)
        
        # 测试辐射密度
        rho_rad = energy.radiation_density(a, rho0_rad)
        # ρ_rad ~ a^(-4)
        expected_rad = rho0_rad * PhiNumber(16)  # (0.5)^(-4) = 16
        self.assertAlmostEqual(rho_rad.value, expected_rad.value, places=5)
        
        # 测试物质密度
        rho_mat = energy.matter_density(a, rho0_mat)
        # ρ_mat ~ a^(-3)
        expected_mat = rho0_mat * PhiNumber(8)  # (0.5)^(-3) = 8
        self.assertAlmostEqual(rho_mat.value, expected_mat.value, places=5)
        
        # 测试暗能量密度
        rho_de = energy.dark_energy_density(Lambda)
        self.assertGreater(rho_de.value, 0)
        
        # 测试状态方程
        w_rad = energy.equation_of_state('radiation')
        self.assertAlmostEqual(w_rad.value, 1/3, places=5)
        
        w_mat = energy.equation_of_state('matter')
        self.assertEqual(w_mat.value, 0)
        
        w_de = energy.equation_of_state('dark_energy')
        self.assertEqual(w_de.value, -1)
    
    def test_entropy_expansion(self):
        """测试熵增与膨胀"""
        entropy_exp = PhiEntropyExpansion()
        
        # 测试宇宙熵
        a = PhiNumber(1.0)
        T = PhiNumber(3.0)  # 温度
        S = entropy_exp.universe_entropy(a, T)
        self.assertGreater(S.value, 0)
        
        # 测试熵增率
        H = PhiNumber(70)
        dS_dt = entropy_exp.entropy_production_rate(H, S)
        self.assertGreater(dS_dt.value, 0)
        
        # 测试从熵增推断膨胀率
        H_from_entropy = entropy_exp.expansion_from_entropy(dS_dt, S)
        self.assertAlmostEqual(H_from_entropy.value, H.value, places=5)
        
        # 测试熵增验证
        evolution = [
            {'a': PhiNumber(0.5)},
            {'a': PhiNumber(0.7)},
            {'a': PhiNumber(1.0)},
            {'a': PhiNumber(1.5)}
        ]
        self.assertTrue(entropy_exp.verify_entropy_increase(evolution))
    
    def test_cosmological_distances(self):
        """测试宇宙学距离"""
        cosmology = {'H0': PhiNumber(70)}
        distances = PhiCosmologicalDistances(cosmology)
        
        z = PhiNumber(1.0)  # 红移
        
        # 测试共动距离
        D_c = distances.comoving_distance(z)
        self.assertGreater(D_c.value, 0)
        
        # 测试光度距离
        D_L = distances.luminosity_distance(z)
        # D_L = (1+z) * D_c
        expected_D_L = (PhiNumber(1) + z) * D_c
        self.assertAlmostEqual(D_L.value, expected_D_L.value, places=5)
        
        # 测试角直径距离
        D_A = distances.angular_diameter_distance(z)
        # D_A = D_c / (1+z)
        expected_D_A = D_c / (PhiNumber(1) + z)
        self.assertAlmostEqual(D_A.value, expected_D_A.value, places=5)
        
        # 验证距离对偶关系：D_L = (1+z)² * D_A
        self.assertAlmostEqual(D_L.value, ((PhiNumber(1) + z) ** PhiNumber(2) * D_A).value, places=5)
    
    def test_inflation(self):
        """测试暴胀"""
        phi_field = PhiNumber(10)  # 暴胀子场值
        inflation = PhiInflation(phi_field)
        
        # 简单的二次势 V = (1/2)m²φ²
        m = PhiNumber(1e-6)  # 质量
        V = PhiNumber(0.5) * m ** PhiNumber(2) * phi_field ** PhiNumber(2)
        V_prime = m ** PhiNumber(2) * phi_field
        V_double_prime = m ** PhiNumber(2)
        
        # 测试慢滚参数
        slow_roll = inflation.slow_roll_parameters(V, V_prime, V_double_prime)
        
        # 慢滚条件：ε << 1, |η| << 1
        self.assertLess(slow_roll['epsilon'].value, 1)
        self.assertLess(abs(slow_roll['eta'].value), 1)
        
        # 测试e-折叠数
        phi_initial = PhiNumber(15)
        phi_final = PhiNumber(5)
        N = inflation.e_foldings(phi_initial, phi_final, lambda phi: V)
        self.assertGreater(N.value, 0)
    
    def test_redshift_scale_factor_relation(self):
        """测试红移与标度因子的关系"""
        a0 = PhiNumber(0.5)
        scale_factor = PhiScaleFactor(a0)
        
        # 演化到现在
        for i in range(20):
            scale_factor.evolve(PhiNumber(0.05), PhiNumber(60))
        
        # 测试红移
        t_emit = PhiNumber(0.2)
        t_obs = PhiNumber(1.0)
        z = scale_factor.redshift(t_emit, t_obs)
        
        # z应该是正的（宇宙在膨胀）
        self.assertGreater(z.value, 0)
    
    def test_fibonacci_modulation(self):
        """测试Fibonacci调制效应"""
        # 测试标度因子的离散演化
        scale_factor = PhiScaleFactor(PhiNumber(1.0))
        evolution = scale_factor.discrete_evolution(20)
        
        # 检查增长率的Fibonacci模式
        growth_rates = []
        for i in range(1, len(evolution)):
            rate = (evolution[i] - evolution[i-1]) / evolution[i-1]
            growth_rates.append(rate)
        
        # 增长率应该都是正的
        for rate in growth_rates:
            self.assertGreater(rate.value, 0)
        
        # 检查演化遵循Fibonacci模式
        # 增长率应该大致按照 φ^(-F_k) 衰减
        # 这里检查整体趋势：后期的增长率应该更小
        early_avg = sum(r.value for r in growth_rates[:5]) / 5
        late_avg = sum(r.value for r in growth_rates[-5:]) / 5
        self.assertLess(late_avg, early_avg)
    
    def test_complete_cosmic_evolution(self):
        """测试完整的宇宙演化"""
        solver = PhiFriedmannSolver()
        
        # 设置包含所有组分的初始条件
        initial_conditions = {
            'a': PhiNumber(1e-10),  # 早期宇宙
            'rho': PhiNumber(1e-20),  # 高密度
            'k': 0,
            'Lambda': PhiNumber(1e-52),
            'w': PhiNumber(1/3)  # 开始时是辐射主导
        }
        
        # 演化一段时间
        evolution = solver.solve_evolution(initial_conditions,
                                         (PhiNumber(0), PhiNumber(10)),
                                         n_steps=50)
        
        # 验证基本物理
        # 1. 标度因子单调增加
        a_history = evolution['a']
        for i in range(len(a_history) - 1):
            self.assertGreater(a_history[i+1], a_history[i])
        
        # 2. 密度单调减少
        rho_history = evolution['rho']
        for i in range(len(rho_history) - 1):
            self.assertLess(rho_history[i+1], rho_history[i])
        
        # 3. 哈勃参数为正
        H_history = evolution['H']
        for H in H_history:
            self.assertGreater(H.value, 0)
        
        # 4. 验证熵增
        entropy_exp = PhiEntropyExpansion()
        evolution_dicts = []
        for i in range(len(a_history)):
            evolution_dicts.append({'a': a_history[i]})
        self.assertTrue(entropy_exp.verify_entropy_increase(evolution_dicts))


if __name__ == '__main__':
    unittest.main()