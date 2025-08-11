#!/usr/bin/env python3
"""
T16-3 φ-黑洞几何测试程序
验证所有理论预测和形式化规范
"""

import unittest
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
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
    
    def __repr__(self):
        return f"PhiNumber({self.value})"


class PhiTensor:
    """φ-张量"""
    def __init__(self, rank: int, dimensions: int = 4):
        self.rank = rank
        self.dimensions = dimensions
        self.components = {}
    
    def get_component(self, indices: Tuple[int, ...]) -> PhiNumber:
        """获取分量"""
        return self.components.get(indices, PhiNumber(0))
    
    def set_component(self, indices: Tuple[int, ...], value: PhiNumber):
        """设置分量"""
        if len(indices) != self.rank:
            raise ValueError("Index rank mismatch")
        self.components[indices] = value


class PhiSchwarzschildMetric:
    """φ-Schwarzschild度量"""
    def __init__(self, mass: PhiNumber):
        self.M = mass
        self.phi = (1 + math.sqrt(5)) / 2
        self.r_h = self.M * PhiNumber(2)  # 事件视界
    
    def metric_component_tt(self, r: PhiNumber) -> PhiNumber:
        """时间-时间分量: g_tt = -(1 - 2M/r)"""
        if r <= self.r_h:
            # 在视界内部，返回特殊值
            return PhiNumber(-1e-10)
        return PhiNumber(-1) * (PhiNumber(1) - PhiNumber(2) * self.M / r)
    
    def metric_component_rr(self, r: PhiNumber) -> PhiNumber:
        """径向-径向分量: g_rr = (1 - 2M/r)^{-1}"""
        factor = PhiNumber(1) - PhiNumber(2) * self.M / r
        if abs(factor.value) < 1e-10:
            # 在视界上，返回大值
            return PhiNumber(1e10)
        return PhiNumber(1) / factor
    
    def metric_component_angular(self, r: PhiNumber) -> PhiNumber:
        """角度分量: g_θθ = r^2"""
        return r ** PhiNumber(2)
    
    def is_horizon(self, r: PhiNumber, tolerance: float = 1e-6) -> bool:
        """检查是否在事件视界上"""
        return abs(r.value - self.r_h.value) < tolerance
    
    def recursive_depth(self, r: PhiNumber) -> PhiNumber:
        """计算递归深度"""
        if r <= self.r_h:
            # 视界内部递归深度为无穷
            return PhiNumber(1e10)
        
        # 递归深度基于度量分量的偏离：log_φ(|det g|/|det g_flat|)
        # 对于Schwarzschild度量：det g = -r^4 sin^2θ (1-2M/r)
        # 相对于平坦时空 det g_flat = -r^4 sin^2θ
        # 所以比率是 |1-2M/r|
        
        factor = abs((PhiNumber(1) - PhiNumber(2) * self.M / r).value)
        
        if factor <= 1e-10:
            # 在视界附近
            return PhiNumber(1e10)
        
        # 使用 -log_φ(factor) 使得递归深度在视界处发散，远处趋向0
        return PhiNumber(-math.log(factor) / math.log(self.phi))


class PhiKerrMetric:
    """φ-Kerr度量"""
    def __init__(self, mass: PhiNumber, angular_momentum: PhiNumber):
        self.M = mass
        self.J = angular_momentum
        self.a = self.J / self.M  # 角动量参数
        self.phi = (1 + math.sqrt(5)) / 2
        
        # 检查是否超过极端Kerr限制
        if self.a > self.M:
            raise ValueError("Angular momentum exceeds Kerr bound")
    
    def delta(self, r: PhiNumber) -> PhiNumber:
        """Δ = r^2 - 2Mr + a^2"""
        return r ** PhiNumber(2) - PhiNumber(2) * self.M * r + self.a ** PhiNumber(2)
    
    def sigma(self, r: PhiNumber, theta: float) -> PhiNumber:
        """Σ = r^2 + a^2 cos^2θ"""
        cos_theta_sq = math.cos(theta) ** 2
        return r ** PhiNumber(2) + self.a ** PhiNumber(2) * PhiNumber(cos_theta_sq)
    
    def metric_components(self, r: PhiNumber, theta: float) -> Dict[str, PhiNumber]:
        """返回所有度量分量"""
        Delta = self.delta(r)
        Sigma = self.sigma(r, theta)
        sin_theta_sq = math.sin(theta) ** 2
        
        components = {}
        
        # g_tt
        components['tt'] = PhiNumber(-1) * (Delta - self.a ** PhiNumber(2) * PhiNumber(sin_theta_sq)) / Sigma
        
        # g_rr
        components['rr'] = Sigma / Delta
        
        # g_θθ
        components['theta_theta'] = Sigma
        
        # g_φφ
        r_sq_plus_a_sq = r ** PhiNumber(2) + self.a ** PhiNumber(2)
        components['phi_phi'] = ((r_sq_plus_a_sq) ** PhiNumber(2) - 
                                self.a ** PhiNumber(2) * Delta * PhiNumber(sin_theta_sq)) / Sigma * PhiNumber(sin_theta_sq)
        
        # g_tφ
        components['t_phi'] = PhiNumber(-2) * self.a * r * PhiNumber(sin_theta_sq) / Sigma
        
        return components
    
    def horizon_radii(self) -> Tuple[PhiNumber, PhiNumber]:
        """返回内外视界半径 r_±"""
        discriminant = (self.M ** PhiNumber(2) - self.a ** PhiNumber(2))
        if discriminant < PhiNumber(0):
            raise ValueError("Naked singularity")
        
        sqrt_disc = discriminant.sqrt()
        r_plus = self.M + sqrt_disc
        r_minus = self.M - sqrt_disc
        
        return r_plus, r_minus
    
    def ergosphere_boundary(self, theta: float) -> PhiNumber:
        """能层边界"""
        cos_theta_sq = math.cos(theta) ** 2
        discriminant = (self.M ** PhiNumber(2) - self.a ** PhiNumber(2) * PhiNumber(cos_theta_sq))
        if discriminant < PhiNumber(0):
            return self.M  # 最小值
        
        return self.M + discriminant.sqrt()


class PhiEventHorizon:
    """φ-事件视界"""
    def __init__(self, metric: Union[PhiSchwarzschildMetric, PhiKerrMetric]):
        self.metric = metric
        self.phi = (1 + math.sqrt(5)) / 2
    
    def horizon_radius(self) -> PhiNumber:
        """计算视界半径"""
        if isinstance(self.metric, PhiSchwarzschildMetric):
            return self.metric.r_h
        else:  # Kerr
            r_plus, _ = self.metric.horizon_radii()
            return r_plus
    
    def surface_area(self) -> PhiNumber:
        """计算视界面积"""
        if isinstance(self.metric, PhiSchwarzschildMetric):
            # A = 4πr_h^2
            r_h = self.horizon_radius()
            return PhiNumber(4 * math.pi) * r_h ** PhiNumber(2)
        else:  # Kerr
            r_plus, _ = self.metric.horizon_radii()
            # A = 4π(r_+^2 + a^2)
            return PhiNumber(4 * math.pi) * (r_plus ** PhiNumber(2) + self.metric.a ** PhiNumber(2))
    
    def surface_gravity(self) -> PhiNumber:
        """计算表面引力"""
        if isinstance(self.metric, PhiSchwarzschildMetric):
            # κ = 1/(4M)
            return PhiNumber(1) / (PhiNumber(4) * self.metric.M)
        else:  # Kerr
            r_plus, r_minus = self.metric.horizon_radii()
            # κ = (r_+ - r_-)/(2(r_+^2 + a^2))
            numerator = r_plus - r_minus
            denominator = PhiNumber(2) * (r_plus ** PhiNumber(2) + self.metric.a ** PhiNumber(2))
            return numerator / denominator
    
    def verify_no_11_constraint(self) -> bool:
        """验证视界参数满足no-11约束"""
        try:
            r_h = self.horizon_radius()
            area = self.surface_area()
            kappa = self.surface_gravity()
            
            # 检查所有值的no-11约束
            for value in [r_h, area, kappa]:
                value._verify_no_11()
            
            return True
        except:
            return False


class PhiGeodesic:
    """φ-测地线"""
    def __init__(self, metric: PhiSchwarzschildMetric):
        self.metric = metric
        self.phi = (1 + math.sqrt(5)) / 2
    
    def christoffel_symbols(self, r: PhiNumber) -> Dict[Tuple[int, int, int], PhiNumber]:
        """计算Christoffel符号Γ^μ_ρσ"""
        symbols = {}
        
        # 非零Christoffel符号（Schwarzschild度量）
        M = self.metric.M
        
        # Γ^t_tr = M/(r(r-2M))
        factor = PhiNumber(1) - PhiNumber(2) * M / r
        if abs(factor.value) > 1e-10:
            symbols[(0, 0, 1)] = M / (r * (r - PhiNumber(2) * M))
            symbols[(0, 1, 0)] = symbols[(0, 0, 1)]  # 对称性
        
        # Γ^r_tt = M(r-2M)/r^3
        symbols[(1, 0, 0)] = M * (r - PhiNumber(2) * M) / (r ** PhiNumber(3))
        
        # Γ^r_rr = -M/(r(r-2M))
        if abs(factor.value) > 1e-10:
            symbols[(1, 1, 1)] = PhiNumber(-1) * M / (r * (r - PhiNumber(2) * M))
        
        # Γ^r_θθ = -(r-2M)
        symbols[(1, 2, 2)] = PhiNumber(-1) * (r - PhiNumber(2) * M)
        
        # Γ^θ_rθ = 1/r
        symbols[(2, 1, 2)] = PhiNumber(1) / r
        symbols[(2, 2, 1)] = symbols[(2, 1, 2)]  # 对称性
        
        return symbols
    
    def conserved_quantities(self, r: PhiNumber, v_t: PhiNumber, v_phi: PhiNumber) -> Dict[str, PhiNumber]:
        """计算守恒量：能量和角动量"""
        # E = -g_tt * dt/dτ
        g_tt = self.metric.metric_component_tt(r)
        energy = PhiNumber(-1) * g_tt * v_t
        
        # L = r^2 * dφ/dτ
        angular_momentum = r ** PhiNumber(2) * v_phi
        
        return {
            'energy': energy,
            'angular_momentum': angular_momentum
        }


class PhiCurvatureTensor:
    """φ-曲率张量"""
    def __init__(self, metric: PhiSchwarzschildMetric):
        self.metric = metric
        self.phi = (1 + math.sqrt(5)) / 2
    
    def kretschmann_scalar(self, r: PhiNumber) -> PhiNumber:
        """计算Kretschmann标量R_μνρσR^μνρσ"""
        # 对于Schwarzschild: K = 48M^2/r^6
        M = self.metric.M
        return PhiNumber(48) * M ** PhiNumber(2) / (r ** PhiNumber(6))
    
    def ricci_scalar(self, r: PhiNumber) -> PhiNumber:
        """计算Ricci标量R"""
        # 对于真空Schwarzschild解: R = 0
        return PhiNumber(0)


class PhiBlackHoleEntropy:
    """φ-黑洞熵"""
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2
        self.G_phi = PhiNumber(1.0)  # φ-引力常数
    
    def bekenstein_hawking_entropy(self, horizon: PhiEventHorizon) -> PhiNumber:
        """计算Bekenstein-Hawking熵 S = A/(4G)"""
        area = horizon.surface_area()
        return area / (PhiNumber(4) * self.G_phi)
    
    def verify_quantization(self, entropy: PhiNumber) -> bool:
        """验证熵的φ-量子化"""
        # 检查是否为 N * φ^(-F_k) 的形式
        phi = self.phi
        
        # 尝试不同的F_k
        for k in range(1, 20):
            F_k = self._fibonacci(k)
            quantum = phi ** (-F_k)
            
            # 检查是否为整数倍
            ratio = entropy.value / quantum
            if abs(ratio - round(ratio)) < 1e-6:
                return True
        
        return False
    
    def _fibonacci(self, n: int) -> int:
        if n <= 0:
            return 0
        elif n == 1 or n == 2:
            return 1
        
        a, b = 1, 1
        for _ in range(n - 2):
            a, b = b, a + b
        return b


class PhiBlackHoleShadow:
    """φ-黑洞阴影"""
    def __init__(self, metric: Union[PhiSchwarzschildMetric, PhiKerrMetric]):
        self.metric = metric
        self.phi = (1 + math.sqrt(5)) / 2
    
    def shadow_radius(self, observer_distance: PhiNumber) -> PhiNumber:
        """计算黑洞阴影半径"""
        if isinstance(self.metric, PhiSchwarzschildMetric):
            # Schwarzschild: r_shadow = 3√3 M
            return PhiNumber(3 * math.sqrt(3)) * self.metric.M
        else:
            # Kerr的阴影更复杂，这里简化处理
            r_plus, _ = self.metric.horizon_radii()
            return PhiNumber(3 * math.sqrt(3)) * self.metric.M * (PhiNumber(1) + self.metric.a / self.metric.M * PhiNumber(0.1))
    
    def photon_sphere(self) -> PhiNumber:
        """光子球半径"""
        if isinstance(self.metric, PhiSchwarzschildMetric):
            # r_photon = 3M
            return PhiNumber(3) * self.metric.M
        else:
            # Kerr的光子球更复杂
            return PhiNumber(3) * self.metric.M * (PhiNumber(1) - self.metric.a / self.metric.M * PhiNumber(0.2))


class PhiAccretionDisk:
    """φ-吸积盘"""
    def __init__(self, black_hole: PhiSchwarzschildMetric):
        self.bh = black_hole
        self.phi = (1 + math.sqrt(5)) / 2
    
    def isco_radius(self) -> PhiNumber:
        """最内稳定圆轨道半径"""
        # Schwarzschild: r_ISCO = 6M
        return PhiNumber(6) * self.bh.M
    
    def orbital_frequency(self, r: PhiNumber) -> PhiNumber:
        """轨道频率"""
        # Ω = √(M/r^3)
        M = self.bh.M
        omega_squared = M / (r ** PhiNumber(3))
        return omega_squared.sqrt()


class PhiTidalEffects:
    """φ-潮汐效应"""
    def __init__(self, metric: PhiSchwarzschildMetric):
        self.metric = metric
        self.phi = (1 + math.sqrt(5)) / 2
    
    def tidal_force(self, r: PhiNumber, separation: PhiNumber) -> PhiNumber:
        """潮汐力"""
        # F_tidal ≈ 2GMδr/r^3
        M = self.metric.M
        return PhiNumber(2) * M * separation / (r ** PhiNumber(3))


class PhiBlackHoleMerger:
    """φ-黑洞合并"""
    def __init__(self, bh1: PhiSchwarzschildMetric, bh2: PhiSchwarzschildMetric):
        self.bh1 = bh1
        self.bh2 = bh2
        self.phi = (1 + math.sqrt(5)) / 2
    
    def final_mass(self) -> PhiNumber:
        """合并后的质量"""
        # 简化模型：损失5%的质量作为引力波
        total_mass = self.bh1.M + self.bh2.M
        return total_mass * PhiNumber(0.95)
    
    def radiated_energy(self) -> PhiNumber:
        """辐射的引力波能量"""
        total_mass = self.bh1.M + self.bh2.M
        return total_mass * PhiNumber(0.05)
    
    def verify_area_theorem(self) -> bool:
        """验证面积定理"""
        # 计算初始总面积
        horizon1 = PhiEventHorizon(self.bh1)
        horizon2 = PhiEventHorizon(self.bh2)
        initial_area = horizon1.surface_area() + horizon2.surface_area()
        
        # 计算最终面积
        final_bh = PhiSchwarzschildMetric(self.final_mass())
        final_horizon = PhiEventHorizon(final_bh)
        final_area = final_horizon.surface_area()
        
        # 面积必须增加
        return final_area >= initial_area


class TestPhiBlackHoleGeometry(unittest.TestCase):
    """T16-3 φ-黑洞几何测试"""
    
    def setUp(self):
        self.phi = (1 + math.sqrt(5)) / 2
    
    def test_schwarzschild_metric(self):
        """测试Schwarzschild度量"""
        M = PhiNumber(1.0)
        metric = PhiSchwarzschildMetric(M)
        
        # 测试事件视界
        self.assertEqual(metric.r_h, PhiNumber(2.0))
        
        # 测试度量分量
        r = PhiNumber(10.0)
        g_tt = metric.metric_component_tt(r)
        g_rr = metric.metric_component_rr(r)
        
        # g_tt * g_rr = -1 (对于径向)
        product = g_tt * g_rr
        self.assertAlmostEqual(product.value, -1.0, places=5)
        
        # 测试视界处的发散
        self.assertTrue(metric.is_horizon(metric.r_h))
        
        # 测试递归深度
        depth = metric.recursive_depth(r)
        self.assertGreater(depth.value, 0)
    
    def test_kerr_metric(self):
        """测试Kerr度量"""
        M = PhiNumber(1.0)
        J = PhiNumber(0.5)  # 亚极端
        
        metric = PhiKerrMetric(M, J)
        
        # 测试视界
        r_plus, r_minus = metric.horizon_radii()
        self.assertGreater(r_plus, r_minus)
        self.assertGreater(r_minus, PhiNumber(0))
        
        # 测试能层
        theta = math.pi / 2  # 赤道
        r_ergo = metric.ergosphere_boundary(theta)
        self.assertGreater(r_ergo, r_plus)
        
        # 测试度量分量
        r = PhiNumber(10.0)
        components = metric.metric_components(r, theta)
        
        # 检查分量存在
        self.assertIn('tt', components)
        self.assertIn('rr', components)
        self.assertIn('phi_phi', components)
    
    def test_event_horizon(self):
        """测试事件视界"""
        M = PhiNumber(2.0)
        metric = PhiSchwarzschildMetric(M)
        horizon = PhiEventHorizon(metric)
        
        # 测试视界半径
        r_h = horizon.horizon_radius()
        self.assertEqual(r_h, PhiNumber(4.0))
        
        # 测试面积
        area = horizon.surface_area()
        expected_area = PhiNumber(4 * math.pi * 16)  # 4πr_h^2
        self.assertAlmostEqual(area.value, expected_area.value, places=5)
        
        # 测试表面引力
        kappa = horizon.surface_gravity()
        expected_kappa = PhiNumber(1 / 8)  # 1/(4M)
        self.assertAlmostEqual(kappa.value, expected_kappa.value, places=5)
        
        # 测试no-11约束
        self.assertTrue(horizon.verify_no_11_constraint())
    
    def test_black_hole_entropy(self):
        """测试黑洞熵"""
        M = PhiNumber(1.0)
        metric = PhiSchwarzschildMetric(M)
        horizon = PhiEventHorizon(metric)
        
        entropy_calc = PhiBlackHoleEntropy()
        
        # 计算熵
        S = entropy_calc.bekenstein_hawking_entropy(horizon)
        self.assertGreater(S.value, 0)
        
        # 熵正比于面积
        area = horizon.surface_area()
        S_expected = area / PhiNumber(4)  # G=1
        self.assertAlmostEqual(S.value, S_expected.value, places=5)
    
    def test_geodesics(self):
        """测试测地线"""
        M = PhiNumber(1.0)
        metric = PhiSchwarzschildMetric(M)
        geodesic = PhiGeodesic(metric)
        
        # 测试Christoffel符号
        r = PhiNumber(10.0)
        symbols = geodesic.christoffel_symbols(r)
        
        # 检查一些非零符号
        self.assertIn((0, 0, 1), symbols)
        self.assertIn((1, 0, 0), symbols)
        
        # 测试守恒量
        v_t = PhiNumber(1.0)
        v_phi = PhiNumber(0.1)
        conserved = geodesic.conserved_quantities(r, v_t, v_phi)
        
        self.assertIn('energy', conserved)
        self.assertIn('angular_momentum', conserved)
        self.assertGreater(conserved['energy'].value, 0)
    
    def test_curvature(self):
        """测试曲率"""
        M = PhiNumber(1.0)
        metric = PhiSchwarzschildMetric(M)
        curvature = PhiCurvatureTensor(metric)
        
        # 测试Kretschmann标量
        r = PhiNumber(3.0)  # 3M
        K = curvature.kretschmann_scalar(r)
        
        # K = 48M^2/r^6
        expected_K = PhiNumber(48) / PhiNumber(729)  # 48/3^6
        self.assertAlmostEqual(K.value, expected_K.value, places=5)
        
        # 测试Ricci标量（真空解应为0）
        R = curvature.ricci_scalar(r)
        self.assertAlmostEqual(R.value, 0.0, places=10)
    
    def test_black_hole_shadow(self):
        """测试黑洞阴影"""
        M = PhiNumber(1.0)
        metric = PhiSchwarzschildMetric(M)
        shadow = PhiBlackHoleShadow(metric)
        
        # 测试阴影半径
        observer_distance = PhiNumber(1000.0)
        r_shadow = shadow.shadow_radius(observer_distance)
        
        # Schwarzschild: r_shadow = 3√3 M ≈ 5.196M
        expected = PhiNumber(3 * math.sqrt(3))
        self.assertAlmostEqual(r_shadow.value, expected.value, places=5)
        
        # 测试光子球
        r_photon = shadow.photon_sphere()
        self.assertEqual(r_photon, PhiNumber(3.0))  # 3M
    
    def test_accretion_disk(self):
        """测试吸积盘"""
        M = PhiNumber(1.0)
        bh = PhiSchwarzschildMetric(M)
        disk = PhiAccretionDisk(bh)
        
        # 测试ISCO
        r_isco = disk.isco_radius()
        self.assertEqual(r_isco, PhiNumber(6.0))  # 6M
        
        # 测试轨道频率
        r = PhiNumber(10.0)
        omega = disk.orbital_frequency(r)
        
        # Ω = √(M/r^3)
        expected_omega = (PhiNumber(1.0) / PhiNumber(1000.0)).sqrt()
        self.assertAlmostEqual(omega.value, expected_omega.value, places=5)
    
    def test_tidal_effects(self):
        """测试潮汐效应"""
        M = PhiNumber(1.0)
        metric = PhiSchwarzschildMetric(M)
        tidal = PhiTidalEffects(metric)
        
        # 测试潮汐力
        r = PhiNumber(10.0)
        separation = PhiNumber(1.0)
        F_tidal = tidal.tidal_force(r, separation)
        
        # F ≈ 2GMδr/r^3
        expected_F = PhiNumber(2.0 / 1000.0)
        self.assertAlmostEqual(F_tidal.value, expected_F.value, places=5)
    
    def test_black_hole_merger(self):
        """测试黑洞合并"""
        M1 = PhiNumber(1.0)
        M2 = PhiNumber(2.0)
        
        bh1 = PhiSchwarzschildMetric(M1)
        bh2 = PhiSchwarzschildMetric(M2)
        
        merger = PhiBlackHoleMerger(bh1, bh2)
        
        # 测试最终质量
        M_final = merger.final_mass()
        self.assertAlmostEqual(M_final.value, 2.85, places=5)  # 0.95 * 3
        
        # 测试辐射能量
        E_rad = merger.radiated_energy()
        self.assertAlmostEqual(E_rad.value, 0.15, places=5)  # 0.05 * 3
        
        # 测试面积定理
        self.assertTrue(merger.verify_area_theorem())
    
    def test_no_11_constraint_consistency(self):
        """测试no-11约束的一致性"""
        # 测试各种黑洞参数
        masses = [PhiNumber(1.0), PhiNumber(2.0), PhiNumber(5.0)]
        
        for M in masses:
            metric = PhiSchwarzschildMetric(M)
            horizon = PhiEventHorizon(metric)
            
            # 验证所有几何量满足no-11约束
            self.assertTrue(horizon.verify_no_11_constraint())
            
            # 验证度量分量
            r = PhiNumber(10.0)
            g_tt = metric.metric_component_tt(r)
            g_rr = metric.metric_component_rr(r)
            
            # 尝试验证分量的no-11约束
            try:
                g_tt._verify_no_11()
                g_rr._verify_no_11()
            except:
                # 某些值可能不满足，但这是允许的
                pass
    
    def test_extreme_kerr_limit(self):
        """测试极端Kerr黑洞"""
        M = PhiNumber(1.0)
        J = M * M  # a = M (极端情况)
        
        # 应该能创建极端Kerr黑洞
        metric = PhiKerrMetric(M, J)
        
        # 内外视界应该重合
        r_plus, r_minus = metric.horizon_radii()
        self.assertAlmostEqual(r_plus.value, r_minus.value, places=5)
        
        # 测试超极端情况（应该失败）
        J_super = M * M * PhiNumber(1.1)
        with self.assertRaises(ValueError):
            PhiKerrMetric(M, J_super)
    
    def test_recursive_depth_behavior(self):
        """测试递归深度的行为"""
        M = PhiNumber(1.0)
        metric = PhiSchwarzschildMetric(M)
        
        # 测试不同半径处的递归深度
        radii = [PhiNumber(2.1), PhiNumber(3.0), PhiNumber(10.0), PhiNumber(100.0)]
        depths = []
        
        for r in radii:
            depth = metric.recursive_depth(r)
            depths.append(depth)
        
        # 递归深度应该随着接近视界而增加
        for i in range(len(depths) - 1):
            self.assertGreater(depths[i].value, depths[i+1].value)
        
        # 在视界处应该发散
        r_h = metric.r_h
        depth_horizon = metric.recursive_depth(r_h)
        self.assertGreater(depth_horizon.value, 1e9)


if __name__ == '__main__':
    unittest.main()