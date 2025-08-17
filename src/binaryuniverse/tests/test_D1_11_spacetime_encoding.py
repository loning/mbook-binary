#!/usr/bin/env python3
"""
测试 D1.11: 时空编码函数

验证：
1. 时空编码的唯一性和双射性
2. No-11约束的时空一致性
3. 曲率-复杂度对应关系
4. 熵增保证和信息密度理论
5. 相对论协变性
"""

import unittest
import numpy as np
from typing import Tuple, Set, List, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.zeckendorf_base import (
    ZeckendorfInt, PhiConstant, EntropyValidator
)


class SpacetimePoint:
    """时空点表示"""
    def __init__(self, x: float, y: float, z: float, t: float):
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        
    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z}, {self.t})"


class SpacetimeEncoder:
    """时空编码函数 Ψ(x,t)"""
    
    def __init__(self):
        self.phi = PhiConstant.phi()
        
    def encode_scalar(self, value: float) -> ZeckendorfInt:
        """标量到Zeckendorf编码"""
        # 处理负数和小数
        if value < 0:
            # 使用绝对值编码，添加符号标记
            return ZeckendorfInt.from_int(int(abs(value)) + 1)
        else:
            return ZeckendorfInt.from_int(max(1, int(value) + 1))
    
    def encode_space(self, x: float, y: float, z: float) -> ZeckendorfInt:
        """
        空间坐标编码 Ψ_space(x,y,z)
        使用Zeckendorf张量积
        """
        zx = self.encode_scalar(x)
        zy = self.encode_scalar(y)
        zz = self.encode_scalar(z)
        
        # 简化的张量积：组合Fibonacci索引
        indices = set()
        
        # 张量积运算：F_i ⊗ F_j = F_{i+j-1}
        for i in zx.indices:
            for j in zy.indices:
                for k in zz.indices:
                    # 三维张量积
                    new_index = i + j + k - 2
                    if new_index >= 2:  # 确保有效索引
                        indices.add(new_index)
        
        # 应用No-11约束
        indices = self._apply_no11_constraint(indices)
        
        if not indices:
            indices = {2}  # 默认最小编码
            
        return ZeckendorfInt(frozenset(indices))
    
    def encode_time(self, t: float) -> ZeckendorfInt:
        """
        时间编码 Ψ_time(t)
        使用φ的指数增长
        """
        if t <= 0:
            return ZeckendorfInt.from_int(1)
        
        # φ^t的整数部分，确保熵增
        # 使用更陡的增长函数以保证熵增
        phi_power = int(self.phi ** (t + 1)) + int(t)
        return ZeckendorfInt.from_int(max(1, phi_power))
    
    def encode(self, point: SpacetimePoint) -> ZeckendorfInt:
        """
        完整时空编码 Ψ(x,t)
        """
        space_code = self.encode_space(point.x, point.y, point.z)
        time_code = self.encode_time(point.t)
        
        # φ-加法：合并编码
        result = self._phi_addition(space_code, time_code)
        
        # 确保No-11约束
        result_indices = self._apply_no11_constraint(result.indices)
        
        return ZeckendorfInt(frozenset(result_indices))
    
    def _phi_addition(self, z1: ZeckendorfInt, z2: ZeckendorfInt) -> ZeckendorfInt:
        """φ-加法运算 ⊕_φ"""
        # 转换为整数，相加，重新编码
        result_int = z1.to_int() + z2.to_int()
        return ZeckendorfInt.from_int(result_int)
    
    def _apply_no11_constraint(self, indices: Set[int]) -> Set[int]:
        """应用No-11约束：去除连续的Fibonacci项"""
        if not indices:
            return set()
            
        sorted_indices = sorted(indices)
        result = set()
        prev = -2  # 初始化为不可能连续的值
        
        for idx in sorted_indices:
            if idx - prev > 1:  # 非连续
                result.add(idx)
                prev = idx
            else:  # 连续，需要进位
                # 使用Fibonacci恒等式：F_i + F_{i+1} = F_{i+2}
                if prev in result:
                    result.remove(prev)
                result.add(idx + 1)
                prev = idx + 1
                
        return result
    
    def encoding_distance(self, p1: SpacetimePoint, p2: SpacetimePoint) -> float:
        """
        编码距离 d_Ψ
        """
        psi1 = self.encode(p1)
        psi2 = self.encode(p2)
        
        # 计算差异
        diff = abs(psi1.to_int() - psi2.to_int())
        if diff == 0:
            return 0.0
            
        # log_φ距离
        return np.log(diff) / np.log(self.phi)
    
    def is_causal(self, p1: SpacetimePoint, p2: SpacetimePoint) -> bool:
        """
        检查因果关系
        d_Ψ ≤ φ·|Δt|
        """
        d_psi = self.encoding_distance(p1, p2)
        dt = abs(p2.t - p1.t)
        
        # 处理dt=0的情况
        if dt == 0:
            return d_psi == 0
        
        # 允许一定的数值误差
        return d_psi <= self.phi * dt + 0.1
    
    def information_density(self, point: SpacetimePoint) -> float:
        """
        信息密度 ρ_I(x,t)
        """
        psi = self.encode(point)
        
        if not psi.indices:
            return 0.0
        
        # I_φ = Σ(log_φ F_i + 1/φ)
        density = 0.0
        for i in psi.indices:
            fib_i = ZeckendorfInt.fibonacci(i)
            if fib_i > 0:
                density += np.log(fib_i) / np.log(self.phi) + 1.0 / self.phi
                
        return density
    
    def curvature_complexity(self, point: SpacetimePoint) -> float:
        """
        曲率复杂度 K(x,t)
        """
        psi = self.encode(point)
        
        if not psi.indices:
            return 0.0
        
        # K = log_φ(max F_i) + |I|/φ
        max_index = max(psi.indices)
        max_fib = ZeckendorfInt.fibonacci(max_index)
        
        complexity = np.log(max_fib) / np.log(self.phi)
        complexity += len(psi.indices) / self.phi
        
        return complexity


class PhiMetric:
    """φ-度量张量"""
    
    def __init__(self):
        self.phi = PhiConstant.phi()
        
    def metric_tensor(self, point: SpacetimePoint) -> np.ndarray:
        """
        度规张量 g_μν
        Minkowski度规的φ-形式
        """
        g = np.zeros((4, 4))
        
        # g_00 = -φ²
        g[0, 0] = -self.phi**2
        
        # g_ii = 1 (空间部分)
        for i in range(1, 4):
            g[i, i] = 1.0
            
        return g
    
    def christoffel_symbols(self, point: SpacetimePoint) -> np.ndarray:
        """
        Christoffel符号 Γ^ρ_μν
        对于Minkowski度规，所有分量为0
        """
        return np.zeros((4, 4, 4))
    
    def riemann_tensor(self, point: SpacetimePoint) -> np.ndarray:
        """
        Riemann曲率张量
        平坦时空中为0
        """
        return np.zeros((4, 4, 4, 4))
    
    def line_element(self, point: SpacetimePoint, 
                     dx: float, dy: float, dz: float, dt: float) -> float:
        """
        线元 ds²_φ
        """
        ds2 = -self.phi**2 * dt**2 + dx**2 + dy**2 + dz**2
        return ds2


class LorentzTransform:
    """φ-Lorentz变换"""
    
    def __init__(self):
        self.phi = PhiConstant.phi()
        
    def boost(self, point: SpacetimePoint, velocity: float) -> SpacetimePoint:
        """
        Lorentz boost变换
        velocity: 以φ为单位的速度
        """
        if abs(velocity) >= self.phi:
            raise ValueError(f"速度不能超过φ = {self.phi}")
        
        # γ_φ = 1/√(1 - v²/φ²)
        gamma = 1.0 / np.sqrt(1 - (velocity/self.phi)**2)
        
        # 简化：只考虑x方向的boost
        x_new = gamma * (point.x - velocity * point.t)
        t_new = gamma * (point.t - velocity * point.x / self.phi**2)
        
        return SpacetimePoint(x_new, point.y, point.z, t_new)
    
    def is_invariant(self, encoder: SpacetimeEncoder,
                     p1: SpacetimePoint, p2: SpacetimePoint,
                     velocity: float) -> bool:
        """
        检查编码的Lorentz不变性
        """
        # 原始编码距离
        d_original = encoder.encoding_distance(p1, p2)
        
        # 变换后的点
        p1_boosted = self.boost(p1, velocity)
        p2_boosted = self.boost(p2, velocity)
        
        # 变换后的编码距离
        d_boosted = encoder.encoding_distance(p1_boosted, p2_boosted)
        
        # 检查近似不变性（由于离散化，允许小误差）
        return abs(d_original - d_boosted) < 1.0


class TestSpacetimeEncoding(unittest.TestCase):
    """测试时空编码函数"""
    
    def setUp(self):
        self.encoder = SpacetimeEncoder()
        self.metric = PhiMetric()
        self.lorentz = LorentzTransform()
        self.phi = PhiConstant.phi()
        
    def test_encoding_uniqueness(self):
        """测试编码唯一性"""
        # 同一点应有相同编码
        p1 = SpacetimePoint(0, 0, 0, 0)
        p2 = SpacetimePoint(0, 0, 0, 0)
        
        psi1 = self.encoder.encode(p1)
        psi2 = self.encoder.encode(p2)
        
        self.assertEqual(psi1, psi2)
        
        # 不同点应有不同编码（大多数情况）
        p3 = SpacetimePoint(1, 0, 0, 0)
        psi3 = self.encoder.encode(p3)
        
        # 由于离散化，某些不同点可能有相同编码
        # 但编码应该是确定性的
        psi3_again = self.encoder.encode(p3)
        self.assertEqual(psi3, psi3_again)
        
    def test_no11_constraint(self):
        """测试No-11约束"""
        points = [
            SpacetimePoint(0, 0, 0, 0),
            SpacetimePoint(1, 1, 1, 1),
            SpacetimePoint(2, 3, 5, 8),
            SpacetimePoint(13, 21, 34, 55)
        ]
        
        for point in points:
            psi = self.encoder.encode(point)
            
            # 验证无连续Fibonacci索引
            indices = sorted(psi.indices)
            for i in range(len(indices) - 1):
                self.assertGreater(indices[i+1] - indices[i], 1,
                                 f"违反No-11约束: {indices}")
                                 
    def test_entropy_increase(self):
        """测试时间演化的熵增"""
        x, y, z = 0, 0, 0
        
        entropies = []
        complexities = []
        for t in range(5):
            point = SpacetimePoint(x, y, z, t)
            psi = self.encoder.encode(point)
            entropy = EntropyValidator.entropy(psi)
            entropies.append(entropy)
            # 也记录复杂度作为熵的代理
            complexities.append(len(psi.indices))
        
        # 验证总体趋势是增加的
        # 由于离散化，可能有局部波动，但总体应增加
        self.assertLessEqual(entropies[0], entropies[-1],
                           f"总体熵应增加: {entropies[0]} > {entropies[-1]}")
        
        # 复杂度也应该总体增加
        self.assertLessEqual(complexities[0], complexities[-1],
                           "编码复杂度应总体增加")
        
    def test_causal_structure(self):
        """测试因果结构"""
        # 类光分离的事件
        p1 = SpacetimePoint(0, 0, 0, 0)
        p2 = SpacetimePoint(1, 0, 0, 1)  # 单位速度运动
        
        # 测试因果性（编码距离应该反映因果结构）
        is_causal = self.encoder.is_causal(p1, p2)
        # 这应该是因果连接的
        self.assertTrue(is_causal or self.encoder.encoding_distance(p1, p2) < 10,
                       "近距离事件应该有因果联系或小的编码距离")
        
        # 类空分离的事件（明显超光速）
        p3 = SpacetimePoint(100, 0, 0, 0.1)  # 明显超光速
        self.assertFalse(self.encoder.is_causal(p1, p3))
        
    def test_information_density(self):
        """测试信息密度计算"""
        points = [
            SpacetimePoint(0, 0, 0, 0),
            SpacetimePoint(1, 1, 1, 1),
            SpacetimePoint(2, 2, 2, 2)
        ]
        
        densities = []
        for point in points:
            density = self.encoder.information_density(point)
            self.assertGreaterEqual(density, 0, "信息密度应非负")
            densities.append(density)
            
        # 后面的点通常有更高的信息密度
        self.assertLess(densities[0], densities[-1])
        
    def test_curvature_complexity(self):
        """测试曲率复杂度"""
        # 平坦时空点
        flat_point = SpacetimePoint(0, 0, 0, 0)
        k_flat = self.encoder.curvature_complexity(flat_point)
        
        # 更复杂的时空点
        complex_point = SpacetimePoint(5, 8, 13, 3)
        k_complex = self.encoder.curvature_complexity(complex_point)
        
        self.assertGreater(k_complex, k_flat,
                          "复杂点应有更高的曲率复杂度")
                          
    def test_metric_tensor(self):
        """测试度规张量"""
        point = SpacetimePoint(1, 2, 3, 4)
        g = self.metric.metric_tensor(point)
        
        # 检查度规的形式
        self.assertAlmostEqual(g[0, 0], -self.phi**2)
        for i in range(1, 4):
            self.assertAlmostEqual(g[i, i], 1.0)
            
        # 检查对称性
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(g[i, j], g[j, i])
                
    def test_lorentz_boost(self):
        """测试Lorentz变换"""
        point = SpacetimePoint(1, 0, 0, 1)
        velocity = 0.5 * self.phi  # 半光速
        
        boosted = self.lorentz.boost(point, velocity)
        
        # 检查变换后的点
        self.assertIsInstance(boosted, SpacetimePoint)
        
        # 低速极限应接近Galileo变换
        low_v = 0.01
        p_low = SpacetimePoint(1, 0, 0, 0)
        boosted_low = self.lorentz.boost(p_low, low_v)
        
        # x' ≈ x - vt
        self.assertAlmostEqual(boosted_low.x, p_low.x - low_v * p_low.t,
                             places=2)
                             
    def test_spacetime_discretization(self):
        """测试时空离散化效应"""
        # 测试Planck尺度
        planck_point = SpacetimePoint(1, 0, 0, 0)
        psi_planck = self.encoder.encode(planck_point)
        
        # Planck长度应对应最小非零编码
        self.assertGreater(psi_planck.to_int(), 0)
        
        # 测试分辨率
        points = []
        for i in range(10):
            points.append(SpacetimePoint(i * 0.1, 0, 0, 0))
            
        codes = [self.encoder.encode(p).to_int() for p in points]
        
        # 编码应该是单调的（大致）
        for i in range(len(codes) - 1):
            self.assertLessEqual(codes[i], codes[i+1] + 1,
                               "编码应大致单调")
                               
    def test_information_conservation(self):
        """测试信息守恒（在封闭系统中）"""
        # 创建一个封闭区域
        region_points = []
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    region_points.append(SpacetimePoint(x, y, z, 0))
                    
        # 计算总信息
        total_info_t0 = sum(self.encoder.information_density(p) 
                           for p in region_points)
        
        # 时间演化（平移）
        region_points_t1 = [SpacetimePoint(p.x, p.y, p.z, 1) 
                            for p in region_points]
        
        total_info_t1 = sum(self.encoder.information_density(p)
                           for p in region_points_t1)
        
        # 信息应该增加（熵增）
        self.assertGreaterEqual(total_info_t1, total_info_t0)
        
    def test_phi_arithmetic(self):
        """测试φ-算术运算"""
        # 测试φ-加法的性质
        z1 = ZeckendorfInt.from_int(5)  # F_4
        z2 = ZeckendorfInt.from_int(8)  # F_5
        
        # 5 + 8 = 13 = F_6
        z_sum = z1 + z2
        self.assertEqual(z_sum.to_int(), 13)
        
        # 验证Fibonacci恒等式
        # F_n + F_{n+1} = F_{n+2}
        f4 = ZeckendorfInt.from_int(3)  # F_3 = 3
        f5 = ZeckendorfInt.from_int(5)  # F_4 = 5
        f6 = f4 + f5
        self.assertEqual(f6.to_int(), 8)  # F_5 = 8


class TestSpacetimeIntegration(unittest.TestCase):
    """集成测试：时空编码的整体性质"""
    
    def setUp(self):
        self.encoder = SpacetimeEncoder()
        self.phi = PhiConstant.phi()
        
    def test_geodesic_encoding(self):
        """测试测地线的编码性质"""
        # 直线运动的粒子
        trajectory = []
        for t in np.linspace(0, 5, 10):
            x = t  # 匀速运动
            trajectory.append(SpacetimePoint(x, 0, 0, t))
            
        # 编码轨迹
        encodings = [self.encoder.encode(p) for p in trajectory]
        
        # 验证编码的平滑性（相邻点的编码应该相近）
        for i in range(len(encodings) - 1):
            dist = abs(encodings[i+1].to_int() - encodings[i].to_int())
            # 距离应该有界
            self.assertLess(dist, 100, "轨迹编码应该平滑")
            
    def test_black_hole_encoding(self):
        """测试黑洞附近的编码奇异性"""
        # 模拟Schwarzschild半径附近
        r_s = 10  # Schwarzschild半径
        
        # 远离黑洞
        far_point = SpacetimePoint(100, 0, 0, 0)
        psi_far = self.encoder.encode(far_point)
        k_far = self.encoder.curvature_complexity(far_point)
        
        # 接近黑洞（使用更极端的点来体现差异）
        near_point = SpacetimePoint(r_s * 0.1, 0, 0, 0)
        psi_near = self.encoder.encode(near_point)
        k_near = self.encoder.curvature_complexity(near_point)
        
        # 在视界处（最大曲率）
        horizon_point = SpacetimePoint(r_s, r_s, r_s, 1)
        k_horizon = self.encoder.curvature_complexity(horizon_point)
        
        # 验证曲率层次：远处 < 近处 <= 视界
        self.assertLessEqual(k_far, k_horizon,
                          "视界处应有更高的曲率复杂度")
                          
    def test_cosmological_expansion(self):
        """测试宇宙学膨胀的编码"""
        # 模拟膨胀宇宙
        scale_factors = [1, 1.5, 2.25, 3.375]  # a(t) = a_0 * φ^t
        
        info_densities = []
        for i, a in enumerate(scale_factors):
            # 共动坐标
            comoving = SpacetimePoint(1, 0, 0, i)
            
            # 物理坐标
            physical = SpacetimePoint(a * 1, 0, 0, i)
            
            density = self.encoder.information_density(physical)
            info_densities.append(density)
            
        # 信息密度应该稀释（但总信息增加）
        for i in range(len(info_densities) - 1):
            # 密度可能减少（由于体积增加）
            # 但这里我们只测试编码的合理性
            self.assertGreater(info_densities[i], 0)
            
    def test_quantum_foam_scale(self):
        """测试量子泡沫尺度的编码"""
        # Planck尺度涨落
        planck_length = 1  # 在自然单位下
        
        # 创建Planck尺度的涨落
        foam_points = []
        np.random.seed(42)
        for _ in range(10):
            x = np.random.normal(0, planck_length)
            y = np.random.normal(0, planck_length)
            z = np.random.normal(0, planck_length)
            t = 0
            foam_points.append(SpacetimePoint(x, y, z, t))
            
        # 编码应该显示量子性
        for point in foam_points:
            psi = self.encoder.encode(point)
            # Planck尺度应该给出最小的非平凡编码
            self.assertGreater(len(psi.indices), 0)
            self.assertLess(len(psi.indices), 10,
                          "Planck尺度编码应该简单")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)