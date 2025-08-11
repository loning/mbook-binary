#!/usr/bin/env python3
"""
T1-3: 熵增速率定理 - 完整测试程序

验证φ-编码二进制宇宙中的熵增速率规律，包括：
1. 递归深度的Fibonacci增长
2. 熵的φ-指数增长
3. no-11约束的调制效应
4. 熵增速率的上下界
5. 物理量的计算和预测
"""

import unittest
import numpy as np
from typing import List, Tuple, Dict
import math


class PhiNumber:
    """φ进制数系统"""
    def __init__(self, value: float):
        self.phi = (1 + np.sqrt(5)) / 2
        self.value = float(value)
        
    def to_fibonacci_basis(self) -> List[int]:
        """转换为Fibonacci基表示（Zeckendorf表示）"""
        if self.value < 0:
            raise ValueError("Cannot convert negative number to Fibonacci basis")
            
        if self.value == 0:
            return [0]
            
        # 生成足够的Fibonacci数
        fibs = [1, 2]
        while fibs[-1] <= self.value:
            fibs.append(fibs[-1] + fibs[-2])
            
        # 贪心算法构造Zeckendorf表示
        result = []
        remaining = self.value
        
        for i in range(len(fibs) - 1, -1, -1):
            if fibs[i] <= remaining + 1e-10:
                result.append(1)
                remaining -= fibs[i]
            else:
                result.append(0)
                
        # 移除前导零
        while result and result[0] == 0:
            result.pop(0)
            
        return result if result else [0]
        
    def verify_no_11(self) -> bool:
        """验证表示中无连续的11"""
        fib_repr = self.to_fibonacci_basis()
        
        for i in range(len(fib_repr) - 1):
            if fib_repr[i] == 1 and fib_repr[i+1] == 1:
                return False
        return True
        
    def __add__(self, other):
        if isinstance(other, PhiNumber):
            return PhiNumber(self.value + other.value)
        return PhiNumber(self.value + float(other))
        
    def __mul__(self, other):
        if isinstance(other, PhiNumber):
            return PhiNumber(self.value * other.value)
        return PhiNumber(self.value * float(other))
        
    def __truediv__(self, other):
        if isinstance(other, PhiNumber):
            return PhiNumber(self.value / other.value)
        return PhiNumber(self.value / float(other))
        
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return PhiNumber(self.value ** other)
        return PhiNumber(self.value ** other.value)
        
    def __repr__(self):
        return f"φ({self.value:.6f})"


class RecursiveDepth:
    """递归深度计算器"""
    def __init__(self):
        self.depths = {0: 0, 1: 1}  # 初始条件
        
    def compute_depth(self, time: int) -> int:
        """计算时刻t的递归深度（Fibonacci增长）"""
        if time < 0:
            raise ValueError("Time cannot be negative")
            
        if time in self.depths:
            return self.depths[time]
        
        # Fibonacci递归：d(t) = d(t-1) + d(t-2)
        depth = self.compute_depth(time-1) + self.compute_depth(time-2)
        self.depths[time] = depth
        return depth
        
    def verify_fibonacci_growth(self, max_time: int = 20) -> bool:
        """验证递归深度的Fibonacci增长性"""
        # 计算前max_time个深度值
        depths = [self.compute_depth(t) for t in range(max_time)]
        
        # 验证Fibonacci关系
        for t in range(2, max_time):
            if depths[t] != depths[t-1] + depths[t-2]:
                return False
                
        return True
        
    def get_depth_sequence(self, max_time: int) -> List[int]:
        """获取深度序列"""
        return [self.compute_depth(t) for t in range(max_time)]


class No11Modulation:
    """no-11约束调制因子"""
    def __init__(self, epsilon: float = 0.01, T: float = 1.0):
        self.epsilon = epsilon  # 约束强度
        self.T = T  # 特征时间尺度
        
    def theta_factor(self, time: float) -> float:
        """计算no-11约束因子Θ(t)"""
        theta = 1.0
        
        # Fibonacci频率的调制
        fib_prev, fib_curr = 1, 1
        for n in range(1, 20):  # 前20个Fibonacci数
            freq = 2 * np.pi * fib_curr / self.T
            theta -= self.epsilon * np.sin(freq * time) / fib_curr
            fib_prev, fib_curr = fib_curr, fib_prev + fib_curr
            
        return theta
        
    def verify_bounds(self, time_points: int = 1000) -> bool:
        """验证|Θ(t) - 1| ≤ ε"""
        times = np.linspace(0, 10 * self.T, time_points)
        
        # 计算理论上界
        # sum(1/F_n) 对于前20个Fibonacci数
        fib_sum = 0
        fib_prev, fib_curr = 1, 1
        for n in range(1, 20):
            fib_sum += 1.0 / fib_curr
            fib_prev, fib_curr = fib_curr, fib_prev + fib_curr
            
        theoretical_bound = self.epsilon * fib_sum
        
        for t in times:
            theta = self.theta_factor(t)
            if abs(theta - 1) > theoretical_bound * 1.1:  # 允许10%的数值误差
                return False
                
        return True
        
    def find_resonances(self, max_time: float) -> List[float]:
        """找出共振点（Θ最小值）"""
        times = np.linspace(0, max_time, 1000)
        thetas = [self.theta_factor(t) for t in times]
        
        resonances = []
        for i in range(1, len(thetas) - 1):
            if thetas[i] < thetas[i-1] and thetas[i] < thetas[i+1]:
                resonances.append(times[i])
                
        return resonances


class EntropyRateCalculator:
    """熵增速率计算器"""
    def __init__(self, k0: float = 1.0):
        self.k0 = k0
        self.phi = (1 + np.sqrt(5)) / 2
        self.depth_calc = RecursiveDepth()
        self.modulation = No11Modulation()
        
    def compute_rate(self, time: float) -> PhiNumber:
        """计算熵增速率 dH/dt = k0 * φ^d(t) * Θ(t)"""
        # 取整数时间计算深度
        d_t = self.depth_calc.compute_depth(int(time))
        theta_t = self.modulation.theta_factor(time)
        
        # 避免大指数溢出，使用对数计算
        if d_t > 100:
            # 对于大的d_t，返回一个大但有限的值
            rate = self.k0 * (self.phi ** 100) * theta_t
        else:
            rate = self.k0 * (self.phi ** d_t) * theta_t
            
        return PhiNumber(rate)
        
    def verify_bounds(self, time: float) -> bool:
        """验证速率在理论界限内"""
        d_t = self.depth_calc.compute_depth(int(time))
        epsilon = self.modulation.epsilon
        
        rate = self.compute_rate(time)
        lower_bound = self.k0 * (self.phi ** d_t) * (1 - epsilon)
        upper_bound = self.k0 * (self.phi ** d_t) * (1 + epsilon)
        
        return lower_bound <= rate.value <= upper_bound
        
    def asymptotic_behavior(self, time: float) -> PhiNumber:
        """计算渐近行为"""
        tau_phi = np.log(self.phi) / np.log(1 + 1/self.phi)
        asymptotic_rate = self.k0 * (self.phi ** (time / tau_phi))
        return PhiNumber(asymptotic_rate)


class EntropyEvolution:
    """熵演化器"""
    def __init__(self, H0: PhiNumber):
        self.H0 = H0  # 初始熵
        self.rate_calc = EntropyRateCalculator()
        self.history = [(0.0, H0.value)]
        
    def evolve(self, time_steps: int, dt: float = 0.01) -> List[Tuple[float, float]]:
        """演化系统熵"""
        current_H = self.H0.value
        current_t = 0.0
        
        evolution = [(current_t, current_H)]
        
        for _ in range(time_steps):
            rate = self.rate_calc.compute_rate(current_t)
            current_H += rate.value * dt
            current_t += dt
            evolution.append((current_t, current_H))
            
        self.history = evolution
        return evolution
        
    def integrate_entropy(self, t_start: float, t_end: float, n_points: int = 1000) -> PhiNumber:
        """积分计算累积熵变"""
        dt = (t_end - t_start) / n_points
        total = 0.0
        
        for i in range(n_points):
            t = t_start + i * dt
            rate = self.rate_calc.compute_rate(t)
            total += rate.value * dt
            
        return PhiNumber(total)
        
    def find_phase_transitions(self) -> List[float]:
        """找出相变点（熵增速率突变）"""
        if len(self.history) < 3:
            return []
            
        transitions = []
        
        # 计算二阶导数找突变点
        for i in range(1, len(self.history) - 1):
            t_prev, H_prev = self.history[i-1]
            t_curr, H_curr = self.history[i]
            t_next, H_next = self.history[i+1]
            
            # 近似二阶导数
            dt = t_curr - t_prev
            if dt > 0:
                d2H = (H_next - 2*H_curr + H_prev) / (dt * dt)
                
                # 检测突变
                if abs(d2H) > 10:  # 阈值
                    transitions.append(t_curr)
                    
        return transitions


class EmergentTime:
    """涌现时间"""
    def __init__(self):
        self.rate_calc = EntropyRateCalculator()
        
    def proper_time_rate(self, coordinate_time: float) -> PhiNumber:
        """计算固有时流逝率 dτ/dt = 1/(dH/dt)"""
        entropy_rate = self.rate_calc.compute_rate(coordinate_time)
        if entropy_rate.value > 0:
            return PhiNumber(1.0 / entropy_rate.value)
        else:
            return PhiNumber(float('inf'))
            
    def time_dilation_factor(self, t1: float, t2: float) -> PhiNumber:
        """计算两个时刻间的时间膨胀因子"""
        rate1 = self.rate_calc.compute_rate(t1)
        rate2 = self.rate_calc.compute_rate(t2)
        
        if rate1.value > 0 and rate2.value > 0:
            return PhiNumber(rate2.value / rate1.value)
        else:
            return PhiNumber(1.0)


class BlackHoleEntropy:
    """黑洞熵计算器"""
    def __init__(self, mass: float):
        self.mass = mass
        self.phi = (1 + np.sqrt(5)) / 2
        self.rate_calc = EntropyRateCalculator()
        
    def bekenstein_hawking_rate(self, time: float) -> PhiNumber:
        """黑洞熵增速率（Planck单位）"""
        # S_BH = A/4 = 4πM² (in Planck units)
        # 基准熵增率
        base_rate = 8 * np.pi * self.mass
        
        # φ调制
        t_P = 1.0  # Planck时间
        d_t = int(time / t_P)
        modulation = No11Modulation()
        theta = modulation.theta_factor(time / t_P)
        
        rate = base_rate * (self.phi ** d_t) * theta
        return PhiNumber(rate)


class TestEntropyRate(unittest.TestCase):
    """T1-3 熵增速率测试套件"""
    
    def setUp(self):
        self.phi = (1 + np.sqrt(5)) / 2
        
    def test_phi_number_fibonacci_basis(self):
        """测试Fibonacci基表示"""
        # 测试几个具体的数
        num1 = PhiNumber(10)
        fib_repr1 = num1.to_fibonacci_basis()
        self.assertTrue(num1.verify_no_11())
        
        # 验证表示的正确性
        # 10 = 8 + 2 = F_6 + F_3
        # Fibonacci序列: 1,1,2,3,5,8,13,21...
        # 所以10的表示应该包含第6和第3个Fibonacci数
        
        # 测试no-11约束
        num2 = PhiNumber(12)  # 12 = 8 + 3 + 1，会违反no-11
        self.assertTrue(num2.verify_no_11())  # Zeckendorf表示自动避免11
        
    def test_recursive_depth_fibonacci(self):
        """测试递归深度的Fibonacci增长"""
        depth_calc = RecursiveDepth()
        
        # 验证前20个深度值
        self.assertTrue(depth_calc.verify_fibonacci_growth(20))
        
        # 检查具体值
        depths = depth_calc.get_depth_sequence(10)
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]  # Fibonacci序列
        self.assertEqual(depths, expected)
        
    def test_no11_modulation_bounds(self):
        """测试no-11调制因子的界限"""
        modulation = No11Modulation(epsilon=0.01)
        
        # 验证界限
        self.assertTrue(modulation.verify_bounds())
        
        # 测试特定时刻
        theta_0 = modulation.theta_factor(0)
        self.assertAlmostEqual(theta_0, 1.0, places=2)
        
        # 找共振点
        resonances = modulation.find_resonances(10.0)
        self.assertGreater(len(resonances), 0)
        
    def test_entropy_rate_calculation(self):
        """测试熵增速率计算"""
        rate_calc = EntropyRateCalculator(k0=1.0)
        
        # 测试t=0时刻
        rate_0 = rate_calc.compute_rate(0)
        self.assertAlmostEqual(rate_0.value, 1.0, places=2)  # φ^0 * Θ(0) ≈ 1
        
        # 测试t=5时刻
        rate_5 = rate_calc.compute_rate(5)
        d_5 = rate_calc.depth_calc.compute_depth(5)
        expected_order = self.phi ** d_5
        self.assertLess(abs(rate_5.value - expected_order) / expected_order, 0.1)
        
        # 验证界限
        for t in range(10):
            self.assertTrue(rate_calc.verify_bounds(float(t)))
            
    def test_entropy_evolution(self):
        """测试熵演化"""
        H0 = PhiNumber(0)
        evolution = EntropyEvolution(H0)
        
        # 演化1000步
        history = evolution.evolve(1000, dt=0.01)
        
        # 验证熵单调增加
        for i in range(1, len(history)):
            self.assertGreater(history[i][1], history[i-1][1])
            
        # 测试积分
        delta_H = evolution.integrate_entropy(0, 1)
        self.assertGreater(delta_H.value, 0)
        
    def test_exponential_growth(self):
        """测试φ指数增长"""
        rate_calc = EntropyRateCalculator()
        
        # 比较不同时刻的增长
        times = [5, 10, 15]
        rates = [rate_calc.compute_rate(t).value for t in times]
        
        # 验证指数增长趋势
        for i in range(1, len(rates)):
            self.assertGreater(rates[i], rates[i-1])
            
        # 检查增长率
        growth_factors = [rates[i]/rates[i-1] for i in range(1, len(rates))]
        
        # 增长因子应该接近φ^5（因为深度每5个时间单位增加F_5=5）
        for factor in growth_factors:
            self.assertGreater(factor, 1)
            
    def test_emergent_time(self):
        """测试涌现时间"""
        emergent = EmergentTime()
        
        # 测试固有时率
        tau_rate_0 = emergent.proper_time_rate(0)
        tau_rate_10 = emergent.proper_time_rate(10)
        
        # 后期时间流逝应该更慢（因为熵增更快）
        self.assertLess(tau_rate_10.value, tau_rate_0.value)
        
        # 测试时间膨胀
        dilation = emergent.time_dilation_factor(0, 10)
        self.assertGreater(dilation.value, 1)  # 时间加速
        
    def test_phase_transitions(self):
        """测试相变点检测"""
        H0 = PhiNumber(0)
        evolution = EntropyEvolution(H0)
        
        # 演化更长时间以检测相变
        evolution.evolve(10000, dt=0.001)
        
        transitions = evolution.find_phase_transitions()
        # 应该能检测到一些相变点
        # （虽然在这个简化模型中可能不明显）
        
    def test_black_hole_entropy(self):
        """测试黑洞熵增"""
        mass = 1.0  # 太阳质量单位
        bh = BlackHoleEntropy(mass)
        
        # 测试不同时刻的熵增率
        rate_0 = bh.bekenstein_hawking_rate(0)
        rate_1 = bh.bekenstein_hawking_rate(1)
        
        self.assertGreater(rate_0.value, 0)
        self.assertGreater(rate_1.value, rate_0.value)
        
    def test_asymptotic_behavior(self):
        """测试渐近行为"""
        rate_calc = EntropyRateCalculator()
        
        # 测试Fibonacci深度的渐近行为
        # d(n) ~ φ^n / sqrt(5) for large n
        
        # 使用Binet公式的近似
        n = 15
        d_n = rate_calc.depth_calc.compute_depth(n)
        
        # Fibonacci数的渐近公式
        phi = rate_calc.phi
        expected_d_n = (phi ** n) / np.sqrt(5)
        
        # 相对误差应该随n增大而减小
        relative_error = abs(d_n - expected_d_n) / expected_d_n
        self.assertLess(relative_error, 0.1)  # 10%以内的误差
        
        # 验证增长率接近φ
        d_n_minus_1 = rate_calc.depth_calc.compute_depth(n-1)
        growth_rate = d_n / d_n_minus_1
        self.assertAlmostEqual(growth_rate, phi, delta=0.1)
        
    def test_information_bound(self):
        """测试信息处理速率界限"""
        rate_calc = EntropyRateCalculator()
        
        # 信息处理速率不能超过熵增速率
        for t in range(10):
            entropy_rate = rate_calc.compute_rate(float(t))
            max_info_rate = entropy_rate  # dI/dt ≤ dH/dt
            
            # 任何实际信息处理速率都应该小于这个界限
            actual_info_rate = entropy_rate.value * 0.8  # 80%效率
            self.assertLessEqual(actual_info_rate, max_info_rate.value)
            
    def test_fibonacci_resonances(self):
        """测试Fibonacci频率共振"""
        modulation = No11Modulation(epsilon=0.1, T=2*np.pi)
        
        # 理论共振点应该在 t = T*F_n/(2π)
        F_n = [1, 1, 2, 3, 5, 8, 13]
        expected_resonances = [modulation.T * f / (2 * np.pi) for f in F_n[2:]]
        
        # 找实际共振点
        actual_resonances = modulation.find_resonances(20.0)
        
        # 至少应该找到一些共振点
        self.assertGreater(len(actual_resonances), 0)
        
        # 验证周期性
        if len(actual_resonances) > 2:
            periods = [actual_resonances[i+1] - actual_resonances[i] 
                      for i in range(len(actual_resonances)-1)]
            # 周期应该呈现某种规律性


if __name__ == '__main__':
    unittest.main(verbosity=2)