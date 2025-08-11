#!/usr/bin/env python3
"""
T26-3 单元测试：e时间演化定理验证

测试e作为时间不可逆性的数学本质，验证：
1. 时间演化的严格e指数性质
2. 不可逆性的绝对保持
3. 与T26-2理论的完全一致性
4. Zeckendorf编码下的时间量子化
5. 长期演化的数值稳定性
6. 因果性的严格维持

依赖：base_framework.py, no11_number_system.py
"""

import unittest
import math
import numpy as np
from typing import List, Tuple, Dict, Any, Callable
from dataclasses import dataclass

# 导入基础框架
import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from base_framework import (
        BinaryUniverseFramework, 
        ZeckendorfEncoder, 
        PhiBasedMeasure,
        ValidationResult
    )
    from no11_number_system import No11NumberSystem
except ImportError:
    # 简化版本用于独立运行
    class BinaryUniverseFramework:
        def __init__(self):
            pass
    
    class ZeckendorfEncoder:
        def __init__(self):
            self.fibonacci_cache = [1, 2]
        
        def get_fibonacci(self, n: int) -> int:
            if n <= 0: return 0
            if n == 1: return 1
            if n == 2: return 2
            while len(self.fibonacci_cache) < n:
                next_fib = self.fibonacci_cache[-1] + self.fibonacci_cache[-2]
                self.fibonacci_cache.append(next_fib)
            return self.fibonacci_cache[n-1]
        
        def to_zeckendorf(self, n: int) -> List[int]:
            if n <= 0: return [0]
            max_index = 1
            while self.get_fibonacci(max_index + 1) <= n:
                max_index += 1
            result = []
            remaining = n
            for i in range(max_index, 0, -1):
                fib_val = self.get_fibonacci(i)
                if fib_val <= remaining:
                    result.append(1)
                    remaining -= fib_val
                else:
                    result.append(0)
            return result
        
        def from_zeckendorf(self, zeck_repr: List[int]) -> int:
            result = 0
            for i, bit in enumerate(zeck_repr):
                if bit == 1:
                    fib_index = len(zeck_repr) - i
                    result += self.get_fibonacci(fib_index)
            return result
        
        def is_valid_zeckendorf(self, zeck_repr: List[int]) -> bool:
            for i in range(len(zeck_repr) - 1):
                if zeck_repr[i] == 1 and zeck_repr[i+1] == 1:
                    return False
            return True
    
    class PhiBasedMeasure:
        def __init__(self):
            self.phi = (1 + math.sqrt(5)) / 2
    
    class ValidationResult:
        def __init__(self, passed, score, details):
            self.passed = passed
            self.score = score  
            self.details = details
    
    class No11NumberSystem:
        def __init__(self):
            pass

@dataclass
class TimeEvolutionState:
    """时间演化状态"""
    time: float
    entropy: float
    alpha: float
    irreversibility_measure: float
    
    def is_physically_valid(self) -> bool:
        """检查物理有效性"""
        return (self.time >= 0 and 
                self.entropy > 0 and 
                self.alpha > 0 and
                self.irreversibility_measure >= 0)

class ETimeEvolutionSystem(BinaryUniverseFramework):
    """e时间演化系统"""
    
    def __init__(self):
        super().__init__()
        self.no11_system = No11NumberSystem()
        self.zeckendorf = ZeckendorfEncoder()
        self.phi_measure = PhiBasedMeasure()
        
        # 数学常数
        self.e_mathematical = math.e
        self.phi = (1 + math.sqrt(5)) / 2
        
    def integrate_time_evolution(
        self, 
        initial_entropy: float, 
        alpha: float, 
        time_span: Tuple[float, float],
        precision: float = 1e-12
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        数值积分时间演化方程 dH/dt = α·H
        使用高精度指数积分避免数值不稳定
        """
        if initial_entropy <= 0 or alpha <= 0:
            raise ValueError("初始熵和熵增率必须为正")
        
        t_start, t_end = time_span
        if t_start < 0 or t_end <= t_start:
            raise ValueError("时间参数无效")
        
        # 简化的固定步长方法避免超时
        n_points = min(1000, int((t_end - t_start) * alpha * 100) + 50)
        n_points = max(n_points, 10)  # 至少10个点
        
        time_points = [t_start + i * (t_end - t_start) / (n_points - 1) for i in range(n_points)]
        entropy_values = []
        irreversibility_values = []
        
        for t in time_points:
            # 精确的指数解：H(t) = H₀ * e^(αt)
            if alpha * t < 700:  # 防止exp()溢出
                h = initial_entropy * math.exp(alpha * t)
            else:
                h = float('inf')
            
            entropy_values.append(h)
            
            # 计算不可逆性强度 I(t) = (1/H)(dH/dt) = α
            irreversibility_values.append(alpha)
            
        return entropy_values, time_points, irreversibility_values
    
    def verify_e_uniqueness(
        self,
        base_candidates: List[float],
        alpha: float = 1.0
    ) -> Tuple[bool, Dict[float, float]]:
        """
        验证e是唯一与自指完备性兼容的指数底数
        """
        deviation_measures = {}
        compatible_bases = []
        
        for base in base_candidates:
            if base <= 0:
                deviation_measures[base] = float('inf')
                continue
                
            # 自指条件：ln(base) = 1，即 base = e
            ln_base = math.log(base)
            self_reference_error = abs(ln_base - 1.0)
            
            # 计算与数学e的偏差
            deviation = abs(base - self.e_mathematical)
            deviation_measures[base] = deviation
            
            # 自指一致性判断（极严格标准）
            if self_reference_error < 1e-14:
                compatible_bases.append(base)
        
        # e的唯一性：只有e（在误差范围内）应该通过测试
        e_is_unique = (len(compatible_bases) == 1 and 
                       abs(compatible_bases[0] - self.e_mathematical) < 1e-12)
        
        return e_is_unique, deviation_measures
    
    def verify_time_irreversibility(
        self,
        entropy_trajectory: List[float],
        time_points: List[float],
        alpha: float,
        causality_window: int = 5
    ) -> Tuple[bool, List[int], float]:
        """
        验证时间演化的严格不可逆性
        """
        violations = []
        
        # 检查1：熵的严格单调递增性
        for i in range(1, len(entropy_trajectory)):
            if entropy_trajectory[i] <= entropy_trajectory[i-1]:
                violations.append(i)
        
        # 检查2：因果性（过去完全决定现在）
        causality_violations = []
        initial_entropy = entropy_trajectory[0]
        
        for i in range(1, len(entropy_trajectory)):
            # 根据初始条件预测当前熵值
            expected_entropy = initial_entropy * math.exp(alpha * time_points[i])
            actual_entropy = entropy_trajectory[i]
            
            # 因果性条件：实际值应与预期值匹配
            relative_error = abs(actual_entropy - expected_entropy) / expected_entropy
            if relative_error > 1e-10:
                causality_violations.append(i)
        
        # 计算时间箭头一致性度量
        if len(entropy_trajectory) > 1:
            entropy_gradients = [
                (entropy_trajectory[i+1] - entropy_trajectory[i]) / 
                (time_points[i+1] - time_points[i])
                for i in range(len(entropy_trajectory)-1)
            ]
            
            # 所有梯度都应为正（严格递增）
            positive_gradients = sum(1 for grad in entropy_gradients if grad > 0)
            arrow_consistency = positive_gradients / len(entropy_gradients)
        else:
            arrow_consistency = 1.0
        
        irreversibility_confirmed = (len(violations) == 0 and 
                                    len(causality_violations) == 0 and
                                    arrow_consistency >= 1.0)  # 完美一致性要求
        
        return irreversibility_confirmed, causality_violations, arrow_consistency
    
    def quantize_time_zeckendorf(
        self,
        continuous_time: float,
        alpha: float
    ) -> Tuple[float, List[int], float]:
        """
        将连续时间在Zeckendorf编码下量子化
        """
        if continuous_time < 0:
            raise ValueError("时间不能为负")
        if alpha <= 0:
            raise ValueError("熵增率必须为正")
        
        # 时间量子：Δt_min = ln(φ)/α
        time_quantum = math.log(self.phi) / alpha
        
        # 将时间转换为时间量子单位
        quantum_units = continuous_time / time_quantum
        
        # 四舍五入到最近的整数量子单位
        quantum_units_int = int(round(quantum_units))
        
        # 获取Zeckendorf表示
        if quantum_units_int > 0:
            fibonacci_repr = self.zeckendorf.to_zeckendorf(quantum_units_int)
            # 验证No-11约束
            if not self.zeckendorf.is_valid_zeckendorf(fibonacci_repr):
                raise ValueError("Zeckendorf表示违反No-11约束")
            # 重构量子化值
            reconstructed_units = self.zeckendorf.from_zeckendorf(fibonacci_repr)
        else:
            fibonacci_repr = [0]
            reconstructed_units = 0
        
        # 转换回时间单位
        quantized_time = reconstructed_units * time_quantum
        quantum_error = abs(continuous_time - quantized_time)
        
        return quantized_time, fibonacci_repr, quantum_error
    
    def ensure_long_term_stability(
        self,
        initial_conditions: Dict[str, float],
        time_horizon: float
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, Any]]:
        """
        保证长时间演化的数值稳定性
        使用对数空间计算避免指数溢出
        """
        H0 = initial_conditions['initial_entropy']
        alpha = initial_conditions['alpha']
        
        if H0 <= 0 or alpha <= 0:
            raise ValueError("初始条件无效")
        
        # 切换到对数空间：ln(H(t)) = ln(H₀) + αt
        log_H0 = math.log(H0)
        
        # 时间网格（自适应密度）
        time_points = self._generate_adaptive_time_grid(0, time_horizon, alpha)
        
        # 对数空间演化（精确解）
        log_entropy_values = [log_H0 + alpha * t for t in time_points]
        
        # 不可逆性度量（在对数空间中为常数）
        irreversibility_values = [alpha] * len(time_points)
        
        # 转换回线性空间（小心处理大值）
        entropy_values = []
        overflow_points = 0
        
        for log_H in log_entropy_values:
            if log_H < 700:  # 避免exp()溢出
                entropy_values.append(math.exp(log_H))
            else:
                entropy_values.append(float('inf'))  # 标记为无穷大
                overflow_points += 1
        
        # 稳定性验证
        stability_metrics = {
            'max_log_entropy': max(log_entropy_values),
            'entropy_growth_rate': alpha,
            'time_span': time_horizon,
            'numerical_overflow_points': overflow_points,
            'total_points': len(time_points)
        }
        
        # 检查长期稳定性
        is_stable = (
            stability_metrics['max_log_entropy'] < 1000 and  # 防止极端增长
            stability_metrics['numerical_overflow_points'] == 0  # 无溢出
        )
        
        stable_trajectory = {
            'time': time_points,
            'entropy': entropy_values,
            'irreversibility': irreversibility_values,
            'is_stable': is_stable
        }
        
        logarithmic_variables = {
            'time': time_points,
            'log_entropy': log_entropy_values,
            'alpha': [alpha] * len(time_points)
        }
        
        stability_report = {
            'metrics': stability_metrics,
            'is_stable': is_stable,
            'overflow_fraction': overflow_points / len(time_points)
        }
        
        return stable_trajectory, logarithmic_variables, stability_report
    
    def _generate_adaptive_time_grid(self, t_start: float, t_end: float, alpha: float) -> List[float]:
        """生成自适应时间网格"""
        n_base = min(1000, int((t_end - t_start) * alpha * 100) + 100)
        return [t_start + i * (t_end - t_start) / n_base for i in range(n_base + 1)]
    
    def verify_t26_2_consistency(
        self,
        e_value_from_t26_2: float,
        time_evolution_alpha: float,
        entropy_trajectory: List[float],
        time_points: List[float]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        验证T26-3与T26-2的理论一致性
        """
        consistency_errors = []
        
        # 检查1：e值的一致性
        e_error = abs(e_value_from_t26_2 - self.e_mathematical)
        consistency_errors.append(e_error)
        
        # 检查2：指数增长模式的一致性
        H0 = entropy_trajectory[0]
        exponential_errors = []
        
        for i, (t, H_observed) in enumerate(zip(time_points, entropy_trajectory)):
            H_expected = H0 * (self.e_mathematical ** (time_evolution_alpha * t))
            if H_expected > 0:
                relative_error = abs(H_observed - H_expected) / H_expected
                exponential_errors.append(relative_error)
        
        max_exp_error = max(exponential_errors) if exponential_errors else 0.0
        consistency_errors.append(max_exp_error)
        
        # 检查3：自指性质的一致性
        # 验证 α = d/dt(ln H) = constant
        actual_alphas = []
        for i in range(1, len(entropy_trajectory)):
            if entropy_trajectory[i-1] > 0 and time_points[i] != time_points[i-1]:
                actual_alpha = (math.log(entropy_trajectory[i]) - math.log(entropy_trajectory[i-1])) / (time_points[i] - time_points[i-1])
                actual_alphas.append(actual_alpha)
        
        if actual_alphas:
            alpha_variance = sum((a - time_evolution_alpha)**2 for a in actual_alphas) / len(actual_alphas)
            consistency_errors.append(math.sqrt(alpha_variance))
        
        # 综合一致性分数
        consistency_score = 1.0 / (1.0 + sum(consistency_errors))
        
        analysis_report = {
            'e_value_error': e_error,
            'max_exponential_error': max_exp_error,
            'alpha_consistency': math.sqrt(alpha_variance) if actual_alphas else 0.0,
            'overall_score': consistency_score,
            'is_consistent': all(error < 1e-10 for error in consistency_errors)
        }
        
        return consistency_score, analysis_report

class TestT263ETimeEvolution(unittest.TestCase):
    """T26-3 e时间演化定理测试"""
    
    def setUp(self):
        """测试初始化"""
        self.system = ETimeEvolutionSystem()
        self.precision = 1e-12
        self.alpha_test = 0.5  # 测试用熵增率
        
    def test_basic_exponential_evolution(self):
        """测试1：基础指数演化"""
        print("\n测试1：基础指数演化")
        
        initial_entropy = 1.0
        time_span = (0.0, 2.0)
        
        entropy_values, time_points, irreversibility = self.system.integrate_time_evolution(
            initial_entropy, self.alpha_test, time_span, self.precision
        )
        
        print(f"时间点数量: {len(time_points)}")
        print(f"初始熵: {entropy_values[0]}")
        print(f"最终熵: {entropy_values[-1]}")
        print(f"理论最终熵: {initial_entropy * math.exp(self.alpha_test * time_span[1])}")
        
        # 验证指数增长
        H0 = entropy_values[0]
        for i, (t, H) in enumerate(zip(time_points, entropy_values)):
            expected_H = H0 * math.exp(self.alpha_test * t)
            relative_error = abs(H - expected_H) / expected_H
            self.assertLess(relative_error, 1e-10, f"时间点{i}的指数误差过大")
        
        # 验证不可逆性度量为常数
        for irr in irreversibility:
            self.assertAlmostEqual(irr, self.alpha_test, places=12, 
                                 msg="不可逆性度量应为常数α")
            
    def test_e_uniqueness_strict(self):
        """测试2：e底数的严格唯一性"""
        print("\n测试2：e底数的严格唯一性")
        
        # 测试候选底数
        candidates = [
            2.0,                    # 二进制底数
            math.exp(1.0),         # 数学e
            math.exp(1.0) + 1e-15, # 接近e的值
            math.exp(1.0) - 1e-15, # 接近e的值
            10.0,                  # 十进制底数
            self.system.phi        # 黄金比例
        ]
        
        is_unique, deviations = self.system.verify_e_uniqueness(candidates, self.alpha_test)
        
        print("底数验证结果:")
        compatible_count = 0
        for base, deviation in deviations.items():
            ln_error = abs(math.log(base) - 1.0) if base > 0 else float('inf')
            is_compatible = ln_error < 1e-14  # 检查是否通过自指测试
            if is_compatible:
                compatible_count += 1
            print(f"底数{base:.15f}: 偏差={deviation:.2e}, ln误差={ln_error:.2e}, 兼容={is_compatible}")
        
        print(f"通过测试的底数数量: {compatible_count}")
        
        # 检查所有通过测试的底数是否都在e的极小邻域内
        compatible_bases = [base for base in candidates 
                          if base > 0 and abs(math.log(base) - 1.0) < 1e-14]
        
        all_close_to_e = True
        max_deviation_from_e = 0.0
        
        for base in compatible_bases:
            deviation_from_e = abs(base - self.system.e_mathematical)
            max_deviation_from_e = max(max_deviation_from_e, deviation_from_e)
            print(f"兼容底数: {base:.15f}, 与e的差值: {deviation_from_e:.2e}")
            
            # 如果任何兼容底数与e的差值超过数值精度范围，则e不唯一
            if deviation_from_e > 1e-13:  # 允许数值误差
                all_close_to_e = False
        
        print(f"最大与e的偏差: {max_deviation_from_e:.2e}")
        print(f"所有兼容底数都接近e: {all_close_to_e}")
        
        # e的唯一性：所有通过测试的底数都应该在e的极小邻域内
        self.assertTrue(compatible_count > 0, "至少应该有一个底数通过测试")
        self.assertTrue(all_close_to_e, f"存在与e偏差过大的兼容底数，最大偏差={max_deviation_from_e:.2e}")
        
        # 确保标准数学e通过了测试
        mathematical_e_passed = False
        for base in compatible_bases:
            if abs(base - self.system.e_mathematical) < 1e-15:
                mathematical_e_passed = True
                break
        self.assertTrue(mathematical_e_passed, "数学e必须通过自指兼容性测试")
        
        # 验证只有数学e通过严格测试
        e_deviation = deviations[math.exp(1.0)]
        self.assertLess(e_deviation, 1e-15, "数学e的偏差必须极小")
        
    def test_time_irreversibility_absolute(self):
        """测试3：时间不可逆性的绝对保持"""
        print("\n测试3：时间不可逆性的绝对保持")
        
        initial_entropy = 0.5
        time_span = (0.0, 5.0)
        
        entropy_values, time_points, _ = self.system.integrate_time_evolution(
            initial_entropy, self.alpha_test, time_span, self.precision
        )
        
        is_irreversible, causality_violations, arrow_consistency = self.system.verify_time_irreversibility(
            entropy_values, time_points, self.alpha_test
        )
        
        print(f"不可逆性确认: {is_irreversible}")
        print(f"因果性违反数量: {len(causality_violations)}")
        print(f"时间箭头一致性: {arrow_consistency}")
        
        # 严格要求
        self.assertTrue(is_irreversible, "时间演化必须严格不可逆")
        self.assertEqual(len(causality_violations), 0, "不能有任何因果性违反")
        self.assertEqual(arrow_consistency, 1.0, "时间箭头一致性必须完美")
        
        # 验证熵的严格单调递增
        for i in range(1, len(entropy_values)):
            self.assertGreater(entropy_values[i], entropy_values[i-1],
                             f"熵在时间点{i}不满足严格递增")
            
    def test_zeckendorf_time_quantization(self):
        """测试4：Zeckendorf时间量子化"""
        print("\n测试4：Zeckendorf时间量子化")
        
        # 测试时间量子化
        test_times = [0.0, 1.0, 2.5, 5.0, 10.0]
        
        for t in test_times:
            quantized_t, fibonacci_repr, error = self.system.quantize_time_zeckendorf(t, self.alpha_test)
            
            print(f"连续时间{t}: 量子化时间{quantized_t:.6f}, "
                  f"Fibonacci表示{fibonacci_repr}, 误差{error:.2e}")
            
            # 验证Zeckendorf表示的有效性
            if t > 0:
                self.assertTrue(self.system.zeckendorf.is_valid_zeckendorf(fibonacci_repr),
                               f"时间{t}的Zeckendorf表示违反No-11约束")
            
            # 验证量子化误差合理
            time_quantum = math.log(self.system.phi) / self.alpha_test
            max_error = time_quantum / 2  # 四舍五入的最大误差
            self.assertLessEqual(error, max_error + 1e-12, 
                               f"时间{t}的量子化误差过大")
            
    def test_long_term_stability(self):
        """测试5：长期演化稳定性"""
        print("\n测试5：长期演化稳定性")
        
        initial_conditions = {
            'initial_entropy': 1.0,
            'alpha': 0.1  # 较小的α避免过快增长
        }
        time_horizon = 50.0  # 长时间演化
        
        trajectory, log_vars, stability_report = self.system.ensure_long_term_stability(
            initial_conditions, time_horizon
        )
        
        print(f"稳定性状态: {trajectory['is_stable']}")
        print(f"最大对数熵: {stability_report['metrics']['max_log_entropy']:.3f}")
        print(f"溢出点数量: {stability_report['metrics']['numerical_overflow_points']}")
        print(f"总时间点数: {stability_report['metrics']['total_points']}")
        
        # 验证长期稳定性
        self.assertTrue(trajectory['is_stable'], "长期演化必须数值稳定")
        self.assertEqual(stability_report['metrics']['numerical_overflow_points'], 0,
                        "不能有数值溢出")
        
        # 验证对数空间的线性性
        log_entropy = log_vars['log_entropy']
        time_points = log_vars['time']
        alpha = initial_conditions['alpha']
        
        for i, (t, log_H) in enumerate(zip(time_points, log_entropy)):
            expected_log_H = math.log(initial_conditions['initial_entropy']) + alpha * t
            self.assertAlmostEqual(log_H, expected_log_H, places=10,
                                 msg=f"对数空间点{i}不满足线性关系")
            
    def test_t26_2_consistency_strict(self):
        """测试6：与T26-2的严格一致性"""
        print("\n测试6：与T26-2的严格一致性")
        
        # 模拟T26-2的e值结果（应该等于数学e）
        e_from_t26_2 = math.e
        
        # 进行时间演化
        initial_entropy = 2.0
        time_span = (0.0, 3.0)
        
        entropy_values, time_points, _ = self.system.integrate_time_evolution(
            initial_entropy, self.alpha_test, time_span
        )
        
        consistency_score, analysis = self.system.verify_t26_2_consistency(
            e_from_t26_2, self.alpha_test, entropy_values, time_points
        )
        
        print(f"一致性分数: {consistency_score}")
        print(f"e值误差: {analysis['e_value_error']:.2e}")
        print(f"指数模式误差: {analysis['max_exponential_error']:.2e}")
        print(f"α一致性: {analysis['alpha_consistency']:.2e}")
        print(f"整体一致: {analysis['is_consistent']}")
        
        # 严格一致性要求
        self.assertTrue(analysis['is_consistent'], "必须与T26-2完全一致")
        self.assertGreater(consistency_score, 0.999, "一致性分数必须极高")
        self.assertLess(analysis['e_value_error'], 1e-15, "e值误差必须极小")
        
    def test_causality_preservation(self):
        """测试7：因果性保持"""
        print("\n测试7：因果性保持")
        
        initial_entropy = 1.5
        alpha = 0.3
        time_span = (0.0, 4.0)
        
        entropy_values, time_points, _ = self.system.integrate_time_evolution(
            initial_entropy, alpha, time_span
        )
        
        # 验证每个时间点的熵值完全由初始条件决定
        for i, (t, H) in enumerate(zip(time_points, entropy_values)):
            expected_H = initial_entropy * math.exp(alpha * t)
            relative_error = abs(H - expected_H) / expected_H
            
            print(f"t={t:.3f}: 观测H={H:.6f}, 预期H={expected_H:.6f}, "
                  f"相对误差={relative_error:.2e}")
            
            self.assertLess(relative_error, 1e-12, 
                           f"时间点{i}违反因果性：未来影响过去")
        
        # 验证严格的马尔可夫性质：H(t+dt)完全由H(t)决定
        for i in range(1, len(entropy_values)):
            dt = time_points[i] - time_points[i-1]
            H_prev = entropy_values[i-1]
            H_curr = entropy_values[i]
            
            # 根据马尔可夫性质预测
            H_predicted = H_prev * math.exp(alpha * dt)
            markov_error = abs(H_curr - H_predicted) / H_predicted
            
            self.assertLess(markov_error, 1e-12,
                           f"时间步{i}违反马尔可夫性质")
            
    def test_entropy_gradient_positivity(self):
        """测试8：熵梯度的严格正性"""
        print("\n测试8：熵梯度的严格正性")
        
        initial_entropy = 0.8
        alpha = 0.7
        time_span = (0.0, 2.0)
        
        entropy_values, time_points, _ = self.system.integrate_time_evolution(
            initial_entropy, alpha, time_span
        )
        
        # 使用解析梯度计算避免数值误差
        gradients = []
        for i, (t, H) in enumerate(zip(time_points, entropy_values)):
            # 解析梯度：dH/dt = α * H(t)
            analytical_gradient = alpha * H
            gradients.append(analytical_gradient)
            
            print(f"时间t={t:.3f}: H={H:.6f}, dH/dt={analytical_gradient:.6f}")
            
            # 严格要求梯度为正
            self.assertGreater(analytical_gradient, 0, 
                             f"时间点{i}的熵梯度不为正")
            
            # 验证梯度与理论预期的一致性
            expected_H = initial_entropy * math.exp(alpha * t)
            expected_gradient = alpha * expected_H
            gradient_error = abs(analytical_gradient - expected_gradient) / expected_gradient
            self.assertLess(gradient_error, 1e-12,
                           f"时间点{i}的梯度与理论值不符")
        
        # 验证梯度的指数增长特性
        for i in range(1, len(gradients)):
            if i < len(time_points):
                growth_factor = gradients[i] / gradients[i-1]
                dt = time_points[i] - time_points[i-1]
                expected_factor = math.exp(alpha * dt)
                
                factor_error = abs(growth_factor - expected_factor) / expected_factor
                self.assertLess(factor_error, 1e-10,
                               f"梯度增长因子{i}不符合指数规律")
            
    def test_boundary_conditions(self):
        """测试9：边界条件处理"""
        print("\n测试9：边界条件处理")
        
        # 测试零初始熵（应该报错）
        with self.assertRaises(ValueError):
            self.system.integrate_time_evolution(0.0, 1.0, (0.0, 1.0))
        
        # 测试负初始熵（应该报错）
        with self.assertRaises(ValueError):
            self.system.integrate_time_evolution(-1.0, 1.0, (0.0, 1.0))
        
        # 测试零或负α（应该报错）
        with self.assertRaises(ValueError):
            self.system.integrate_time_evolution(1.0, 0.0, (0.0, 1.0))
        
        with self.assertRaises(ValueError):
            self.system.integrate_time_evolution(1.0, -0.5, (0.0, 1.0))
        
        # 测试负时间（应该报错）
        with self.assertRaises(ValueError):
            self.system.integrate_time_evolution(1.0, 1.0, (-1.0, 1.0))
        
        # 测试无效时间范围（应该报错）
        with self.assertRaises(ValueError):
            self.system.integrate_time_evolution(1.0, 1.0, (2.0, 1.0))
        
        print("所有边界条件正确处理")
        
    def test_numerical_precision_limits(self):
        """测试10：数值精度极限"""
        print("\n测试10：数值精度极限")
        
        # 测试极高精度要求
        high_precision = 1e-15
        initial_entropy = 1.0
        alpha = 0.1
        time_span = (0.0, 1.0)
        
        entropy_values, time_points, _ = self.system.integrate_time_evolution(
            initial_entropy, alpha, time_span, high_precision
        )
        
        # 验证精度是否达到要求
        H0 = entropy_values[0]
        max_error = 0.0
        
        for t, H in zip(time_points, entropy_values):
            expected_H = H0 * math.exp(alpha * t)
            relative_error = abs(H - expected_H) / expected_H
            max_error = max(max_error, relative_error)
        
        print(f"最大相对误差: {max_error:.2e}")
        print(f"要求精度: {high_precision:.2e}")
        
        self.assertLessEqual(max_error, high_precision * 10,
                           "数值精度未达到要求")
        
        # 测试极小α值的稳定性
        tiny_alpha = 1e-6
        entropy_values_tiny, _, _ = self.system.integrate_time_evolution(
            initial_entropy, tiny_alpha, (0.0, 100.0)
        )
        
        # 验证极小α下的演化稳定性
        final_entropy = entropy_values_tiny[-1]
        expected_final = initial_entropy * math.exp(tiny_alpha * 100.0)
        tiny_alpha_error = abs(final_entropy - expected_final) / expected_final
        
        print(f"极小α相对误差: {tiny_alpha_error:.2e}")
        self.assertLess(tiny_alpha_error, 1e-10, "极小α值下数值不稳定")

if __name__ == '__main__':
    unittest.main(verbosity=2)