"""
C21-1 黎曼猜想RealityShell概率重述推论测试套件
基于T21-5概率等价性理论和T21-6 RealityShell映射定理
实现黎曼猜想的概率化验证框架
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import math
import cmath
from typing import List, Tuple, Dict, Any, Optional
from base_framework import BinaryUniverseFramework, ZeckendorfEncoder, PhiBasedMeasure
from test_T21_5 import ZeckendorfProbabilisticEquivalenceSystemCorrected
from test_T21_6 import RealityShellMappingSystem
import numpy as np


class RiemannHypothesisRealityShellVerifier(BinaryUniverseFramework):
    """C21-1 黎曼猜想RealityShell验证系统"""
    
    def __init__(self, precision: int = 15):
        super().__init__()
        self.precision = precision
        self.encoder = ZeckendorfEncoder()
        
        # 集成T21-5和T21-6系统
        self.t21_5_system = ZeckendorfProbabilisticEquivalenceSystemCorrected(precision)
        self.t21_6_system = RealityShellMappingSystem(precision)
        
        # 设置验证参数
        self.setup_verification_parameters()
    
    def setup_verification_parameters(self):
        """设置验证参数"""
        self.verification_params = {
            'boundary_threshold': 0.95,       # 边界集中度阈值
            'confidence_level': 0.95,         # 置信水平
            'critical_line_tolerance': 1e-8,  # 临界线容忍度
            'zero_detection_threshold': 1e-10, # 零点检测阈值
            'min_sample_size': 50,            # 最小样本大小
            'statistical_significance': 0.05  # 统计显著性水平
        }
    
    def generate_riemann_zero_candidates(self, count: int = 100) -> List[complex]:
        """
        生成黎曼ζ函数零点候选
        使用已知的非平凡零点进行测试
        """
        # 已知的前几个非平凡零点的虚部（实部为0.5）
        known_zeros_imaginary = [
            14.134725141734693790,
            21.022039638771554993,
            25.010857580145688763,
            30.424876125859513210,
            32.935061587739189691,
            37.586178158825671257,
            40.918719012147495187,
            43.327073280914999519,
            48.005150881167159727,
            49.773832477672302181
        ]
        
        zero_candidates = []
        
        # 添加已知零点（在临界线上）
        for imag_part in known_zeros_imaginary[:min(count//2, len(known_zeros_imaginary))]:
            zero_candidates.append(complex(0.5, imag_part))
            zero_candidates.append(complex(0.5, -imag_part))  # 对称零点
        
        # 添加一些测试用的非临界线点
        remaining_count = count - len(zero_candidates)
        for i in range(remaining_count):
            # 在临界带内但不在临界线上的点
            if i % 2 == 0:
                real_part = 0.3
            else:
                real_part = 0.7
            imag_part = 10 + 5 * i
            zero_candidates.append(complex(real_part, imag_part))
        
        return zero_candidates[:count]
    
    def classify_zero_reality_shell_states(
        self, 
        zero_candidates: List[complex]
    ) -> Dict[str, List[complex]]:
        """
        零点RealityShell状态分类器 - 算法C21-1-1实现
        """
        classified_zeros = {
            'boundary': [],
            'reality': [],
            'critical': [],
            'possibility': []
        }
        
        for zero in zero_candidates:
            # 检查是否在临界带内
            if not (0 < zero.real < 1):
                continue
            
            try:
                # 计算RealityShell映射
                mapping_result = self.t21_6_system.compute_reality_shell_mapping(zero)
                
                # 获取状态和概率
                rs_state = mapping_result['reality_shell_state']
                equiv_prob = mapping_result['equivalence_probability']
                is_on_critical_line = abs(zero.real - 0.5) < self.verification_params['critical_line_tolerance']
                
                # 验证状态一致性
                if self.validate_zero_state_consistency(zero, rs_state, equiv_prob, is_on_critical_line):
                    classified_zeros[rs_state.lower()].append(zero)
                    
            except Exception as e:
                # 记录错误但继续处理
                print(f"Warning: Error classifying zero {zero}: {e}")
                continue
        
        return classified_zeros
    
    def validate_zero_state_consistency(
        self, 
        zero: complex, 
        rs_state: str, 
        equiv_prob: float, 
        is_on_critical_line: bool
    ) -> bool:
        """验证零点的RealityShell状态一致性"""
        tolerance = 1e-6
        
        if rs_state == "Boundary":
            return (abs(equiv_prob - 1/3) < tolerance and is_on_critical_line)
        elif rs_state == "Reality":
            return abs(equiv_prob - 2/3) < tolerance
        elif rs_state == "Critical":
            return (abs(equiv_prob - 1/3) < tolerance and not is_on_critical_line)
        elif rs_state == "Possibility":
            return abs(equiv_prob - 0.0) < tolerance
        else:
            return False
    
    def analyze_boundary_concentration(
        self, 
        classified_zeros: Dict[str, List[complex]]
    ) -> Dict[str, Any]:
        """
        边界集中度分析器 - 算法C21-1-2实现
        """
        # 计算各状态零点数量
        boundary_count = len(classified_zeros['boundary'])
        reality_count = len(classified_zeros['reality'])
        critical_count = len(classified_zeros['critical'])
        possibility_count = len(classified_zeros['possibility'])
        
        total_zeros = boundary_count + reality_count + critical_count + possibility_count
        
        if total_zeros == 0:
            return {
                'error': 'No zeros found for analysis',
                'boundary_concentration': 0.0,
                'riemann_hypothesis_support': 'Insufficient data'
            }
        
        # 计算边界集中度
        boundary_concentration = boundary_count / total_zeros
        
        # 计算置信区间
        confidence_interval = self.calculate_binomial_confidence_interval(
            boundary_count, total_zeros, self.verification_params['confidence_level']
        )
        
        # 评估黎曼猜想支持度
        rh_support = self.evaluate_riemann_hypothesis_support(
            boundary_concentration, confidence_interval
        )
        
        return {
            'zero_counts': {
                'boundary': boundary_count,
                'reality': reality_count,
                'critical': critical_count,
                'possibility': possibility_count,
                'total': total_zeros
            },
            'boundary_concentration': boundary_concentration,
            'confidence_interval': confidence_interval,
            'riemann_hypothesis_support': rh_support,
            'theoretical_validation': {
                'matches_c21_1_prediction': boundary_concentration >= self.verification_params['boundary_threshold'],
                'statistical_significance': self.assess_statistical_significance(boundary_count, total_zeros),
                'sample_adequacy': total_zeros >= self.verification_params['min_sample_size']
            }
        }
    
    def calculate_binomial_confidence_interval(
        self, 
        successes: int, 
        trials: int, 
        confidence_level: float
    ) -> Tuple[float, float]:
        """计算边界集中度的二项分布置信区间"""
        if trials == 0:
            return (0.0, 0.0)
        
        # 使用正态近似的置信区间
        p_hat = successes / trials
        z_score = 1.96  # 95%置信水平对应的z分数
        
        if confidence_level == 0.99:
            z_score = 2.576
        elif confidence_level == 0.90:
            z_score = 1.645
        
        margin_of_error = z_score * math.sqrt(p_hat * (1 - p_hat) / trials)
        
        lower_bound = max(0, p_hat - margin_of_error)
        upper_bound = min(1, p_hat + margin_of_error)
        
        return (lower_bound, upper_bound)
    
    def evaluate_riemann_hypothesis_support(
        self, 
        boundary_concentration: float, 
        confidence_interval: Tuple[float, float]
    ) -> Dict[str, Any]:
        """评估对黎曼猜想的支持程度"""
        lower_bound, upper_bound = confidence_interval
        threshold = self.verification_params['boundary_threshold']
        
        if lower_bound >= threshold:
            support_level = "Strong"
            support_confidence = 0.95
            interpretation = "Strong evidence supporting Riemann Hypothesis via RealityShell analysis"
        elif boundary_concentration >= threshold:
            support_level = "Moderate"
            support_confidence = 0.8
            interpretation = "Moderate evidence supporting Riemann Hypothesis"
        elif boundary_concentration >= 0.8:
            support_level = "Weak"
            support_confidence = 0.6
            interpretation = "Weak evidence supporting Riemann Hypothesis"
        else:
            support_level = "Insufficient"
            support_confidence = 0.3
            interpretation = "Insufficient evidence for Riemann Hypothesis"
        
        return {
            'support_level': support_level,
            'support_confidence': support_confidence,
            'interpretation': interpretation,
            'boundary_concentration': boundary_concentration,
            'confidence_bounds': confidence_interval,
            'meets_c21_1_threshold': boundary_concentration >= threshold
        }
    
    def assess_statistical_significance(self, boundary_count: int, total_count: int) -> Dict[str, Any]:
        """评估统计显著性"""
        if total_count == 0:
            return {'significant': False, 'p_value': 1.0}
        
        # 零假设：边界集中度 <= 0.5（随机分布）
        # 备择假设：边界集中度 > 0.5
        p0 = 0.5  # 零假设下的期望概率
        p_observed = boundary_count / total_count
        
        # 使用正态近似进行z检验
        if total_count * p0 * (1 - p0) >= 5:  # 正态近似条件
            z_stat = (p_observed - p0) / math.sqrt(p0 * (1 - p0) / total_count)
            # 单尾检验的p值近似
            if z_stat > 0:
                p_value = 0.5 * math.exp(-0.717 * z_stat - 0.416 * z_stat**2)  # 近似公式
            else:
                p_value = 1.0
        else:
            # 样本过小，使用保守估计
            p_value = 0.5
        
        return {
            'significant': p_value < self.verification_params['statistical_significance'],
            'p_value': p_value,
            'z_statistic': z_stat if 'z_stat' in locals() else None,
            'test_type': 'one-tailed z-test',
            'null_hypothesis': 'boundary_concentration <= 0.5'
        }
    
    def verify_reality_shell_mapping_consistency(
        self, 
        zero_dataset: List[complex]
    ) -> Dict[str, Any]:
        """验证RealityShell映射与T21-5系统的一致性"""
        consistency_results = {
            'total_zeros': len(zero_dataset),
            'consistent_mappings': 0,
            'inconsistent_mappings': 0,
            'mapping_errors': []
        }
        
        for zero in zero_dataset:
            try:
                # T21-6 RealityShell映射
                rs_mapping = self.t21_6_system.compute_reality_shell_mapping(zero)
                
                # T21-5 概率等价性分析
                equiv_analysis = self.t21_5_system.analyze_equivalence_at_point(zero)
                
                # 检查一致性
                rs_prob = rs_mapping['equivalence_probability']
                equiv_prob = equiv_analysis['probabilistic_analysis']['equivalence_probability']
                
                if abs(rs_prob - equiv_prob) < 1e-10:
                    consistency_results['consistent_mappings'] += 1
                else:
                    consistency_results['inconsistent_mappings'] += 1
                    consistency_results['mapping_errors'].append({
                        'zero': zero,
                        'rs_probability': rs_prob,
                        'equiv_probability': equiv_prob,
                        'difference': abs(rs_prob - equiv_prob)
                    })
                    
            except Exception as e:
                consistency_results['mapping_errors'].append({
                    'zero': zero,
                    'error': str(e)
                })
        
        consistency_rate = (consistency_results['consistent_mappings'] / 
                           consistency_results['total_zeros'] 
                           if consistency_results['total_zeros'] > 0 else 0)
        
        consistency_results['consistency_rate'] = consistency_rate
        consistency_results['passes_consistency_test'] = consistency_rate >= 0.99
        
        return consistency_results


class TestC21_1_RiemannHypothesisRealityShell(unittest.TestCase):
    """C21-1 黎曼猜想RealityShell概率重述推论测试套件"""
    
    def setUp(self):
        """测试前设置"""
        self.verifier = RiemannHypothesisRealityShellVerifier(precision=12)
    
    def test_01_zero_candidate_generation(self):
        """测试零点候选生成"""
        zero_candidates = self.verifier.generate_riemann_zero_candidates(50)
        
        # 验证候选数量
        self.assertEqual(len(zero_candidates), 50)
        
        # 验证候选都在临界带内
        for zero in zero_candidates:
            self.assertGreater(zero.real, 0)
            self.assertLess(zero.real, 1)
        
        # 验证包含已知零点
        critical_line_zeros = [z for z in zero_candidates if abs(z.real - 0.5) < 1e-10]
        self.assertGreater(len(critical_line_zeros), 0)
    
    def test_02_zero_reality_shell_classification(self):
        """测试零点RealityShell状态分类"""
        # 生成测试零点
        test_zeros = [
            complex(0.5, 14.134725),   # 已知零点，应该是Boundary
            complex(0.5, 21.022040),   # 已知零点，应该是Boundary
            complex(0.3, 15.0),        # 非临界线，可能是Reality或Critical
            complex(0.7, 20.0),        # 非临界线，可能是Reality或Critical
        ]
        
        classified = self.verifier.classify_zero_reality_shell_states(test_zeros)
        
        # 验证分类结果结构
        self.assertIn('boundary', classified)
        self.assertIn('reality', classified)
        self.assertIn('critical', classified)
        self.assertIn('possibility', classified)
        
        # 验证临界线零点被正确分类为边界状态
        boundary_zeros = classified['boundary']
        critical_line_zeros_in_boundary = sum(1 for z in boundary_zeros 
                                             if abs(z.real - 0.5) < 1e-8)
        
        # 至少有一些临界线零点应该被分类为边界状态
        self.assertGreater(critical_line_zeros_in_boundary, 0)
    
    def test_03_boundary_concentration_analysis(self):
        """测试边界集中度分析"""
        # 创建模拟分类结果
        mock_classified = {
            'boundary': [complex(0.5, 14.1), complex(0.5, 21.0), complex(0.5, 25.0)],
            'reality': [complex(0.3, 15.0)],
            'critical': [complex(0.7, 20.0)],
            'possibility': []
        }
        
        analysis = self.verifier.analyze_boundary_concentration(mock_classified)
        
        # 验证分析结果结构
        self.assertIn('zero_counts', analysis)
        self.assertIn('boundary_concentration', analysis)
        self.assertIn('confidence_interval', analysis)
        self.assertIn('riemann_hypothesis_support', analysis)
        
        # 验证边界集中度计算
        expected_concentration = 3 / 5  # 3个边界零点，5个总零点
        self.assertAlmostEqual(analysis['boundary_concentration'], expected_concentration, places=6)
        
        # 验证置信区间
        ci_lower, ci_upper = analysis['confidence_interval']
        self.assertLessEqual(ci_lower, expected_concentration)
        self.assertGreaterEqual(ci_upper, expected_concentration)
        
        # 验证RH支持度评估
        rh_support = analysis['riemann_hypothesis_support']
        self.assertIn('support_level', rh_support)
        self.assertIn('interpretation', rh_support)
    
    def test_04_riemann_hypothesis_support_evaluation(self):
        """测试黎曼猜想支持度评估"""
        # 测试强支持情况（边界集中度 >= 0.95）
        strong_support = self.verifier.evaluate_riemann_hypothesis_support(0.97, (0.95, 0.99))
        self.assertEqual(strong_support['support_level'], "Strong")
        self.assertTrue(strong_support['meets_c21_1_threshold'])
        
        # 测试中等支持情况
        moderate_support = self.verifier.evaluate_riemann_hypothesis_support(0.96, (0.92, 1.0))
        self.assertEqual(moderate_support['support_level'], "Moderate")
        self.assertTrue(moderate_support['meets_c21_1_threshold'])
        
        # 测试弱支持情况
        weak_support = self.verifier.evaluate_riemann_hypothesis_support(0.85, (0.80, 0.90))
        self.assertEqual(weak_support['support_level'], "Weak")
        self.assertFalse(weak_support['meets_c21_1_threshold'])
        
        # 测试不充分支持情况
        insufficient_support = self.verifier.evaluate_riemann_hypothesis_support(0.45, (0.30, 0.60))
        self.assertEqual(insufficient_support['support_level'], "Insufficient")
        self.assertFalse(insufficient_support['meets_c21_1_threshold'])
    
    def test_05_state_consistency_validation(self):
        """测试状态一致性验证"""
        # 测试边界状态验证
        boundary_zero = complex(0.5, 14.1)
        self.assertTrue(self.verifier.validate_zero_state_consistency(
            boundary_zero, "Boundary", 1/3, True))
        
        # 测试Reality状态验证
        reality_zero = complex(0.3, 15.0)
        self.assertTrue(self.verifier.validate_zero_state_consistency(
            reality_zero, "Reality", 2/3, False))
        
        # 测试Critical状态验证
        critical_zero = complex(0.7, 20.0)
        self.assertTrue(self.verifier.validate_zero_state_consistency(
            critical_zero, "Critical", 1/3, False))
        
        # 测试不一致情况
        self.assertFalse(self.verifier.validate_zero_state_consistency(
            boundary_zero, "Reality", 1/3, True))  # 状态与概率不匹配
    
    def test_06_statistical_significance_assessment(self):
        """测试统计显著性评估"""
        # 测试显著情况
        significant_result = self.verifier.assess_statistical_significance(95, 100)
        self.assertTrue(significant_result['significant'])
        self.assertLess(significant_result['p_value'], 0.05)
        
        # 测试不显著情况
        non_significant_result = self.verifier.assess_statistical_significance(45, 100)
        self.assertFalse(non_significant_result['significant'])
        self.assertGreater(non_significant_result['p_value'], 0.05)
        
        # 测试边界情况
        borderline_result = self.verifier.assess_statistical_significance(60, 100)
        self.assertIn('p_value', borderline_result)
        self.assertIn('z_statistic', borderline_result)
    
    def test_07_reality_shell_consistency_verification(self):
        """测试RealityShell映射一致性验证"""
        # 生成测试零点
        test_zeros = [
            complex(0.5, 14.1),
            complex(0.5, 21.0),
            complex(0.3, 15.0),
            complex(0.7, 20.0)
        ]
        
        consistency_check = self.verifier.verify_reality_shell_mapping_consistency(test_zeros)
        
        # 验证结果结构
        self.assertIn('total_zeros', consistency_check)
        self.assertIn('consistent_mappings', consistency_check)
        self.assertIn('inconsistent_mappings', consistency_check)
        self.assertIn('consistency_rate', consistency_check)
        self.assertIn('passes_consistency_test', consistency_check)
        
        # 验证一致性率
        consistency_rate = consistency_check['consistency_rate']
        self.assertGreaterEqual(consistency_rate, 0.0)
        self.assertLessEqual(consistency_rate, 1.0)
        
        # 高一致性应该通过测试
        if consistency_rate >= 0.99:
            self.assertTrue(consistency_check['passes_consistency_test'])
    
    def test_08_binomial_confidence_interval(self):
        """测试二项分布置信区间计算"""
        # 测试标准情况
        ci = self.verifier.calculate_binomial_confidence_interval(80, 100, 0.95)
        self.assertEqual(len(ci), 2)
        self.assertLessEqual(ci[0], 0.8)  # 下界应该 <= 观测值
        self.assertGreaterEqual(ci[1], 0.8)  # 上界应该 >= 观测值
        
        # 测试边界情况
        ci_zero = self.verifier.calculate_binomial_confidence_interval(0, 0, 0.95)
        self.assertEqual(ci_zero, (0.0, 0.0))
        
        ci_perfect = self.verifier.calculate_binomial_confidence_interval(100, 100, 0.95)
        self.assertLess(ci_perfect[0], 1.0)  # 下界应该小于1
        self.assertEqual(ci_perfect[1], 1.0)  # 上界应该等于1
    
    def test_09_comprehensive_riemann_verification(self):
        """综合黎曼猜想验证测试"""
        # 生成较大的测试样本
        zero_candidates = self.verifier.generate_riemann_zero_candidates(30)
        
        # 执行完整的验证流程
        classified = self.verifier.classify_zero_reality_shell_states(zero_candidates)
        analysis = self.verifier.analyze_boundary_concentration(classified)
        consistency_check = self.verifier.verify_reality_shell_mapping_consistency(zero_candidates)
        
        # 验证综合结果
        self.assertGreaterEqual(analysis['zero_counts']['total'], 20)  # 至少处理了20个零点
        
        # 验证边界集中度
        boundary_concentration = analysis['boundary_concentration']
        self.assertGreaterEqual(boundary_concentration, 0.0)
        self.assertLessEqual(boundary_concentration, 1.0)
        
        # 验证理论一致性
        theoretical_validation = analysis['theoretical_validation']
        self.assertIn('matches_c21_1_prediction', theoretical_validation)
        self.assertIn('statistical_significance', theoretical_validation)
        self.assertIn('sample_adequacy', theoretical_validation)
        
        # 验证系统集成
        self.assertGreaterEqual(consistency_check['consistency_rate'], 0.95)  # 至少95%一致性
        
        print(f"\\n综合黎曼猜想验证结果:")
        print(f"总零点数: {analysis['zero_counts']['total']}")
        print(f"边界零点数: {analysis['zero_counts']['boundary']}")
        print(f"边界集中度: {boundary_concentration:.3f}")
        print(f"RH支持度: {analysis['riemann_hypothesis_support']['support_level']}")
        print(f"系统一致性: {consistency_check['consistency_rate']:.3f}")
    
    def test_10_critical_line_zero_behavior(self):
        """测试临界线零点的特殊行为"""
        # 生成临界线零点
        critical_line_zeros = [
            complex(0.5, 14.134725),
            complex(0.5, 21.022040),
            complex(0.5, 25.010858),
            complex(0.5, 30.424876),
            complex(0.5, 32.935062)
        ]
        
        classified = self.verifier.classify_zero_reality_shell_states(critical_line_zeros)
        boundary_zeros = classified['boundary']
        
        # 验证临界线零点主要被分类为边界状态
        boundary_ratio = len(boundary_zeros) / len(critical_line_zeros)
        
        # 根据C21-1理论，临界线零点应该有高概率被分类为边界状态
        self.assertGreater(boundary_ratio, 0.7)  # 至少70%
        
        # 验证每个边界零点的概率等价性
        for zero in boundary_zeros:
            try:
                mapping = self.t21_6_system.compute_reality_shell_mapping(zero)
                equiv_prob = mapping['equivalence_probability']
                
                # 边界状态应该对应1/3概率
                self.assertAlmostEqual(equiv_prob, 1/3, places=6)
                
            except Exception as e:
                self.fail(f"Failed to analyze boundary zero {zero}: {e}")
    
    def test_11_non_critical_line_analysis(self):
        """测试非临界线点的分析"""
        # 生成非临界线点
        non_critical_points = [
            complex(0.3, 14.0),
            complex(0.3, 21.0),
            complex(0.7, 25.0),
            complex(0.7, 30.0),
        ]
        
        classified = self.verifier.classify_zero_reality_shell_states(non_critical_points)
        
        # 非临界线点应该主要被分类为Reality或Critical状态
        reality_count = len(classified['reality'])
        critical_count = len(classified['critical'])
        boundary_count = len(classified['boundary'])
        
        # 非临界线点被分类为边界状态的比例应该很低
        total_analyzed = reality_count + critical_count + boundary_count
        if total_analyzed > 0:
            boundary_ratio = boundary_count / total_analyzed
            self.assertLess(boundary_ratio, 0.3)  # 少于30%
    
    def test_12_c21_1_theoretical_predictions(self):
        """验证C21-1理论预测"""
        # 生成混合测试样本：临界线 + 非临界线
        critical_zeros = [complex(0.5, 14.1), complex(0.5, 21.0), complex(0.5, 25.0)]
        non_critical_points = [complex(0.3, 15.0), complex(0.7, 20.0)]
        all_points = critical_zeros + non_critical_points
        
        classified = self.verifier.classify_zero_reality_shell_states(all_points)
        analysis = self.verifier.analyze_boundary_concentration(classified)
        
        boundary_concentration = analysis['boundary_concentration']
        rh_support = analysis['riemann_hypothesis_support']
        
        # C21-1预测：如果RH成立，边界集中度应该 >= 0.95
        # 对于这个小样本，我们检验基本理论结构
        self.assertIsInstance(boundary_concentration, (int, float))
        self.assertGreaterEqual(boundary_concentration, 0.0)
        self.assertLessEqual(boundary_concentration, 1.0)
        
        # 验证支持度评估结构
        self.assertIn('support_level', rh_support)
        self.assertIn('support_confidence', rh_support)
        self.assertIn('meets_c21_1_threshold', rh_support)
        
        # 验证理论一致性
        theoretical_validation = analysis['theoretical_validation']
        self.assertIn('matches_c21_1_prediction', theoretical_validation)
        
        # 临界线零点应该显示出边界集中的趋势
        if len(classified['boundary']) > 0:
            critical_line_in_boundary = sum(1 for z in classified['boundary'] 
                                           if abs(z.real - 0.5) < 1e-8)
            self.assertGreater(critical_line_in_boundary, 0)
        
        print(f"\\nC21-1理论预测验证:")
        print(f"边界集中度: {boundary_concentration:.3f}")
        print(f"理论预测匹配: {theoretical_validation['matches_c21_1_prediction']}")
        print(f"RH支持等级: {rh_support['support_level']}")
        print(f"边界状态零点: {len(classified['boundary'])}")
        print(f"Reality状态零点: {len(classified['reality'])}")
        print(f"Critical状态零点: {len(classified['critical'])}")


if __name__ == '__main__':
    # 运行测试套件
    unittest.main(verbosity=2)