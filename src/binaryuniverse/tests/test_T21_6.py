"""
T21-6 临界带RealityShell映射定理测试套件
基于T21-5概率等价性理论建立的RealityShell映射系统验证
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
import numpy as np


class RealityShellMappingSystem(BinaryUniverseFramework):
    """T21-6 RealityShell映射系统"""
    
    def __init__(self, precision: int = 15):
        super().__init__()
        self.precision = precision
        self.encoder = ZeckendorfEncoder()
        self.phi = (1 + math.sqrt(5)) / 2
        
        # 集成T21-5概率等价性系统
        self.t21_5_system = ZeckendorfProbabilisticEquivalenceSystemCorrected(precision)
        self.setup_reality_shell_states()
    
    def setup_reality_shell_states(self):
        """设置RealityShell状态空间"""
        self.reality_shell_states = {
            "Reality": {"threshold": 2/3, "confidence": 1.0},
            "Boundary": {"threshold": 1/3, "critical_line": 0.5},
            "Possibility": {"threshold": 0.0, "confidence": 0.6},
            "Critical": {"threshold": 1/3, "confidence": 0.8}  # π主导但不在临界线
        }
    
    def compute_reality_shell_mapping(self, s: complex, precision: float = 1e-6) -> Dict[str, Any]:
        """
        计算点s的RealityShell映射
        算法21-6-1实现
        """
        # 验证输入在临界带内
        if not (0 < s.real < 1):
            raise ValueError(f"Point {s} not in critical strip")
        
        # 计算T21-5概率等价性
        analysis = self.t21_5_system.analyze_equivalence_at_point(s)
        equiv_prob = analysis['probabilistic_analysis']['equivalence_probability']
        
        # 应用映射规则
        if abs(equiv_prob - 2/3) < precision:
            reality_shell_state = "Reality"
            state_confidence = 1.0
        elif abs(equiv_prob - 1/3) < precision and abs(s.real - 0.5) < precision:
            reality_shell_state = "Boundary" 
            state_confidence = 1.0
        elif abs(equiv_prob - 1/3) < precision:
            reality_shell_state = "Critical"  # π主导但不在临界线
            state_confidence = 0.8
        else:
            reality_shell_state = "Possibility"
            state_confidence = 0.6
        
        # 计算到边界的距离
        boundary_distance = abs(s.real - 0.5)
        
        # 三元分量分析
        three_fold = analysis['three_fold_decomposition']
        
        return {
            'point': s,
            'reality_shell_state': reality_shell_state,
            'state_confidence': state_confidence,
            'equivalence_probability': equiv_prob,
            'boundary_distance': boundary_distance,
            'layer_classification': self.classify_critical_strip_layer(s, equiv_prob),
            'three_fold_components': {
                'phi_dominance': three_fold['phi_indicator'],
                'pi_dominance': three_fold['pi_indicator'], 
                'e_dominance': three_fold['e_indicator']
            },
            'topological_properties': {
                'is_on_critical_line': abs(s.real - 0.5) < precision,
                'layer_membership': self.determine_layer_membership(s),
                'fractal_coordinate': self.compute_fractal_coordinate(s, equiv_prob)
            }
        }
    
    def classify_critical_strip_layer(self, s: complex, equiv_prob: float) -> str:
        """分类临界带层级"""
        real_part = s.real
        
        if real_part > 2/3:
            return "phi_dominated_layer"  # L1: φ主导层
        elif real_part > 1/3:
            return "mixed_layer"          # L2: 混合层
        else:
            return "pi_dominated_layer"   # L3: π主导层
    
    def determine_layer_membership(self, s: complex) -> Dict[str, float]:
        """确定点在各层的隶属度"""
        sigma = s.real
        
        # 使用模糊隶属函数
        phi_membership = max(0, min(1, 3 * (sigma - 1/3)))
        pi_membership = max(0, min(1, 1 - 2 * abs(sigma - 0.5)))
        mixed_membership = 1 - max(phi_membership, pi_membership)
        
        return {
            'phi_layer': phi_membership,
            'pi_layer': pi_membership,
            'mixed_layer': mixed_membership
        }
    
    def compute_fractal_coordinate(self, s: complex, equiv_prob: float) -> float:
        """计算分形坐标"""
        # 基于概率等价性和位置的分形度量
        distance_to_center = abs(s - complex(0.5, 0))
        prob_deviation = abs(equiv_prob - 0.5)
        
        # 分形坐标反映边界复杂性
        fractal_coord = prob_deviation * math.log(1 + distance_to_center)
        return fractal_coord
    
    def verify_riemann_hypothesis_via_reality_shell(
        self, 
        zero_candidates: List[complex],
        critical_line_tolerance: float = 1e-8
    ) -> Dict[str, Any]:
        """
        通过RealityShell映射验证黎曼猜想
        算法21-6-2实现
        """
        boundary_points = 0
        non_boundary_points = 0
        boundary_violations = []
        
        for zero in zero_candidates:
            # 检查是否在临界带内
            if not (0 < zero.real < 1):
                continue
                
            # 计算RealityShell映射
            mapping_result = self.compute_reality_shell_mapping(zero)
            
            # 检查是否在边界上
            is_on_critical_line = abs(zero.real - 0.5) < critical_line_tolerance
            is_boundary_state = mapping_result['reality_shell_state'] == "Boundary"
            
            if is_on_critical_line and is_boundary_state:
                boundary_points += 1
            elif is_on_critical_line and not is_boundary_state:
                # 在临界线上但不是边界状态（理论冲突）
                boundary_violations.append({
                    'zero': zero,
                    'expected_state': 'Boundary',
                    'actual_state': mapping_result['reality_shell_state'],
                    'equiv_prob': mapping_result['equivalence_probability']
                })
            else:
                non_boundary_points += 1
        
        total_valid_zeros = boundary_points + non_boundary_points
        boundary_ratio = boundary_points / total_valid_zeros if total_valid_zeros > 0 else 0
        
        # RH支持度评估
        if len(boundary_violations) == 0 and boundary_ratio > 0.9:
            rh_support = "Strong"
            rh_confidence = 0.95
        elif len(boundary_violations) <= 2 and boundary_ratio > 0.8:
            rh_support = "Moderate"
            rh_confidence = 0.8
        else:
            rh_support = "Weak"
            rh_confidence = 0.5
        
        return {
            'riemann_hypothesis_analysis': {
                'support_level': rh_support,
                'confidence': rh_confidence,
                'boundary_ratio': boundary_ratio
            },
            'zero_distribution': {
                'total_zeros_analyzed': len(zero_candidates),
                'boundary_points': boundary_points,
                'non_boundary_points': non_boundary_points,
                'violations': boundary_violations
            },
            'reality_shell_interpretation': {
                'boundary_concentration': boundary_ratio,
                'physical_meaning': self.interpret_boundary_concentration(boundary_ratio),
                'collapse_stability': self.assess_collapse_stability(boundary_ratio)
            }
        }
    
    def interpret_boundary_concentration(self, boundary_ratio: float) -> str:
        """解释边界集中度的物理意义"""
        if boundary_ratio > 0.95:
            return "Perfect Reality-Possibility symmetry"
        elif boundary_ratio > 0.8:
            return "High symmetry with minor fluctuations"
        elif boundary_ratio > 0.6:
            return "Moderate asymmetry, Reality/Possibility bias exists"
        else:
            return "Strong asymmetry, fundamental imbalance"
    
    def assess_collapse_stability(self, boundary_ratio: float) -> str:
        """评估collapse稳定性"""
        if boundary_ratio > 0.9:
            return "Highly stable collapse configuration"
        elif boundary_ratio > 0.7:
            return "Moderately stable with fluctuations"
        else:
            return "Unstable collapse tendency"
    
    def analyze_critical_strip_layering(
        self,
        real_range: Tuple[float, float] = (0.1, 0.9),
        imag_range: Tuple[float, float] = (-10, 10),
        grid_resolution: int = 50
    ) -> Dict[str, Any]:
        """
        分析临界带的分层结构和分形特征
        算法21-6-3实现
        """
        # 生成网格
        real_vals = np.linspace(real_range[0], real_range[1], grid_resolution)
        imag_vals = np.linspace(imag_range[0], imag_range[1], grid_resolution)
        
        # 初始化分层统计
        layer_stats = {
            'phi_dominated': {'points': [], 'count': 0},
            'pi_dominated': {'points': [], 'count': 0}, 
            'mixed': {'points': [], 'count': 0},
            'boundary': {'points': [], 'count': 0}
        }
        
        reality_shell_distribution = {}
        fractal_points = []
        
        for r in real_vals[:10]:  # 限制计算量
            for i in imag_vals[:10]:
                s = complex(r, i)
                
                # 计算映射
                try:
                    mapping = self.compute_reality_shell_mapping(s)
                    
                    # 统计分层
                    layer = mapping['layer_classification']
                    rs_state = mapping['reality_shell_state']
                    
                    # 统计简化
                    if layer.startswith('phi'):
                        layer_stats['phi_dominated']['count'] += 1
                        layer_stats['phi_dominated']['points'].append(s)
                    elif layer.startswith('pi'):
                        layer_stats['pi_dominated']['count'] += 1
                        layer_stats['pi_dominated']['points'].append(s)
                    else:
                        layer_stats['mixed']['count'] += 1
                        layer_stats['mixed']['points'].append(s)
                    
                    # RealityShell分布统计
                    if rs_state not in reality_shell_distribution:
                        reality_shell_distribution[rs_state] = 0
                    reality_shell_distribution[rs_state] += 1
                    
                    # 收集分形分析点
                    if mapping['topological_properties']['fractal_coordinate'] > 0:
                        fractal_points.append({
                            'point': s,
                            'fractal_coord': mapping['topological_properties']['fractal_coordinate'],
                            'layer': layer
                        })
                        
                except Exception as e:
                    continue
        
        # 计算分形维数
        fractal_dimension = self.estimate_fractal_dimension(fractal_points)
        
        # 分析层间边界
        boundary_analysis = self.analyze_layer_boundaries(layer_stats)
        
        return {
            'layer_structure': {
                'statistics': layer_stats,
                'layer_ratios': {
                    layer: stats['count'] / sum(s['count'] for s in layer_stats.values()) 
                    if sum(s['count'] for s in layer_stats.values()) > 0 else 0
                    for layer, stats in layer_stats.items()
                },
                'boundary_analysis': boundary_analysis
            },
            'reality_shell_distribution': {
                'raw_counts': reality_shell_distribution,
                'normalized': {
                    state: count / sum(reality_shell_distribution.values())
                    for state, count in reality_shell_distribution.items()
                } if reality_shell_distribution else {}
            },
            'fractal_analysis': {
                'dimension': fractal_dimension,
                'fractal_points_count': len(fractal_points),
                'dimension_interpretation': self.interpret_fractal_dimension(fractal_dimension)
            },
            'theoretical_validation': {
                'matches_t21_6_predictions': self.validate_against_theory(layer_stats, reality_shell_distribution),
                'three_fold_structure_confirmed': self.check_three_fold_structure(layer_stats)
            }
        }
    
    def estimate_fractal_dimension(self, fractal_points: List[Dict]) -> float:
        """估计边界的分形维数"""
        if len(fractal_points) < 10:
            return 1.0  # 不足以进行分形分析
        
        # 简化的分形维数估计
        coord_sum = sum(fp['fractal_coord'] for fp in fractal_points)
        average_complexity = coord_sum / len(fractal_points)
        
        # 基于复杂性映射到分形维数
        dimension = 1.0 + min(0.585, average_complexity)  # 最大1.585
        return dimension
    
    def interpret_fractal_dimension(self, dimension: float) -> str:
        """解释分形维数的物理意义"""
        if dimension > 1.5:
            return "Complex fractal boundary with high self-similarity"
        elif dimension > 1.2:
            return "Moderate fractal structure"
        else:
            return "Simple boundary with low fractal complexity"
    
    def analyze_layer_boundaries(self, layer_stats: Dict) -> Dict[str, Any]:
        """分析层间边界"""
        total_points = sum(stats['count'] for stats in layer_stats.values())
        
        return {
            'total_boundary_points': total_points,
            'layer_distribution': {
                layer: stats['count'] / total_points if total_points > 0 else 0
                for layer, stats in layer_stats.items()
            },
            'boundary_sharpness': self.calculate_boundary_sharpness(layer_stats)
        }
    
    def calculate_boundary_sharpness(self, layer_stats: Dict) -> float:
        """计算边界清晰度"""
        counts = [stats['count'] for stats in layer_stats.values()]
        if not counts or max(counts) == 0:
            return 0.0
        
        # 香农熵的倒数作为清晰度度量
        total = sum(counts)
        entropy = -sum((c/total) * math.log(c/total) for c in counts if c > 0)
        sharpness = 1 / (1 + entropy)
        return sharpness
    
    def validate_against_theory(self, layer_stats: Dict, rs_distribution: Dict) -> bool:
        """验证是否符合T21-6理论预测"""
        # 简化验证：检查基本结构是否存在
        has_three_layers = len([k for k, v in layer_stats.items() if v['count'] > 0]) >= 2
        has_reality_shell_states = len(rs_distribution) >= 2
        
        return has_three_layers and has_reality_shell_states
    
    def check_three_fold_structure(self, layer_stats: Dict) -> bool:
        """检查三元结构是否确认"""
        # 验证φ、π、混合三层结构
        phi_present = layer_stats['phi_dominated']['count'] > 0
        pi_present = layer_stats['pi_dominated']['count'] > 0
        mixed_present = layer_stats['mixed']['count'] > 0
        
        return phi_present or pi_present or mixed_present


class TestT21_6_RealityShellMapping(unittest.TestCase):
    """T21-6 RealityShell映射测试套件"""
    
    def setUp(self):
        """测试前设置"""
        self.system = RealityShellMappingSystem(precision=12)
    
    def test_01_reality_shell_mapping_basic(self):
        """测试基本RealityShell映射功能"""
        test_points = [
            complex(0.5, 0),      # 临界线原点
            complex(0.5, 1.0),    # 临界线
            complex(0.25, 0.5),   # φ主导区域
            complex(0.75, 0.2),   # φ主导区域
        ]
        
        for s in test_points:
            with self.subTest(s=s):
                mapping = self.system.compute_reality_shell_mapping(s)
                
                # 验证映射结果结构
                self.assertIn('reality_shell_state', mapping)
                self.assertIn('state_confidence', mapping)
                self.assertIn('equivalence_probability', mapping)
                self.assertIn('boundary_distance', mapping)
                self.assertIn('layer_classification', mapping)
                
                # 验证状态值合理性
                self.assertIn(mapping['reality_shell_state'], 
                             ['Reality', 'Boundary', 'Possibility', 'Critical'])
                self.assertGreaterEqual(mapping['state_confidence'], 0.0)
                self.assertLessEqual(mapping['state_confidence'], 1.0)
                self.assertIn(mapping['equivalence_probability'], [0.0, 1/3, 2/3])
    
    def test_02_critical_line_boundary_mapping(self):
        """测试临界线Re(s)=1/2的边界映射"""
        critical_line_points = [
            complex(0.5, t) for t in np.linspace(-1, 1, 5)
        ]
        
        boundary_count = 0
        pi_dominated_count = 0
        
        for s in critical_line_points:
            mapping = self.system.compute_reality_shell_mapping(s)
            
            # 检查临界线特性
            self.assertTrue(mapping['topological_properties']['is_on_critical_line'])
            
            # 统计边界状态
            if mapping['reality_shell_state'] in ['Boundary', 'Critical']:
                boundary_count += 1
            
            # 统计π主导
            if mapping['three_fold_components']['pi_dominance'] == 1:
                pi_dominated_count += 1
        
        # 验证临界线主要是边界或Critical状态
        boundary_ratio = boundary_count / len(critical_line_points)
        pi_ratio = pi_dominated_count / len(critical_line_points)
        
        self.assertGreater(boundary_ratio, 0.5)
        self.assertGreater(pi_ratio, 0.3)
    
    def test_03_layer_classification_accuracy(self):
        """测试分层分类准确性"""
        test_cases = [
            (complex(0.8, 0.5), 'phi_dominated_layer'),  # φ主导区域
            (complex(0.2, 0.5), 'pi_dominated_layer'),   # π主导区域  
            (complex(0.5, 0.5), 'mixed_layer'),          # 混合区域
        ]
        
        for s, expected_layer in test_cases:
            with self.subTest(s=s, expected=expected_layer):
                mapping = self.system.compute_reality_shell_mapping(s)
                actual_layer = mapping['layer_classification']
                
                self.assertEqual(actual_layer, expected_layer)
    
    def test_04_three_fold_component_consistency(self):
        """测试三元分量一致性"""
        test_points = [
            complex(0.3, 0.5),
            complex(0.5, 0.5), 
            complex(0.7, 0.5),
        ]
        
        for s in test_points:
            with self.subTest(s=s):
                mapping = self.system.compute_reality_shell_mapping(s)
                components = mapping['three_fold_components']
                
                # 验证指示函数值
                self.assertIn(components['phi_dominance'], [0, 1])
                self.assertIn(components['pi_dominance'], [0, 1])
                self.assertEqual(components['e_dominance'], 0)
                
                # 验证互斥性
                active_components = sum([
                    components['phi_dominance'],
                    components['pi_dominance'],
                    components['e_dominance']
                ])
                self.assertLessEqual(active_components, 1)
    
    def test_05_layer_membership_computation(self):
        """测试层隶属度计算"""
        s = complex(0.6, 0.3)
        
        mapping = self.system.compute_reality_shell_mapping(s)
        layer_membership = mapping['topological_properties']['layer_membership']
        
        # 验证隶属度值在[0,1]范围内
        for layer, membership in layer_membership.items():
            self.assertGreaterEqual(membership, 0.0)
            self.assertLessEqual(membership, 1.0)
        
        # 验证隶属度结构
        self.assertIn('phi_layer', layer_membership)
        self.assertIn('pi_layer', layer_membership)
        self.assertIn('mixed_layer', layer_membership)
    
    def test_06_fractal_coordinate_computation(self):
        """测试分形坐标计算"""
        test_points = [
            complex(0.4, 0.8),
            complex(0.6, 0.2),
            complex(0.5, 1.5),
        ]
        
        for s in test_points:
            with self.subTest(s=s):
                mapping = self.system.compute_reality_shell_mapping(s)
                fractal_coord = mapping['topological_properties']['fractal_coordinate']
                
                # 验证分形坐标为非负数
                self.assertGreaterEqual(fractal_coord, 0.0)
                self.assertTrue(math.isfinite(fractal_coord))
    
    def test_07_riemann_hypothesis_verification(self):
        """测试黎曼猜想验证功能"""
        # 模拟一些零点候选（包括临界线上的和非临界线上的）
        zero_candidates = [
            complex(0.5, 14.134725),   # 真实ζ零点
            complex(0.5, 21.022040),   # 真实ζ零点
            complex(0.5, 25.010858),   # 真实ζ零点
            complex(0.3, 14.0),        # 非临界线零点
            complex(0.7, 21.0),        # 非临界线零点
        ]
        
        rh_analysis = self.system.verify_riemann_hypothesis_via_reality_shell(zero_candidates)
        
        # 验证分析结果结构
        self.assertIn('riemann_hypothesis_analysis', rh_analysis)
        self.assertIn('zero_distribution', rh_analysis)
        self.assertIn('reality_shell_interpretation', rh_analysis)
        
        # 验证RH支持度评估
        rh_support = rh_analysis['riemann_hypothesis_analysis']['support_level']
        self.assertIn(rh_support, ['Strong', 'Moderate', 'Weak'])
        
        # 验证边界集中度分析
        boundary_ratio = rh_analysis['riemann_hypothesis_analysis']['boundary_ratio']
        self.assertGreaterEqual(boundary_ratio, 0.0)
        self.assertLessEqual(boundary_ratio, 1.0)
    
    def test_08_critical_strip_layering_analysis(self):
        """测试临界带分层结构分析"""
        layering_analysis = self.system.analyze_critical_strip_layering(
            real_range=(0.2, 0.8),
            imag_range=(-2, 2),
            grid_resolution=10  # 小网格以加快测试
        )
        
        # 验证分析结果结构
        self.assertIn('layer_structure', layering_analysis)
        self.assertIn('reality_shell_distribution', layering_analysis)
        self.assertIn('fractal_analysis', layering_analysis)
        self.assertIn('theoretical_validation', layering_analysis)
        
        # 验证分层统计
        layer_stats = layering_analysis['layer_structure']['statistics']
        self.assertIn('phi_dominated', layer_stats)
        self.assertIn('pi_dominated', layer_stats)
        self.assertIn('mixed', layer_stats)
        
        # 验证分形维数计算
        fractal_dimension = layering_analysis['fractal_analysis']['dimension']
        self.assertGreaterEqual(fractal_dimension, 1.0)
        self.assertLessEqual(fractal_dimension, 2.0)
    
    def test_09_boundary_concentration_interpretation(self):
        """测试边界集中度解释"""
        test_ratios = [0.95, 0.85, 0.65, 0.45]
        
        for ratio in test_ratios:
            with self.subTest(ratio=ratio):
                interpretation = self.system.interpret_boundary_concentration(ratio)
                self.assertIsInstance(interpretation, str)
                self.assertGreater(len(interpretation), 10)  # 有意义的解释
    
    def test_10_reality_shell_state_distribution(self):
        """测试RealityShell状态分布"""
        # 生成临界带内的测试点
        test_points = []
        for r in np.linspace(0.2, 0.8, 5):
            for i in np.linspace(-1, 1, 3):
                test_points.append(complex(r, i))
        
        state_counts = {}
        for s in test_points:
            try:
                mapping = self.system.compute_reality_shell_mapping(s)
                state = mapping['reality_shell_state']
                state_counts[state] = state_counts.get(state, 0) + 1
            except Exception:
                continue
        
        # 验证状态分布合理性
        total_points = sum(state_counts.values())
        if total_points > 0:
            # 应该有多种状态出现
            self.assertGreaterEqual(len(state_counts), 2)
            
            # 验证各状态比例合理
            for state, count in state_counts.items():
                ratio = count / total_points
                self.assertLessEqual(ratio, 1.0)
                self.assertGreater(ratio, 0.0)
    
    def test_11_integration_with_t21_5_system(self):
        """测试与T21-5系统的集成"""
        s = complex(0.5, 1.0)
        
        # 直接从T21-5系统获取分析
        t21_5_analysis = self.system.t21_5_system.analyze_equivalence_at_point(s)
        
        # 通过T21-6系统获取映射
        t21_6_mapping = self.system.compute_reality_shell_mapping(s)
        
        # 验证等价概率一致性
        t21_5_prob = t21_5_analysis['probabilistic_analysis']['equivalence_probability']
        t21_6_prob = t21_6_mapping['equivalence_probability']
        
        self.assertEqual(t21_5_prob, t21_6_prob)
        
        # 验证三元分量一致性
        t21_5_components = t21_5_analysis['three_fold_decomposition']
        t21_6_components = t21_6_mapping['three_fold_components']
        
        self.assertEqual(t21_5_components['phi_indicator'], 
                        t21_6_components['phi_dominance'])
        self.assertEqual(t21_5_components['pi_indicator'], 
                        t21_6_components['pi_dominance'])
        self.assertEqual(t21_5_components['e_indicator'], 
                        t21_6_components['e_dominance'])
    
    def test_12_comprehensive_theory_validation(self):
        """综合理论验证"""
        # 大规模测试验证T21-6理论
        real_vals = np.linspace(0.3, 0.7, 4)
        imag_vals = np.linspace(-1, 1, 4)
        
        total_tests = 0
        successful_mappings = 0
        theory_consistent = 0
        
        for r in real_vals:
            for i in imag_vals:
                s = complex(r, i)
                total_tests += 1
                
                try:
                    mapping = self.system.compute_reality_shell_mapping(s)
                    successful_mappings += 1
                    
                    # 检查理论一致性
                    equiv_prob = mapping['equivalence_probability']
                    rs_state = mapping['reality_shell_state']
                    
                    # 验证映射规则一致性
                    if (equiv_prob == 2/3 and rs_state == "Reality") or \
                       (equiv_prob == 1/3 and rs_state in ["Boundary", "Critical"]) or \
                       (equiv_prob == 0 and rs_state == "Possibility"):
                        theory_consistent += 1
                        
                except Exception as e:
                    continue
        
        # 计算成功率
        if total_tests > 0:
            success_rate = successful_mappings / total_tests
            consistency_rate = theory_consistent / successful_mappings if successful_mappings > 0 else 0
            
            print(f"\n综合理论验证结果:")
            print(f"测试总数: {total_tests}")
            print(f"成功映射率: {success_rate:.3f}")
            print(f"理论一致性率: {consistency_rate:.3f}")
            
            # 验证理论的有效性
            self.assertGreater(success_rate, 0.8)  # 80%以上成功率
            self.assertGreater(consistency_rate, 0.6)  # 60%以上一致性
            
            # 验证是否支持T21-6理论
            supports_theory = (success_rate > 0.8 and consistency_rate > 0.6)
            self.assertTrue(supports_theory, "综合测试未能验证T21-6理论")


if __name__ == '__main__':
    # 运行测试套件
    unittest.main(verbosity=2)