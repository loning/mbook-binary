"""
Unit tests for D1-4: Time Metric
D1-4：时间度量的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
from formal_system import SystemState, TimeMetric, create_initial_system, simulate_evolution
import numpy as np


class TestD1_4_TimeMetric(VerificationTest):
    """D1-4时间度量的形式化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        self.time_metric = TimeMetric()
        
        # 创建测试用的系统状态序列
        self.states = simulate_evolution(5)  # 生成6个状态 (0到5)
        
    def test_non_negativity(self):
        """测试非负性 - 验证检查点1"""
        # 测试所有状态对的时间距离
        for i, state_i in enumerate(self.states):
            for j, state_j in enumerate(self.states):
                distance = self.time_metric.distance(state_i, state_j)
                
                # 对于时间度量，只有前向时间（i<j）才需要非负
                # 后向时间（i>j）为负值，表示逆向
                if i <= j:
                    # 验证前向时间非负性
                    self.assertGreaterEqual(
                        distance, 0,
                        f"Forward time distance from t={i} to t={j} should be non-negative"
                    )
                
                # 验证零距离当且仅当相同状态
                if i == j:
                    self.assertEqual(
                        distance, 0,
                        f"Distance from state at t={i} to itself should be 0"
                    )
                else:
                    self.assertNotEqual(
                        distance, 0,
                        f"Distance between different states (t={i}, t={j}) should be non-zero"
                    )
                    
    def test_monotonicity(self):
        """测试单调性 - 验证检查点2"""
        # 测试 i < j < k 时的单调性
        for i in range(len(self.states) - 2):
            for j in range(i + 1, len(self.states) - 1):
                for k in range(j + 1, len(self.states)):
                    d_ij = self.time_metric.distance(self.states[i], self.states[j])
                    d_ik = self.time_metric.distance(self.states[i], self.states[k])
                    
                    self.assertLess(
                        d_ij, d_ik,
                        f"Distance should be monotonic: τ({i},{j}) < τ({i},{k})"
                    )
                    
    def test_additivity(self):
        """测试可加性 - 验证检查点3"""
        # 测试 τ(i,k) = τ(i,j) + τ(j,k)
        for i in range(len(self.states) - 2):
            for j in range(i + 1, len(self.states) - 1):
                for k in range(j + 1, len(self.states)):
                    d_ij = self.time_metric.distance(self.states[i], self.states[j])
                    d_jk = self.time_metric.distance(self.states[j], self.states[k])
                    d_ik = self.time_metric.distance(self.states[i], self.states[k])
                    
                    # 允许小的数值误差
                    self.assertAlmostEqual(
                        d_ik, d_ij + d_jk,
                        places=10,
                        msg=f"Additivity: τ({i},{k}) should equal τ({i},{j}) + τ({j},{k})"
                    )
                    
    def test_directionality(self):
        """测试方向性 - 验证检查点4"""
        # 测试时间方向性：τ(i,j) > 0 当且仅当 i < j
        for i, state_i in enumerate(self.states):
            for j, state_j in enumerate(self.states):
                distance = self.time_metric.distance(state_i, state_j)
                
                if i < j:
                    self.assertGreater(
                        distance, 0,
                        f"Forward time distance τ({i},{j}) should be positive"
                    )
                elif i > j:
                    self.assertLess(
                        distance, 0,
                        f"Backward time distance τ({i},{j}) should be negative"
                    )
                else:  # i == j
                    self.assertEqual(
                        distance, 0,
                        f"Same time distance τ({i},{i}) should be zero"
                    )
                    
    def test_structural_distance(self):
        """测试结构距离度量"""
        # 验证相邻状态的结构距离
        for i in range(len(self.states) - 1):
            s1 = self.states[i]
            s2 = self.states[i + 1]
            
            # 计算新增元素数量
            new_elements = s2.elements - s1.elements
            expected_distance = np.sqrt(len(new_elements))
            
            # 验证距离计算
            actual_distance = self.time_metric.distance(s1, s2)
            self.assertAlmostEqual(
                actual_distance, expected_distance,
                places=10,
                msg=f"Structural distance between t={i} and t={i+1}"
            )
            
    def test_information_distance(self):
        """测试信息距离度量"""
        # 验证基于熵的时间度量
        for i in range(len(self.states) - 1):
            s1 = self.states[i]
            s2 = self.states[i + 1]
            
            # 计算熵增
            entropy_increase = s2.entropy() - s1.entropy()
            
            # 验证熵增为正（第二定律）
            self.assertGreaterEqual(
                entropy_increase, 0,
                f"Entropy should increase from t={i} to t={i+1}"
            )
            
    def test_discreteness(self):
        """测试离散性"""
        # 收集所有非零距离
        non_zero_distances = []
        
        for i, state_i in enumerate(self.states):
            for j, state_j in enumerate(self.states):
                if i != j:
                    distance = abs(self.time_metric.distance(state_i, state_j))
                    non_zero_distances.append(distance)
                    
        # 验证最小非零距离存在
        if non_zero_distances:
            min_distance = min(non_zero_distances)
            self.assertGreater(
                min_distance, 0,
                "Minimum non-zero distance should be positive (discrete time)"
            )
            
    def test_irreversibility(self):
        """测试不可逆性"""
        # 验证系统状态演化的不可逆性
        for i in range(len(self.states) - 1):
            s1 = self.states[i]
            s2 = self.states[i + 1]
            
            # 验证元素只增不减（简化的不可逆性）
            self.assertTrue(
                s1.elements.issubset(s2.elements),
                f"Elements should only increase (irreversibility) from t={i} to t={i+1}"
            )
            
            # 验证新元素的存在
            new_elements = s2.elements - s1.elements
            self.assertGreater(
                len(new_elements), 0,
                f"New elements should emerge from t={i} to t={i+1}"
            )
            
    def test_cumulative_time(self):
        """测试累积时间"""
        # 计算总时间
        total_time = 0
        for i in range(len(self.states) - 1):
            step_time = self.time_metric.distance(self.states[i], self.states[i+1])
            total_time += step_time
            
        # 验证总时间等于首尾距离
        direct_distance = self.time_metric.distance(self.states[0], self.states[-1])
        self.assertAlmostEqual(
            total_time, direct_distance,
            places=10,
            msg="Total time should equal direct distance from start to end"
        )
        
    def test_time_metric_properties(self):
        """测试时间度量的所有性质"""
        # 使用验证器检查所有性质
        properties = self.time_metric.verify_time_properties(self.states)
        
        # 验证单调性
        self.assertTrue(
            properties["monotonic"],
            "Time should be monotonic"
        )
        
        # 验证非负性（对于前向时间）
        self.assertTrue(
            properties["non_negative"],
            "Forward time distances should be non-negative"
        )
        
        # 验证可加性
        self.assertTrue(
            properties["additive"],
            "Time metric should satisfy additivity"
        )


if __name__ == "__main__":
    unittest.main()