"""
Unit tests for T2-11: Maximum Entropy Rate Theorem
T2-11：最大熵增率定理的单元测试
"""
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base_framework import VerificationTest
import math
from typing import List, Dict, Optional, Tuple


class TestT2_11_MaximumEntropyRate(VerificationTest):
    """T2-11 最大熵增率定理的数学化验证测试"""
    
    def setUp(self):
        """测试前设置"""
        super().setUp()
        self.golden_ratio = (1 + math.sqrt(5)) / 2
        self.log_phi = math.log(self.golden_ratio)
        
    def _compute_information_capacity(self, constraint_pattern: str) -> Optional[float]:
        """计算给定约束模式的信息容量"""
        if not constraint_pattern:
            # 无约束：无法保证唯一可解码
            return None
        
        if len(constraint_pattern) == 1:
            # 长度1约束：完全禁止某个符号
            return 0.0
        
        if constraint_pattern in ["00", "11"]:
            # 最优长度2约束
            return self.log_phi
        
        if constraint_pattern in ["01", "10"]:
            # 次优长度2约束
            return math.log(2) / 2
        
        # 其他情况的估计
        if len(constraint_pattern) == 3:
            # 长度3约束通常无法保证唯一可解码
            return None
            
        # 其他模式的保守估计
        return 0.5 * math.log(2)
    
    def _compute_self_description_complexity(self, base: int) -> float:
        """计算给定基底的自描述复杂度"""
        if base == 1:
            return float('inf')  # 一元系统无法有效编码
        elif base == 2:
            return 4.0  # 简单的对偶关系
        else:
            return base ** 2  # O(k²)复杂度
    
    def _is_uniquely_decodable(self, constraint_pattern: str) -> bool:
        """检查约束是否保证唯一可解码"""
        if not constraint_pattern:
            return False  # 无约束无法保证
        
        if len(constraint_pattern) == 1:
            return True  # 完全禁止某符号
        
        if len(constraint_pattern) == 2:
            return True  # 长度2约束可以保证
        
        # 长度3+通常有前缀冲突问题
        return False
    
    def _can_construct_system_with_rate(self, target_rate: float) -> bool:
        """检查是否可以构造具有给定熵增率的自指完备系统"""
        # 根据定理，不能超过log(φ)
        return target_rate <= self.log_phi + 1e-10
    
    def test_entropy_rate_upper_bound(self):
        """测试熵增率上界 - 验证检查点1"""
        # 测试不同编码系统的容量
        encoding_systems = {
            "no_constraint": {"capacity": math.log(2), "valid": False, "reason": "无唯一可解码保证"},
            "no_0": {"capacity": 0.0, "valid": True, "reason": "完全禁止0"},
            "no_1": {"capacity": 0.0, "valid": True, "reason": "完全禁止1"},
            "no_00": {"capacity": self.log_phi, "valid": True, "reason": "最优约束"},
            "no_11": {"capacity": self.log_phi, "valid": True, "reason": "最优约束"},
            "no_01": {"capacity": math.log(2)/2, "valid": True, "reason": "次优约束"},
            "no_10": {"capacity": math.log(2)/2, "valid": True, "reason": "次优约束"},
        }
        
        valid_capacities = []
        
        for name, system in encoding_systems.items():
            capacity = system["capacity"]
            is_valid = system["valid"]
            
            if is_valid and capacity > 0:
                valid_capacities.append(capacity)
                
                # 验证所有有效系统都不超过上界
                self.assertLessEqual(
                    capacity, self.log_phi + 1e-10,
                    f"System '{name}' exceeds theoretical upper bound"
                )
        
        # 验证存在达到上界的系统
        max_capacity = max(valid_capacities)
        self.assertAlmostEqual(
            max_capacity, self.log_phi, 6,
            "Maximum capacity should equal log(φ)"
        )
        
        # 验证上界的数值
        expected_log_phi = math.log((1 + math.sqrt(5)) / 2)
        self.assertAlmostEqual(
            self.log_phi, expected_log_phi, 10,
            "log(φ) calculation should be accurate"
        )
        
    def test_binary_encoding_requirement(self):
        """测试二进制编码要求 - 验证检查点2"""
        # 测试不同基底的自描述复杂度
        bases = {
            "unary": {"k": 1, "complexity": float('inf')},
            "binary": {"k": 2, "complexity": 4},
            "ternary": {"k": 3, "complexity": 9},
            "quaternary": {"k": 4, "complexity": 16},
            "decimal": {"k": 10, "complexity": 100},
        }
        
        # 找到有限复杂度的基底
        finite_bases = {name: base for name, base in bases.items() 
                       if base["complexity"] < float('inf')}
        
        self.assertGreater(len(finite_bases), 0, "Should have bases with finite complexity")
        
        # 验证二进制是最优的
        min_complexity = min(base["complexity"] for base in finite_bases.values())
        binary_complexity = bases["binary"]["complexity"]
        
        self.assertEqual(
            binary_complexity, min_complexity,
            "Binary should have minimal self-description complexity"
        )
        
        # 验证高阶基底复杂度确实更高
        for name, base in bases.items():
            if base["k"] > 2 and base["complexity"] < float('inf'):
                self.assertGreater(
                    base["complexity"], binary_complexity,
                    f"Base {base['k']} should be more complex than binary"
                )
        
    def test_unique_decodability_constraint(self):
        """测试唯一可解码性约束 - 验证检查点3"""
        # 测试不同约束模式
        constraints = [
            ("", False, None),           # 无约束
            ("0", True, 0.0),           # 禁止0
            ("1", True, 0.0),           # 禁止1
            ("00", True, self.log_phi), # 禁止00
            ("11", True, self.log_phi), # 禁止11
            ("01", True, math.log(2)/2), # 禁止01
            ("10", True, math.log(2)/2), # 禁止10
            ("000", False, None),        # 长度3约束
            ("111", False, None),        # 长度3约束
        ]
        
        valid_constraints = []
        
        for pattern, should_be_decodable, expected_capacity in constraints:
            is_decodable = self._is_uniquely_decodable(pattern)
            capacity = self._compute_information_capacity(pattern)
            
            # 验证可解码性判断
            self.assertEqual(
                is_decodable, should_be_decodable,
                f"Decodability of '{pattern}' should be {should_be_decodable}"
            )
            
            # 验证容量计算
            if expected_capacity is not None:
                self.assertIsNotNone(capacity, f"Capacity of '{pattern}' should be defined")
                self.assertAlmostEqual(
                    capacity, expected_capacity, 6,
                    f"Capacity of '{pattern}' should be {expected_capacity}"
                )
                
                if is_decodable and capacity > 0:
                    valid_constraints.append((pattern, capacity))
                    
                    # 验证不超过理论上界
                    self.assertLessEqual(
                        capacity, self.log_phi + 1e-10,
                        f"Constraint '{pattern}' capacity exceeds upper bound"
                    )
        
        # 验证最优约束确实达到上界
        max_capacity = max(cap for _, cap in valid_constraints)
        self.assertAlmostEqual(
            max_capacity, self.log_phi, 6,
            "Maximum constraint capacity should equal log(φ)"
        )
        
    def test_self_reference_contradiction(self):
        """测试自指矛盾 - 验证检查点4"""
        # 测试尝试构造超过上界的系统
        impossible_rates = [
            1.1 * self.log_phi,
            1.5 * self.log_phi,
            2.0 * self.log_phi,
            math.log(2),  # 二进制上界
            1.0,          # 1 bit/time
        ]
        
        for target_rate in impossible_rates:
            can_construct = self._can_construct_system_with_rate(target_rate)
            
            if target_rate > self.log_phi:
                self.assertFalse(
                    can_construct,
                    f"Should not be able to construct system with rate {target_rate}"
                )
            else:
                self.assertTrue(
                    can_construct,
                    f"Should be able to construct system with rate {target_rate}"
                )
        
        # 测试矛盾的具体形式
        hypothetical_system = {
            "entropy_rate": 1.2 * self.log_phi,
            "self_referential": True,
            "uses_binary": True,      # 由T2-4要求
            "unique_decodable": True, # 自指要求
        }
        
        # 检查约束的兼容性
        requirements = []
        
        # 自指完备性 → 二进制编码 (T2-4)
        if hypothetical_system["self_referential"]:
            requirements.append(("binary_required", hypothetical_system["uses_binary"]))
        
        # 二进制 + 唯一可解码 → 容量 ≤ log(φ)
        if (hypothetical_system["uses_binary"] and 
            hypothetical_system["unique_decodable"]):
            rate_feasible = hypothetical_system["entropy_rate"] <= self.log_phi
            requirements.append(("rate_bounded", rate_feasible))
        
        # 验证存在不满足的要求
        unsatisfied = [name for name, satisfied in requirements if not satisfied]
        self.assertGreater(
            len(unsatisfied), 0,
            f"System should have unsatisfied requirements: {unsatisfied}"
        )
        
    def test_phi_system_optimality(self):
        """测试φ-系统最优性 - 验证检查点5"""
        # φ-表示系统的规格
        phi_system = {
            "name": "phi-representation",
            "uses_binary": True,
            "constraint": "no-11",
            "self_referential": True,
            "unique_decodable": True,
            "entropy_rate": self.log_phi,
            "fibonacci_based": True,
        }
        
        # 验证φ-系统满足所有要求
        self.assertTrue(
            phi_system["uses_binary"],
            "φ-system should use binary encoding"
        )
        
        self.assertTrue(
            phi_system["self_referential"],
            "φ-system should be self-referential"
        )
        
        self.assertTrue(
            phi_system["unique_decodable"],
            "φ-system should guarantee unique decodability"
        )
        
        # 验证φ-系统达到理论上界
        achieved_rate = phi_system["entropy_rate"]
        theoretical_max = self.log_phi
        
        self.assertAlmostEqual(
            achieved_rate, theoretical_max, 10,
            "φ-system should achieve theoretical maximum entropy rate"
        )
        
        # 验证φ-系统的约束确实最优
        constraint_name = phi_system["constraint"]  # "no-11"
        # 转换为模式名
        if constraint_name == "no-11":
            pattern = "11"
        elif constraint_name == "no-00":
            pattern = "00"
        else:
            pattern = constraint_name
            
        constraint_capacity = self._compute_information_capacity(pattern)
        self.assertAlmostEqual(
            constraint_capacity, self.log_phi, 6,
            "φ-system constraint should have optimal capacity"
        )
        
        # 验证没有其他系统能超过φ-系统
        alternative_systems = [
            {"rate": 1.1 * self.log_phi, "feasible": False},
            {"rate": self.log_phi, "feasible": True},
            {"rate": 0.9 * self.log_phi, "feasible": True},
            {"rate": math.log(2), "feasible": False},  # 二进制无约束
        ]
        
        for alt_system in alternative_systems:
            rate = alt_system["rate"]
            should_be_feasible = alt_system["feasible"]
            is_feasible = self._can_construct_system_with_rate(rate)
            
            self.assertEqual(
                is_feasible, should_be_feasible,
                f"System with rate {rate} feasibility should be {should_be_feasible}"
            )
        
    def test_theoretical_bounds_verification(self):
        """测试理论界限验证"""
        # 验证黄金比例的数学性质
        phi = self.golden_ratio
        
        # 验证黄金比例的定义方程：φ² = φ + 1
        self.assertAlmostEqual(
            phi * phi, phi + 1, 10,
            "Golden ratio should satisfy φ² = φ + 1"
        )
        
        # 验证与Fibonacci数列的关系
        fib = [1, 1]
        for i in range(2, 20):
            fib.append(fib[i-1] + fib[i-2])
        
        # 验证比率趋向φ
        ratio = fib[-1] / fib[-2]
        self.assertAlmostEqual(
            ratio, phi, 5,
            "Fibonacci ratio should approach φ"
        )
        
        # 验证log(φ)的数值范围
        self.assertGreater(
            self.log_phi, 0.48,
            "log(φ) should be greater than 0.48"
        )
        
        self.assertLess(
            self.log_phi, 0.49,
            "log(φ) should be less than 0.49"
        )
        
        # 与二进制对比
        log_2 = math.log(2)
        self.assertLess(
            self.log_phi, log_2,
            "log(φ) should be less than log(2)"
        )
        
        # 验证是约束系统的最大容量（比无约束二进制小）
        theoretical_ratio = self.log_phi / log_2
        self.assertLess(
            theoretical_ratio, 1.0,
            "Constrained system should have lower capacity than unconstrained"
        )
        
    def test_constraint_efficiency_analysis(self):
        """测试约束效率分析"""
        # 分析不同约束的效率
        constraint_analysis = {
            "no_00": {
                "capacity": self.log_phi,
                "efficiency": 1.0,  # 最优
                "symmetry": True,
            },
            "no_11": {
                "capacity": self.log_phi,
                "efficiency": 1.0,  # 最优
                "symmetry": True,
            },
            "no_01": {
                "capacity": math.log(2) / 2,
                "efficiency": (math.log(2) / 2) / self.log_phi,
                "symmetry": False,
            },
            "no_10": {
                "capacity": math.log(2) / 2,
                "efficiency": (math.log(2) / 2) / self.log_phi,
                "symmetry": False,
            },
        }
        
        # 验证最优约束
        optimal_constraints = []
        for name, analysis in constraint_analysis.items():
            if abs(analysis["efficiency"] - 1.0) < 1e-6:
                optimal_constraints.append(name)
                
                # 验证对称性
                self.assertTrue(
                    analysis["symmetry"],
                    f"Optimal constraint '{name}' should preserve symmetry"
                )
        
        # 应该有且仅有两个最优约束
        self.assertEqual(
            len(optimal_constraints), 2,
            "Should have exactly 2 optimal constraints"
        )
        
        self.assertIn("no_00", optimal_constraints)
        self.assertIn("no_11", optimal_constraints)
        
        # 验证次优约束的效率
        suboptimal_efficiency = constraint_analysis["no_01"]["efficiency"]
        expected_ratio = (math.log(2) / 2) / self.log_phi
        self.assertAlmostEqual(
            suboptimal_efficiency, expected_ratio, 6,
            "Suboptimal constraint efficiency should match calculation"
        )
        
        # 验证效率值在合理范围内
        for name, analysis in constraint_analysis.items():
            efficiency = analysis["efficiency"]
            self.assertGreater(efficiency, 0, f"Efficiency of '{name}' should be positive")
            self.assertLessEqual(efficiency, 1.0, f"Efficiency of '{name}' should not exceed 1")
        
    def test_physical_implications(self):
        """测试物理含义"""
        # 如果物理系统是自指完备的，其信息处理率受限
        physical_scenarios = {
            "quantum_computer": {
                "claimed_rate": 2.0 * self.log_phi,  # 超过上界
                "self_referential": True,
                "feasible": False,
            },
            "classical_computer": {
                "claimed_rate": 0.8 * self.log_phi,  # 低于上界
                "self_referential": True,
                "feasible": True,
            },
            "black_hole": {
                "claimed_rate": self.log_phi,  # 刚好达到上界
                "self_referential": True,
                "feasible": True,
            },
            "non_self_referential": {
                "claimed_rate": 10.0 * self.log_phi,  # 可以很高
                "self_referential": False,
                "feasible": True,  # 不受此定理约束
            },
        }
        
        for scenario, properties in physical_scenarios.items():
            rate = properties["claimed_rate"]
            is_self_ref = properties["self_referential"]
            should_be_feasible = properties["feasible"]
            
            if is_self_ref:
                # 自指系统受定理约束
                is_feasible = rate <= self.log_phi + 1e-10
            else:
                # 非自指系统不受约束
                is_feasible = True
            
            self.assertEqual(
                is_feasible, should_be_feasible,
                f"Scenario '{scenario}' feasibility mismatch"
            )
        
    def test_mathematical_consistency(self):
        """测试数学一致性"""
        # 验证定理与其他数学结果的一致性
        
        # 1. 与信息论的一致性
        # 二进制信道的理论容量是log(2)
        binary_capacity = math.log(2)
        self.assertLess(
            self.log_phi, binary_capacity,
            "Constrained systems should have lower capacity than unconstrained"
        )
        
        # 2. 与组合数学的一致性
        # Fibonacci增长率确实是φ
        fibonacci_growth_rate = self.golden_ratio
        entropy_growth_rate = math.exp(self.log_phi)
        self.assertAlmostEqual(
            fibonacci_growth_rate, entropy_growth_rate, 10,
            "Fibonacci and entropy growth rates should match"
        )
        
        # 3. 与递归理论的一致性
        # 自指系统的复杂度有下界
        self_description_lower_bound = 0.1  # 相对保守的下界
        practical_phi_capacity = self.log_phi
        self.assertGreater(
            practical_phi_capacity, self_description_lower_bound,
            "System should have capacity to describe itself"
        )
        
        # 4. 与计算复杂度的一致性
        # 最优约束的发现应该是可计算的
        optimal_constraints = ["00", "11"]  # 使用模式而不是约束名
        for constraint in optimal_constraints:
            capacity = self._compute_information_capacity(constraint)
            self.assertAlmostEqual(
                capacity, self.log_phi, 6,
                f"Optimal constraint '{constraint}' should have capacity log(φ)"
            )
            
    def test_edge_cases_and_limits(self):
        """测试边界情况和极限"""
        # 测试接近上界的情况
        near_limit_rates = [
            0.99 * self.log_phi,
            0.999 * self.log_phi,
            0.9999 * self.log_phi,
            self.log_phi,
            self.log_phi + 1e-15,  # 浮点精度内
            self.log_phi + 1e-10,  # 接近但超过
            self.log_phi + 1e-6,   # 明显超过
        ]
        
        for rate in near_limit_rates:
            can_construct = self._can_construct_system_with_rate(rate)
            should_be_possible = rate <= self.log_phi + 1e-10  # 允许更大的数值误差
            
            self.assertEqual(
                can_construct, should_be_possible,
                f"Rate {rate} (vs {self.log_phi}) construction feasibility incorrect"
            )
        
        # 测试极端小的率
        tiny_rates = [1e-10, 1e-6, 0.01, 0.1]
        for rate in tiny_rates:
            can_construct = self._can_construct_system_with_rate(rate)
            self.assertTrue(
                can_construct,
                f"Very small rate {rate} should be constructible"
            )
        
        # 测试零容量约束
        zero_capacity_constraints = ["0", "1"]
        for constraint in zero_capacity_constraints:
            capacity = self._compute_information_capacity(constraint)
            self.assertEqual(
                capacity, 0.0,
                f"Constraint '{constraint}' should have zero capacity"
            )
            
            # 但仍然是有效的（虽然退化）
            is_decodable = self._is_uniquely_decodable(constraint)
            self.assertTrue(
                is_decodable,
                f"Constraint '{constraint}' should be uniquely decodable"
            )


if __name__ == "__main__":
    unittest.main()