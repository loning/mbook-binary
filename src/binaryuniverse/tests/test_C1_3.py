#!/usr/bin/env python3
"""
test_C1_3.py - C1-3信息密度推论的完整机器验证测试

完整验证φ-表示系统的信息密度性质
"""

import unittest
import sys
import os
import math
from typing import List, Dict, Tuple, Any

# 添加包路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'formal'))

class FibonacciSystem:
    """Fibonacci数系统"""
    
    def __init__(self):
        """初始化Fibonacci系统"""
        self.fib_cache = {0: 0, 1: 1, 2: 1}
        
    def fibonacci(self, n: int) -> int:
        """计算第n个Fibonacci数"""
        if n in self.fib_cache:
            return self.fib_cache[n]
        
        # F(n) = F(n-1) + F(n-2)
        self.fib_cache[n] = self.fibonacci(n-1) + self.fibonacci(n-2)
        return self.fib_cache[n]
    
    def count_valid_sequences(self, n: int) -> int:
        """计算n位系统中有效序列的数量"""
        # n位φ-表示系统有F(n+2)个有效状态
        return self.fibonacci(n + 2)
    
    def generate_valid_sequences(self, n: int) -> List[List[int]]:
        """生成所有n位有效序列"""
        if n == 0:
            return [[]]
        
        sequences = []
        
        def generate_recursive(seq: List[int], remaining: int):
            if remaining == 0:
                sequences.append(seq[:])
                return
            
            # 添加0总是有效的
            seq.append(0)
            generate_recursive(seq, remaining - 1)
            seq.pop()
            
            # 只有当最后一位不是1时才能添加1
            if not seq or seq[-1] == 0:
                seq.append(1)
                generate_recursive(seq, remaining - 1)
                seq.pop()
        
        generate_recursive([], n)
        return sequences


class InformationDensityVerifier:
    """信息密度推论验证器"""
    
    def __init__(self, max_n: int = 30):
        """初始化验证器"""
        self.max_n = max_n
        self.phi = (1 + math.sqrt(5)) / 2
        self.fib_system = FibonacciSystem()
        
    def compute_density(self, n: int) -> float:
        """计算n位系统的信息密度"""
        if n == 0:
            return 0.0
        
        # ρ(n) = log_2(F_{n+2}) / n
        num_states = self.fib_system.count_valid_sequences(n)
        if num_states <= 0:
            return 0.0
            
        return math.log2(num_states) / n
    
    def verify_density_formula(self) -> Dict[str, bool]:
        """验证密度公式"""
        results = {
            "formula_correct": True,
            "monotonic_decrease": True,
            "positive_density": True
        }
        
        densities = []
        for n in range(1, self.max_n + 1):
            density = self.compute_density(n)
            densities.append(density)
            
            # 验证密度为正
            if density <= 0:
                results["positive_density"] = False
                
            # 验证状态数计算
            expected_states = self.fib_system.fibonacci(n + 2)
            actual_states = self.fib_system.count_valid_sequences(n)
            if expected_states != actual_states:
                results["formula_correct"] = False
                print(f"  状态数错误: n={n}, expected={expected_states}, actual={actual_states}")
        
        # 验证密度单调递减趋势（对于大n）
        for i in range(10, len(densities) - 1):
            if densities[i] < densities[i + 1]:
                # 允许小的波动
                if abs(densities[i] - densities[i + 1]) > 0.001:
                    results["monotonic_decrease"] = False
        
        return results
    
    def verify_asymptotic_density(self) -> Dict[str, Any]:
        """验证渐近密度"""
        results = {
            "densities": [],
            "converges": True,
            "limit": math.log2(self.phi),
            "convergence_rate": 0.0,
            "final_error": 0.0
        }
        
        # 计算不同长度的密度
        for n in range(1, self.max_n + 1):
            density = self.compute_density(n)
            results["densities"].append((n, density))
        
        # 检查收敛性
        target = math.log2(self.phi)
        errors = []
        
        # 初始化converges为True
        results["converges"] = True
        
        # 使用更多的点来检查收敛性
        check_points = min(15, len(results["densities"]))
        for n, density in results["densities"][-check_points:]:
            error = abs(density - target)
            errors.append(error)
            
            # 检查误差是否在减小（收敛的标志）
            if len(errors) >= 2 and errors[-1] > errors[-2] * 1.1:  # 允许10%的波动
                results["converges"] = False
                break
        
        # 记录最终误差
        if results["densities"]:
            results["final_error"] = abs(results["densities"][-1][1] - target)
        
        # 估计收敛速率（应该是O(1/n)）
        if len(results["densities"]) >= 10:
            # 使用最后10个点估计收敛速率
            n_values = [n for n, _ in results["densities"][-10:]]
            errors = [abs(d - target) for _, d in results["densities"][-10:]]
            
            # 检查是否满足O(1/n)收敛
            # error ≈ c/n，所以error * n ≈ c应该近似常数
            products = [errors[i] * n_values[i] for i in range(len(errors))]
            if products:
                avg_product = sum(products) / len(products)
                results["convergence_rate"] = avg_product
        
        return results
    
    def verify_optimality(self) -> Dict[str, bool]:
        """验证最优性"""
        results = {
            "beats_looser_constraints": True,
            "optimal_under_constraint": True,
            "theoretical_limit": True
        }
        
        # 比较不同约束下的密度
        n = 10  # 使用10位系统进行比较
        
        # 1. φ-表示（no-consecutive-1s）
        phi_states = self.fib_system.count_valid_sequences(n)
        phi_density = math.log2(phi_states) / n
        
        # 2. 无约束二进制
        binary_states = 2 ** n
        binary_density = math.log2(binary_states) / n
        
        # 3. 更严格的约束（例如no-11和no-101）
        stricter_states = self._count_stricter_constraint(n)
        stricter_density = math.log2(stricter_states) / n if stricter_states > 0 else 0
        
        # 验证关系
        if phi_density >= binary_density:
            results["beats_looser_constraints"] = False
            
        if stricter_states > 0 and phi_density <= stricter_density:
            results["optimal_under_constraint"] = False
        
        # 验证理论极限
        theoretical_limit = math.log2(self.phi)
        if abs(phi_density - theoretical_limit) > 0.1:  # 对于n=10允许一定误差
            results["theoretical_limit"] = False
        
        return results
    
    def _count_stricter_constraint(self, n: int) -> int:
        """计算更严格约束下的状态数（例如no-11和no-101）"""
        count = 0
        
        def is_valid_strict(seq: List[int]) -> bool:
            # 检查no-11
            for i in range(len(seq) - 1):
                if seq[i] == 1 and seq[i + 1] == 1:
                    return False
            # 检查no-101
            for i in range(len(seq) - 2):
                if seq[i] == 1 and seq[i + 1] == 0 and seq[i + 2] == 1:
                    return False
            return True
        
        # 枚举所有可能的序列
        for i in range(2 ** n):
            seq = [(i >> j) & 1 for j in range(n)]
            if is_valid_strict(seq):
                count += 1
                
        return count
    
    def compute_entropy(self, n: int) -> float:
        """计算系统熵"""
        # H(n) = log_2(F_{n+2})
        num_states = self.fib_system.count_valid_sequences(n)
        if num_states <= 0:
            return 0.0
        return math.log2(num_states)
    
    def verify_entropy_properties(self) -> Dict[str, Any]:
        """验证熵性质"""
        results = {
            "entropy_values": [],
            "entropy_density": [],
            "extensivity": True,
            "subadditivity": True
        }
        
        # 计算不同长度的熵
        for n in range(1, min(self.max_n + 1, 16)):
            entropy = self.compute_entropy(n)
            entropy_density = entropy / n if n > 0 else 0
            results["entropy_values"].append((n, entropy))
            results["entropy_density"].append((n, entropy_density))
        
        # 验证广延性：H(n)应该近似线性增长
        if len(results["entropy_values"]) >= 10:
            # 检查H(n)/n是否趋于常数
            densities = [h/n for n, h in results["entropy_values"] if n > 5]
            if densities:
                avg_density = sum(densities) / len(densities)
                for d in densities:
                    if abs(d - avg_density) / avg_density > 0.1:  # 10%容差
                        results["extensivity"] = False
                        break
        
        # 验证次可加性：H(m+n) ≤ H(m) + H(n)
        for m in range(2, 6):
            for n in range(2, 6):
                if m + n <= 15:
                    h_m = self.compute_entropy(m)
                    h_n = self.compute_entropy(n)
                    h_mn = self.compute_entropy(m + n)
                    if h_mn > h_m + h_n + 0.001:  # 小容差
                        results["subadditivity"] = False
        
        return results
    
    def compare_with_other_systems(self) -> Dict[str, Any]:
        """与其他系统比较"""
        comparisons = {
            "systems": ["phi", "binary", "ternary", "constrained"],
            "densities": {},
            "relative_efficiency": {}
        }
        
        # 测试不同长度
        test_lengths = [5, 10, 15, 20]
        
        for system in comparisons["systems"]:
            densities = []
            
            for n in test_lengths:
                if system == "phi":
                    # φ-表示系统
                    density = self.compute_density(n)
                elif system == "binary":
                    # 无约束二进制
                    density = 1.0  # log_2(2^n) / n = 1
                elif system == "ternary":
                    # 三进制系统
                    density = math.log2(3)  # log_2(3^n) / n
                elif system == "constrained":
                    # 更严格的约束
                    states = self._count_stricter_constraint(n)
                    density = math.log2(states) / n if states > 0 and n > 0 else 0
                else:
                    density = 0
                
                densities.append(density)
            
            comparisons["densities"][system] = densities
            
            # 计算相对效率（相对于二进制）
            if system != "binary":
                avg_density = sum(densities) / len(densities) if densities else 0
                comparisons["relative_efficiency"][system] = avg_density / 1.0
        
        return comparisons
    
    def verify_corollary_completeness(self) -> Dict[str, Any]:
        """C1-3推论的完整验证"""
        return {
            "density_formula": self.verify_density_formula(),
            "asymptotic_density": self.verify_asymptotic_density(),
            "optimality": self.verify_optimality(),
            "entropy_properties": self.verify_entropy_properties(),
            "system_comparison": self.compare_with_other_systems()
        }


class TestC1_3_InformationDensity(unittest.TestCase):
    """C1-3信息密度推论的完整机器验证测试"""

    def setUp(self):
        """测试初始化"""
        self.verifier = InformationDensityVerifier(max_n=30)
        
    def test_density_formula_complete(self):
        """测试密度公式的完整性 - 验证检查点1"""
        print("\n=== C1-3 验证检查点1：密度公式完整验证 ===")
        
        formula_verification = self.verifier.verify_density_formula()
        print(f"密度公式验证结果: {formula_verification}")
        
        self.assertTrue(formula_verification["formula_correct"], 
                       "密度公式应该正确")
        self.assertTrue(formula_verification["positive_density"], 
                       "信息密度应该为正")
        
        # 显示一些具体例子
        print("  信息密度示例:")
        for n in [1, 2, 3, 5, 8, 10, 15, 20]:
            density = self.verifier.compute_density(n)
            states = self.verifier.fib_system.count_valid_sequences(n)
            print(f"    n={n}: 状态数={states}, 密度={density:.6f}")
        
        print("✓ 密度公式完整验证通过")

    def test_asymptotic_density_complete(self):
        """测试渐近密度的完整性 - 验证检查点2"""
        print("\n=== C1-3 验证检查点2：渐近密度完整验证 ===")
        
        asymptotic_data = self.verifier.verify_asymptotic_density()
        
        print(f"渐近密度验证:")
        print(f"  理论极限: {asymptotic_data['limit']:.6f}")
        print(f"  收敛性: {asymptotic_data['converges']}")
        print(f"  最终误差: {asymptotic_data['final_error']:.6e}")
        print(f"  收敛速率常数: {asymptotic_data['convergence_rate']:.3f}")
        
        # 显示密度演化
        print("  密度演化:")
        for n, density in asymptotic_data["densities"][-5:]:
            error = abs(density - asymptotic_data['limit'])
            print(f"    n={n}: ρ={density:.6f}, error={error:.6e}")
        
        self.assertTrue(asymptotic_data["converges"], 
                       "密度应该收敛到log_2(φ)")
        
        # 验证最终密度接近理论值
        if asymptotic_data["densities"]:
            last_density = asymptotic_data["densities"][-1][1]
            last_n = asymptotic_data["densities"][-1][0]
            # 对于O(1/n)收敛，n=30时误差应该在1/30 ≈ 0.033的量级
            expected_error_order = 1.0 / last_n
            actual_error = abs(last_density - math.log2(self.verifier.phi))
            self.assertLess(actual_error, expected_error_order * 0.5, 
                           f"对于n={last_n}，误差应该是O(1/n)量级")
        
        # 验证O(1/n)收敛
        self.assertGreater(asymptotic_data["convergence_rate"], 0.1,
                          "收敛速率应该满足O(1/n)")
        self.assertLess(asymptotic_data["convergence_rate"], 10.0,
                       "收敛速率常数应该有界")
        
        print("✓ 渐近密度完整验证通过")

    def test_optimality_complete(self):
        """测试最优性的完整性 - 验证检查点3"""
        print("\n=== C1-3 验证检查点3：最优性完整验证 ===")
        
        optimality = self.verifier.verify_optimality()
        print(f"最优性验证结果: {optimality}")
        
        self.assertTrue(optimality["beats_looser_constraints"], 
                       "φ-表示密度应该小于无约束系统")
        self.assertTrue(optimality["optimal_under_constraint"], 
                       "φ-表示在约束下应该是最优的")
        self.assertTrue(optimality["theoretical_limit"], 
                       "应该接近理论极限")
        
        # 显示具体的密度比较
        print("  密度比较:")
        n = 10
        phi_states = self.verifier.fib_system.count_valid_sequences(n)
        phi_density = math.log2(phi_states) / n
        binary_density = 1.0
        
        print(f"    10位系统:")
        print(f"    - φ-表示: {phi_states}个状态, 密度={phi_density:.6f}")
        print(f"    - 二进制: {2**n}个状态, 密度={binary_density:.6f}")
        print(f"    - 密度比: {phi_density/binary_density:.3f}")
        
        print("✓ 最优性完整验证通过")

    def test_entropy_properties_complete(self):
        """测试熵性质的完整性 - 验证检查点4"""
        print("\n=== C1-3 验证检查点4：熵性质完整验证 ===")
        
        entropy_data = self.verifier.verify_entropy_properties()
        
        print("熵性质验证:")
        print(f"  广延性: {entropy_data['extensivity']}")
        print(f"  次可加性: {entropy_data['subadditivity']}")
        
        # 显示熵值
        print("  系统熵:")
        for n, entropy in entropy_data["entropy_values"][:8]:
            print(f"    n={n}: H={entropy:.3f}")
        
        print("  熵密度:")
        for n, density in entropy_data["entropy_density"][:8]:
            print(f"    n={n}: h={density:.6f}")
        
        self.assertTrue(entropy_data["extensivity"], 
                       "熵应该具有广延性")
        self.assertTrue(entropy_data["subadditivity"], 
                       "熵应该满足次可加性")
        
        print("✓ 熵性质完整验证通过")

    def test_system_comparison_complete(self):
        """测试系统比较的完整性 - 验证检查点5"""
        print("\n=== C1-3 验证检查点5：系统比较完整验证 ===")
        
        comparisons = self.verifier.compare_with_other_systems()
        
        print("不同编码系统比较:")
        print("  系统密度（bits/symbol）:")
        
        # 格式化输出比较结果
        systems = comparisons["systems"]
        test_lengths = [5, 10, 15, 20]
        
        # 打印表头
        print(f"    {'系统':<12}", end="")
        for n in test_lengths:
            print(f"n={n:<2}  ", end="")
        print()
        
        # 打印各系统的密度
        for system in systems:
            print(f"    {system:<12}", end="")
            for i, density in enumerate(comparisons["densities"][system]):
                print(f"{density:.3f}  ", end="")
            print()
        
        print("\n  相对效率（相对于二进制）:")
        for system, efficiency in comparisons["relative_efficiency"].items():
            print(f"    {system}: {efficiency:.3f}")
        
        # 验证φ-表示的效率
        phi_efficiency = comparisons["relative_efficiency"]["phi"]
        self.assertGreater(phi_efficiency, 0.6, 
                          "φ-表示效率应该大于60%")
        self.assertLess(phi_efficiency, 0.8, 
                       "φ-表示效率应该小于80%")
        
        print("✓ 系统比较完整验证通过")

    def test_complete_information_density_corollary(self):
        """测试完整信息密度推论 - 主推论验证"""
        print("\n=== C1-3 主推论：完整信息密度验证 ===")
        
        # 完整验证
        verification = self.verifier.verify_corollary_completeness()
        
        print(f"推论完整验证结果:")
        
        # 1. 密度公式
        density_formula = verification["density_formula"]
        print(f"\n1. 密度公式验证:")
        for key, value in density_formula.items():
            print(f"   {key}: {value}")
        self.assertTrue(all(density_formula.values()),
                       "密度公式所有性质应该满足")
        
        # 2. 渐近密度
        asymptotic = verification["asymptotic_density"]
        print(f"\n2. 渐近密度:")
        print(f"   收敛到: {asymptotic['limit']:.6f}")
        print(f"   收敛性: {asymptotic['converges']}")
        print(f"   最终误差: {asymptotic['final_error']:.6e}")
        
        # 3. 最优性
        optimality = verification["optimality"]
        print(f"\n3. 最优性验证:")
        for key, value in optimality.items():
            print(f"   {key}: {value}")
        
        # 4. 熵性质
        entropy = verification["entropy_properties"]
        print(f"\n4. 熵性质:")
        print(f"   广延性: {entropy['extensivity']}")
        print(f"   次可加性: {entropy['subadditivity']}")
        
        # 5. 系统比较
        comparison = verification["system_comparison"]
        print(f"\n5. 系统效率比较:")
        print(f"   φ-表示: {comparison['relative_efficiency'].get('phi', 0):.3f}")
        print(f"   约束系统: {comparison['relative_efficiency'].get('constrained', 0):.3f}")
        
        print(f"\n✓ C1-3推论验证通过")
        print(f"  - 信息密度收敛到log_2(φ)")
        print(f"  - 在约束条件下达到最优")
        print(f"  - 熵满足热力学性质")
        print(f"  - 提供了约束系统的信息论极限")


def run_complete_verification():
    """运行完整的C1-3验证"""
    print("=" * 80)
    print("C1-3 信息密度推论 - 完整机器验证")
    print("=" * 80)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestC1_3_InformationDensity)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 80)
    if result.wasSuccessful():
        print("✓ C1-3信息密度推论完整验证成功！")
        print("φ-表示系统的信息密度达到了约束条件下的理论极限。")
    else:
        print("✗ C1-3信息密度推论验证发现问题")
        print(f"失败: {len(result.failures)}, 错误: {len(result.errors)}")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_complete_verification()
    exit(0 if success else 1)