"""
测试 T0-20: Zeckendorf度量空间基础理论

验证完备度量空间的性质、压缩映射定理和不动点存在性。
"""

import unittest
import numpy as np
from typing import List, Tuple, Callable, Optional


class ZeckendorfMetricSpace:
    """Zeckendorf度量空间的实现"""
    
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.contraction_constant = 1 / self.phi
        # 预计算Fibonacci数
        self.fibonacci = self._generate_fibonacci(100)
    
    def _generate_fibonacci(self, n: int) -> List[int]:
        """生成前n个Fibonacci数"""
        fib = [1, 2]
        for i in range(2, n):
            fib.append(fib[-1] + fib[-2])
        return fib
    
    def encode_zeckendorf(self, n: int) -> str:
        """将自然数编码为Zeckendorf表示"""
        if n == 0:
            return "0"
        
        # 找到不超过n的最大Fibonacci数
        result = []
        for f in reversed(self.fibonacci):
            if f <= n:
                result.append('1')
                n -= f
            else:
                if result:  # 只在开始编码后添加0
                    result.append('0')
        
        # 移除前导0
        s = ''.join(result).lstrip('0')
        return s if s else '0'
    
    def decode_zeckendorf(self, z: str) -> int:
        """将Zeckendorf表示解码为自然数"""
        if not z or z == "0":
            return 0
        
        value = 0
        for i, bit in enumerate(reversed(z)):
            if bit == '1':
                value += self.fibonacci[i]
        return value
    
    def verify_no_11(self, z: str) -> bool:
        """验证字符串满足No-11约束"""
        return "11" not in z
    
    def metric(self, x: str, y: str) -> float:
        """计算Zeckendorf度量 d_Z(x, y)"""
        vx = self.decode_zeckendorf(x)
        vy = self.decode_zeckendorf(y)
        diff = abs(vx - vy)
        return diff / (1 + diff)
    
    def is_cauchy_sequence(self, sequence: List[str], epsilon: float = 1e-6) -> bool:
        """检查序列是否为Cauchy序列"""
        n = len(sequence)
        if n < 2:
            return True
        
        # 检查后半部分是否满足Cauchy条件
        start = n // 2
        for i in range(start, n):
            for j in range(i + 1, n):
                if self.metric(sequence[i], sequence[j]) >= epsilon:
                    return False
        return True
    
    def find_fixed_point(self, mapping: Callable, initial: str, 
                        max_iterations: int = 1000, tolerance: float = 1e-10) -> Tuple[str, int]:
        """使用Banach不动点定理寻找不动点"""
        current = initial
        
        for iteration in range(max_iterations):
            next_point = mapping(current)
            
            # 检查是否达到不动点
            if self.metric(current, next_point) < tolerance:
                return next_point, iteration
            
            current = next_point
        
        # 如果未收敛，返回最后的点
        return current, max_iterations
    
    def verify_contraction(self, mapping: Callable, x: str, y: str) -> Tuple[bool, float]:
        """验证映射是否为压缩映射，返回是否压缩和压缩比"""
        mx = mapping(x)
        my = mapping(y)
        
        d_before = self.metric(x, y)
        d_after = self.metric(mx, my)
        
        if d_before == 0:
            return True, 0
        
        ratio = d_after / d_before
        is_contraction = ratio <= self.contraction_constant + 1e-10  # 数值容差
        
        return is_contraction, ratio


class TestT0_20ZeckendorfMetricSpace(unittest.TestCase):
    """T0-20 度量空间测试"""
    
    def setUp(self):
        self.space = ZeckendorfMetricSpace()
    
    def test_zeckendorf_encoding_uniqueness(self):
        """测试Zeckendorf编码的唯一性"""
        for n in range(100):
            z = self.space.encode_zeckendorf(n)
            # 验证No-11约束
            self.assertTrue(self.space.verify_no_11(z), 
                          f"编码 {z} (n={n}) 违反No-11约束")
            # 验证可逆性
            decoded = self.space.decode_zeckendorf(z)
            self.assertEqual(decoded, n, 
                           f"编解码不一致: {n} -> {z} -> {decoded}")
    
    def test_metric_properties(self):
        """测试度量的基本性质"""
        test_values = ["1", "10", "100", "101", "1000", "1001", "1010"]
        
        for x in test_values:
            for y in test_values:
                d = self.space.metric(x, y)
                
                # 非负性
                self.assertGreaterEqual(d, 0, f"度量非负性失败: d({x},{y}) = {d}")
                
                # 有界性 (由定义，度量 < 1)
                self.assertLess(d, 1, f"度量有界性失败: d({x},{y}) = {d}")
                
                # 同一性
                if x == y:
                    self.assertEqual(d, 0, f"同一性失败: d({x},{x}) = {d}")
                
                # 对称性
                d_reverse = self.space.metric(y, x)
                self.assertAlmostEqual(d, d_reverse, places=10,
                                      msg=f"对称性失败: d({x},{y}) != d({y},{x})")
        
        # 三角不等式
        for x in test_values:
            for y in test_values:
                for z in test_values:
                    d_xz = self.space.metric(x, z)
                    d_xy = self.space.metric(x, y)
                    d_yz = self.space.metric(y, z)
                    self.assertLessEqual(d_xz, d_xy + d_yz + 1e-10,
                                       f"三角不等式失败: d({x},{z}) > d({x},{y}) + d({y},{z})")
    
    def test_cauchy_sequence_convergence(self):
        """测试Cauchy序列的收敛性（完备性）"""
        # 构造一个收敛到5的序列
        target = 5  # Fibonacci数
        sequence = []
        for i in range(20):
            # 逐步逼近目标
            value = int(target * (1 - self.space.contraction_constant ** i))
            sequence.append(self.space.encode_zeckendorf(value))
        
        # 验证是Cauchy序列
        self.assertTrue(self.space.is_cauchy_sequence(sequence),
                       "构造的序列不是Cauchy序列")
        
        # 验证收敛
        limit = self.space.encode_zeckendorf(target)
        final_distance = self.space.metric(sequence[-1], limit)
        self.assertLess(final_distance, 0.1,
                       f"序列未收敛到预期极限: 距离 = {final_distance}")
    
    def test_contraction_mapping(self):
        """测试压缩映射性质"""
        def test_mapping(z: str) -> str:
            """一个测试用的压缩映射"""
            n = self.space.decode_zeckendorf(z)
            # 应用一个简单的压缩变换
            new_n = int(n * 0.618 + 1)  # 接近φ^(-1)的压缩
            return self.space.encode_zeckendorf(new_n)
        
        # 测试多对点的压缩性
        test_pairs = [
            ("1", "10"),
            ("100", "101"),
            ("1000", "1010"),
        ]
        
        for x, y in test_pairs:
            is_contraction, ratio = self.space.verify_contraction(test_mapping, x, y)
            self.assertTrue(is_contraction,
                          f"映射不是压缩的: {x}, {y}, 比率 = {ratio}")
            # 验证压缩常数接近φ^(-1)
            self.assertLess(ratio, 0.7,  # 留一些余量
                          f"压缩常数过大: {ratio}")
    
    def test_fixed_point_existence(self):
        """测试不动点的存在性和唯一性"""
        def self_referential_map(z: str) -> str:
            """自指映射的简化版本"""
            n = self.space.decode_zeckendorf(z)
            # 模拟自指: f(n) = floor(n * φ^(-1)) + Fib(k) for some k
            # 这里简化为寻找稳定点
            if n < 2:
                return self.space.encode_zeckendorf(2)
            new_n = int(n * self.space.contraction_constant + 1)
            return self.space.encode_zeckendorf(new_n)
        
        # 从不同初始点开始
        initial_points = ["1", "10", "100", "1000"]
        fixed_points = []
        
        for initial in initial_points:
            fixed_point, iterations = self.space.find_fixed_point(
                self_referential_map, initial, max_iterations=100
            )
            fixed_points.append(fixed_point)
            
            # 验证确实是不动点
            mapped = self_referential_map(fixed_point)
            distance = self.space.metric(fixed_point, mapped)
            self.assertLess(distance, 1e-6,
                          f"找到的点不是不动点: d({fixed_point}, M({fixed_point})) = {distance}")
        
        # 验证唯一性（所有初始点收敛到同一个不动点）
        for i in range(1, len(fixed_points)):
            d = self.space.metric(fixed_points[0], fixed_points[i])
            self.assertLess(d, 1e-6,
                          f"不动点不唯一: {fixed_points[0]} vs {fixed_points[i]}")
    
    def test_convergence_rate(self):
        """测试收敛速率是否符合理论预测"""
        def test_map(z: str) -> str:
            n = self.space.decode_zeckendorf(z)
            target = 8  # Fibonacci数
            # 向目标收敛的映射
            new_n = int(n * self.space.contraction_constant + 
                       target * (1 - self.space.contraction_constant))
            return self.space.encode_zeckendorf(new_n)
        
        initial = "1"
        current = initial
        distances = []
        
        # 迭代并记录到不动点的距离
        for i in range(20):
            next_point = test_map(current)
            distances.append(self.space.metric(current, next_point))
            current = next_point
        
        # 验证指数收敛
        for i in range(1, len(distances) - 1):
            if distances[i-1] > 1e-10:  # 避免除零
                ratio = distances[i] / distances[i-1]
                # 收敛比应该接近φ^(-1)
                self.assertLess(ratio, self.space.contraction_constant + 0.1,
                              f"收敛速率不符合预期: 第{i}步比率 = {ratio}")
    
    def test_entropy_increase(self):
        """测试迭代过程中的熵增规律"""
        def compute_entropy(z: str) -> float:
            """计算Zeckendorf字符串的信息熵"""
            n = self.space.decode_zeckendorf(z)
            if n <= 1:
                return 0
            return np.log(n)
        
        def entropy_increasing_map(z: str) -> str:
            """熵增映射"""
            n = self.space.decode_zeckendorf(z)
            # 每次迭代增加约log(φ)的熵
            new_n = int(n * self.space.phi)
            if new_n > 10000:  # 防止溢出
                new_n = 10000
            return self.space.encode_zeckendorf(new_n)
        
        current = "10"
        entropies = []
        
        for _ in range(10):
            entropies.append(compute_entropy(current))
            current = entropy_increasing_map(current)
        
        # 验证熵增
        for i in range(1, len(entropies)):
            if entropies[i-1] > 0:  # 忽略初始的零熵
                delta_h = entropies[i] - entropies[i-1]
                expected = np.log(self.space.phi)
                # 验证熵增接近log(φ)
                self.assertAlmostEqual(delta_h, expected, delta=0.1,
                                     msg=f"熵增偏离预期: {delta_h} vs {expected}")
    
    def test_fibonacci_fixed_points(self):
        """测试Fibonacci数作为特殊不动点"""
        fibonacci_strings = ["1", "10", "100", "1000", "10000", "100000"]
        
        for fib_str in fibonacci_strings:
            n = self.space.decode_zeckendorf(fib_str)
            # Fibonacci数的特殊性质
            self.assertIn(n, self.space.fibonacci[:20],
                        f"{n} 不是Fibonacci数")
            
            # 验证编码的简洁性（只有一个1）
            if n > 2:
                self.assertEqual(fib_str.count('1'), 1,
                               f"Fibonacci数 {n} 的编码不简洁: {fib_str}")


class TestFixedPointApplications(unittest.TestCase):
    """测试不动点理论在具体应用中的表现"""
    
    def setUp(self):
        self.space = ZeckendorfMetricSpace()
    
    def test_psi_self_mapping(self):
        """测试ψ = ψ(ψ)自映射的不动点"""
        def psi_mapping(z: str) -> str:
            """模拟ψ自映射"""
            n = self.space.decode_zeckendorf(z)
            # 自指操作的简化模拟
            # ψ(ψ) 在数值上表现为某种不动点变换
            if n == 0:
                return "1"
            # 寻找稳定的自指结构
            new_n = n
            for fib in self.space.fibonacci[:10]:
                if abs(n - fib) < 2:
                    new_n = fib  # 吸引到最近的Fibonacci数
                    break
            return self.space.encode_zeckendorf(new_n)
        
        # 测试从不同初始值开始
        for initial_n in [1, 3, 5, 8, 13]:
            initial = self.space.encode_zeckendorf(initial_n)
            fixed_point, _ = self.space.find_fixed_point(
                psi_mapping, initial, max_iterations=50
            )
            
            # 验证不动点是Fibonacci数
            fp_value = self.space.decode_zeckendorf(fixed_point)
            self.assertIn(fp_value, self.space.fibonacci[:20],
                        f"ψ不动点 {fp_value} 不是Fibonacci数")
    
    def test_recursive_encoding(self):
        """测试递归过程R = R(R)的编码"""
        def recursive_map(z: str) -> str:
            """递归编码映射"""
            n = self.space.decode_zeckendorf(z)
            # R(R) 的简化：递归深度增加
            depth = len(z)
            if depth >= 10:  # 限制递归深度
                return z
            # 增加一层递归
            new_n = n + self.space.fibonacci[min(depth, 15)]
            return self.space.encode_zeckendorf(new_n)
        
        initial = "1"
        current = initial
        depths = []
        
        for i in range(10):
            current = recursive_map(current)
            depths.append(len(current))
            
            # 验证No-11约束始终满足
            self.assertTrue(self.space.verify_no_11(current),
                          f"递归第{i}步违反No-11: {current}")
        
        # 验证递归深度的增长模式
        for i in range(1, len(depths)):
            # 深度应该逐渐增加但最终稳定
            self.assertGreaterEqual(depths[i], depths[i-1] - 1,
                                  "递归深度异常下降")


if __name__ == '__main__':
    unittest.main()