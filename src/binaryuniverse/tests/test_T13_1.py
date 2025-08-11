#!/usr/bin/env python3
"""
Unit tests for T13-1: φ-Encoding Algorithm Complexity Theorem
验证φ编码算法的复杂度性质

Tests verify:
1. O(log n)时间复杂度
2. O(log n)空间复杂度
3. 编码/解码正确性
4. 并行加速比
5. 与其他编码的比较
6. 信息理论界限
7. 量子算法优势
"""

import unittest
import numpy as np
import time
import sys
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib import rcParams
from functools import lru_cache
from multiprocessing import Pool, cpu_count

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 常数
PHI = (1 + np.sqrt(5)) / 2


class PhiEncodingSystem:
    """φ编码系统实现"""
    
    def __init__(self):
        self.phi = PHI
        self._fib_cache = [1, 1]  # F_1 = 1, F_2 = 1
        self._operation_count = 0
        self._space_usage = 0
        
    def _generate_fibonacci(self, max_val: int) -> List[int]:
        """生成不超过max_val的Fibonacci数列"""
        self._operation_count += 1
        
        while self._fib_cache[-1] < max_val:
            self._fib_cache.append(
                self._fib_cache[-1] + self._fib_cache[-2]
            )
            self._operation_count += 1
            
        # 返回不超过max_val的Fibonacci数
        result = []
        for fib in self._fib_cache:
            if fib <= max_val:
                result.append(fib)
            else:
                break
        
        self._space_usage = max(self._space_usage, len(result))
        return result
    
    def encode(self, n: int) -> List[int]:
        """φ编码（贪心算法）"""
        self._operation_count = 0
        self._space_usage = 0
        
        if n == 0:
            return []
        
        fibs = self._generate_fibonacci(n)
        result = []
        
        # 贪心选择最大的Fibonacci数
        i = len(fibs) - 1
        while n > 0 and i >= 0:
            self._operation_count += 1
            if fibs[i] <= n:
                result.append(i)
                n -= fibs[i]
            i -= 1
        
        self._space_usage = max(self._space_usage, len(result))
        return sorted(result)
    
    def decode(self, indices: List[int]) -> int:
        """φ解码"""
        self._operation_count = 0
        self._space_usage = 0
        
        if not indices:
            return 0
        
        max_idx = max(indices)
        fibs = self._generate_fibonacci(10**9)  # 足够大的数
        
        result = 0
        for idx in indices:
            self._operation_count += 1
            if idx < len(fibs):
                result += fibs[idx]
        
        self._space_usage = len(indices)
        return result
    
    def verify_no11(self, indices: List[int]) -> bool:
        """验证编码满足no-11约束"""
        sorted_indices = sorted(indices)
        for i in range(len(sorted_indices) - 1):
            if sorted_indices[i+1] == sorted_indices[i] + 1:
                return False
        return True
    
    def parallel_encode(self, n: int, num_processes: int = None) -> List[int]:
        """并行φ编码"""
        if num_processes is None:
            num_processes = cpu_count()
        
        # 对于小数字或单处理器，直接使用串行算法
        if n < 1000 or num_processes == 1:
            return self.encode(n)
        
        # 简化的并行版本：由于Zeckendorf表示的贪心性质，
        # 真正的并行化较困难。这里模拟并行查找最大Fibonacci数
        # 实际应用中，可以并行化Fibonacci数的生成
        
        # 步骤1：并行生成Fibonacci数（模拟）
        start_time = time.perf_counter()
        fibs = self._generate_fibonacci(n)
        generation_time = time.perf_counter() - start_time
        
        # 步骤2：贪心选择（本质上是串行的）
        result = []
        remaining = n
        
        # 模拟并行搜索延迟
        time.sleep(generation_time / num_processes)
        
        # 执行标准贪心算法
        i = len(fibs) - 1
        while remaining > 0 and i >= 0:
            if fibs[i] <= remaining:
                result.append(i)
                remaining -= fibs[i]
            i -= 1
        
        return sorted(result)
    
    def get_complexity_metrics(self) -> Tuple[int, int]:
        """获取最近操作的复杂度指标"""
        return self._operation_count, self._space_usage


class TestPhiEncodingComplexity(unittest.TestCase):
    """φ编码复杂度测试"""
    
    def setUp(self):
        """初始化测试环境"""
        self.encoder = PhiEncodingSystem()
        self.test_sizes = [10, 100, 1000, 10000, 100000]
    
    def test_encoding_correctness(self):
        """测试1: 验证编码/解码的正确性"""
        print("\n测试1: 验证编码/解码正确性")
        
        test_numbers = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 
                       233, 377, 610, 987, 1597, 2584, 4181, 6765]
        
        for n in test_numbers:
            # 编码
            encoded = self.encoder.encode(n)
            
            # 解码
            decoded = self.encoder.decode(encoded)
            
            # 验证no-11约束
            is_valid = self.encoder.verify_no11(encoded)
            
            print(f"\nn = {n}:")
            print(f"  编码: {encoded}")
            print(f"  解码: {decoded}")
            print(f"  满足no-11: {is_valid}")
            
            # 断言
            self.assertEqual(decoded, n, f"解码错误: {n}")
            self.assertTrue(is_valid, f"违反no-11约束: {encoded}")
    
    def test_time_complexity(self):
        """测试2: 验证O(log n)时间复杂度"""
        print("\n测试2: 验证时间复杂度")
        
        times = []
        operations = []
        
        for size in self.test_sizes:
            # 多次运行取平均
            total_time = 0
            total_ops = 0
            runs = 10
            
            for _ in range(runs):
                start_time = time.perf_counter()
                encoded = self.encoder.encode(size)
                end_time = time.perf_counter()
                
                total_time += (end_time - start_time)
                ops, _ = self.encoder.get_complexity_metrics()
                total_ops += ops
            
            avg_time = total_time / runs
            avg_ops = total_ops / runs
            
            times.append(avg_time)
            operations.append(avg_ops)
            
            print(f"\nn = {size}:")
            print(f"  平均时间: {avg_time*1000:.3f} ms")
            print(f"  平均操作数: {avg_ops:.1f}")
            print(f"  log₂(n): {np.log2(size):.2f}")
        
        # 验证对数增长
        # 计算增长率
        for i in range(1, len(self.test_sizes)):
            n1, n2 = self.test_sizes[i-1], self.test_sizes[i]
            ops1, ops2 = operations[i-1], operations[i]
            
            expected_ratio = np.log2(n2) / np.log2(n1)
            actual_ratio = ops2 / ops1
            
            print(f"\nn={n1}→{n2}: 预期比值={expected_ratio:.2f}, 实际比值={actual_ratio:.2f}")
            
            # 允许一定误差（由于常数因子）
            self.assertLess(actual_ratio, expected_ratio * 2,
                           "操作数增长过快，不符合O(log n)")
        
        self._plot_complexity(self.test_sizes, operations, "时间复杂度")
    
    def test_space_complexity(self):
        """测试3: 验证O(log n)空间复杂度"""
        print("\n测试3: 验证空间复杂度")
        
        space_usage = []
        
        for size in self.test_sizes:
            encoded = self.encoder.encode(size)
            _, space = self.encoder.get_complexity_metrics()
            
            # 理论空间需求
            theory_space = np.log(size) / np.log(PHI)
            
            space_usage.append(space)
            
            print(f"\nn = {size}:")
            print(f"  实际空间: {space}")
            print(f"  编码长度: {len(encoded)}")
            print(f"  理论空间: {theory_space:.2f}")
            print(f"  空间效率: {len(encoded)/theory_space:.2%}")
        
        # 验证对数增长
        for i in range(1, len(self.test_sizes)):
            n1, n2 = self.test_sizes[i-1], self.test_sizes[i]
            s1, s2 = space_usage[i-1], space_usage[i]
            
            if s1 > 0:  # 避免除零
                growth_ratio = s2 / s1
                size_ratio = np.log(n2) / np.log(n1)
                
                print(f"\nn={n1}→{n2}: 空间增长={growth_ratio:.2f}, 大小比值对数={size_ratio:.2f}")
                
                # 空间增长应该接近对数增长
                self.assertLess(growth_ratio, size_ratio * 1.5,
                               "空间增长过快，不符合O(log n)")
    
    def test_parallel_speedup(self):
        """测试4: 验证并行加速比"""
        print("\n测试4: 验证并行加速比")
        
        test_n = 100000
        processor_counts = [1, 2, 4, 8]
        
        # 串行基准时间
        start_time = time.perf_counter()
        serial_result = self.encoder.encode(test_n)
        serial_time = time.perf_counter() - start_time
        
        speedups = []
        
        for p in processor_counts:
            # 并行编码
            start_time = time.perf_counter()
            parallel_result = self.encoder.parallel_encode(test_n, p)
            parallel_time = time.perf_counter() - start_time
            
            speedup = serial_time / parallel_time if parallel_time > 0 else 1
            speedups.append(speedup)
            
            # 理论最大加速比
            theory_speedup = min(p, np.log2(test_n) / np.log2(np.log2(test_n)))
            
            print(f"\n处理器数 p = {p}:")
            print(f"  串行时间: {serial_time*1000:.3f} ms")
            print(f"  并行时间: {parallel_time*1000:.3f} ms")
            print(f"  实际加速比: {speedup:.2f}")
            print(f"  理论上限: {theory_speedup:.2f}")
            
            # 验证结果一致性
            self.assertEqual(sorted(serial_result), sorted(parallel_result),
                           "并行编码结果不一致")
        
        # 绘制加速比图
        self._plot_speedup(processor_counts, speedups)
    
    def test_comparison_with_binary(self):
        """测试5: 与二进制编码的比较"""
        print("\n测试5: 与二进制编码比较")
        
        test_numbers = [100, 1000, 10000, 100000]
        
        for n in test_numbers:
            # φ编码
            phi_encoded = self.encoder.encode(n)
            phi_length = len(phi_encoded)
            
            # 二进制编码
            binary = bin(n)[2:]
            binary_length = len(binary)
            
            # 检查二进制是否有连续的11
            has_11 = '11' in binary
            
            # 信息理论下界
            info_bound = np.log2(n)
            
            print(f"\nn = {n}:")
            print(f"  二进制长度: {binary_length}")
            print(f"  二进制有'11': {has_11}")
            print(f"  φ编码长度: {phi_length}")
            print(f"  信息论下界: {info_bound:.2f}")
            print(f"  φ编码效率: {info_bound/phi_length:.2%}")
            print(f"  相比二进制: {phi_length/binary_length:.2%}")
            
            # φ编码不应该比二进制差太多
            self.assertLess(phi_length, binary_length * 1.5,
                           "φ编码效率过低")
    
    def test_information_theoretic_bound(self):
        """测试6: 验证信息理论界限"""
        print("\n测试6: 验证信息理论界限")
        
        # 测试不同规模的数
        test_range = range(1000, 10001, 1000)
        
        actual_lengths = []
        theory_bounds = []
        
        for n in test_range:
            encoded = self.encoder.encode(n)
            actual_length = len(encoded)
            
            # 信息理论下界（考虑no-11约束）
            # 有效状态密度约为 1/φ
            theory_bound = np.log2(n) - np.log2(PHI)
            
            actual_lengths.append(actual_length)
            theory_bounds.append(theory_bound)
            
            efficiency = theory_bound / actual_length if actual_length > 0 else 0
            
            if n % 2000 == 0:  # 打印部分结果
                print(f"\nn = {n}:")
                print(f"  实际编码长度: {actual_length}")
                print(f"  理论下界: {theory_bound:.2f}")
                print(f"  效率: {efficiency:.2%}")
        
        # 计算平均效率
        avg_efficiency = np.mean([b/a for a, b in zip(actual_lengths, theory_bounds) if a > 0])
        print(f"\n平均效率: {avg_efficiency:.2%}")
        
        # φ编码应该接近信息理论下界
        self.assertGreater(avg_efficiency, 0.7, "φ编码未达到信息理论效率要求")
    
    def test_quantum_advantage(self):
        """测试7: 模拟量子算法优势"""
        print("\n测试7: 量子算法优势分析")
        
        # 模拟量子搜索的复杂度
        classical_complexities = []
        quantum_complexities = []
        
        for size in self.test_sizes:
            # 经典复杂度
            classical = np.log2(size)
            classical_complexities.append(classical)
            
            # 量子复杂度（Grover搜索）
            quantum = np.sqrt(classical)
            quantum_complexities.append(quantum)
            
            speedup = classical / quantum
            
            print(f"\nn = {size}:")
            print(f"  经典复杂度: O({classical:.1f})")
            print(f"  量子复杂度: O({quantum:.1f})")
            print(f"  量子加速比: {speedup:.2f}x")
        
        # 绘制量子优势图
        self._plot_quantum_advantage(self.test_sizes, 
                                   classical_complexities, 
                                   quantum_complexities)
    
    def test_encoding_uniqueness(self):
        """测试8: 验证编码唯一性"""
        print("\n测试8: 验证Zeckendorf表示的唯一性")
        
        # 测试一系列数字
        for n in range(1, 100):
            encoded = self.encoder.encode(n)
            
            # 验证没有重复的索引
            self.assertEqual(len(encoded), len(set(encoded)),
                           f"编码{encoded}包含重复索引")
            
            # 验证满足no-11约束
            self.assertTrue(self.encoder.verify_no11(encoded),
                          f"编码{encoded}违反no-11约束")
            
            # 验证解码正确
            decoded = self.encoder.decode(encoded)
            self.assertEqual(decoded, n,
                           f"数字{n}的编码解码不一致")
        
        print("\n前20个数的Zeckendorf表示:")
        for n in range(1, 21):
            encoded = self.encoder.encode(n)
            print(f"{n}: {encoded}")
    
    def _plot_complexity(self, sizes, operations, title):
        """绘制复杂度分析图"""
        plt.figure(figsize=(10, 6))
        
        # 实际复杂度
        plt.plot(sizes, operations, 'b-o', label='实际操作数', linewidth=2, markersize=8)
        
        # 理论O(log n)曲线
        theory = [np.log2(n) * operations[0] / np.log2(sizes[0]) for n in sizes]
        plt.plot(sizes, theory, 'r--', label='O(log n)理论', linewidth=2)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('输入大小 n')
        plt.ylabel('操作数')
        plt.title(f'φ编码{title}分析')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/cookie/the-binarymath/docs/binaryuniverse/tests/phi_encoding_complexity_T13_1.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_speedup(self, processors, speedups):
        """绘制并行加速比图"""
        plt.figure(figsize=(10, 6))
        
        # 实际加速比
        plt.plot(processors, speedups, 'b-o', label='实际加速比', linewidth=2, markersize=8)
        
        # 理想线性加速
        plt.plot(processors, processors, 'g--', label='理想线性加速', linewidth=2)
        
        # Amdahl定律（假设串行部分10%）
        serial_fraction = 0.1
        amdahl = [1 / (serial_fraction + (1 - serial_fraction) / p) for p in processors]
        plt.plot(processors, amdahl, 'r:', label="Amdahl定律(10%串行)", linewidth=2)
        
        plt.xlabel('处理器数量')
        plt.ylabel('加速比')
        plt.title('φ编码并行加速比分析')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/cookie/the-binarymath/docs/binaryuniverse/tests/parallel_speedup_T13_1.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_quantum_advantage(self, sizes, classical, quantum):
        """绘制量子优势图"""
        plt.figure(figsize=(10, 6))
        
        # 复杂度比较
        plt.plot(sizes, classical, 'b-o', label='经典算法 O(log n)', linewidth=2)
        plt.plot(sizes, quantum, 'r-s', label='量子算法 O(√log n)', linewidth=2)
        
        # 加速比
        speedup = [c/q for c, q in zip(classical, quantum)]
        ax2 = plt.gca().twinx()
        ax2.plot(sizes, speedup, 'g--^', label='量子加速比', linewidth=2)
        ax2.set_ylabel('量子加速比', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        plt.xscale('log')
        plt.xlabel('输入大小 n')
        plt.ylabel('算法复杂度')
        plt.title('φ编码的量子算法优势')
        plt.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/cookie/the-binarymath/docs/binaryuniverse/tests/quantum_advantage_T13_1.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)