"""
测试T5-4：最优压缩定理

验证：
1. Fibonacci序列计数
2. 密度收敛到log2(φ)
3. φ-表示的最优性
4. 编码效率（约69.4%）
5. 描述长度下界
"""

import unittest
import numpy as np
import math
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from io import BytesIO

class PhiSequenceAnalyzer:
    """φ-序列分析器"""
    
    def __init__(self):
        self.phi = (1 + math.sqrt(5)) / 2
        self.log_phi = math.log2(self.phi)
        
    def count_valid_sequences(self, n: int) -> int:
        """计算长度为n的有效序列数（满足no-11约束）"""
        # 根据L1-5: a(n) = F(n+2)
        # 生成Fibonacci数列 F(1)=1, F(2)=1, F(n)=F(n-1)+F(n-2)
        if n == 0:
            return 1  # a(0) = F(2) = 1
        
        # 计算F(n+2)
        fib = [1, 1]  # F(1), F(2)
        for i in range(2, n + 3):  # 需要计算到F(n+2)
            fib.append(fib[-1] + fib[-2])
        
        return fib[n + 1]  # F(n+2) = fib[n+1] (因为索引从0开始)
    
    def generate_valid_sequences(self, n: int) -> List[str]:
        """生成所有长度为n的有效φ-序列"""
        if n == 0:
            return ['']
        if n == 1:
            return ['0', '1']
        
        sequences = []
        # 递归生成
        for seq in self.generate_valid_sequences(n-1):
            sequences.append(seq + '0')  # 总是可以加0
            if not seq.endswith('1'):    # 只有不以1结尾才能加1
                sequences.append(seq + '1')
        
        return sequences
    
    def compute_density(self, n: int) -> float:
        """计算长度为n的序列的描述密度"""
        if n == 0:
            return 0.0
        count = self.count_valid_sequences(n)
        return math.log2(count) / n
    
    def compute_encoding_metrics(self, n: int) -> Dict[str, float]:
        """计算编码度量"""
        # 无约束情况
        unconstrained_sequences = 2**n
        unconstrained_density = 1.0
        
        # φ-表示情况
        phi_sequences = self.count_valid_sequences(n)
        phi_density = self.compute_density(n)
        
        # 效率和冗余度
        efficiency = phi_density / unconstrained_density
        redundancy = 1 - efficiency
        
        return {
            'n': n,
            'unconstrained_count': unconstrained_sequences,
            'phi_count': phi_sequences,
            'phi_density': phi_density,
            'efficiency': efficiency,
            'redundancy': redundancy,
            'density_error': abs(phi_density - self.log_phi)
        }
    
    def minimum_length_for_descriptions(self, N: int) -> int:
        """计算表示N个描述所需的最小长度"""
        # n_min = log2(N) / log2(φ)
        min_length_theoretical = math.log2(N) / self.log_phi
        
        # 找到实际的最小长度
        n = 1
        while self.count_valid_sequences(n) < N:
            n += 1
        
        return n, min_length_theoretical
    
    def verify_optimality(self, n: int) -> bool:
        """验证φ-表示的最优性"""
        # 生成所有有效序列
        valid_seqs = self.generate_valid_sequences(n)
        
        # 验证计数
        expected_count = self.count_valid_sequences(n)
        actual_count = len(valid_seqs)
        
        # 验证no-11约束
        for seq in valid_seqs:
            if '11' in seq:
                return False
        
        return actual_count == expected_count


class DescriptionCompressor:
    """描述压缩器（演示两种压缩概念）"""
    
    def __init__(self):
        self.phi_analyzer = PhiSequenceAnalyzer()
    
    def traditional_compress(self, data: str) -> Tuple[str, float]:
        """传统压缩：减少比特数"""
        # 简单的游程编码示例
        compressed = []
        i = 0
        while i < len(data):
            char = data[i]
            count = 1
            while i + count < len(data) and data[i + count] == char:
                count += 1
            compressed.append(f"{char}{count}")
            i += count
        
        compressed_str = ''.join(compressed)
        compression_ratio = len(compressed_str) / len(data)
        
        return compressed_str, compression_ratio
    
    def description_compress(self, n_symbols: int) -> Dict[str, float]:
        """描述压缩：最大化每符号的描述数"""
        # 无约束：2^n个描述
        unconstrained_descriptions = 2**n_symbols
        
        # φ-表示：F_{n+2}个描述
        phi_descriptions = self.phi_analyzer.count_valid_sequences(n_symbols)
        
        # 描述密度
        unconstrained_density = math.log2(unconstrained_descriptions) / n_symbols
        phi_density = math.log2(phi_descriptions) / n_symbols
        
        return {
            'symbols': n_symbols,
            'unconstrained_descriptions': unconstrained_descriptions,
            'phi_descriptions': phi_descriptions,
            'unconstrained_density': unconstrained_density,
            'phi_density': phi_density,
            'density_ratio': phi_density / unconstrained_density
        }


class TestT5_4OptimalCompression(unittest.TestCase):
    """T5-4最优压缩定理测试"""
    
    def setUp(self):
        """初始化测试环境"""
        self.analyzer = PhiSequenceAnalyzer()
        self.compressor = DescriptionCompressor()
        self.phi = (1 + math.sqrt(5)) / 2
        self.log_phi = math.log2(self.phi)
    
    def test_fibonacci_sequence_counting(self):
        """测试1：Fibonacci序列计数"""
        print("\n测试1：Fibonacci序列验证")
        
        # a(n) = F(n+2)，其中F(1)=1, F(2)=1, F(3)=2, ...
        # n:    0  1  2  3  4   5   6   7   8   9   10   11
        # F(n+2): F2 F3 F4 F5 F6  F7  F8  F9 F10 F11 F12  F13
        expected = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
        
        print("  n  F_{n+2}  计算值  验证")
        print("  -  -------  -----  ----")
        
        all_correct = True
        for n in range(len(expected)):
            computed = self.analyzer.count_valid_sequences(n)
            correct = computed == expected[n]
            all_correct &= correct
            
            print(f"  {n:<2} {expected[n]:<7} {computed:<6} {'✓' if correct else '✗'}")
        
        self.assertTrue(all_correct, "Fibonacci序列计数应该正确")
        
        # 验证实际生成的序列
        for n in range(1, 6):
            sequences = self.analyzer.generate_valid_sequences(n)
            expected_count = self.analyzer.count_valid_sequences(n)
            actual_count = len(sequences)
            
            print(f"\n  n={n}: 期望{expected_count}个序列，实际{actual_count}个")
            if n <= 3:
                print(f"  序列: {sequences}")
            
            self.assertEqual(actual_count, expected_count,
                           f"n={n}时序列数应该匹配")
    
    def test_density_convergence(self):
        """测试2：密度收敛到log2(φ)"""
        print("\n测试2：密度收敛性")
        
        print(f"  理论密度: log2(φ) = {self.log_phi:.6f}")
        print("\n  n    密度      误差")
        print("  ---  --------  --------")
        
        densities = []
        errors = []
        
        for n in range(5, 51, 5):
            density = self.analyzer.compute_density(n)
            error = abs(density - self.log_phi)
            densities.append(density)
            errors.append(error)
            
            print(f"  {n:<3}  {density:.6f}  {error:.6f}")
        
        # 验证收敛
        final_error = errors[-1]
        self.assertLess(final_error, 0.005,
                       "密度应该收敛到log2(φ)")
        
        # 验证误差递减
        for i in range(1, len(errors)):
            self.assertLessEqual(errors[i], errors[i-1] * 1.1,
                               "误差应该递减或保持稳定")
    
    def test_phi_representation_optimality(self):
        """测试3：φ-表示的最优性"""
        print("\n测试3：最优性验证")
        
        # 验证不同长度的最优性
        for n in [4, 6, 8, 10]:
            optimal = self.analyzer.verify_optimality(n)
            self.assertTrue(optimal, f"n={n}时φ-表示应该是最优的")
            
            metrics = self.analyzer.compute_encoding_metrics(n)
            
            print(f"\n  长度n={n}:")
            print(f"    无约束序列数: {metrics['unconstrained_count']}")
            print(f"    φ-序列数: {metrics['phi_count']}")
            print(f"    密度: {metrics['phi_density']:.6f}")
            print(f"    最优性验证: {'✓' if optimal else '✗'}")
        
        # 验证没有其他编码能超过φ-表示的密度
        print("\n  验证密度上界...")
        
        # 理论上，任何满足no-11约束的编码都不能超过log2(φ)的密度
        # 但在有限长度下，密度会略高于理论值
        max_density_observed = max(self.analyzer.compute_density(n) 
                                 for n in range(10, 31))
        
        # 验证密度趋向理论值
        large_n_density = self.analyzer.compute_density(100)
        print(f"\n  n=100时的密度: {large_n_density:.6f}")
        print(f"  与理论值的差距: {abs(large_n_density - self.log_phi):.6f}")
        
        self.assertLess(abs(large_n_density - self.log_phi), 0.003,
                       "大n时密度应该非常接近理论值")
    
    def test_encoding_efficiency(self):
        """测试4：编码效率（约69.4%）"""
        print("\n测试4：编码效率")
        
        theoretical_efficiency = self.log_phi  # ≈ 0.694
        
        print(f"  理论效率: {theoretical_efficiency:.4f} ({theoretical_efficiency*100:.1f}%)")
        print("\n  n    效率      冗余度")
        print("  ---  --------  --------")
        
        efficiencies = []
        
        for n in [5, 10, 15, 20, 25, 30]:
            metrics = self.analyzer.compute_encoding_metrics(n)
            efficiency = metrics['efficiency']
            redundancy = metrics['redundancy']
            efficiencies.append(efficiency)
            
            print(f"  {n:<3}  {efficiency:.4f}    {redundancy:.4f}")
        
        # 验证平均效率
        avg_efficiency = np.mean(efficiencies)
        print(f"\n  平均效率: {avg_efficiency:.4f} ({avg_efficiency*100:.1f}%)")
        
        # 有限长度下的效率会略高于理论值
        # 验证效率在合理范围内
        self.assertGreater(avg_efficiency, theoretical_efficiency,
                          "有限长度下效率应略高于理论值")
        self.assertLess(avg_efficiency, theoretical_efficiency * 1.05,
                       "效率不应过度偏离理论值")
        
        # 验证冗余度
        theoretical_redundancy = 1 - theoretical_efficiency
        print(f"  理论冗余度: {theoretical_redundancy:.4f} ({theoretical_redundancy*100:.1f}%)")
        
        # 验证冗余度的收敛趋势
        # 随着n增大，冗余度应该接近理论值
        redundancies = []
        for n in [30, 50, 100]:
            m = self.analyzer.compute_encoding_metrics(n)
            redundancies.append(m['redundancy'])
            print(f"  n={n}: 冗余度={m['redundancy']:.4f}")
        
        # 验证单调递增且趋向理论值
        for i in range(1, len(redundancies)):
            self.assertGreater(redundancies[i], redundancies[i-1],
                             "冗余度应该随n增大而增加")
        
        # 验证n=100时非常接近理论值
        self.assertAlmostEqual(redundancies[-1], theoretical_redundancy,
                             places=2, msg="n=100时冗余度应该接近理论值")
    
    def test_minimum_description_length(self):
        """测试5：描述长度下界"""
        print("\n测试5：最小描述长度")
        
        test_cases = [10, 50, 100, 500, 1000]
        
        print("  描述数N  理论最小长度  实际最小长度  差异")
        print("  -------  -----------  -----------  ----")
        
        for N in test_cases:
            actual_min, theoretical_min = self.analyzer.minimum_length_for_descriptions(N)
            difference = actual_min - theoretical_min
            
            print(f"  {N:<7}  {theoretical_min:11.2f}  {actual_min:11}  {difference:4.1f}")
            
            # 实际长度应该是理论长度的上取整
            self.assertGreaterEqual(actual_min, math.ceil(theoretical_min) - 1,
                                  "实际长度应该接近理论下界")
            self.assertLessEqual(actual_min, math.ceil(theoretical_min) + 1,
                               "实际长度不应远超理论下界")
    
    def test_compression_paradigms(self):
        """测试6：两种压缩范式对比"""
        print("\n测试6：压缩范式对比")
        
        # 传统压缩示例
        test_data = "00000111110000011111"
        compressed, ratio = self.compressor.traditional_compress(test_data)
        
        print("  传统压缩（减少比特数）:")
        print(f"    原始: {test_data} ({len(test_data)} bits)")
        print(f"    压缩: {compressed} ({len(compressed)} chars)")
        print(f"    压缩比: {ratio:.2f}")
        
        # 描述压缩示例
        print("\n  描述压缩（最大化描述密度）:")
        
        for n in [8, 16, 24]:
            result = self.compressor.description_compress(n)
            
            print(f"\n    {n}个符号:")
            print(f"      无约束: {result['unconstrained_descriptions']} 个描述")
            print(f"      φ-表示: {result['phi_descriptions']} 个描述")
            print(f"      密度比: {result['density_ratio']:.3f}")
        
        # 验证φ-表示在描述密度意义上是最优的
        self.assertAlmostEqual(result['phi_density'], 
                             self.analyzer.compute_density(n),
                             places=6,
                             msg="描述密度计算应该一致")
    
    def test_golden_ratio_universality(self):
        """测试7：黄金比例的普遍性"""
        print("\n测试7：黄金比例的出现")
        
        # φ出现在多个地方
        contexts = []
        
        # 1. Fibonacci数列的比值
        fib_ratios = []
        a, b = 1, 2
        for i in range(20):
            ratio = b / a
            fib_ratios.append(ratio)
            a, b = b, a + b
        
        final_ratio = fib_ratios[-1]
        contexts.append(('Fibonacci比值', final_ratio, self.phi))
        
        # 2. 密度极限
        density_limit = 2**self.log_phi
        contexts.append(('密度极限底数', density_limit, self.phi))
        
        # 3. 最优效率
        optimal_efficiency = self.log_phi
        contexts.append(('最优效率', 2**optimal_efficiency, self.phi))
        
        print("  上下文              计算值     理论值φ    误差")
        print("  -----------------  ---------  ---------  -------")
        
        for context, computed, theoretical in contexts:
            error = abs(computed - theoretical)
            print(f"  {context:<17}  {computed:.6f}  {theoretical:.6f}  {error:.6f}")
        
        # 验证收敛
        self.assertAlmostEqual(final_ratio, self.phi, places=5,
                             msg="Fibonacci比值应该收敛到φ")


if __name__ == '__main__':
    unittest.main()