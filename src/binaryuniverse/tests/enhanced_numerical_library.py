#!/usr/bin/env python3
"""
T0理论体系的增强数值计算库
提供高精度φ-算术、Zeckendorf编码优化、并行计算支持

核心模块:
1. HighPrecisionPhi - 高精度黄金比例计算
2. OptimizedZeckendorf - 优化的Zeckendorf编码系统
3. ParallelCompute - 并行计算支持
4. PhiArithmetic - φ-算术运算
5. NoElevenValidator - No-11约束验证
6. QuantumCalculations - 量子态计算
7. InformationMeasures - 信息熵测量
"""

import numpy as np
from decimal import Decimal, getcontext
from typing import List, Tuple, Dict, Optional, Union, Callable
import math
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache, wraps
import time
import warnings

# 设置高精度
getcontext().prec = 100  # 100位小数精度


class HighPrecisionPhi:
    """高精度黄金比例计算器"""
    
    def __init__(self, precision: int = 100):
        self.precision = precision
        getcontext().prec = precision
        
        # 计算高精度φ
        self._phi_decimal = self._calculate_phi_decimal()
        self._phi_float = float(self._phi_decimal)
        
        # 预计算常用幂次
        self._phi_powers_cache = {}
        self._inverse_phi_powers_cache = {}
        
    def _calculate_phi_decimal(self) -> Decimal:
        """使用高精度算术计算φ"""
        five = Decimal(5)
        one = Decimal(1)
        two = Decimal(2)
        
        phi = (one + five.sqrt()) / two
        return phi
    
    @property
    def phi(self) -> float:
        """返回φ的float值"""
        return self._phi_float
    
    @property
    def phi_decimal(self) -> Decimal:
        """返回φ的高精度Decimal值"""
        return self._phi_decimal
    
    @lru_cache(maxsize=1000)
    def power(self, n: float, use_decimal: bool = False) -> Union[float, Decimal]:
        """计算φ^n"""
        if use_decimal:
            return self._phi_decimal ** Decimal(str(n))
        else:
            return self._phi_float ** n
    
    @lru_cache(maxsize=1000)
    def inverse_power(self, n: float, use_decimal: bool = False) -> Union[float, Decimal]:
        """计算φ^(-n) = (1/φ)^n"""
        inv_phi = 1.0 / self._phi_float if not use_decimal else Decimal(1) / self._phi_decimal
        
        if use_decimal:
            return inv_phi ** Decimal(str(n))
        else:
            return inv_phi ** n
    
    def convergent_ratio(self, fib_n: int, fib_n_minus_1: int) -> Tuple[float, float]:
        """计算Fibonacci比值收敛到φ的误差"""
        if fib_n_minus_1 == 0:
            return float('inf'), float('inf')
        
        ratio = fib_n / fib_n_minus_1
        error = abs(ratio - self._phi_float)
        relative_error = error / self._phi_float
        
        return error, relative_error
    
    def binet_formula(self, n: int, use_decimal: bool = False) -> Union[int, Decimal]:
        """使用Binet公式计算第n个Fibonacci数"""
        if use_decimal:
            phi = self._phi_decimal
            psi = (Decimal(1) - Decimal(5).sqrt()) / Decimal(2)  # 共轭根
            sqrt5 = Decimal(5).sqrt()
            
            fib = (phi ** n - psi ** n) / sqrt5
            return int(fib.to_integral_value())
        else:
            phi = self._phi_float
            psi = (1 - math.sqrt(5)) / 2
            sqrt5 = math.sqrt(5)
            
            fib = (phi ** n - psi ** n) / sqrt5
            return round(fib)
    
    def phi_derivative(self, func: Callable, x: float, n: int = 10) -> float:
        """计算φ-导数：使用Fibonacci增量的导数"""
        if n >= len(FibonacciSequence().sequence):
            n = len(FibonacciSequence().sequence) - 1
        
        # 使用φ^(-2n)作为增量
        delta = self.inverse_power(2 * n)
        
        # 数值导数
        derivative = (func(x + delta) - func(x)) / delta
        return derivative
    
    def generate_phi_sequence(self, length: int) -> List[float]:
        """生成φ-序列：1, φ, φ², φ³, ..."""
        sequence = []
        for n in range(length):
            sequence.append(self.power(n))
        return sequence


class FibonacciSequence:
    """高效Fibonacci序列生成器"""
    
    def __init__(self, max_length: int = 200):
        self.max_length = max_length
        self.sequence = self._generate_sequence()
        self.phi_calc = HighPrecisionPhi()
        
    def _generate_sequence(self) -> List[int]:
        """生成Fibonacci序列"""
        if self.max_length <= 0:
            return []
        elif self.max_length == 1:
            return [1]
        elif self.max_length == 2:
            return [1, 2]  # 使用F_1=1, F_2=2避免退化
        
        fib = [1, 2]
        for i in range(2, self.max_length):
            fib.append(fib[-1] + fib[-2])
        
        return fib
    
    def get(self, n: int) -> int:
        """获取第n个Fibonacci数"""
        if n < 1:
            return 0
        elif n <= len(self.sequence):
            return self.sequence[n - 1]
        else:
            # 使用Binet公式扩展
            return self.phi_calc.binet_formula(n)
    
    def extend_to(self, new_length: int):
        """扩展序列到指定长度"""
        if new_length <= len(self.sequence):
            return
        
        current_length = len(self.sequence)
        for i in range(current_length, new_length):
            self.sequence.append(self.sequence[-1] + self.sequence[-2])
        
        self.max_length = new_length
    
    def ratios(self) -> List[float]:
        """返回连续比值 F_n/F_{n-1}"""
        if len(self.sequence) < 2:
            return []
        
        ratios = []
        for i in range(1, len(self.sequence)):
            ratios.append(self.sequence[i] / self.sequence[i-1])
        
        return ratios


class OptimizedZeckendorf:
    """优化的Zeckendorf编码系统"""
    
    def __init__(self):
        self.fibonacci = FibonacciSequence(100)
        self.phi_calc = HighPrecisionPhi()
        
        # 缓存
        self._encoding_cache = {}
        self._decoding_cache = {}
        
    @lru_cache(maxsize=10000)
    def encode(self, n: int) -> Tuple[int, ...]:
        """将整数编码为Zeckendorf表示（返回Fibonacci索引）"""
        if n <= 0:
            return ()
        
        indices = []
        remaining = n
        i = len(self.fibonacci.sequence) - 1
        
        while i >= 0 and remaining > 0:
            fib_val = self.fibonacci.get(i + 1)  # Fibonacci编号从1开始
            if fib_val <= remaining:
                indices.append(i + 1)  # 存储Fibonacci编号
                remaining -= fib_val
                i -= 2  # 跳过下一个避免连续
            else:
                i -= 1
        
        return tuple(reversed(indices))  # 从小到大排列
    
    @lru_cache(maxsize=10000)
    def decode(self, indices: Tuple[int, ...]) -> int:
        """从Zeckendorf索引解码为整数"""
        if not indices:
            return 0
        
        total = 0
        for idx in indices:
            total += self.fibonacci.get(idx)
        
        return total
    
    def to_binary_string(self, n: int) -> str:
        """转换为二进制字符串表示"""
        if n <= 0:
            return "0"
        
        indices = self.encode(n)
        if not indices:
            return "0"
        
        # 找到最大索引
        max_index = max(indices)
        
        # 构建二进制字符串
        binary = ['0'] * max_index
        for idx in indices:
            binary[idx - 1] = '1'  # 索引从1开始，数组从0开始
        
        return ''.join(reversed(binary))  # 高位在前
    
    def from_binary_string(self, binary_str: str) -> int:
        """从二进制字符串解码"""
        if not binary_str or binary_str == "0":
            return 0
        
        indices = []
        for i, bit in enumerate(reversed(binary_str)):
            if bit == '1':
                indices.append(i + 1)  # Fibonacci编号从1开始
        
        return self.decode(tuple(indices))
    
    def validate_no_11(self, binary_str: str) -> bool:
        """验证No-11约束"""
        return "11" not in binary_str
    
    def validate_encoding(self, n: int) -> bool:
        """验证编码的有效性"""
        binary = self.to_binary_string(n)
        return self.validate_no_11(binary)
    
    def all_valid_up_to(self, max_n: int) -> List[int]:
        """返回所有不超过max_n的有效Zeckendorf数"""
        valid = []
        for n in range(1, max_n + 1):
            if self.validate_encoding(n):
                valid.append(n)
        return valid
    
    def efficiency_ratio(self, max_n: int = 1000) -> float:
        """计算编码效率：有效数字的比例"""
        valid_count = len(self.all_valid_up_to(max_n))
        return valid_count / max_n


class PhiArithmetic:
    """φ-算术运算系统"""
    
    def __init__(self):
        self.phi_calc = HighPrecisionPhi()
        self.phi = self.phi_calc.phi
        
    def phi_addition(self, a: float, b: float) -> float:
        """φ-加法：a ⊕_φ b = (a + b) / φ"""
        return (a + b) / self.phi
    
    def phi_multiplication(self, a: float, b: float) -> float:
        """φ-乘法：a ⊗_φ b = (a × b) × φ"""
        return (a * b) * self.phi
    
    def phi_power(self, base: float, exponent: float) -> float:
        """φ-幂运算：base^φ_exp = base^(exp/φ)"""
        return base ** (exponent / self.phi)
    
    def phi_logarithm(self, x: float, base: Optional[float] = None) -> float:
        """φ-对数：log_φ(x) or log_base^φ(x)"""
        if base is None:
            # 自然φ-对数
            return math.log(x) * self.phi
        else:
            # φ-底对数
            return math.log(x) / math.log(base * self.phi)
    
    def phi_distance(self, x: float, y: float) -> float:
        """φ-距离：d_φ(x,y) = |x-y|^φ"""
        return abs(x - y) ** self.phi
    
    def phi_norm(self, vector: List[float]) -> float:
        """φ-范数：||v||_φ = (Σ|v_i|^φ)^(1/φ)"""
        if not vector:
            return 0.0
        
        sum_powers = sum(abs(v) ** self.phi for v in vector)
        return sum_powers ** (1.0 / self.phi)
    
    def phi_inner_product(self, v1: List[float], v2: List[float]) -> float:
        """φ-内积：⟨v₁,v₂⟩_φ = Σ(v₁ᵢ·v₂ᵢ)·τᵢ，τ=1/φ"""
        if len(v1) != len(v2):
            raise ValueError("Vectors must have the same length")
        
        tau = 1.0 / self.phi
        result = 0.0
        
        for i, (a, b) in enumerate(zip(v1, v2)):
            weight = tau ** i
            result += a * b * weight
        
        return result
    
    def phi_orthogonalize(self, vectors: List[List[float]]) -> List[List[float]]:
        """φ-Gram-Schmidt正交化"""
        if not vectors:
            return []
        
        orthogonal = []
        
        for v in vectors:
            u = v.copy()
            
            # 减去在已有正交向量上的投影
            for ortho in orthogonal:
                # 投影系数
                proj_coeff = self.phi_inner_product(v, ortho) / self.phi_inner_product(ortho, ortho)
                
                # 减去投影
                for i in range(len(u)):
                    u[i] -= proj_coeff * ortho[i]
            
            # 归一化
            norm = self.phi_norm(u)
            if norm > 1e-10:  # 避免数值错误
                u = [x / norm for x in u]
                orthogonal.append(u)
        
        return orthogonal


class ParallelCompute:
    """并行计算支持系统"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (threading.cpu_count() or 1) + 4)
        
    def parallel_map(self, func: Callable, iterable, use_processes: bool = False):
        """并行映射"""
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            results = list(executor.map(func, iterable))
        
        return results
    
    def parallel_fibonacci_batch(self, indices: List[int]) -> List[int]:
        """并行计算Fibonacci数批次"""
        phi_calc = HighPrecisionPhi()
        
        def compute_fib(n):
            return phi_calc.binet_formula(n)
        
        return self.parallel_map(compute_fib, indices, use_processes=True)
    
    def parallel_zeckendorf_batch(self, numbers: List[int]) -> List[str]:
        """并行Zeckendorf编码批次"""
        zeck = OptimizedZeckendorf()
        
        def encode_number(n):
            return zeck.to_binary_string(n)
        
        return self.parallel_map(encode_number, numbers)
    
    def parallel_phi_powers(self, exponents: List[float]) -> List[float]:
        """并行计算φ幂次"""
        phi_calc = HighPrecisionPhi()
        
        def compute_power(exp):
            return phi_calc.power(exp)
        
        return self.parallel_map(compute_power, exponents, use_processes=True)


class QuantumCalculations:
    """量子态计算工具"""
    
    def __init__(self):
        self.phi_calc = HighPrecisionPhi()
        self.phi = self.phi_calc.phi
    
    def normalize_state(self, amplitudes: List[complex]) -> List[complex]:
        """归一化量子态"""
        norm_sq = sum(abs(amp) ** 2 for amp in amplitudes)
        norm = math.sqrt(norm_sq)
        
        if norm < 1e-10:
            # 零态处理
            normalized = [complex(1, 0)] + [complex(0, 0)] * (len(amplitudes) - 1)
        else:
            normalized = [amp / norm for amp in amplitudes]
        
        return normalized
    
    def phi_qubit(self) -> Tuple[complex, complex]:
        """创建φ-量子比特：|ψ⟩ = α|0⟩ + β|1⟩，满足φ-比例"""
        # α² : β² = φ : 1
        alpha_sq = self.phi / (self.phi + 1)
        beta_sq = 1 / (self.phi + 1)
        
        alpha = complex(math.sqrt(alpha_sq), 0)
        beta = complex(math.sqrt(beta_sq), 0)
        
        return alpha, beta
    
    def phi_entangled_state(self) -> np.ndarray:
        """创建φ-纠缠态"""
        # |ψ⟩ = α|01⟩ + β|10⟩ (避免|11⟩违反No-11)
        alpha, beta = self.phi_qubit()
        
        state = np.zeros(4, dtype=complex)
        state[1] = alpha  # |01⟩
        state[2] = beta   # |10⟩
        
        return self.normalize_state(state.tolist())
    
    def von_neumann_entropy(self, density_matrix: np.ndarray) -> float:
        """计算von Neumann熵"""
        eigenvals = np.linalg.eigvals(density_matrix)
        eigenvals = eigenvals[eigenvals > 1e-12]  # 过滤数值误差
        
        entropy = -sum(lam * math.log2(lam) for lam in eigenvals if lam > 0)
        return entropy
    
    def phi_rotation(self, angle_factor: float = 1.0) -> np.ndarray:
        """创建φ-旋转矩阵"""
        theta = angle_factor * math.pi / self.phi
        
        rotation = np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)]
        ], dtype=complex)
        
        return rotation
    
    def evolve_phi_state(self, initial_state: List[complex], steps: int) -> List[List[complex]]:
        """使用φ-哈密顿量演化量子态"""
        evolution = [initial_state]
        current_state = np.array(initial_state)
        
        # φ-哈密顿量（简化）
        H_phi = np.array([
            [0, 1/self.phi],
            [1/self.phi, 0]
        ], dtype=complex)
        
        dt = 1.0 / self.phi  # φ-时间步长
        
        for step in range(steps):
            # 使用欧拉方法演化：|ψ(t+dt)⟩ = |ψ(t)⟩ - i·H·|ψ(t)⟩·dt
            derivative = -1j * np.dot(H_phi, current_state) * dt
            current_state = current_state + derivative
            
            # 重归一化
            current_state = np.array(self.normalize_state(current_state.tolist()))
            evolution.append(current_state.tolist())
        
        return evolution


class InformationMeasures:
    """信息测量工具"""
    
    def __init__(self):
        self.phi_calc = HighPrecisionPhi()
        self.phi = self.phi_calc.phi
    
    def shannon_entropy(self, probabilities: List[float]) -> float:
        """Shannon熵"""
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    def phi_entropy(self, probabilities: List[float]) -> float:
        """φ-熵：H_φ = -Σp_i·log_φ(p_i)"""
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                # log_φ(p) = log(p) / log(φ)
                entropy -= p * math.log(p) / math.log(self.phi)
        return entropy
    
    def shannon_to_phi_entropy(self, shannon_entropy: float) -> float:
        """Shannon熵转换为φ-熵"""
        return shannon_entropy / math.log2(self.phi)
    
    def mutual_information(self, joint_probs: np.ndarray) -> float:
        """互信息"""
        # 计算边际概率
        p_x = np.sum(joint_probs, axis=1)
        p_y = np.sum(joint_probs, axis=0)
        
        mi = 0.0
        for i, px in enumerate(p_x):
            for j, py in enumerate(p_y):
                pxy = joint_probs[i, j]
                if pxy > 0 and px > 0 and py > 0:
                    mi += pxy * math.log2(pxy / (px * py))
        
        return mi
    
    def phi_mutual_information(self, joint_probs: np.ndarray) -> float:
        """φ-互信息"""
        shannon_mi = self.mutual_information(joint_probs)
        return self.shannon_to_phi_entropy(shannon_mi)
    
    def relative_entropy(self, p: List[float], q: List[float]) -> float:
        """相对熵（KL散度）"""
        if len(p) != len(q):
            raise ValueError("Probability distributions must have the same length")
        
        kl_div = 0.0
        for pi, qi in zip(p, q):
            if pi > 0:
                if qi > 0:
                    kl_div += pi * math.log2(pi / qi)
                else:
                    return float('inf')  # KL散度无穷大
        
        return kl_div
    
    def information_gain(self, prior: List[float], posterior: List[float]) -> float:
        """信息增益"""
        return self.shannon_entropy(prior) - self.shannon_entropy(posterior)


class PerformanceProfiler:
    """性能分析工具"""
    
    def __init__(self):
        self.timings = {}
        
    def time_function(self, func_name: str):
        """装饰器：测量函数执行时间"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    
                    if func_name not in self.timings:
                        self.timings[func_name] = []
                    self.timings[func_name].append(execution_time)
            
            return wrapper
        return decorator
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """获取性能统计"""
        stats = {}
        for func_name, times in self.timings.items():
            stats[func_name] = {
                'count': len(times),
                'total': sum(times),
                'mean': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'std': math.sqrt(sum((t - sum(times)/len(times))**2 for t in times) / len(times))
            }
        return stats
    
    def print_report(self):
        """打印性能报告"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("PERFORMANCE PROFILING REPORT")
        print("="*60)
        
        for func_name, stat in stats.items():
            print(f"\n{func_name}:")
            print(f"  Calls: {stat['count']}")
            print(f"  Total time: {stat['total']:.6f}s")
            print(f"  Mean time: {stat['mean']:.6f}s")
            print(f"  Min time: {stat['min']:.6f}s")
            print(f"  Max time: {stat['max']:.6f}s")
            print(f"  Std dev: {stat['std']:.6f}s")
        
        print("="*60)


# 全局实例
phi_calc = HighPrecisionPhi()
fibonacci = FibonacciSequence()
zeckendorf = OptimizedZeckendorf()
phi_arithmetic = PhiArithmetic()
parallel_compute = ParallelCompute()
quantum_calc = QuantumCalculations()
info_measures = InformationMeasures()
profiler = PerformanceProfiler()


def benchmark_library():
    """基准测试"""
    print("运行增强数值计算库基准测试...")
    
    # φ计算精度测试
    print(f"\n高精度φ计算:")
    print(f"φ (100位精度): {phi_calc.phi_decimal}")
    print(f"φ² = {phi_calc.power(2, use_decimal=True)}")
    print(f"1/φ = {phi_calc.inverse_power(1, use_decimal=True)}")
    
    # Fibonacci效率测试
    print(f"\nFibonacci计算:")
    start = time.time()
    large_fib = fibonacci.get(100)
    fib_time = time.time() - start
    print(f"F_100 = {large_fib} (用时: {fib_time:.6f}s)")
    
    # Zeckendorf编码测试
    print(f"\nZeckendorf编码:")
    test_numbers = [100, 200, 500, 1000]
    for n in test_numbers:
        binary = zeckendorf.to_binary_string(n)
        valid = zeckendorf.validate_no_11(binary)
        print(f"{n} -> {binary} ({'Valid' if valid else 'Invalid'})")
    
    # 并行计算测试
    print(f"\n并行计算测试:")
    start = time.time()
    fib_batch = parallel_compute.parallel_fibonacci_batch(list(range(20, 30)))
    parallel_time = time.time() - start
    print(f"并行计算F_20到F_29: {parallel_time:.6f}s")
    
    # φ-算术测试
    print(f"\nφ-算术测试:")
    v1 = [1, 2, 3]
    v2 = [4, 5, 6]
    phi_inner = phi_arithmetic.phi_inner_product(v1, v2)
    phi_norm1 = phi_arithmetic.phi_norm(v1)
    print(f"⟨{v1}, {v2}⟩_φ = {phi_inner:.6f}")
    print(f"||{v1}||_φ = {phi_norm1:.6f}")
    
    # 量子计算测试
    print(f"\n量子计算测试:")
    alpha, beta = quantum_calc.phi_qubit()
    print(f"φ-qubit: α={alpha:.6f}, β={beta:.6f}")
    print(f"|α|² + |β|² = {abs(alpha)**2 + abs(beta)**2:.10f}")
    
    print("\n基准测试完成！")


if __name__ == "__main__":
    benchmark_library()