#!/usr/bin/env python3
"""
Zeckendorf编码基础库
提供二进制宇宙理论的核心编码功能
"""

import numpy as np
from typing import List, Tuple, Optional
import math


class ZeckendorfEncoder:
    """Zeckendorf编码器 - 二进制宇宙的基础编码系统"""
    
    def __init__(self, max_length: int = 128):
        """
        初始化编码器
        
        Args:
            max_length: 支持的最大二进制串长度
        """
        self.phi = (1 + np.sqrt(5)) / 2  # 黄金比率
        self.max_length = max_length
        self.fibonacci_cache = self._generate_fibonacci(max_length)
        
    def _generate_fibonacci(self, n: int) -> List[int]:
        """生成Fibonacci数列缓存"""
        fib = [1, 2]
        while len(fib) < n:
            fib.append(fib[-1] + fib[-2])
        return fib
        
    def encode(self, n: int) -> str:
        """
        将整数编码为Zeckendorf表示（二进制串）
        满足no-11约束：不包含连续的1
        
        Args:
            n: 要编码的非负整数
            
        Returns:
            Zeckendorf二进制表示字符串
        """
        if n == 0:
            return "0"
            
        result = []
        remaining = n
        used_indices = []
        
        # 贪心算法：从大到小尝试Fibonacci数
        i = len(self.fibonacci_cache) - 1
        while i >= 0 and remaining > 0:
            if self.fibonacci_cache[i] <= remaining:
                # 检查no-11约束
                if not used_indices or used_indices[-1] != i + 1:
                    remaining -= self.fibonacci_cache[i]
                    used_indices.append(i)
            i -= 1
                    
        # 构建二进制串
        if not used_indices:
            return "0"
            
        max_idx = max(used_indices)
        result = ['0'] * (max_idx + 1)
        for idx in used_indices:
            result[max_idx - idx] = '1'
                
        return ''.join(result).lstrip('0') or "0"
        
    def decode(self, zeck_str: str) -> int:
        """
        将Zeckendorf表示解码为整数
        
        Args:
            zeck_str: Zeckendorf二进制字符串
            
        Returns:
            对应的整数值
        """
        if not zeck_str or zeck_str == "0":
            return 0
            
        value = 0
        fib_index = len(zeck_str) - 1
        
        for bit in zeck_str:
            if bit == '1':
                if fib_index < len(self.fibonacci_cache):
                    value += self.fibonacci_cache[fib_index]
            fib_index -= 1
            
        return value
        
    def verify_no_11(self, zeck_str: str) -> bool:
        """验证是否满足no-11约束"""
        return "11" not in zeck_str
        
    def compute_capacity(self, length: int) -> float:
        """
        计算给定长度的Zeckendorf熵容量
        
        理论结果：log_2(φ) * L ≈ 0.694 * L
        
        Args:
            length: 二进制串长度
            
        Returns:
            熵容量（比特）
        """
        # 精确值：log_2(φ) ≈ 0.6942419136
        return 0.694 * length
        
    def count_valid_strings(self, length: int) -> int:
        """
        计算给定长度下有效Zeckendorf串的数量
        
        使用动态规划计算满足no-11约束的串数
        
        Args:
            length: 串长度
            
        Returns:
            有效串的数量
        """
        if length == 0:
            return 1
        if length == 1:
            return 2  # '0' 和 '1'
        
        # dp[i] = 长度为i的无11串的个数
        dp = [0] * (length + 1)
        dp[0] = 1
        dp[1] = 2
        
        for i in range(2, length + 1):
            # 末尾是0：前i-1位可以是任何有效串
            # 末尾是1：前一位必须是0，所以是dp[i-2]
            dp[i] = dp[i-1] + dp[i-2]
        
        return dp[length]
        
    def fibonacci_quantize(self, value: float) -> int:
        """
        将连续值量子化到最近的Fibonacci数
        
        Args:
            value: 要量子化的浮点数
            
        Returns:
            最近的Fibonacci数
        """
        if value <= 0:
            return 0
            
        # 找到最接近的Fibonacci数
        min_diff = float('inf')
        closest_fib = 0
        
        for fib in self.fibonacci_cache:
            if abs(fib - value) < min_diff:
                min_diff = abs(fib - value)
                closest_fib = fib
            if fib > value * 2:  # 优化：不需要检查太大的数
                break
                
        return closest_fib


class GoldenConstants:
    """黄金比率相关常数"""
    
    PHI = (1 + np.sqrt(5)) / 2  # φ = 1.618...
    PHI_INVERSE = 2 / (1 + np.sqrt(5))  # φ^(-1) = 0.618...
    LOG2_PHI = np.log2((1 + np.sqrt(5)) / 2)  # log_2(φ) = 0.694...
    
    @staticmethod
    def lucas_number(n: int) -> int:
        """计算第n个Lucas数"""
        if n == 0:
            return 2
        if n == 1:
            return 1
        
        phi = GoldenConstants.PHI
        psi = 1 - phi  # 共轭
        return int(phi**n + psi**n + 0.5)
    
    @staticmethod
    def is_fibonacci(n: int) -> bool:
        """判断是否为Fibonacci数"""
        # 一个数是Fibonacci数当且仅当5n²+4或5n²-4是完全平方数
        def is_perfect_square(x):
            s = int(np.sqrt(x))
            return s * s == x
        
        return is_perfect_square(5 * n * n + 4) or is_perfect_square(5 * n * n - 4)


class EntropyCalculator:
    """熵计算工具"""
    
    @staticmethod
    def binary_entropy(p: float) -> float:
        """计算二元熵函数 H(p) = -p*log2(p) - (1-p)*log2(1-p)"""
        if p <= 0 or p >= 1:
            return 0
        return -p * np.log2(p) - (1-p) * np.log2(1-p)
    
    @staticmethod
    def zeckendorf_entropy(state: str) -> float:
        """
        计算Zeckendorf状态的熵
        
        Args:
            state: Zeckendorf二进制串
            
        Returns:
            状态熵
        """
        if not state or state == "0":
            return 0
            
        # 计算1的密度
        ones = state.count('1')
        length = len(state)
        
        if ones == 0:
            return 0
            
        # 基础熵：基于1的分布
        p = ones / length
        base_entropy = EntropyCalculator.binary_entropy(p)
        
        # Zeckendorf约束带来的额外信息
        # no-11约束减少了可能状态数，增加了每个状态的信息量
        constraint_factor = GoldenConstants.LOG2_PHI
        
        return base_entropy * length * constraint_factor
    
    @staticmethod
    def system_entropy(components: List[str]) -> float:
        """
        计算多组件系统的总熵
        
        Args:
            components: 组件状态列表（Zeckendorf串）
            
        Returns:
            系统总熵
        """
        return sum(EntropyCalculator.zeckendorf_entropy(comp) for comp in components)