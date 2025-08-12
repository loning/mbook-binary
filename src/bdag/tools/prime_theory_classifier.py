#!/usr/bin/env python3
"""
Prime Theory Classifier v3.0
素数理论分类系统 - 实现五类理论分类(AXIOM/PRIME-FIB/FIBONACCI/PRIME/COMPOSITE)
与theory_parser.py和其他工具完全同步
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    from .theory_validator import PrimeChecker
    from .theory_parser import TheoryParser, FibonacciOperationType
except ImportError:
    from theory_validator import PrimeChecker
    from theory_parser import TheoryParser, FibonacciOperationType


# 使用与theory_parser.py完全一致的分类枚举
TheoryClassType = FibonacciOperationType


@dataclass
class TheoryClassification:
    """理论分类信息"""
    theory_number: int
    class_type: TheoryClassType
    is_prime: bool
    is_fibonacci: bool
    is_twin_prime: bool = False
    is_mersenne_prime: bool = False
    is_sophie_germain: bool = False
    prime_factors: Optional[List[Tuple[int, int]]] = None
    special_properties: Optional[List[str]] = None
    

class PrimeTheoryClassifier:
    """素数理论分类器 v3.0 - 与其他工具完全同步"""
    
    def __init__(self, max_theory: int = 997):
        self.max_theory = max_theory
        self.prime_checker = PrimeChecker()
        self.parser = TheoryParser(max_theory)
        
        # 生成完整的Fibonacci序列和素数集合
        self.fibonacci_sequence = self.parser.fibonacci_sequence
        self.fibonacci_set = set(self.fibonacci_sequence)
        self.primes = self.prime_checker.get_primes_up_to(max_theory)
        self.prime_set = set(self.primes)
    
    def classify_theory(self, n: int) -> TheoryClassification:
        """分类单个理论（五类分类系统）"""
        # 基础属性检测
        is_prime = n in self.prime_set
        is_fibonacci = n in self.fibonacci_set
        
        # 确定分类类型 - 使用一致的逻辑
        class_type = self._verify_classification(n, is_prime, is_fibonacci)
        
        # 创建分类对象
        classification = TheoryClassification(
            theory_number=n,
            class_type=class_type,
            is_prime=is_prime,
            is_fibonacci=is_fibonacci
        )
        
        # 添加素数特殊性质
        if is_prime:
            self._add_prime_properties(classification)
        
        # 添加素因子分解（对于合数）
        if not is_prime and n > 1:
            classification.prime_factors = self.prime_checker.prime_factorize(n)
        
        # 添加特殊性质描述
        classification.special_properties = self._get_special_properties(classification)
        
        return classification
    
    def _verify_classification(self, n: int, is_prime: bool, is_fibonacci: bool) -> FibonacciOperationType:
        """验证分类逻辑与parser一致"""
        if n == 1:
            return FibonacciOperationType.AXIOM
        elif is_prime and is_fibonacci:
            return FibonacciOperationType.PRIME_FIB
        elif is_fibonacci:
            return FibonacciOperationType.FIBONACCI
        elif is_prime:
            return FibonacciOperationType.PRIME
        else:
            return FibonacciOperationType.COMPOSITE
    
    def classify_range(self, start: int, end: int) -> Dict[int, TheoryClassification]:
        """分类一个范围内的所有理论"""
        classifications = {}
        for n in range(start, end + 1):
            classifications[n] = self.classify_theory(n)
        return classifications
    
    def get_theory_dependencies(self, classification: TheoryClassification) -> List[int]:
        """根据分类获取理论依赖"""
        n = classification.theory_number
        
        if classification.class_type == TheoryClassType.AXIOM:
            return []  # T1无依赖
        
        elif classification.class_type == TheoryClassType.PRIME_FIB:
            # 素数-Fibonacci理论依赖前两个Fibonacci理论
            if n == 2:
                return [1]  # T2依赖T1
            elif n == 3:
                return [2, 1]  # T3依赖T2和T1
            elif n == 5:
                return [3, 2]  # T5依赖T3和T2
            else:
                # 其他素数-Fib理论的递归依赖
                fib_index = self._get_fibonacci_index(n)
                if fib_index >= 2:
                    prev_fib = self.parser.fibonacci_sequence[fib_index - 1]
                    prev_prev_fib = self.parser.fibonacci_sequence[fib_index - 2]
                    return [prev_fib, prev_prev_fib]
                return []
        
        elif classification.class_type == TheoryClassType.FIBONACCI:
            # 纯Fibonacci理论的递归依赖
            fib_index = self._get_fibonacci_index(n)
            if fib_index >= 2:
                prev_fib = self.parser.fibonacci_sequence[fib_index - 1]
                prev_prev_fib = self.parser.fibonacci_sequence[fib_index - 2]
                return [prev_fib, prev_prev_fib]
            return []
        
        elif classification.class_type == TheoryClassType.PRIME:
            # 纯素数理论基于Zeckendorf分解
            return self._get_zeckendorf_dependencies(n)
        
        else:  # COMPOSITE
            # 合数理论基于Zeckendorf分解
            return self._get_zeckendorf_dependencies(n)
    
    def get_prime_significance(self, n: int) -> Dict[str, any]:
        """获取素数理论的特殊意义"""
        if not self.prime_checker.is_prime(n):
            return {}
        
        classification = self.classify_theory(n)
        significance = {
            'is_prime': True,
            'class_type': classification.class_type.value,
            'special_types': []
        }
        
        # 特殊素数类型
        if classification.is_twin_prime:
            significance['special_types'].append('孪生素数')
            significance['twin_pair'] = self._get_twin_pair(n)
        
        if classification.is_mersenne_prime:
            significance['special_types'].append('梅森素数')
            significance['mersenne_form'] = f"2^{self._get_mersenne_exponent(n)} - 1"
        
        if classification.is_sophie_germain:
            significance['special_types'].append('Sophie Germain素数')
            significance['safe_prime'] = 2 * n + 1
        
        # 素数在理论系统中的作用
        if classification.class_type == FibonacciOperationType.PRIME_FIB:
            significance['role'] = '双重基础理论（素数+Fibonacci）'
            significance['importance'] = '极高'
        else:
            significance['role'] = '原子理论（不可分解）'
            significance['importance'] = '高'
        
        # 密码学意义
        if n > 100:
            significance['cryptographic'] = '可用于密码学应用'
        
        return significance
    
    def get_classification_statistics(self) -> Dict:
        """获取分类统计信息"""
        stats = {
            'AXIOM': 0,
            'PRIME-FIB': 0,
            'FIBONACCI': 0,
            'PRIME': 0,
            'COMPOSITE': 0
        }
        
        for n in range(1, self.max_theory + 1):
            is_prime = n in self.prime_set
            is_fibonacci = n in self.fibonacci_set
            class_type = self._verify_classification(n, is_prime, is_fibonacci)
            stats[class_type.value] += 1
        
        return stats
    
    def _add_prime_properties(self, classification: TheoryClassification):
        """添加素数的特殊性质"""
        n = classification.theory_number
        classification.is_twin_prime = self.prime_checker.is_twin_prime(n)
        classification.is_mersenne_prime = self.prime_checker.is_mersenne_prime(n)
        classification.is_sophie_germain = self.prime_checker.is_sophie_germain_prime(n)
    
    def _get_special_properties(self, classification: TheoryClassification) -> List[str]:
        """获取理论的特殊性质描述"""
        properties = []
        
        if classification.class_type == FibonacciOperationType.AXIOM:
            properties.append("宇宙唯一公理")
            properties.append("所有理论的根源")
        
        elif classification.class_type == FibonacciOperationType.PRIME_FIB:
            properties.append("素数-Fibonacci双重性质")
            properties.append("系统核心基础理论")
            
        elif classification.class_type == FibonacciOperationType.FIBONACCI:
            properties.append("纯Fibonacci递归理论")
            properties.append("递归涌现性质")
            
        elif classification.class_type == FibonacciOperationType.PRIME:
            properties.append("原子理论")
            properties.append("不可分解")
            
        if classification.is_twin_prime:
            properties.append("孪生素数")
        if classification.is_mersenne_prime:
            properties.append("梅森素数")
        if classification.is_sophie_germain:
            properties.append("Sophie Germain素数")
        
        if classification.prime_factors:
            factor_str = self._format_prime_factors(classification.prime_factors)
            properties.append(f"素因子分解: {factor_str}")
        
        return properties
    
    def _get_fibonacci_index(self, n: int) -> int:
        """获取Fibonacci数的索引"""
        try:
            return self.parser.fibonacci_sequence.index(n)
        except ValueError:
            return -1
    
    def _get_zeckendorf_dependencies(self, n: int) -> List[int]:
        """获取基于Zeckendorf分解的依赖"""
        # 这里简化处理，实际应该调用Zeckendorf分解算法
        deps = []
        remaining = n
        for fib in reversed(self.parser.fibonacci_sequence):
            if fib <= remaining and fib < n:
                deps.append(fib)
                remaining -= fib
                if remaining == 0:
                    break
        return sorted(deps)
    
    def _get_twin_pair(self, n: int) -> Tuple[int, int]:
        """获取孪生素数对"""
        if self.prime_checker.is_prime(n - 2):
            return (n - 2, n)
        elif self.prime_checker.is_prime(n + 2):
            return (n, n + 2)
        return None
    
    def _get_mersenne_exponent(self, n: int) -> int:
        """获取梅森素数的指数"""
        import math
        if self.prime_checker.is_mersenne_prime(n):
            return int(math.log2(n + 1))
        return None
    
    def _format_prime_factors(self, factors: List[Tuple[int, int]]) -> str:
        """格式化素因子分解"""
        parts = []
        for prime, exp in factors:
            if exp == 1:
                parts.append(str(prime))
            else:
                parts.append(f"{prime}^{exp}")
        return " × ".join(parts)
    
    def generate_classification_report(self, max_n: int = 100) -> str:
        """生成分类报告"""
        classifications = self.classify_range(1, max_n)
        
        # 统计各类理论
        stats = {
            FibonacciOperationType.AXIOM: [],
            FibonacciOperationType.PRIME_FIB: [],
            FibonacciOperationType.FIBONACCI: [],
            FibonacciOperationType.PRIME: [],
            FibonacciOperationType.COMPOSITE: []
        }
        
        for n, cls in classifications.items():
            stats[cls.class_type].append(n)
        
        # 生成报告
        report = []
        report.append("=" * 60)
        report.append(f"素数理论分类报告 (T1-T{max_n})")
        report.append("=" * 60)
        
        report.append(f"\n统计概览:")
        report.append(f"  AXIOM (公理): {len(stats[FibonacciOperationType.AXIOM])} 个")
        report.append(f"  PRIME-FIB (素数-Fibonacci): {len(stats[FibonacciOperationType.PRIME_FIB])} 个")
        report.append(f"  FIBONACCI (纯Fibonacci): {len(stats[FibonacciOperationType.FIBONACCI])} 个")
        report.append(f"  PRIME (纯素数): {len(stats[FibonacciOperationType.PRIME])} 个")
        report.append(f"  COMPOSITE (合数): {len(stats[FibonacciOperationType.COMPOSITE])} 个")
        
        # 详细列表
        report.append(f"\n详细分类:")
        
        if stats[FibonacciOperationType.AXIOM]:
            report.append(f"\n【AXIOM - 公理】")
            report.append(f"  T{stats[FibonacciOperationType.AXIOM]}")
        
        if stats[FibonacciOperationType.PRIME_FIB]:
            report.append(f"\n【PRIME-FIB - 素数-Fibonacci双重理论】")
            report.append(f"  T{stats[FibonacciOperationType.PRIME_FIB]}")
        
        if stats[FibonacciOperationType.FIBONACCI]:
            report.append(f"\n【FIBONACCI - 纯Fibonacci理论】")
            report.append(f"  T{stats[FibonacciOperationType.FIBONACCI][:20]}")  # 只显示前20个
            if len(stats[FibonacciOperationType.FIBONACCI]) > 20:
                report.append(f"  ... 还有 {len(stats[FibonacciOperationType.FIBONACCI])-20} 个")
        
        if stats[FibonacciOperationType.PRIME]:
            report.append(f"\n【PRIME - 纯素数理论】")
            report.append(f"  T{stats[FibonacciOperationType.PRIME][:30]}")  # 只显示前30个
            if len(stats[FibonacciOperationType.PRIME]) > 30:
                report.append(f"  ... 还有 {len(stats[FibonacciOperationType.PRIME])-30} 个")
        
        # 特殊素数
        report.append(f"\n特殊素数理论:")
        special_primes = []
        for n in stats[FibonacciOperationType.PRIME] + stats[FibonacciOperationType.PRIME_FIB]:
            cls = classifications[n]
            if cls.is_twin_prime or cls.is_mersenne_prime or cls.is_sophie_germain:
                special_desc = []
                if cls.is_twin_prime:
                    special_desc.append("孪生")
                if cls.is_mersenne_prime:
                    special_desc.append("梅森")
                if cls.is_sophie_germain:
                    special_desc.append("Sophie Germain")
                special_primes.append(f"T{n}({','.join(special_desc)})")
        
        if special_primes:
            report.append(f"  {', '.join(special_primes[:15])}")
            if len(special_primes) > 15:
                report.append(f"  ... 还有 {len(special_primes)-15} 个特殊素数")
        
        return "\n".join(report)


def main():
    """测试分类器"""
    classifier = PrimeTheoryClassifier()
    
    # 测试单个理论分类
    test_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 17, 19, 21, 23, 29, 31, 34, 55, 89]
    
    print("单个理论分类测试:")
    print("=" * 60)
    for n in test_numbers:
        cls = classifier.classify_theory(n)
        deps = classifier.get_theory_dependencies(cls)
        print(f"T{n:3d}: {cls.class_type.value:12s} | 素数:{cls.is_prime} | Fib:{cls.is_fibonacci} | 依赖:T{deps}")
        if cls.special_properties:
            print(f"      特性: {', '.join(cls.special_properties)}")
    
    # 生成分类报告
    print("\n" + classifier.generate_classification_report(100))
    
    # 测试素数意义分析
    print("\n素数理论意义分析:")
    print("=" * 60)
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        sig = classifier.get_prime_significance(p)
        if sig:
            print(f"T{p}: {sig}")


if __name__ == "__main__":
    main()