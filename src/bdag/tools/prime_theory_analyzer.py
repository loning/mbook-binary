#!/usr/bin/env python3
"""
Prime Theory Analyzer v3.0
专门分析素数理论特性的高级工具
分析素数理论在T{n}系统中的数学性质和宇宙意义
"""

import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict

try:
    from .theory_validator import PrimeChecker
    from .fibonacci_tensor_space import UniversalTensorSpace, TensorClassification
except ImportError:
    from theory_validator import PrimeChecker
    from fibonacci_tensor_space import UniversalTensorSpace, TensorClassification

class PrimeType(Enum):
    """素数类型分类"""
    REGULAR_PRIME = "regular_prime"
    TWIN_PRIME = "twin_prime"
    MERSENNE_PRIME = "mersenne_prime"
    SOPHIE_GERMAIN_PRIME = "sophie_germain_prime"
    SAFE_PRIME = "safe_prime"
    PALINDROMIC_PRIME = "palindromic_prime"
    CIRCULAR_PRIME = "circular_prime"

@dataclass
class PrimeTheoryAnalysis:
    """素数理论分析结果"""
    theory_number: int
    prime_type: PrimeType
    is_prime_fib: bool                    # 是否为PRIME-FIB双重理论
    twin_prime_partner: Optional[int]     # 孪生素数伙伴
    sophie_germain_safe: Optional[int]    # Sophie Germain对应的安全素数
    gap_to_next_prime: int               # 到下一个素数的间隔
    gap_to_prev_prime: int               # 到上一个素数的间隔
    cumulative_prime_index: int          # 在素数序列中的位置
    theoretical_density: float           # 理论密度（根据素数定理）
    actual_density: float               # 实际密度
    primality_strength: float           # 素数强度指标
    universe_significance: str          # 宇宙意义描述
    zeckendorf_representation: List[int] # Zeckendorf分解
    prime_factorization_context: Dict   # 与其他理论的素因子关系

class PrimeTheoryAnalyzer:
    """素数理论分析器"""
    
    def __init__(self, max_theory: int = 997):
        self.max_theory = max_theory
        self.prime_checker = PrimeChecker()
        self.tensor_space = UniversalTensorSpace(max_theory)
        
        # 生成基础数据
        self.primes = self.prime_checker.get_primes_up_to(max_theory)
        self.prime_set = set(self.primes)
        self.fibonacci_set = set(self.tensor_space.fibonacci_sequence)
        
        # 分析缓存
        self._analysis_cache: Dict[int, PrimeTheoryAnalysis] = {}
        
    def analyze_prime_theory(self, prime_number: int) -> PrimeTheoryAnalysis:
        """深度分析单个素数理论"""
        if prime_number in self._analysis_cache:
            return self._analysis_cache[prime_number]
        
        if not self.prime_checker.is_prime(prime_number):
            raise ValueError(f"{prime_number} is not a prime number")
        
        # 确定素数类型
        prime_type = self._classify_prime_type(prime_number)
        
        # 分析孪生素数
        twin_partner = self._find_twin_prime_partner(prime_number)
        
        # 分析Sophie Germain素数
        sophie_safe = self._find_sophie_germain_safe(prime_number)
        
        # 计算素数间隔
        prev_gap, next_gap = self._calculate_prime_gaps(prime_number)
        
        # 计算素数位置
        prime_index = self.primes.index(prime_number) + 1
        
        # 计算密度
        theoretical_density = self._theoretical_prime_density(prime_number)
        actual_density = prime_index / prime_number
        
        # 计算素数强度
        primality_strength = self._calculate_primality_strength(prime_number)
        
        # 判断是否为PRIME-FIB
        is_prime_fib = prime_number in self.fibonacci_set
        
        # 获取宇宙意义
        universe_significance = self._get_universe_significance(prime_number)
        
        # Zeckendorf分解
        zeckendorf = self.tensor_space._to_zeckendorf(prime_number)
        
        # 素因子关系分析
        factorization_context = self._analyze_factorization_context(prime_number)
        
        analysis = PrimeTheoryAnalysis(
            theory_number=prime_number,
            prime_type=prime_type,
            is_prime_fib=is_prime_fib,
            twin_prime_partner=twin_partner,
            sophie_germain_safe=sophie_safe,
            gap_to_next_prime=next_gap,
            gap_to_prev_prime=prev_gap,
            cumulative_prime_index=prime_index,
            theoretical_density=theoretical_density,
            actual_density=actual_density,
            primality_strength=primality_strength,
            universe_significance=universe_significance,
            zeckendorf_representation=zeckendorf,
            prime_factorization_context=factorization_context
        )
        
        self._analysis_cache[prime_number] = analysis
        return analysis
    
    def _classify_prime_type(self, p: int) -> PrimeType:
        """分类素数类型"""
        if self.prime_checker.is_mersenne_prime(p):
            return PrimeType.MERSENNE_PRIME
        elif self.prime_checker.is_twin_prime(p):
            return PrimeType.TWIN_PRIME
        elif self.prime_checker.is_sophie_germain_prime(p):
            return PrimeType.SOPHIE_GERMAIN_PRIME
        elif self._is_safe_prime(p):
            return PrimeType.SAFE_PRIME
        elif self._is_palindromic_prime(p):
            return PrimeType.PALINDROMIC_PRIME
        elif self._is_circular_prime(p):
            return PrimeType.CIRCULAR_PRIME
        else:
            return PrimeType.REGULAR_PRIME
    
    def _is_safe_prime(self, p: int) -> bool:
        """检查是否为安全素数"""
        if p <= 2:
            return False
        return self.prime_checker.is_prime((p - 1) // 2)
    
    def _is_palindromic_prime(self, p: int) -> bool:
        """检查是否为回文素数"""
        return str(p) == str(p)[::-1]
    
    def _is_circular_prime(self, p: int) -> bool:
        """检查是否为循环素数"""
        s = str(p)
        for i in range(len(s)):
            rotated = int(s[i:] + s[:i])
            if not self.prime_checker.is_prime(rotated):
                return False
        return True
    
    def _find_twin_prime_partner(self, p: int) -> Optional[int]:
        """找到孪生素数伙伴"""
        if p + 2 in self.prime_set:
            return p + 2
        elif p - 2 in self.prime_set:
            return p - 2
        return None
    
    def _find_sophie_germain_safe(self, p: int) -> Optional[int]:
        """找到Sophie Germain对应的安全素数"""
        safe_candidate = 2 * p + 1
        if safe_candidate <= self.max_theory and safe_candidate in self.prime_set:
            return safe_candidate
        return None
    
    def _calculate_prime_gaps(self, p: int) -> Tuple[int, int]:
        """计算到前后素数的间隔"""
        try:
            p_index = self.primes.index(p)
        except ValueError:
            return 0, 0
        
        prev_gap = 0
        next_gap = 0
        
        if p_index > 0:
            prev_gap = p - self.primes[p_index - 1]
        
        if p_index < len(self.primes) - 1:
            next_gap = self.primes[p_index + 1] - p
        
        return prev_gap, next_gap
    
    def _theoretical_prime_density(self, n: int) -> float:
        """根据素数定理计算理论密度"""
        if n <= 1:
            return 0.0
        return 1.0 / math.log(n)
    
    def _calculate_primality_strength(self, p: int) -> float:
        """计算素数强度指标"""
        # 基础强度 = 1（所有素数）
        strength = 1.0
        
        # 特殊类型加权
        prime_type = self._classify_prime_type(p)
        type_weights = {
            PrimeType.MERSENNE_PRIME: 3.0,
            PrimeType.TWIN_PRIME: 2.0,
            PrimeType.SOPHIE_GERMAIN_PRIME: 2.5,
            PrimeType.SAFE_PRIME: 1.8,
            PrimeType.PALINDROMIC_PRIME: 1.5,
            PrimeType.CIRCULAR_PRIME: 1.7,
            PrimeType.REGULAR_PRIME: 1.0
        }
        strength *= type_weights[prime_type]
        
        # PRIME-FIB双重基础加权
        if p in self.fibonacci_set:
            strength *= 5.0
        
        # 大小调节因子
        size_factor = math.log10(p) / math.log10(self.max_theory)
        strength *= (1.0 + size_factor)
        
        return strength
    
    def _get_universe_significance(self, p: int) -> str:
        """获取素数的宇宙意义"""
        significance_map = {
            2: "熵增定理 - 热力学基础原子",
            3: "约束定理 - 秩序涌现原子", 
            5: "空间定理 - 维度基础原子",
            7: "编码定理 - 信息原子基础",
            11: "十一维定理 - 弦论基础原子",
            13: "统一场定理 - 力的统一原子",
            17: "周期定理 - 循环原子基础",
            19: "间隙定理 - 分布原子基础",
            23: "对称定理 - 不变原子基础",
            29: "孪生定理 - 关联原子基础",
            31: "梅森定理 - 完美原子基础",
            37: "螺旋定理 - 动态原子基础",
            41: "维度定理 - 高维原子基础",
            43: "共振定理 - 谐波原子基础",
            47: "质数定理 - 元原子基础"
        }
        
        if p in significance_map:
            return significance_map[p]
        
        # 动态生成意义
        if p in self.fibonacci_set:
            return f"双重基础定理 - 第{p}原子-递归"
        elif self._classify_prime_type(p) == PrimeType.TWIN_PRIME:
            return f"孪生原子定理 - 第{p}关联原子"
        elif self._classify_prime_type(p) == PrimeType.MERSENNE_PRIME:
            return f"梅森原子定理 - 第{p}完美原子"
        else:
            return f"原子定理 - 第{p}不可分解构建块"
    
    def _analyze_factorization_context(self, p: int) -> Dict:
        """分析素因子关系上下文"""
        context = {
            'appears_in_theories': [],      # 作为素因子出现在哪些理论中
            'factorizes_theories': [],      # 该素数可以因子化哪些理论
            'multiplicative_relationships': {}  # 乘积关系
        }
        
        # 查找该素数作为素因子出现的理论
        for n in range(2, min(self.max_theory + 1, p * 50)):  # 限制搜索范围
            if n in self.tensor_space.basis_tensors:
                tensor = self.tensor_space.basis_tensors[n]
                for prime_factor, power in tensor.prime_factors:
                    if prime_factor == p:
                        context['appears_in_theories'].append(n)
                        break
        
        # 分析乘积关系（仅检查小范围）
        for other_prime in self.primes:
            if other_prime != p and other_prime <= self.max_theory // p:
                product = p * other_prime
                if product <= self.max_theory:
                    context['multiplicative_relationships'][other_prime] = product
        
        return context
    
    def analyze_all_prime_theories(self) -> Dict[int, PrimeTheoryAnalysis]:
        """分析所有素数理论"""
        all_analyses = {}
        
        for prime in self.primes:
            if prime <= self.max_theory:
                all_analyses[prime] = self.analyze_prime_theory(prime)
        
        return all_analyses
    
    def get_prime_statistics(self) -> Dict:
        """获取素数理论统计信息"""
        analyses = self.analyze_all_prime_theories()
        
        # 按类型统计
        type_counts = defaultdict(int)
        prime_fib_count = 0
        strength_distribution = []
        gap_distribution = []
        
        for analysis in analyses.values():
            type_counts[analysis.prime_type.value] += 1
            if analysis.is_prime_fib:
                prime_fib_count += 1
            strength_distribution.append(analysis.primality_strength)
            gap_distribution.append(analysis.gap_to_next_prime)
        
        return {
            'total_prime_theories': len(analyses),
            'prime_fib_theories': prime_fib_count,
            'type_distribution': dict(type_counts),
            'average_strength': np.mean(strength_distribution) if strength_distribution else 0,
            'max_strength': max(strength_distribution) if strength_distribution else 0,
            'average_gap': np.mean([g for g in gap_distribution if g > 0]) if gap_distribution else 0,
            'max_gap': max(gap_distribution) if gap_distribution else 0,
            'strength_distribution': strength_distribution,
            'gap_distribution': gap_distribution
        }
    
    def find_strongest_prime_theories(self, top_n: int = 10) -> List[Tuple[int, float]]:
        """找到强度最高的素数理论"""
        analyses = self.analyze_all_prime_theories()
        
        strength_pairs = [(prime, analysis.primality_strength) 
                         for prime, analysis in analyses.items()]
        
        return sorted(strength_pairs, key=lambda x: x[1], reverse=True)[:top_n]
    
    def find_prime_clusters(self, max_gap: int = 6) -> List[List[int]]:
        """找到素数聚类（间隔较小的素数组）"""
        clusters = []
        current_cluster = [self.primes[0]] if self.primes else []
        
        for i in range(1, len(self.primes)):
            gap = self.primes[i] - self.primes[i-1]
            
            if gap <= max_gap:
                current_cluster.append(self.primes[i])
            else:
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [self.primes[i]]
        
        # 添加最后一个聚类
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
        
        return clusters
    
    def generate_prime_analysis_report(self, prime_number: int) -> str:
        """生成单个素数理论的详细分析报告"""
        analysis = self.analyze_prime_theory(prime_number)
        
        report = []
        report.append(f"🔢 素数理论T{prime_number}深度分析报告")
        report.append("=" * 50)
        
        report.append(f"\n📊 基本信息:")
        report.append(f"  素数: {prime_number}")
        report.append(f"  类型: {analysis.prime_type.value}")
        report.append(f"  是否PRIME-FIB: {'✅ 是' if analysis.is_prime_fib else '❌ 否'}")
        report.append(f"  素数序列位置: #{analysis.cumulative_prime_index}")
        
        report.append(f"\n🎯 数学性质:")
        report.append(f"  前向间隔: {analysis.gap_to_prev_prime}")
        report.append(f"  后向间隔: {analysis.gap_to_next_prime}")
        report.append(f"  理论密度: {analysis.theoretical_density:.6f}")
        report.append(f"  实际密度: {analysis.actual_density:.6f}")
        report.append(f"  素数强度: {analysis.primality_strength:.3f}")
        
        if analysis.twin_prime_partner:
            report.append(f"  孪生素数伙伴: {analysis.twin_prime_partner}")
        
        if analysis.sophie_germain_safe:
            report.append(f"  Sophie Germain安全素数: {analysis.sophie_germain_safe}")
        
        report.append(f"\n🌌 宇宙意义:")
        report.append(f"  {analysis.universe_significance}")
        
        report.append(f"\n📐 Zeckendorf分解:")
        report.append(f"  {' + '.join(f'F{f}' for f in analysis.zeckendorf_representation)}")
        
        if analysis.prime_factorization_context['appears_in_theories']:
            appears_in = analysis.prime_factorization_context['appears_in_theories'][:10]
            report.append(f"\n🔗 作为素因子出现在:")
            report.append(f"  {appears_in} ({'...' if len(appears_in) == 10 else ''})")
        
        if analysis.prime_factorization_context['multiplicative_relationships']:
            mult_rel = analysis.prime_factorization_context['multiplicative_relationships']
            report.append(f"\n✖️ 乘积关系 (前5个):")
            for other_p, product in list(mult_rel.items())[:5]:
                report.append(f"  T{prime_number} × T{other_p} = T{product}")
        
        return "\n".join(report)

def main():
    """主函数 - 演示素数理论分析器"""
    print("🔍 素数理论分析器演示")
    print("=" * 50)
    
    analyzer = PrimeTheoryAnalyzer(max_theory=100)  # 限制范围便于演示
    
    # 分析几个重要的素数理论
    important_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    print("\n⭐ 重要素数理论分析:")
    for p in important_primes:
        analysis = analyzer.analyze_prime_theory(p)
        classification = "PRIME-FIB" if analysis.is_prime_fib else "PRIME"
        print(f"  T{p}: {analysis.prime_type.value} - {classification} - 强度:{analysis.primality_strength:.2f}")
    
    # 统计信息
    stats = analyzer.get_prime_statistics()
    print(f"\n📊 素数理论统计:")
    print(f"  总素数理论数: {stats['total_prime_theories']}")
    print(f"  PRIME-FIB理论数: {stats['prime_fib_theories']}")
    print(f"  平均强度: {stats['average_strength']:.3f}")
    print(f"  最大强度: {stats['max_strength']:.3f}")
    print(f"  平均间隔: {stats['average_gap']:.2f}")
    
    # 最强素数理论
    strongest = analyzer.find_strongest_prime_theories(5)
    print(f"\n💪 强度最高的素数理论:")
    for i, (prime, strength) in enumerate(strongest, 1):
        print(f"  #{i}: T{prime} - 强度 {strength:.3f}")
    
    # 素数聚类
    clusters = analyzer.find_prime_clusters(max_gap=6)
    print(f"\n🔗 素数聚类 (间隔≤6):")
    for i, cluster in enumerate(clusters[:5], 1):
        print(f"  聚类{i}: {cluster}")
    
    # 详细分析示例
    print(f"\n📝 T13详细分析报告:")
    print(analyzer.generate_prime_analysis_report(13))

if __name__ == "__main__":
    main()