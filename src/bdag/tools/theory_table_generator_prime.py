#!/usr/bin/env python3
"""
T{n} 理论表生成器 (素数增强版) v3.0
基于五类分类系统：AXIOM/PRIME-FIB/FIBONACCI/PRIME/COMPOSITE
包含素数标记、素因子分解和特殊素数类检测
"""

from typing import List, Dict, Tuple, Optional
try:
    from .prime_theory_classifier import PrimeTheoryClassifier
    from .theory_validator import PrimeChecker
    from .theory_parser import FibonacciOperationType
except ImportError:
    from prime_theory_classifier import PrimeTheoryClassifier
    from theory_validator import PrimeChecker
    from theory_parser import FibonacciOperationType

# 使用统一的分类枚举
TheoryClassType = FibonacciOperationType


def generate_fibonacci(max_val):
    """生成Fibonacci序列直到超过max_val"""
    standard_fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
    # 扩展序列如果需要
    if max_val > 987:
        a, b = 610, 987
        while b <= max_val:
            a, b = b, a + b
            standard_fib.append(b)
    
    fib_set = set(standard_fib)
    fib_index = {val: i+1 for i, val in enumerate(standard_fib)}
    
    return standard_fib, fib_set, fib_index


def zeckendorf_decompose(n, fib_seq):
    """Zeckendorf分解：将n表示为非连续Fibonacci数之和"""
    if n == 0:
        return []
    
    result = []
    remaining = n
    
    for fib in reversed(fib_seq):
        if fib <= remaining:
            result.append(fib)
            remaining -= fib
            if remaining == 0:
                break
    
    return sorted(result, reverse=True)


def format_prime_factors(factors: List[Tuple[int, int]]) -> str:
    """格式化素因子分解"""
    if not factors:
        return ""
    
    parts = []
    for prime, exp in factors:
        if exp == 1:
            parts.append(str(prime))
        else:
            parts.append(f"{prime}^{exp}")
    return " × ".join(parts)


def get_special_prime_types(n: int, prime_checker: PrimeChecker) -> List[str]:
    """获取特殊素数类型"""
    types = []
    if not prime_checker.is_prime(n):
        return types
    
    if prime_checker.is_twin_prime(n):
        types.append("Twin")
    if prime_checker.is_mersenne_prime(n):
        types.append("Mersenne")
    if prime_checker.is_sophie_germain_prime(n):
        types.append("Sophie")
    
    return types


def generate_enhanced_theory_table(max_n=997):
    """生成增强版理论表，包含五类分类和素数信息"""
    
    # 初始化工具
    classifier = PrimeTheoryClassifier(max_n)
    prime_checker = PrimeChecker()
    
    # 生成Fibonacci序列
    fib_seq, fib_set, fib_index = generate_fibonacci(max_n)
    
    # 理论名称映射
    theory_names = {
        1: "SelfReferenceAxiom",
        2: "EntropyTheorem", 
        3: "ConstraintTheorem",
        5: "SpaceTheorem",
        7: "CodingPrimeTheorem",  # 新增素数理论
        8: "ComplexityTheorem",
        11: "ConstraintComplexityPrime",  # 新增素数理论
        13: "UnifiedFieldTheorem",
        17: "TriplePrimeUnification",  # 新增素数理论
        19: "SpaceUnifiedPrime",  # 新增素数理论
        21: "ConsciousnessTheorem",
        23: "EntropyConsciousnessPrime",  # 新增素数理论
        29: "ComplexConsciousnessPrime",  # 新增素数理论
        31: "MersennePrimeTheorem",  # 新增素数理论
        34: "UniverseMindTheorem",
        55: "MetaUniverseTheorem",
        89: "InfiniteRecursionTheorem",
        144: "CosmicHarmonyTheorem",
        233: "TranscendenceTheorem",
        377: "OmegaPointTheorem",
        610: "SingularityTheorem",
        987: "UltimateRealityTheorem"
    }
    
    # 生成表格
    table = []
    table.append("# T{n} 完整理论系统表 (T1-T997) - 五类分类素数增强版 v3.0")
    table.append("\n## 🎯 五类分类系统说明")
    table.append("\n- 🔴 **AXIOM**: 唯一公理基础（T1）")
    table.append("- ⭐ **PRIME-FIB**: 素数+Fibonacci双重基础理论")  
    table.append("- 🔵 **FIBONACCI**: 纯Fibonacci递归理论")
    table.append("- 🟢 **PRIME**: 纯素数原子理论")
    table.append("- 🟡 **COMPOSITE**: 合数组合理论")
    table.append("\n## 📊 统计概览")
    
    # 统计各类理论
    stats = {
        TheoryClassType.AXIOM: 0,
        TheoryClassType.PRIME_FIB: 0,
        TheoryClassType.FIBONACCI: 0,
        TheoryClassType.PRIME: 0,
        TheoryClassType.COMPOSITE: 0
    }
    
    # 收集素数统计
    prime_count = 0
    twin_primes = []
    mersenne_primes = []
    sophie_germain_primes = []
    
    # 预计算所有分类
    classifications = {}
    for n in range(1, max_n + 1):
        cls = classifier.classify_theory(n)
        classifications[n] = cls
        stats[cls.class_type] += 1
        
        if cls.is_prime:
            prime_count += 1
            if cls.is_twin_prime:
                twin_primes.append(n)
            if cls.is_mersenne_prime:
                mersenne_primes.append(n)
            if cls.is_sophie_germain:
                sophie_germain_primes.append(n)
    
    # 输出统计 
    table.append(f"\n- **总理论数**: {max_n}")
    table.append(f"- 🔴 **AXIOM (公理)**: {stats[TheoryClassType.AXIOM]} 个 ({stats[TheoryClassType.AXIOM]/max_n*100:.1f}%)")
    table.append(f"- ⭐ **PRIME-FIB (双重基础)**: {stats[TheoryClassType.PRIME_FIB]} 个 ({stats[TheoryClassType.PRIME_FIB]/max_n*100:.2f}%)")
    table.append(f"- 🔵 **FIBONACCI (递归)**: {stats[TheoryClassType.FIBONACCI]} 个 ({stats[TheoryClassType.FIBONACCI]/max_n*100:.2f}%)")
    table.append(f"- 🟢 **PRIME (原子)**: {stats[TheoryClassType.PRIME]} 个 ({stats[TheoryClassType.PRIME]/max_n*100:.1f}%)")
    table.append(f"- 🟡 **COMPOSITE (组合)**: {stats[TheoryClassType.COMPOSITE]} 个 ({stats[TheoryClassType.COMPOSITE]/max_n*100:.1f}%)")
    table.append(f"\n- **素数理论总数**: {prime_count}")
    table.append(f"- **孪生素数**: {len(twin_primes)}个")
    table.append(f"- **梅森素数**: {len(mersenne_primes)}个")
    table.append(f"- **Sophie Germain素数**: {len(sophie_germain_primes)}个")
    
    # 列出重要的素数-Fibonacci理论
    table.append("\n## ⭐ PRIME-FIB双重基础理论")
    table.append("\n这些理论同时具有素数性和Fibonacci性的双重数学基础，是系统的最核心支柱：\n")
    for n in range(1, min(max_n + 1, 1000)):
        cls = classifications[n]
        if cls.class_type == TheoryClassType.PRIME_FIB:
            theory_name = theory_names.get(n, f"Theory_{n}")
            special = get_special_prime_types(n, prime_checker)
            special_str = f" ({', '.join(special)})" if special else ""
            table.append(f"- **T{n}** = F{fib_index.get(n, '?')} - {theory_name}{special_str}")
    
    # 列出纯Fibonacci理论
    table.append("\n## 🔵 纯Fibonacci递归理论")
    table.append("\n这些理论是Fibonacci数但不是素数，体现纯递归涌现性质：\n")
    pure_fibs = [n for n in range(1, max_n + 1) if classifications[n].class_type == TheoryClassType.FIBONACCI]
    for n in pure_fibs:
        theory_name = theory_names.get(n, f"FibonacciTheory_{n}")
        prime_factors = prime_checker.prime_factorize(n) if n > 1 else []
        factor_str = format_prime_factors(prime_factors)
        table.append(f"- **T{n}** = F{fib_index.get(n, '?')} = {factor_str} - {theory_name}")
    
    # 列出纯素数理论（前30个）
    table.append("\n## 🟢 纯素数原子理论（前30个）")
    table.append("\n这些理论位于素数位置但不是Fibonacci数，代表不可分解的原子构建块：\n")
    pure_primes = [n for n in range(1, max_n + 1) if classifications[n].class_type == TheoryClassType.PRIME]
    for n in pure_primes[:30]:
        zeck = zeckendorf_decompose(n, fib_seq)
        zeck_str = "+".join([f"F{fib_index.get(f, '?')}" for f in zeck if f in fib_index])
        deps = "+".join([f"T{f}" for f in zeck])
        theory_name = theory_names.get(n, f"PrimeTheory_{n}")
        special = get_special_prime_types(n, prime_checker)
        special_str = f" [{', '.join(special)}]" if special else ""
        table.append(f"- **T{n}** = {zeck_str} ← FROM {deps} - {theory_name}{special_str}")
    
    # 列出特殊素数
    table.append("\n## 🎯 特殊素数理论")
    
    if twin_primes:
        table.append("\n### 孪生素数")
        pairs = []
        i = 0
        while i < len(twin_primes):
            n = twin_primes[i]
            if i + 1 < len(twin_primes) and twin_primes[i + 1] == n + 2:
                pairs.append(f"(T{n}, T{n+2})")
                i += 2
            else:
                # 单个孪生素数（对在范围外）
                if prime_checker.is_prime(n - 2):
                    pairs.append(f"(T{n-2}, T{n})")
                else:
                    pairs.append(f"(T{n}, T{n+2})")
                i += 1
        table.append(", ".join(pairs[:15]))
        if len(pairs) > 15:
            table.append(f"... 还有{len(pairs)-15}对")
    
    if mersenne_primes[:10]:
        table.append("\n### 梅森素数 (2^p - 1形式)")
        mersenne_list = []
        for n in mersenne_primes[:10]:
            import math
            p = int(math.log2(n + 1))
            mersenne_list.append(f"T{n} = 2^{p}-1")
        table.append(", ".join(mersenne_list))
    
    if sophie_germain_primes[:15]:
        table.append("\n### Sophie Germain素数 (p和2p+1都是素数)")
        sophie_list = [f"T{n} (安全素数:T{2*n+1})" if prime_checker.is_prime(2*n+1) and 2*n+1 <= max_n 
                      else f"T{n}" for n in sophie_germain_primes[:15]]
        table.append(", ".join(sophie_list))
    
    # 完整理论表（前100个）
    table.append("\n## 📋 完整理论表（前100个）")
    table.append("\n| T{n} | 类型 | 素数 | Fib | Zeckendorf | 素因子 | 特殊性质 | 理论名称 |")
    table.append("|------|------|------|-----|------------|--------|----------|----------|")
    
    for n in range(1, min(101, max_n + 1)):
        cls = classifications[n]
        
        # 类型
        type_map = {
            TheoryClassType.AXIOM: "**AXIOM**",
            TheoryClassType.PRIME_FIB: "**PRIME-FIB**",
            TheoryClassType.FIBONACCI: "FIBONACCI",
            TheoryClassType.PRIME: "PRIME",
            TheoryClassType.COMPOSITE: "COMPOSITE"
        }
        type_str = type_map[cls.class_type]
        
        # 素数标记
        prime_str = "✓" if cls.is_prime else "-"
        
        # Fibonacci标记
        fib_str = f"F{fib_index[n]}" if n in fib_index else "-"
        
        # Zeckendorf分解
        if n == 1:
            zeck_str = "F1"
        elif n in fib_set:
            zeck_str = f"F{fib_index[n]}"
        else:
            zeck = zeckendorf_decompose(n, fib_seq)
            zeck_str = "+".join([f"F{fib_index.get(f, '?')}" for f in zeck if f in fib_index])
        
        # 素因子分解
        if cls.prime_factors:
            factor_str = format_prime_factors(cls.prime_factors)
        elif cls.is_prime:
            factor_str = "素数"
        else:
            factor_str = "-"
        
        # 特殊性质
        special = []
        if cls.is_twin_prime:
            special.append("Twin")
        if cls.is_mersenne_prime:
            special.append("Mers")
        if cls.is_sophie_germain:
            special.append("SG")
        special_str = ",".join(special) if special else "-"
        
        # 理论名称
        theory_name = theory_names.get(n, f"Theory_{n}")
        
        # 加粗重要理论
        if cls.class_type in [TheoryClassType.AXIOM, TheoryClassType.PRIME_FIB]:
            row = f"| **T{n}** | {type_str} | {prime_str} | {fib_str} | {zeck_str} | {factor_str} | {special_str} | **{theory_name}** |"
        else:
            row = f"| T{n} | {type_str} | {prime_str} | {fib_str} | {zeck_str} | {factor_str} | {special_str} | {theory_name} |"
        
        table.append(row)
    
    # 素数密度分析
    table.append("\n## 📈 素数密度分析")
    ranges = [(1, 100), (101, 200), (201, 300), (301, 400), (401, 500), 
              (501, 600), (601, 700), (701, 800), (801, 900), (901, 997)]
    
    table.append("\n| 范围 | 素数个数 | 密度 | 预期(素数定理) |")
    table.append("|------|----------|------|---------------|")
    
    import math
    for start, end in ranges:
        if end > max_n:
            end = max_n
        if start > max_n:
            break
        
        primes_in_range = sum(1 for n in range(start, end + 1) 
                              if classifications[n].is_prime)
        density = primes_in_range / (end - start + 1)
        expected = 1 / math.log(start + (end - start) / 2)  # 素数定理近似
        
        table.append(f"| T{start}-T{end} | {primes_in_range} | {density:.2%} | {expected:.2%} |")
    
    return "\n".join(table)


def main():
    """生成增强版理论表"""
    # 生成完整表格
    table_content = generate_enhanced_theory_table(997)
    
    # 保存到文件
    output_file = "T1_T997_prime_enhanced_table.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(table_content)
    
    print(f"已生成增强版理论表: {output_file}")
    
    # 打印前100行预览
    lines = table_content.split('\n')
    for line in lines[:100]:
        print(line)
    
    print(f"\n... 完整表格已保存到 {output_file}")


if __name__ == "__main__":
    main()