#!/usr/bin/env python3
"""
T{n} 理论表生成器 v3.0 - 扩展到T997
基于五类分类系统的完整理论框架：
- AXIOM: 唯一公理（T1）
- PRIME-FIB: 素数+Fibonacci双重基础
- FIBONACCI: 纯Fibonacci递归理论 
- PRIME: 纯素数原子理论
- COMPOSITE: 合数组合理论
"""

try:
    from .theory_validator import PrimeChecker
except ImportError:
    from theory_validator import PrimeChecker

def generate_fibonacci(max_val):
    """生成Fibonacci序列直到超过max_val"""
    fib_seq = [1, 1]  # F0=0, F1=1, F2=1, 但我们从F1=1开始
    fib_dict = {1: 1, 2: 1}
    k = 2
    
    while fib_seq[-1] <= max_val:
        next_fib = fib_seq[-1] + fib_seq[-2]
        k += 1
        fib_seq.append(next_fib)
        fib_dict[k] = next_fib
    
    # 修正为标准Fibonacci: F1=1, F2=2, F3=3, F4=5...
    standard_fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
    fib_set = set(standard_fib)
    fib_index = {val: i+1 for i, val in enumerate(standard_fib)}
    
    return standard_fib, fib_set, fib_index

def zeckendorf_decompose(n, fib_seq):
    """Zeckendorf分解：将n表示为非连续Fibonacci数之和"""
    if n == 0:
        return []
    
    # 使用贪心算法，从最大的Fibonacci数开始
    result = []
    remaining = n
    
    # 从大到小遍历Fibonacci数
    for fib in reversed(fib_seq):
        if fib <= remaining:
            result.append(fib)
            remaining -= fib
            if remaining == 0:
                break
    
    return sorted(result, reverse=True)

def get_theory_type(n, fib_set, prime_set):
    """确定理论类型（五类分类系统）"""
    if n == 1:
        return "AXIOM"
    elif n in fib_set and n in prime_set:
        return "PRIME-FIB"
    elif n in fib_set:
        return "FIBONACCI"
    elif n in prime_set:
        return "PRIME"
    else:
        return "COMPOSITE"

def get_theory_name(n, theory_type, fib_index):
    """生成理论名称（五类分类系统）"""
    theory_names = {
        # AXIOM
        1: "SelfReferenceAxiom",
        
        # PRIME-FIB (双重基础)
        2: "EntropyTheorem", 
        3: "ConstraintTheorem",
        5: "SpaceTheorem",
        13: "UnifiedFieldTheorem",
        89: "InfiniteRecursionTheorem",
        233: "TranscendenceTheorem",
        
        # FIBONACCI (纯递归)
        8: "ComplexityTheorem",
        21: "ConsciousnessTheorem",
        34: "UniverseMindTheorem",
        55: "MetaUniverseTheorem",
        144: "CosmicHarmonyTheorem",
        377: "OmegaPointTheorem",
        610: "SingularityTheorem",
        987: "UltimateRealityTheorem",
        
        # PRIME (纯原子)
        7: "CodingTheorem",
        11: "DimensionTheorem",
        17: "CyclicTheorem",
        19: "GapTheorem",
        23: "SymmetryTheorem",
        29: "TwinTheorem",
        31: "MersenneTheorem",
        37: "SpiralTheorem",
        41: "DimensionalTheorem",
        43: "ResonanceTheorem",
        47: "PrimalityTheorem",
        
        # COMPOSITE (组合)
        4: "TimeExtended",
        6: "QuantumExtended", 
        9: "ObserverExtended",
        10: "PhiComplexExtended",
        12: "TripleExtended",
        14: "SymmetryExtended",
        15: "ProductExtended",
        16: "PowerExtended",
        18: "DoubleExtended",
        20: "DecimalExtended"
    }
    
    if n in theory_names:
        return theory_names[n]
    
    # 为其他理论生成名称
    if theory_type == "PRIME-FIB":
        return f"DualFoundationTheorem_F{fib_index.get(n, 'X')}_P{n}"
    elif theory_type == "FIBONACCI":
        return f"RecursiveTheorem_F{fib_index.get(n, 'X')}"
    elif theory_type == "PRIME":
        return f"AtomicTheorem_P{n}"
    elif theory_type == "COMPOSITE":
        return f"CompositeTheory_{n}"
    else:
        return f"Theory_{n}"

def get_tensor_name(n, theory_type):
    """生成张量名称（五类分类系统）"""
    tensor_names = {
        # AXIOM
        1: "SelfRefTensor",
        
        # PRIME-FIB (双重张量)
        2: "EntropyTensor",
        3: "ConstraintTensor", 
        5: "SpaceTensor",
        13: "UnifiedTensor",
        89: "InfiniteRecursionTensor",
        233: "TranscendenceTensor",
        
        # FIBONACCI (递归张量)
        8: "ComplexityTensor",
        21: "ConsciousnessTensor",
        34: "UniverseMindTensor",
        55: "MetaUniverseTensor",
        144: "CosmicHarmonyTensor",
        377: "OmegaPointTensor",
        610: "SingularityTensor",
        987: "UltimateRealityTensor",
        
        # PRIME (原子张量)
        7: "CodingTensor",
        11: "DimensionTensor",
        17: "CyclicTensor",
        19: "GapTensor",
        23: "SymmetryTensor",
        29: "TwinTensor",
        31: "MersenneTensor",
        37: "SpiralTensor",
        41: "DimensionalTensor",
        43: "ResonanceTensor",
        47: "PrimalityTensor",
        
        # COMPOSITE (组合张量)
        4: "TimeTensor",
        6: "QuantumTensor",
        9: "ObserverTensor",
        10: "PhiTensor",
        12: "TripleTensor",
        14: "SymmetryTensor",
        15: "ProductTensor",
        16: "PowerTensor",
        18: "DoubleTensor",
        20: "DecimalTensor"
    }
    
    if n in tensor_names:
        return tensor_names[n]
    
    # 为其他理论生成张量名称
    if theory_type == "PRIME-FIB":
        return f"DualTensor_{n}"
    elif theory_type == "FIBONACCI":
        return f"RecursiveTensor_{n}"
    elif theory_type == "PRIME":
        return f"AtomicTensor_{n}"
    elif theory_type == "COMPOSITE":
        return f"CompositeTensor_{n}"
    else:
        return f"Tensor_{n}"

def generate_from_source(zeck_components):
    """根据Zeckendorf分解生成FROM来源"""
    if not zeck_components:
        return "UNIVERSE"
    return "+".join(f"T{c}" for c in sorted(zeck_components))

def get_zeck_fibonacci_notation(zeck_components, fib_index):
    """生成Zeckendorf的Fibonacci表示法"""
    if not zeck_components:
        return "UNIVERSE"
    
    fib_notations = []
    for comp in zeck_components:
        if comp in fib_index:
            fib_notations.append(f"F{fib_index[comp]}")
        else:
            fib_notations.append(f"F?{comp}")  # 不应该发生
    
    return "+".join(sorted(fib_notations))

def generate_theory_description(n, theory_type, theory_name, zeck_components, is_prime=False):
    """生成理论描述（五类分类系统）"""
    descriptions = {
        # AXIOM
        1: "自指完备公理 - 宇宙的唯一基础假设",
        
        # PRIME-FIB (双重基础)
        2: "熵增定理 - 双重基础：自指系统必然熵增的原子递归机制", 
        3: "约束定理 - 双重基础：熵增与自指产生约束的原子递归耦合",
        5: "空间定理 - 双重基础：空间维度必然性的原子递归涌现",
        13: "统一场定理 - 双重基础：物理力统一的原子递归框架",
        89: "无限递归定理 - 双重基础：无限深度自指的原子递归本质",
        233: "超越定理 - 双重基础：超越有限机制的原子递归突破",
        
        # FIBONACCI (纯递归)
        8: "复杂性定理 - 复杂性的纯Fibonacci递归涌现",
        21: "意识定理 - 意识的必然Fibonacci递归涌现",
        34: "宇宙心智定理 - 宇宙自我认知的Fibonacci递归结构",
        55: "元宇宙定理 - 多层现实的Fibonacci递归架构",
        144: "宇宙和谐定理 - 万物数学和谐的Fibonacci递归律",
        377: "Ω点定理 - 进化终极目标的Fibonacci递归汇聚",
        610: "奇点定理 - 复杂性奇点的Fibonacci递归临界",
        987: "终极现实定理 - 现实最终本质的Fibonacci递归揭示",
        
        # PRIME (纯原子)
        7: "编码定理 - 信息编码的原子优化机制",
        11: "十一维定理 - 弦论基础的11维原子空间",
        17: "周期定理 - 循环结构的原子周期律",
        19: "间隙定理 - 分布间隙的原子规律性",
        23: "对称定理 - 对称性的原子不变性",
        29: "孪生定理 - 关联结构的原子对偶性",
        31: "梅森定理 - 完美数的原子构造律",
        37: "螺旋定理 - 动态结构的原子螺旋性",
        41: "维度定理 - 高维空间的原子基础",
        43: "共振定理 - 谐波共振的原子机制",
        47: "素数定理 - 素数分布的原子本质",
        
        # COMPOSITE (组合)
        4: "时间扩展定理 - 时间涌现的组合机制",
        6: "量子扩展定理 - 量子现象的组合起源",
        9: "观察者扩展定理 - 观察者效应的组合构造",
        10: "φ复合定理 - 黄金比例的组合展现",
        12: "三重扩展定理 - 三元组合的扩展机制",
        14: "对称扩展定理 - 对称性的组合扩展",
        15: "乘积扩展定理 - 素数乘积的组合效应",
        16: "幂次扩展定理 - 指数增长的组合律",
        18: "双重扩展定理 - 二重结构的组合性",
        20: "十进制定理 - 十进制基础的组合表示"
    }
    
    if n in descriptions:
        return descriptions[n]
    
    # 为其他理论生成描述
    if theory_type == "PRIME-FIB":
        return f"双重基础定理 - 第{n}原子的递归涌现机制"
    elif theory_type == "FIBONACCI":
        return f"Fibonacci递归定理 - F{n}维度的纯递归结构"
    elif theory_type == "PRIME":
        return f"原子定理 - 第{n}不可分解的原子构建块"
    elif theory_type == "COMPOSITE":
        components_str = "+".join(str(c) for c in sorted(zeck_components))
        return f"组合定理 - 基于T{components_str}的合成结构"
    else:
        return f"理论{n} - 待定义类型"

def generate_complete_theory_table(max_n=997):
    """生成完整的T1-T997理论表（五类分类系统）"""
    
    print("🔬 正在生成T{n}理论系统完整表 v3.0...")
    print(f"📊 范围: T1 到 T{max_n}")
    print("🎯 分类系统: AXIOM | PRIME-FIB | FIBONACCI | PRIME | COMPOSITE")
    
    # 生成Fibonacci序列
    fib_seq, fib_set, fib_index = generate_fibonacci(max_n)
    print(f"🔢 Fibonacci序列生成完成: {len(fib_seq)} 个数")
    
    # 生成素数序列
    prime_checker = PrimeChecker()
    primes = prime_checker.get_primes_up_to(max_n)
    prime_set = set(primes)
    print(f"🔣 素数序列生成完成: {len(primes)} 个素数")
    
    # 生成理论表
    theories = []
    classification_counts = {"AXIOM": 0, "PRIME-FIB": 0, "FIBONACCI": 0, "PRIME": 0, "COMPOSITE": 0}
    
    for n in range(1, max_n + 1):
        # Zeckendorf分解
        if n == 1:
            zeck_components = [1]  # F1
        else:
            zeck_components = zeckendorf_decompose(n, fib_seq)
        
        # 确定理论类型（五类分类）
        theory_type = get_theory_type(n, fib_set, prime_set)
        classification_counts[theory_type] += 1
        
        # 生成各种属性
        theory_name = get_theory_name(n, theory_type, fib_index)
        tensor_name = get_tensor_name(n, theory_type)
        from_source = generate_from_source(zeck_components if n > 1 else [])
        zeck_fib_notation = get_zeck_fibonacci_notation(zeck_components, fib_index)
        description = generate_theory_description(n, theory_type, theory_name, zeck_components, n in prime_set)
        
        # 素因子分解
        prime_factors = prime_checker.prime_factorize(n) if n > 1 else []
        
        # 构造理论条目
        theory_entry = {
            'n': n,
            'is_fibonacci': n in fib_set,
            'is_prime': n in prime_set,
            'fibonacci_index': fib_index.get(n, None),
            'prime_factors': prime_factors,
            'zeckendorf_components': zeck_components,
            'zeckendorf_sum': sum(zeck_components),
            'theory_type': theory_type,
            'theory_name': theory_name,
            'tensor_name': tensor_name,
            'from_source': from_source,
            'zeck_fibonacci_notation': zeck_fib_notation,
            'description': description
        }
        
        theories.append(theory_entry)
        
        # 显示进度（重要理论和里程碑）
        if (n % 100 == 0 or n in fib_set or n in prime_set or n <= 25 or 
            theory_type in ["AXIOM", "PRIME-FIB"] or n in [50, 200, 500, 750]):
            status_symbol = {"AXIOM": "🔴", "PRIME-FIB": "⭐", "FIBONACCI": "🔵", 
                            "PRIME": "🟢", "COMPOSITE": "🟡"}[theory_type]
            print(f"✅ T{n}: {status_symbol} {theory_type} - {theory_name}")
    
    print(f"🎯 理论表生成完成: {len(theories)} 个理论")
    print(f"📊 分类统计:")
    for cls, count in classification_counts.items():
        percentage = count / len(theories) * 100
        print(f"   {cls}: {count} 个 ({percentage:.1f}%)")
    
    return theories, fib_seq, fib_set, fib_index, prime_set

def export_markdown_table(theories, output_file="T1_T997_five_class_table.md"):
    """导出为五类分类系统Markdown表格"""
    
    # 统计五类分类
    axiom_count = sum(1 for t in theories if t['theory_type'] == 'AXIOM')
    prime_fib_count = sum(1 for t in theories if t['theory_type'] == 'PRIME-FIB')
    fibonacci_count = sum(1 for t in theories if t['theory_type'] == 'FIBONACCI')
    prime_count = sum(1 for t in theories if t['theory_type'] == 'PRIME')
    composite_count = sum(1 for t in theories if t['theory_type'] == 'COMPOSITE')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# T{n} 五类分类理论系统完整表 v3.0 (T1-T997)\n\n")
        
        f.write("## 🎯 五类分类系统\n\n")
        f.write("本系统基于**素数性**和**Fibonacci性**的双重数学特性进行分类：\n\n")
        f.write("- 🔴 **AXIOM**: 唯一公理基础（T1）\n")
        f.write("- ⭐ **PRIME-FIB**: 素数+Fibonacci双重基础理论\n")
        f.write("- 🔵 **FIBONACCI**: 纯Fibonacci递归理论\n")
        f.write("- 🟢 **PRIME**: 纯素数原子理论\n")
        f.write("- 🟡 **COMPOSITE**: 合数组合理论\n\n")
        
        f.write("## 📊 统计概览\n\n")
        f.write(f"- **总理论数**: {len(theories)}\n")
        f.write(f"- 🔴 **AXIOM**: {axiom_count} 个 ({axiom_count/len(theories)*100:.1f}%)\n")
        f.write(f"- ⭐ **PRIME-FIB**: {prime_fib_count} 个 ({prime_fib_count/len(theories)*100:.1f}%)\n") 
        f.write(f"- 🔵 **FIBONACCI**: {fibonacci_count} 个 ({fibonacci_count/len(theories)*100:.1f}%)\n")
        f.write(f"- 🟢 **PRIME**: {prime_count} 个 ({prime_count/len(theories)*100:.1f}%)\n")
        f.write(f"- 🟡 **COMPOSITE**: {composite_count} 个 ({composite_count/len(theories)*100:.1f}%)\n\n")
        
        f.write("## ⭐ PRIME-FIB 双重基础理论\n\n")
        prime_fib_theories = [t for t in theories if t['theory_type'] == 'PRIME-FIB']
        f.write("同时具备素数性和Fibonacci性的稀有理论：\n\n")
        for t in prime_fib_theories:
            f.write(f"- **T{t['n']}** = F{t['fibonacci_index']} = P{t['n']} - {t['theory_name']}\n")
            f.write(f"  - 素因子: {t['prime_factors']}\n")
            f.write(f"  - 描述: {t['description']}\n\n")
        
        f.write("## 🔵 纯Fibonacci理论\n\n")
        fibonacci_theories = [t for t in theories if t['theory_type'] == 'FIBONACCI']
        for t in fibonacci_theories[:10]:  # 只显示前10个
            f.write(f"- **T{t['n']}** = F{t['fibonacci_index']} - {t['theory_name']}\n")
            f.write(f"  - 素因子: {t['prime_factors']}\n")
            f.write(f"  - 描述: {t['description'][:100]}...\n\n")
        if len(fibonacci_theories) > 10:
            f.write(f"... 以及其他 {len(fibonacci_theories)-10} 个Fibonacci理论\n\n")
        
        f.write("## 🟢 重要素数理论（前20个）\n\n")
        prime_theories = [t for t in theories if t['theory_type'] == 'PRIME'][:20]
        for t in prime_theories:
            f.write(f"- **T{t['n']}** (素数) - {t['theory_name']}\n")
            f.write(f"  - 描述: {t['description'][:80]}...\n\n")
        
        f.write("## 📋 完整理论表\n\n")
        f.write("| T{n} | 类型 | 素数 | Fibonacci | 素因子 | Zeckendorf | FROM来源 | 理论名称 | 张量空间 | 描述 |\n")
        f.write("|------|------|------|-----------|--------|------------|----------|----------|----------|------|\n")
        
        for theory in theories:
            n = theory['n']
            
            # 类型符号
            type_symbols = {"AXIOM": "🔴", "PRIME-FIB": "⭐", "FIBONACCI": "🔵", 
                           "PRIME": "🟢", "COMPOSITE": "🟡"}
            type_display = f"{type_symbols[theory['theory_type']]} **{theory['theory_type']}**"
            
            # 素数标记
            is_prime = "✅" if theory['is_prime'] else "❌"
            
            # Fibonacci标记
            is_fib = "✅" if theory['is_fibonacci'] else "❌"
            fib_notation = f"F{theory['fibonacci_index']}" if theory['is_fibonacci'] else "-"
            
            # 素因子
            prime_factors_str = "×".join(f"{p}^{e}" if e > 1 else str(p) 
                                        for p, e in theory['prime_factors']) if theory['prime_factors'] else "1"
            
            zeck_notation = theory['zeck_fibonacci_notation']
            from_source = theory['from_source']
            theory_name = theory['theory_name']
            tensor_name = theory['tensor_name']
            description = theory['description'][:60] + "..." if len(theory['description']) > 60 else theory['description']
            
            f.write(f"| **T{n}** | {type_display} | {is_prime} | {is_fib} {fib_notation} | {prime_factors_str} | {zeck_notation} | {from_source} | {theory_name} | {tensor_name} | {description} |\n")
    
    print(f"📄 五类分类Markdown表格已导出到: {output_file}")

def export_summary_stats(theories):
    """导出五类分类统计摘要"""
    print("\n" + "="*70)
    print("📊 T{n}五类分类理论系统完整统计 v3.0")
    print("="*70)
    
    # 统计各类数量
    axiom_count = sum(1 for t in theories if t['theory_type'] == 'AXIOM')
    prime_fib_count = sum(1 for t in theories if t['theory_type'] == 'PRIME-FIB')
    fibonacci_count = sum(1 for t in theories if t['theory_type'] == 'FIBONACCI')
    prime_count = sum(1 for t in theories if t['theory_type'] == 'PRIME')
    composite_count = sum(1 for t in theories if t['theory_type'] == 'COMPOSITE')
    
    print(f"🎯 总理论数: {len(theories)}")
    print(f"🔴 AXIOM (公理): {axiom_count}")
    print(f"⭐ PRIME-FIB (双重基础): {prime_fib_count}") 
    print(f"🔵 FIBONACCI (递归): {fibonacci_count}")
    print(f"🟢 PRIME (原子): {prime_count}")
    print(f"🟡 COMPOSITE (组合): {composite_count}")
    
    print(f"\n📈 分布比例:")
    total = len(theories)
    print(f"  - 公理比例: {axiom_count/total*100:.2f}%")
    print(f"  - 双重基础比例: {prime_fib_count/total*100:.2f}%")
    print(f"  - 递归比例: {fibonacci_count/total*100:.2f}%")
    print(f"  - 原子比例: {prime_count/total*100:.2f}%")
    print(f"  - 组合比例: {composite_count/total*100:.2f}%")
    
    print(f"\n⭐ PRIME-FIB双重基础理论 (最稀有):")
    prime_fib_theories = [t for t in theories if t['theory_type'] == 'PRIME-FIB']
    for t in prime_fib_theories:
        print(f"  T{t['n']} = F{t['fibonacci_index']} = P{t['n']} - {t['theory_name']}")
    
    print(f"\n🔵 Fibonacci理论:")
    fibonacci_theories = [t for t in theories if t['is_fibonacci']]
    print(f"  总计: {len(fibonacci_theories)} 个")
    for t in fibonacci_theories[:8]:  # 显示前8个
        classification = "⭐" if t['theory_type'] == 'PRIME-FIB' else "🔵"
        print(f"  {classification} T{t['n']} = F{t['fibonacci_index']} - {t['theory_name']}")
    
    print(f"\n🟢 重要素数理论 (前10个):")
    prime_theories = [t for t in theories if t['is_prime']]
    print(f"  总素数理论: {len(prime_theories)} 个")
    for t in prime_theories[:10]:
        classification = "⭐" if t['theory_type'] == 'PRIME-FIB' else "🟢"
        print(f"  {classification} T{t['n']} - {t['theory_name']}")
    
    print(f"\n🎲 数学性质分析:")
    print(f"  - 总Fibonacci数: {len([t for t in theories if t['is_fibonacci']])}")
    print(f"  - 总素数: {len([t for t in theories if t['is_prime']])}")
    print(f"  - 双重性质(PRIME-FIB): {prime_fib_count}")
    print(f"  - 稀有度分析: PRIME-FIB占总数的 {prime_fib_count/total*100:.3f}%")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    # 生成完整理论表（五类分类系统）
    theories, fib_seq, fib_set, fib_index, prime_set = generate_complete_theory_table(997)
    
    # 导出统计
    export_summary_stats(theories)
    
    # 导出Markdown表格
    export_markdown_table(theories, "T1_T997_five_class_table.md")
    
    print("\n🚀 T{n}五类分类理论系统完整表生成完成！")
    print("🔗 查看文件: T1_T997_five_class_table.md")
    print("📊 新增功能:")
    print("  - ⭐ PRIME-FIB双重基础理论识别")
    print("  - 🔵 纯Fibonacci递归理论分类")  
    print("  - 🟢 纯素数原子理论分类")
    print("  - 🟡 合数组合理论分类")
    print("  - 🔴 唯一公理基础（T1）")
    print("  - 🧮 完整素因子分解")
    print("  - 📈 精确分类统计")