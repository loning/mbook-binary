#!/usr/bin/env python3
"""
T{n} 理论表生成器 - 扩展到T997
基于Fibonacci序列和Zeckendorf分解的完整理论系统
"""

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

def get_theory_type(n, fib_set):
    """确定理论类型"""
    if n == 1:
        return "AXIOM"
    elif n in fib_set:
        return "THEOREM"
    else:
        return "EXTENDED"

def get_theory_name(n, theory_type, fib_index):
    """生成理论名称"""
    theory_names = {
        1: "SelfReferenceAxiom",
        2: "EntropyTheorem", 
        3: "ConstraintTheorem",
        4: "TimeExtended",
        5: "SpaceTheorem",
        6: "QuantumExtended", 
        7: "CodingExtended",
        8: "ComplexityTheorem",
        9: "ObserverExtended",
        10: "PhiComplexExtended",
        11: "ConstraintComplexExtended",
        12: "TripleExtended",
        13: "UnifiedFieldTheorem",
        21: "ConsciousnessTheorem",
        34: "UniverseMindTheorem",
        55: "MetaUniverseTheorem",
        89: "InfiniteRecursionTheorem",
        144: "CosmicHarmonyTheorem",
        233: "TranscendenceTheorem",
        377: "OmegaPointTheorem",
        610: "SingularityTheorem",
        987: "UltimateRealityTheorem"
    }
    
    if n in theory_names:
        return theory_names[n]
    
    # 为其他理论生成名称
    if theory_type == "THEOREM":
        return f"FibonacciTheorem_F{fib_index.get(n, 'X')}"
    elif theory_type == "EXTENDED":
        return f"ExtendedTheory_{n}"
    else:
        return f"Theory_{n}"

def get_tensor_name(n, theory_type):
    """生成张量名称"""
    tensor_names = {
        1: "SelfRefTensor",
        2: "EntropyTensor",
        3: "ConstraintTensor", 
        4: "TimeTensor",
        5: "SpaceTensor",
        6: "QuantumTensor",
        7: "CodingTensor",
        8: "ComplexityTensor",
        9: "ObserverTensor",
        10: "PhiTensor",
        11: "ConstraintComplexTensor",
        12: "TripleTensor",
        13: "UnifiedTensor",
        21: "ConsciousnessTensor",
        34: "UniverseMindTensor",
        55: "MetaUniverseTensor",
        89: "InfiniteRecursionTensor",
        144: "CosmicHarmonyTensor",
        233: "TranscendenceTensor",
        377: "OmegaPointTensor",
        610: "SingularityTensor",
        987: "UltimateRealityTensor"
    }
    
    if n in tensor_names:
        return tensor_names[n]
    
    if theory_type == "THEOREM":
        return f"FibTensor_{n}"
    else:
        return f"ExtTensor_{n}"

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

def generate_theory_description(n, theory_type, theory_name, zeck_components):
    """生成理论描述"""
    descriptions = {
        1: "自指完备公理 - 宇宙的基础假设",
        2: "熵增定理 - 自指系统必然熵增", 
        3: "约束定理 - 熵增与自指产生约束",
        4: "时间扩展定理 - 时间的涌现机制",
        5: "空间定理 - 空间维度的必然性",
        6: "量子扩展定理 - 量子现象的起源",
        7: "编码扩展定理 - 信息编码的优化",
        8: "复杂性定理 - 复杂性的递归涌现",
        13: "统一场定理 - 物理力的统一",
        21: "意识定理 - 意识的必然涌现",
        34: "宇宙心智定理 - 宇宙的自我认知",
        55: "元宇宙定理 - 多层现实的结构",
        89: "无限递归定理 - 无限深度的自指",
        144: "宇宙和谐定理 - 万物的数学和谐",
        233: "超越定理 - 超越有限的机制",
        377: "Ω点定理 - 进化的终极目标",
        610: "奇点定理 - 复杂性奇点",
        987: "终极现实定理 - 现实的最终本质"
    }
    
    if n in descriptions:
        return descriptions[n]
    
    if theory_type == "THEOREM":
        return f"Fibonacci递归定理 - F{n}维度的基础结构"
    elif theory_type == "EXTENDED":
        components_str = "+".join(str(c) for c in sorted(zeck_components))
        return f"扩展定理 - 基于T{components_str}的组合结构"
    else:
        return f"理论{n} - 待定义"

def generate_complete_theory_table(max_n=997):
    """生成完整的T1-T997理论表"""
    
    print("🔬 正在生成T{n}理论系统完整表...")
    print(f"📊 范围: T1 到 T{max_n}")
    
    # 生成Fibonacci序列
    fib_seq, fib_set, fib_index = generate_fibonacci(max_n)
    print(f"🔢 Fibonacci序列生成完成: {len(fib_seq)} 个数")
    
    # 生成理论表
    theories = []
    
    for n in range(1, max_n + 1):
        # Zeckendorf分解
        if n == 1:
            zeck_components = [1]  # F1
        else:
            zeck_components = zeckendorf_decompose(n, fib_seq)
        
        # 确定理论类型
        theory_type = get_theory_type(n, fib_set)
        
        # 生成各种属性
        theory_name = get_theory_name(n, theory_type, fib_index)
        tensor_name = get_tensor_name(n, theory_type)
        from_source = generate_from_source(zeck_components if n > 1 else [])
        zeck_fib_notation = get_zeck_fibonacci_notation(zeck_components, fib_index)
        description = generate_theory_description(n, theory_type, theory_name, zeck_components)
        
        # 构造理论条目
        theory_entry = {
            'n': n,
            'is_fibonacci': n in fib_set,
            'fibonacci_index': fib_index.get(n, None),
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
        
        if n % 100 == 0 or n in fib_set or n <= 20:
            print(f"✅ T{n}: {theory_type} - {theory_name}")
    
    print(f"🎯 理论表生成完成: {len(theories)} 个理论")
    
    return theories, fib_seq, fib_set, fib_index

def export_markdown_table(theories, output_file="complete_theory_table.md"):
    """导出为Markdown表格"""
    
    axiom_count = sum(1 for t in theories if t['theory_type'] == 'AXIOM')
    theorem_count = sum(1 for t in theories if t['theory_type'] == 'THEOREM')  
    extended_count = sum(1 for t in theories if t['theory_type'] == 'EXTENDED')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# T{n} 完整理论系统表 (T1-T997)\n\n")
        f.write("## 📊 统计概览\n\n")
        f.write(f"- **总理论数**: {len(theories)}\n")
        f.write(f"- **AXIOM (公理)**: {axiom_count}\n") 
        f.write(f"- **THEOREM (Fibonacci定理)**: {theorem_count}\n")
        f.write(f"- **EXTENDED (扩展定理)**: {extended_count}\n\n")
        
        f.write("## 🔢 Fibonacci理论位置\n\n")
        fibonacci_theories = [t for t in theories if t['is_fibonacci']]
        for t in fibonacci_theories:
            f.write(f"- **T{t['n']}** = F{t['fibonacci_index']} - {t['theory_name']} ({t['description']})\n")
        
        f.write("\n## 📋 完整理论表\n\n")
        f.write("| T{n} | 值 | F{k} | Zeckendorf分解 | 类型 | FROM来源 | 理论名称 | 张量空间 | 描述 |\n")
        f.write("|------|----|----|----------------|------|----------|----------|----------|------|\n")
        
        for theory in theories:
            n = theory['n']
            is_fib = "✅" if theory['is_fibonacci'] else "❌"
            fib_notation = f"F{theory['fibonacci_index']}" if theory['is_fibonacci'] else "-"
            zeck_notation = theory['zeck_fibonacci_notation']
            theory_type = f"**{theory['theory_type']}**"
            from_source = theory['from_source']
            theory_name = theory['theory_name']
            tensor_name = theory['tensor_name']
            description = theory['description']
            
            f.write(f"| **T{n}** | {n} | {is_fib} {fib_notation} | {zeck_notation} | {theory_type} | {from_source} | {theory_name} | {tensor_name} | {description} |\n")
    
    print(f"📄 Markdown表格已导出到: {output_file}")

def export_summary_stats(theories):
    """导出统计摘要"""
    print("\n" + "="*60)
    print("📊 T{n}理论系统完整统计")
    print("="*60)
    
    axiom_count = sum(1 for t in theories if t['theory_type'] == 'AXIOM')
    theorem_count = sum(1 for t in theories if t['theory_type'] == 'THEOREM')  
    extended_count = sum(1 for t in theories if t['theory_type'] == 'EXTENDED')
    
    print(f"🎯 总理论数: {len(theories)}")
    print(f"🔴 AXIOM (公理): {axiom_count}")
    print(f"🔵 THEOREM (Fibonacci定理): {theorem_count}") 
    print(f"🟡 EXTENDED (扩展定理): {extended_count}")
    
    print(f"\n📈 分布比例:")
    print(f"  - 公理比例: {axiom_count/len(theories)*100:.1f}%")
    print(f"  - 定理比例: {theorem_count/len(theories)*100:.1f}%")
    print(f"  - 扩展比例: {extended_count/len(theories)*100:.1f}%")
    
    print(f"\n🌟 特殊Fibonacci理论:")
    fibonacci_theories = [t for t in theories if t['is_fibonacci']]
    for t in fibonacci_theories:
        print(f"  T{t['n']} = F{t['fibonacci_index']} - {t['theory_name']}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # 生成完整理论表
    theories, fib_seq, fib_set, fib_index = generate_complete_theory_table(997)
    
    # 导出统计
    export_summary_stats(theories)
    
    # 导出Markdown表格
    export_markdown_table(theories, "T1_T997_complete_table.md")
    
    print("\n🚀 T{n}理论系统完整表生成完成！")
    print("🔗 查看文件: T1_T997_complete_table.md")