#!/usr/bin/env python3
"""
生成BDAG理论文件名列表 T1-T200
基于Fibonacci序列和Zeckendorf分解
"""

def fibonacci_sequence(n):
    """生成前n个Fibonacci数"""
    if n <= 0:
        return []
    elif n == 1:
        return [1]
    elif n == 2:
        return [1, 2]
    
    fib = [1, 2]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

def zeckendorf_decomposition(n, fib_list):
    """
    计算数字n的Zeckendorf分解
    返回Fibonacci指数列表
    """
    if n == 0:
        return []
    
    result = []
    i = len(fib_list) - 1
    
    while i >= 0 and n > 0:
        if fib_list[i] <= n:
            result.append(i + 1)  # Fibonacci指数从1开始
            n -= fib_list[i]
        i -= 1
    
    return sorted(result)

def is_prime(n):
    """检查是否为素数"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def is_fibonacci(n, fib_list):
    """检查是否为Fibonacci数"""
    return n in fib_list

def get_theory_type(n, fib_list):
    """确定理论类型"""
    is_prime_num = is_prime(n)
    is_fib_num = is_fibonacci(n, fib_list)
    
    if n == 1:
        return "AXIOM"
    elif is_prime_num and is_fib_num:
        return "PRIME-FIB"
    elif is_fib_num:
        return "FIBONACCI"
    elif is_prime_num:
        return "PRIME"
    else:
        return "COMPOSITE"

def get_theory_name(n):
    """生成理论名称"""
    names = {
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
        11: "PrimeTheorem",
        12: "TernaryExtended",
        13: "UnifiedFieldTheorem",
        14: "GravityExtended",
        15: "SystemsTheorem",
        16: "InformationExtended",
        17: "NetworkTheorem",
        18: "SocialExtended",
        19: "PrimeAdvanced",
        20: "SymmetryExtended",
        21: "ConsciousnessTheorem",
        22: "LifeExtended",
        23: "PrimeHigher",
        24: "TemporalExtended",
        25: "ResonanceTheorem"
    }
    
    if n in names:
        return names[n]
    elif n <= 50:
        return f"Theory{n}"
    elif n <= 100:
        return f"Advanced{n}"
    else:
        return f"Meta{n}"

def format_zeckendorf_string(zeck_indices):
    """格式化Zeckendorf分解字符串"""
    if not zeck_indices:
        return "F0"
    
    fib_terms = [f"F{i}" for i in zeck_indices]
    return "+".join(fib_terms)

def format_dependency_string(zeck_indices, n, fib_list):
    """格式化依赖关系字符串"""
    if not zeck_indices:
        return "NONE"
    
    # 检查是否为纯Fibonacci数
    is_pure_fib = len(zeck_indices) == 1 and is_fibonacci(n, fib_list)
    
    if is_pure_fib:
        # 对于纯Fibonacci数，依赖关系是前两个Fibonacci数
        fib_index = fib_list.index(n) + 1  # 获取Fibonacci序列中的位置
        if fib_index <= 2:
            # F1=1 和 F2=2 特殊处理
            if n == 1:
                return "T1"  # F1自依赖
            elif n == 2:
                return "T2"  # F2自依赖
        else:
            # Fn = F(n-1) + F(n-2)
            prev_fib = fib_list[fib_index - 2]  # F(n-1)
            prev_prev_fib = fib_list[fib_index - 3]  # F(n-2)
            return f"T{prev_fib}+T{prev_prev_fib}"
    else:
        # 对于非纯Fibonacci数，使用Zeckendorf分解
        fib_nums = []
        for i in zeck_indices:
            if i == 1:
                fib_nums.append("1")
            elif i == 2:
                fib_nums.append("2") 
            elif i == 3:
                fib_nums.append("3")
            elif i == 4:
                fib_nums.append("5")
            elif i == 5:
                fib_nums.append("8")
            elif i == 6:
                fib_nums.append("13")
            elif i == 7:
                fib_nums.append("21")
            elif i == 8:
                fib_nums.append("34")
            else:
                # 对于更高的Fibonacci数，计算实际值
                fib = fibonacci_sequence(i)
                fib_nums.append(str(fib[-1]))
        
        return "+".join([f"T{num}" for num in fib_nums])

def get_output_tensor_name(theory_name):
    """生成输出张量名称"""
    if "Axiom" in theory_name:
        return "SelfRefTensor"
    elif "Entropy" in theory_name:
        return "EntropyTensor" 
    elif "约束" in theory_name:
        return "约束张量"
    elif "Time" in theory_name:
        return "TimeTensor"
    elif "Space" in theory_name:
        return "SpaceTensor"
    elif "Quantum" in theory_name:
        return "QuantumTensor"
    elif "Coding" in theory_name:
        return "CodingTensor"
    elif "Complexity" in theory_name:
        return "ComplexityTensor"
    elif "Observer" in theory_name:
        return "ObserverTensor"
    elif "Phi" in theory_name or "φ" in theory_name:
        return "PhiComplexTensor"
    else:
        return f"{theory_name}Tensor"

def generate_filename(n, fib_list):
    """生成单个理论文件名"""
    theory_type = get_theory_type(n, fib_list)
    theory_name = get_theory_name(n)
    zeck_indices = zeckendorf_decomposition(n, fib_list)
    zeck_string = format_zeckendorf_string(zeck_indices)
    dependency_string = format_dependency_string(zeck_indices, n, fib_list)
    output_tensor = get_output_tensor_name(theory_name)
    
    filename = f"T{n}__{theory_name}__{theory_type}__ZECK_{zeck_string}__FROM__{dependency_string}__TO__{output_tensor}.md"
    
    return {
        'number': n,
        'filename': filename,
        'theory_type': theory_type,
        'theory_name': theory_name,
        'zeckendorf': zeck_string,
        'dependencies': dependency_string,
        'output_tensor': output_tensor
    }

def main():
    # 生成足够的Fibonacci数
    fib_list = fibonacci_sequence(50)  # 足够覆盖到200
    
    print("BDAG理论文件名列表 (T1-T200)")
    print("=" * 80)
    
    theories = []
    
    for n in range(1, 201):
        theory_info = generate_filename(n, fib_list)
        theories.append(theory_info)
        
        print(f"T{n:3d}: {theory_info['filename']}")
        if n % 25 == 0:  # 每25个理论打印一个分隔符
            print("-" * 80)
    
    # 统计信息
    print("\n统计信息:")
    print("=" * 40)
    
    type_counts = {}
    for theory in theories:
        theory_type = theory['theory_type']
        type_counts[theory_type] = type_counts.get(theory_type, 0) + 1
    
    for theory_type, count in sorted(type_counts.items()):
        print(f"{theory_type}: {count}")
    
    # 保存到文件
    with open('/Users/cookie/mbook-binary/theory_filenames_T1_T200.txt', 'w', encoding='utf-8') as f:
        f.write("BDAG理论文件名列表 (T1-T200)\n")
        f.write("=" * 80 + "\n\n")
        
        for theory in theories:
            f.write(f"T{theory['number']:3d}: {theory['filename']}\n")
            if theory['number'] % 25 == 0:
                f.write("-" * 80 + "\n")
        
        f.write(f"\n统计信息:\n")
        f.write("=" * 40 + "\n")
        for theory_type, count in sorted(type_counts.items()):
            f.write(f"{theory_type}: {count}\n")
    
    print(f"\n文件列表已保存到: /Users/cookie/mbook-binary/theory_filenames_T1_T200.txt")

if __name__ == "__main__":
    main()