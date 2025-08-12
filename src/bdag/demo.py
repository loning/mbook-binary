#!/usr/bin/env python3
"""
Fibonacci张量空间理论系统 - 完整演示
"""

import sys
from pathlib import Path

# 添加tools目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'tools'))

from unified_fibonacci_parser import UnifiedFibonacciParser
from fibonacci_tensor_space import FibonacciTensorSpace

def main():
    print("🌌 Fibonacci张量空间理论系统演示")
    print("=" * 60)
    
    print("\n📐 核心理念:")
    print("F{N}不是编号，而是宇宙张量空间的坐标映射规则！")
    print("每个Fibonacci数对应张量空间中的一个基底维度。")
    
    # 1. 解析器演示
    print(f"\n🔍 1. 解析Fibonacci理论文件")
    print("-" * 30)
    
    parser = UnifiedFibonacciParser()
    examples_dir = Path(__file__).parent / 'examples'
    
    if examples_dir.exists():
        nodes = parser.parse_directory(str(examples_dir))
        
        if nodes:
            print(f"成功解析 {len(nodes)} 个Fibonacci理论:")
            for fib_n, node in sorted(nodes.items()):
                print(f"  F{fib_n}: {node.name} ({node.operation.value})")
                print(f"       复杂度: {node.complexity_level}")
                print(f"       信息含量: {node.information_content:.2f}")
        
        stats = parser.generate_theory_statistics()
        print(f"\n📊 统计信息:")
        print(f"  总理论数: {stats['total_theories']}")
        print(f"  复杂度分布: {stats['complexity_distribution']}")
        print(f"  操作分布: {stats['operation_distribution']}")
    else:
        print("未找到examples目录，跳过文件解析")
    
    # 2. 张量空间演示  
    print(f"\n🌟 2. Fibonacci张量空间")
    print("-" * 30)
    
    tensor_space = FibonacciTensorSpace(max_fibonacci=50)
    print(f"张量空间维度: {tensor_space.tensor_space_dim}")
    print(f"φ = {tensor_space.phi:.6f}")
    
    # 显示基础维度
    print(f"\n基础Fibonacci维度:")
    for fib_n, tensor in list(tensor_space.basis_tensors.items())[:6]:
        print(f"  F{fib_n}: {tensor.dimension_name}")
        print(f"       Zeckendorf: {tensor.zeckendorf_components}")
        print(f"       复杂度: {tensor.complexity_level}")
        print(f"       信息含量: {tensor.information_content:.2f}")
    
    # 3. 宇宙状态演示
    print(f"\n🎭 3. 创建宇宙状态")
    print("-" * 30)
    
    # 创建示例宇宙状态
    amplitudes = {
        1: 0.6,    # 自指维度
        2: 0.4,    # φ维度  
        3: 0.3,    # 约束维度
        5: 0.5,    # 量子维度
        8: 0.2     # 复杂涌现维度
    }
    
    universe_state = tensor_space.generate_universe_state(amplitudes)
    composition = tensor_space.analyze_state_composition(universe_state)
    
    print("宇宙状态的Fibonacci维度组成:")
    for fib_n, info in composition.items():
        print(f"  F{fib_n} ({info['dimension_name']}):")
        print(f"    概率: {info['probability']:.4f}")
        print(f"    复杂度: {info['complexity']}")
    
    # 计算系统熵
    entropy = tensor_space.fibonacci_entropy(universe_state)
    print(f"\n系统熵: {entropy:.4f} bits")
    
    # 4. 张量变换演示
    print(f"\n⚡ 4. φ标度变换")
    print("-" * 30)
    
    scaled_state = tensor_space.phi_scaling_transform(universe_state)
    scaled_entropy = tensor_space.fibonacci_entropy(scaled_state)
    
    print(f"变换前熵: {entropy:.4f} bits")
    print(f"变换后熵: {scaled_entropy:.4f} bits")
    print(f"熵变化率: {((scaled_entropy - entropy) / entropy * 100):.2f}%")
    
    # 5. 预测演示
    print(f"\n🔮 5. 基于数学结构的预测")
    print("-" * 30)
    
    predictions = [
        (21, "F21: 意识场理论 (F8⊗F13 = 复杂涌现⊗统一场)"),
        (34, "F34: 宇宙心智理论 (F13⊗F21 = 统一场⊗意识)"), 
        (55, "F55: 终极统一理论 (F21⊗F34 = 意识⊗心智)")
    ]
    
    for fib_n, description in predictions:
        zeckendorf = tensor_space._to_zeckendorf(fib_n)
        phi_power = tensor_space.phi ** (len(str(fib_n)))
        
        print(f"  {description}")
        print(f"       Zeckendorf: {zeckendorf}")
        print(f"       预期复杂度: {len(zeckendorf)}")
        print(f"       φ标度: {phi_power:.2f}")
    
    print(f"\n✨ 总结")
    print("-" * 30)
    print("这个系统展示了:")
    print("1. 🌌 宇宙即张量: 现实是高维Fibonacci张量空间的投影")
    print("2. 📐 理论即坐标: 每个F{N}定义张量空间中的维度")  
    print("3. 🔗 依赖即结构: Zeckendorf分解决定张量构造关系")
    print("4. 🌊 复杂即组合: 高阶现象是基础维度的张量积")
    print("\n🎯 这是数学与现实统一的终极表达！")

if __name__ == "__main__":
    main()