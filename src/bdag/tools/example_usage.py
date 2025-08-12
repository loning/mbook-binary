#!/usr/bin/env python3
"""
T{n}理论工具集使用示例
Example usage of T{n} theory tools
"""

from pathlib import Path
from theory_parser import TheoryParser, FibonacciOperationType
from theory_validator import TheorySystemValidator

def example_parse_theories():
    """示例：解析理论文件"""
    print("🔬 解析理论系统示例")
    print("="*40)
    
    parser = TheoryParser()
    examples_dir = Path(__file__).parent.parent / 'examples'
    
    # 解析目录
    nodes = parser.parse_directory(str(examples_dir))
    
    print(f"发现 {len(nodes)} 个理论:")
    for theory_num, node in sorted(nodes.items()):
        print(f"  T{theory_num}: {node.name}")
        print(f"    类型: {node.operation.value}")
        print(f"    Zeckendorf: {node.zeckendorf_decomp}")
        print(f"    依赖: T{node.theory_dependencies}")
        print(f"    信息量: {node.information_content:.2f} φ-bits")
        print()

def example_validate_system():
    """示例：验证理论系统"""
    print("🔍 验证理论系统示例")
    print("="*40)
    
    validator = TheorySystemValidator()
    examples_dir = Path(__file__).parent.parent / 'examples'
    
    # 验证系统
    report = validator.validate_directory(str(examples_dir))
    
    print(f"系统状态: {report.system_health}")
    print(f"理论总数: {report.total_theories}")
    print(f"有效理论: {report.valid_theories}")
    print(f"问题统计:")
    print(f"  严重: {len(report.critical_issues)}")
    print(f"  错误: {len(report.errors)}")
    print(f"  警告: {len(report.warnings)}")

def example_theory_analysis():
    """示例：分析特定理论"""
    print("📊 理论分析示例")
    print("="*40)
    
    parser = TheoryParser()
    
    # 分析T8复杂性定理
    test_filename = "T8__ComplexityTheorem__THEOREM__ZECK_F5__FROM__T7+T6__TO__ComplexTensor.md"
    node = parser.parse_filename(test_filename)
    
    if node:
        print(f"理论: T{node.theory_number} - {node.name}")
        print(f"类型描述: {node.theory_type_description}")
        print(f"是否Fibonacci理论: {node.is_fibonacci_theory}")
        print(f"复杂度等级: {node.complexity_level}")
        print(f"信息含量: {node.information_content:.3f} φ-bits")
        print(f"一致性: {'✅' if node.is_consistent else '❌'}")
        
        # Zeckendorf分析
        expected_zeck = parser.to_zeckendorf(node.theory_number)
        print(f"Zeckendorf分解:")
        print(f"  声明: {node.zeckendorf_decomp}")
        print(f"  期望: {expected_zeck}")
        print(f"  匹配: {'✅' if set(node.zeckendorf_decomp) == set(expected_zeck) else '❌'}")

def example_fibonacci_sequence():
    """示例：Fibonacci序列和Zeckendorf分解"""
    print("🔢 Fibonacci序列示例")  
    print("="*40)
    
    parser = TheoryParser()
    
    print("Fibonacci序列:")
    for i, fib in enumerate(parser.fibonacci_sequence[:10], 1):
        print(f"  F{i} = {fib}")
    
    print("\n自然数的Zeckendorf分解:")
    for n in range(1, 16):
        zeck = parser.to_zeckendorf(n)
        fib_indices = []
        for fib_val in zeck:
            try:
                idx = parser.fibonacci_sequence.index(fib_val) + 1
                fib_indices.append(f"F{idx}")
            except ValueError:
                fib_indices.append(f"?{fib_val}")
        print(f"  {n:2d} = {zeck} = {'+'.join(fib_indices)}")

def main():
    """运行所有示例"""
    examples = [
        example_fibonacci_sequence,
        example_parse_theories,
        example_theory_analysis, 
        example_validate_system
    ]
    
    for example in examples:
        try:
            example()
            print("\n" + "="*60 + "\n")
        except Exception as e:
            print(f"示例 {example.__name__} 执行失败: {e}")
            print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()