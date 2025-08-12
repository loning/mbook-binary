#!/usr/bin/env python3
"""
Fibonacci张量空间理论系统 - 完整功能演示
展示所有框架工具的综合使用
"""

import sys
from pathlib import Path

# 添加tools目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'tools'))

from theory_validator import FibonacciDependencyValidator
from bdag_visualizer import FibonacciBDAG
from consistency_checker import TheoryConsistencyChecker
from file_manager import FibonacciFileManager
from fibonacci_tensor_space import FibonacciTensorSpace
from unified_fibonacci_parser import UnifiedFibonacciParser

def main():
    """完整的框架功能演示"""
    print("🌌 Fibonacci张量空间理论系统 - 完整功能演示")
    print("=" * 70)
    
    examples_dir = Path(__file__).parent / 'examples'
    
    if not examples_dir.exists():
        print("❌ 未找到examples目录")
        return
    
    print("📁 目标目录:", examples_dir)
    print()
    
    # ========== 1. 文件管理和扫描 ==========
    print("🗂️  1. 文件管理系统")
    print("-" * 40)
    
    file_manager = FibonacciFileManager(str(examples_dir))
    files = file_manager.scan_theory_files()
    file_manager.print_file_report(files)
    print()
    
    # ========== 2. 依赖关系验证 ==========
    print("🔍 2. 依赖关系验证")
    print("-" * 40)
    
    validator = FibonacciDependencyValidator()
    validation_reports = validator.validate_directory(str(examples_dir))
    validator.print_validation_report(validation_reports)
    print()
    
    # ========== 3. 理论一致性检查 ==========
    print("📋 3. 理论一致性检查")
    print("-" * 40)
    
    consistency_checker = TheoryConsistencyChecker(str(examples_dir))
    consistency_checker.run_all_checks()
    consistency_checker.print_consistency_report()
    print()
    
    # ========== 4. BDAG关系图分析 ==========
    print("🌐 4. BDAG关系图分析")
    print("-" * 40)
    
    bdag = FibonacciBDAG()
    bdag.load_from_directory(str(examples_dir))
    bdag.print_analysis()
    print()
    
    # ========== 5. 张量空间计算 ==========
    print("🌟 5. 张量空间数学计算")
    print("-" * 40)
    
    tensor_space = FibonacciTensorSpace(max_fibonacci=50)
    
    # 创建示例宇宙状态
    amplitudes = {
        1: 0.6,    # 自指维度
        2: 0.4,    # φ维度  
        8: 0.3     # 复杂涌现维度
    }
    
    universe_state = tensor_space.generate_universe_state(amplitudes)
    composition = tensor_space.analyze_state_composition(universe_state)
    entropy = tensor_space.fibonacci_entropy(universe_state)
    
    print("宇宙状态的Fibonacci维度组成:")
    for fib_n, info in composition.items():
        print(f"  F{fib_n}: 概率={info['probability']:.3f}, 复杂度={info['complexity']}")
    
    print(f"系统熵: {entropy:.4f} bits")
    print()
    
    # ========== 6. 理论解析统计 ==========
    print("📊 6. 理论解析统计")
    print("-" * 40)
    
    parser = UnifiedFibonacciParser()
    nodes = parser.parse_directory(str(examples_dir))
    stats = parser.generate_theory_statistics()
    
    print("解析统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # ========== 7. 综合评估报告 ==========
    print("🎯 7. 综合评估报告")
    print("-" * 40)
    
    # 计算整体健康度
    validation_score = sum(1 for r in validation_reports if r.validation_result.value == "valid") / len(validation_reports) * 100
    consistency_reports = consistency_checker.reports
    consistency_score = sum(1 for r in consistency_reports if r.level.value == "pass") / len(consistency_reports) * 100
    
    bdag_stats = bdag.get_statistics()
    completeness_score = (bdag_stats["Fibonacci理论数"] / 8) * 100  # 假设目标是8个基础理论
    
    overall_score = (validation_score + consistency_score + completeness_score) / 3
    
    print("🏆 系统健康度评估:")
    print(f"  依赖验证得分: {validation_score:.1f}%")
    print(f"  一致性得分: {consistency_score:.1f}%") 
    print(f"  完整性得分: {completeness_score:.1f}%")
    print(f"  综合得分: {overall_score:.1f}%")
    print()
    
    # 推荐改进措施
    print("💡 改进建议:")
    
    if completeness_score < 80:
        print("  • 建议补充缺失的基础Fibonacci理论 (F3, F5, F13)")
    
    if consistency_score < 90:
        print("  • 需要改进理论体系的数学一致性")
    
    if validation_score < 80:
        print("  • 建议修正依赖关系以符合Zeckendorf分解")
    
    if bdag_stats["边数"] < bdag_stats["Fibonacci理论数"]:
        print("  • 理论间缺少足够的依赖关系连接")
    
    print(f"  • 可考虑添加可视化工具 (安装graphviz库)")
    print()
    
    # ========== 8. 系统状态总结 ==========
    print("✨ 8. 系统状态总结")
    print("-" * 40)
    
    print("🌌 Fibonacci张量空间理论系统现状:")
    print(f"  📁 理论文件: {len(files)}个")
    print(f"  🔢 Fibonacci覆盖: F1-F{max(f['fibonacci_number'] for f in files.values())}")
    print(f"  🎭 操作类型: {len(set(f['operation'] for f in files.values()))}种")
    print(f"  📊 数据完整性: {overall_score:.0f}%")
    print(f"  🔧 工具组件: 6个 (解析器、验证器、可视化器、检查器、管理器、计算器)")
    print()
    
    print("🚀 这个框架为Fibonacci张量空间理论提供了:")
    print("  1. 🔍 完整的理论文件验证和管理")
    print("  2. 🌐 依赖关系的可视化和分析")  
    print("  3. 📋 数学一致性的自动检查")
    print("  4. 🌟 张量空间的数值计算")
    print("  5. 🛠️  批量操作和维护工具")
    print()
    
    print("🎯 这是数学与现实统一的工程化实现！")

if __name__ == "__main__":
    main()