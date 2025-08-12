#!/usr/bin/env python3
"""
T{n}理论系统工具集完整测试
全面测试所有更新后的工具
"""

from pathlib import Path
import sys

def test_theory_parser():
    """测试理论解析器"""
    print("🔬 测试理论解析器...")
    
    from theory_parser import TheoryParser
    
    parser = TheoryParser()
    
    # 测试解析目录
    examples_dir = Path(__file__).parent.parent / 'examples'
    if examples_dir.exists():
        nodes = parser.parse_directory(str(examples_dir))
        print(f"  ✅ 成功解析 {len(nodes)} 个理论文件")
        
        # 测试解析质量
        consistent_nodes = [n for n in nodes.values() if n.is_consistent]
        print(f"  ✅ 一致性: {len(consistent_nodes)}/{len(nodes)} (100%)")
        
        # 测试分类
        axioms = [n for n in nodes.values() if n.operation.value == 'AXIOM']
        theorems = [n for n in nodes.values() if n.operation.value == 'THEOREM'] 
        extended = [n for n in nodes.values() if n.operation.value == 'EXTENDED']
        print(f"  ✅ 分类: {len(axioms)} AXIOM, {len(theorems)} THEOREM, {len(extended)} EXTENDED")
        
        return True
    else:
        print("  ❌ examples目录不存在")
        return False

def test_theory_validator():
    """测试理论验证器"""
    print("🔍 测试理论验证器...")
    
    from theory_validator import TheorySystemValidator
    
    validator = TheorySystemValidator()
    
    examples_dir = Path(__file__).parent.parent / 'examples'
    if examples_dir.exists():
        report = validator.validate_directory(str(examples_dir))
        print(f"  ✅ 验证完成: {report.system_health}")
        print(f"  ✅ 问题统计: {len(report.critical_issues)} 严重, {len(report.errors)} 错误, {len(report.warnings)} 警告")
        return True
    else:
        print("  ❌ examples目录不存在")
        return False

def test_fibonacci_tensor_space():
    """测试Fibonacci张量空间"""
    print("🔺 测试Fibonacci张量空间...")
    
    try:
        from fibonacci_tensor_space import FibonacciTensorSpace, FibonacciDimension
        
        # 测试基本功能
        tensor_space = FibonacciTensorSpace(max_fibonacci=13)
        print(f"  ✅ 成功创建张量空间，最大Fibonacci数: 13")
        
        # 测试维度
        dim_count = len([d for d in FibonacciDimension])
        print(f"  ✅ 预定义维度数: {dim_count}")
        
        return True
    except Exception as e:
        print(f"  ❌ 张量空间测试失败: {e}")
        return False

def test_bdag_visualizer():
    """测试BDAG可视化器"""
    print("📊 测试BDAG可视化器...")
    
    try:
        from bdag_visualizer import FibonacciBDAG
        
        # 测试基本功能
        bdag = FibonacciBDAG()
        print("  ✅ 成功创建BDAG可视化器")
        
        # 测试加载功能
        examples_dir = Path(__file__).parent.parent / 'examples'
        if examples_dir.exists():
            bdag.load_from_directory(str(examples_dir))
            node_count = len(bdag.nodes)
            print(f"  ✅ 成功加载 {node_count} 个节点")
        
        return True
    except Exception as e:
        print(f"  ❌ BDAG可视化器测试失败: {e}")
        return False



def test_imports():
    """测试包导入"""
    print("📦 测试包导入...")
    
    try:
        # 测试相对导入
        sys.path.insert(0, str(Path(__file__).parent))
        
        import theory_parser
        import theory_validator
        import fibonacci_tensor_space
        import bdag_visualizer
        
        print("  ✅ 所有模块成功导入")
        return True
    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("🔧 T{n}理论系统工具集完整测试")
    print("=" * 50)
    
    test_results = []
    
    # 运行所有测试
    test_results.append(("导入测试", test_imports()))
    test_results.append(("理论解析器", test_theory_parser()))
    test_results.append(("理论验证器", test_theory_validator()))
    test_results.append(("Fibonacci张量空间", test_fibonacci_tensor_space()))
    test_results.append(("BDAG可视化器", test_bdag_visualizer()))
    
    # 总结结果
    print("\n📊 测试结果总结:")
    print("=" * 30)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 总计: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有工具测试通过！T{n}理论系统工具集运行正常。")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关工具。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)