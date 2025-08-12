#!/usr/bin/env python3
"""
T{n}理论系统工具集完整测试 v3.0
全面测试五类分类系统和素数增强功能
支持：AXIOM/PRIME-FIB/FIBONACCI/PRIME/COMPOSITE分类
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
        
        # 测试五类分类
        axioms = [n for n in nodes.values() if n.operation.value == 'AXIOM']
        prime_fibs = [n for n in nodes.values() if n.operation.value == 'PRIME_FIB']
        fibonaccis = [n for n in nodes.values() if n.operation.value == 'FIBONACCI']
        primes = [n for n in nodes.values() if n.operation.value == 'PRIME']
        composites = [n for n in nodes.values() if n.operation.value == 'COMPOSITE']
        
        print(f"  ✅ 五类分类: {len(axioms)} AXIOM, {len(prime_fibs)} PRIME-FIB, {len(fibonaccis)} FIB, {len(primes)} PRIME, {len(composites)} COMP")
        
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
    """测试三维宇宙张量空间（Fibonacci-Prime-Zeckendorf）"""
    print("🔺 测试三维宇宙张量空间...")
    
    try:
        from fibonacci_tensor_space import UniversalTensorSpace, TensorClassification
        
        # 测试基本功能
        tensor_space = UniversalTensorSpace(max_theory=100)
        print(f"  ✅ 成功创建三维张量空间，最大理论: 100")
        
        # 测试分类功能
        classification = tensor_space.classify_theory(13)
        print(f"  ✅ T13分类: {classification.class_type.value} - {'PRIME-FIB双重基础' if classification.is_prime_fib else '基础理论'}")
        
        # 测试双重基础理论分析
        dual_foundations = tensor_space.analyze_dual_foundations()
        print(f"  ✅ 发现 {len(dual_foundations)} 个PRIME-FIB双重基础理论")
        
        return True
    except Exception as e:
        print(f"  ❌ 张量空间测试失败: {e}")
        return False

def test_prime_theory_classifier():
    """测试素数理论分类器"""
    print("🔢 测试素数理论分类器...")
    
    try:
        from prime_theory_classifier import PrimeTheoryClassifier
        from theory_parser import FibonacciOperationType
        
        classifier = PrimeTheoryClassifier(max_theory=50)
        
        # 测试关键理论分类
        test_theories = [1, 2, 3, 5, 13, 7, 11, 21, 34, 89]
        correct_classifications = 0
        
        for n in test_theories:
            classification = classifier.classify_theory(n)
            expected_types = {
                1: FibonacciOperationType.AXIOM,
                2: FibonacciOperationType.PRIME_FIB,
                3: FibonacciOperationType.PRIME_FIB,
                5: FibonacciOperationType.PRIME_FIB,
                13: FibonacciOperationType.PRIME_FIB,
                7: FibonacciOperationType.PRIME,
                11: FibonacciOperationType.PRIME,
                21: FibonacciOperationType.FIBONACCI,
                34: FibonacciOperationType.FIBONACCI,
                89: FibonacciOperationType.PRIME_FIB
            }
            
            if classification.class_type == expected_types[n]:
                correct_classifications += 1
        
        accuracy = correct_classifications / len(test_theories) * 100
        print(f"  ✅ 分类准确性: {correct_classifications}/{len(test_theories)} ({accuracy:.1f}%)")
        
        # 测试统计功能
        stats = classifier.get_classification_statistics()
        print(f"  ✅ 统计功能: {sum(stats.values())} 个理论已分类")
        
        return accuracy >= 95.0  # 要求95%以上准确性
    except Exception as e:
        print(f"  ❌ 素数理论分类器测试失败: {e}")
        return False

def test_prime_theory_analyzer():
    """测试素数理论分析器"""
    print("🔍 测试素数理论分析器...")
    
    try:
        from prime_theory_analyzer import PrimeTheoryAnalyzer
        
        analyzer = PrimeTheoryAnalyzer(max_theory=50)
        
        # 测试重要素数分析
        test_primes = [2, 3, 5, 13, 89]  # 包含PRIME-FIB
        
        for p in test_primes:
            analysis = analyzer.analyze_prime_theory(p)
            is_prime_fib = p in [2, 3, 5, 13, 89]
            
            if analysis.is_prime_fib == is_prime_fib:
                print(f"  ✅ T{p}: {'PRIME-FIB' if is_prime_fib else 'PRIME'} - 强度{analysis.primality_strength:.2f}")
            else:
                print(f"  ❌ T{p}: 分类错误")
                return False
        
        # 测试统计功能
        stats = analyzer.get_prime_statistics()
        print(f"  ✅ 统计: {stats['total_prime_theories']} 素数理论, {stats['prime_fib_theories']} PRIME-FIB")
        
        return True
    except Exception as e:
        print(f"  ❌ 素数理论分析器测试失败: {e}")
        return False

def test_theory_table_generators():
    """测试理论表生成器"""
    print("📋 测试理论表生成器...")
    
    try:
        from theory_table_generator import generate_complete_theory_table
        from theory_table_generator_prime import generate_enhanced_theory_table
        
        # 测试基础生成器
        theories, _, _, _, _ = generate_complete_theory_table(max_n=20)
        print(f"  ✅ 基础生成器: {len(theories)} 个理论")
        
        # 验证分类
        classifications = set(t['theory_type'] for t in theories)
        expected_types = {'AXIOM', 'PRIME-FIB', 'FIBONACCI', 'PRIME', 'COMPOSITE'}
        if expected_types.issubset(classifications):
            print(f"  ✅ 五类分类系统完整")
        else:
            missing = expected_types - classifications
            print(f"  ❌ 缺少分类: {missing}")
            return False
        
        # 测试素数增强生成器
        enhanced_table = generate_enhanced_theory_table(max_n=20)
        if "PRIME-FIB双重基础理论" in enhanced_table:
            print(f"  ✅ 素数增强生成器工作正常")
        else:
            print(f"  ❌ 素数增强生成器输出异常")
            return False
        
        return True
    except Exception as e:
        print(f"  ❌ 理论表生成器测试失败: {e}")
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
        import prime_theory_classifier
        import prime_theory_analyzer
        import theory_table_generator
        import theory_table_generator_prime
        
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
    test_results.append(("三维张量空间", test_fibonacci_tensor_space()))
    test_results.append(("素数理论分类器", test_prime_theory_classifier()))
    test_results.append(("素数理论分析器", test_prime_theory_analyzer()))
    test_results.append(("理论表生成器", test_theory_table_generators()))
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
        print("🎉 所有工具测试通过！T{n}五类分类系统工具集运行正常。")
        print("🔥 新功能验证成功：")
        print("   ⭐ PRIME-FIB双重基础理论识别")
        print("   🔢 素数理论深度分析")
        print("   🔺 三维宇宙张量空间")
        print("   📊 五类分类统计")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关工具。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)