#!/usr/bin/env python3
"""
BDAG Tools Test Script
Test the BDAG parsing, validation, and visualization functionality
"""

import sys
import os
from pathlib import Path

# Add the tools directory to the path
sys.path.insert(0, str(Path(__file__).parent))

import bdag_core
import bdag_validator
import bdag_visualizer

BDAGParser = bdag_core.BDAGParser
BDAGValidator = bdag_validator.BDAGValidator
BDAGVisualizer = bdag_visualizer.BDAGVisualizer

def test_parsing():
    """Test BDAG parsing functionality"""
    print("=== 测试BDAG解析功能 ===")
    
    # 测试有效文件名
    valid_filenames = [
        "A001__SelfReference__DEFINE__FROM__Axiom__TO__SelfRefTensor__ATTR__Recursive_Entropic.md",
        "A002__Phi__DEFINE__FROM__Constant__TO__GoldenRatio__ATTR__Irrational_Algebraic.md",
        "B101__EntropyIncrease__APPLY__FROM__A001_SelfReference__TO__EntropyTensor__ATTR__Monotonic_Irreversible.md",
        "C201__InformationEntropy__COMBINE__FROM__B101_EntropyIncrease__B103_PhiEncoding__TO__InfoTensor__ATTR__Quantized_Compressed.md"
    ]
    
    parser = BDAGParser()
    nodes = {}
    
    for filename in valid_filenames:
        print(f"\n解析文件: {filename}")
        node = parser.parse_filename(filename)
        if node:
            print(f"  ✓ 成功解析")
            print(f"    层级: {node.layer.value}")
            print(f"    序号: {node.sequence:03d}")
            print(f"    节点名: {node.name}")
            print(f"    操作: {node.operation.value}")
            print(f"    输入: {[f'{inp.id}_{inp.name}' if inp.id else inp.name for inp in node.inputs]}")
            print(f"    输出: {node.output_type}")
            print(f"    属性: {node.attributes}")
            # Add the node to our collection
            nodes[node.full_id] = node
        else:
            print(f"  ✗ 解析失败")
    
    # 测试错误
    if parser.get_errors():
        print(f"\n解析错误:")
        for error in parser.get_errors():
            print(f"  {error}")
    
    # Compute terminal nodes for the collected nodes
    referenced_ids = set()
    for node in nodes.values():
        for inp in node.inputs:
            if inp.id:  # 排除 Axiom 等基础输入
                referenced_ids.add(inp.id)
    
    for node in nodes.values():
        node.is_terminal = node.full_id not in referenced_ids
    
    return nodes

def test_validation(nodes):
    """Test BDAG validation functionality"""
    print("\n=== 测试BDAG验证功能 ===")
    
    if not nodes:
        print("没有有效节点可供验证")
        return
    
    validator = BDAGValidator(nodes)
    errors = validator.validate_all()
    
    if errors:
        print(f"发现 {len(errors)} 个验证错误:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
    else:
        print("✓ 所有验证检查通过")

def test_visualization(nodes):
    """Test BDAG visualization functionality"""
    print("\n=== 测试BDAG可视化功能 ===")
    
    if not nodes:
        print("没有有效节点可供可视化")
        return
    
    visualizer = BDAGVisualizer(nodes)
    
    # 生成统计信息
    stats = visualizer.generate_statistics()
    print("\n统计信息:")
    print(f"  总节点数: {stats['total_nodes']}")
    print(f"  总边数: {stats['total_edges']}")
    print(f"  源节点: {stats['source_nodes']}")
    print(f"  终端节点: {stats['terminal_nodes']}")
    print(f"  层级分布: {stats['by_layer']}")
    print(f"  操作分布: {stats['by_operation']}")
    
    # 生成关键路径
    critical_path = visualizer.get_critical_path()
    print(f"  关键路径: {' → '.join(critical_path)}")
    
    # 生成Mermaid图表
    print("\n生成Mermaid图表:")
    mermaid = visualizer.generate_mermaid()
    print("```mermaid")
    print(mermaid)
    print("```")
    
    # 生成层级分析
    layer_analysis = visualizer.generate_layer_analysis()
    print("\n层级分析:")
    for layer, data in layer_analysis['layer_distribution'].items():
        if data['count'] > 0:
            print(f"  {layer}层: {data['count']}个节点, 平均输入{data['avg_inputs']:.1f}, 操作{data['operations']}")

def test_invalid_filenames():
    """Test parsing of invalid filenames"""
    print("\n=== 测试无效文件名解析 ===")
    
    invalid_filenames = [
        "T001_Invalid_Format.md",  # 错误格式
        "A001__Test__INVALID__FROM__Axiom__TO__Test__ATTR__Test.md",  # 无效操作
        "A001-Wrong-Separator.md",  # 错误分隔符
        "Z001__Test__DEFINE__FROM__Axiom__TO__Test__ATTR__Test.md",  # 无效层级
    ]
    
    parser = BDAGParser()
    
    for filename in invalid_filenames:
        print(f"\n测试无效文件名: {filename}")
        node = parser.parse_filename(filename)
        if node:
            print(f"  ✗ 意外成功解析（应该失败）")
        else:
            print(f"  ✓ 正确拒绝解析")
    
    errors = parser.get_errors()
    if errors:
        print(f"\n捕获的错误:")
        for error in errors:
            print(f"  {error}")

def main():
    """Main test function"""
    print("BDAG工具测试")
    print("=" * 50)
    
    # 解析测试
    nodes = test_parsing()
    
    # 验证测试
    test_validation(nodes)
    
    # 可视化测试
    test_visualization(nodes)
    
    # 无效文件名测试
    test_invalid_filenames()
    
    print("\n" + "=" * 50)
    print("测试完成")

if __name__ == '__main__':
    main()