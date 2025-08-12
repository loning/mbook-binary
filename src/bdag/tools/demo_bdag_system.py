#!/usr/bin/env python3
"""
BDAG System Demonstration
Comprehensive demo of the Binary Universe DAG system capabilities
"""

import sys
import os
from pathlib import Path

# Add the tools directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from bdag_core import BDAGParser
from bdag_validator import BDAGValidator
from bdag_visualizer import BDAGVisualizer

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_subheader(title):
    """Print a formatted subheader"""
    print(f"\n--- {title} ---")

def demo_parsing():
    """Demonstrate BDAG parsing capabilities"""
    print_header("BDAG解析系统演示")
    
    examples_dir = Path(__file__).parent.parent / "examples"
    
    print(f"解析目录: {examples_dir}")
    
    parser = BDAGParser()
    nodes = parser.parse_directory(str(examples_dir))
    
    print(f"\n成功解析 {len(nodes)} 个节点:")
    
    for node_id, node in sorted(nodes.items()):
        print(f"\n  {node_id}: {node.name}")
        print(f"    层级: {node.layer.value}层")
        print(f"    操作: {node.operation.value}")
        print(f"    输入: {[inp.name if not inp.id else f'{inp.id}_{inp.name}' for inp in node.inputs]}")
        print(f"    输出: {node.output_type}")
        print(f"    属性: {', '.join(node.attributes)}")
        print(f"    源节点: {'是' if node.is_source else '否'}")
        print(f"    终端节点: {'是' if node.is_terminal else '否'}")
    
    return nodes

def demo_validation(nodes):
    """Demonstrate BDAG validation capabilities"""
    print_header("BDAG验证系统演示")
    
    validator = BDAGValidator(nodes)
    errors = validator.validate_all()
    
    if errors:
        print(f"⚠️  发现 {len(errors)} 个验证问题:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
    else:
        print("✅ 所有验证检查通过!")
        
        print_subheader("验证检查项目")
        print("  ✓ 层级约束检查")
        print("  ✓ 依赖层级合理性")
        print("  ✓ 操作类型适配性")
        print("  ✓ 序号唯一性")
        print("  ✓ 无环性检查")
        print("  ✓ 输入节点存在性")

def demo_statistics(nodes):
    """Demonstrate BDAG statistics and analysis"""
    print_header("BDAG统计分析演示")
    
    visualizer = BDAGVisualizer(nodes)
    stats = visualizer.generate_statistics()
    
    print_subheader("基本统计信息")
    print(f"  总节点数: {stats['total_nodes']}")
    print(f"  总边数: {stats['total_edges']}")
    print(f"  源节点数: {len(stats['source_nodes'])}")
    print(f"  终端节点数: {len(stats['terminal_nodes'])}")
    
    print_subheader("层级分布")
    layer_names = {'A': '公理层', 'B': '基础操作层', 'C': '复合操作层', 'E': '涌现层', 'U': '统一层'}
    for layer, count in sorted(stats['by_layer'].items()):
        layer_name = layer_names.get(layer, f"{layer}层")
        print(f"  {layer_name}: {count} 个节点")
    
    print_subheader("操作类型分布")
    op_names = {
        'DEFINE': '定义操作',
        'APPLY': '应用操作', 
        'TRANSFORM': '变换操作',
        'COMBINE': '组合操作',
        'EMERGE': '涌现操作',
        'DERIVE': '推导操作'
    }
    for op, count in sorted(stats['by_operation'].items()):
        op_name = op_names.get(op, op)
        print(f"  {op_name}: {count} 个节点")
    
    print_subheader("关键路径分析")
    critical_path = visualizer.get_critical_path()
    if critical_path:
        path_desc = []
        for node_id in critical_path:
            if node_id in nodes:
                node = nodes[node_id]
                path_desc.append(f"{node_id}({node.name})")
            else:
                path_desc.append(node_id)
        print(f"  最长路径: {' → '.join(path_desc)}")
        print(f"  路径长度: {len(critical_path)} 步")
    else:
        print("  未找到关键路径")
    
    print_subheader("属性使用频率")
    sorted_attrs = sorted(stats['by_attributes'].items(), key=lambda x: x[1], reverse=True)
    for attr, count in sorted_attrs[:8]:  # 显示前8个
        print(f"  {attr}: {count} 次")

def demo_layer_analysis(nodes):
    """Demonstrate detailed layer analysis"""
    print_header("BDAG层级分析演示")
    
    visualizer = BDAGVisualizer(nodes)
    layer_analysis = visualizer.generate_layer_analysis()
    
    layer_names = {'A': '公理层', 'B': '基础操作层', 'C': '复合操作层', 'E': '涌现层', 'U': '统一层'}
    
    for layer, data in layer_analysis['layer_distribution'].items():
        if data['count'] > 0:
            print_subheader(f"{layer_names.get(layer, layer)}分析")
            print(f"  节点数量: {data['count']}")
            print(f"  平均输入数: {data['avg_inputs']:.1f}")
            print(f"  最大输入数: {data['max_inputs']}")
            
            if data['operations']:
                print("  操作类型分布:")
                for op, count in data['operations'].items():
                    print(f"    {op}: {count} 个")

def demo_visualization(nodes):
    """Demonstrate visualization capabilities"""
    print_header("BDAG可视化演示")
    
    visualizer = BDAGVisualizer(nodes)
    
    print_subheader("Mermaid 流程图")
    mermaid = visualizer.generate_mermaid()
    print("```mermaid")
    print(mermaid)
    print("```")
    
    print_subheader("层级关系说明")
    print("  🔴 公理层 (A): 基础定义和常数")
    print("  🔵 基础层 (B): 单输入操作和变换")  
    print("  🟢 复合层 (C): 多输入组合操作")
    print("  🟠 涌现层 (E): 系统级新性质涌现")
    print("  🟣 统一层 (U): 最高级别的理论统一")

def demo_theoretical_insights(nodes):
    """Demonstrate theoretical insights from the BDAG"""
    print_header("二进制宇宙理论洞察")
    
    print_subheader("理论结构分析")
    print("  🔬 原子化原理: 每个文件代表一个张量操作")
    print("  🌐 DAG结构: 严格的层级依赖，无循环引用")
    print("  📐 φ量化: 黄金比例作为信息编码基础")
    print("  🔄 自指完备: A001自指张量是整个理论的根基")
    
    print_subheader("关键理论关系")
    
    # 分析A001 -> B101路径
    if 'A001' in nodes and 'B101' in nodes:
        print("  自指 → 熵增:")
        print("    A001(自指完备) → B101(熵增算子)")
        print("    体现了系统自我引用必然导致熵增的基本原理")
    
    # 分析A002 -> B103路径  
    if 'A002' in nodes and 'B103' in nodes:
        print("  φ比例 → Zeckendorf编码:")
        print("    A002(黄金比例) → B103(φ编码系统)")
        print("    实现了无连续'11'的最优信息编码")
    
    # 分析C201组合
    if 'C201' in nodes:
        print("  信息熵量化:")
        print("    B101(熵增) + B103(φ编码) → C201(量化信息熵)")
        print("    将连续熵增过程量化为Fibonacci步长")
    
    print_subheader("哲学意义")
    print("  🧠 意识涌现: 复杂系统中信息整合产生主观体验")
    print("  ⏰ 时间本质: 时间是熵增过程的量化表现")
    print("  🌌 宇宙结构: 整个宇宙可能是一个自指完备系统")
    print("  💫 量子引力: φ编码可能是统一量子力学和引力的关键")

def main():
    """Main demonstration function"""
    print("🌌 二进制宇宙DAG系统完整演示")
    print("Binary Universe DAG System Comprehensive Demo")
    
    # 1. 解析演示
    nodes = demo_parsing()
    
    # 2. 验证演示
    demo_validation(nodes)
    
    # 3. 统计分析演示
    demo_statistics(nodes)
    
    # 4. 层级分析演示
    demo_layer_analysis(nodes)
    
    # 5. 可视化演示
    demo_visualization(nodes)
    
    # 6. 理论洞察演示
    demo_theoretical_insights(nodes)
    
    print_header("演示总结")
    print("✅ BDAG系统完整功能演示完成")
    print(f"📊 成功处理 {len(nodes)} 个原子化张量操作")
    print("🔧 解析、验证、可视化工具全部就绪")
    print("📚 从T0理论到BDAG格式的迁移路径清晰")
    print("🚀 系统已准备好支持大规模理论开发")
    
    print("\n下一步建议:")
    print("1. 将现有T0-T33理论系统化迁移到BDAG格式")
    print("2. 建立自动化CI/CD验证流程")
    print("3. 开发更多涌现层(E)和统一层(U)理论")
    print("4. 集成到mdBook文档系统")

if __name__ == '__main__':
    main()