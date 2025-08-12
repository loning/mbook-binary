#!/usr/bin/env python3
"""
BDAG Command Line Interface
Binary Universe DAG processing toolkit
"""

import argparse
import json
import sys
from pathlib import Path

from bdag_core import BDAGParser
from bdag_validator import BDAGValidator
from bdag_visualizer import BDAGVisualizer

def main():
    parser = argparse.ArgumentParser(description='BDAG文件处理工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 解析命令
    parse_parser = subparsers.add_parser('parse', help='解析BDAG文件')
    parse_parser.add_argument('directory', help='包含BDAG文件的目录')
    parse_parser.add_argument('--output', '-o', help='输出文件路径')
    parse_parser.add_argument('--format', choices=['json', 'yaml'], 
                             default='json', help='输出格式')
    
    # 验证命令
    validate_parser = subparsers.add_parser('validate', help='验证BDAG结构')
    validate_parser.add_argument('directory', help='包含BDAG文件的目录')
    validate_parser.add_argument('--strict', action='store_true',
                               help='严格模式验证')
    
    # 可视化命令
    viz_parser = subparsers.add_parser('visualize', help='生成可视化图表')
    viz_parser.add_argument('directory', help='包含BDAG文件的目录')
    viz_parser.add_argument('--format', choices=['mermaid', 'graphviz'],
                           default='mermaid', help='图表格式')
    viz_parser.add_argument('--output', '-o', help='输出文件路径')
    
    # 统计命令
    stats_parser = subparsers.add_parser('stats', help='生成统计信息')
    stats_parser.add_argument('directory', help='包含BDAG文件的目录')
    stats_parser.add_argument('--detailed', action='store_true',
                             help='详细统计信息')
    
    args = parser.parse_args()
    
    if args.command == 'parse':
        cmd_parse(args)
    elif args.command == 'validate':
        cmd_validate(args)
    elif args.command == 'visualize':
        cmd_visualize(args)
    elif args.command == 'stats':
        cmd_stats(args)
    else:
        parser.print_help()

def cmd_parse(args):
    """解析命令实现"""
    parser = BDAGParser()
    nodes = parser.parse_directory(args.directory)
    
    if parser.get_errors():
        print("解析错误:", file=sys.stderr)
        for error in parser.get_errors():
            print(f"  {error}", file=sys.stderr)
        sys.exit(1)
    
    # 转换为可序列化格式
    result = {}
    for node_id, node in nodes.items():
        result[node_id] = {
            'layer': node.layer.value,
            'sequence': node.sequence,
            'name': node.name,
            'operation': node.operation.value,
            'inputs': [{'id': inp.id, 'name': inp.name} for inp in node.inputs],
            'output_type': node.output_type,
            'attributes': node.attributes,
            'filename': node.filename,
            'is_source': node.is_source,
            'is_terminal': node.is_terminal
        }
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到 {args.output}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

def cmd_validate(args):
    """验证命令实现"""
    parser = BDAGParser()
    nodes = parser.parse_directory(args.directory)
    
    # 解析错误
    parse_errors = parser.get_errors()
    
    # 验证错误
    validation_errors = []
    if not parse_errors:  # 只有解析成功才进行验证
        validator = BDAGValidator(nodes)
        validation_errors = validator.validate_all()
    
    total_errors = parse_errors + validation_errors
    
    if total_errors:
        print(f"发现 {len(total_errors)} 个错误:")
        for i, error in enumerate(total_errors, 1):
            print(f"{i:2d}. {error}")
        sys.exit(1)
    else:
        print(f"验证通过! 共处理 {len(nodes)} 个节点。")

def cmd_visualize(args):
    """可视化命令实现"""
    parser = BDAGParser()
    nodes = parser.parse_directory(args.directory)
    
    if parser.get_errors():
        print("解析错误，无法生成可视化:", file=sys.stderr)
        for error in parser.get_errors():
            print(f"  {error}", file=sys.stderr)
        sys.exit(1)
    
    visualizer = BDAGVisualizer(nodes)
    
    if args.format == 'mermaid':
        content = visualizer.generate_mermaid()
    else:  # graphviz
        content = visualizer.generate_graphviz()
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"可视化图表已保存到 {args.output}")
    else:
        print(content)

def cmd_stats(args):
    """统计命令实现"""
    parser = BDAGParser()
    nodes = parser.parse_directory(args.directory)
    
    if parser.get_errors():
        print("解析错误，无法生成统计:", file=sys.stderr)
        for error in parser.get_errors():
            print(f"  {error}", file=sys.stderr)
        sys.exit(1)
    
    visualizer = BDAGVisualizer(nodes)
    stats = visualizer.generate_statistics()
    
    print(f"BDAG统计信息:")
    print(f"  总节点数: {stats['total_nodes']}")
    print(f"  总边数: {stats['total_edges']}")
    print(f"  源节点数: {len(stats['source_nodes'])}")
    print(f"  终端节点数: {len(stats['terminal_nodes'])}")
    
    print(f"\n按层级分布:")
    for layer, count in sorted(stats['by_layer'].items()):
        print(f"  {layer}层: {count} 个节点")
    
    print(f"\n按操作类型分布:")
    for op, count in sorted(stats['by_operation'].items()):
        print(f"  {op}: {count} 个节点")
    
    if args.detailed:
        print(f"\n源节点: {', '.join(stats['source_nodes'])}")
        print(f"\n终端节点: {', '.join(stats['terminal_nodes'])}")
        
        critical_path = visualizer.get_critical_path()
        print(f"\n关键路径: {' → '.join(critical_path)}")
        
        print(f"\n属性使用频率:")
        for attr, count in sorted(stats['by_attributes'].items(), 
                                 key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {attr}: {count} 次")
        
        # 层级分析
        layer_analysis = visualizer.generate_layer_analysis()
        print(f"\n层级分析:")
        for layer, data in layer_analysis['layer_distribution'].items():
            if data['count'] > 0:
                print(f"  {layer}层:")
                print(f"    节点数: {data['count']}")
                print(f"    平均输入: {data['avg_inputs']:.1f}")
                print(f"    最大输入: {data['max_inputs']}")
                print(f"    操作分布: {dict(data['operations'])}")

if __name__ == '__main__':
    main()