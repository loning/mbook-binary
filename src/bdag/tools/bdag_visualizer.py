#!/usr/bin/env python3
"""
BDAG Visualizer v2.0
生成T{n}理论依赖关系的有向无环图(DAG)可视化
支持新的THEOREM/EXTENDED分类系统
"""

import re
import os
from typing import List, Dict, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
try:
    from .theory_parser import TheoryParser, TheoryNode as ParsedTheoryNode, FibonacciOperationType
except ImportError:
    from theory_parser import TheoryParser, TheoryNode as ParsedTheoryNode, FibonacciOperationType

try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("⚠️ graphviz库未安装，可视化功能受限")

@dataclass
class VisualizerNode:
    """可视化节点（重命名以避免冲突）"""
    theory_number: int
    name: str
    operation: str
    dependencies: List[int]
    file_path: str
    is_fibonacci_theory: bool = False
    information_content: float = 0.0

class FibonacciBDAG:
    """T{n}理论有向无环图可视化器 v2.0"""
    
    def __init__(self):
        self.parser = TheoryParser()
        self.nodes: Dict[int, VisualizerNode] = {}
        self.edges: List[Tuple[int, int]] = []
    
    def load_from_directory(self, directory_path: str):
        """从目录加载T{n}理论文件"""
        nodes_dict = self.parser.parse_directory(directory_path)
        
        # 转换为可视化节点
        for theory_num, parsed_node in nodes_dict.items():
            vis_node = VisualizerNode(
                theory_number=theory_num,
                name=parsed_node.name,
                operation=parsed_node.operation.value,
                dependencies=parsed_node.theory_dependencies,
                file_path=parsed_node.filename,
                is_fibonacci_theory=parsed_node.is_fibonacci_theory,
                information_content=parsed_node.information_content
            )
            self.nodes[theory_num] = vis_node
            
            # 添加边
            for dep in parsed_node.theory_dependencies:
                self.edges.append((dep, theory_num))
    
    def get_node_levels(self) -> Dict[int, int]:
        """计算节点层级（用于布局）"""
        levels = {}
        
        # 使用拓扑排序计算层级
        changed = True
        max_iterations = 10
        iteration = 0
        
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            
            for theory_num, node in self.nodes.items():
                if theory_num in levels:
                    continue
                
                # 检查所有依赖是否已有层级
                deps_levels = []
                all_deps_resolved = True
                
                for dep in node.dependencies:
                    if dep in levels:
                        deps_levels.append(levels[dep])
                    else:
                        all_deps_resolved = False
                        break
                
                # 如果所有依赖都有层级，设置当前节点层级
                if all_deps_resolved:
                    if deps_levels:
                        levels[theory_num] = max(deps_levels) + 1
                    else:
                        levels[theory_num] = 1  # 没有依赖的理论
                    changed = True
        
        # 处理剩余节点（可能有循环依赖）
        for theory_num in self.nodes:
            if theory_num not in levels:
                levels[theory_num] = 99  # 标记为未解析
        
        return levels
    
    def get_statistics(self) -> Dict:
        """获取图统计信息"""
        fibonacci_nodes = [n for n in self.nodes.values() if n.is_fibonacci_theory]
        non_fibonacci_nodes = [n for n in self.nodes.values() if not n.is_fibonacci_theory]
        
        # 操作类型统计
        operation_counts = defaultdict(int)
        for node in self.nodes.values():
            operation_counts[node.operation] += 1
        
        # 层级统计
        levels = self.get_node_levels()
        level_counts = defaultdict(int)
        for level in levels.values():
            level_counts[level] += 1
        
        return {
            "总节点数": len(self.nodes),
            "Fibonacci理论数": len(fibonacci_nodes),
            "非Fibonacci理论数": len(non_fibonacci_nodes),
            "边数": len(self.edges),
            "操作类型分布": dict(operation_counts),
            "层级分布": dict(level_counts)
        }
    
    def generate_dot_graph(self) -> str:
        """生成Graphviz DOT格式的图"""
        lines = [
            "digraph TheoryBDAG {",
            "    rankdir=TB;",
            "    node [fontname=\"Arial Unicode MS\"];",
            "    edge [fontname=\"Arial Unicode MS\"];",
            "    "
        ]
        
        # 节点样式定义
        node_styles = {
            "AXIOM": 'shape=ellipse, style=filled, fillcolor=lightgreen',
            "THEOREM": 'shape=diamond, style=filled, fillcolor=orange',
            "EXTENDED": 'shape=hexagon, style=filled, fillcolor=lightcyan'
        }
        
        # 获取层级信息
        levels = self.get_node_levels()
        
        # 按层级分组节点
        level_groups = defaultdict(list)
        for theory_num, level in levels.items():
            level_groups[level].append(theory_num)
        
        # 生成子图（相同层级）
        for level in sorted(level_groups.keys()):
            if level != 99:  # 跳过未解析的节点
                lines.append(f"    // Level {level}")
                lines.append("    {")
                lines.append("        rank=same;")
                for theory_num in level_groups[level]:
                    lines.append(f"        \"T{theory_num}\";")
                lines.append("    }")
                lines.append("")
        
        # 添加节点
        for theory_num, node in self.nodes.items():
            style = node_styles.get(node.operation, 'shape=circle')
            
            # 显示理论编号和名称
            label = f"T{theory_num}\\n{node.name}"
            
            lines.append(f'    "T{theory_num}" [{style}, label="{label}"];')
        
        lines.append("")
        
        # 添加边
        for src, dst in self.edges:
            lines.append(f'    "T{src}" -> "T{dst}";')
        
        lines.append("}")
        return "\n".join(lines)
    
    def save_graph(self, output_path: str, format: str = 'png'):
        """保存图形文件"""
        if not HAS_GRAPHVIZ:
            print("❌ 需要安装graphviz库: pip install graphviz")
            return False
        
        dot_source = self.generate_dot_graph()
        
        try:
            graph = graphviz.Source(dot_source)
            graph.render(output_path, format=format, cleanup=True)
            print(f"✅ 图形已保存: {output_path}.{format}")
            return True
        except Exception as e:
            print(f"❌ 保存图形失败: {e}")
            return False
    
    def print_analysis(self):
        """打印图分析结果"""
        print("📊 Fibonacci理论依赖关系图分析")
        print("=" * 50)
        
        stats = self.get_statistics()
        
        print("\n🔢 基本统计:")
        for key, value in stats.items():
            if not isinstance(value, dict):
                print(f"  {key}: {value}")
        
        print("\n🎭 操作类型分布:")
        for op, count in stats["操作类型分布"].items():
            print(f"  {op}: {count}")
        
        print("\n🏗️ 层级结构:")
        for level, count in sorted(stats["层级分布"].items()):
            if level == 99:
                print(f"  未解析: {count}")
            else:
                print(f"  第{level}层: {count}个节点")
        
        # 显示层级详情
        levels = self.get_node_levels()
        print("\n📋 节点层级详情:")
        for level in sorted(set(levels.values())):
            if level == 99:
                continue
            nodes_at_level = [f"T{n}" for n, l in levels.items() if l == level]
            print(f"  第{level}层: {', '.join(nodes_at_level)}")

def main():
    """演示T{n}理论BDAG可视化器"""
    print("🌐 T{n}理论BDAG可视化器")
    print("=" * 50)
    
    bdag = FibonacciBDAG()
    
    # 加载理论目录
    theory_dir = Path(__file__).parent.parent / 'examples'
    
    if theory_dir.exists():
        print(f"加载理论目录: {theory_dir}")
        bdag.load_from_directory(str(theory_dir))
        bdag.print_analysis()
        
        # 生成DOT源码
        print(f"\n🔧 Graphviz DOT源码:")
        print("-" * 30)
        print(bdag.generate_dot_graph())
        
        # 尝试保存图形
        output_path = Path(__file__).parent.parent / 'theory_bdag'
        if bdag.save_graph(str(output_path)):
            print(f"✅ 图形文件已生成")
        else:
            print("💾 DOT源码已生成，可以手动使用Graphviz处理")
    else:
        print("❌ 未找到理论目录")

if __name__ == "__main__":
    main()