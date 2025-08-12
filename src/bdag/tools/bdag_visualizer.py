#!/usr/bin/env python3
"""
BDAG Visualizer
生成Fibonacci理论依赖关系的有向无环图(DAG)可视化
"""

import re
import os
from typing import List, Dict, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("⚠️ graphviz库未安装，可视化功能受限")

@dataclass
class TheoryNode:
    """理论节点"""
    fibonacci_number: int
    name: str
    operation: str
    dependencies: List[str]
    file_path: str
    is_base_concept: bool = False  # 是否为基础概念(Universe, Math等)

class FibonacciBDAG:
    """Fibonacci理论有向无环图"""
    
    def __init__(self):
        self.nodes: Dict[str, TheoryNode] = {}
        self.edges: List[Tuple[str, str]] = []
        self.fibonacci_sequence = self._generate_fibonacci_sequence(100)
    
    def _generate_fibonacci_sequence(self, max_fib: int) -> List[int]:
        """生成Fibonacci序列"""
        fib = [1, 2]
        while fib[-1] < max_fib:
            next_fib = fib[-1] + fib[-2]
            if next_fib <= max_fib:
                fib.append(next_fib)
            else:
                break
        return fib
    
    def _parse_theory_filename(self, filename: str) -> Tuple[int, str, str, List[str]]:
        """解析理论文件名"""
        pattern = r'F(\d+)__(.+?)__(.+?)__FROM__(.+?)__TO__'
        match = re.match(pattern, filename)
        
        if match:
            fib_num = int(match.group(1))
            theory_name = match.group(2)
            operation = match.group(3)
            from_deps = match.group(4)
            
            # 提取依赖项
            dependencies = self._extract_dependencies(from_deps)
            return fib_num, theory_name, operation, dependencies
        
        raise ValueError(f"无法解析文件名: {filename}")
    
    def _extract_dependencies(self, deps_string: str) -> List[str]:
        """提取依赖项"""
        # 查找F数字模式
        fib_pattern = r'F(\d+)'
        fib_matches = re.findall(fib_pattern, deps_string)
        
        dependencies = []
        
        # 添加Fibonacci依赖
        for match in fib_matches:
            dependencies.append(f"F{match}")
        
        # 如果没有F依赖，添加基础概念
        if not dependencies:
            base_concepts = ["Universe", "Math", "Physics", "Information", "Cosmos"]
            for concept in base_concepts:
                if concept in deps_string:
                    dependencies.append(concept)
        
        return dependencies
    
    def load_from_directory(self, directory_path: str):
        """从目录加载理论文件"""
        theory_dir = Path(directory_path)
        
        if not theory_dir.exists():
            raise ValueError(f"目录不存在: {directory_path}")
        
        # 清空现有数据
        self.nodes.clear()
        self.edges.clear()
        
        # 加载理论文件
        for file_path in theory_dir.glob("F*__*.md"):
            try:
                fib_num, name, operation, deps = self._parse_theory_filename(file_path.name)
                
                node_id = f"F{fib_num}"
                node = TheoryNode(
                    fibonacci_number=fib_num,
                    name=name,
                    operation=operation,
                    dependencies=deps,
                    file_path=str(file_path)
                )
                
                self.nodes[node_id] = node
                
                # 添加边
                for dep in deps:
                    self.edges.append((dep, node_id))
                    
                    # 如果依赖不是F数字，创建基础概念节点
                    if not dep.startswith('F') and dep not in self.nodes:
                        base_node = TheoryNode(
                            fibonacci_number=-1,
                            name=dep,
                            operation="BASE_CONCEPT",
                            dependencies=[],
                            file_path="",
                            is_base_concept=True
                        )
                        self.nodes[dep] = base_node
                        
            except Exception as e:
                print(f"解析文件失败 {file_path.name}: {e}")
    
    def get_node_levels(self) -> Dict[str, int]:
        """计算节点层级（用于布局）"""
        levels = {}
        
        # 初始化基础概念为第0层
        for node_id, node in self.nodes.items():
            if node.is_base_concept:
                levels[node_id] = 0
        
        # 使用拓扑排序计算层级
        changed = True
        max_iterations = 10
        iteration = 0
        
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            
            for node_id, node in self.nodes.items():
                if node_id in levels:
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
                        levels[node_id] = max(deps_levels) + 1
                    else:
                        levels[node_id] = 1  # 没有依赖的F数字
                    changed = True
        
        # 处理剩余节点（可能有循环依赖）
        for node_id in self.nodes:
            if node_id not in levels:
                levels[node_id] = 99  # 标记为未解析
        
        return levels
    
    def get_statistics(self) -> Dict:
        """获取图统计信息"""
        fibonacci_nodes = [n for n in self.nodes.values() if not n.is_base_concept]
        base_nodes = [n for n in self.nodes.values() if n.is_base_concept]
        
        # 操作类型统计
        operation_counts = defaultdict(int)
        for node in fibonacci_nodes:
            operation_counts[node.operation] += 1
        
        # 层级统计
        levels = self.get_node_levels()
        level_counts = defaultdict(int)
        for level in levels.values():
            level_counts[level] += 1
        
        return {
            "总节点数": len(self.nodes),
            "Fibonacci理论数": len(fibonacci_nodes),
            "基础概念数": len(base_nodes),
            "边数": len(self.edges),
            "操作类型分布": dict(operation_counts),
            "层级分布": dict(level_counts)
        }
    
    def generate_dot_graph(self) -> str:
        """生成Graphviz DOT格式的图"""
        lines = [
            "digraph FibonacciBDAG {",
            "    rankdir=TB;",
            "    node [fontname=\"Arial Unicode MS\"];",
            "    edge [fontname=\"Arial Unicode MS\"];",
            "    "
        ]
        
        # 节点样式定义
        node_styles = {
            "BASE_CONCEPT": 'shape=box, style=filled, fillcolor=lightblue',
            "AXIOM": 'shape=ellipse, style=filled, fillcolor=lightgreen',
            "DEFINE": 'shape=ellipse, style=filled, fillcolor=lightgreen', 
            "EMERGE": 'shape=diamond, style=filled, fillcolor=orange',
            "COMBINE": 'shape=diamond, style=filled, fillcolor=yellow',
            "DERIVE": 'shape=hexagon, style=filled, fillcolor=pink',
            "APPLY": 'shape=hexagon, style=filled, fillcolor=lightcyan'
        }
        
        # 获取层级信息
        levels = self.get_node_levels()
        
        # 按层级分组节点
        level_groups = defaultdict(list)
        for node_id, level in levels.items():
            level_groups[level].append(node_id)
        
        # 生成子图（相同层级）
        for level in sorted(level_groups.keys()):
            if level != 99:  # 跳过未解析的节点
                lines.append(f"    // Level {level}")
                lines.append("    {")
                lines.append("        rank=same;")
                for node_id in level_groups[level]:
                    lines.append(f"        \"{node_id}\";")
                lines.append("    }")
                lines.append("")
        
        # 添加节点
        for node_id, node in self.nodes.items():
            style = node_styles.get(node.operation, 'shape=circle')
            
            if node.is_base_concept:
                label = node.name
            else:
                # 显示F数字和理论名
                label = f"F{node.fibonacci_number}\\n{node.name}"
            
            lines.append(f'    "{node_id}" [{style}, label="{label}"];')
        
        lines.append("")
        
        # 添加边
        for src, dst in self.edges:
            lines.append(f'    "{src}" -> "{dst}";')
        
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
            nodes_at_level = [n for n, l in levels.items() if l == level]
            print(f"  第{level}层: {', '.join(nodes_at_level)}")

def main():
    """演示BDAG可视化器"""
    print("🌐 Fibonacci理论BDAG可视化器")
    print("=" * 50)
    
    bdag = FibonacciBDAG()
    
    # 加载examples目录
    examples_dir = Path(__file__).parent.parent / 'examples'
    
    if examples_dir.exists():
        bdag.load_from_directory(str(examples_dir))
        bdag.print_analysis()
        
        # 生成DOT源码
        print(f"\n🔧 Graphviz DOT源码:")
        print("-" * 30)
        print(bdag.generate_dot_graph())
        
        # 尝试保存图形
        output_path = Path(__file__).parent.parent / 'fibonacci_bdag'
        if bdag.save_graph(str(output_path)):
            print(f"✅ 图形文件已生成")
        else:
            print("💾 DOT源码已生成，可以手动使用Graphviz处理")
    else:
        print("❌ 未找到examples目录")

if __name__ == "__main__":
    main()