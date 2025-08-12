#!/usr/bin/env python3
"""
BDAG Visualizer - DAG visualization and analysis
Binary Universe DAG visualization engine
"""

from typing import List, Dict
from bdag_core import TensorNode

class BDAGVisualizer:
    """BDAG可视化工具"""
    
    def __init__(self, nodes: Dict[str, TensorNode]):
        self.nodes = nodes
    
    def generate_mermaid(self) -> str:
        """生成Mermaid流程图"""
        lines = ["graph TD"]
        
        # 添加节点定义
        for node in self.nodes.values():
            label = f"{node.full_id}<br/>{node.name}<br/>[{node.operation.value}]"
            lines.append(f'    {node.full_id}["{label}"]')
        
        # 添加边
        for node in self.nodes.values():
            for inp in node.inputs:
                if inp.id in self.nodes:
                    lines.append(f"    {inp.id} --> {node.full_id}")
        
        # 添加层级样式
        layer_colors = {
            'A': '#ffebee',  # 浅红色
            'B': '#e3f2fd',  # 浅蓝色
            'C': '#e8f5e8',  # 浅绿色
            'E': '#fff3e0',  # 浅橙色
            'U': '#f3e5f5'   # 浅紫色
        }
        
        for layer, color in layer_colors.items():
            layer_nodes = [n.full_id for n in self.nodes.values() 
                          if n.layer.value == layer]
            if layer_nodes:
                lines.append(f"    classDef layer{layer} fill:{color}")
                lines.append(f"    class {','.join(layer_nodes)} layer{layer}")
        
        return '\n'.join(lines)
    
    def generate_graphviz(self) -> str:
        """生成Graphviz DOT格式"""
        lines = [
            "digraph BDAG {",
            "    rankdir=TB;",
            "    node [shape=box, style=rounded];",
            ""
        ]
        
        # 按层级分组
        layers = {}
        for node in self.nodes.values():
            layer = node.layer.value
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node)
        
        # 添加子图（层级）
        for layer, nodes in layers.items():
            lines.append(f"    subgraph cluster_{layer} {{")
            lines.append(f'        label="{layer} Layer";')
            
            for node in nodes:
                label = f"{node.full_id}\\\\n{node.name}\\\\n[{node.operation.value}]"
                lines.append(f'        {node.full_id} [label="{label}"];')
            
            lines.append("    }")
            lines.append("")
        
        # 添加边
        for node in self.nodes.values():
            for inp in node.inputs:
                if inp.id in self.nodes:
                    lines.append(f"    {inp.id} -> {node.full_id};")
        
        lines.append("}")
        return '\n'.join(lines)
    
    def generate_statistics(self) -> Dict:
        """生成统计信息"""
        stats = {
            'total_nodes': len(self.nodes),
            'by_layer': {},
            'by_operation': {},
            'by_attributes': {},
            'source_nodes': [],
            'terminal_nodes': [],
            'max_depth': 0,
            'total_edges': 0
        }
        
        # 按层级统计
        for node in self.nodes.values():
            layer = node.layer.value
            stats['by_layer'][layer] = stats['by_layer'].get(layer, 0) + 1
        
        # 按操作类型统计
        for node in self.nodes.values():
            op = node.operation.value
            stats['by_operation'][op] = stats['by_operation'].get(op, 0) + 1
        
        # 按属性统计
        for node in self.nodes.values():
            for attr in node.attributes:
                stats['by_attributes'][attr] = stats['by_attributes'].get(attr, 0) + 1
        
        # 源节点和终端节点
        for node in self.nodes.values():
            if node.is_source:
                stats['source_nodes'].append(node.full_id)
            if node.is_terminal:
                stats['terminal_nodes'].append(node.full_id)
        
        # 边的总数
        stats['total_edges'] = sum(
            len([inp for inp in node.inputs if inp.id in self.nodes])
            for node in self.nodes.values()
        )
        
        return stats
    
    def get_critical_path(self) -> List[str]:
        """获取关键路径（最长路径）"""
        # 计算每个节点的最大深度
        depths = {}
        
        def compute_depth(node_id: str) -> int:
            if node_id in depths:
                return depths[node_id]
            
            if node_id not in self.nodes:
                return 0
            
            node = self.nodes[node_id]
            max_input_depth = 0
            
            for inp in node.inputs:
                if inp.id in self.nodes:
                    input_depth = compute_depth(inp.id)
                    max_input_depth = max(max_input_depth, input_depth)
            
            depths[node_id] = max_input_depth + 1
            return depths[node_id]
        
        # 计算所有节点的深度
        for node_id in self.nodes:
            compute_depth(node_id)
        
        # 找到最大深度的节点
        if not depths:
            return []
        
        max_depth = max(depths.values())
        terminal_nodes = [node_id for node_id, depth in depths.items() 
                         if depth == max_depth]
        
        # 回溯找到路径
        def backtrack(node_id: str, path: List[str]) -> List[str]:
            if node_id not in self.nodes:
                return [node_id] + path
            
            current_depth = depths[node_id]
            if current_depth == 1:
                return [node_id] + path
            
            node = self.nodes[node_id]
            for inp in node.inputs:
                if inp.id in self.nodes and depths[inp.id] == current_depth - 1:
                    return backtrack(inp.id, [node_id] + path)
            
            return [node_id] + path
        
        if terminal_nodes:
            return backtrack(terminal_nodes[0], [])
        else:
            return []
    
    def generate_layer_analysis(self) -> Dict:
        """生成层级分析报告"""
        analysis = {
            'layer_distribution': {},
            'operation_distribution': {},
            'dependency_patterns': {},
            'complexity_metrics': {}
        }
        
        # 层级分布
        for layer in ['A', 'B', 'C', 'E', 'U']:
            layer_nodes = [n for n in self.nodes.values() if n.layer.value == layer]
            analysis['layer_distribution'][layer] = {
                'count': len(layer_nodes),
                'operations': {},
                'avg_inputs': 0,
                'max_inputs': 0
            }
            
            if layer_nodes:
                # 操作类型分布
                for node in layer_nodes:
                    op = node.operation.value
                    analysis['layer_distribution'][layer]['operations'][op] = \
                        analysis['layer_distribution'][layer]['operations'].get(op, 0) + 1
                
                # 输入统计
                input_counts = [len([inp for inp in node.inputs if inp.id in self.nodes]) 
                               for node in layer_nodes]
                analysis['layer_distribution'][layer]['avg_inputs'] = \
                    sum(input_counts) / len(input_counts) if input_counts else 0
                analysis['layer_distribution'][layer]['max_inputs'] = \
                    max(input_counts) if input_counts else 0
        
        return analysis