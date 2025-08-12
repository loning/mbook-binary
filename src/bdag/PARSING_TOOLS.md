# 解析工具和实现

## 工具概述

BDAG解析工具套件提供了完整的文件名解析、验证、可视化和管理功能，确保DAG结构的正确性和一致性。

## 核心解析器

### Python解析器实现

```python
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from enum import Enum
import json

class LayerType(Enum):
    AXIOM = 'A'
    BASIC = 'B'
    COMPOSITE = 'C'
    EMERGENT = 'E'
    UNIFIED = 'U'

class OperationType(Enum):
    DEFINE = 'DEFINE'
    APPLY = 'APPLY'
    TRANSFORM = 'TRANSFORM'
    COMBINE = 'COMBINE'
    EMERGE = 'EMERGE'
    DERIVE = 'DERIVE'

@dataclass
class InputNode:
    """输入节点信息"""
    id: str
    name: str
    layer: Optional[LayerType] = None
    
    def __post_init__(self):
        if self.id and self.id[0] in ['A', 'B', 'C', 'E', 'U']:
            self.layer = LayerType(self.id[0])

@dataclass
class TensorNode:
    """张量节点完整信息"""
    # 基本标识
    layer: LayerType
    sequence: int
    full_id: str
    name: str
    operation: OperationType
    
    # 依赖关系
    inputs: List[InputNode]
    output_type: str
    attributes: List[str]
    
    # 元信息
    filename: str
    is_terminal: bool = False
    is_source: bool = False
    
    def __post_init__(self):
        """计算派生属性"""
        self.is_source = len(self.inputs) == 0 or all(
            inp.id in ['Axiom', 'Constant', 'Sequence', 'Constraint'] 
            for inp in self.inputs
        )
        # is_terminal 需要在构建完整DAG后计算

class BDAGParser:
    """BDAG文件名解析器"""
    
    # 完整文件名正则表达式
    FILENAME_PATTERN = re.compile(
        r'^([ABCEU])(\d{3})__([A-Z][a-zA-Z0-9]*)__([A-Z]+)__'
        r'FROM__((?:[A-Z]\d{3}_[A-Z][a-zA-Z0-9]*(?:__[A-Z]\d{3}_[A-Z][a-zA-Z0-9]*)*)|'
        r'(?:Axiom|Constant|Sequence|Constraint))__'
        r'TO__([A-Z][a-zA-Z0-9]*)__'
        r'ATTR__([A-Z][a-zA-Z]*(?:_[A-Z][a-zA-Z]*)*)'
        r'\.md$'
    )
    
    def __init__(self):
        self.nodes: Dict[str, TensorNode] = {}
        self.errors: List[str] = []
    
    def parse_filename(self, filename: str) -> Optional[TensorNode]:
        """解析单个文件名"""
        match = self.FILENAME_PATTERN.match(filename)
        if not match:
            self.errors.append(f"文件名格式错误: {filename}")
            return None
        
        try:
            # 提取基本信息
            layer_char = match.group(1)
            sequence = int(match.group(2))
            name = match.group(3)
            operation = match.group(4)
            inputs_str = match.group(5)
            output_type = match.group(6)
            attributes_str = match.group(7)
            
            # 验证和转换
            layer = LayerType(layer_char)
            full_id = f"{layer_char}{sequence:03d}"
            
            try:
                op_type = OperationType(operation)
            except ValueError:
                self.errors.append(f"未知操作类型: {operation} in {filename}")
                return None
            
            # 解析输入节点
            inputs = self._parse_inputs(inputs_str, filename)
            if inputs is None:
                return None
            
            # 解析属性
            attributes = attributes_str.split('_')
            
            return TensorNode(
                layer=layer,
                sequence=sequence,
                full_id=full_id,
                name=name,
                operation=op_type,
                inputs=inputs,
                output_type=output_type,
                attributes=attributes,
                filename=filename
            )
            
        except Exception as e:
            self.errors.append(f"解析错误 {filename}: {str(e)}")
            return None
    
    def _parse_inputs(self, inputs_str: str, filename: str) -> Optional[List[InputNode]]:
        """解析输入节点字符串"""
        inputs = []
        
        # 处理基础输入类型
        if inputs_str in ['Axiom', 'Constant', 'Sequence', 'Constraint']:
            return [InputNode('', inputs_str)]
        
        # 解析节点输入
        try:
            input_parts = inputs_str.split('__')
            for part in input_parts:
                if '_' in part:
                    input_id, input_name = part.split('_', 1)
                    inputs.append(InputNode(input_id, input_name))
                else:
                    self.errors.append(f"输入节点格式错误: {part} in {filename}")
                    return None
            
            return inputs
            
        except Exception as e:
            self.errors.append(f"输入解析错误 {filename}: {str(e)}")
            return None
    
    def parse_directory(self, directory_path: str) -> Dict[str, TensorNode]:
        """解析目录中的所有BDAG文件"""
        from pathlib import Path
        
        self.nodes.clear()
        self.errors.clear()
        
        try:
            for file_path in Path(directory_path).glob("*.md"):
                node = self.parse_filename(file_path.name)
                if node:
                    if node.full_id in self.nodes:
                        self.errors.append(f"重复的节点ID: {node.full_id}")
                    else:
                        self.nodes[node.full_id] = node
        
        except Exception as e:
            self.errors.append(f"目录读取错误: {str(e)}")
        
        # 计算终端节点
        self._compute_terminal_nodes()
        
        return self.nodes
    
    def _compute_terminal_nodes(self):
        """计算哪些节点是终端节点（没有被其他节点依赖）"""
        referenced_ids = set()
        
        for node in self.nodes.values():
            for inp in node.inputs:
                if inp.id:  # 排除 Axiom 等基础输入
                    referenced_ids.add(inp.id)
        
        for node in self.nodes.values():
            node.is_terminal = node.full_id not in referenced_ids
    
    def get_errors(self) -> List[str]:
        """获取解析过程中的错误"""
        return self.errors.copy()
    
    def validate_dag(self) -> List[str]:
        """验证DAG结构的正确性"""
        validator = BDAGValidator(self.nodes)
        return validator.validate_all()
```

### DAG验证器

```python
class BDAGValidator:
    """BDAG结构验证器"""
    
    def __init__(self, nodes: Dict[str, TensorNode]):
        self.nodes = nodes
        self.errors: List[str] = []
    
    def validate_all(self) -> List[str]:
        """执行所有验证检查"""
        self.errors.clear()
        
        self._validate_layer_constraints()
        self._validate_dependency_hierarchy()
        self._validate_operation_types()
        self._validate_sequence_uniqueness()
        self._validate_no_cycles()
        self._validate_input_existence()
        
        return self.errors.copy()
    
    def _validate_layer_constraints(self):
        """验证层级约束"""
        layer_order = {'A': 0, 'B': 1, 'C': 2, 'E': 3, 'U': 4}
        
        for node in self.nodes.values():
            current_level = layer_order[node.layer.value]
            
            for inp in node.inputs:
                if inp.layer and inp.id in self.nodes:
                    dep_level = layer_order[inp.layer.value]
                    
                    # 依赖层级必须低于当前层级
                    if dep_level >= current_level:
                        self.errors.append(
                            f"违反层级约束: {node.full_id} 依赖同级或更高级的 {inp.id}"
                        )
                    
                    # 同层级依赖必须是前序节点
                    if (inp.layer.value == node.layer.value and 
                        int(inp.id[1:4]) >= node.sequence):
                        self.errors.append(
                            f"违反序号约束: {node.full_id} 依赖后序节点 {inp.id}"
                        )
    
    def _validate_dependency_hierarchy(self):
        """验证依赖层级的合理性"""
        allowed_deps = {
            'A': [],
            'B': ['A'],
            'C': ['A', 'B'],
            'E': ['A', 'B', 'C'],
            'U': ['A', 'B', 'C', 'E']
        }
        
        for node in self.nodes.values():
            layer = node.layer.value
            allowed = allowed_deps[layer]
            
            for inp in node.inputs:
                if inp.layer and inp.layer.value not in allowed:
                    self.errors.append(
                        f"不允许的依赖关系: {layer}层的 {node.full_id} "
                        f"依赖 {inp.layer.value}层的 {inp.id}"
                    )
    
    def _validate_operation_types(self):
        """验证操作类型的合理性"""
        allowed_operations = {
            'A': ['DEFINE'],
            'B': ['APPLY', 'TRANSFORM', 'DERIVE'],
            'C': ['COMBINE', 'DERIVE', 'TRANSFORM'],
            'E': ['EMERGE', 'DERIVE'],
            'U': ['DERIVE', 'EMERGE', 'COMBINE', 'TRANSFORM']
        }
        
        for node in self.nodes.values():
            layer = node.layer.value
            operation = node.operation.value
            
            if operation not in allowed_operations[layer]:
                self.errors.append(
                    f"不适合的操作类型: {layer}层不应使用 {operation} 操作 "
                    f"({node.full_id})"
                )
    
    def _validate_sequence_uniqueness(self):
        """验证序号唯一性"""
        layer_sequences = {}
        
        for node in self.nodes.values():
            layer = node.layer.value
            seq = node.sequence
            
            if layer not in layer_sequences:
                layer_sequences[layer] = set()
            
            if seq in layer_sequences[layer]:
                self.errors.append(
                    f"重复的序号: {layer}层的序号 {seq:03d} 被多次使用"
                )
            else:
                layer_sequences[layer].add(seq)
    
    def _validate_no_cycles(self):
        """验证无环性"""
        # 使用DFS检测环
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node_id: WHITE for node_id in self.nodes}
        
        def dfs(node_id: str) -> bool:
            if colors[node_id] == GRAY:
                return True  # 发现环
            if colors[node_id] == BLACK:
                return False  # 已访问完成
            
            colors[node_id] = GRAY
            
            node = self.nodes[node_id]
            for inp in node.inputs:
                if inp.id in self.nodes and dfs(inp.id):
                    return True
            
            colors[node_id] = BLACK
            return False
        
        for node_id in self.nodes:
            if colors[node_id] == WHITE:
                if dfs(node_id):
                    self.errors.append(f"检测到循环依赖，涉及节点: {node_id}")
                    break
    
    def _validate_input_existence(self):
        """验证输入节点存在性"""
        for node in self.nodes.values():
            for inp in node.inputs:
                if (inp.id and 
                    inp.id not in ['Axiom', 'Constant', 'Sequence', 'Constraint'] and
                    inp.id not in self.nodes):
                    self.errors.append(
                        f"输入节点不存在: {node.full_id} 引用了不存在的 {inp.id}"
                    )
```

### DAG可视化工具

```python
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
                label = f"{node.full_id}\\n{node.name}\\n[{node.operation.value}]"
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
        max_depth = max(depths.values())
        terminal_nodes = [node_id for node_id, depth in depths.items() 
                         if depth == max_depth]
        
        # 回溯找到路径
        def backtrack(node_id: str, path: List[str]) -> List[str]:
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
```

## 命令行工具

### BDAG CLI工具

```python
#!/usr/bin/env python3
"""
BDAG命令行工具
"""

import argparse
import json
import sys
from pathlib import Path

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

if __name__ == '__main__':
    main()
```

## 使用示例

### 基本解析
```bash
# 解析目录中的所有BDAG文件
python bdag_cli.py parse /path/to/bdag/files

# 保存解析结果到JSON文件
python bdag_cli.py parse /path/to/bdag/files -o parsed.json
```

### 验证DAG结构
```bash
# 验证DAG结构
python bdag_cli.py validate /path/to/bdag/files

# 严格模式验证
python bdag_cli.py validate /path/to/bdag/files --strict
```

### 生成可视化
```bash
# 生成Mermaid图表
python bdag_cli.py visualize /path/to/bdag/files --format mermaid -o dag.mmd

# 生成Graphviz图表
python bdag_cli.py visualize /path/to/bdag/files --format graphviz -o dag.dot
```

### 统计分析
```bash
# 基本统计
python bdag_cli.py stats /path/to/bdag/files

# 详细统计
python bdag_cli.py stats /path/to/bdag/files --detailed
```

## 集成到构建系统

### Makefile集成
```makefile
# 验证BDAG文件
validate-bdag:
	python tools/bdag_cli.py validate src/bdag/files

# 生成可视化
generate-dag-viz:
	python tools/bdag_cli.py visualize src/bdag/files \
		--format mermaid -o docs/dag.mmd
	python tools/bdag_cli.py visualize src/bdag/files \
		--format graphviz -o docs/dag.dot

# 生成统计报告
generate-stats:
	python tools/bdag_cli.py stats src/bdag/files \
		--detailed > docs/bdag_stats.txt

.PHONY: validate-bdag generate-dag-viz generate-stats
```

### Pre-commit Hook
```bash
#!/bin/sh
# .git/hooks/pre-commit

echo "验证BDAG文件..."
python tools/bdag_cli.py validate src/bdag/files

if [ $? -ne 0 ]; then
    echo "BDAG验证失败，提交被拒绝。"
    exit 1
fi

echo "BDAG验证通过。"
```

### CI/CD集成
```yaml
# .github/workflows/bdag-validation.yml
name: BDAG Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Validate BDAG files
      run: |
        python tools/bdag_cli.py validate src/bdag/files
    - name: Generate visualization
      run: |
        python tools/bdag_cli.py visualize src/bdag/files \
          --format mermaid -o dag.mmd
    - name: Upload artifacts
      uses: actions/upload-artifact@v2
      with:
        name: dag-visualization
        path: dag.mmd
```