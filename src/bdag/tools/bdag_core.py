#!/usr/bin/env python3
"""
BDAG Core Classes and Data Structures
Binary Universe DAG parsing and validation core
"""

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
        if self.id and len(self.id) > 0 and self.id[0] in ['A', 'B', 'C', 'E', 'U']:
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
        
        # 检查目录是否存在
        dir_path = Path(directory_path)
        if not dir_path.exists():
            self.errors.append(f"目录不存在: {directory_path}")
            return self.nodes
        
        if not dir_path.is_dir():
            self.errors.append(f"路径不是目录: {directory_path}")
            return self.nodes
        
        try:
            md_files = list(dir_path.glob("*.md"))
            if not md_files:
                self.errors.append(f"目录中没有找到 .md 文件: {directory_path}")
                return self.nodes
            
            print(f"找到 {len(md_files)} 个 .md 文件")
            
            for file_path in md_files:
                print(f"正在解析: {file_path.name}")
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
        
        print(f"成功解析 {len(self.nodes)} 个节点")
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