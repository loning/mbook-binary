#!/usr/bin/env python3
"""
BDAG Validator - DAG structure validation
Binary Universe DAG validation engine
"""

from typing import List, Dict
from bdag_core import TensorNode, LayerType, OperationType

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