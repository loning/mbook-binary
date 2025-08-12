#!/usr/bin/env python3
"""
BDAG Semantic Validator
Deep validation of theoretical consistency and mathematical correctness
"""

import re
from typing import List, Dict, Set, Tuple
from pathlib import Path
from bdag_core import TensorNode

class SemanticValidator:
    """BDAG语义验证器"""
    
    def __init__(self, nodes: Dict[str, TensorNode]):
        self.nodes = nodes
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        # 定义已知的数学关系
        self.math_relations = {
            'phi_fibonacci': r'F_n.*φ.*golden.*ratio',
            'entropy_increase': r'H\(.*\).*>.*H\(',
            'zeckendorf_unique': r'unique.*representation.*Fibonacci',
            'self_reference': r'S\(S\)|S.*=.*S\(S\)'
        }
        
        # 物理维度定义
        self.dimensions = {
            'SelfRefTensor': {'information': 1, 'time': 0},
            'EntropyTensor': {'information': 1, 'time': -1},
            'GoldenRatio': {'dimensionless': 1},
            'ZeckendorfSystem': {'information': 1},
            'InfoTensor': {'information': 1, 'time': -1}
        }
    
    def validate_semantic_consistency(self) -> Tuple[List[str], List[str]]:
        """执行语义一致性验证"""
        self.errors.clear()
        self.warnings.clear()
        
        self._validate_mathematical_consistency()
        self._validate_dimensional_analysis()
        self._validate_phi_encoding_rules()
        self._validate_entropy_monotonicity()
        self._validate_content_filename_match()
        
        return self.errors.copy(), self.warnings.copy()
    
    def _validate_mathematical_consistency(self):
        """验证数学一致性"""
        for node in self.nodes.values():
            if node.operation.value == 'DEFINE':
                self._check_definition_completeness(node)
            elif node.operation.value == 'APPLY':
                self._check_application_validity(node)
            elif node.operation.value == 'COMBINE':
                self._check_combination_rules(node)
    
    def _check_definition_completeness(self, node: TensorNode):
        """检查定义的完整性"""
        # 读取对应的文件内容
        content = self._read_node_content(node)
        if not content:
            self.errors.append(f"无法读取 {node.full_id} 的文件内容")
            return
        
        # 检查是否有数学表示
        if '$$' not in content and '$' not in content:
            self.warnings.append(f"{node.full_id}: 缺少数学表示")
        
        # 检查特定的数学关系
        if node.name == 'Phi' and 'varphi' not in content:
            self.errors.append(f"{node.full_id}: φ定义中缺少数学符号")
        
        if node.name == 'SelfReference' and not re.search(self.math_relations['self_reference'], content):
            self.warnings.append(f"{node.full_id}: 自指关系表达不清晰")
    
    def _check_application_validity(self, node: TensorNode):
        """检查应用操作的有效性"""
        content = self._read_node_content(node)
        if not content:
            return
        
        # 检查熵增操作
        if 'Entropy' in node.name and not re.search(self.math_relations['entropy_increase'], content):
            self.warnings.append(f"{node.full_id}: 熵增性质未明确表达")
        
        # 检查φ编码操作
        if 'Phi' in node.name and not re.search(self.math_relations['zeckendorf_unique'], content):
            self.warnings.append(f"{node.full_id}: Zeckendorf唯一性未说明")
    
    def _check_combination_rules(self, node: TensorNode):
        """检查组合操作的规则"""
        if len(node.inputs) < 2:
            self.errors.append(f"{node.full_id}: COMBINE操作需要至少2个输入")
        
        # 检查输入之间的兼容性
        input_types = []
        for inp in node.inputs:
            if inp.id in self.nodes:
                input_node = self.nodes[inp.id]
                input_types.append(input_node.output_type)
        
        # 验证组合的数学意义
        content = self._read_node_content(node)
        if content and 'otimes' not in content and 'tensor' not in content.lower():
            self.warnings.append(f"{node.full_id}: 组合操作的数学形式不清晰")
    
    def _validate_dimensional_analysis(self):
        """验证物理维度分析"""
        for node in self.nodes.values():
            if node.output_type in self.dimensions:
                expected_dims = self.dimensions[node.output_type]
                
                # 检查输入维度的一致性
                for inp in node.inputs:
                    if inp.id in self.nodes:
                        input_node = self.nodes[inp.id]
                        if input_node.output_type in self.dimensions:
                            self._check_dimensional_compatibility(node, input_node)
    
    def _check_dimensional_compatibility(self, node: TensorNode, input_node: TensorNode):
        """检查维度兼容性"""
        output_dims = self.dimensions.get(node.output_type, {})
        input_dims = self.dimensions.get(input_node.output_type, {})
        
        # 对于APPLY操作，输入输出维度应该有明确关系
        if node.operation.value == 'APPLY':
            if 'Entropy' in node.name and 'information' not in output_dims:
                self.errors.append(f"{node.full_id}: 熵操作应该输出信息维度")
    
    def _validate_phi_encoding_rules(self):
        """验证φ编码规则"""
        phi_nodes = [n for n in self.nodes.values() if 'Phi' in n.name or 'phi' in n.name.lower()]
        
        for node in phi_nodes:
            content = self._read_node_content(node)
            if content:
                # 检查No-11约束
                if 'No-11' not in content and '11' in content:
                    self.warnings.append(f"{node.full_id}: 应该明确No-11约束")
                
                # 检查Fibonacci关系
                if 'Fibonacci' not in content and 'F_n' not in content:
                    self.warnings.append(f"{node.full_id}: 缺少Fibonacci关系说明")
    
    def _validate_entropy_monotonicity(self):
        """验证熵的单调性"""
        entropy_nodes = [n for n in self.nodes.values() if 'Entropy' in n.name]
        
        for node in entropy_nodes:
            if 'Monotonic' not in node.attributes:
                self.warnings.append(f"{node.full_id}: 熵操作应该标记为Monotonic")
            
            if 'Irreversible' not in node.attributes:
                self.warnings.append(f"{node.full_id}: 熵操作应该标记为Irreversible")
    
    def _validate_content_filename_match(self):
        """验证文件内容与文件名的匹配"""
        for node in self.nodes.values():
            content = self._read_node_content(node)
            if not content:
                continue
            
            # 检查操作类型是否在内容中体现
            op_keywords = {
                'DEFINE': ['定义', 'define', '张量定义'],
                'APPLY': ['应用', 'apply', '算子'],
                'COMBINE': ['组合', 'combine', '张量积', 'otimes'],
                'EMERGE': ['涌现', 'emerge', '新性质'],
                'DERIVE': ['推导', 'derive', '推导出']
            }
            
            expected_keywords = op_keywords.get(node.operation.value, [])
            if not any(keyword in content.lower() for keyword in expected_keywords):
                self.warnings.append(f"{node.full_id}: 内容与操作类型 {node.operation.value} 不匹配")
    
    def _read_node_content(self, node: TensorNode) -> str:
        """读取节点对应的文件内容"""
        try:
            # 假设文件在examples目录中
            file_path = Path(__file__).parent.parent / "examples" / node.filename
            if file_path.exists():
                return file_path.read_text(encoding='utf-8')
        except Exception as e:
            pass
        return ""
    
    def generate_report(self) -> str:
        """生成验证报告"""
        report = ["BDAG语义验证报告", "=" * 40]
        
        if self.errors:
            report.append(f"\n🚨 错误 ({len(self.errors)} 个):")
            for i, error in enumerate(self.errors, 1):
                report.append(f"  {i}. {error}")
        
        if self.warnings:
            report.append(f"\n⚠️  警告 ({len(self.warnings)} 个):")
            for i, warning in enumerate(self.warnings, 1):
                report.append(f"  {i}. {warning}")
        
        if not self.errors and not self.warnings:
            report.append("\n✅ 所有语义检查通过!")
        
        return "\n".join(report)

# 测试代码
if __name__ == "__main__":
    # 这里需要加载实际的节点数据
    print("语义验证器已准备就绪")