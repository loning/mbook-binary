#!/usr/bin/env python3
"""
BDAG Naming System Optimizer
Improved naming scheme with aliases and readability
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class NamingConfig:
    """命名配置"""
    verbosity: str = "medium"  # short, medium, full
    use_aliases: bool = True
    max_filename_length: int = 100
    
class ImprovedBDAGNaming:
    """改进的BDAG命名系统"""
    
    def __init__(self, config: NamingConfig = None):
        self.config = config or NamingConfig()
        
        # 常用术语的别名映射
        self.aliases = {
            'SelfReference': 'SelfRef',
            'InformationEntropy': 'InfoEntropy', 
            'EntropyIncrease': 'EntropyInc',
            'PhiEncoding': 'PhiEnc',
            'ObserverDifferentiation': 'ObsDiff',
            'ZeckendorfSystem': 'ZeckSys',
            'QuantumState': 'QState',
            'SpacetimeGeometry': 'STGeom',
            'MeasurementCollapse': 'MeasCollapse'
        }
        
        # 属性简化映射
        self.attr_short = {
            'Recursive': 'Rec',
            'Entropic': 'Ent', 
            'Monotonic': 'Mono',
            'Irreversible': 'Irrev',
            'Quantized': 'Quant',
            'Compressed': 'Comp',
            'Irrational': 'Irrat',
            'Algebraic': 'Alg'
        }
    
    def generate_filename(self, layer: str, seq: int, name: str, 
                         operation: str, inputs: List[str], 
                         output: str, attributes: List[str]) -> str:
        """生成优化的文件名"""
        
        if self.config.verbosity == "short":
            return self._generate_short_name(layer, seq, name, operation, inputs, attributes)
        elif self.config.verbosity == "medium":
            return self._generate_medium_name(layer, seq, name, operation, inputs, output, attributes)
        else:  # full
            return self._generate_full_name(layer, seq, name, operation, inputs, output, attributes)
    
    def _generate_short_name(self, layer: str, seq: int, name: str, 
                           operation: str, inputs: List[str], attributes: List[str]) -> str:
        """生成简短文件名"""
        short_name = self.aliases.get(name, name[:8])
        short_inputs = "+".join([inp.split('_')[0] for inp in inputs if '_' in inp])
        short_attrs = "_".join([self.attr_short.get(attr, attr[:4]) for attr in attributes[:2]])
        
        return f"{layer}{seq:03d}__{short_name}__FROM__{short_inputs}__ATTR__{short_attrs}.md"
    
    def _generate_medium_name(self, layer: str, seq: int, name: str, 
                            operation: str, inputs: List[str], output: str, attributes: List[str]) -> str:
        """生成中等长度文件名"""
        medium_name = self.aliases.get(name, name)
        
        # 简化输入表示
        if len(inputs) == 1:
            input_part = inputs[0] if '_' in inputs[0] else inputs[0]
        else:
            input_part = "+".join([inp.split('_')[0] if '_' in inp else inp for inp in inputs])
        
        # 限制属性数量
        attrs = attributes[:3]
        attr_part = "_".join([self.attr_short.get(attr, attr) for attr in attrs])
        
        filename = f"{layer}{seq:03d}__{medium_name}__{operation}__FROM__{input_part}__TO__{output}__ATTR__{attr_part}.md"
        
        # 如果太长，进一步简化
        if len(filename) > self.config.max_filename_length:
            return self._generate_short_name(layer, seq, name, operation, inputs, attributes)
        
        return filename
    
    def _generate_full_name(self, layer: str, seq: int, name: str, 
                          operation: str, inputs: List[str], output: str, attributes: List[str]) -> str:
        """生成完整文件名（原始格式）"""
        input_part = "__".join(inputs)
        attr_part = "_".join(attributes)
        
        return f"{layer}{seq:03d}__{name}__{operation}__FROM__{input_part}__TO__{output}__ATTR__{attr_part}.md"
    
    def create_alias_mapping(self, directory: str) -> Dict[str, str]:
        """为目录中的文件创建别名映射"""
        mapping = {}
        
        for file_path in Path(directory).glob("*.md"):
            filename = file_path.name
            
            # 解析原始文件名
            if self._is_bdag_filename(filename):
                short_name = self._extract_short_form(filename)
                mapping[short_name] = filename
        
        return mapping
    
    def _is_bdag_filename(self, filename: str) -> bool:
        """检查是否为BDAG文件名"""
        pattern = r'^[ABCEU]\d{3}__.*\.md$'
        return bool(re.match(pattern, filename))
    
    def _extract_short_form(self, filename: str) -> str:
        """从完整文件名提取短形式"""
        # 这里可以实现反向映射逻辑
        parts = filename.split('__')
        if len(parts) >= 2:
            layer_seq = parts[0]
            name = parts[1]
            short_name = self.aliases.get(name, name[:8])
            return f"{layer_seq}__{short_name}"
        return filename
    
    def suggest_improvements(self, directory: str) -> List[str]:
        """分析并建议命名改进"""
        suggestions = []
        
        for file_path in Path(directory).glob("*.md"):
            filename = file_path.name
            
            if len(filename) > self.config.max_filename_length:
                suggestions.append(f"文件名过长: {filename}")
            
            # 检查重复或相似的名称
            # 检查是否使用了最佳别名
            
        return suggestions

# 使用示例
if __name__ == "__main__":
    optimizer = ImprovedBDAGNaming(NamingConfig(verbosity="medium"))
    
    # 测试不同长度的文件名
    test_cases = [
        ("A", 1, "SelfReference", "DEFINE", ["Axiom"], "SelfRefTensor", ["Recursive", "Entropic"]),
        ("C", 201, "InformationEntropy", "COMBINE", ["B101_EntropyIncrease", "B103_PhiEncoding"], "InfoTensor", ["Quantized", "Compressed", "Optimal"])
    ]
    
    for case in test_cases:
        short = optimizer.generate_filename(*case)
        print(f"优化命名: {short}")