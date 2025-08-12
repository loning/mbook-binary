"""
T{n}理论系统工具集
Binary Universe T{n} Theory System Tools v2.0
"""

# 主要工具 - 新版本
from .theory_parser import (
    TheoryParser,
    TheoryNode,
    FibonacciOperationType
)
from .theory_validator import (
    TheorySystemValidator,
    ValidationReport,
    ValidationLevel,
    ValidationIssue
)
from .fibonacci_tensor_space import (
    FibonacciTensorSpace,
    FibonacciTensor,
    FibonacciDimension
)
from .bdag_visualizer import (
    FibonacciBDAG,
    VisualizerNode
)

__all__ = [
    # 核心解析器
    'TheoryParser',
    'TheoryNode', 
    'FibonacciOperationType',
    
    # 验证系统
    'TheorySystemValidator',
    'ValidationReport',
    'ValidationLevel',
    'ValidationIssue',
    
    # 张量空间
    'FibonacciTensorSpace',
    'FibonacciTensor',
    'FibonacciDimension',
    
    # 可视化
    'FibonacciBDAG',
    'VisualizerNode'
]

__version__ = '2.1.0'
__author__ = 'Binary Universe Theory Project'
__description__ = 'T{n}理论系统工具集 - 支持THEOREM/EXTENDED分类的统一框架'