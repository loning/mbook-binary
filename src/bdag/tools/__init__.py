"""
Fibonacci张量空间理论系统
Binary Universe Fibonacci Tensor Space Theory System
"""

from .unified_fibonacci_parser import (
    UnifiedFibonacciParser,
    FibonacciNode,
    FibonacciOperationType
)
from .fibonacci_tensor_space import (
    FibonacciTensorSpace,
    FibonacciTensor,
    FibonacciDimension
)

__all__ = [
    'UnifiedFibonacciParser',
    'FibonacciNode', 
    'FibonacciOperationType',
    'FibonacciTensorSpace',
    'FibonacciTensor',
    'FibonacciDimension'
]

__version__ = '2.0.0'
__author__ = 'Binary Universe Theory Project'
__description__ = 'Fibonacci张量空间映射系统 - 宇宙理论的数学本体论'