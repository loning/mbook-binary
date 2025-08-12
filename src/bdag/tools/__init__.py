"""
T{n}理论系统工具集
Binary Universe T{n} Theory System Tools v3.0
五类分类系统：AXIOM/PRIME-FIB/FIBONACCI/PRIME/COMPOSITE
"""

# 主要工具 - v3.0版本
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
    UniversalTensorSpace,
    TensorClassification
)
from .prime_theory_classifier import (
    PrimeTheoryClassifier,
    PrimeClassification
)
from .prime_theory_analyzer import (
    PrimeTheoryAnalyzer,
    PrimeAnalysis
)
from .classification_statistics import (
    ClassificationStatisticsGenerator
)
from .theory_table_generator import (
    generate_complete_theory_table
)
from .theory_table_generator_prime import (
    generate_enhanced_theory_table
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
    
    # 三维张量空间
    'UniversalTensorSpace',
    'TensorClassification',
    
    # 素数理论工具
    'PrimeTheoryClassifier',
    'PrimeClassification',
    'PrimeTheoryAnalyzer', 
    'PrimeAnalysis',
    
    # 统计分析工具
    'ClassificationStatisticsGenerator',
    
    # 表格生成工具
    'generate_complete_theory_table',
    'generate_enhanced_theory_table'
]

__version__ = '3.0.0'
__author__ = 'Binary Universe Theory Project'
__description__ = 'T{n}理论系统工具集 - 五类分类系统（AXIOM/PRIME-FIB/FIBONACCI/PRIME/COMPOSITE）的统一框架'