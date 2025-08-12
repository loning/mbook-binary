#!/usr/bin/env python3
"""
Test the semantic validator
"""

import sys
from pathlib import Path

# Add the tools directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from bdag_core import BDAGParser
from semantic_validator import SemanticValidator

def test_semantic_validation():
    """Test semantic validation functionality"""
    print("=== 测试语义验证器 ===")
    
    examples_dir = Path(__file__).parent.parent / "examples"
    
    # 解析BDAG文件
    parser = BDAGParser()
    nodes = parser.parse_directory(str(examples_dir))
    
    if parser.get_errors():
        print("解析错误:")
        for error in parser.get_errors():
            print(f"  {error}")
        return
    
    print(f"成功解析 {len(nodes)} 个节点")
    
    # 执行语义验证
    validator = SemanticValidator(nodes)
    errors, warnings = validator.validate_semantic_consistency()
    
    # 生成报告
    report = validator.generate_report()
    print(report)

if __name__ == "__main__":
    test_semantic_validation()