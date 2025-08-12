#!/usr/bin/env python3
"""
T{n} Natural Number Theory Parser
解析T{n}自然数理论文件，支持Zeckendorf分解依赖关系
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class TheoryOperationType(Enum):
    """理论操作类型"""
    AXIOM = 'AXIOM'           # 基础公理 (Fibonacci数理论)
    EMERGE = 'EMERGE'         # 涌现理论 (组合数理论)
    DERIVE = 'DERIVE'         # 推导理论
    COMBINE = 'COMBINE'       # 组合理论
    UNIFY = 'UNIFY'           # 统一理论

@dataclass
class TheoryNode:
    """T{n}理论节点"""
    theory_number: int                    # 理论编号T{n}
    name: str                            # 理论名称
    operation: TheoryOperationType       # 操作类型
    zeckendorf_decomp: List[int]         # 声明的Zeckendorf分解
    expected_zeckendorf: List[int]       # 数学期望的Zeckendorf分解
    theory_dependencies: List[int]       # T{n}理论依赖
    output_type: str                     # 输出类型
    filename: str                        # 文件名
    
    # 计算属性
    complexity_level: int = 0            # 复杂度
    is_fibonacci_theory: bool = False    # 是否Fibonacci数理论
    information_content: float = 0.0     # 信息含量
    is_consistent: bool = True           # 是否一致
    
    def __post_init__(self):
        """计算派生属性"""
        self.complexity_level = len(self.zeckendorf_decomp)
        self.is_fibonacci_theory = (len(self.expected_zeckendorf) == 1)
        self.information_content = self._calculate_info_content()
        self.is_consistent = (set(self.zeckendorf_decomp) == set(self.expected_zeckendorf))
    
    def _calculate_info_content(self) -> float:
        """计算信息含量"""
        import math
        phi = (1 + math.sqrt(5)) / 2
        if self.theory_number > 0:
            return math.log(self.theory_number) / math.log(phi)
        return 0.0

class NaturalNumberTheoryParser:
    """T{n}自然数理论解析器"""
    
    def __init__(self, max_fib: int = 100):
        self.max_fib = max_fib
        self.fibonacci_sequence = self._generate_fibonacci_sequence()
        self.nodes: Dict[int, TheoryNode] = {}
        self.errors: List[str] = []
        
        # T{n}文件名正则表达式
        self.filename_pattern = re.compile(
            r'^T(\d+)__([A-Za-z][A-Za-z0-9_]*)__([A-Z]+)__'
            r'ZECK_(F\d+(?:\+F\d+)*)__'
            r'FROM__((?:T\d+(?:\+T\d+)*)|(?:Universe|Math|Physics|Information|Cosmos|Binary))__'
            r'TO__([A-Za-z][A-Za-z0-9_]*)'
            r'\.md$'
        )
    
    def _generate_fibonacci_sequence(self) -> List[int]:
        """生成Fibonacci序列 (F1=1, F2=2, F3=3, F4=5, F5=8...)"""
        fib = [1, 2]
        while fib[-1] < self.max_fib:
            next_fib = fib[-1] + fib[-2]
            if next_fib <= self.max_fib:
                fib.append(next_fib)
            else:
                break
        return fib
    
    def to_zeckendorf(self, n: int) -> List[int]:
        """计算自然数n的Zeckendorf分解"""
        if n <= 0:
            return []
        
        result = []
        for fib in reversed(self.fibonacci_sequence):
            if fib <= n:
                result.append(fib)
                n -= fib
                if n == 0:
                    break
        
        return sorted(result)
    
    def parse_filename(self, filename: str) -> Optional[TheoryNode]:
        """解析T{n}理论文件名"""
        match = self.filename_pattern.match(filename)
        if not match:
            self.errors.append(f"文件名格式错误: {filename}")
            return None
        
        try:
            theory_num = int(match.group(1))           # T{n}
            name = match.group(2)                      # 理论名称
            operation_str = match.group(3)             # 操作类型
            zeck_str = match.group(4)                  # ZECK声明
            from_str = match.group(5)                  # FROM依赖
            output_type = match.group(6)               # TO输出
            
            # 解析操作类型
            try:
                operation = TheoryOperationType(operation_str)
            except ValueError:
                self.errors.append(f"未知操作类型 {operation_str} in {filename}")
                return None
            
            # 解析声明的Zeckendorf分解
            declared_zeck = self._parse_zeckendorf_string(zeck_str)
            if not declared_zeck:
                self.errors.append(f"无效Zeckendorf分解 {zeck_str} in {filename}")
                return None
            
            # 计算数学期望的Zeckendorf分解
            expected_zeck = self.to_zeckendorf(theory_num)
            
            # 解析理论依赖
            theory_deps = self._parse_theory_dependencies(from_str)
            
            node = TheoryNode(
                theory_number=theory_num,
                name=name,
                operation=operation,
                zeckendorf_decomp=declared_zeck,
                expected_zeckendorf=expected_zeck,
                theory_dependencies=theory_deps,
                output_type=output_type,
                filename=filename
            )
            
            return node
            
        except Exception as e:
            self.errors.append(f"解析错误 {filename}: {str(e)}")
            return None
    
    def _parse_zeckendorf_string(self, zeck_str: str) -> List[int]:
        """解析Zeckendorf分解字符串 'F1+F3+F8' -> [1,3,8]"""
        fib_nums = []
        for match in re.finditer(r'F(\d+)', zeck_str):
            fib_num = int(match.group(1))
            if fib_num in self.fibonacci_sequence:
                fib_nums.append(fib_num)
        return sorted(fib_nums)
    
    def _parse_theory_dependencies(self, from_str: str) -> List[int]:
        """解析理论依赖 'T1+T3' -> [1,3]"""
        # 基础概念无依赖
        base_concepts = {'Universe', 'Math', 'Physics', 'Information', 'Cosmos', 'Binary'}
        if from_str in base_concepts:
            return []
        
        # 解析T{n}依赖
        theory_nums = []
        for match in re.finditer(r'T(\d+)', from_str):
            theory_num = int(match.group(1))
            theory_nums.append(theory_num)
        return sorted(theory_nums)
    
    def parse_directory(self, directory_path: str) -> Dict[int, TheoryNode]:
        """解析目录中的所有T{n}理论文件"""
        directory = Path(directory_path)
        self.nodes.clear()
        self.errors.clear()
        
        if not directory.exists():
            self.errors.append(f"目录不存在: {directory_path}")
            return {}
        
        # 查找T开头的理论文件
        theory_files = list(directory.glob("T*__*.md"))
        
        for file_path in theory_files:
            node = self.parse_filename(file_path.name)
            if node:
                self.nodes[node.theory_number] = node
        
        return self.nodes
    
    def validate_dependencies(self) -> List[str]:
        """验证所有理论的依赖关系"""
        validation_errors = []
        
        for theory_num, node in self.nodes.items():
            # 检查Zeckendorf一致性
            if not node.is_consistent:
                validation_errors.append(
                    f"T{theory_num}: Zeckendorf不一致 - "
                    f"声明{node.zeckendorf_decomp}, 期望{node.expected_zeckendorf}"
                )
            
            # 检查依赖是否匹配Zeckendorf分解
            # 对于复合理论，依赖应该对应Zeckendorf中的Fibonacci数
            if not node.is_fibonacci_theory:
                expected_deps = node.expected_zeckendorf.copy()  # 直接依赖Fibonacci对应的理论
                declared_deps = node.theory_dependencies
                
                if set(expected_deps) != set(declared_deps):
                    validation_errors.append(
                        f"T{theory_num}: 依赖关系不匹配Zeckendorf - "
                        f"声明依赖T{declared_deps}, Zeckendorf期望T{expected_deps}"
                    )
        
        return validation_errors
    
    def generate_theory_statistics(self) -> Dict:
        """生成理论统计信息"""
        if not self.nodes:
            return {}
        
        total = len(self.nodes)
        fibonacci_theories = sum(1 for n in self.nodes.values() if n.is_fibonacci_theory)
        composite_theories = total - fibonacci_theories
        
        # 复杂度分布
        complexity_dist = {}
        for node in self.nodes.values():
            level = node.complexity_level
            complexity_dist[level] = complexity_dist.get(level, 0) + 1
        
        # 操作类型分布
        operation_dist = {}
        for node in self.nodes.values():
            op = node.operation.value
            operation_dist[op] = operation_dist.get(op, 0) + 1
        
        # 一致性统计
        consistent_count = sum(1 for n in self.nodes.values() if n.is_consistent)
        
        return {
            'total_theories': total,
            'fibonacci_theories': fibonacci_theories,
            'composite_theories': composite_theories,
            'complexity_distribution': complexity_dist,
            'operation_distribution': operation_dist,
            'consistency_rate': f"{consistent_count}/{total} ({consistent_count/total*100:.1f}%)"
        }
    
    def print_analysis_report(self):
        """打印分析报告"""
        print("📊 T{n}自然数理论分析报告")
        print("=" * 50)
        
        if self.errors:
            print(f"\n❌ 解析错误 ({len(self.errors)}个):")
            for error in self.errors[:5]:  # 只显示前5个错误
                print(f"  • {error}")
            if len(self.errors) > 5:
                print(f"  ... 还有{len(self.errors)-5}个错误")
        
        if not self.nodes:
            print("\n⚠️ 未找到有效的理论文件")
            return
        
        stats = self.generate_theory_statistics()
        
        print(f"\n📈 基本统计:")
        for key, value in stats.items():
            if not isinstance(value, dict):
                print(f"  {key}: {value}")
        
        print(f"\n🎭 操作类型分布:")
        for op, count in stats['operation_distribution'].items():
            print(f"  {op}: {count}")
        
        print(f"\n📊 复杂度分布:")
        for level, count in sorted(stats['complexity_distribution'].items()):
            print(f"  复杂度{level}: {count}个理论")
        
        # 验证依赖关系
        validation_errors = self.validate_dependencies()
        if validation_errors:
            print(f"\n⚠️ 依赖关系验证错误 ({len(validation_errors)}个):")
            for error in validation_errors:
                print(f"  • {error}")
        else:
            print(f"\n✅ 所有理论的依赖关系都符合Zeckendorf分解！")

def main():
    """测试解析器"""
    parser = NaturalNumberTheoryParser()
    
    # 测试解析目录
    examples_dir = Path(__file__).parent.parent / 'examples'
    if examples_dir.exists():
        parser.parse_directory(str(examples_dir))
        parser.print_analysis_report()
    else:
        print("测试用例：")
        # 测试文件名解析
        test_filenames = [
            "T1__UniversalSelfReference__AXIOM__ZECK_F1__FROM__Universe__TO__SelfRefTensor.md",
            "T4__TemporalEmergence__EMERGE__ZECK_F1+F3__FROM__T1+T3__TO__TimeTensor.md",
            "T8__ComplexEmergence__AXIOM__ZECK_F8__FROM__Cosmos__TO__ComplexTensor.md"
        ]
        
        for filename in test_filenames:
            node = parser.parse_filename(filename)
            if node:
                print(f"✅ {filename}")
                print(f"   T{node.theory_number}: {node.name}")
                print(f"   Zeckendorf: {node.zeckendorf_decomp} (期望: {node.expected_zeckendorf})")
                print(f"   依赖: T{node.theory_dependencies}")
                print(f"   一致性: {'✅' if node.is_consistent else '❌'}")
            else:
                print(f"❌ {filename}")
        
        if parser.errors:
            print(f"\n解析错误:")
            for error in parser.errors:
                print(f"  • {error}")

if __name__ == "__main__":
    main()