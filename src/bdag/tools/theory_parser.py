#!/usr/bin/env python3
"""
T{n} Theory Parser v2.0
统一的T{n}理论解析器，支持新的THEOREM/EXTENDED分类系统
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import math

class FibonacciOperationType(Enum):
    """T{n}理论操作类型 - 基于素数-Fibonacci分类系统"""
    AXIOM = 'AXIOM'           # 公理理论（只有T1）
    PRIME_FIB = 'PRIME-FIB'   # 既是素数又是Fibonacci的理论
    FIBONACCI = 'FIBONACCI'   # 纯Fibonacci理论（非素数）
    PRIME = 'PRIME'           # 纯素数理论（非Fibonacci）
    COMPOSITE = 'COMPOSITE'   # 合数理论（既非素数也非Fibonacci）

@dataclass
class TheoryNode:
    """T{n}理论节点 - 支持新分类系统"""
    theory_number: int                      # 理论编号T{n}
    name: str                              # 理论名称
    operation: FibonacciOperationType      # 操作类型
    zeckendorf_decomp: List[int]           # Zeckendorf分解
    theory_dependencies: List[int]         # T{n}理论依赖
    output_type: str                       # 输出张量类型
    filename: str                          # 文件名
    
    # 计算属性
    complexity_level: int = 0              # 复杂度等级
    is_fibonacci_theory: bool = False      # 是否为单个Fibonacci数
    is_single_axiom_system: bool = False   # 是否为单公理系统
    information_content: float = 0.0       # 信息含量
    is_consistent: bool = True             # 依赖一致性
    
    def __post_init__(self):
        """计算派生属性"""
        self.complexity_level = len(self.zeckendorf_decomp) 
        self.is_fibonacci_theory = (len(self.zeckendorf_decomp) == 1)
        self.is_single_axiom_system = (self.theory_number == 1)  # 只有T1是真正的公理
        self.information_content = self._calculate_info_content()
        self.is_consistent = self._validate_consistency()
    
    def _calculate_info_content(self) -> float:
        """计算信息含量 log_φ(n)"""
        phi = (1 + math.sqrt(5)) / 2
        if self.theory_number > 0:
            return math.log(self.theory_number) / math.log(phi)
        return 0.0
    
    def _validate_consistency(self) -> bool:
        """验证理论一致性"""
        expected_zeck = TheoryParser.to_zeckendorf_static(self.theory_number)
        
        # 基本一致性检查
        if set(self.zeckendorf_decomp) != set(expected_zeck):
            return False
            
        # 操作类型一致性检查
        if self.theory_number == 1 and self.operation != FibonacciOperationType.AXIOM:
            return False
        elif self.theory_number > 1 and self.operation == FibonacciOperationType.AXIOM:
            return False
            
        return True
    
    @property
    def theory_type_description(self) -> str:
        """理论类型描述"""
        if self.theory_number == 1:
            return "唯一公理 (自指完备)"
        elif self.is_fibonacci_theory:
            return "Fibonacci递归定理"
        else:
            return "Zeckendorf扩展定理"

class TheoryParser:
    """统一T{n}理论解析器"""
    
    def __init__(self, max_theory: int = 100):
        self.max_theory = max_theory
        self.fibonacci_sequence = self._generate_fibonacci_sequence()
        self.fibonacci_set = set(self.fibonacci_sequence)
        self.nodes: Dict[int, TheoryNode] = {}
        self.errors: List[str] = []
        
        # T{n}文件名正则表达式 - 支持素数分类格式
        self.filename_pattern = re.compile(
            r'^T(\d+)__([A-Za-z][A-Za-z0-9_]*)__(AXIOM|PRIME-FIB|FIBONACCI|PRIME|COMPOSITE)__'
            r'ZECK_(F\d+(?:\+F\d+)*)__'
            r'FROM__((?:T\d+(?:\+T\d+)*)|(?:UNIVERSE|Universe|Math|Physics|Information|Cosmos|Binary))__'
            r'TO__([A-Za-z][A-Za-z0-9_]*)'
            r'\.md$'
        )
    
    @staticmethod
    def is_prime(n: int) -> bool:
        """检查数字是否为素数"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def get_theory_classification(self, n: int) -> FibonacciOperationType:
        """获取理论的完整分类（包含素数信息）"""
        if n == 1:
            return FibonacciOperationType.AXIOM
        elif n in self.fibonacci_set and self.is_prime(n):
            return FibonacciOperationType.PRIME_FIB
        elif n in self.fibonacci_set:
            return FibonacciOperationType.FIBONACCI
        elif self.is_prime(n):
            return FibonacciOperationType.PRIME
        else:
            return FibonacciOperationType.COMPOSITE
    
    def _generate_fibonacci_sequence(self) -> List[int]:
        """生成Fibonacci序列 (F1=1, F2=2, F3=3, F4=5, F5=8...)"""
        fib = [1, 2, 3]  # F1=1, F2=2, F3=3
        while fib[-1] < self.max_theory:
            next_fib = fib[-1] + fib[-2]  # F4=F3+F2=3+2=5, F5=F4+F3=5+3=8
            if next_fib <= self.max_theory:
                fib.append(next_fib)
            else:
                break
        return fib
    
    @staticmethod
    def to_zeckendorf_static(n: int) -> List[int]:
        """静态方法：计算自然数n的Zeckendorf分解"""
        if n <= 0:
            return []
        
        # 生成到n的Fibonacci序列
        fib = [1, 2, 3]  # F1=1, F2=2, F3=3
        while fib[-1] < n:
            next_fib = fib[-1] + fib[-2]  # F4=F3+F2=3+2=5, F5=F4+F3=5+3=8
            if next_fib <= n:
                fib.append(next_fib)
            else:
                break
        
        result = []
        for fib_num in reversed(fib):
            if fib_num <= n:
                result.append(fib_num)
                n -= fib_num
                if n == 0:
                    break
        
        return sorted(result)
    
    def to_zeckendorf(self, n: int) -> List[int]:
        """实例方法：计算自然数n的Zeckendorf分解"""
        return self.to_zeckendorf_static(n)
    
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
                operation = FibonacciOperationType(operation_str)
            except ValueError:
                self.errors.append(f"未知操作类型 {operation_str} in {filename}")
                return None
            
            # 解析声明的Zeckendorf分解
            declared_zeck = self._parse_zeckendorf_string(zeck_str)
            if not declared_zeck:
                self.errors.append(f"无效Zeckendorf分解 {zeck_str} in {filename}")
                return None
            
            # 解析理论依赖
            theory_deps = self._parse_theory_dependencies(from_str)
            
            node = TheoryNode(
                theory_number=theory_num,
                name=name,
                operation=operation,
                zeckendorf_decomp=declared_zeck,
                theory_dependencies=theory_deps,
                output_type=output_type,
                filename=filename
            )
            
            return node
            
        except Exception as e:
            self.errors.append(f"解析错误 {filename}: {str(e)}")
            return None
    
    def _parse_zeckendorf_string(self, zeck_str: str) -> List[int]:
        """解析Zeckendorf分解字符串 'F1+F3+F4' -> [1,3,5] (F1=1, F3=3, F4=5)"""
        fib_nums = []
        for match in re.finditer(r'F(\d+)', zeck_str):
            fib_index = int(match.group(1))  # Fibonacci索引
            # 将Fibonacci索引转换为实际值
            if 1 <= fib_index <= len(self.fibonacci_sequence):
                fib_value = self.fibonacci_sequence[fib_index - 1]  # F1是索引0
                fib_nums.append(fib_value)
            else:
                self.errors.append(f"无效Fibonacci索引: F{fib_index} in {zeck_str}")
        return sorted(fib_nums)
    
    def _parse_theory_dependencies(self, from_str: str) -> List[int]:
        """解析理论依赖 'T1+T3' -> [1,3]"""
        # 基础概念无依赖
        base_concepts = {
            'Universe', 'UNIVERSE', 'Math', 'Physics', 
            'Information', 'Cosmos', 'Binary'
        }
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
    
    def validate_all_theories(self) -> List[str]:
        """验证所有理论的一致性"""
        validation_errors = []
        
        for theory_num, node in self.nodes.items():
            # 基本一致性检查
            if not node.is_consistent:
                expected_zeck = self.to_zeckendorf(theory_num)
                validation_errors.append(
                    f"T{theory_num}: Zeckendorf不一致 - "
                    f"声明{node.zeckendorf_decomp}, 期望{expected_zeck}"
                )
            
            # 单公理系统检查
            if theory_num == 1:
                if node.operation != FibonacciOperationType.AXIOM:
                    validation_errors.append(
                        f"T1: 必须是AXIOM类型，当前为{node.operation.value}"
                    )
                if node.theory_dependencies:
                    validation_errors.append(
                        f"T1: 作为唯一公理不应有理论依赖，当前依赖{node.theory_dependencies}"
                    )
            
            # 非公理理论检查
            elif theory_num > 1:
                if node.operation == FibonacciOperationType.AXIOM:
                    validation_errors.append(
                        f"T{theory_num}: 不应是AXIOM类型（只有T1是公理），当前为{node.operation.value}"
                    )
            
            # 依赖关系检查
            self._validate_theory_dependencies(node, validation_errors)
        
        return validation_errors
    
    def _validate_theory_dependencies(self, node: TheoryNode, errors: List[str]):
        """验证单个理论的依赖关系"""
        theory_num = node.theory_number
        declared_deps = node.theory_dependencies
        
        # 对于Fibonacci递归定理，依赖应该对应递归关系
        if node.is_fibonacci_theory and theory_num > 2:
            # F_n = F_{n-1} + F_{n-2}，但依赖基于理论逻辑而非数值递归
            pass  # 暂时跳过，因为理论依赖基于语义而非纯数学递归
        
        # 对于扩展定理，依赖应该对应Zeckendorf分解
        elif not node.is_fibonacci_theory:
            # 检查依赖是否存在于系统中
            for dep in declared_deps:
                if dep not in self.nodes and dep != 1:  # T1可能未解析但应存在
                    errors.append(
                        f"T{theory_num}: 依赖T{dep}不存在于系统中"
                    )
    
    def generate_statistics(self) -> Dict:
        """生成详细统计信息"""
        if not self.nodes:
            return {'total_theories': 0}
        
        total = len(self.nodes)
        axiom_count = sum(1 for n in self.nodes.values() if n.operation == FibonacciOperationType.AXIOM)
        prime_fib_count = sum(1 for n in self.nodes.values() if n.operation == FibonacciOperationType.PRIME_FIB)
        fibonacci_count = sum(1 for n in self.nodes.values() if n.operation == FibonacciOperationType.FIBONACCI)
        prime_count = sum(1 for n in self.nodes.values() if n.operation == FibonacciOperationType.PRIME)
        composite_count = sum(1 for n in self.nodes.values() if n.operation == FibonacciOperationType.COMPOSITE)
        fibonacci_theories = sum(1 for n in self.nodes.values() if n.is_fibonacci_theory)
        prime_theories = sum(1 for n in self.nodes.values() if self.is_prime(n.theory_number))
        
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
        
        # 理论层次分析
        max_theory_num = max(self.nodes.keys()) if self.nodes else 0
        fibonacci_coverage = sum(1 for f in self.fibonacci_sequence if f <= max_theory_num and f in self.nodes) / len([f for f in self.fibonacci_sequence if f <= max_theory_num]) if self.fibonacci_sequence else 0
        
        return {
            'total_theories': total,
            'axiom_theories': axiom_count,
            'prime_fib_theories': prime_fib_count,
            'fibonacci_theories': fibonacci_count,
            'prime_theories': prime_count,
            'composite_theories': composite_count,
            'total_fibonacci': fibonacci_theories,
            'total_prime': prime_theories,
            'complexity_distribution': complexity_dist,
            'operation_distribution': operation_dist,
            'consistency_rate': f"{consistent_count}/{total} ({consistent_count/total*100:.1f}%)" if total > 0 else "N/A",
            'fibonacci_coverage': f"{fibonacci_coverage*100:.1f}%",
            'max_theory_number': max_theory_num,
            'single_axiom_system': axiom_count == 1 and 1 in self.nodes
        }
    
    def print_comprehensive_report(self):
        """打印综合分析报告"""
        print("🔬 T{n}理论系统综合分析报告 v2.0")
        print("=" * 60)
        
        if self.errors:
            print(f"\n❌ 解析错误 ({len(self.errors)}个):")
            for error in self.errors[:10]:  # 显示前10个错误
                print(f"  • {error}")
            if len(self.errors) > 10:
                print(f"  ... 还有{len(self.errors)-10}个错误")
        
        if not self.nodes:
            print("\n⚠️ 未找到有效的理论文件")
            return
        
        stats = self.generate_statistics()
        
        print(f"\n📊 系统概览:")
        print(f"  总理论数: {stats['total_theories']}")
        print(f"  公理理论: {stats['axiom_theories']} ({'✅' if stats['single_axiom_system'] else '❌'} 单公理系统)")
        print(f"  素数-Fibonacci: {stats['prime_fib_theories']}")
        print(f"  纯Fibonacci: {stats['fibonacci_theories']}")
        print(f"  纯素数: {stats['prime_theories']}")
        print(f"  合数理论: {stats['composite_theories']}")
        print(f"  Fibonacci覆盖: {stats['fibonacci_coverage']}")
        print(f"  最高理论: T{stats['max_theory_number']}")
        print(f"  一致性率: {stats['consistency_rate']}")
        
        print(f"\n🎭 操作类型分布:")
        for op, count in stats['operation_distribution'].items():
            print(f"  {op}: {count}")
        
        print(f"\n📈 复杂度分布:")
        for level, count in sorted(stats['complexity_distribution'].items()):
            print(f"  复杂度{level}: {count}个理论")
        
        # 详细理论列表
        print(f"\n📚 理论详情:")
        for theory_num in sorted(self.nodes.keys()):
            node = self.nodes[theory_num]
            status = "✅" if node.is_consistent else "❌"
            deps_str = f"←T{node.theory_dependencies}" if node.theory_dependencies else "←Universe"
            print(f"  T{theory_num:2d}: {node.name:20s} [{node.operation.value:8s}] {deps_str:15s} {status}")
        
        # 验证依赖关系
        validation_errors = self.validate_all_theories()
        if validation_errors:
            print(f"\n⚠️ 系统一致性问题 ({len(validation_errors)}个):")
            for error in validation_errors:
                print(f"  • {error}")
        else:
            print(f"\n✅ 系统完全一致！符合单公理理论体系。")

def main():
    """测试解析器"""
    parser = TheoryParser()
    
    # 测试解析目录
    examples_dir = Path(__file__).parent.parent / 'examples'
    if examples_dir.exists():
        print(f"解析目录: {examples_dir}")
        parser.parse_directory(str(examples_dir))
        parser.print_comprehensive_report()
    else:
        print("examples目录不存在，使用测试用例：")
        
        # 测试文件名解析
        test_filenames = [
            "T1__SelfReferenceAxiom__AXIOM__ZECK_F1__FROM__UNIVERSE__TO__SelfRefTensor.md",
            "T2__EntropyTheorem__THEOREM__ZECK_F2__FROM__T1__TO__EntropyTensor.md",
            "T3__ConstraintTheorem__THEOREM__ZECK_F3__FROM__T2+T1__TO__ConstraintTensor.md",
            "T4__TimeExtended__EXTENDED__ZECK_F1+F3__FROM__T1+T3__TO__TimeTensor.md",
            "T5__SpaceTheorem__THEOREM__ZECK_F4__FROM__T3+T2__TO__SpaceTensor.md",
            "T6__QuantumExtended__EXTENDED__ZECK_F1+F4__FROM__T1+T5__TO__QuantumTensor.md"
        ]
        
        for filename in test_filenames:
            node = parser.parse_filename(filename)
            if node:
                print(f"✅ {filename}")
                print(f"   T{node.theory_number}: {node.name} ({node.theory_type_description})")
                print(f"   操作: {node.operation.value}")
                print(f"   Zeckendorf: {node.zeckendorf_decomp}")
                print(f"   依赖: T{node.theory_dependencies}")
                print(f"   信息: {node.information_content:.2f} φ-bits")
                print(f"   一致性: {'✅' if node.is_consistent else '❌'}")
            else:
                print(f"❌ {filename}")
        
        if parser.errors:
            print(f"\n解析错误:")
            for error in parser.errors:
                print(f"  • {error}")

if __name__ == "__main__":
    main()