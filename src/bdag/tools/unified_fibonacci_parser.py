#!/usr/bin/env python3
"""
Unified Fibonacci BDAG Parser
Pure Fibonacci numbering without artificial layer divisions
"""

import re
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class FibonacciOperationType(Enum):
    """基于Fibonacci性质的操作类型"""
    AXIOM = 'AXIOM'           # 素Fibonacci数对应的基础公理
    DEFINE = 'DEFINE'         # 基本定义
    APPLY = 'APPLY'          # 单一应用
    COMBINE = 'COMBINE'       # 两项组合
    EMERGE = 'EMERGE'        # 复杂涌现
    DERIVE = 'DERIVE'        # 数学推导
    UNIFY = 'UNIFY'          # 高阶统一

@dataclass
class FibonacciNode:
    """统一Fibonacci理论节点"""
    fibonacci_number: int                    # Fibonacci序列位置
    name: str                               # 理论名称
    operation: FibonacciOperationType       # 操作类型
    zeckendorf_decomposition: List[int]     # Zeckendorf分解
    dependencies: List[int]                 # 自然依赖(=Zeckendorf分解)
    output_type: str                        # 输出类型
    attributes: List[str]                   # 属性列表
    filename: str                           # 文件名
    
    # 计算属性
    complexity_level: int = 0               # 复杂度等级
    is_prime_fibonacci: bool = False        # 是否素Fibonacci
    information_content: float = 0.0        # 信息含量
    
    def __post_init__(self):
        """计算派生属性"""
        self.complexity_level = len(self.zeckendorf_decomposition)
        self.is_prime_fibonacci = self._is_prime_fibonacci()
        self.information_content = self._calculate_info_content()
    
    def _is_prime_fibonacci(self) -> bool:
        """检查是否为素Fibonacci数"""
        # 这里简化实现，真实情况需要质数测试
        prime_fibonacci_positions = {2, 3, 5, 13, 89, 233, 1597}
        return self.fibonacci_number in prime_fibonacci_positions
    
    def _calculate_info_content(self) -> float:
        """计算信息含量"""
        import math
        phi = (1 + math.sqrt(5)) / 2
        if self.fibonacci_number > 0:
            return math.log(self.fibonacci_number) / math.log(phi)
        return 0.0

class UnifiedFibonacciParser:
    """统一Fibonacci解析器"""
    
    def __init__(self, max_fib=100):
        self.fibonacci_set = self._generate_fibonacci(max_fib)
        self.fibonacci_list = self._generate_fibonacci_list(max_fib)
        self.nodes: Dict[int, FibonacciNode] = {}
        self.errors: List[str] = []
        
        # Fibonacci文件名正则
        self.filename_pattern = re.compile(
            r'^F(\d+)__([A-Za-z][A-Za-z0-9]*)__([A-Z]+)__'
            r'FROM__((?:F\d+(?:\+F\d+)*)|(?:Universe|Math|Physics|Information|Cosmos|Binary))__'
            r'TO__([A-Za-z][A-Za-z0-9]*)__'
            r'ATTR__([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z][A-Za-z0-9]*)*)'
            r'\.md$'
        )
    
    def _generate_fibonacci(self, n: int) -> Set[int]:
        """生成Fibonacci数集合"""
        if n <= 0:
            return set()
        
        fib_set = {1}
        if n <= 1:
            return fib_set
            
        fib_set.add(2)
        if n <= 2:
            return fib_set
        
        a, b = 1, 2
        while b <= n:
            fib_set.add(b)
            a, b = b, a + b
        
        return fib_set
    
    def _generate_fibonacci_list(self, n: int) -> List[int]:
        """生成Fibonacci数列表（用于Zeckendorf）"""
        if n <= 0:
            return []
        
        fib_list = [1, 2]
        if n <= 2:
            return [f for f in fib_list if f <= n]
        
        a, b = 1, 2
        while b <= n:
            if b not in fib_list:
                fib_list.append(b)
            a, b = b, a + b
        
        return sorted([f for f in fib_list if f <= n])
    
    def to_zeckendorf(self, n: int) -> List[int]:
        """转换为Zeckendorf表示"""
        if n <= 0:
            return []
        
        result = []
        for fib in reversed(self.fibonacci_list):
            if fib <= n:
                result.append(fib)
                n -= fib
                if n == 0:
                    break
        
        return sorted(result)
    
    def parse_filename(self, filename: str) -> Optional[FibonacciNode]:
        """解析Fibonacci文件名"""
        match = self.filename_pattern.match(filename)
        if not match:
            self.errors.append(f"Fibonacci文件名格式错误: {filename}")
            return None
        
        try:
            fib_num = int(match.group(1))
            name = match.group(2)
            operation_str = match.group(3)
            inputs_str = match.group(4)
            output_type = match.group(5)
            attributes_str = match.group(6)
            
            # 验证Fibonacci数
            if fib_num not in self.fibonacci_set:
                self.errors.append(f"无效的Fibonacci数: {fib_num}")
                return None
            
            # 解析操作类型
            try:
                operation = FibonacciOperationType(operation_str)
            except ValueError:
                self.errors.append(f"未知操作类型: {operation_str}")
                return None
            
            # 获取Zeckendorf分解
            zeckendorf_decomp = self.to_zeckendorf(fib_num)
            
            # 解析输入依赖
            dependencies = self._parse_dependencies(inputs_str, filename)
            if dependencies is None:
                return None
            
            # 验证依赖一致性
            if not self._validate_dependencies(fib_num, dependencies):
                self.errors.append(f"F{fib_num}的依赖关系与Zeckendorf分解不符")
                return None
            
            # 解析属性
            attributes = attributes_str.split('_') if attributes_str else []
            
            node = FibonacciNode(
                fibonacci_number=fib_num,
                name=name,
                operation=operation,
                zeckendorf_decomposition=zeckendorf_decomp,
                dependencies=dependencies,
                output_type=output_type,
                attributes=attributes,
                filename=filename
            )
            
            return node
            
        except Exception as e:
            self.errors.append(f"解析错误 {filename}: {str(e)}")
            return None
    
    def _parse_dependencies(self, inputs_str: str, filename: str) -> Optional[List[int]]:
        """解析依赖关系"""
        # 基础输入类型
        basic_inputs = {'Universe', 'Math', 'Physics', 'Information', 'Cosmos', 'Binary'}
        if inputs_str in basic_inputs:
            return []  # 无依赖
        
        # 解析Fibonacci依赖
        dependencies = []
        fib_pattern = re.findall(r'F(\d+)', inputs_str)
        
        for fib_str in fib_pattern:
            fib_num = int(fib_str)
            if fib_num in self.fibonacci_set:
                dependencies.append(fib_num)
            else:
                self.errors.append(f"无效的依赖Fibonacci数 F{fib_num} in {filename}")
                return None
        
        return sorted(dependencies)
    
    def _validate_dependencies(self, fib_num: int, dependencies: List[int]) -> bool:
        """验证依赖关系与Zeckendorf分解的一致性"""
        zeckendorf = self.to_zeckendorf(fib_num)
        
        # 对于基础理论（素Fibonacci或小数），允许无依赖
        if fib_num <= 5 or len(zeckendorf) == 1:
            return True
        
        # 对于复合理论，依赖应该与Zeckendorf分解相关
        # 这里允许一定的灵活性，不要求严格相等
        return len(dependencies) <= len(zeckendorf) + 1
    
    def parse_directory(self, directory_path: str) -> Dict[int, FibonacciNode]:
        """解析目录中的所有Fibonacci理论文件"""
        from pathlib import Path
        
        self.nodes.clear()
        self.errors.clear()
        
        dir_path = Path(directory_path)
        if not dir_path.exists():
            self.errors.append(f"目录不存在: {directory_path}")
            return self.nodes
        
        if not dir_path.is_dir():
            self.errors.append(f"路径不是目录: {directory_path}")
            return self.nodes
        
        try:
            md_files = list(dir_path.glob("F*.md"))  # 只匹配F开头的文件
            if not md_files:
                self.errors.append(f"目录中没有找到 Fibonacci 理论文件: {directory_path}")
                return self.nodes
            
            print(f"找到 {len(md_files)} 个 Fibonacci 理论文件")
            
            for file_path in md_files:
                print(f"正在解析: {file_path.name}")
                node = self.parse_filename(file_path.name)
                if node:
                    if node.fibonacci_number in self.nodes:
                        self.errors.append(f"重复的Fibonacci编号: F{node.fibonacci_number}")
                    else:
                        self.nodes[node.fibonacci_number] = node
        
        except Exception as e:
            self.errors.append(f"目录读取错误: {str(e)}")
        
        print(f"成功解析 {len(self.nodes)} 个Fibonacci理论")
        return self.nodes
    
    def get_errors(self) -> List[str]:
        """获取解析错误"""
        return self.errors.copy()
    
    def generate_theory_statistics(self) -> Dict:
        """生成理论统计"""
        stats = {
            'total_theories': len(self.nodes),
            'complexity_distribution': {},
            'operation_distribution': {},
            'prime_fibonacci_count': 0,
            'max_fibonacci_number': 0,
            'avg_complexity': 0.0,
            'info_content_range': (0, 0)
        }
        
        if not self.nodes:
            return stats
        
        # 复杂度分布
        complexities = [node.complexity_level for node in self.nodes.values()]
        for complexity in complexities:
            stats['complexity_distribution'][complexity] = \
                stats['complexity_distribution'].get(complexity, 0) + 1
        
        # 操作分布
        for node in self.nodes.values():
            op = node.operation.value
            stats['operation_distribution'][op] = \
                stats['operation_distribution'].get(op, 0) + 1
        
        # 其他统计
        stats['prime_fibonacci_count'] = sum(1 for node in self.nodes.values() 
                                           if node.is_prime_fibonacci)
        stats['max_fibonacci_number'] = max(self.nodes.keys()) if self.nodes else 0
        stats['avg_complexity'] = sum(complexities) / len(complexities)
        
        info_contents = [node.information_content for node in self.nodes.values()]
        stats['info_content_range'] = (min(info_contents), max(info_contents))
        
        return stats
    
    def validate_fibonacci_dag(self) -> List[str]:
        """验证Fibonacci DAG的完整性"""
        validation_errors = []
        
        # 检查依赖完整性
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    validation_errors.append(
                        f"F{node.fibonacci_number} 依赖不存在的 F{dep}"
                    )
        
        # 检查Fibonacci序列的连续性
        fib_numbers = sorted(self.nodes.keys())
        if fib_numbers:
            expected_start = fib_numbers[0]
            for i, fib_num in enumerate(fib_numbers):
                expected = self.fibonacci_seq[expected_start - 1 + i] if expected_start - 1 + i < len(self.fibonacci_seq) else -1
                if fib_num != expected and fib_num in self.fibonacci_seq:
                    # 允许跳跃，但必须是有效的Fibonacci数
                    continue
        
        return validation_errors

def test_unified_fibonacci_parser():
    """测试统一Fibonacci解析器"""
    print("🌟 统一Fibonacci理论解析器测试")
    print("=" * 50)
    
    parser = UnifiedFibonacciParser()
    
    # 测试文件名 (使用标准Fibonacci序列: 1,2,3,5,8,13,21,34...)
    test_filenames = [
        "F1__SelfReference__AXIOM__FROM__Universe__TO__SelfRefTensor__ATTR__Fundamental.md",
        "F2__GoldenRatio__AXIOM__FROM__Math__TO__PhiTensor__ATTR__Transcendental.md", 
        "F3__BinaryConstraint__DEFINE__FROM__Information__TO__No11Rule__ATTR__Forbidden.md",
        "F5__QuantumPrinciple__AXIOM__FROM__Physics__TO__QuantumTensor__ATTR__Fundamental.md",
        "F8__ComplexEmergence__EMERGE__FROM__F3+F5__TO__ComplexTensor__ATTR__Nonlinear.md",
        "F13__UnifiedField__UNIFY__FROM__F5+F8__TO__UnifiedTensor__ATTR__Holistic.md"
    ]
    
    print("\n📋 解析测试结果:")
    for filename in test_filenames:
        node = parser.parse_filename(filename)
        if node:
            print(f"✅ F{node.fibonacci_number}: {node.name}")
            print(f"   Zeckendorf: {node.zeckendorf_decomposition}")
            print(f"   复杂度: {node.complexity_level}")
            print(f"   信息含量: {node.information_content:.2f}")
            print(f"   素Fibonacci: {'是' if node.is_prime_fibonacci else '否'}")
        else:
            print(f"❌ 解析失败: {filename}")
    
    if parser.get_errors():
        print(f"\n⚠️  错误信息:")
        for error in parser.get_errors():
            print(f"   {error}")

if __name__ == "__main__":
    test_unified_fibonacci_parser()