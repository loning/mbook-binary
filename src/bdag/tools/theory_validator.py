#!/usr/bin/env python3
"""
Theory Validation System v3.0
验证T{n}理论系统的完整性和一致性，包含素数理论支持
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
try:
    from .theory_parser import TheoryParser, TheoryNode, FibonacciOperationType
except ImportError:
    from theory_parser import TheoryParser, TheoryNode, FibonacciOperationType

class ValidationLevel(Enum):
    """验证级别"""
    CRITICAL = "critical"    # 严重错误，理论系统不可用
    ERROR = "error"         # 错误，影响理论正确性
    WARNING = "warning"     # 警告，可能的问题
    INFO = "info"          # 信息，建议改进

@dataclass
class ValidationIssue:
    """验证问题"""
    level: ValidationLevel
    category: str           # 问题类别
    theory_number: int      # 相关理论编号
    message: str           # 问题描述
    suggestion: Optional[str] = None  # 改进建议

@dataclass
class ValidationReport:
    """综合验证报告"""
    total_theories: int
    valid_theories: int
    critical_issues: List[ValidationIssue]
    errors: List[ValidationIssue]
    warnings: List[ValidationIssue]
    info: List[ValidationIssue]
    system_health: str      # 系统健康状态
    
    @property
    def all_issues(self) -> List[ValidationIssue]:
        """所有问题"""
        return self.critical_issues + self.errors + self.warnings + self.info
    
    @property
    def has_critical_issues(self) -> bool:
        """是否有严重问题"""
        return len(self.critical_issues) > 0
    
    @property
    def has_errors(self) -> bool:
        """是否有错误"""
        return len(self.errors) > 0

class PrimeChecker:
    """素数检测工具"""
    
    @staticmethod
    def is_prime(n: int) -> bool:
        """检查是否为素数"""
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
    
    @staticmethod
    def prime_factorize(n: int) -> List[Tuple[int, int]]:
        """素因子分解，返回[(素数, 幂次)]"""
        if n < 2:
            return []
        
        factors = []
        # 检查2的因子
        count = 0
        while n % 2 == 0:
            count += 1
            n //= 2
        if count > 0:
            factors.append((2, count))
        
        # 检查奇数因子
        i = 3
        while i * i <= n:
            count = 0
            while n % i == 0:
                count += 1
                n //= i
            if count > 0:
                factors.append((i, count))
            i += 2
        
        # 如果n本身是素数
        if n > 2:
            factors.append((n, 1))
        
        return factors
    
    @staticmethod
    def get_primes_up_to(n: int) -> List[int]:
        """获取n以内的所有素数"""
        if n < 2:
            return []
        
        # 埃拉托斯特尼筛法
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        
        return [i for i in range(2, n + 1) if sieve[i]]
    
    @staticmethod
    def is_twin_prime(n: int) -> bool:
        """检查是否为孪生素数"""
        if not PrimeChecker.is_prime(n):
            return False
        return PrimeChecker.is_prime(n - 2) or PrimeChecker.is_prime(n + 2)
    
    @staticmethod
    def is_mersenne_prime(n: int) -> bool:
        """检查是否为梅森素数 (2^p - 1形式)"""
        if not PrimeChecker.is_prime(n):
            return False
        
        # 检查n+1是否是2的幂
        m = n + 1
        return m > 0 and (m & (m - 1)) == 0
    
    @staticmethod
    def is_sophie_germain_prime(n: int) -> bool:
        """检查是否为Sophie Germain素数 (p和2p+1都是素数)"""
        if not PrimeChecker.is_prime(n):
            return False
        return PrimeChecker.is_prime(2 * n + 1)

class TheorySystemValidator:
    """T{n}理论系统验证器"""
    
    def __init__(self):
        self.parser = TheoryParser()
        self.prime_checker = PrimeChecker()
        self.issues: List[ValidationIssue] = []
    
    def validate_directory(self, directory_path: str) -> ValidationReport:
        """验证目录中的理论系统"""
        self.issues.clear()
        
        # 解析理论文件
        nodes = self.parser.parse_directory(directory_path)
        
        if not nodes:
            self.issues.append(ValidationIssue(
                level=ValidationLevel.CRITICAL,
                category="System",
                theory_number=0,
                message="未找到任何理论文件",
                suggestion="请确认目录路径正确且包含T*.md文件"
            ))
        
        # 基础验证
        self._validate_parser_errors()
        self._validate_axiom_system(nodes)
        self._validate_theory_completeness(nodes)
        self._validate_dependency_graph(nodes)
        self._validate_fibonacci_coverage(nodes)
        self._validate_operation_types(nodes)
        self._validate_prime_theories(nodes)  # 新增：素数理论验证
        
        return self._generate_report(nodes)
    
    def _validate_parser_errors(self):
        """验证解析错误"""
        for error in self.parser.errors:
            self.issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category="Parsing",
                theory_number=0,
                message=f"解析错误: {error}",
                suggestion="请检查文件名格式是否符合规范"
            ))
    
    def _validate_axiom_system(self, nodes: Dict[int, TheoryNode]):
        """验证单公理系统"""
        axiom_theories = [n for n in nodes.values() if n.operation == FibonacciOperationType.AXIOM]
        
        if len(axiom_theories) == 0:
            self.issues.append(ValidationIssue(
                level=ValidationLevel.CRITICAL,
                category="Axiom",
                theory_number=0,
                message="理论系统缺少公理基础",
                suggestion="应该有且仅有T1作为唯一公理"
            ))
        elif len(axiom_theories) > 1:
            theory_nums = [n.theory_number for n in axiom_theories]
            self.issues.append(ValidationIssue(
                level=ValidationLevel.CRITICAL,
                category="Axiom",
                theory_number=0,
                message=f"发现多个公理理论: T{theory_nums}",
                suggestion="单公理系统只应有T1作为唯一公理"
            ))
        elif axiom_theories[0].theory_number != 1:
            self.issues.append(ValidationIssue(
                level=ValidationLevel.CRITICAL,
                category="Axiom",
                theory_number=axiom_theories[0].theory_number,
                message=f"公理理论应为T1，实际为T{axiom_theories[0].theory_number}",
                suggestion="将T1设为AXIOM，其他理论改为THEOREM或EXTENDED"
            ))
        
        # 检查T1是否有依赖
        if 1 in nodes and nodes[1].theory_dependencies:
            self.issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category="Axiom",
                theory_number=1,
                message=f"公理T1不应有理论依赖，当前依赖: T{nodes[1].theory_dependencies}",
                suggestion="T1作为唯一公理应从Universe基础产生"
            ))
    
    def _validate_theory_completeness(self, nodes: Dict[int, TheoryNode]):
        """验证理论完整性"""
        max_theory = max(nodes.keys()) if nodes else 0
        
        # 检查Fibonacci数理论的覆盖
        fib_sequence = self.parser.fibonacci_sequence
        missing_fibs = []
        for fib in fib_sequence:
            if fib <= max_theory and fib not in nodes:
                missing_fibs.append(fib)
        
        if missing_fibs:
            self.issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category="Completeness",
                theory_number=0,
                message=f"缺少Fibonacci数理论: T{missing_fibs}",
                suggestion="考虑添加这些理论以完善Fibonacci覆盖"
            ))
        
        # 检查依赖完整性
        for theory_num, node in nodes.items():
            for dep in node.theory_dependencies:
                if dep not in nodes:
                    self.issues.append(ValidationIssue(
                        level=ValidationLevel.ERROR,
                        category="Dependency",
                        theory_number=theory_num,
                        message=f"依赖T{dep}不存在",
                        suggestion=f"添加T{dep}理论文件或修正T{theory_num}的依赖"
                    ))
    
    def _validate_dependency_graph(self, nodes: Dict[int, TheoryNode]):
        """验证依赖图结构"""
        # 检查循环依赖
        visited = set()
        rec_stack = set()
        
        def has_cycle(theory_num: int) -> bool:
            if theory_num not in nodes:
                return False
            
            visited.add(theory_num)
            rec_stack.add(theory_num)
            
            for dep in nodes[theory_num].theory_dependencies:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(theory_num)
            return False
        
        for theory_num in nodes.keys():
            if theory_num not in visited:
                if has_cycle(theory_num):
                    self.issues.append(ValidationIssue(
                        level=ValidationLevel.CRITICAL,
                        category="Dependency",
                        theory_number=theory_num,
                        message=f"检测到循环依赖，涉及T{theory_num}",
                        suggestion="检查并修正依赖关系，确保无环"
                    ))
    
    def _validate_fibonacci_coverage(self, nodes: Dict[int, TheoryNode]):
        """验证Fibonacci覆盖"""
        max_theory = max(nodes.keys()) if nodes else 0
        fib_theories = {n for n in nodes.keys() if n in self.parser.fibonacci_set and n <= max_theory}
        expected_fibs = {f for f in self.parser.fibonacci_sequence if f <= max_theory}
        
        coverage_rate = len(fib_theories) / len(expected_fibs) if expected_fibs else 0
        
        if coverage_rate < 0.8:
            self.issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category="Coverage",
                theory_number=0,
                message=f"Fibonacci理论覆盖率较低: {coverage_rate*100:.1f}%",
                suggestion="考虑增加更多Fibonacci数理论以提高系统完备性"
            ))
    
    def _validate_operation_types(self, nodes: Dict[int, TheoryNode]):
        """验证操作类型"""
        for theory_num, node in nodes.items():
            # 检查Fibonacci数理论的操作类型
            if node.is_fibonacci_theory:
                if theory_num == 1:
                    expected_op = FibonacciOperationType.AXIOM
                else:
                    expected_op = FibonacciOperationType.THEOREM
                
                if node.operation != expected_op:
                    self.issues.append(ValidationIssue(
                        level=ValidationLevel.ERROR,
                        category="Operation",
                        theory_number=theory_num,
                        message=f"T{theory_num}操作类型应为{expected_op.value}，实际为{node.operation.value}",
                        suggestion=f"Fibonacci数理论T{theory_num}应标记为{expected_op.value}"
                    ))
            
            # 检查复合理论的操作类型
            else:
                if node.operation == FibonacciOperationType.AXIOM:
                    self.issues.append(ValidationIssue(
                        level=ValidationLevel.ERROR,
                        category="Operation",
                        theory_number=theory_num,
                        message=f"复合理论T{theory_num}不应标记为AXIOM",
                        suggestion="复合理论应标记为EXTENDED"
                    ))
    
    def _validate_prime_theories(self, nodes: Dict[int, TheoryNode]):
        """验证素数理论的特殊性质"""
        max_theory = max(nodes.keys()) if nodes else 0
        
        # 获取所有素数
        primes = self.prime_checker.get_primes_up_to(max_theory)
        
        # 统计素数理论的覆盖情况
        prime_theories = [p for p in primes if p in nodes]
        prime_fib_theories = [p for p in prime_theories if p in self.parser.fibonacci_set]
        pure_prime_theories = [p for p in prime_theories if p not in self.parser.fibonacci_set]
        
        # 记录素数-Fibonacci双重理论
        if prime_fib_theories:
            self.issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                category="Prime",
                theory_number=0,
                message=f"发现{len(prime_fib_theories)}个素数-Fibonacci双重理论: T{prime_fib_theories}",
                suggestion="这些理论具有特殊的双重意义，建议重点关注"
            ))
        
        # 分析缺失的重要素数理论
        missing_primes = [p for p in primes[:20] if p not in nodes]  # 检查前20个素数
        if missing_primes:
            self.issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                category="Prime",
                theory_number=0,
                message=f"缺少重要素数位置的理论: T{missing_primes[:10]}",
                suggestion="考虑为这些素数位置开发独特的理论内容"
            ))
        
        # 检查特殊素数类
        for theory_num in prime_theories:
            special_types = []
            if self.prime_checker.is_twin_prime(theory_num):
                special_types.append("孪生素数")
            if self.prime_checker.is_mersenne_prime(theory_num):
                special_types.append("梅森素数")
            if self.prime_checker.is_sophie_germain_prime(theory_num):
                special_types.append("Sophie Germain素数")
            
            if special_types:
                self.issues.append(ValidationIssue(
                    level=ValidationLevel.INFO,
                    category="Prime",
                    theory_number=theory_num,
                    message=f"T{theory_num}是特殊素数理论: {', '.join(special_types)}",
                    suggestion="可以为这个理论添加相应的特殊性质"
                ))
        
        # 分析合数理论的素因子结构
        composite_theories = [n for n in nodes.keys() if n > 1 and not self.prime_checker.is_prime(n)]
        for theory_num in composite_theories[:10]:  # 只分析前10个作为示例
            factors = self.prime_checker.prime_factorize(theory_num)
            if factors:
                factor_str = ' × '.join([f"{p}^{e}" if e > 1 else str(p) for p, e in factors])
                prime_deps = [p for p, _ in factors if p in nodes]
                if prime_deps:
                    self.issues.append(ValidationIssue(
                        level=ValidationLevel.INFO,
                        category="Prime",
                        theory_number=theory_num,
                        message=f"T{theory_num} = {factor_str}，可考虑与素数理论T{prime_deps}的深层关联",
                        suggestion="合数理论可能继承其素因子理论的某些性质"
                    ))
    
    def _generate_report(self, nodes: Dict[int, TheoryNode]) -> ValidationReport:
        """生成验证报告"""
        # 按级别分类问题
        critical_issues = [i for i in self.issues if i.level == ValidationLevel.CRITICAL]
        errors = [i for i in self.issues if i.level == ValidationLevel.ERROR]
        warnings = [i for i in self.issues if i.level == ValidationLevel.WARNING]
        info = [i for i in self.issues if i.level == ValidationLevel.INFO]
        
        # 计算有效理论数
        valid_theories = sum(1 for n in nodes.values() if n.is_consistent)
        
        # 确定系统健康状态
        if critical_issues:
            system_health = "CRITICAL - 系统存在严重问题"
        elif errors:
            system_health = "ERROR - 系统存在错误需要修正"
        elif warnings:
            system_health = "WARNING - 系统基本正常但有改进空间"
        else:
            system_health = "HEALTHY - 系统完全健康"
        
        return ValidationReport(
            total_theories=len(nodes),
            valid_theories=valid_theories,
            critical_issues=critical_issues,
            errors=errors,
            warnings=warnings,
            info=info,
            system_health=system_health
        )
    
    def print_validation_report(self, report: ValidationReport):
        """打印验证报告"""
        print("🔍 T{n}理论系统验证报告")
        print("=" * 50)
        
        # 系统概览
        print(f"\n📊 系统状态: {report.system_health}")
        print(f"总理论数: {report.total_theories}")
        print(f"有效理论: {report.valid_theories}")
        print(f"一致性: {report.valid_theories}/{report.total_theories} ({report.valid_theories/report.total_theories*100:.1f}%)" if report.total_theories > 0 else "一致性: N/A")
        
        # 问题统计
        print(f"\n🚨 问题统计:")
        print(f"  严重问题: {len(report.critical_issues)}")
        print(f"  错误: {len(report.errors)}")
        print(f"  警告: {len(report.warnings)}")
        print(f"  信息: {len(report.info)}")
        
        # 详细问题列表
        if report.critical_issues:
            print(f"\n🔴 严重问题:")
            for issue in report.critical_issues:
                print(f"  • T{issue.theory_number}: {issue.message}")
                if issue.suggestion:
                    print(f"    建议: {issue.suggestion}")
        
        if report.errors:
            print(f"\n🟡 错误:")
            for issue in report.errors:
                print(f"  • T{issue.theory_number}: {issue.message}")
                if issue.suggestion:
                    print(f"    建议: {issue.suggestion}")
        
        if report.warnings:
            print(f"\n🔵 警告:")
            for issue in report.warnings[:5]:  # 只显示前5个警告
                print(f"  • T{issue.theory_number}: {issue.message}")
                if issue.suggestion:
                    print(f"    建议: {issue.suggestion}")
            if len(report.warnings) > 5:
                print(f"  ... 还有{len(report.warnings)-5}个警告")
        
        # 系统建议
        if not report.has_critical_issues and not report.has_errors:
            print(f"\n✅ 理论系统验证通过！")
        else:
            print(f"\n⚠️ 建议优先处理严重问题和错误，确保理论系统的数学一致性。")

def main():
    """测试验证器"""
    validator = TheorySystemValidator()
    
    # 验证examples目录
    examples_dir = Path(__file__).parent.parent / 'examples'
    if examples_dir.exists():
        print(f"验证目录: {examples_dir}")
        report = validator.validate_directory(str(examples_dir))
        validator.print_validation_report(report)
    else:
        print("examples目录不存在")

if __name__ == "__main__":
    main()