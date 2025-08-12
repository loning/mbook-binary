#!/usr/bin/env python3
"""
Theory Validation System
验证理论文件的Fibonacci依赖关系是否符合数学结构
"""

import re
import os
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class ValidationResult(Enum):
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"

@dataclass
class TheoryValidationReport:
    """理论验证报告"""
    theory_file: str
    fibonacci_number: int
    declared_dependencies: List[str]
    expected_dependencies: List[int]
    validation_result: ValidationResult
    issues: List[str]
    suggestions: List[str]

class FibonacciDependencyValidator:
    """Fibonacci依赖关系验证器"""
    
    def __init__(self, max_fibonacci: int = 100):
        self.max_fib = max_fibonacci
        self.fibonacci_sequence = self._generate_fibonacci_sequence()
        
    def _generate_fibonacci_sequence(self) -> List[int]:
        """生成Fibonacci序列"""
        fib = [1, 2]
        while fib[-1] < self.max_fib:
            next_fib = fib[-1] + fib[-2]
            if next_fib <= self.max_fib:
                fib.append(next_fib)
            else:
                break
        return fib
    
    def _to_zeckendorf(self, n: int) -> List[int]:
        """转换为Zeckendorf表示"""
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
    
    def _parse_theory_filename(self, filename: str) -> Optional[Tuple[int, str, str, str]]:
        """解析理论文件名
        返回: (fibonacci_number, theory_name, operation, from_dependencies)
        """
        pattern = r'F(\d+)__(.+?)__(.+?)__FROM__(.+?)__TO__'
        match = re.match(pattern, filename)
        
        if match:
            fib_num = int(match.group(1))
            theory_name = match.group(2)
            operation = match.group(3)
            from_deps = match.group(4)
            return fib_num, theory_name, operation, from_deps
        
        return None
    
    def _extract_dependencies_from_string(self, deps_string: str) -> List[str]:
        """从依赖字符串中提取依赖项"""
        # 处理各种可能的格式
        # F1+F3, F1__F3, Universe, Math等
        
        # 查找F数字模式
        fib_pattern = r'F(\d+)'
        fib_matches = re.findall(fib_pattern, deps_string)
        
        dependencies = []
        
        # 添加Fibonacci依赖
        for match in fib_matches:
            dependencies.append(f"F{match}")
        
        # 如果没有F依赖，可能是基础概念
        if not dependencies:
            # 基础概念如Universe, Math, Physics等
            base_concepts = ["Universe", "Math", "Physics", "Information", "Cosmos"]
            for concept in base_concepts:
                if concept in deps_string:
                    dependencies.append(concept)
        
        return dependencies
    
    def validate_theory_file(self, file_path: str) -> TheoryValidationReport:
        """验证单个理论文件"""
        filename = os.path.basename(file_path)
        
        # 解析文件名
        parsed = self._parse_theory_filename(filename)
        if not parsed:
            return TheoryValidationReport(
                theory_file=filename,
                fibonacci_number=-1,
                declared_dependencies=[],
                expected_dependencies=[],
                validation_result=ValidationResult.INVALID,
                issues=["无法解析文件名格式"],
                suggestions=["请使用标准Fibonacci理论文件命名格式"]
            )
        
        fib_num, theory_name, operation, from_deps = parsed
        
        # 提取声明的依赖
        declared_deps = self._extract_dependencies_from_string(from_deps)
        
        # 计算期望的依赖 (基于Zeckendorf分解)
        expected_deps = []
        if fib_num in self.fibonacci_sequence:
            zeckendorf = self._to_zeckendorf(fib_num)
            if len(zeckendorf) == 1:
                # 基础Fibonacci数，可以依赖基本概念
                expected_deps = []  # 允许依赖Universe, Math等
            else:
                # 复合Fibonacci数，应该依赖其Zeckendorf分解
                expected_deps = zeckendorf
        
        # 验证逻辑
        issues = []
        suggestions = []
        validation_result = ValidationResult.VALID
        
        # 检查Fibonacci数是否在序列中
        if fib_num not in self.fibonacci_sequence:
            issues.append(f"F{fib_num}不是有效的Fibonacci数")
            validation_result = ValidationResult.INVALID
        
        # 检查依赖关系
        if expected_deps:  # 复合Fibonacci数
            # 提取声明依赖中的Fibonacci数字
            declared_fib_nums = []
            for dep in declared_deps:
                if dep.startswith('F'):
                    try:
                        num = int(dep[1:])
                        declared_fib_nums.append(num)
                    except ValueError:
                        pass
            
            # 检查是否匹配Zeckendorf分解
            if set(declared_fib_nums) != set(expected_deps):
                issues.append(f"依赖关系不符合Zeckendorf分解")
                issues.append(f"声明依赖: {declared_fib_nums}")
                issues.append(f"期望依赖: {expected_deps}")
                suggestions.append(f"应该依赖: {[f'F{x}' for x in expected_deps]}")
                validation_result = ValidationResult.INVALID
        
        # 检查操作类型
        valid_operations = ["AXIOM", "DEFINE", "EMERGE", "COMBINE", "APPLY", "DERIVE"]
        if operation not in valid_operations:
            issues.append(f"未知操作类型: {operation}")
            suggestions.append(f"有效操作类型: {valid_operations}")
            validation_result = ValidationResult.WARNING
        
        # 基础Fibonacci数应该是AXIOM或DEFINE
        zeckendorf_decomp = self._to_zeckendorf(fib_num)
        if len(zeckendorf_decomp) == 1 and fib_num > 1:
            if operation not in ["AXIOM", "DEFINE"]:
                issues.append("基础Fibonacci数应该使用AXIOM或DEFINE操作")
                validation_result = ValidationResult.WARNING
        elif len(zeckendorf_decomp) > 1:
            # 复合Fibonacci数应该使用EMERGE或COMBINE
            if operation not in ["EMERGE", "COMBINE", "DERIVE"]:
                issues.append("复合Fibonacci数应该使用EMERGE、COMBINE或DERIVE操作")
                validation_result = ValidationResult.WARNING
        
        return TheoryValidationReport(
            theory_file=filename,
            fibonacci_number=fib_num,
            declared_dependencies=declared_deps,
            expected_dependencies=expected_deps,
            validation_result=validation_result,
            issues=issues,
            suggestions=suggestions
        )
    
    def validate_directory(self, directory_path: str) -> List[TheoryValidationReport]:
        """验证目录中的所有理论文件"""
        reports = []
        theory_dir = Path(directory_path)
        
        if not theory_dir.exists():
            print(f"目录不存在: {directory_path}")
            return reports
        
        # 查找所有理论文件
        for file_path in theory_dir.glob("F*__*.md"):
            report = self.validate_theory_file(str(file_path))
            reports.append(report)
        
        return reports
    
    def generate_validation_summary(self, reports: List[TheoryValidationReport]) -> Dict:
        """生成验证总结"""
        total = len(reports)
        valid = sum(1 for r in reports if r.validation_result == ValidationResult.VALID)
        invalid = sum(1 for r in reports if r.validation_result == ValidationResult.INVALID)
        warning = sum(1 for r in reports if r.validation_result == ValidationResult.WARNING)
        
        # 统计问题类型
        issue_types = {}
        for report in reports:
            for issue in report.issues:
                issue_types[issue] = issue_types.get(issue, 0) + 1
        
        return {
            "总文件数": total,
            "有效": valid,
            "无效": invalid,
            "警告": warning,
            "成功率": f"{(valid/total*100):.1f}%" if total > 0 else "0%",
            "常见问题": issue_types
        }
    
    def print_validation_report(self, reports: List[TheoryValidationReport]):
        """打印验证报告"""
        print("🔍 Fibonacci理论依赖关系验证报告")
        print("=" * 60)
        
        for report in reports:
            status_icon = {
                ValidationResult.VALID: "✅",
                ValidationResult.INVALID: "❌", 
                ValidationResult.WARNING: "⚠️"
            }[report.validation_result]
            
            print(f"\n{status_icon} {report.theory_file}")
            print(f"   Fibonacci数: F{report.fibonacci_number}")
            
            if report.expected_dependencies:
                print(f"   期望依赖: {[f'F{x}' for x in report.expected_dependencies]}")
            
            if report.declared_dependencies:
                print(f"   声明依赖: {report.declared_dependencies}")
            
            if report.issues:
                print("   问题:")
                for issue in report.issues:
                    print(f"     • {issue}")
            
            if report.suggestions:
                print("   建议:")
                for suggestion in report.suggestions:
                    print(f"     → {suggestion}")
        
        # 打印总结
        summary = self.generate_validation_summary(reports)
        print(f"\n📊 验证总结")
        print("-" * 30)
        for key, value in summary.items():
            if key != "常见问题":
                print(f"{key}: {value}")
        
        if summary["常见问题"]:
            print(f"\n🔥 常见问题:")
            for issue, count in sorted(summary["常见问题"].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {issue}: {count}次")

def main():
    """演示验证器功能"""
    print("🔍 Fibonacci理论依赖关系验证器")
    print("=" * 50)
    
    validator = FibonacciDependencyValidator()
    
    # 验证examples目录
    examples_dir = Path(__file__).parent.parent / 'examples'
    
    if examples_dir.exists():
        reports = validator.validate_directory(str(examples_dir))
        validator.print_validation_report(reports)
    else:
        print("未找到examples目录，创建测试示例...")
        
        # 测试示例
        test_files = [
            "F1__UniversalSelfReference__AXIOM__FROM__Universe__TO__SelfRefTensor__ATTR__Fundamental.md",
            "F2__GoldenRatioPrinciple__AXIOM__FROM__Math__TO__PhiTensor__ATTR__Transcendental.md", 
            "F8__ComplexEmergence__EMERGE__FROM__F3+F5__TO__ComplexTensor__ATTR__Nonlinear.md",
            "F4__InvalidExample__EMERGE__FROM__F1__TO__TimeTensor__ATTR__Wrong.md"  # 错误示例
        ]
        
        reports = []
        for filename in test_files:
            # 创建临时文件路径进行测试
            temp_path = f"/tmp/{filename}"
            report = validator.validate_theory_file(temp_path)
            reports.append(report)
        
        validator.print_validation_report(reports)

if __name__ == "__main__":
    main()