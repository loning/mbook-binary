#!/usr/bin/env python3
"""
Theory Consistency Checker
验证理论体系是否遵循A1公理和φ-编码约束
"""

import re
import os
from typing import List, Dict, Tuple, Set, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import math

class ConsistencyLevel(Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"

@dataclass
class ConsistencyReport:
    """一致性检查报告"""
    check_name: str
    level: ConsistencyLevel
    message: str
    details: List[str] = None
    affected_files: List[str] = None

class TheoryConsistencyChecker:
    """理论一致性检查器"""
    
    def __init__(self, theory_directory: str):
        self.theory_dir = Path(theory_directory)
        self.fibonacci_sequence = self._generate_fibonacci_sequence(1000)
        self.phi = (1 + math.sqrt(5)) / 2
        self.reports: List[ConsistencyReport] = []
    
    def _generate_fibonacci_sequence(self, max_fib: int) -> List[int]:
        """生成Fibonacci序列"""
        fib = [1, 2]
        while fib[-1] < max_fib:
            next_fib = fib[-1] + fib[-2]
            if next_fib <= max_fib:
                fib.append(next_fib)
            else:
                break
        return fib
    
    def _has_consecutive_ones_in_binary(self, n: int) -> bool:
        """检查数字的二进制表示是否有连续的1（违反No-11约束）"""
        binary = bin(n)[2:]  # 去掉'0b'前缀
        return '11' in binary
    
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
    
    def _parse_theory_files(self) -> Dict[str, Dict]:
        """解析所有理论文件"""
        theories = {}
        
        if not self.theory_dir.exists():
            return theories
        
        for file_path in self.theory_dir.glob("F*__*.md"):
            # 解析文件名
            pattern = r'F(\d+)__(.+?)__(.+?)__FROM__(.+?)__TO__(.+?)__ATTR__(.+?)\.md'
            match = re.match(pattern, file_path.name)
            
            if match:
                fib_num = int(match.group(1))
                name = match.group(2)
                operation = match.group(3)
                from_deps = match.group(4)
                to_output = match.group(5)
                attributes = match.group(6)
                
                theories[f"F{fib_num}"] = {
                    'fibonacci_number': fib_num,
                    'name': name,
                    'operation': operation,
                    'dependencies': from_deps,
                    'output': to_output,
                    'attributes': attributes,
                    'file_path': str(file_path),
                    'file_name': file_path.name
                }
        
        return theories
    
    def check_a1_axiom_compliance(self, theories: Dict[str, Dict]) -> ConsistencyReport:
        """检查A1公理合规性：自指完备的系统必然熵增"""
        
        issues = []
        warnings = []
        
        # 检查是否有F1理论（自指基础）
        if "F1" not in theories:
            issues.append("缺少F1理论 - 自指完备的基础")
        else:
            f1_theory = theories["F1"]
            # 检查F1是否涉及自指
            if "Self" not in f1_theory['name'] and "Ref" not in f1_theory['name']:
                warnings.append("F1理论名称未明确体现自指概念")
        
        # 检查理论是否体现熵增特性
        entropy_indicators = ["Entropy", "Increase", "Complex", "Emerge", "Evolution"]
        entropy_theories = []
        
        for theory_id, theory in theories.items():
            for indicator in entropy_indicators:
                if indicator.lower() in theory['name'].lower() or indicator.lower() in theory['attributes'].lower():
                    entropy_theories.append(theory_id)
                    break
        
        if not entropy_theories:
            issues.append("没有找到明确的熵增理论")
        
        # 检查自指循环
        self_referential_theories = []
        for theory_id, theory in theories.items():
            if theory_id in theory['dependencies']:
                self_referential_theories.append(theory_id)
        
        if self_referential_theories:
            warnings.append(f"发现自指循环: {self_referential_theories}")
        
        level = ConsistencyLevel.FAIL if issues else (ConsistencyLevel.WARNING if warnings else ConsistencyLevel.PASS)
        
        return ConsistencyReport(
            check_name="A1公理合规性",
            level=level,
            message=f"自指完备系统熵增原理检查",
            details=issues + warnings,
            affected_files=[theories[tid]['file_name'] for tid in entropy_theories[:5]]  # 最多显示5个
        )
    
    def check_no11_constraint(self, theories: Dict[str, Dict]) -> ConsistencyReport:
        """检查No-11约束：二进制中不能有连续的11"""
        
        violations = []
        
        for theory_id, theory in theories.items():
            fib_num = theory['fibonacci_number']
            
            if self._has_consecutive_ones_in_binary(fib_num):
                binary = bin(fib_num)[2:]
                violations.append({
                    'theory': theory_id,
                    'number': fib_num,
                    'binary': binary,
                    'file': theory['file_name']
                })
        
        level = ConsistencyLevel.FAIL if violations else ConsistencyLevel.PASS
        details = [f"{v['theory']} (F{v['number']}) = {v['binary']}b 包含连续11" for v in violations]
        
        return ConsistencyReport(
            check_name="No-11约束",
            level=level,
            message=f"二进制连续11检查",
            details=details,
            affected_files=[v['file'] for v in violations]
        )
    
    def check_phi_encoding_consistency(self, theories: Dict[str, Dict]) -> ConsistencyReport:
        """检查φ-编码一致性"""
        
        issues = []
        warnings = []
        
        # 检查是否有F2理论（φ原理）
        if "F2" not in theories:
            issues.append("缺少F2理论 - φ比例原理")
        else:
            f2_theory = theories["F2"]
            phi_indicators = ["Golden", "Phi", "Ratio", "φ"]
            has_phi = any(indicator in f2_theory['name'] for indicator in phi_indicators)
            if not has_phi:
                warnings.append("F2理论未明确体现黄金比例概念")
        
        # 检查Fibonacci数序列的φ收敛性
        convergence_ratios = []
        for i in range(2, min(len(self.fibonacci_sequence), 10)):
            ratio = self.fibonacci_sequence[i] / self.fibonacci_sequence[i-1]
            convergence_ratios.append(ratio)
            
            # 检查是否有对应的理论
            fib_num = self.fibonacci_sequence[i]
            theory_id = f"F{fib_num}"
            if theory_id in theories:
                expected_phi_power = math.log(fib_num) / math.log(self.phi)
                # 这里可以添加更多φ相关的检查
        
        # 检查信息含量的φ量化
        for theory_id, theory in theories.items():
            fib_num = theory['fibonacci_number']
            if fib_num in self.fibonacci_sequence and fib_num > 1:
                info_content = math.log(fib_num) / math.log(self.phi)
                # 可以检查理论复杂度是否与信息含量匹配
        
        level = ConsistencyLevel.FAIL if issues else (ConsistencyLevel.WARNING if warnings else ConsistencyLevel.PASS)
        
        return ConsistencyReport(
            check_name="φ-编码一致性",
            level=level,
            message="黄金比例编码原理检查",
            details=issues + warnings + [f"φ收敛验证: 最后5项比值 = {convergence_ratios[-5:]}"]
        )
    
    def check_zeckendorf_dependency_consistency(self, theories: Dict[str, Dict]) -> ConsistencyReport:
        """检查Zeckendorf分解依赖一致性"""
        
        inconsistencies = []
        missing_deps = []
        
        for theory_id, theory in theories.items():
            fib_num = theory['fibonacci_number']
            
            if fib_num not in self.fibonacci_sequence:
                continue
            
            # 获取Zeckendorf分解
            zeckendorf = self._to_zeckendorf(fib_num)
            
            # 如果是复合Fibonacci数，检查依赖
            if len(zeckendorf) > 1:
                expected_deps = [f"F{x}" for x in zeckendorf if x != fib_num]
                
                # 从依赖字符串中提取F数字
                dep_pattern = r'F(\d+)'
                declared_deps = re.findall(dep_pattern, theory['dependencies'])
                declared_deps = [f"F{x}" for x in declared_deps]
                
                if set(declared_deps) != set(expected_deps):
                    inconsistencies.append({
                        'theory': theory_id,
                        'expected': expected_deps,
                        'declared': declared_deps,
                        'zeckendorf': zeckendorf
                    })
                
                # 检查依赖的理论是否存在
                for dep in expected_deps:
                    if dep not in theories:
                        missing_deps.append({
                            'missing': dep,
                            'required_by': theory_id
                        })
        
        details = []
        for inc in inconsistencies:
            details.append(f"{inc['theory']}: 期望{inc['expected']}, 声明{inc['declared']}")
        
        for miss in missing_deps:
            details.append(f"缺少依赖: {miss['missing']} (被{miss['required_by']}需要)")
        
        level = ConsistencyLevel.FAIL if inconsistencies or missing_deps else ConsistencyLevel.PASS
        
        return ConsistencyReport(
            check_name="Zeckendorf依赖一致性",
            level=level,
            message="数学依赖关系检查",
            details=details
        )
    
    def check_fibonacci_completeness(self, theories: Dict[str, Dict]) -> ConsistencyReport:
        """检查Fibonacci序列完整性"""
        
        present_numbers = set()
        for theory_id, theory in theories.items():
            fib_num = theory['fibonacci_number']
            if fib_num in self.fibonacci_sequence:
                present_numbers.add(fib_num)
        
        # 检查前几个基础Fibonacci数
        basic_fibs = [1, 2, 3, 5, 8, 13]
        missing_basic = [f"F{f}" for f in basic_fibs if f not in present_numbers]
        
        # 检查序列连续性
        max_present = max(present_numbers) if present_numbers else 0
        expected_sequence = [f for f in self.fibonacci_sequence if f <= max_present]
        missing_in_sequence = [f"F{f}" for f in expected_sequence if f not in present_numbers]
        
        details = []
        if missing_basic:
            details.append(f"缺少基础Fibonacci理论: {missing_basic}")
        
        if missing_in_sequence:
            details.append(f"序列不连续，缺少: {missing_in_sequence}")
        
        coverage = len(present_numbers) / len(expected_sequence) * 100 if expected_sequence else 0
        details.append(f"Fibonacci序列覆盖率: {coverage:.1f}%")
        
        level = ConsistencyLevel.WARNING if missing_basic else ConsistencyLevel.PASS
        
        return ConsistencyReport(
            check_name="Fibonacci完整性",
            level=level,
            message="Fibonacci序列覆盖度检查",
            details=details
        )
    
    def run_all_checks(self) -> List[ConsistencyReport]:
        """运行所有一致性检查"""
        
        print("🔍 理论一致性检查中...")
        theories = self._parse_theory_files()
        
        if not theories:
            return [ConsistencyReport(
                check_name="文件解析",
                level=ConsistencyLevel.FAIL,
                message="无法找到或解析理论文件"
            )]
        
        self.reports = [
            self.check_a1_axiom_compliance(theories),
            self.check_no11_constraint(theories),
            self.check_phi_encoding_consistency(theories),
            self.check_zeckendorf_dependency_consistency(theories),
            self.check_fibonacci_completeness(theories)
        ]
        
        return self.reports
    
    def print_consistency_report(self):
        """打印完整性报告"""
        print("📋 Fibonacci理论体系一致性报告")
        print("=" * 60)
        
        # 统计
        pass_count = sum(1 for r in self.reports if r.level == ConsistencyLevel.PASS)
        warning_count = sum(1 for r in self.reports if r.level == ConsistencyLevel.WARNING)
        fail_count = sum(1 for r in self.reports if r.level == ConsistencyLevel.FAIL)
        
        print(f"\n📊 检查结果统计:")
        print(f"✅ 通过: {pass_count}")
        print(f"⚠️  警告: {warning_count}")
        print(f"❌ 失败: {fail_count}")
        
        total_score = (pass_count * 100 + warning_count * 50) / (len(self.reports) * 100) * 100
        print(f"🎯 总体评分: {total_score:.1f}%")
        
        # 详细报告
        print(f"\n🔍 详细检查结果:")
        print("-" * 40)
        
        for report in self.reports:
            icon = {"pass": "✅", "warning": "⚠️", "fail": "❌"}[report.level.value]
            print(f"\n{icon} {report.check_name}")
            print(f"   {report.message}")
            
            if report.details:
                for detail in report.details:
                    print(f"   • {detail}")
            
            if report.affected_files:
                files_str = ", ".join(report.affected_files[:3])
                if len(report.affected_files) > 3:
                    files_str += f" (+{len(report.affected_files)-3}个)"
                print(f"   📁 相关文件: {files_str}")

def main():
    """演示一致性检查器"""
    print("🔧 Fibonacci理论一致性检查器")
    print("=" * 50)
    
    # 检查examples目录
    examples_dir = Path(__file__).parent.parent / 'examples'
    
    if examples_dir.exists():
        checker = TheoryConsistencyChecker(str(examples_dir))
        checker.run_all_checks()
        checker.print_consistency_report()
    else:
        print("❌ 未找到examples目录")

if __name__ == "__main__":
    main()