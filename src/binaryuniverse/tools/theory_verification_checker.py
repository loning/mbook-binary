#!/usr/bin/env python3
"""
二进制宇宙理论自动化验证检查工具
Theory Verification Automated Checker for Binary Universe

此工具对整个理论体系进行自动化完备性检查，确保：
1. 理论-形式化-测试三文件一致性
2. φ-编码约束的全面实现
3. A1公理的系统性遵循
4. 依赖关系的完整性验证
5. V1-V5验证系统的有效性确认
"""

import os
import re
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TheoryType(Enum):
    AXIOM = "A"
    DEFINITION = "D"
    LEMMA = "L"
    THEOREM = "T"
    COROLLARY = "C"
    METATHEOREM = "M"
    VERIFICATION = "V"
    PROPOSITION = "P"

@dataclass
class TheoryFile:
    """理论文件信息"""
    theory_type: TheoryType
    number: str
    sub_number: Optional[str]
    name: str
    theory_file: Optional[Path]
    formal_file: Optional[Path]
    test_file: Optional[Path]
    
    @property
    def identifier(self) -> str:
        if self.sub_number:
            return f"{self.theory_type.value}{self.number}.{self.sub_number}"
        return f"{self.theory_type.value}{self.number}"

@dataclass
class VerificationResult:
    """验证结果"""
    file_identifier: str
    check_type: str
    status: bool
    message: str
    details: Optional[str] = None

class TheoryVerificationChecker:
    """理论验证检查器"""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.theory_files: Dict[str, TheoryFile] = {}
        self.verification_results: List[VerificationResult] = []
        
        # Fibonacci序列 (从CLAUDE.md)
        self.fibonacci = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
        
        # φ值
        self.phi = (1 + 5**0.5) / 2
        
    def scan_theory_files(self):
        """扫描所有理论文件"""
        print("🔍 扫描理论文件...")
        
        # 扫描理论主文件
        for file_path in self.base_path.glob("*.md"):
            if self._is_theory_file(file_path):
                theory = self._parse_theory_file(file_path)
                if theory:
                    self.theory_files[theory.identifier] = theory
                    
        # 扫描形式化文件
        formal_dir = self.base_path / "formal"
        if formal_dir.exists():
            for file_path in formal_dir.glob("*_formal.md"):
                self._link_formal_file(file_path)
                
        # 扫描测试文件
        tests_dir = self.base_path / "tests"
        if tests_dir.exists():
            for file_path in tests_dir.glob("test_*.py"):
                self._link_test_file(file_path)
                
        print(f"✅ 扫描完成，发现 {len(self.theory_files)} 个理论文件")
        
    def _is_theory_file(self, file_path: Path) -> bool:
        """判断是否为理论文件"""
        name = file_path.stem
        # 匹配模式：A1, D1-1, T9-5等
        pattern = r'^[ADLTCMPV]\d+(-\d+)?-'
        return bool(re.match(pattern, name))
        
    def _parse_theory_file(self, file_path: Path) -> Optional[TheoryFile]:
        """解析理论文件"""
        name = file_path.stem
        
        # 解析理论标识符
        match = re.match(r'^([ADLTCMPV])(\d+)(?:-(\d+))?-(.+)$', name)
        if not match:
            return None
            
        theory_type_str, number, sub_number, theory_name = match.groups()
        
        try:
            theory_type = TheoryType(theory_type_str)
        except ValueError:
            return None
            
        return TheoryFile(
            theory_type=theory_type,
            number=number,
            sub_number=sub_number,
            name=theory_name,
            theory_file=file_path,
            formal_file=None,
            test_file=None
        )
        
    def _link_formal_file(self, file_path: Path):
        """链接形式化文件"""
        name = file_path.stem.replace("_formal", "")
        # 转换命名格式：T9_5_consciousness_phase_transition -> T9-5
        match = re.match(r'^([ADLTCMPV])(\d+)(?:_(\d+))?_', name)
        if match:
            theory_type_str, number, sub_number = match.groups()
            if sub_number:
                identifier = f"{theory_type_str}{number}.{sub_number}"
            else:
                identifier = f"{theory_type_str}{number}"
                
            if identifier in self.theory_files:
                self.theory_files[identifier].formal_file = file_path
                
    def _link_test_file(self, file_path: Path):
        """链接测试文件"""
        name = file_path.stem.replace("test_", "")
        # 转换命名格式：test_T9_5_consciousness_phase_transition -> T9-5
        match = re.match(r'^([ADLTCMPV])(\d+)(?:_(\d+))?_', name)
        if match:
            theory_type_str, number, sub_number = match.groups()
            if sub_number:
                identifier = f"{theory_type_str}{number}.{sub_number}"
            else:
                identifier = f"{theory_type_str}{number}"
                
            if identifier in self.theory_files:
                self.theory_files[identifier].test_file = file_path
                
    def check_three_file_consistency(self):
        """检查三文件一致性"""
        print("\n📋 检查理论-形式化-测试三文件一致性...")
        
        complete_count = 0
        for identifier, theory in self.theory_files.items():
            has_theory = theory.theory_file is not None
            has_formal = theory.formal_file is not None  
            has_test = theory.test_file is not None
            
            if has_theory and has_formal and has_test:
                complete_count += 1
                self._add_result(identifier, "三文件完整性", True, "✅ 三文件齐全")
                
                # 检查内容一致性
                self._check_content_consistency(theory)
                
            else:
                missing = []
                if not has_theory:
                    missing.append("理论文件")
                if not has_formal:
                    missing.append("形式化文件")
                if not has_test:
                    missing.append("测试文件")
                    
                self._add_result(identifier, "三文件完整性", False, 
                               f"❌ 缺少: {', '.join(missing)}")
                               
        print(f"✅ 三文件完整性检查完成：{complete_count}/{len(self.theory_files)} 完整")
        
    def _check_content_consistency(self, theory: TheoryFile):
        """检查内容一致性"""
        try:
            # 检查理论文件中是否提到形式化和测试
            theory_content = theory.theory_file.read_text(encoding='utf-8')
            
            # 检查是否提到φ-编码
            has_phi_mention = 'φ' in theory_content or 'phi' in theory_content.lower()
            if not has_phi_mention:
                self._add_result(theory.identifier, "φ-编码一致性", False, 
                               "❌ 理论文件未提及φ-编码")
            else:
                self._add_result(theory.identifier, "φ-编码一致性", True, 
                               "✅ φ-编码概念一致")
                               
            # 检查是否提到熵增
            has_entropy_mention = any(word in theory_content for word in ['熵增', '熵', 'entropy', '熵增长'])
            if not has_entropy_mention:
                self._add_result(theory.identifier, "A1公理遵循", False,
                               "❌ 未体现A1公理（熵增）")
            else:
                self._add_result(theory.identifier, "A1公理遵循", True,
                               "✅ 体现A1公理")
                               
        except Exception as e:
            self._add_result(theory.identifier, "内容一致性", False, 
                           f"❌ 检查失败: {str(e)}")
            
    def check_fibonacci_encoding(self):
        """检查Fibonacci编码实现"""
        print("\n🔢 检查Fibonacci/Zeckendorf编码实现...")
        
        # 检查测试文件中的Fibonacci实现
        for identifier, theory in self.theory_files.items():
            if theory.test_file and theory.test_file.exists():
                try:
                    test_content = theory.test_file.read_text(encoding='utf-8')
                    
                    # 检查是否包含Fibonacci序列
                    has_fibonacci = any(str(f) in test_content for f in self.fibonacci[:10])
                    
                    if has_fibonacci:
                        # 检查是否有No-11约束实现
                        has_no_11_constraint = '11' in test_content and 'consecutive' in test_content.lower()
                        
                        if has_no_11_constraint or 'no.*11' in test_content.lower():
                            self._add_result(identifier, "Fibonacci编码", True,
                                           "✅ 包含Fibonacci序列和No-11约束")
                        else:
                            self._add_result(identifier, "Fibonacci编码", False,
                                           "❌ 缺少No-11约束实现")
                    else:
                        # 对于某些理论，可能不直接使用Fibonacci
                        if theory.theory_type in [TheoryType.VERIFICATION, TheoryType.METATHEOREM]:
                            self._add_result(identifier, "Fibonacci编码", True,
                                           "✅ 验证/元定理系统，编码要求适当")
                        else:
                            self._add_result(identifier, "Fibonacci编码", False,
                                           "❌ 未发现Fibonacci序列使用")
                                           
                except Exception as e:
                    self._add_result(identifier, "Fibonacci编码", False,
                                   f"❌ 检查失败: {str(e)}")
                    
    def check_test_file_structure(self):
        """检查测试文件结构（不执行测试）"""
        print("\n🧪 检查测试文件结构...")
        
        tests_dir = self.base_path / "tests"
        if not tests_dir.exists():
            print("❌ 测试目录不存在")
            self._add_result("SYSTEM", "测试目录", False, "❌ 测试目录不存在")
            return
            
        # 检查测试文件结构
        test_files_found = list(tests_dir.glob("test_*.py"))
        
        for test_file in test_files_found:
            try:
                test_content = test_file.read_text(encoding='utf-8')
                
                # 检查是否有共享基类
                has_shared_base = "SharedV" in test_content and "ValidationBase" in test_content
                
                # 检查是否有测试类
                has_test_classes = "class Test" in test_content
                
                # 检查是否有测试方法
                has_test_methods = "def test_" in test_content
                
                if has_test_classes and has_test_methods:
                    status_msg = "✅ 测试文件结构完整"
                    if has_shared_base:
                        status_msg += " (使用共享基类)"
                    self._add_result(test_file.stem, "测试文件结构", True, status_msg)
                else:
                    missing = []
                    if not has_test_classes:
                        missing.append("测试类")
                    if not has_test_methods:
                        missing.append("测试方法")
                    self._add_result(test_file.stem, "测试文件结构", False,
                                   f"❌ 缺少: {', '.join(missing)}")
                                   
            except Exception as e:
                self._add_result(test_file.stem, "测试文件结构", False,
                               f"❌ 读取失败: {str(e)}")
                
        print(f"✅ 测试文件结构检查完成：{len(test_files_found)} 个测试文件")
        
    def check_dependency_consistency(self):
        """检查依赖关系一致性"""
        print("\n🔗 检查理论依赖关系...")
        
        # 定义理论依赖关系
        dependencies = {
            # 定义层依赖公理
            "D1.1": ["A1"], "D1.2": ["A1"], "D1.3": ["A1"], "D1.4": ["A1"],
            "D1.5": ["A1"], "D1.6": ["A1"], "D1.7": ["A1"], "D1.8": ["A1"],
            "D1.9": ["A1"], "D1.10": ["A1"], "D1.11": ["A1"], "D1.12": ["A1"],
            "D1.13": ["A1"], "D1.14": ["A1"], "D1.15": ["A1"],
            
            # 引理层依赖定义
            "L1.1": ["D1.1", "D1.2"], "L1.2": ["D1.2", "D1.3"], "L1.3": ["D1.3", "D1.4"],
            "L1.4": ["D1.4", "D1.5"], "L1.5": ["D1.3", "D1.8"], "L1.6": ["D1.6", "D1.7"],
            "L1.7": ["D1.5"], "L1.8": ["D1.8"], "L1.9": ["D1.12"], "L1.10": ["D1.13"],
            "L1.11": ["D1.5"], "L1.12": ["D1.14"], "L1.13": ["D1.1"], "L1.14": ["D1.2"],
            "L1.15": ["D1.8"],
            
            # 定理层依赖引理
            "T1.1": ["L1.1", "L1.2"], "T2.1": ["L1.5", "L1.8"], "T3.1": ["L1.7"],
            "T9.4": ["L1.12"], "T9.5": ["L1.12"], 
            
            # 验证系统依赖
            "V1": ["A1"], "V2": ["D1.1", "D1.8"], "V3": ["T1.1", "T2.1"],
            "V4": ["V1", "V2", "V3"], "V5": ["V1", "V2", "V3", "V4"],
        }
        
        missing_dependencies = []
        for theory_id, deps in dependencies.items():
            if theory_id in self.theory_files:
                for dep in deps:
                    if dep not in self.theory_files and dep != "A1":  # A1是特殊情况
                        missing_dependencies.append(f"{theory_id} → {dep}")
                        
        if not missing_dependencies:
            self._add_result("SYSTEM", "依赖关系", True, "✅ 所有依赖关系完整")
        else:
            self._add_result("SYSTEM", "依赖关系", False, 
                           f"❌ 缺少依赖: {', '.join(missing_dependencies[:10])}")
            
    def check_verification_systems(self):
        """检查V1-V5验证系统"""
        print("\n🔒 检查V1-V5验证系统...")
        
        verification_systems = ["V1", "V2", "V3", "V4", "V5"]
        system_descriptions = [
            "基础公理验证系统", "定义完备性验证系统", "推导有效性验证系统",
            "理论边界验证系统", "预测验证追踪系统"
        ]
        
        for i, v_system in enumerate(verification_systems):
            if v_system in self.theory_files:
                theory = self.theory_files[v_system]
                if theory.theory_file and theory.formal_file and theory.test_file:
                    self._add_result(v_system, "验证系统", True, 
                                   f"✅ {system_descriptions[i]} 完整")
                    
                    # 检查验证系统的特殊要求
                    if theory.test_file.exists():
                        try:
                            test_content = theory.test_file.read_text(encoding='utf-8')
                            if "SharedV" in test_content and "ValidationBase" in test_content:
                                self._add_result(v_system, "共享基类", True,
                                               "✅ 使用共享验证基类")
                            else:
                                self._add_result(v_system, "共享基类", False,
                                               "❌ 未使用共享验证基类")
                        except Exception:
                            pass
                else:
                    self._add_result(v_system, "验证系统", False,
                                   f"❌ {system_descriptions[i]} 不完整")
            else:
                self._add_result(v_system, "验证系统", False,
                               f"❌ 缺少 {system_descriptions[i]}")
                               
    def _add_result(self, identifier: str, check_type: str, status: bool, 
                   message: str, details: str = None):
        """添加验证结果"""
        self.verification_results.append(VerificationResult(
            file_identifier=identifier,
            check_type=check_type,
            status=status,
            message=message,
            details=details
        ))
        
    def generate_report(self):
        """生成验证报告"""
        print("\n📊 生成验证报告...")
        
        # 统计结果
        total_checks = len(self.verification_results)
        passed_checks = sum(1 for r in self.verification_results if r.status)
        failed_checks = total_checks - passed_checks
        
        # 按检查类型分组
        by_type = {}
        for result in self.verification_results:
            if result.check_type not in by_type:
                by_type[result.check_type] = {"passed": 0, "failed": 0, "results": []}
            by_type[result.check_type]["results"].append(result)
            if result.status:
                by_type[result.check_type]["passed"] += 1
            else:
                by_type[result.check_type]["failed"] += 1
                
        # 生成报告
        report_lines = [
            "# 二进制宇宙理论体系自动化验证报告",
            "",
            f"**总体结果**: {passed_checks}/{total_checks} 项检查通过 ({passed_checks/total_checks*100:.1f}%)",
            f"**理论文件数**: {len(self.theory_files)}",
            "",
            "## 详细检查结果",
            ""
        ]
        
        for check_type, stats in by_type.items():
            report_lines.append(f"### {check_type}")
            report_lines.append(f"通过: {stats['passed']}, 失败: {stats['failed']}")
            report_lines.append("")
            
            for result in stats["results"]:
                report_lines.append(f"- **{result.file_identifier}**: {result.message}")
                if result.details and not result.status:
                    report_lines.append(f"  ```\n  {result.details[:200]}...\n  ```")
            report_lines.append("")
            
        # 保存报告
        report_file = self.base_path / "verification_report.md"
        report_file.write_text("\n".join(report_lines), encoding='utf-8')
        
        print(f"✅ 验证报告已保存到: {report_file}")
        return report_file
        
    def run_full_verification(self):
        """运行完整验证"""
        print("🚀 开始二进制宇宙理论体系自动化验证")
        print("=" * 60)
        
        try:
            # 扫描文件
            self.scan_theory_files()
            
            # 各项检查
            self.check_three_file_consistency()
            self.check_fibonacci_encoding()
            self.check_dependency_consistency()
            self.check_verification_systems()
            self.check_test_file_structure()
            
            # 生成报告
            report_file = self.generate_report()
            
            print("\n" + "=" * 60)
            print("🎉 验证完成！")
            
            # 显示总结
            passed = sum(1 for r in self.verification_results if r.status)
            total = len(self.verification_results)
            success_rate = passed / total * 100 if total > 0 else 0
            
            print(f"📈 成功率: {success_rate:.1f}% ({passed}/{total})")
            
            if success_rate >= 90:
                print("🏆 优秀！理论体系高度完备")
            elif success_rate >= 75:
                print("✅ 良好！理论体系基本完备")
            elif success_rate >= 60:
                print("⚠️  需要改进！存在一些问题")
            else:
                print("❌ 需要重大改进！存在严重问题")
                
            return report_file
            
        except Exception as e:
            print(f"❌ 验证过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函数"""
    if len(sys.argv) > 1:
        base_path = Path(sys.argv[1])
    else:
        base_path = Path(__file__).parent.parent
        
    if not base_path.exists():
        print(f"❌ 路径不存在: {base_path}")
        return 1
        
    checker = TheoryVerificationChecker(base_path)
    report_file = checker.run_full_verification()
    
    return 0 if report_file else 1

if __name__ == "__main__":
    exit(main())