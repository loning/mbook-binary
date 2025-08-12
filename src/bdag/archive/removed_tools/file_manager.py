#!/usr/bin/env python3
"""
Batch File Operations Manager
批量管理和维护Fibonacci理论文件的工具
"""

import re
import os
import shutil
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import json

@dataclass
class FileOperation:
    """文件操作记录"""
    operation: str  # rename, move, update, create, delete
    old_path: str
    new_path: str
    status: str     # success, failed, skipped
    message: str

class FibonacciFileManager:
    """Fibonacci理论文件管理器"""
    
    def __init__(self, base_directory: str):
        self.base_dir = Path(base_directory)
        self.operations_log: List[FileOperation] = []
        
    def scan_theory_files(self) -> Dict[str, Dict]:
        """扫描所有理论文件"""
        files = {}
        
        if not self.base_dir.exists():
            return files
        
        # 扫描F开头的markdown文件
        for file_path in self.base_dir.glob("F*__*.md"):
            try:
                parsed = self._parse_filename(file_path.name)
                if parsed:
                    fib_num, name, operation, from_deps, to_output, attributes = parsed
                    files[file_path.name] = {
                        'path': str(file_path),
                        'fibonacci_number': fib_num,
                        'theory_name': name,
                        'operation': operation,
                        'from_dependencies': from_deps,
                        'to_output': to_output,
                        'attributes': attributes,
                        'file_size': file_path.stat().st_size,
                        'modified_time': file_path.stat().st_mtime
                    }
            except Exception as e:
                print(f"解析文件失败 {file_path.name}: {e}")
        
        return files
    
    def _parse_filename(self, filename: str) -> Optional[Tuple]:
        """解析文件名格式"""
        pattern = r'F(\d+)__(.+?)__(.+?)__FROM__(.+?)__TO__(.+?)__ATTR__(.+?)\.md'
        match = re.match(pattern, filename)
        
        if match:
            return (
                int(match.group(1)),  # fibonacci_number
                match.group(2),       # name
                match.group(3),       # operation
                match.group(4),       # from_dependencies
                match.group(5),       # to_output
                match.group(6)        # attributes
            )
        return None
    
    def _build_filename(self, fib_num: int, name: str, operation: str, 
                       from_deps: str, to_output: str, attributes: str) -> str:
        """构建标准文件名"""
        return f"F{fib_num}__{name}__{operation}__FROM__{from_deps}__TO__{to_output}__ATTR__{attributes}.md"
    
    def validate_filename_format(self, files: Dict[str, Dict]) -> List[str]:
        """验证文件名格式"""
        invalid_files = []
        
        for filename, info in files.items():
            # 重新构建文件名并比较
            expected_name = self._build_filename(
                info['fibonacci_number'],
                info['theory_name'],
                info['operation'],
                info['from_dependencies'],
                info['to_output'],
                info['attributes']
            )
            
            if filename != expected_name:
                invalid_files.append({
                    'current': filename,
                    'expected': expected_name,
                    'issues': self._identify_naming_issues(filename, expected_name)
                })
        
        return invalid_files
    
    def _identify_naming_issues(self, current: str, expected: str) -> List[str]:
        """识别命名问题"""
        issues = []
        
        # 提取各部分进行比较
        current_parts = current.split('__')
        expected_parts = expected.split('__')
        
        if len(current_parts) != len(expected_parts):
            issues.append("文件名段数不正确")
        else:
            part_names = ['Fibonacci编号', '理论名称', '操作类型', 'FROM依赖', 'TO输出', 'ATTR属性']
            for i, (curr, exp) in enumerate(zip(current_parts, expected_parts)):
                if curr != exp and i < len(part_names):
                    issues.append(f"{part_names[i]}不匹配: '{curr}' vs '{exp}'")
        
        return issues
    
    def standardize_filenames(self, files: Dict[str, Dict], dry_run: bool = True) -> List[FileOperation]:
        """标准化文件名"""
        operations = []
        
        for filename, info in files.items():
            old_path = Path(info['path'])
            
            # 构建标准文件名
            new_filename = self._build_filename(
                info['fibonacci_number'],
                info['theory_name'],
                info['operation'],
                info['from_dependencies'],
                info['to_output'],
                info['attributes']
            )
            
            new_path = old_path.parent / new_filename
            
            # 如果文件名已经标准
            if filename == new_filename:
                operation = FileOperation(
                    operation="rename",
                    old_path=str(old_path),
                    new_path=str(new_path),
                    status="skipped",
                    message="文件名已标准化"
                )
            else:
                if not dry_run:
                    try:
                        old_path.rename(new_path)
                        status = "success"
                        message = "文件名已标准化"
                    except Exception as e:
                        status = "failed"
                        message = f"重命名失败: {e}"
                else:
                    status = "pending"
                    message = "等待执行"
                
                operation = FileOperation(
                    operation="rename",
                    old_path=str(old_path),
                    new_path=str(new_path),
                    status=status,
                    message=message
                )
            
            operations.append(operation)
        
        self.operations_log.extend(operations)
        return operations
    
    def fix_common_naming_issues(self, files: Dict[str, Dict]) -> Dict[str, str]:
        """修复常见命名问题"""
        fixes = {}
        
        for filename, info in files.items():
            suggested_fixes = {}
            
            # 修复理论名称中的空格
            name = info['theory_name']
            if ' ' in name:
                suggested_fixes['theory_name'] = name.replace(' ', '')
            
            # 修复操作名称标准化
            operation = info['operation']
            operation_mapping = {
                'Define': 'DEFINE',
                'define': 'DEFINE',
                'Axiom': 'AXIOM', 
                'axiom': 'AXIOM',
                'Emerge': 'EMERGE',
                'emerge': 'EMERGE',
                'Combine': 'COMBINE',
                'combine': 'COMBINE'
            }
            if operation in operation_mapping:
                suggested_fixes['operation'] = operation_mapping[operation]
            
            # 修复属性格式
            attributes = info['attributes']
            if ' ' in attributes:
                suggested_fixes['attributes'] = attributes.replace(' ', '_')
            
            if suggested_fixes:
                fixes[filename] = suggested_fixes
        
        return fixes
    
    def backup_files(self, files: Dict[str, Dict], backup_dir: str = None) -> bool:
        """备份文件"""
        if backup_dir is None:
            backup_dir = str(self.base_dir / 'backup')
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)
        
        try:
            for filename, info in files.items():
                source = Path(info['path'])
                target = backup_path / filename
                shutil.copy2(source, target)
            
            # 创建备份清单
            manifest = {
                'backup_time': os.times(),
                'files': list(files.keys()),
                'total_files': len(files)
            }
            
            with open(backup_path / 'manifest.json', 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 备份完成: {len(files)}个文件 -> {backup_dir}")
            return True
            
        except Exception as e:
            print(f"❌ 备份失败: {e}")
            return False
    
    def generate_missing_theory_templates(self, missing_fibs: List[int]) -> List[str]:
        """为缺失的Fibonacci理论生成模板文件"""
        templates = []
        
        fibonacci_properties = {
            3: ("BinaryConstraint", "AXIOM", "Information", "No11Rule", "Forbidden_Pattern"),
            5: ("QuantumDiscretization", "AXIOM", "Physics", "QuantumTensor", "Fundamental_Discrete"),
            13: ("UnifiedField", "AXIOM", "Physics", "FieldTensor", "Unified_Continuous"),
            21: ("ConsciousEmergence", "EMERGE", "F8+F13", "ConsciousnessTensor", "Emergent_Subjective")
        }
        
        for fib_num in missing_fibs:
            if fib_num in fibonacci_properties:
                name, op, from_dep, to_out, attr = fibonacci_properties[fib_num]
                
                filename = self._build_filename(fib_num, name, op, from_dep, to_out, attr)
                
                template_content = f"""# F{fib_num} {name}

## 理论定义
**编号**: F{fib_num} (Fibonacci序列第{fib_num}位)  
**操作**: {op}  
**输入**: {from_dep}  
**输出**: {to_out}  

## 数学表示
[待补充数学公式]

## 基本性质
[待补充理论性质]

## 物理意义
[待补充物理解释]

## Fibonacci位置的意义
- **F{fib_num}**: [待补充]
- **Zeckendorf分解**: [待补充]
- **复杂度等级**: [待补充]
- **信息含量**: [待补充]

## 验证条件
[待补充验证标准]
"""
                
                templates.append({
                    'filename': filename,
                    'content': template_content,
                    'fibonacci_number': fib_num
                })
        
        return templates
    
    def print_file_report(self, files: Dict[str, Dict]):
        """打印文件管理报告"""
        print("📁 Fibonacci理论文件管理报告")
        print("=" * 50)
        
        if not files:
            print("❌ 未找到理论文件")
            return
        
        # 基本统计
        total_files = len(files)
        total_size = sum(info['file_size'] for info in files.values())
        
        print(f"\n📊 文件统计:")
        print(f"总文件数: {total_files}")
        print(f"总大小: {total_size/1024:.1f} KB")
        
        # Fibonacci数分布
        fib_numbers = [info['fibonacci_number'] for info in files.values()]
        fib_range = f"F{min(fib_numbers)} - F{max(fib_numbers)}" if fib_numbers else "无"
        print(f"Fibonacci范围: {fib_range}")
        
        # 操作类型分布
        operations = {}
        for info in files.values():
            op = info['operation']
            operations[op] = operations.get(op, 0) + 1
        
        print(f"\n🎭 操作类型分布:")
        for op, count in sorted(operations.items()):
            print(f"  {op}: {count}个")
        
        # 文件名格式验证
        invalid_files = self.validate_filename_format(files)
        if invalid_files:
            print(f"\n⚠️  文件名格式问题: {len(invalid_files)}个")
            for issue in invalid_files[:3]:  # 只显示前3个
                print(f"  {issue['current']}")
                for prob in issue['issues']:
                    print(f"    • {prob}")
        else:
            print(f"\n✅ 所有文件名格式正确")

def main():
    """演示文件管理器"""
    print("🗂️ Fibonacci理论文件管理器")
    print("=" * 50)
    
    examples_dir = Path(__file__).parent.parent / 'examples'
    
    if examples_dir.exists():
        manager = FibonacciFileManager(str(examples_dir))
        files = manager.scan_theory_files()
        manager.print_file_report(files)
        
        # 检查缺失的基础理论
        present_fibs = [info['fibonacci_number'] for info in files.values()]
        basic_fibs = [1, 2, 3, 5, 8, 13]
        missing_fibs = [f for f in basic_fibs if f not in present_fibs]
        
        if missing_fibs:
            print(f"\n📝 可生成缺失理论模板:")
            templates = manager.generate_missing_theory_templates(missing_fibs)
            for template in templates:
                print(f"  {template['filename']}")
        
        # 显示操作日志
        if manager.operations_log:
            print(f"\n📋 操作日志:")
            for op in manager.operations_log[-5:]:  # 显示最后5个操作
                print(f"  {op.operation}: {op.status} - {op.message}")
    else:
        print("❌ 未找到examples目录")

if __name__ == "__main__":
    main()