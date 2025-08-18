#!/usr/bin/env python3
"""
MD文件行尾空格修复工具
修复Markdown文件的行尾格式，确保每行末尾有适当的空格
"""

import os
import re
from pathlib import Path

def fix_md_line_endings(file_path):
    """
    修复单个MD文件的行尾空格
    
    Args:
        file_path: MD文件路径
    
    Returns:
        bool: 是否进行了修改
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        lines = content.split('\n')
        modified_lines = []
        
        for line in lines:
            # 移除行尾的所有空格和制表符
            cleaned_line = line.rstrip(' \t')
            
            # 如果是空行，保持空行
            if not cleaned_line:
                modified_lines.append('')
            # 如果行以两个空格结尾（Markdown换行符），保持
            elif line.endswith('  '):
                modified_lines.append(cleaned_line + '  ')
            # 其他行末尾添加两个空格（Markdown换行）
            else:
                modified_lines.append(cleaned_line + '  ')
        
        new_content = '\n'.join(modified_lines)
        
        # 如果内容有变化，写回文件
        if new_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False

def find_md_files(directory):
    """
    递归查找目录中的所有MD文件
    
    Args:
        directory: 搜索目录路径
    
    Returns:
        list: MD文件路径列表
    """
    md_files = []
    directory_path = Path(directory)
    
    for file_path in directory_path.rglob('*.md'):
        md_files.append(str(file_path))
    
    return md_files

def main():
    """主函数"""
    # 设置要处理的目录
    base_dir = "/Users/cookie/mbook-binary-org/src/existencephilosophy"
    
    print("MD文件行尾空格修复工具")
    print("=" * 50)
    print(f"处理目录: {base_dir}")
    
    # 查找所有MD文件
    md_files = find_md_files(base_dir)
    
    if not md_files:
        print("未找到MD文件")
        return
    
    print(f"找到 {len(md_files)} 个MD文件")
    
    # 处理每个文件
    modified_count = 0
    for file_path in md_files:
        print(f"处理: {os.path.relpath(file_path, base_dir)}")
        
        if fix_md_line_endings(file_path):
            modified_count += 1
            print(f"  ✓ 已修改")
        else:
            print(f"  - 无需修改")
    
    print(f"\n完成! 共处理 {len(md_files)} 个文件，修改了 {modified_count} 个文件")

if __name__ == "__main__":
    main()