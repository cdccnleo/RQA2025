#!/usr/bin/env python3
"""
代码格式自动修复脚本

修复常见的Flake8格式问题：
- 移除尾部空格
- 移除空白行中的空格
- 移除文件末尾的空行
"""

import os
import sys
from pathlib import Path


def fix_file_format(file_path: Path) -> tuple:
    """
    修复单个文件的格式问题
    
    Returns:
        tuple: (是否修改, 修复数量)
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        fixes = 0
        
        # 按行处理
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # 移除尾部空格
            if line.rstrip() != line:
                fixes += 1
            fixed_lines.append(line.rstrip())
        
        # 移除文件末尾的多余空行
        while fixed_lines and fixed_lines[-1] == '':
            fixed_lines.pop()
            fixes += 1
        
        # 重新组合（确保文件末尾有一个换行）
        new_content = '\n'.join(fixed_lines)
        if new_content and not new_content.endswith('\n'):
            new_content += '\n'
        
        # 保存（如果有修改）
        if new_content != original_content:
            file_path.write_text(new_content, encoding='utf-8')
            return True, fixes
        
        return False, 0
    
    except Exception as e:
        print(f"❌ 处理文件失败 {file_path}: {e}")
        return False, 0


def fix_directory(directory: str):
    """修复目录中所有Python文件的格式"""
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"❌ 目录不存在: {directory}")
        return
    
    print(f"\n🔧 开始修复目录: {directory}\n")
    print("="*80)
    
    total_files = 0
    modified_files = 0
    total_fixes = 0
    
    # 遍历所有Python文件
    for py_file in dir_path.rglob('*.py'):
        total_files += 1
        modified, fixes = fix_file_format(py_file)
        
        if modified:
            modified_files += 1
            total_fixes += fixes
            print(f"✅ {py_file.relative_to(dir_path)}: 修复{fixes}个问题")
    
    print("="*80)
    print(f"\n📊 修复统计:")
    print(f"  总文件数: {total_files}")
    print(f"  修改文件: {modified_files}")
    print(f"  总修复数: {total_fixes}")
    
    if modified_files > 0:
        print(f"\n✅ 格式修复完成！")
    else:
        print(f"\n✅ 无需修复，格式良好！")


if __name__ == '__main__':
    # 修复Task 1的组件
    print("\n" + "="*80)
    print("🎯 Phase 2 Week 1 - 代码格式自动修复")
    print("="*80)
    
    directories = [
        'src/core/business/optimizer/components',
        'src/core/business/optimizer/configs',
        'src/core/business/optimizer',
        'src/core/orchestration/components',
        'src/core/orchestration/models',
        'src/core/orchestration/configs'
    ]
    
    for directory in directories:
        if Path(directory).exists():
            fix_directory(directory)
    
    print("\n" + "="*80)
    print("🎉 所有目录格式修复完成！")
    print("="*80)
    print("\n建议下一步:")
    print("  1. 运行 flake8 检查验证")
    print("  2. 运行测试确保无破坏")
    print("  3. 提交代码")
    print()

