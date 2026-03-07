#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修复exception_utils导入路径
将绝对导入改为相对导入
"""

import re
from pathlib import Path
from datetime import datetime

def fix_exception_utils_imports(file_path, verbose=False):
    """修复单个文件中的exception_utils导入"""
    try:
        if verbose:
            print(f"  📄 处理文件: {file_path.name}", flush=True)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        modified = False
        modified_count = 0
        modified_line_numbers = []
        
        lines = content.split('\n')
        new_lines = []
        
        for idx, line in enumerate(lines, start=1):
            original_line = line
            changed = False
            
            # 修复绝对导入：from src.infrastructure.utils.exception_utils import ...
            if 'from src.infrastructure.utils.exception_utils' in line:
                # 确定是src/infrastructure/utils下的文件还是tests下的文件
                if 'src/infrastructure/utils' in str(file_path):
                    # 在同一目录下，使用相对导入
                    line = re.sub(
                        r'from src\.infrastructure\.utils\.exception_utils',
                        'from .exception_utils',
                        line
                    )
                else:
                    # 在tests下，使用绝对导入但去掉src前缀
                    line = re.sub(
                        r'from src\.infrastructure\.utils\.exception_utils',
                        'from infrastructure.utils.exception_utils',
                        line
                    )
                changed = True
            
            # 修复import语句中的绝对路径
            if 'import src.infrastructure.utils.exception_utils' in line:
                if 'src/infrastructure/utils' in str(file_path):
                    line = re.sub(
                        r'import src\.infrastructure\.utils\.exception_utils',
                        'from .exception_utils import *',
                        line
                    )
                else:
                    line = re.sub(
                        r'import src\.infrastructure\.utils\.exception_utils',
                        'from infrastructure.utils.exception_utils import *',
                        line
                    )
                changed = True
            
            if changed:
                modified = True
                modified_count += 1
                modified_line_numbers.append(idx)
                if verbose:
                    print(f"    ✏️  第{idx}行: {original_line.strip()[:60]}... → {line.strip()[:60]}...", flush=True)
            
            new_lines.append(line)
        
        if modified:
            content = '\n'.join(new_lines)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return (True, modified_count, None, modified_line_numbers)
        
        return (False, 0, None, [])
    except Exception as e:
        error_msg = f"修复 {file_path} 时出错: {e}"
        if verbose:
            print(f"    ❌ {error_msg}", flush=True)
        return (False, 0, error_msg, [])

def find_files_with_exception_utils_import():
    """查找所有包含exception_utils导入的文件"""
    files = []
    
    # 查找src目录
    for path in Path('src').rglob('*.py'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'src.infrastructure.utils.exception_utils' in content:
                    files.append(path)
        except:
            pass
    
    # 查找tests目录
    for path in Path('tests').rglob('*.py'):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'src.infrastructure.utils.exception_utils' in content:
                    files.append(path)
        except:
            pass
    
    return files

def main():
    """主函数"""
    print("=" * 80)
    print("🔧 批量修复exception_utils导入路径")
    print("=" * 80)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("💡 说明:")
    print("  修复所有使用绝对路径导入exception_utils的文件")
    print("  - src/infrastructure/utils下的文件: 改为相对导入")
    print("  - tests下的文件: 改为去掉src前缀的绝对导入")
    print()
    
    files = find_files_with_exception_utils_import()
    
    if not files:
        print("  ✅ 没有找到需要修复的文件")
        return
    
    print(f"📁 找到 {len(files)} 个需要修复的文件")
    print()
    
    fixed_count = 0
    error_count = 0
    total_modified_lines = 0
    
    for file_path in files:
        try:
            print(f"📄 处理: {file_path.relative_to(Path.cwd())}")
            modified, line_count, error, line_numbers = fix_exception_utils_imports(file_path, verbose=True)
            
            if error:
                error_count += 1
                print(f"    ❌ 处理失败: {error}")
            elif modified:
                fixed_count += 1
                total_modified_lines += line_count
                print(f"    ✅ 修复成功 (修改{line_count}行, 涉及行号: {line_numbers[:10]}{'...' if len(line_numbers) > 10 else ''})")
            else:
                print(f"    ⏭️  无需修改")
            
            print()
            
        except Exception as e:
            error_count += 1
            print(f"    ❌ 异常: {e}")
            print()
    
    print("=" * 80)
    print("📊 修复完成")
    print("=" * 80)
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"📁 总文件数: {len(files)}")
    print(f"✅ 修复文件数: {fixed_count}")
    print(f"📝 总修改行数: {total_modified_lines}")
    print(f"❌ 错误数: {error_count}")
    print()
    
    if fixed_count > 0:
        print(f"🎉 成功修复 {fixed_count} 个文件，共 {total_modified_lines} 行！")
    
    print("=" * 80)

if __name__ == '__main__':
    main()

