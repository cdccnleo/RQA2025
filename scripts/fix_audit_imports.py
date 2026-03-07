#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修复测试文件中的audit模块导入问题

将测试文件中的：
- from src.infrastructure.utils.audit import ...
- from infrastructure.utils.audit import ...
注释掉或修复
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime

def fix_audit_import(file_path, verbose=False):
    """修复单个文件中的audit导入"""
    try:
        if verbose:
            print(f"  📄 处理文件: {file_path.name}", flush=True)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        lines = content.split('\n')
        modified_lines = []
        modified = False
        modified_count = 0
        modified_line_numbers = []
        
        for idx, line in enumerate(lines, start=1):
            original_line = line
            changed = False
            
            # 修复各种形式的audit导入
            if re.search(r'from\s+(src\.)?infrastructure\.utils\.audit\s+import', line, re.IGNORECASE):
                line = f"# {line}  # 已移除，audit模块不在utils下"
                changed = True
            elif re.search(r'from\s+\.audit\s+import', line, re.IGNORECASE):
                line = f"# {line}  # 已移除，audit模块不在utils下"
                changed = True
            elif re.search(r'import\s+.*audit', line, re.IGNORECASE) and 'from' not in line:
                # 直接import audit的情况
                if 'audit' in line.lower() and '#' not in line:
                    line = f"# {line}  # 已移除，audit模块不在utils下"
                    changed = True
            
            if changed:
                modified = True
                modified_count += 1
                modified_line_numbers.append(idx)
                if verbose:
                    print(f"    ✏️  第{idx}行: {original_line.strip()[:60]}... → {line.strip()[:60]}...", flush=True)
            
            modified_lines.append(line)
        
        if modified:
            content = '\n'.join(modified_lines)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return (True, modified_count, None, modified_line_numbers)
        
        return (False, 0, None, [])
    except Exception as e:
        error_msg = f"修复 {file_path} 时出错: {e}"
        if verbose:
            print(f"    ❌ {error_msg}", flush=True)
        return (False, 0, error_msg, [])

def main():
    """主函数"""
    print("=" * 80)
    print("🔧 批量修复audit导入问题")
    print("=" * 80)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_dir = Path('tests/unit/infrastructure/utils')
    
    if not test_dir.exists():
        print(f"❌ 测试目录不存在: {test_dir}")
        sys.exit(1)
    
    print(f"📂 扫描目录: {test_dir.absolute()}")
    
    test_files = list(test_dir.rglob('test_*.py'))
    total_files = len(test_files)
    
    print(f"📊 找到 {total_files} 个测试文件")
    print("-" * 80)
    print()
    
    fixed_count = 0
    error_count = 0
    total_modified_lines = 0
    errors = []
    
    for idx, test_file in enumerate(test_files, start=1):
        try:
            print(f"[{idx}/{total_files}] 处理: {test_file.relative_to(test_dir)}")
            
            modified, line_count, error, line_numbers = fix_audit_import(test_file, verbose=True)
            
            if error:
                errors.append((test_file, error))
                error_count += 1
                print(f"    ❌ 处理失败")
            elif modified:
                fixed_count += 1
                total_modified_lines += line_count
                if line_numbers:
                    line_nums_str = ', '.join(map(str, line_numbers))
                    print(f"    ✅ 修复成功 (修改{line_count}行, 行号: {line_nums_str})")
                else:
                    print(f"    ✅ 修复成功 (修改{line_count}行)")
            else:
                print(f"    ⏭️  无需修改")
            
            print()
            
        except Exception as e:
            error_msg = f"{test_file} - {e}"
            errors.append((test_file, error_msg))
            error_count += 1
            print(f"    ❌ 异常: {e}")
            print()
    
    print("=" * 80)
    print("📊 修复完成统计")
    print("=" * 80)
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"📁 总文件数: {total_files}")
    print(f"✅ 修复文件数: {fixed_count}")
    print(f"📝 总修改行数: {total_modified_lines}")
    print(f"⏭️  无需修改: {total_files - fixed_count - error_count}")
    print(f"❌ 错误数: {error_count}")
    print()
    
    if errors:
        print("❌ 错误详情:")
        print("-" * 80)
        for file_path, error in errors:
            print(f"  • {file_path.relative_to(test_dir)}")
            print(f"    {error}")
        print()
    
    if fixed_count > 0:
        print(f"🎉 成功修复 {fixed_count} 个文件，共 {total_modified_lines} 行！")
    
    print("=" * 80)

if __name__ == '__main__':
    main()

