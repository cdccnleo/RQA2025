#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修复测试文件中的validate_dates导入问题

将测试文件中的：
- from src.infrastructure.utils.tools import validate_dates
- from infrastructure.utils.tools import validate_dates
替换为：
- 注释掉或移除validate_dates导入

功能：
- 扫描所有测试文件
- 识别并修复validate_dates导入错误
- 输出详细进度和日志
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime

def fix_validate_dates_import(file_path, verbose=False):
    """
    修复单个文件中的validate_dates导入
    
    Args:
        file_path: 文件路径
        verbose: 是否输出详细信息
    
    Returns:
        tuple: (是否修改, 修改的行数, 错误信息, 修改的行号列表)
    """
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
            
            # 修复各种形式的validate_dates导入
            # 1. from src.infrastructure.utils.tools import validate_dates
            # 2. from infrastructure.utils.tools import validate_dates  
            # 3. from .tools import validate_dates
            if re.search(r'from\s+(src\.)?infrastructure\.utils\.tools\s+import.*validate_dates', line, re.IGNORECASE):
                if 'validate_dates' in line:
                    # 检查是否有多个导入
                    import_part = line.split('import')[1] if 'import' in line else ''
                    if ',' in import_part:
                        # 多个导入，移除validate_dates
                        imports = import_part.strip()
                        import_list = [imp.strip() for imp in imports.split(',')]
                        import_list = [imp for imp in import_list if 'validate_dates' not in imp.lower()]
                        if import_list:
                            line = line.split('import')[0] + 'import ' + ', '.join(import_list)
                            changed = True
                        else:
                            # 所有导入都被移除了，注释整行
                            line = f"# {line}  # 已移除，validate_dates不在tools模块"
                            changed = True
                    else:
                        # 单独的validate_dates导入，直接注释
                        line = f"# {line}  # 已移除，validate_dates不在tools模块"
                        changed = True
            
            elif re.search(r'from\s+\.tools\s+import.*validate_dates', line, re.IGNORECASE):
                # 相对导入
                if 'validate_dates' in line:
                    import_part = line.split('import')[1] if 'import' in line else ''
                    if ',' in import_part:
                        imports = import_part.strip()
                        import_list = [imp.strip() for imp in imports.split(',')]
                        import_list = [imp for imp in import_list if 'validate_dates' not in imp.lower()]
                        if import_list:
                            line = line.split('import')[0] + 'import ' + ', '.join(import_list)
                            changed = True
                        else:
                            line = f"# {line}  # 已移除，validate_dates不在tools模块"
                            changed = True
                    else:
                        line = f"# {line}  # 已移除，validate_dates不在tools模块"
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
    print("🔧 批量修复validate_dates导入问题")
    print("=" * 80)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_dir = Path('tests/unit/infrastructure/utils')
    
    if not test_dir.exists():
        print(f"❌ 测试目录不存在: {test_dir}")
        sys.exit(1)
    
    print(f"📂 扫描目录: {test_dir.absolute()}")
    
    # 收集所有测试文件
    test_files = list(test_dir.rglob('test_*.py'))
    total_files = len(test_files)
    
    print(f"📊 找到 {total_files} 个测试文件")
    print("-" * 80)
    print()
    
    fixed_count = 0
    error_count = 0
    total_modified_lines = 0
    errors = []
    
    # 遍历所有测试文件
    for idx, test_file in enumerate(test_files, start=1):
        try:
            print(f"[{idx}/{total_files}] 处理: {test_file.relative_to(test_dir)}")
            
            modified, line_count, error, line_numbers = fix_validate_dates_import(test_file, verbose=True)
            
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
    
    # 输出总结
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

