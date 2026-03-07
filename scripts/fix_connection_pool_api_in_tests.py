#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修复连接池测试中的API调用
将 get_connection 改为 acquire，put_connection 改为 release
"""

import re
from pathlib import Path
from datetime import datetime

def fix_connection_pool_api(file_path, verbose=False):
    """修复单个文件中的连接池API调用"""
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
            
            # 1. 将 pool.get_connection(...) 改为 pool.acquire(...)
            if 'pool.get_connection' in line:
                line = line.replace('pool.get_connection', 'pool.acquire')
                changed = True
            
            # 2. 将 pool.put_connection(...) 改为 pool.release(...)
            # 处理带timeout的情况：pool.put_connection(conn, timeout=0.1) -> pool.release(conn)
            if 'pool.put_connection' in line:
                # 先处理带timeout的情况
                line = re.sub(r'pool\.put_connection\s*\(\s*([^,)]+)\s*,\s*timeout\s*=[^)]+\)', r'pool.release(\1)', line)
                # 再处理不带timeout的情况
                line = line.replace('pool.put_connection', 'pool.release')
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

def main():
    """主函数"""
    print("=" * 80)
    print("🔧 批量修复连接池测试API调用")
    print("=" * 80)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("💡 说明:")
    print("  将测试中的连接池API调用统一：")
    print("  - pool.get_connection() → pool.acquire()")
    print("  - pool.put_connection() → pool.release()")
    print()
    
    test_files = [
        Path('tests/unit/infrastructure/utils/test_connection_pool_comprehensive.py'),
        Path('tests/unit/infrastructure/utils/test_connection_health_checker_edge_cases.py')
    ]
    
    fixed_count = 0
    error_count = 0
    total_modified_lines = 0
    
    for test_file in test_files:
        if not test_file.exists():
            print(f"  ⏭️  文件不存在: {test_file}")
            continue
        
        try:
            print(f"📄 处理: {test_file.name}")
            modified, line_count, error, line_numbers = fix_connection_pool_api(test_file, verbose=True)
            
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
    print(f"📁 总文件数: {len(test_files)}")
    print(f"✅ 修复文件数: {fixed_count}")
    print(f"📝 总修改行数: {total_modified_lines}")
    print(f"❌ 错误数: {error_count}")
    print()
    
    if fixed_count > 0:
        print(f"🎉 成功修复 {fixed_count} 个文件，共 {total_modified_lines} 行！")
    
    print("=" * 80)

if __name__ == '__main__':
    main()

