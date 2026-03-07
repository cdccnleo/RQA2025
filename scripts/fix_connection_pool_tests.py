#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修复连接池相关测试

将测试中的连接池API调用统一：
- 确保使用正确的方法名
- 处理异常类型不匹配问题
"""

import re
import sys
from pathlib import Path
from datetime import datetime

def fix_connection_pool_test(file_path, verbose=False):
    """修复单个文件中的连接池测试问题"""
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
            
            # 1. 修复异常类型：如果测试期望Empty但实际是RuntimeError
            # 检查是否有with pytest.raises(Empty)但实际可能抛出RuntimeError
            if 'pytest.raises(Empty)' in line and 'get_connection' in lines[idx:idx+3] if idx < len(lines) else False:
                # 保持Empty，因为我们已经修改了代码在timeout时抛出Empty
                pass
            
            # 2. 确保使用put_connection而不是直接调用release（如果测试中有）
            # 这个不需要修改，因为put_connection已经添加到ConnectionPool类中
            
            # 3. 其他可能的修复...
            
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
    print("🔧 分析连接池测试问题")
    print("=" * 80)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("💡 说明:")
    print("  连接池测试失败主要是因为API变更。")
    print("  已添加get_connection()和put_connection()方法到ConnectionPool类。")
    print("  如果仍有失败，可能需要进一步调整异常类型或测试逻辑。")
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
            print(f"📄 分析: {test_file.name}")
            modified, line_count, error, line_numbers = fix_connection_pool_test(test_file, verbose=True)
            
            if error:
                error_count += 1
                print(f"    ❌ 处理失败")
            elif modified:
                fixed_count += 1
                total_modified_lines += line_count
                print(f"    ✅ 修复成功 (修改{line_count}行)")
            else:
                print(f"    ⏭️  无需修改")
            
            print()
            
        except Exception as e:
            error_count += 1
            print(f"    ❌ 异常: {e}")
            print()
    
    print("=" * 80)
    print("📊 分析完成")
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

