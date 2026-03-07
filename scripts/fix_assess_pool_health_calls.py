#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修复_assess_pool_health函数调用
将3参数调用改为4参数调用（添加connections参数）
"""

import re
from pathlib import Path
from datetime import datetime

def fix_assess_pool_health_calls(file_path, verbose=False):
    """修复单个文件中的_assess_pool_health调用"""
    try:
        if verbose:
            print(f"  📄 处理文件: {file_path.name}", flush=True)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        new_lines = []
        modified = False
        modified_count = 0
        modified_line_numbers = []
        
        for idx, line in enumerate(lines, start=1):
            original_line = line
            changed = False
            
            # 匹配模式：_assess_pool_health(available, active, max_size)
            # 需要改为：_assess_pool_health(connections, available, active, max_size)
            if '_assess_pool_health(' in line and line.count(',') == 2:
                # 提取参数
                match = re.search(r'_assess_pool_health\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\)', line)
                if match:
                    available_param = match.group(1).strip()
                    active_param = match.group(2).strip()
                    max_size_param = match.group(3).strip()
                    
                    # 根据第一个参数推断connections参数
                    # 如果是Queue，connections应该是对应的列表
                    if 'queue' in available_param.lower() or 'available' in available_param.lower():
                        # 从available_param推断connections
                        if 'empty_queue' in available_param:
                            connections_param = '[]'
                        elif 'available_queue' in available_param:
                            # 尝试从上下文找connections列表，否则创建新变量名
                            connections_param = f'_connections_from_{available_param}'
                            # 简化：直接使用列表字面量
                            connections_param = '[]'  # 临时方案
                        else:
                            connections_param = '[]'
                    else:
                        connections_param = '[]'
                    
                    # 如果有上下文，尝试找到connections变量
                    # 检查前几行是否有connections相关变量
                    context_lines = lines[max(0, idx-5):idx]
                    for ctx_line in reversed(context_lines):
                        if 'connections' in ctx_line.lower() and '=' in ctx_line:
                            # 提取变量名
                            var_match = re.search(r'(\w+)\s*=\s*', ctx_line)
                            if var_match:
                                connections_param = var_match.group(1)
                                break
                    
                    # 替换调用
                    new_call = f'_assess_pool_health({connections_param}, {available_param}, {active_param}, {max_size_param})'
                    line = re.sub(r'_assess_pool_health\s*\(\s*[^)]+\)', new_call, line)
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
    print("🔧 批量修复_assess_pool_health函数调用")
    print("=" * 80)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_file = Path('tests/unit/infrastructure/utils/test_connection_health_checker_edge_cases.py')
    
    if not test_file.exists():
        print(f"  ⏭️  文件不存在: {test_file}")
        return
    
    print(f"📄 处理: {test_file.name}")
    modified, line_count, error, line_numbers = fix_assess_pool_health_calls(test_file, verbose=True)
    
    if error:
        print(f"    ❌ 处理失败: {error}")
    elif modified:
        print(f"    ✅ 修复成功 (修改{line_count}行, 涉及行号: {line_numbers[:10]}{'...' if len(line_numbers) > 10 else ''})")
    else:
        print(f"    ⏭️  无需修改")
    
    print()
    print("=" * 80)
    print("📊 修复完成")
    print("=" * 80)
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

if __name__ == '__main__':
    main()

