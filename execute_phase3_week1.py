#!/usr/bin/env python3
"""
Phase 3 Week 1: 代码格式化和基础修复执行脚本
"""

import re
import ast
from pathlib import Path
from datetime import datetime


def log(message):
    """打印日志"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")


def fix_whitespace_and_blank_lines(file_path):
    """修复空白字符和空行问题"""
    try:
        content = Path(file_path).read_text(encoding='utf-8')
        original_content = content
        
        # 修复W291: 去除行尾空格
        content = re.sub(r' +\n', '\n', content)
        
        # 修复W293: 去除空行中的空格
        content = re.sub(r'^[ \t]+$', '', content, flags=re.MULTILINE)
        
        # 修复W391: 文件末尾只保留一个换行
        content = content.rstrip() + '\n'
        
        # 修复E302: 类定义前应该有2个空行
        # 匹配不是以class开头的行后面紧跟class的情况
        content = re.sub(r'([^\n]\n)(class\s+\w+)', r'\1\n\n\2', content)
        
        # 修复E305: 类/函数结束后应该有2个空行
        # 这个比较复杂，暂时跳过
        
        if content != original_content:
            Path(file_path).write_text(content, encoding='utf-8')
            return True
        return False
    except Exception as e:
        log(f"❌ 修复失败 {file_path}: {e}")
        return False


def fix_long_lines(file_path, max_length=100):
    """修复行过长问题"""
    try:
        content = Path(file_path).read_text(encoding='utf-8')
        lines = content.split('\n')
        fixed_lines = []
        modified = False
        
        for line in lines:
            if len(line) > max_length:
                # 尝试智能换行
                fixed_line = smart_line_break(line, max_length)
                if fixed_line != line:
                    modified = True
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        
        if modified:
            Path(file_path).write_text('\n'.join(fixed_lines), encoding='utf-8')
            return True
        return False
    except Exception as e:
        log(f"❌ 修复失败 {file_path}: {e}")
        return False


def smart_line_break(line, max_length=100):
    """智能换行"""
    if len(line) <= max_length:
        return line
    
    # 如果是注释，在句号或逗号后换行
    if line.strip().startswith('#'):
        return break_comment_line(line, max_length)
    
    # 如果是字符串赋值，尝试在逗号后换行
    if '=' in line and ('"' in line or "'" in line):
        return break_string_line(line, max_length)
    
    # 如果是函数调用，尝试在参数间换行
    if '(' in line and ')' in line:
        return break_function_call(line, max_length)
    
    return line


def break_comment_line(line, max_length):
    """在注释中换行"""
    if len(line) <= max_length:
        return line
    
    # 找到合适的换行位置（空格或逗号后）
    break_point = max_length
    while break_point > max_length - 20 and break_point > 0:
        if line[break_point] in ' ,.;：':
            break
        break_point -= 1
    
    if break_point > 0:
        indent = len(line) - len(line.lstrip())
        first_part = line[:break_point].rstrip()
        second_part = ' ' * indent + '# ' + line[break_point:].lstrip()
        return first_part + '\n' + second_part
    
    return line


def break_string_line(line, max_length):
    """在字符串中换行"""
    # 暂时保持原样，避免破坏字符串
    return line


def break_function_call(line, max_length):
    """在函数调用中换行"""
    if len(line) <= max_length:
        return line
    
    # 简单处理：如果函数调用很长，尝试在多行显示
    match = re.match(r'^(\s*)(\w+)\((.*)\)(.*)$', line)
    if match:
        indent, func_name, args, suffix = match.groups()
        arg_list = [a.strip() for a in args.split(',')]
        
        if len(arg_list) > 2:  # 参数较多时才换行
            result = f"{indent}{func_name}(\n"
            for arg in arg_list:
                result += f"{indent}    {arg},\n"
            result += f"{indent}){suffix}"
            return result
    
    return line


def remove_unused_imports(file_path):
    """移除未使用的导入"""
    try:
        content = Path(file_path).read_text(encoding='utf-8')
        original_content = content
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return False
        
        # 收集所有导入
        imports = {}
        used_names = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    imports[name] = node
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    if name != '*':
                        imports[name] = node
            elif isinstance(node, ast.Name):
                used_names.add(node.id)
        
        # 找出未使用的导入
        unused = set(imports.keys()) - used_names
        
        # 移除未使用的导入（简单处理：注释掉）
        lines = content.split('\n')
        new_lines = []
        
        for line in lines:
            stripped = line.strip()
            should_keep = True
            
            for unused_name in unused:
                # 检查是否是导入未使用名称的行
                if stripped.startswith('import ') or stripped.startswith('from '):
                    if unused_name in stripped:
                        should_keep = False
                        break
            
            if should_keep:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
        
        if content != original_content:
            Path(file_path).write_text(content, encoding='utf-8')
            return True
        return False
    except Exception as e:
        log(f"❌ 修复失败 {file_path}: {e}")
        return False


def process_all_files():
    """处理所有文件"""
    src_dir = Path("src")
    
    stats = {
        'whitespace_fixed': 0,
        'long_lines_fixed': 0,
        'imports_fixed': 0,
        'total_files': 0,
        'error_files': []
    }
    
    log("="*70)
    log("Phase 3 Week 1: 代码格式化和基础修复")
    log("="*70)
    
    # 获取所有Python文件
    py_files = list(src_dir.rglob("*.py"))
    
    # 排除目录
    py_files = [f for f in py_files if not any(
        skip in str(f) for skip in ['backups', 'production_simulation', 'docs', 'reports', '__pycache__', '.git']
    )]
    
    stats['total_files'] = len(py_files)
    log(f"找到 {len(py_files)} 个Python文件")
    log("")
    
    # 步骤1: 修复空白字符和空行
    log("步骤1: 修复空白字符和空行问题 (W291, W293, W391, E302)...")
    for i, py_file in enumerate(py_files, 1):
        try:
            if fix_whitespace_and_blank_lines(py_file):
                stats['whitespace_fixed'] += 1
            if i % 50 == 0:
                log(f"  进度: {i}/{len(py_files)} 文件")
        except Exception as e:
            stats['error_files'].append(f"{py_file}: {e}")
    
    log(f"✅ 修复了 {stats['whitespace_fixed']} 个文件的空白字符问题")
    log("")
    
    # 步骤2: 修复行过长
    log("步骤2: 修复行过长问题 (E501)...")
    for i, py_file in enumerate(py_files, 1):
        try:
            if fix_long_lines(py_file):
                stats['long_lines_fixed'] += 1
            if i % 50 == 0:
                log(f"  进度: {i}/{len(py_files)} 文件")
        except Exception as e:
            stats['error_files'].append(f"{py_file}: {e}")
    
    log(f"✅ 修复了 {stats['long_lines_fixed']} 个文件的行过长问题")
    log("")
    
    # 步骤3: 移除未使用的导入
    log("步骤3: 移除未使用的导入 (F401)...")
    for i, py_file in enumerate(py_files, 1):
        try:
            if remove_unused_imports(py_file):
                stats['imports_fixed'] += 1
            if i % 50 == 0:
                log(f"  进度: {i}/{len(py_files)} 文件")
        except Exception as e:
            stats['error_files'].append(f"{py_file}: {e}")
    
    log(f"✅ 修复了 {stats['imports_fixed']} 个文件的导入问题")
    log("")
    
    # 显示统计
    log("="*70)
    log("Week 1 修复统计")
    log("="*70)
    log(f"总文件数: {stats['total_files']}")
    log(f"空白字符修复: {stats['whitespace_fixed']} 文件")
    log(f"行过长修复: {stats['long_lines_fixed']} 文件")
    log(f"导入优化: {stats['imports_fixed']} 文件")
    log(f"错误文件: {len(stats['error_files'])} 个")
    
    if stats['error_files']:
        log("\n错误详情 (前5个):")
        for error in stats['error_files'][:5]:
            log(f"  - {error}")
    
    log("="*70)
    log("Week 1 完成！")
    log("="*70)
    
    # 保存统计
    stats_file = Path('phase3_week1_stats.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("Phase 3 Week 1 修复统计\n")
        f.write("="*50 + "\n")
        f.write(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总文件数: {stats['total_files']}\n")
        f.write(f"空白字符修复: {stats['whitespace_fixed']} 文件\n")
        f.write(f"行过长修复: {stats['long_lines_fixed']} 文件\n")
        f.write(f"导入优化: {stats['imports_fixed']} 文件\n")
        f.write(f"错误文件: {len(stats['error_files'])} 个\n")
    
    log(f"\n统计已保存到: {stats_file}")


if __name__ == "__main__":
    process_all_files()
