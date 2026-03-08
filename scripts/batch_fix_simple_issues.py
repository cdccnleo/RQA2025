#!/usr/bin/env python3
"""
批量修复简单Flake8错误的脚本
修复: E501(行过长), W291(行尾空格), W293(空行空格), W391(文件末尾空行)
"""

import re
from pathlib import Path


def fix_long_line(line, max_length=100):
    """智能修复长行"""
    if len(line) <= max_length:
        return line
    
    # 如果是字符串，尝试在逗号或操作符处换行
    if '"' in line or "'" in line:
        # 尝试在逗号后换行
        parts = line.split(', ')
        if len(parts) > 1:
            result = parts[0] + ','
            current_length = len(result)
            
            for part in parts[1:]:
                if current_length + len(part) + 2 > max_length:
                    result += '\n' + ' ' * 8  # 缩进8个空格
                    current_length = 8
                result += ' ' + part + ','
                current_length += len(part) + 2
            
            return result.rstrip(',')
    
    # 如果是函数调用，尝试在参数间换行
    if '(' in line and ')' in line:
        match = re.match(r'(\s*)(\w+)\((.*)\)(.*)', line)
        if match:
            indent, func_name, args, suffix = match.groups()
            arg_list = args.split(', ')
            if len(arg_list) > 1:
                result = f"{indent}{func_name}(\n"
                for arg in arg_list:
                    result += f"{indent}    {arg},\n"
                result += f"{indent}){suffix}"
                return result
    
    return line


def fix_file(file_path):
    """修复单个文件"""
    try:
        content = Path(file_path).read_text(encoding='utf-8')
        original_content = content
        
        # 修复W291: 去除行尾空格
        content = re.sub(r' +\n', '\n', content)
        
        # 修复W293: 去除空行中的空格
        content = re.sub(r'^[ \t]+$', '', content, flags=re.MULTILINE)
        
        # 修复W391: 文件末尾只保留一个换行
        content = content.rstrip() + '\n'
        
        # 修复E501: 长行
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            if len(line) > 100:
                fixed_line = fix_long_line(line, 100)
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        
        content = '\n'.join(fixed_lines)
        
        # 如果内容有变化，写回文件
        if content != original_content:
            Path(file_path).write_text(content, encoding='utf-8')
            return True
        
        return False
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False


def main():
    """主函数"""
    src_dir = Path("src")
    fixed_count = 0
    error_files = []
    
    print("="*60)
    print("批量修复简单Flake8错误")
    print("="*60)
    
    # 遍历所有Python文件
    for py_file in src_dir.rglob("*.py"):
        # 跳过排除的目录
        if any(skip in str(py_file) for skip in ['backups', 'production_simulation', 'docs', 'reports', '__pycache__']):
            continue
        
        try:
            if fix_file(py_file):
                fixed_count += 1
                print(f"✅ 已修复: {py_file}")
        except Exception as e:
            error_files.append(f"{py_file}: {e}")
    
    print("\n" + "="*60)
    print(f"修复完成: {fixed_count} 个文件已修复")
    if error_files:
        print(f"错误: {len(error_files)} 个文件出错")
        for err in error_files[:5]:
            print(f"  - {err}")
    print("="*60)


if __name__ == "__main__":
    main()
