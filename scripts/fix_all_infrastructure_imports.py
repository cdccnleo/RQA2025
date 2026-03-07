#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Infrastructure模块导入路径批量修复脚本
修复所有 'from infrastructure.' 改为 'from src.infrastructure.'
"""

import os
import re
from pathlib import Path


def fix_imports_in_file(file_path):
    """修复单个文件的导入"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复模式：将 'from infrastructure.' 改为 'from src.infrastructure.'
        # 但不修改已经有src前缀的
        pattern = r'^from infrastructure\.'
        replacement = r'from src.infrastructure.'
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        # 修复 import infrastructure.
        pattern2 = r'^import infrastructure\.'
        replacement2 = r'import src.infrastructure.'
        content = re.sub(pattern2, replacement2, content, flags=re.MULTILINE)
        
        if content != original_content:
            # 备份
            backup_path = str(file_path) + '.bak'
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            # 保存修改
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
        
        return False
    
    except Exception as e:
        print(f"❌ 错误: {file_path} - {e}")
        return None


def main():
    """主函数"""
    print("=" * 80)
    print("Infrastructure模块导入路径批量修复")
    print("=" * 80)
    print()
    
    # 查找所有src/infrastructure下的Python文件
    src_dir = Path('src/infrastructure')
    py_files = list(src_dir.rglob('*.py'))
    
    print(f"找到 {len(py_files)} 个Python文件")
    print()
    
    fixed_count = 0
    skipped_count = 0
    error_count = 0
    
    for file_path in py_files:
        result = fix_imports_in_file(file_path)
        
        if result is True:
            fixed_count += 1
            print(f"✅ 修复: {file_path}")
        elif result is False:
            skipped_count += 1
        else:
            error_count += 1
    
    # 总结
    print("\n" + "=" * 80)
    print("修复完成！")
    print("=" * 80)
    print(f"✅ 成功修复: {fixed_count} 个文件")
    print(f"⚪ 无需修复: {skipped_count} 个文件")
    print(f"❌ 修复失败: {error_count} 个文件")
    print(f"📊 总计: {len(py_files)} 个文件")
    print()
    
    if fixed_count > 0:
        print("⚠️ 重要：请清理Python缓存后重新测试")
        print("Get-ChildItem -Path src -Recurse -Filter '__pycache__' | Remove-Item -Recurse -Force")


if __name__ == '__main__':
    main()

