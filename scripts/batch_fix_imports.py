#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量修复导入错误脚本
修复Infrastructure/utils测试文件的导入问题
"""

import os
import re
from pathlib import Path


def read_error_files(error_file='test_logs/infra_utils_errors_detailed.txt'):
    """读取错误文件列表"""
    if not os.path.exists(error_file):
        print(f"❌ 错误文件列表不存在: {error_file}")
        return []
    
    with open(error_file, 'r', encoding='utf-8') as f:
        files = [line.strip() for line in f if line.strip()]
    
    print(f"✅ 读取到 {len(files)} 个错误文件")
    return files


def backup_file(file_path):
    """备份文件"""
    if os.path.exists(file_path):
        backup_path = file_path + '.bak'
        import shutil
        shutil.copy2(file_path, backup_path)
        return True
    return False


def fix_environment_import(content):
    """修复environment导入"""
    # 模式1: from .environment import
    pattern1 = r'from \.environment import'
    replacement1 = 'from .components.environment import'
    content = re.sub(pattern1, replacement1, content)
    
    # 模式2: from infrastructure.utils.environment import
    pattern2 = r'from infrastructure\.utils\.environment import'
    replacement2 = 'from infrastructure.utils.components.environment import'
    content = re.sub(pattern2, replacement2, content)
    
    # 模式3: from src.infrastructure.utils.environment import
    pattern3 = r'from src\.infrastructure\.utils\.environment import'
    replacement3 = 'from src.infrastructure.utils.components.environment import'
    content = re.sub(pattern3, replacement3, content)
    
    return content


def fix_src_prefix(content):
    """确保有src前缀"""
    # 模式: from infrastructure. 但不是 from src.infrastructure.
    pattern = r'(?<!src\.)from infrastructure\.'
    replacement = 'from src.infrastructure.'
    content = re.sub(pattern, replacement, content)
    
    # 模式: import infrastructure. 但不是 import src.infrastructure.
    pattern2 = r'(?<!src\.)import infrastructure\.'
    replacement2 = 'import src.infrastructure.'
    content = re.sub(pattern2, replacement2, content)
    
    return content


def fix_file(file_path):
    """修复单个文件"""
    try:
        if not os.path.exists(file_path):
            print(f"⚠️ 文件不存在: {file_path}")
            return False
        
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # 应用修复
        content = original_content
        content = fix_environment_import(content)
        content = fix_src_prefix(content)
        
        # 检查是否有变化
        if content == original_content:
            return False
        
        # 备份
        backup_file(file_path)
        
        # 保存修复后的文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 已修复: {file_path}")
        return True
    
    except Exception as e:
        print(f"❌ 修复失败: {file_path} - {e}")
        return False


def main():
    """主函数"""
    print("=" * 80)
    print("批量修复Infrastructure/Utils测试导入错误")
    print("=" * 80)
    print()
    
    # 读取错误文件列表
    error_files = read_error_files()
    
    if not error_files:
        print("⚠️ 没有需要修复的文件")
        return
    
    print(f"\n准备修复 {len(error_files)} 个文件...\n")
    
    # 批量修复
    fixed_count = 0
    failed_count = 0
    skipped_count = 0
    
    for file_path in error_files:
        result = fix_file(file_path)
        if result:
            fixed_count += 1
        elif result is False and os.path.exists(file_path):
            skipped_count += 1
        else:
            failed_count += 1
    
    # 总结
    print("\n" + "=" * 80)
    print("修复完成！")
    print("=" * 80)
    print(f"✅ 成功修复: {fixed_count} 个文件")
    print(f"⚪ 无需修复: {skipped_count} 个文件")
    print(f"❌ 修复失败: {failed_count} 个文件")
    print(f"📊 总计: {len(error_files)} 个文件")
    print()
    
    if fixed_count > 0:
        print("⚠️ 重要提示：")
        print("1. 请清理Python缓存:")
        print("   Get-ChildItem -Path . -Recurse -Filter '__pycache__' -Directory | Remove-Item -Recurse -Force")
        print()
        print("2. 验证修复效果:")
        print("   pytest tests/unit/infrastructure/utils/ --co -q")


if __name__ == '__main__':
    main()

