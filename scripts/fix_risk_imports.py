#!/usr/bin/env python3
"""
修复风险控制层测试文件导入路径脚本
Fix Risk Control Layer Test Import Paths Script

批量修复风险控制层测试文件中的导入路径问题

Author: RQA2025 Development Team
Date: 2025-12-01
"""

import os
import re
from pathlib import Path

def fix_import_in_file(file_path):
    """修复单个文件中的导入路径"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 记录原始内容，用于检查是否需要修改
        original_content = content

        # 修复常见的导入路径问题
        # 1. 将 'from risk.' 替换为 'from src.risk.'
        content = re.sub(r'from risk\.', r'from src.risk.', content)

        # 2. 将 'import risk.' 替换为 'import src.risk.'
        content = re.sub(r'import risk\.', r'import src.risk.', content)

        # 3. 修复相对导入问题（如果有的话）
        # content = re.sub(r'from \.\.', r'from src.risk', content)

        # 如果内容有变化，写入文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 修复: {file_path}")
            return True
        else:
            print(f"⏭️ 跳过: {file_path} (无需修改)")
            return False

    except Exception as e:
        print(f"❌ 错误: {file_path} - {e}")
        return False

def main():
    """主函数"""
    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent
    risk_test_dir = project_root / "tests" / "unit" / "risk"

    if not risk_test_dir.exists():
        print(f"❌ 风险控制层测试目录不存在: {risk_test_dir}")
        return 1

    print("🚀 开始修复风险控制层测试文件导入路径...")
    print(f"测试目录: {risk_test_dir}")
    print("-" * 60)

    # 统计信息
    total_files = 0
    fixed_files = 0
    skipped_files = 0
    error_files = 0

    # 遍历所有Python测试文件
    for root, dirs, files in os.walk(risk_test_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                total_files += 1

                # 跳过conftest.py（它已经正确配置了）
                if file == 'conftest.py':
                    print(f"⏭️ 跳过: {file_path} (配置文件)")
                    skipped_files += 1
                    continue

                # 跳过__init__.py文件
                if file == '__init__.py':
                    print(f"⏭️ 跳过: {file_path} (包初始化文件)")
                    skipped_files += 1
                    continue

                if fix_import_in_file(file_path):
                    fixed_files += 1
                else:
                    skipped_files += 1

    print("\n" + "="*60)
    print("📊 修复统计:")
    print(f"总文件数: {total_files}")
    print(f"修复文件数: {fixed_files}")
    print(f"跳过文件数: {skipped_files}")
    print(f"错误文件数: {error_files}")
    print("="*60)

    if fixed_files > 0:
        print("\n🎯 修复完成！建议重新运行测试验证效果。")
    else:
        print("\nℹ️  所有文件都已经正确配置，无需修复。")

    return 0

if __name__ == "__main__":
    exit(main())
