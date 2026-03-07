#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量更新FPGA测试文件的导入路径
将 src.fpga 更新为 src.acceleration.fpga
"""

import os
import re


def update_imports_in_file(file_path):
    """更新单个文件中的导入路径"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 更新导入路径
        old_content = content

        # 更新 from src.fpga 导入
        content = re.sub(
            r'from src\.fpga\.([a-zA-Z_]+) import',
            r'from src.acceleration.fpga.\1 import',
            content
        )

        # 更新 from src.fpga import
        content = re.sub(
            r'from src\.fpga import',
            r'from src.acceleration.fpga import',
            content
        )

        # 更新 patch 中的路径
        content = re.sub(
            r'@patch\([\'"](src\.fpga\.([a-zA-Z_]+))[\'"]',
            r'@patch(\'src.acceleration.fpga.\2\'',
            content
        )

        # 更新其他引用
        content = re.sub(
            r'src\.fpga\.([a-zA-Z_]+)',
            r'src.acceleration.fpga.\1',
            content
        )

        # 如果内容有变化，写回文件
        if content != old_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 已更新: {file_path}")
            return True
        else:
            print(f"⏭️  无需更新: {file_path}")
            return False

    except Exception as e:
        print(f"❌ 更新失败: {file_path} - {str(e)}")
        return False


def find_and_update_fpga_tests():
    """查找并更新所有FPGA相关的测试文件"""
    # 需要更新的目录
    test_dirs = [
        'tests/unit/acceleration/fpga',
        'tests/unit/integration',
        'tests/unit'
    ]

    updated_files = []
    total_files = 0

    for test_dir in test_dirs:
        if not os.path.exists(test_dir):
            continue

        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    total_files += 1

                    # 检查文件是否包含FPGA相关导入
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        if 'src.fpga' in content:
                            if update_imports_in_file(file_path):
                                updated_files.append(file_path)
                    except Exception as e:
                        print(f"❌ 读取失败: {file_path} - {str(e)}")

    return updated_files, total_files


def main():
    """主函数"""
    print("🔄 开始更新FPGA测试文件导入路径...")
    print("=" * 50)

    updated_files, total_files = find_and_update_fpga_tests()

    print("=" * 50)
    print(f"📊 更新统计:")
    print(f"   总文件数: {total_files}")
    print(f"   更新文件数: {len(updated_files)}")
    print(f"   更新率: {len(updated_files)/total_files*100:.1f}%" if total_files > 0 else "   更新率: 0%")

    if updated_files:
        print(f"\n✅ 已更新的文件:")
        for file in updated_files:
            print(f"   - {file}")

    print("\n🎉 导入路径更新完成!")


if __name__ == "__main__":
    main()
