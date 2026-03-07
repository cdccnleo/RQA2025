#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修复基础设施层测试文件的导入问题
"""

import os
import re
import glob
from pathlib import Path
from typing import List, Dict

def find_test_files() -> List[str]:
    """查找所有测试文件"""
    test_files = []
    test_root = Path(__file__).parent.parent / "tests" / "unit" / "infrastructure"

    for pattern in ["**/*.py"]:
        test_files.extend(glob.glob(str(test_root / pattern), recursive=True))

    return [f for f in test_files if f.endswith('.py') and os.path.basename(f).startswith('test_')]

def analyze_import_issues(test_files: List[str]) -> Dict[str, List[str]]:
    """分析导入问题"""
    issues = {}

    for test_file in test_files[:50]:  # 先处理前50个文件
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            file_issues = []

            # 检查src.infrastructure导入
            src_imports = re.findall(r'from src\.infrastructure\..*?import', content)
            if src_imports:
                file_issues.append(f"Found {len(src_imports)} src.infrastructure imports")

            # 检查相对导入
            relative_imports = re.findall(r'from \.\..*?import', content)
            if relative_imports:
                file_issues.append(f"Found {len(relative_imports)} relative imports")

            # 检查导入错误注释
            if 'ImportError' in content or 'ModuleNotFoundError' in content:
                file_issues.append("Contains import error comments")

            if file_issues:
                issues[test_file] = file_issues

        except Exception as e:
            issues[test_file] = [f"Read error: {str(e)}"]

    return issues

def fix_import_paths(test_file: str) -> bool:
    """修复单个文件的导入路径"""
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 替换src.infrastructure导入为相对导入
        # from src.infrastructure.xxx.yyy import ZZZ -> from infrastructure.xxx.yyy import ZZZ
        content = re.sub(
            r'from src\.infrastructure\.([^\'"]*?) import',
            r'from infrastructure.\1 import',
            content
        )

        # 修复一些常见的导入问题
        # 移除不正确的导入
        lines = content.split('\n')
        fixed_lines = []

        skip_next = False
        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue

            # 跳过有问题的导入
            if any(error in line for error in [
                'from src.infrastructure.init_infrastructure import',
                'from src.infrastructure.health import EnhancedHealthChecker',
                'from src.infrastructure.logging.core.unified_logging_interface import'
            ]):
                # 检查下一行是否是导入的继续
                if i + 1 < len(lines) and lines[i + 1].strip().startswith(')'):
                    skip_next = True
                continue

            fixed_lines.append(line)

        content = '\n'.join(fixed_lines)

        # 如果内容有变化，写回文件
        if content != original_content:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"Error fixing {test_file}: {e}")
        return False

def create_init_files():
    """确保必要的__init__.py文件存在"""
    dirs_to_check = [
        "src/infrastructure",
        "src/infrastructure/cache",
        "src/infrastructure/cache/strategies",
        "src/infrastructure/cache/core",
        "src/infrastructure/cache/exceptions",
        "src/infrastructure/config",
        "src/infrastructure/config/core",
        "src/infrastructure/config/validators",
        "src/infrastructure/health",
        "src/infrastructure/health/components",
        "src/infrastructure/logging",
        "src/infrastructure/logging/core",
        "src/infrastructure/monitoring",
        "src/infrastructure/security",
        "src/infrastructure/utils",
        "src/infrastructure/distributed"
    ]

    for dir_path in dirs_to_check:
        init_file = os.path.join(dir_path, "__init__.py")
        if not os.path.exists(init_file):
            try:
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write('"""Infrastructure module"""\n')
                print(f"Created: {init_file}")
            except Exception as e:
                print(f"Error creating {init_file}: {e}")

def main():
    """主函数"""
    print("🔧 基础设施层测试导入问题修复")
    print("=" * 60)

    # 1. 确保__init__.py文件存在
    print("📁 创建必要的__init__.py文件...")
    create_init_files()

    # 2. 查找测试文件
    print("🔍 查找测试文件...")
    test_files = find_test_files()
    print(f"找到 {len(test_files)} 个测试文件")

    # 3. 分析导入问题
    print("📊 分析导入问题...")
    issues = analyze_import_issues(test_files)

    print(f"发现 {len(issues)} 个有问题的文件:")
    for file_path, file_issues in list(issues.items())[:10]:  # 只显示前10个
        print(f"  • {os.path.basename(file_path)}: {', '.join(file_issues)}")

    # 4. 修复导入路径
    print("🔧 修复导入路径...")
    fixed_count = 0
    for test_file in test_files[:100]:  # 先修复前100个文件
        if fix_import_paths(test_file):
            fixed_count += 1
            if fixed_count % 10 == 0:
                print(f"已修复 {fixed_count} 个文件...")

    print(f"✅ 共修复了 {fixed_count} 个文件的导入路径")

    # 5. 生成修复报告
    print("
📋 修复报告:"    print(f"   • 处理的文件总数: {len(test_files)}")
    print(f"   • 有导入问题文件: {len(issues)}")
    print(f"   • 成功修复文件: {fixed_count}")
    print(f"   • 修复成功率: {fixed_count/len(test_files)*100:.1f}%" if test_files else "0%")

    print("
💡 修复内容:"    print("   • 将 'from src.infrastructure.xxx import' 改为 'from infrastructure.xxx import'")
    print("   • 移除了有问题的导入语句")
    print("   • 创建了缺失的 __init__.py 文件")

    print("
🎯 建议下一步:"    print("   • 运行测试验证修复效果")
    print("   • 检查是否还有其他导入问题")
    print("   • 运行覆盖率统计")

if __name__ == "__main__":
    main()
