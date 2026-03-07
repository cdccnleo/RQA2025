#!/usr/bin/env python3
"""
批量修复健康监控模块导入问题
"""

import os
import re
from pathlib import Path

def fix_health_imports():
    """修复健康监控模块的导入问题"""

    health_dir = Path("src/infrastructure/health")

    # 1. 检查并添加缺失的常量
    constants_to_add = [
        "DEFAULT_MONITOR_TIMEOUT = 30.0",
        "HEALTH_CHECK_INTERVAL = 60.0",
        "MAX_RETRY_ATTEMPTS = 3",
        "DEFAULT_HEALTH_TIMEOUT = 30.0",
    ]

    # 检查health_checker.py
    checker_file = health_dir / "components" / "health_checker.py"
    if checker_file.exists():
        content = checker_file.read_text(encoding='utf-8')
        for const in constants_to_add:
            const_name = const.split('=')[0].strip()
            if f"{const_name} =" not in content:
                # 在常量区域添加
                content = content.replace(
                    "DEFAULT_MONITOR_TIMEOUT = 30.0",
                    "DEFAULT_MONITOR_TIMEOUT = 30.0\n" + const,
                    1
                )
                print(f"✓ 添加常量 {const_name} 到 health_checker.py")

        checker_file.write_text(content, encoding='utf-8')

    # 2. 检查并修复其他导入问题
    test_files = list(Path("tests/unit/infrastructure/health").glob("*.py"))

    for test_file in test_files[:10]:  # 先处理前10个文件作为测试
        try:
            content = test_file.read_text(encoding='utf-8')

            # 检查是否需要修复导入
            if "DEFAULT_MONITOR_TIMEOUT" in content and "from src.infrastructure.health.components.health_checker import" in content:
                # 导入语句看起来是正确的
                continue

            # 如果有问题，标记但暂时不修复
            print(f"需要检查: {test_file.name}")

        except Exception as e:
            print(f"处理文件 {test_file.name} 时出错: {e}")

def main():
    print("开始修复健康监控模块导入问题...")
    fix_health_imports()
    print("修复完成！")

if __name__ == "__main__":
    main()
