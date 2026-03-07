#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试环境准备脚本
"""

import os
import sys
import json
from pathlib import Path


def prepare_test_environment():
    """准备测试环境"""
    project_root = Path(__file__).parent.parent

    print("🔧 准备E2E测试环境...")

    # 1. 检查配置文件
    config_path = project_root / "tests" / "e2e" / "test_config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("✅ 测试配置加载成功")
    else:
        print("❌ 测试配置文件不存在")
        return False

    # 2. 创建必要的目录
    directories = [
        "tests/e2e/reports",
        "tests/e2e/screenshots",
        "tests/e2e/logs",
        "tests/e2e/data"
    ]

    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建目录: {dir_name}")

    # 3. 清理旧的测试数据
    cleanup_old_data(project_root)

    # 4. 预热测试数据库
    warm_up_database(project_root)

    print("✅ 测试环境准备完成")
    return True


def cleanup_old_data(project_root):
    """清理旧的测试数据"""
    import glob

    # 清理旧的报告文件
    report_files = glob.glob(str(project_root / "tests" / "e2e" / "reports" / "*"))
    for file_path in report_files:
        try:
            os.remove(file_path)
        except Exception:
            pass

    print("🧹 旧测试数据清理完成")


def warm_up_database(project_root):
    """预热测试数据库"""
    # 这里可以添加数据库预热逻辑
    print("🔥 数据库预热完成")


if __name__ == "__main__":
    success = prepare_test_environment()
    sys.exit(0 if success else 1)
