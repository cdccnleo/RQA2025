#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行特征分析层测试的脚本

解决pytest路径问题
"""

import os
import sys
import subprocess

def main():
    # 设置项目根目录和Python路径
    project_root = os.path.abspath(os.path.dirname(__file__) + "/..")
    src_dir = os.path.join(project_root, "src")

    # 构建pytest命令
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/features/acceleration/test_acceleration_components_coverage.py",
        "-v", "--tb=short", "--no-cov", "-n0"
    ]

    print(f"运行命令: {' '.join(cmd)}")

    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = src_dir

    # 运行测试
    result = subprocess.run(cmd, cwd=project_root, env=env, capture_output=False, text=True)

    print(f"测试执行完成，返回码: {result.returncode}")

if __name__ == "__main__":
    main()
