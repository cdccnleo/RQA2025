#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试运行脚本 - 自动清除PYTHONPATH环境变量
解决Redis导入问题
"""

import os
import sys
import subprocess


def run_tests_without_pythonpath():
    """清除PYTHONPATH并运行测试"""

    # 保存原始PYTHONPATH
    original_pythonpath = os.environ.get('PYTHONPATH')

    try:
        # 清除PYTHONPATH
        if 'PYTHONPATH' in os.environ:
            del os.environ['PYTHONPATH']
            print(f"已清除PYTHONPATH: {original_pythonpath}")

        # 构建pytest命令
        pytest_args = ['python', '-m', 'pytest']

        # 添加用户提供的参数
        if len(sys.argv) > 1:
            pytest_args.extend(sys.argv[1:])

        # 运行pytest
        print(f"运行命令: {' '.join(pytest_args)}")
        result = subprocess.run(pytest_args, check=False)

        return result.returncode

    finally:
        # 恢复原始PYTHONPATH
        if original_pythonpath:
            os.environ['PYTHONPATH'] = original_pythonpath
            print(f"已恢复PYTHONPATH: {original_pythonpath}")


if __name__ == '__main__':
    exit_code = run_tests_without_pythonpath()
    sys.exit(exit_code)
