#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复后的覆盖率运行脚本

这个脚本解决覆盖率工具的路径识别问题，通过以下方式：
1. 直接设置Python路径
2. 使用正确的覆盖率参数
3. 避免路径映射问题
"""

import os
import sys
import subprocess
from pathlib import Path


def setup_environment():
    """设置环境变量和Python路径"""
    project_root = Path(__file__).parent.parent.parent
    src_path = project_root / "src"

    # 设置Python路径
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # 设置环境变量
    os.environ['PYTHONPATH'] = str(src_path)
    os.environ['COVERAGE_FILE'] = str(project_root / '.coverage')

    print(f"✅ 已设置Python路径: {src_path}")
    print(f"✅ 已设置环境变量: PYTHONPATH={src_path}")

    return project_root, src_path


def run_coverage_test():
    """运行覆盖率测试"""
    project_root, src_path = setup_environment()

    print("\n🧪 运行修复后的覆盖率测试...")

    # 使用修复后的覆盖率参数
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/infrastructure/utils/helpers/test_environment_manager_enhanced.py",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "-v"
    ]

    try:
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ 覆盖率测试成功！")

            # 检查输出
            if "Module was never imported" not in result.stdout:
                print("✅ 路径问题已修复！")
            else:
                print("⚠️ 路径问题仍然存在，尝试其他解决方案...")
                return False

        else:
            print(f"❌ 覆盖率测试失败: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ 运行覆盖率测试时出错: {e}")
        return False

    return True


def run_coverage_with_alternative_method():
    """使用替代方法运行覆盖率测试"""
    project_root, src_path = setup_environment()

    print("\n🔄 尝试替代方法...")

    # 方法1：使用相对路径
    cmd1 = [
        sys.executable, "-m", "pytest",
        "tests/unit/infrastructure/utils/helpers/test_environment_manager_enhanced.py",
        "--cov=infrastructure.utils.helpers.environment_manager",
        "--cov-report=term-missing",
        "-v"
    ]

    try:
        result1 = subprocess.run(cmd1, cwd=project_root, capture_output=True, text=True)

        if result1.returncode == 0 and "Module was never imported" not in result1.stdout:
            print("✅ 替代方法1成功！")
            return True

    except Exception as e:
        print(f"⚠️ 替代方法1失败: {e}")

    # 方法2：使用绝对路径
    cmd2 = [
        sys.executable, "-m", "pytest",
        "tests/unit/infrastructure/utils/helpers/test_environment_manager_enhanced.py",
        f"--cov={src_path}/infrastructure/utils/helpers/environment_manager",
        "--cov-report=term-missing",
        "-v"
    ]

    try:
        result2 = subprocess.run(cmd2, cwd=project_root, capture_output=True, text=True)

        if result2.returncode == 0 and "Module was never imported" not in result2.stdout:
            print("✅ 替代方法2成功！")
            return True

    except Exception as e:
        print(f"⚠️ 替代方法2失败: {e}")

    return False


def main():
    """主函数"""
    print("🔧 开始运行修复后的覆盖率测试...")

    # 尝试标准方法
    if run_coverage_test():
        print("\n🎉 覆盖率测试成功！路径问题已修复。")
        return

    # 尝试替代方法
    if run_coverage_with_alternative_method():
        print("\n🎉 使用替代方法成功！")
        return

    print("\n❌ 所有方法都失败了。需要进一步调查路径问题。")


if __name__ == "__main__":
    main()
