#!/usr/bin/env python3
"""
自动运行安全模块重构脚本，选择src/core/security/路径
"""

import subprocess
import sys
from pathlib import Path


def run_restructure():
    """运行重构脚本并自动选择第一个选项"""
    script_path = Path(__file__).parent / "security_module_restructure.py"

    print("🚀 启动安全模块重构...")
    print("📁 目标路径: src/core/security/")
    print()

    # 使用subprocess运行脚本，并通过输入管道发送选择
    try:
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        # 发送输入：选择第一个选项（默认）
        stdout, stderr = process.communicate(input="\n")

        print("=== 重构脚本输出 ===")
        print(stdout)

        if stderr:
            print("=== 错误信息 ===")
            print(stderr)

        return process.returncode == 0

    except Exception as e:
        print(f"❌ 运行脚本失败: {e}")
        return False


if __name__ == "__main__":
    success = run_restructure()
    if success:
        print("\n✅ 重构脚本执行完成！")
    else:
        print("\n❌ 重构脚本执行失败！")
        sys.exit(1)
