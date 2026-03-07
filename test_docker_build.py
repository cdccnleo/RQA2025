#!/usr/bin/env python3
"""
测试Docker构建脚本
"""

import subprocess
import sys
import time

def run_command(cmd, description):
    """运行命令并返回结果"""
    print(f"\n🔧 {description}")
    print(f"执行命令: {' '.join(cmd)}")

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        elapsed = time.time() - start_time

        print(".2f"        print(f"退出码: {result.returncode}")

        if result.returncode == 0:
            print("✅ 命令执行成功"            return True, result.stdout, result.stderr
        else:
            print("❌ 命令执行失败"            print("STDOUT:")
            print(result.stdout)
            print("STDERR:")
            print(result.stderr)
            return False, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        print("⏰ 命令执行超时"        return False, "", "Timeout"
    except Exception as e:
        print(f"💥 命令执行异常: {e}")
        return False, "", str(e)

def main():
    """主函数"""
    print("🐳 Docker构建测试")
    print("=" * 50)

    # 检查Docker是否可用
    success, stdout, stderr = run_command(["docker", "--version"], "检查Docker版本")
    if not success:
        print("❌ Docker不可用，无法继续测试")
        return False

    # 构建镜像
    success, stdout, stderr = run_command(
        ["docker", "build", "-t", "rqa2025-test", "."],
        "构建Docker镜像"
    )
    if not success:
        print("❌ Docker构建失败")
        return False

    # 检查镜像是否创建成功
    success, stdout, stderr = run_command(
        ["docker", "images", "rqa2025-test"],
        "检查镜像是否创建成功"
    )
    if not success or "rqa2025-test" not in stdout:
        print("❌ 镜像创建失败")
        return False

    print("✅ Docker镜像构建成功")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)