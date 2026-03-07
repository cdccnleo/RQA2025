#!/usr/bin/env python3
"""
RQA2025 简单服务启动脚本
"""

import os
import sys
import subprocess
import time


def main():
    print("🚀 RQA2025 量化交易系统启动中...")
    print("=" * 50)

    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()
    env['PYTHONIOENCODING'] = 'utf-8'

    try:
        # 启动服务
        print("启动服务...")
        process = subprocess.Popen([
            sys.executable, 'main.py'
        ], cwd=os.getcwd(), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding='utf-8', errors='replace')

        print("等待服务启动...")
        time.sleep(5)

        # 检查进程是否还在运行
        if process.poll() is None:
            print("✅ 服务启动成功!")
            print("🌐 访问地址:")
            print("   主应用: http://localhost:8000")
            print("   API文档: http://localhost:8000/docs")
            print("   健康检查: http://localhost:8000/health")
            print()
            print("按 Ctrl+C 停止服务")

            # 等待用户中断
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 正在停止服务...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                    print("✅ 服务已停止")
                except subprocess.TimeoutExpired:
                    process.kill()
                    print("⚠️ 服务已被强制停止")

        else:
            # 进程已退出，检查退出码
            stdout, stderr = process.communicate()
            print("❌ 服务启动失败")
            print("错误信息:")
            if stderr:
                print(stderr)
            return 1

    except Exception as e:
        print(f"❌ 启动过程中出现错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
