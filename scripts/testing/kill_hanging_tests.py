#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
紧急测试进程清理脚本
用于解决测试无法退出的问题
"""

import os
import psutil
import time
import subprocess


def kill_python_processes():
    """杀死所有相关的Python测试进程"""
    try:
        current_pid = os.getpid()
        killed_count = 0

        print("🔍 扫描Python进程...")
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                if proc.info['name'] == 'python.exe' and proc.info['pid'] != current_pid:
                    cmdline = proc.info.get('cmdline', [])
                    cmd_str = ' '.join(cmdline) if cmdline else ''

                    # 检查是否是pytest或相关测试进程
                    if any(keyword in cmd_str.lower() for keyword in ['pytest', 'test', 'conftest']):
                        create_time = proc.info.get('create_time', 0)
                        age = time.time() - create_time

                        print(f"🎯 发现测试进程: PID={proc.info['pid']}, 运行时间={age:.1f}秒")
                        print(f"   命令: {cmd_str[:100]}{'...' if len(cmd_str) > 100 else ''}")

                        # 终止进程
                        proc.kill()
                        killed_count += 1
                        print(f"   ✅ 已终止")

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except Exception as e:
                print(f"   ⚠️  处理进程时出错: {e}")

        if killed_count > 0:
            print(f"\n🎉 成功终止 {killed_count} 个测试进程")
            time.sleep(2)  # 等待进程完全退出
        else:
            print("\n✅ 没有发现需要终止的测试进程")

    except ImportError:
        print("⚠️  psutil未安装，使用备用方法...")

        # 备用方法：使用tasklist和taskkill
        if os.name == 'nt':
            try:
                # 获取所有python进程
                result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'],
                                        capture_output=True, text=True)

                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 3:  # 跳过标题行
                        print(f"发现 {len(lines)-3} 个Python进程")
                        # 终止所有python进程（除了当前进程）
                        subprocess.run(['taskkill', '/F', '/FI', 'IMAGENAME eq python.exe'],
                                       capture_output=True)
                        print("✅ 已终止Python进程")
            except Exception as e:
                print(f"备用方法失败: {e}")
    except Exception as e:
        print(f"❌ 清理进程时出错: {e}")


def check_thread_status():
    """检查当前进程的线程状态"""
    try:
        import threading

        current_process = psutil.Process(os.getpid())
        thread_count = current_process.num_threads()

        print(f"\n📊 当前进程线程状态:")
        print(f"   总线程数: {thread_count}")

        # 获取活跃线程
        active_threads = threading.enumerate()
        print(f"   Python线程数: {len(active_threads)}")

        for i, thread in enumerate(active_threads):
            print(f"   {i+1}. {thread.name} (alive: {thread.is_alive()})")

    except Exception as e:
        print(f"⚠️  无法检查线程状态: {e}")


def main():
    """主函数"""
    print("🚨 紧急测试进程清理工具")
    print("=" * 40)

    # 检查线程状态
    check_thread_status()

    # 终止进程
    kill_python_processes()

    # 最终检查
    time.sleep(1)
    check_thread_status()

    print("\n✅ 清理完成")
    print("\n💡 如果问题仍然存在，建议:")
    print("   1. 重启命令行/终端")
    print("   2. 检查代码中的无限循环")
    print("   3. 确认所有线程都有正确的停止机制")


if __name__ == "__main__":
    main()
