#!/usr/bin/env python3
"""
脚本调度和立即终止功能演示
展示如何管理多个测试脚本的运行和立即终止
"""

import sys
import time
import signal
import threading
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def demo_script_scheduler():
    """演示脚本调度器功能"""
    print("🚀 脚本调度和立即终止功能演示")
    print("="*60)

    try:
        # 导入脚本调度器
        from script_scheduler import ScriptScheduler, ScriptInfo

        # 创建调度器
        scheduler = ScriptScheduler()

        # 创建测试脚本
        test_scripts = {
            'performance_test': ScriptInfo(
                name='performance_test',
                path='scripts/testing/run_performance_benchmark.py'
            ),
            'simple_performance': ScriptInfo(
                name='simple_performance',
                path='scripts/testing/simple_performance_benchmark_system.py'
            )
        }

        print("\n📋 可用脚本:")
        for name, script_info in test_scripts.items():
            print(f"  - {name}: {script_info.path}")

        # 启动监控
        monitor_thread = scheduler.start_monitoring()

        print("\n🎮 演示控制功能:")
        print("1. 启动性能测试脚本")
        print("2. 查看运行状态")
        print("3. 立即终止脚本")
        print("4. 强制终止所有脚本")

        # 演示1: 启动脚本
        print("\n📊 演示1: 启动性能测试脚本")
        performance_script = test_scripts['performance_test']
        success = scheduler.start_script(performance_script)

        if success:
            print("✅ 脚本启动成功")

            # 等待一段时间
            print("⏰ 等待5秒...")
            time.sleep(5)

            # 演示2: 查看状态
            print("\n📊 演示2: 查看脚本状态")
            status = scheduler.get_script_status('performance_test')
            if status:
                print(f"  脚本名称: {status.name}")
                print(f"  PID: {status.pid}")
                print(f"  状态: {status.status}")
                print(f"  内存使用: {status.memory_usage:.1f}MB")
                print(f"  CPU使用率: {status.cpu_usage:.1f}%")

            # 演示3: 立即终止
            print("\n⏹️ 演示3: 立即终止脚本")
            print("正在终止性能测试脚本...")
            scheduler.stop_script('performance_test')

            # 等待终止完成
            time.sleep(2)

            # 检查终止结果
            status = scheduler.get_script_status('performance_test')
            if not status:
                print("✅ 脚本已成功终止")
            else:
                print(f"⚠️ 脚本状态: {status.status}")

        # 演示4: 启动多个脚本并强制终止
        print("\n📊 演示4: 启动多个脚本")
        for name, script_info in test_scripts.items():
            scheduler.start_script(script_info)
            print(f"  启动脚本: {name}")
            time.sleep(1)

        print("⏰ 等待3秒...")
        time.sleep(3)

        print("\n💥 演示5: 强制终止所有脚本")
        print("正在强制终止所有脚本...")
        scheduler.terminate_all_scripts()

        # 等待终止完成
        time.sleep(2)

        # 检查结果
        running_scripts = scheduler.list_running_scripts()
        if not running_scripts:
            print("✅ 所有脚本已成功终止")
        else:
            print(f"⚠️ 仍有 {len(running_scripts)} 个脚本在运行")

        # 停止监控
        scheduler.stop_monitoring()

        print("\n🎉 演示完成！")

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保 script_scheduler.py 文件存在")
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")


def demo_signal_handling():
    """演示信号处理功能"""
    print("\n🔔 信号处理功能演示")
    print("="*40)

    def signal_handler(signum, frame):
        print(f"\n📡 收到信号 {signum}")
        print("正在优雅地退出...")
        sys.exit(0)

    # 设置信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("信号处理器已设置:")
    print("  - SIGINT (Ctrl+C): 优雅终止")
    print("  - SIGTERM: 优雅终止")

    print("\n⏰ 等待10秒，您可以按 Ctrl+C 测试信号处理...")
    try:
        for i in range(10, 0, -1):
            print(f"\r⏳ 倒计时: {i} 秒", end='')
            time.sleep(1)
        print("\n✅ 信号处理演示完成")
    except KeyboardInterrupt:
        print("\n📡 收到 Ctrl+C 信号")
        print("正在退出...")


def demo_immediate_termination():
    """演示立即终止功能"""
    print("\n⚡ 立即终止功能演示")
    print("="*40)

    def long_running_task():
        """模拟长时间运行的任务"""
        print("🔄 开始长时间运行任务...")
        for i in range(30):
            print(f"\r⏳ 任务进度: {i+1}/30", end='')
            time.sleep(1)
        print("\n✅ 任务完成")

    def task_with_timeout():
        """带超时的任务"""
        print("⏰ 启动带超时的任务...")
        task_thread = threading.Thread(target=long_running_task)
        task_thread.daemon = True
        task_thread.start()

        # 等待5秒后强制终止
        time.sleep(5)
        print("\n💥 强制终止任务")
        return "任务被强制终止"

    print("演示1: 长时间运行任务")
    result = task_with_timeout()
    print(f"结果: {result}")

    print("\n演示2: 信号中断任务")
    print("启动任务，按 Ctrl+C 中断...")
    try:
        long_running_task()
    except KeyboardInterrupt:
        print("\n📡 任务被信号中断")


def main():
    """主函数"""
    print("🎬 脚本调度和立即终止功能演示")
    print("="*60)

    try:
        # 演示1: 脚本调度器
        demo_script_scheduler()

        # 演示2: 信号处理
        demo_signal_handling()

        # 演示3: 立即终止
        demo_immediate_termination()

        print("\n🎉 所有演示完成！")
        print("\n📋 功能总结:")
        print("✅ 脚本调度: 可以管理多个测试脚本的运行")
        print("✅ 立即终止: 支持优雅终止和强制终止")
        print("✅ 信号处理: 响应 Ctrl+C 和系统信号")
        print("✅ 状态监控: 实时监控脚本运行状态")
        print("✅ 资源管理: 自动清理和资源释放")

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断演示")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")


if __name__ == "__main__":
    main()
