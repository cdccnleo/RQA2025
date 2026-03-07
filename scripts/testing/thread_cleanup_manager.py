#!/usr/bin/env python3
"""
基础设施层线程清理管理器
解决线程退出和超时问题
"""

import sys
import time
import threading
import signal
from pathlib import Path
from typing import Dict, Set, Optional
from dataclasses import dataclass
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ThreadInfo:
    """线程信息"""
    thread_id: int
    thread_name: str
    thread_status: str
    start_time: datetime
    last_check: datetime
    is_daemon: bool
    is_alive: bool
    stack_trace: Optional[str] = None


class ThreadCleanupManager:
    """线程清理管理器"""

    def __init__(self):
        self.project_root = project_root
        self.main_thread = threading.main_thread()
        self.known_threads: Set[int] = set()
        self.thread_history: Dict[int, ThreadInfo] = {}
        self.cleanup_timeout = 30  # 清理超时时间（秒）
        self.monitoring_interval = 5  # 监控间隔（秒）
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # 设置信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n收到信号 {signum}，正在清理...")
        self.stop_monitoring()
        sys.exit(0)

    def start_monitoring(self):
        """启动线程监控"""
        if self.monitoring:
            print("线程监控已在运行")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="ThreadCleanupManager",
            daemon=True
        )
        self.monitor_thread.start()
        print("线程监控已启动")

    def stop_monitoring(self):
        """停止线程监控"""
        if not self.monitoring:
            return

        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        print("线程监控已停止")

    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                self._check_threads()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                print(f"监控循环异常: {e}")
                time.sleep(1)

    def _check_threads(self):
        """检查线程状态"""
        current_threads = threading.enumerate()
        current_thread_ids = {t.ident for t in current_threads if t.ident}

        # 更新已知线程
        self.known_threads.update(current_thread_ids)

        # 检查新线程
        for thread in current_threads:
            if thread.ident and thread.ident not in self.thread_history:
                self._add_thread_to_history(thread)

        # 检查已停止的线程
        stopped_threads = []
        for thread_id in list(self.thread_history.keys()):
            if thread_id not in current_thread_ids:
                stopped_threads.append(thread_id)

        # 清理已停止的线程记录
        for thread_id in stopped_threads:
            del self.thread_history[thread_id]
            self.known_threads.discard(thread_id)

        # 更新现有线程状态
        for thread in current_threads:
            if thread.ident and thread.ident in self.thread_history:
                self._update_thread_status(thread)

    def _add_thread_to_history(self, thread: threading.Thread):
        """添加线程到历史记录"""
        thread_info = ThreadInfo(
            thread_id=thread.ident,
            thread_name=thread.name,
            thread_status="running",
            start_time=datetime.now(),
            last_check=datetime.now(),
            is_daemon=thread.daemon,
            is_alive=thread.is_alive()
        )
        self.thread_history[thread.ident] = thread_info

    def _update_thread_status(self, thread: threading.Thread):
        """更新线程状态"""
        if thread.ident in self.thread_history:
            thread_info = self.thread_history[thread.ident]
            thread_info.is_alive = thread.is_alive()
            thread_info.last_check = datetime.now()

            # 检查线程是否卡住
            if thread_info.is_alive:
                time_since_start = datetime.now() - thread_info.start_time
                if time_since_start.total_seconds() > self.cleanup_timeout:
                    thread_info.thread_status = "stuck"
                else:
                    thread_info.thread_status = "running"

    def get_thread_summary(self) -> Dict:
        """获取线程摘要"""
        total_threads = len(self.thread_history)
        running_threads = len([t for t in self.thread_history.values() if t.is_alive])
        stuck_threads = len([t for t in self.thread_history.values() if t.thread_status == "stuck"])
        daemon_threads = len([t for t in self.thread_history.values() if t.is_daemon])

        return {
            "total_threads": total_threads,
            "running_threads": running_threads,
            "stuck_threads": stuck_threads,
            "daemon_threads": daemon_threads,
            "main_thread": self.main_thread.ident,
            "timestamp": datetime.now().isoformat()
        }

    def cleanup_stuck_threads(self, force: bool = False) -> int:
        """清理卡住的线程"""
        stuck_threads = [t for t in self.thread_history.values() if t.thread_status == "stuck"]

        if not stuck_threads:
            print("没有发现卡住的线程")
            return 0

        print(f"发现 {len(stuck_threads)} 个卡住的线程:")
        for thread_info in stuck_threads:
            print(f"  - {thread_info.thread_name} (ID: {thread_info.thread_id})")

        if not force:
            print("使用 --force 参数强制清理卡住的线程")
            return 0

        cleaned_count = 0
        for thread_info in stuck_threads:
            try:
                # 尝试优雅地停止线程
                if hasattr(thread_info, 'stop'):
                    thread_info.stop()
                    cleaned_count += 1
                    print(f"已停止线程: {thread_info.thread_name}")
            except Exception as e:
                print(f"停止线程 {thread_info.thread_name} 失败: {e}")

        return cleaned_count

    def force_cleanup_all_threads(self) -> int:
        """强制清理所有非主线程"""
        non_main_threads = [t for t in threading.enumerate() if t is not self.main_thread]

        if not non_main_threads:
            print("没有发现非主线程")
            return 0

        print(f"发现 {len(non_main_threads)} 个非主线程，正在强制清理...")

        cleaned_count = 0
        for thread in non_main_threads:
            try:
                if thread.is_alive():
                    # 设置停止标志（如果存在）
                    if hasattr(thread, '_stop_flag'):
                        thread._stop_flag.set()
                    elif hasattr(thread, '_stop_monitoring'):
                        thread._stop_monitoring = True
                    elif hasattr(thread, '_monitoring'):
                        thread._monitoring = False

                    # 等待线程结束
                    thread.join(timeout=2.0)

                    if not thread.is_alive():
                        cleaned_count += 1
                        print(f"已清理线程: {thread.name}")
                    else:
                        print(f"线程仍在运行: {thread.name}")

            except Exception as e:
                print(f"清理线程 {thread.name} 失败: {e}")

        return cleaned_count

    def print_thread_status(self):
        """打印线程状态"""
        summary = self.get_thread_summary()

        print(f"\n{'='*60}")
        print("线程状态摘要")
        print(f"{'='*60}")
        print(f"总线程数: {summary['total_threads']}")
        print(f"运行中: {summary['running_threads']}")
        print(f"卡住: {summary['stuck_threads']}")
        print(f"守护线程: {summary['daemon_threads']}")
        print(f"主线程ID: {summary['main_thread']}")
        print(f"检查时间: {summary['timestamp']}")

        if self.thread_history:
            print(f"\n详细线程信息:")
            print(f"{'ID':<10} {'名称':<20} {'状态':<10} {'守护':<6} {'运行时间':<10}")
            print("-" * 70)

            for thread_info in self.thread_history.values():
                runtime = datetime.now() - thread_info.start_time
                runtime_str = f"{runtime.total_seconds():.1f}s"
                daemon_str = "是" if thread_info.is_daemon else "否"

                print(f"{thread_info.thread_id:<10} {thread_info.thread_name:<20} "
                      f"{thread_info.thread_status:<10} {daemon_str:<6} {runtime_str:<10}")

    def cleanup_test_environment(self):
        """清理测试环境"""
        print("清理测试环境...")

        # 停止监控
        self.stop_monitoring()

        # 强制清理所有非主线程
        cleaned = self.force_cleanup_all_threads()

        # 等待一段时间让线程完全清理
        time.sleep(2)

        # 最终检查
        final_threads = [t for t in threading.enumerate() if t is not self.main_thread]
        if final_threads:
            print(f"警告: 仍有 {len(final_threads)} 个非主线程")
            for thread in final_threads:
                print(f"  - {thread.name} (ID: {thread.ident})")
        else:
            print("✅ 测试环境清理完成")

        return cleaned


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="基础设施层线程清理管理器")
    parser.add_argument("--monitor", action="store_true", help="启动线程监控")
    parser.add_argument("--status", action="store_true", help="显示线程状态")
    parser.add_argument("--cleanup", action="store_true", help="清理卡住的线程")
    parser.add_argument("--force", action="store_true", help="强制清理所有非主线程")
    parser.add_argument("--test-env", action="store_true", help="清理测试环境")
    parser.add_argument("--timeout", type=int, default=30, help="清理超时时间（秒）")

    args = parser.parse_args()

    print("=" * 60)
    print("基础设施层线程清理管理器")
    print("=" * 60)

    manager = ThreadCleanupManager()
    manager.cleanup_timeout = args.timeout

    try:
        if args.monitor:
            manager.start_monitoring()
            try:
                while True:
                    time.sleep(10)
                    manager.print_thread_status()
            except KeyboardInterrupt:
                print("\n用户中断，正在清理...")
                manager.stop_monitoring()

        elif args.status:
            manager.print_thread_status()

        elif args.cleanup:
            cleaned = manager.cleanup_stuck_threads(force=args.force)
            print(f"清理了 {cleaned} 个卡住的线程")

        elif args.force:
            cleaned = manager.force_cleanup_all_threads()
            print(f"强制清理了 {cleaned} 个线程")

        elif args.test_env:
            cleaned = manager.cleanup_test_environment()
            print(f"测试环境清理完成，清理了 {cleaned} 个线程")

        else:
            # 默认显示状态
            manager.print_thread_status()
            print("\n使用 --help 查看可用选项")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"异常: {e}")
    finally:
        manager.stop_monitoring()


if __name__ == "__main__":
    main()
