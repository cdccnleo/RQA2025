#!/usr/bin/env python3
"""
改进的SystemMonitor类
使用threading.Event实现优雅的线程退出，解决time.sleep(60)阻塞问题
"""

import sys
import time
import threading
import signal
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ImprovedSystemMonitor:
    """改进的系统监控器，支持优雅的线程退出"""

    def __init__(self, check_interval: float = 60.0):
        self.check_interval = check_interval
        self._monitoring = False
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self._stats = []

        # 模拟psutil和os模块（用于测试）
        self.psutil = self._mock_psutil()
        self.os = self._mock_os()

        # 设置信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _mock_psutil(self):
        """模拟psutil模块用于测试"""
        class MockPsutil:
            def cpu_percent(self, interval=1):
                return 25.0

            def virtual_memory(self):
                class MockMemory:
                    def __init__(self):
                        self.total = 16 * 1024 * 1024 * 1024  # 16GB
                        self.available = 8 * 1024 * 1024 * 1024  # 8GB
                        self.used = 8 * 1024 * 1024 * 1024  # 8GB
                        self.percent = 50.0
                return MockMemory()

            def disk_usage(self, path):
                class MockDisk:
                    def __init__(self):
                        self.total = 1000 * 1024 * 1024 * 1024  # 1TB
                        self.used = 500 * 1024 * 1024 * 1024  # 500GB
                        self.free = 500 * 1024 * 1024 * 1024  # 500GB
                        self.percent = 50.0
                return MockDisk()

            def net_io_counters(self):
                class MockNetIO:
                    def __init__(self):
                        self.bytes_sent = 1024 * 1024  # 1MB
                        self.bytes_recv = 2048 * 1024  # 2MB
                        self.packets_sent = 1000
                        self.packets_recv = 2000
                return MockNetIO()

            def pids(self):
                return list(range(100))  # 100个进程

        return MockPsutil()

    def _mock_os(self):
        """模拟os模块用于测试"""
        class MockOS:
            def getloadavg(self):
                return [1.0, 1.5, 2.0]
        return MockOS()

    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n收到信号 {signum}，正在优雅关闭...")
        self.stop_monitoring()
        sys.exit(0)

    def start_monitoring(self) -> None:
        """启动系统监控"""
        if self._monitoring:
            return

        self._monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="SystemMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        print("✅ 系统监控已启动")

    def stop_monitoring(self) -> None:
        """停止系统监控"""
        if not self._monitoring:
            return

        print("🛑 正在停止系统监控...")
        self._monitoring = False
        self._stop_event.set()

        if self._monitor_thread and self._monitor_thread.is_alive():
            # 使用较短的超时时间，避免长时间等待
            self._monitor_thread.join(timeout=5.0)

            if self._monitor_thread.is_alive():
                print("⚠️  线程仍在运行，强制清理...")
                # 这里可以添加强制清理逻辑
            else:
                print("✅ 系统监控已优雅停止")

    def _monitor_loop(self) -> None:
        """监控主循环 - 改进版本"""
        print(f"🔄 监控循环已启动，检查间隔: {self.check_interval}秒")

        while self._monitoring:
            try:
                # 检查停止信号
                if self._stop_event.is_set():
                    print("🛑 收到停止信号，退出监控循环")
                    break

                # 收集系统统计
                stats = self._collect_system_stats()

                # 保存统计数据
                self._stats.append(stats)
                if len(self._stats) > 1000:
                    self._stats = self._stats[-1000:]

                # 检查系统状态
                self._check_system_status(stats)

                # 使用Event.wait()替代time.sleep()，支持中断
                if self._stop_event.wait(timeout=self.check_interval):
                    print("🛑 停止事件被触发，退出监控循环")
                    break

            except Exception as e:
                print(f"❌ 监控循环异常: {e}")
                # 即使有异常，也要检查停止信号
                if self._stop_event.is_set():
                    break

    def _collect_system_stats(self) -> dict:
        """收集系统统计数据"""
        # CPU使用率
        cpu_percent = self.psutil.cpu_percent(interval=1)

        # 内存使用
        mem = self.psutil.virtual_memory()

        # 磁盘使用
        disk = self.psutil.disk_usage('/')

        # 网络IO
        net_io = self.psutil.net_io_counters()

        # 系统负载
        load_avg = self._get_load_avg()

        # 进程数量
        process_count = len(self.psutil.pids())

        return {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'cpu': {
                'percent': cpu_percent,
                'load_avg': load_avg
            },
            'memory': {
                'total': mem.total,
                'available': mem.available,
                'used': mem.used,
                'percent': mem.percent
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            },
            'network': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            },
            'process': {
                'count': process_count
            }
        }

    def _get_load_avg(self):
        """获取系统负载"""
        try:
            if hasattr(self.os, 'getloadavg'):
                return list(self.os.getloadavg())
        except Exception:
            pass
        return None

    def _check_system_status(self, stats: dict) -> None:
        """检查系统状态"""
        # 简化的状态检查
        if stats['cpu']['percent'] > 80:
            print(f"⚠️  CPU使用率较高: {stats['cpu']['percent']}%")

        if stats['memory']['percent'] > 80:
            print(f"⚠️  内存使用率较高: {stats['memory']['percent']}%")

    def get_stats(self) -> list:
        """获取统计数据"""
        return self._stats.copy()

    def get_summary(self) -> dict:
        """获取监控摘要"""
        if not self._stats:
            return {}

        latest = self._stats[-1]
        return {
            'status': 'running' if self._monitoring else 'stopped',
            'thread_alive': self._monitor_thread.is_alive() if self._monitor_thread else False,
            'stats_count': len(self._stats),
            'latest_stats': latest
        }


def test_improved_system_monitor():
    """测试改进的系统监控器"""
    print("🧪 测试改进的系统监控器...")

    # 创建监控器实例
    monitor = ImprovedSystemMonitor(check_interval=2.0)  # 2秒间隔用于测试

    try:
        # 启动监控
        print("启动监控...")
        monitor.start_monitoring()

        # 等待一段时间
        print("等待5秒...")
        time.sleep(5)

        # 获取摘要
        summary = monitor.get_summary()
        print(f"监控摘要: {summary}")

        # 停止监控
        print("停止监控...")
        monitor.stop_monitoring()

        # 验证线程状态
        if monitor._monitor_thread:
            print(f"线程状态: {'运行中' if monitor._monitor_thread.is_alive() else '已停止'}")

        print("✅ 测试完成")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("改进的系统监控器测试")
    print("=" * 60)

    success = test_improved_system_monitor()

    if success:
        print("\n🎉 改进的系统监控器测试成功!")
        sys.exit(0)
    else:
        print("\n❌ 改进的系统监控器测试失败!")
        sys.exit(1)
