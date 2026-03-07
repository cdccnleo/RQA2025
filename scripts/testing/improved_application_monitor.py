#!/usr/bin/env python3
"""
改进的ApplicationMonitor类
使用threading.Event实现优雅的线程退出，解决time.sleep(60)阻塞问题
"""

import sys
import time
import threading
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ImprovedApplicationMonitor:
    """改进的应用监控器，支持优雅的线程退出"""

    def __init__(self, app_name: str = "default_app"):
        self.app_name = app_name
        self._metrics = {
            'functions': [],
            'errors': [],
            'custom': []
        }
        self._last_compaction = time.time()
        self._stop_event = threading.Event()
        self._compaction_thread = None

        # 设置信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n收到信号 {signum}，正在优雅关闭...")
        self.close()
        sys.exit(0)

    def start_monitoring(self):
        """启动监控"""
        if self._compaction_thread and self._compaction_thread.is_alive():
            return

        self._stop_event.clear()
        self._compaction_thread = threading.Thread(
            target=self._auto_compact,
            name="ApplicationMonitor-Compaction",
            daemon=True
        )
        self._compaction_thread.start()
        print("✅ 应用监控已启动")

    def close(self):
        """关闭监控器并释放资源"""
        print("🛑 正在关闭应用监控...")

        if self._compaction_thread and self._compaction_thread.is_alive():
            self._stop_event.set()

            # 等待线程退出
            self._compaction_thread.join(timeout=5.0)

            if self._compaction_thread.is_alive():
                print("⚠️  线程仍在运行，强制清理...")
            else:
                print("✅ 应用监控已优雅关闭")

    def _auto_compact(self):
        """自动压缩监控数据 - 改进版本"""
        print(f"🔄 压缩线程已启动，检查间隔: 60秒")

        while not self._stop_event.is_set():
            try:
                # 使用Event.wait()替代time.sleep()，支持中断
                if self._stop_event.wait(timeout=60.0):
                    print("🛑 停止事件被触发，退出压缩循环")
                    break

                # 执行压缩逻辑
                now = time.time()

                # 压缩函数指标
                if len(self._metrics['functions']) > 1000:
                    self._metrics['functions'] = self._metrics['functions'][-1000:]

                # 压缩错误记录
                if len(self._metrics['errors']) > 1000:
                    self._metrics['errors'] = self._metrics['errors'][-1000:]

                # 压缩自定义指标
                if len(self._metrics['custom']) > 1000:
                    self._metrics['custom'] = self._metrics['custom'][-1000:]

                self._last_compaction = now
                print(f"✅ 数据压缩完成，时间: {datetime.fromtimestamp(now).strftime('%H:%M:%S')}")

            except Exception as e:
                print(f"❌ 压缩过程异常: {e}")
                # 即使有异常，也要检查停止信号
                if self._stop_event.is_set():
                    break

    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""
        return {
            'app_name': self.app_name,
            'timestamp': datetime.now().isoformat(),
            'metrics_count': {
                'functions': len(self._metrics['functions']),
                'errors': len(self._metrics['errors']),
                'custom': len(self._metrics['custom'])
            },
            'last_compaction': self._last_compaction
        }

    def get_status(self) -> Dict[str, Any]:
        """获取监控器状态"""
        return {
            'app_name': self.app_name,
            'monitoring': self._compaction_thread and self._compaction_thread.is_alive(),
            'thread_alive': self._compaction_thread.is_alive() if self._compaction_thread else False,
            'stop_event_set': self._stop_event.is_set(),
            'metrics_count': len(self._metrics['functions']) + len(self._metrics['errors']) + len(self._metrics['custom'])
        }


def test_improved_application_monitor():
    """测试改进的应用监控器"""
    print("🧪 测试改进的应用监控器...")

    # 创建监控器实例
    monitor = ImprovedApplicationMonitor("test_app")

    try:
        # 启动监控
        print("启动监控...")
        monitor.start_monitoring()
        time.sleep(2)

        # 获取状态
        status = monitor.get_status()
        print(f"监控状态: {status}")

        # 等待一段时间
        print("等待5秒...")
        time.sleep(5)

        # 获取指标
        metrics = monitor.get_metrics()
        print(f"监控指标: {metrics}")

        # 关闭监控
        print("关闭监控...")
        monitor.close()

        # 验证线程状态
        if monitor._compaction_thread:
            print(f"线程状态: {'运行中' if monitor._compaction_thread.is_alive() else '已停止'}")

        print("✅ 测试完成")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("改进的应用监控器测试")
    print("=" * 60)

    success = test_improved_application_monitor()

    if success:
        print("\n🎉 改进的应用监控器测试成功!")
        sys.exit(0)
    else:
        print("\n❌ 改进的应用监控器测试失败!")
        sys.exit(1)
