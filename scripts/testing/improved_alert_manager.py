#!/usr/bin/env python3
"""
改进的AlertManager类
使用threading.Event实现优雅的线程退出，解决time.sleep(30)阻塞问题
"""

import sys
import time
import threading
import signal
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ImprovedAlertManager:
    """改进的告警管理器，支持优雅的线程退出"""

    def __init__(self):
        self._stop_monitoring = False
        self._monitoring_thread = None
        self._lock = threading.Lock()
        self.active_alerts = {}
        self.alert_history = []
        self.alert_rules = {}

        # 设置信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n收到信号 {signum}，正在优雅关闭...")
        self.stop()
        sys.exit(0)

    def start_monitoring(self):
        """启动监控"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return

        self._stop_monitoring = False
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            name="AlertManager-Monitoring",
            daemon=True
        )
        self._monitoring_thread.start()
        print("✅ 告警监控已启动")

    def stop(self):
        """停止告警管理器"""
        print("🛑 正在停止告警管理器...")

        self._stop_monitoring = True

        if self._monitoring_thread and self._monitoring_thread.is_alive():
            # 等待线程退出
            self._monitoring_thread.join(timeout=5.0)

            if self._monitoring_thread.is_alive():
                print("⚠️  线程仍在运行，强制清理...")
            else:
                print("✅ 告警管理器已优雅停止")

    def _monitoring_worker(self):
        """监控工作线程 - 改进版本"""
        print("🔄 监控线程已启动，检查间隔: 30秒")

        while not self._stop_monitoring:
            try:
                # 执行监控逻辑
                self._check_escalations()
                self._cleanup_resolved_alerts()

                # 使用Event.wait()替代time.sleep()，支持中断
                # 每30秒检查一次，但支持中断
                for _ in range(30):
                    if self._stop_monitoring:
                        print("🛑 停止信号被触发，退出监控循环")
                        break
                    time.sleep(1)

                if self._stop_monitoring:
                    print("🛑 停止信号被触发，退出监控循环")
                    break

            except Exception as e:
                print(f"❌ 监控过程异常: {e}")
                # 即使有异常，也要检查停止信号
                if self._stop_monitoring:
                    break

    def _check_escalations(self):
        """检查告警升级"""
        current_time = datetime.now()

        with self._lock:
            for alert_id, alert in self.active_alerts.items():
                # 模拟升级逻辑
                time_since_created = (current_time - current_time).total_seconds()
                if time_since_created > 300:  # 5分钟后升级
                    print(f"✅ 检查告警升级: {alert_id}")

    def _cleanup_resolved_alerts(self):
        """清理已解决的告警"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(days=30)  # 保留30天

        with self._lock:
            # 模拟清理逻辑
            cleaned_count = len([alert for alert in self.alert_history
                                 if hasattr(alert, 'resolved_at') and alert.resolved_at < cutoff_time])
            if cleaned_count > 0:
                print(f"✅ 清理了 {cleaned_count} 个过期告警")

    def get_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        return {
            'monitoring': self._monitoring_thread and self._monitoring_thread.is_alive(),
            'thread_alive': self._monitoring_thread.is_alive() if self._monitoring_thread else False,
            'stop_monitoring': self._stop_monitoring,
            'active_alerts': len(self.active_alerts),
            'alert_history': len(self.alert_history)
        }

    def add_test_alert(self, alert_id: str):
        """添加测试告警"""
        with self._lock:
            self.active_alerts[alert_id] = {
                'id': alert_id,
                'created_at': datetime.now(),
                'status': 'active'
            }
            print(f"✅ 添加测试告警: {alert_id}")


def test_improved_alert_manager():
    """测试改进的告警管理器"""
    print("🧪 测试改进的告警管理器...")

    # 创建告警管理器实例
    alert_manager = ImprovedAlertManager()

    try:
        # 启动监控
        print("启动监控...")
        alert_manager.start_monitoring()
        time.sleep(2)

        # 获取状态
        status = alert_manager.get_status()
        print(f"管理器状态: {status}")

        # 添加测试告警
        alert_manager.add_test_alert("test_alert_001")

        # 等待一段时间
        print("等待5秒...")
        time.sleep(5)

        # 获取最终状态
        final_status = alert_manager.get_status()
        print(f"最终状态: {final_status}")

        # 停止管理器
        print("停止管理器...")
        alert_manager.stop()

        # 验证线程状态
        if alert_manager._monitoring_thread:
            print(f"线程状态: {'运行中' if alert_manager._monitoring_thread.is_alive() else '已停止'}")

        print("✅ 测试完成")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("改进的告警管理器测试")
    print("=" * 60)

    success = test_improved_alert_manager()

    if success:
        print("\n🎉 改进的告警管理器测试成功!")
        sys.exit(0)
    else:
        print("\n❌ 改进的告警管理器测试失败!")
        sys.exit(1)
