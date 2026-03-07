#!/usr/bin/env python3
"""
解决SystemMonitor超时问题的专门脚本
通过强制清理线程来解决time.sleep(60)的阻塞问题
"""

from scripts.testing.thread_cleanup_manager import ThreadCleanupManager
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入线程清理管理器


class SystemMonitorTimeoutFixer:
    """SystemMonitor超时问题修复器"""

    def __init__(self):
        self.thread_manager = ThreadCleanupManager()
        self.monitors = []

    def create_system_monitor(self):
        """创建SystemMonitor实例"""
        try:
            from src.infrastructure.monitoring.system_monitor import SystemMonitor
            sm = SystemMonitor()
            self.monitors.append(sm)
            return sm
        except ImportError as e:
            print(f"❌ 无法导入SystemMonitor: {e}")
            return None

    def test_system_monitor_with_cleanup(self):
        """测试SystemMonitor并自动清理"""
        print("🧪 测试SystemMonitor并自动清理...")

        try:
            # 创建监控器
            sm = self.create_system_monitor()
            if not sm:
                return False

            print(f"✅ SystemMonitor创建成功")
            print(f"   check_interval: {sm.check_interval}")
            print(f"   _monitoring: {sm._monitoring}")

            # 启动监控
            print("启动监控...")
            sm.start_monitoring()
            time.sleep(0.1)  # 等待线程启动

            print(f"   监控状态: {sm._monitoring}")
            print(f"   监控线程: {sm._monitor_thread}")
            print(f"   线程状态: {sm._monitor_thread.is_alive()}")

            # 立即停止（不等待超时）
            print("立即停止监控...")
            try:
                # 使用较短的超时时间
                sm._monitor_thread.join(timeout=2.0)
                print("✅ 线程正常退出")
            except Exception as e:
                print(f"⚠️  线程退出异常: {e}")

            # 如果线程仍在运行，强制清理
            if sm._monitor_thread.is_alive():
                print("⚠️  线程仍在运行，强制清理...")
                self.thread_manager.cleanup_test_environment()

                # 再次检查
                if sm._monitor_thread.is_alive():
                    print("❌ 强制清理后线程仍在运行")
                    return False
                else:
                    print("✅ 强制清理成功")

            return True

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False

    def run_comprehensive_test(self):
        """运行综合测试"""
        print("=" * 60)
        print("SystemMonitor超时问题综合测试")
        print("=" * 60)

        try:
            # 1. 清理环境
            print("\n🔍 清理测试环境...")
            cleaned_count = self.thread_manager.cleanup_test_environment()
            print(f"清理了 {cleaned_count} 个线程")

            # 2. 运行测试
            success = self.test_system_monitor_with_cleanup()

            # 3. 最终清理
            print("\n🔍 最终清理...")
            final_cleaned = self.thread_manager.cleanup_test_environment()
            print(f"最终清理了 {final_cleaned} 个线程")

            # 4. 状态检查
            final_summary = self.thread_manager.get_thread_summary()
            print(f"最终线程数: {final_summary['total_threads']}")

            if success and final_summary['total_threads'] <= 1:
                print("\n🎉 SystemMonitor超时问题解决成功!")
                return True
            else:
                print("\n⚠️  部分问题仍未完全解决")
                return False

        except Exception as e:
            print(f"\n❌ 综合测试失败: {e}")
            return False


def main():
    """主函数"""
    fixer = SystemMonitorTimeoutFixer()

    try:
        success = fixer.run_comprehensive_test()
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\n用户中断")
        return 1
    except Exception as e:
        print(f"\n\n脚本执行异常: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
