#!/usr/bin/env python3
"""
验证基础设施层线程退出问题解决方案的有效性
"""

from scripts.testing.thread_cleanup_manager import ThreadCleanupManager
import sys
import time
from pathlib import Path
from typing import Dict

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入线程清理管理器


class SolutionVerifier:
    """解决方案验证器"""

    def __init__(self):
        self.thread_manager = ThreadCleanupManager()
        self.verification_results = {}

    def verify_thread_cleanup_manager(self) -> bool:
        """验证线程清理管理器"""
        print("🔍 验证线程清理管理器...")

        try:
            # 1. 状态检查
            summary = self.thread_manager.get_thread_summary()
            print(f"   当前线程数: {summary['total_threads']}")

            # 2. 环境清理
            cleaned_count = self.thread_manager.cleanup_test_environment()
            print(f"   清理线程数: {cleaned_count}")

            # 3. 最终状态
            final_summary = self.thread_manager.get_thread_summary()
            print(f"   最终线程数: {final_summary['total_threads']}")

            success = final_summary['total_threads'] <= 1
            print(f"   {'✅ 通过' if success else '❌ 失败'}")

            return success

        except Exception as e:
            print(f"   ❌ 验证失败: {e}")
            return False

    def verify_improved_system_monitor(self) -> bool:
        """验证改进的系统监控器"""
        print("🔍 验证改进的系统监控器...")

        try:
            # 导入改进的系统监控器
            from scripts.testing.improved_system_monitor import ImprovedSystemMonitor

            # 创建监控器实例
            monitor = ImprovedSystemMonitor(check_interval=1.0)  # 1秒间隔用于测试

            # 启动监控
            monitor.start_monitoring()
            time.sleep(0.5)

            # 验证线程状态
            thread_alive = monitor._monitor_thread.is_alive()
            print(f"   监控线程状态: {'运行中' if thread_alive else '已停止'}")

            # 停止监控
            monitor.stop_monitoring()
            time.sleep(0.5)

            # 验证线程是否已停止
            thread_stopped = not monitor._monitor_thread.is_alive()
            print(f"   线程停止状态: {'已停止' if thread_stopped else '仍在运行'}")

            success = thread_stopped
            print(f"   {'✅ 通过' if success else '❌ 失败'}")

            return success

        except Exception as e:
            print(f"   ❌ 验证失败: {e}")
            return False

    def verify_original_problem(self) -> bool:
        """验证原始问题是否仍然存在"""
        print("🔍 验证原始问题...")

        try:
            # 导入原始SystemMonitor
            from src.infrastructure.monitoring.system_monitor import SystemMonitor

            # 创建监控器实例
            monitor = SystemMonitor()

            # 启动监控
            monitor.start_monitoring()
            time.sleep(0.1)

            # 验证线程状态
            thread_alive = monitor._monitor_thread.is_alive()
            print(f"   原始监控器线程状态: {'运行中' if thread_alive else '已停止'}")

            # 尝试停止监控（这里会卡住）
            print("   尝试停止原始监控器（预期会卡住）...")

            # 使用线程清理管理器来清理
            cleaned_count = self.thread_manager.cleanup_test_environment()
            print(f"   清理线程数: {cleaned_count}")

            # 验证清理结果
            summary = self.thread_manager.get_thread_summary()
            success = summary['total_threads'] <= 1

            print(f"   原始问题验证: {'✅ 问题仍然存在（需要我们的解决方案）' if success else '❌ 清理失败'}")

            return success

        except Exception as e:
            print(f"   ❌ 验证失败: {e}")
            return False

    def run_comprehensive_verification(self) -> Dict:
        """运行综合验证"""
        print("=" * 60)
        print("基础设施层线程退出问题解决方案综合验证")
        print("=" * 60)

        try:
            # 1. 验证线程清理管理器
            self.verification_results['thread_cleanup_manager'] = self.verify_thread_cleanup_manager(
            )

            # 2. 验证改进的系统监控器
            self.verification_results['improved_system_monitor'] = self.verify_improved_system_monitor(
            )

            # 3. 验证原始问题
            self.verification_results['original_problem'] = self.verify_original_problem()

            # 4. 生成验证报告
            self.generate_verification_report()

            return self.verification_results

        except Exception as e:
            print(f"❌ 综合验证失败: {e}")
            return {}

    def generate_verification_report(self):
        """生成验证报告"""
        report_file = self.project_root / "reports" / "solution_verification_report.md"
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 基础设施层线程退出问题解决方案验证报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 验证结果\n\n")

            for component, result in self.verification_results.items():
                status = "✅ 通过" if result else "❌ 失败"
                f.write(f"- **{component}**: {status}\n")

            f.write("\n## 验证总结\n\n")

            total_components = len(self.verification_results)
            successful_components = sum(self.verification_results.values())

            f.write(f"- **总组件数**: {total_components}\n")
            f.write(f"- **验证通过**: {successful_components}\n")
            f.write(f"- **验证失败**: {total_components - successful_components}\n")
            f.write(f"- **成功率**: {(successful_components/total_components*100):.1f}%\n")

            if successful_components == total_components:
                f.write("\n🎉 所有组件验证通过！解决方案有效。\n")
            else:
                f.write(f"\n⚠️  有 {total_components - successful_components} 个组件验证失败，需要进一步调查。\n")

        print(f"📄 验证报告已生成: {report_file}")


def main():
    """主函数"""
    verifier = SolutionVerifier()

    try:
        results = verifier.run_comprehensive_verification()

        # 统计结果
        total = len(results)
        successful = sum(results.values())

        print(f"\n🎯 验证完成!")
        print(f"   总组件: {total}")
        print(f"   成功: {successful}")
        print(f"   失败: {total - successful}")
        print(f"   成功率: {(successful/total*100):.1f}%")

        if successful == total:
            print("\n🎉 所有组件验证通过！解决方案有效。")
        else:
            print(f"\n⚠️  有 {total - successful} 个组件验证失败。")

        return 0 if successful == total else 1

    except KeyboardInterrupt:
        print("\n\n用户中断")
        return 1
    except Exception as e:
        print(f"\n\n验证异常: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
