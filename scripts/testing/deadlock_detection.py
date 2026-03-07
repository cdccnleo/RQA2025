#!/usr/bin/env python3
"""
死锁检测脚本
检测和修复EnhancedDataIntegrationManager中的死锁问题
"""

from src.data.enhanced_integration_manager import EnhancedDataIntegrationManager, DataStreamConfig
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class DeadlockDetector:
    """死锁检测器"""

    def __init__(self, timeout=30):
        self.timeout = timeout
        self.detected_deadlock = False
        self.test_results = {}

    def run_with_timeout(self, func, *args, **kwargs):
        """带超时的函数执行"""
        result = None
        exception = None

        def target():
            nonlocal result, exception
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                exception = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            # 超时，可能存在死锁
            self.detected_deadlock = True
            return None, f"函数执行超时 ({self.timeout}秒)"

        return result, exception

    def test_performance_monitoring(self):
        """测试性能监控功能"""
        print("🔍 测试性能监控功能...")

        manager = None
        try:
            # 创建管理器
            manager = EnhancedDataIntegrationManager()

            # 记录性能指标
            manager.performance_monitor.record_metric("load_time", 1.5)
            manager.performance_monitor.record_metric("load_time", 2.1)
            manager.performance_monitor.record_metric("load_time", 0.8)

            # 测试获取性能指标（可能死锁的地方）
            result, error = self.run_with_timeout(manager.get_performance_metrics)

            if error:
                print(f"❌ 性能监控测试失败: {error}")
                return False

            # 验证结果
            assert "performance" in result
            assert "load_time" in result["performance"]
            load_time_stats = result["performance"]["load_time"]
            assert load_time_stats["count"] == 3

            print("✅ 性能监控测试通过")
            return True

        except Exception as e:
            print(f"❌ 性能监控测试异常: {e}")
            return False
        finally:
            if manager:
                manager.shutdown()

    def test_alert_management(self):
        """测试告警管理功能"""
        print("🔍 测试告警管理功能...")

        manager = None
        try:
            manager = EnhancedDataIntegrationManager()

            # 触发告警
            manager.alert_manager.trigger_alert("performance_warning", {"value": 15.0})

            # 测试获取告警历史（可能死锁的地方）
            result, error = self.run_with_timeout(manager.get_alert_history, hours=1)

            if error:
                print(f"❌ 告警管理测试失败: {error}")
                return False

            print("✅ 告警管理测试通过")
            return True

        except Exception as e:
            print(f"❌ 告警管理测试异常: {e}")
            return False
        finally:
            if manager:
                manager.shutdown()

    def test_concurrent_access(self):
        """测试并发访问"""
        print("🔍 测试并发访问...")

        manager = None
        try:
            manager = EnhancedDataIntegrationManager()

            # 创建多个线程同时访问
            def worker(worker_id):
                try:
                    # 记录指标
                    manager.performance_monitor.record_metric(f"worker_{worker_id}", worker_id)

                    # 获取性能指标
                    metrics = manager.get_performance_metrics()

                    # 获取告警历史
                    history = manager.get_alert_history(hours=1)

                    return True
                except Exception as e:
                    print(f"❌ 工作线程 {worker_id} 失败: {e}")
                    return False

            # 使用线程池执行并发测试
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(worker, i) for i in range(10)]

                # 等待所有任务完成或超时
                results = []
                for future in as_completed(futures, timeout=self.timeout):
                    results.append(future.result())

                success_count = sum(results)
                print(f"✅ 并发访问测试: {success_count}/10 成功")
                return success_count >= 8  # 至少80%成功

        except Exception as e:
            print(f"❌ 并发访问测试异常: {e}")
            return False
        finally:
            if manager:
                manager.shutdown()

    def test_data_stream_operations(self):
        """测试数据流操作"""
        print("🔍 测试数据流操作...")

        manager = None
        try:
            manager = EnhancedDataIntegrationManager()

            # 创建数据流
            stream_config = DataStreamConfig("test_stream", "test_data")
            stream_id = manager.create_data_stream(stream_config)

            # 启动数据流
            manager.start_data_stream(stream_id)

            # 获取性能指标（包含流信息）
            result, error = self.run_with_timeout(manager.get_performance_metrics)

            if error:
                print(f"❌ 数据流操作测试失败: {error}")
                return False

            # 验证流信息
            assert "streams" in result
            assert stream_id in result["streams"]

            # 停止数据流
            manager.stop_data_stream(stream_id)

            print("✅ 数据流操作测试通过")
            return True

        except Exception as e:
            print(f"❌ 数据流操作测试异常: {e}")
            return False
        finally:
            if manager:
                manager.shutdown()

    def run_all_tests(self):
        """运行所有死锁检测测试"""
        print("🔍 开始死锁检测测试")
        print("=" * 60)

        tests = [
            ("性能监控", self.test_performance_monitoring),
            ("告警管理", self.test_alert_management),
            ("并发访问", self.test_concurrent_access),
            ("数据流操作", self.test_data_stream_operations)
        ]

        results = {}
        for test_name, test_func in tests:
            print(f"\n📊 运行测试: {test_name}")
            start_time = time.time()

            try:
                success = test_func()
                duration = time.time() - start_time
                results[test_name] = {
                    'success': success,
                    'duration': duration,
                    'deadlock_detected': self.detected_deadlock
                }

                status = "✅ 通过" if success else "❌ 失败"
                print(f"  {status} - 耗时: {duration:.2f}秒")

            except Exception as e:
                duration = time.time() - start_time
                results[test_name] = {
                    'success': False,
                    'duration': duration,
                    'error': str(e),
                    'deadlock_detected': True
                }
                print(f"  ❌ 异常 - 耗时: {duration:.2f}秒 - {e}")

        # 生成报告
        self.generate_report(results)

        return results

    def generate_report(self, results):
        """生成死锁检测报告"""
        print("\n📈 死锁检测报告")
        print("=" * 60)

        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r['success'])
        deadlock_detected = any(r.get('deadlock_detected', False) for r in results.values())

        print(f"总测试数: {total_tests}")
        print(f"通过测试: {passed_tests}")
        print(f"失败测试: {total_tests - passed_tests}")
        print(f"死锁检测: {'是' if deadlock_detected else '否'}")

        print("\n详细结果:")
        for test_name, result in results.items():
            status = "✅ 通过" if result['success'] else "❌ 失败"
            deadlock = "⚠️ 死锁" if result.get('deadlock_detected', False) else "✅ 正常"
            print(f"  {test_name}: {status} - {deadlock} - {result['duration']:.2f}秒")

        if deadlock_detected:
            print("\n⚠️ 检测到潜在死锁问题，建议检查:")
            print("  1. 锁的获取顺序")
            print("  2. 是否存在循环等待")
            print("  3. 锁的超时机制")
            print("  4. 资源释放的完整性")
        else:
            print("\n✅ 未检测到死锁问题")


def main():
    """主函数"""
    print("🔍 EnhancedDataIntegrationManager 死锁检测")
    print("=" * 60)

    try:
        detector = DeadlockDetector(timeout=30)
        results = detector.run_all_tests()

        # 保存报告
        report_path = Path("reports/optimization/deadlock_detection_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 死锁检测报告\n\n")
            f.write(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            total_tests = len(results)
            passed_tests = sum(1 for r in results.values() if r['success'])
            deadlock_detected = any(r.get('deadlock_detected', False) for r in results.values())

            f.write(f"**总测试数**: {total_tests}\n")
            f.write(f"**通过测试**: {passed_tests}\n")
            f.write(f"**失败测试**: {total_tests - passed_tests}\n")
            f.write(f"**死锁检测**: {'是' if deadlock_detected else '否'}\n\n")

            f.write("## 详细结果\n\n")
            for test_name, result in results.items():
                status = "✅ 通过" if result['success'] else "❌ 失败"
                deadlock = "⚠️ 死锁" if result.get('deadlock_detected', False) else "✅ 正常"
                f.write(f"- **{test_name}**: {status} - {deadlock} - {result['duration']:.2f}秒\n")

        print(f"\n📄 报告已保存到: {report_path}")

    except Exception as e:
        print(f"\n❌ 死锁检测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
