#!/usr/bin/env python3
"""
简单死锁测试脚本
避免复杂依赖，直接测试核心功能
"""

import os
import sys
import time
import threading
import gc
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 设置环境变量避免复杂导入
os.environ['DISABLE_HEAVY_IMPORTS'] = 'true'
os.environ['LIGHTWEIGHT_TEST'] = 'true'


def test_basic_manager_creation():
    """测试基本管理器创建"""
    print("🔍 测试基本管理器创建...")
    
    try:
        # 只导入核心模块
        from src.data.enhanced_integration_manager import EnhancedDataIntegrationManager
        
        # 创建管理器
        manager = EnhancedDataIntegrationManager()
        
        # 基本功能测试
        manager.register_node("test_node", "127.0.0.1", 8080)
        
        # 获取性能指标
        metrics = manager.get_performance_metrics()
        print(f"✅ 基本管理器创建成功，指标: {len(metrics)} 项")
        
        # 清理
        manager.shutdown()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ 基本管理器创建失败: {e}")
        return False


def test_performance_monitoring_simple():
    """简单性能监控测试"""
    print("🔍 测试简单性能监控...")
    
    try:
        from src.data.enhanced_integration_manager import EnhancedDataIntegrationManager
        
        manager = None
        try:
            manager = EnhancedDataIntegrationManager()
            
            # 记录简单指标
            manager.performance_monitor.record_metric("test_metric", 1.0)
            
            # 获取指标
            metrics = manager.get_performance_metrics()
            
            if "performance" in metrics and "test_metric" in metrics["performance"]:
                print("✅ 简单性能监控测试通过")
                return True
            else:
                print("❌ 性能指标获取失败")
                return False
                
        finally:
            if manager:
                manager.shutdown()
                gc.collect()
                
    except Exception as e:
        print(f"❌ 简单性能监控测试失败: {e}")
        return False


def test_alert_management_simple():
    """简单告警管理测试"""
    print("🔍 测试简单告警管理...")
    
    try:
        from src.data.enhanced_integration_manager import EnhancedDataIntegrationManager
        
        manager = None
        try:
            manager = EnhancedDataIntegrationManager()
            
            # 触发简单告警
            manager.alert_manager.trigger_alert("performance_warning", {"value": 10.0})
            
            # 获取告警历史
            history = manager.get_alert_history(hours=1)
            
            print(f"✅ 简单告警管理测试通过，历史记录: {len(history)} 条")
            return True
            
        finally:
            if manager:
                manager.shutdown()
                gc.collect()
                
    except Exception as e:
        print(f"❌ 简单告警管理测试失败: {e}")
        return False


def test_concurrent_access_simple():
    """简单并发访问测试"""
    print("🔍 测试简单并发访问...")
    
    try:
        from src.data.enhanced_integration_manager import EnhancedDataIntegrationManager
        
        manager = None
        try:
            manager = EnhancedDataIntegrationManager()
            
            # 创建多个线程
            def worker(worker_id):
                try:
                    # 记录指标
                    manager.performance_monitor.record_metric(f"worker_{worker_id}", worker_id)
                    
                    # 获取指标
                    metrics = manager.get_performance_metrics()
                    
                    return True
                except Exception as e:
                    print(f"❌ 工作线程 {worker_id} 失败: {e}")
                    return False
            
            # 使用简单线程池
            threads = []
            results = []
            
            for i in range(5):
                thread = threading.Thread(target=lambda: results.append(worker(i)))
                thread.daemon = True
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join(timeout=10)
            
            success_count = sum(results)
            print(f"✅ 简单并发访问测试: {success_count}/5 成功")
            return success_count >= 4
            
        finally:
            if manager:
                manager.shutdown()
                gc.collect()
                
    except Exception as e:
        print(f"❌ 简单并发访问测试失败: {e}")
        return False


def test_memory_cleanup():
    """测试内存清理"""
    print("🔍 测试内存清理...")
    
    try:
        from src.data.enhanced_integration_manager import EnhancedDataIntegrationManager
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # 创建多个管理器
        managers = []
        for i in range(3):
            manager = EnhancedDataIntegrationManager()
            managers.append(manager)
            
            # 执行一些操作
            manager.register_node(f"node_{i}", "127.0.0.1", 8080 + i)
            manager.performance_monitor.record_metric(f"test_{i}", i)
            
            # 立即关闭
            manager.shutdown()
            gc.collect()
        
        # 检查内存使用
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"✅ 内存清理测试: 内存增加 {memory_increase:.2f} MB")
        return memory_increase < 50  # 内存增加应该小于50MB
        
    except Exception as e:
        print(f"❌ 内存清理测试失败: {e}")
        return False


def main():
    """主函数"""
    print("🔍 EnhancedDataIntegrationManager 简单死锁测试")
    print("=" * 60)
    
    tests = [
        ("基本管理器创建", test_basic_manager_creation),
        ("简单性能监控", test_performance_monitoring_simple),
        ("简单告警管理", test_alert_management_simple),
        ("简单并发访问", test_concurrent_access_simple),
        ("内存清理", test_memory_cleanup)
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
                'duration': duration
            }
            
            status = "✅ 通过" if success else "❌ 失败"
            print(f"  {status} - 耗时: {duration:.2f}秒")
            
        except Exception as e:
            duration = time.time() - start_time
            results[test_name] = {
                'success': False,
                'duration': duration,
                'error': str(e)
            }
            print(f"  ❌ 异常 - 耗时: {duration:.2f}秒 - {e}")
    
    # 生成报告
    print("\n📈 简单死锁测试报告")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r['success'])
    
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    
    print("\n详细结果:")
    for test_name, result in results.items():
        status = "✅ 通过" if result['success'] else "❌ 失败"
        print(f"  {test_name}: {status} - {result['duration']:.2f}秒")
    
    if passed_tests == total_tests:
        print("\n✅ 所有测试通过，未检测到死锁问题")
    else:
        print("\n⚠️ 部分测试失败，可能存在死锁问题")
    
    # 保存报告
    report_path = Path("reports/optimization/simple_deadlock_test_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 简单死锁测试报告\n\n")
        f.write(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**总测试数**: {total_tests}\n")
        f.write(f"**通过测试**: {passed_tests}\n")
        f.write(f"**失败测试**: {total_tests - passed_tests}\n\n")
        
        f.write("## 详细结果\n\n")
        for test_name, result in results.items():
            status = "✅ 通过" if result['success'] else "❌ 失败"
            f.write(f"- **{test_name}**: {status} - {result['duration']:.2f}秒\n")
    
    print(f"\n📄 报告已保存到: {report_path}")


if __name__ == "__main__":
    main() 