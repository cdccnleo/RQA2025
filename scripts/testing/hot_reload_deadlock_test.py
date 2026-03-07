#!/usr/bin/env python3
# -*- coding: utf-8
"""
热重载服务死锁检测脚本

专门检测热重载服务中可能存在的死锁问题：
1. 文件监听器的线程安全
2. 配置回调的锁竞争
3. 服务启动/停止的同步问题
4. 资源清理的时序问题
"""

import os
import sys
import time
import threading
import tempfile
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Dict

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', '..', 'src')
sys.path.insert(0, src_dir)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HotReloadDeadlockTester:
    """热重载死锁测试器"""
    
    def __init__(self):
        self.test_results = []
        self.lock_contention_count = 0
        self.deadlock_detected = False
        self.test_timeout = 30  # 测试超时时间（秒）
        
    def test_basic_operations(self) -> bool:
        """测试基本操作是否会导致死锁"""
        logger.info("测试基本操作...")
        
        try:
            from infrastructure.core.config.services.hot_reload_service import HotReloadService
            
            # 创建服务实例
            service = HotReloadService()
            
            # 测试启动
            if not service.start():
                logger.error("服务启动失败")
                return False
            
            # 测试停止
            if not service.stop():
                logger.error("服务停止失败")
                return False
            
            logger.info("基本操作测试通过")
            return True
            
        except Exception as e:
            logger.error(f"基本操作测试失败: {e}")
            return False
    
    def test_concurrent_file_watching(self) -> bool:
        """测试并发文件监听是否会导致死锁"""
        logger.info("测试并发文件监听...")
        
        try:
            from infrastructure.core.config.services.hot_reload_service import HotReloadService
            
            # 创建临时配置文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                config_data = {"test": "value", "number": 42}
                json.dump(config_data, f, indent=2, ensure_ascii=False)
                config_file = f.name
            
            service = HotReloadService()
            
            # 启动服务
            if not service.start():
                logger.error("服务启动失败")
                return False
            
            # 并发监听多个文件
            def watch_file(file_path: str, callback):
                return service.watch_file(file_path, callback)
            
            def on_config_change(file_path: str, config: dict):
                logger.debug(f"配置变更: {file_path}")
                time.sleep(0.1)  # 模拟处理时间
            
            # 使用线程池并发监听
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for i in range(5):
                    future = executor.submit(watch_file, config_file, on_config_change)
                    futures.append(future)
                
                # 等待所有任务完成
                for future in as_completed(futures, timeout=10):
                    try:
                        result = future.result()
                        if not result:
                            logger.warning("文件监听失败")
                    except Exception as e:
                        logger.error(f"文件监听异常: {e}")
                        return False
            
            # 停止服务
            if not service.stop():
                logger.error("服务停止失败")
                return False
            
            # 清理临时文件
            Path(config_file).unlink()
            
            logger.info("并发文件监听测试通过")
            return True
            
        except Exception as e:
            logger.error(f"并发文件监听测试失败: {e}")
            return False
    
    def test_nested_lock_scenarios(self) -> bool:
        """测试嵌套锁场景是否会导致死锁"""
        logger.info("测试嵌套锁场景...")
        
        try:
            from infrastructure.core.config.services.hot_reload_service import HotReloadService
            
            service = HotReloadService()
            
            # 测试在持有锁时调用其他方法
            def test_nested_calls():
                # 启动服务
                service.start()
                
                # 在服务运行时获取状态
                status = service.get_status()
                logger.debug(f"服务状态: {status}")
                
                # 获取监听文件列表
                watched_files = service.get_watched_files()
                logger.debug(f"监听文件: {watched_files}")
                
                # 停止服务
                service.stop()
                
                return True
            
            # 在多个线程中执行嵌套调用
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for i in range(3):
                    future = executor.submit(test_nested_calls)
                    futures.append(future)
                
                # 等待所有任务完成
                for future in as_completed(futures, timeout=15):
                    try:
                        result = future.result()
                        if not result:
                            logger.warning("嵌套调用测试失败")
                            return False
                    except Exception as e:
                        logger.error(f"嵌套调用异常: {e}")
                        return False
            
            logger.info("嵌套锁场景测试通过")
            return True
            
        except Exception as e:
            logger.error(f"嵌套锁场景测试失败: {e}")
            return False
    
    def test_resource_cleanup_race_condition(self) -> bool:
        """测试资源清理的竞态条件"""
        logger.info("测试资源清理竞态条件...")
        
        try:
            from infrastructure.core.config.services.hot_reload_service import HotReloadService
            
            # 创建多个服务实例
            services = []
            for i in range(3):
                service = HotReloadService()
                services.append(service)
            
            # 并发启动和停止服务
            def start_stop_service(service_instance: HotReloadService):
                try:
                    # 启动服务
                    if service_instance.start():
                        time.sleep(0.1)  # 短暂运行
                        # 停止服务
                        service_instance.stop()
                        return True
                    return False
                except Exception as e:
                    logger.error(f"服务启动/停止异常: {e}")
                    return False
            
            # 并发执行
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for service in services:
                    future = executor.submit(start_stop_service, service)
                    futures.append(future)
                
                # 等待所有任务完成
                success_count = 0
                for future in as_completed(futures, timeout=20):
                    try:
                        result = future.result()
                        if result:
                            success_count += 1
                    except Exception as e:
                        logger.error(f"服务操作异常: {e}")
                
                if success_count != 3:
                    logger.warning(f"只有 {success_count}/3 个服务操作成功")
                    return False
            
            logger.info("资源清理竞态条件测试通过")
            return True
            
        except Exception as e:
            logger.error(f"资源清理竞态条件测试失败: {e}")
            return False
    
    def test_callback_deadlock(self) -> bool:
        """测试回调函数是否会导致死锁"""
        logger.info("测试回调函数死锁...")
        
        try:
            from infrastructure.core.config.services.hot_reload_service import HotReloadService
            
            # 创建临时配置文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                config_data = {"test": "value", "number": 42}
                json.dump(config_data, f, indent=2, ensure_ascii=False)
                config_file = f.name
            
            service = HotReloadService()
            
            # 创建一个可能导致死锁的回调函数
            callback_lock = threading.Lock()
            callback_called = False
            
            def problematic_callback(file_path: str, config: dict):
                nonlocal callback_called
                with callback_lock:
                    callback_called = True
                    logger.debug(f"回调被调用: {file_path}")
                    # 模拟长时间处理
                    time.sleep(0.5)
            
            # 启动服务
            if not service.start():
                logger.error("服务启动失败")
                return False
            
            # 监听文件
            if not service.watch_file(config_file, problematic_callback):
                logger.error("文件监听失败")
                return False
            
            # 修改配置文件触发回调
            with open(config_file, 'w', encoding='utf-8') as f:
                new_config = {"test": "new_value", "number": 100}
                json.dump(new_config, f, indent=2, ensure_ascii=False)
            
            # 等待回调执行
            time.sleep(1)
            
            # 检查回调是否被调用
            if not callback_called:
                logger.warning("回调函数未被调用")
            
            # 停止服务
            if not service.stop():
                logger.error("服务停止失败")
                return False
            
            # 清理临时文件
            Path(config_file).unlink()
            
            logger.info("回调函数死锁测试通过")
            return True
            
        except Exception as e:
            logger.error(f"回调函数死锁测试失败: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """运行所有死锁测试"""
        logger.info("开始运行热重载死锁测试...")
        
        tests = [
            ("基本操作", self.test_basic_operations),
            ("并发文件监听", self.test_concurrent_file_watching),
            ("嵌套锁场景", self.test_nested_lock_scenarios),
            ("资源清理竞态条件", self.test_resource_cleanup_race_condition),
            ("回调函数死锁", self.test_callback_deadlock)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"运行测试: {test_name}")
            start_time = time.time()
            
            try:
                result = test_func()
                results[test_name] = result
                
                elapsed_time = time.time() - start_time
                status = "通过" if result else "失败"
                logger.info(f"测试 {test_name}: {status} (耗时: {elapsed_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"测试 {test_name} 异常: {e}")
                results[test_name] = False
        
        return results
    
    def generate_report(self, results: Dict[str, bool]) -> str:
        """生成测试报告"""
        report = []
        report.append("# 热重载服务死锁检测报告")
        report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 测试结果汇总
        report.append("## 测试结果汇总")
        passed_count = sum(1 for result in results.values() if result)
        total_count = len(results)
        
        report.append(f"- 总测试数: {total_count}")
        report.append(f"- 通过测试: {passed_count}")
        report.append(f"- 失败测试: {total_count - passed_count}")
        report.append(f"- 通过率: {passed_count/total_count*100:.1f}%")
        report.append("")
        
        # 详细测试结果
        report.append("## 详细测试结果")
        for test_name, result in results.items():
            status = "✅ 通过" if result else "❌ 失败"
            report.append(f"- {test_name}: {status}")
        report.append("")
        
        # 问题分析
        failed_tests = [name for name, result in results.items() if not result]
        if failed_tests:
            report.append("## 问题分析")
            report.append("以下测试失败，可能存在死锁风险：")
            for test_name in failed_tests:
                report.append(f"- {test_name}")
            report.append("")
            
            report.append("### 建议修复措施")
            report.append("1. 检查锁的获取顺序，确保一致")
            report.append("2. 使用 `threading.RLock()` 替代 `threading.Lock()`")
            report.append("3. 为所有锁操作设置超时时间")
            report.append("4. 避免在持有锁时调用可能获取其他锁的方法")
            report.append("5. 确保资源清理的线程安全")
        else:
            report.append("## 问题分析")
            report.append("所有测试通过，未发现明显的死锁风险。")
            report.append("")
            report.append("### 预防建议")
            report.append("1. 继续监控锁的使用模式")
            report.append("2. 定期运行死锁检测")
            report.append("3. 在添加新功能时保持锁使用的一致性")
        
        return "\n".join(report)

def main():
    """主函数"""
    logger.info("启动热重载死锁检测...")
    
    # 创建测试器
    tester = HotReloadDeadlockTester()
    
    # 运行所有测试
    results = tester.run_all_tests()
    
    # 生成报告
    report = tester.generate_report(results)
    
    # 保存报告
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    report_file = project_root / "reports" / "testing" / "hot_reload_deadlock_report.md"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"死锁检测报告已保存到: {report_file}")
    
    # 打印测试结果
    passed_count = sum(1 for result in results.values() if result)
    total_count = len(results)
    
    if passed_count == total_count:
        logger.info("🎉 所有死锁测试通过！")
    else:
        logger.warning(f"⚠️  {total_count - passed_count} 个测试失败，存在潜在死锁风险")
    
    return passed_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
