#!/usr/bin/env python3
"""
基础设施层模块逐个测试脚本
依次测试各个模块，验证是否存在内存泄漏问题
"""

import os
import sys
import subprocess
import time
import psutil
import gc
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModuleTestResult:
    """模块测试结果"""
    module_name: str
    test_file: str
    success: bool
    memory_before: float
    memory_after: float
    memory_growth: float
    execution_time: float
    error_message: str = ""
    leak_detected: bool = False


class ModuleByModuleTester:
    """模块逐个测试器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.test_dir = self.project_root / "tests" / "unit" / "infrastructure"
        self.results: List[ModuleTestResult] = []
        
        # 定义要测试的模块及其对应的测试文件
        self.module_test_mapping = {
            # 核心模块
            "init_infrastructure": "test_init_infrastructure.py",
            "application_monitor": "test_application_monitor.py", 
            "system_monitor": "test_system_monitor.py",
            "unified_config_manager": "test_unified_config_manager.py",
            
            # 监控模块
            "visual_monitor": "test_visual_monitor.py",
            "monitoring": "test_monitoring.py",
            "performance_monitor": "test_performance_monitor.py",
            "model_monitor": "test_model_monitor.py",
            "network_monitor": "test_network_monitor.py",
            "backtest_monitor": "test_backtest_monitor.py",
            "prometheus_monitor": "test_prometheus_monitor.py",
            
            # 配置模块
            "config_coverage": "test_config_coverage.py",
            "config_version": "test_config_version.py",
            "config_loader_service": "test_config_loader_service.py",
            "yaml_loader": "test_yaml_loader.py",
            "json_loader": "test_json_loader.py",
            "env_loader": "test_env_loader.py",
            
            # 错误处理模块
            "error_handler": "test_error_handler.py",
            "persistent_error_handler": "test_persistent_error_handler_test.py",
            "circuit_breaker": "test_circuit_breaker.py",
            "circuit_breaker_fixed": "test_circuit_breaker_fixed.py",
            "circuit_breaker_manager": "test_circuit_breaker_manager.py",
            
            # 服务模块
            "service_launcher": "test_service_launcher.py",
            "deployment_validator": "test_deployment_validator.py",
            "deployment_manager": "test_deployment_manager.py",
            "final_deployment_check": "test_final_deployment_check.py",
            
            # 数据模块
            "data_sync": "test_data_sync.py",
            "event": "test_event.py",
            "event_service": "test_event_service.py",
            
            # 资源管理模块
            "resource_manager": "test_resource_manager.py",
            "gpu_manager": "test_gpu_manager.py",
            "quota_manager": "test_quota_manager.py",
            "connection_pool": "test_connection_pool.py",
            "load_balancer": "test_load_balancer.py",
            
            # 工具模块
            "lock": "test_lock.py",
            "thread_management": "test_thread_management.py",
            "logging_coverage": "test_logging_coverage.py",
            "metrics_collector": "test_metrics_collector.py",
            
            # 其他模块
            "version": "test_version.py",
            "disaster_recovery": "test_disaster_recovery.py",
            "degradation_manager": "test_degradation_manager.py",
            "alert_manager": "test_alert_manager.py",
            "notification": "test_notification.py",
            "validators": "test_validators.py",
            "schema_validator": "test_schema_validator.py",
            "standard_interfaces": "test_standard_interfaces.py",
            "unified_interface_manager": "test_unified_interface_manager.py",
            "unified_hot_reload": "test_unified_hot_reload.py",
            "optimization_modules": "test_optimization_modules.py",
            "coverage_improvement": "test_coverage_improvement.py",
            "coverage_boost": "test_coverage_boost.py",
            "document_management": "test_document_management.py",
            "market_aware_retry": "test_market_aware_retry_test.py",
            "regulatory_compliance": "test_regulatory_compliance.py",
            "regulatory_reporter": "test_regulatory_reporter.py",
            "infrastructure_core": "test_infrastructure_core.py",
            "infrastructure": "test_infrastructure.py",
            "minimal_infra_main_flow": "test_minimal_infra_main_flow.py",
            "app_factory": "test_app_factory.py",
            "async_inference_engine": "test_async_inference_engine.py",
            "async_inference_engine_top20": "test_async_inference_engine_top20.py",
            "factory": "test_factory.py",
            "resource_dashboard": "test_resource_dashboard.py",
        }
    
    def get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def run_aggressive_cleanup(self):
        """运行激进内存清理"""
        print("🧹 运行激进内存清理...")
        
        # 强制垃圾回收
        for i in range(5):
            collected = gc.collect()
            if collected > 0:
                print(f"✅ 第{i+1}次垃圾回收: 清理了 {collected} 个对象")
        
        # 清理模块缓存
        modules_to_clear = []
        for name in list(sys.modules.keys()):
            if any(keyword in name.lower() for keyword in ['infrastructure', 'monitoring', 'config', 'cache']):
                modules_to_clear.append(name)
        
        for module_name in modules_to_clear:
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        if modules_to_clear:
            print(f"🧹 清理了 {len(modules_to_clear)} 个模块缓存")
        
        # 清理Prometheus注册表
        try:
            from prometheus_client import REGISTRY
            if hasattr(REGISTRY, '_names_to_collectors'):
                # 只清理非系统指标
                system_metrics = [
                    'python_gc_objects_collected', 'python_gc_objects_collected_total',
                    'python_gc_objects_collected_created', 'python_gc_objects_uncollectable',
                    'python_gc_objects_uncollectable_total', 'python_gc_objects_uncollectable_created',
                    'python_gc_collections', 'python_gc_collections_total',
                    'python_gc_collections_created', 'python_info'
                ]
                
                metrics_to_remove = []
                for metric_name in REGISTRY._names_to_collectors.keys():
                    if metric_name not in system_metrics:
                        metrics_to_remove.append(metric_name)
                
                for metric_name in metrics_to_remove:
                    del REGISTRY._names_to_collectors[metric_name]
                
                if metrics_to_remove:
                    print(f"🧹 清理Prometheus注册表: {len(metrics_to_remove)} 个非系统指标")
        except Exception as e:
            print(f"⚠️  Prometheus清理失败: {e}")
    
    def test_single_module(self, module_name: str, test_file: str) -> ModuleTestResult:
        """测试单个模块"""
        print(f"\n🔍 测试模块: {module_name}")
        print(f"   测试文件: {test_file}")
        
        # 检查测试文件是否存在
        test_file_path = self.test_dir / test_file
        if not test_file_path.exists():
            return ModuleTestResult(
                module_name=module_name,
                test_file=test_file,
                success=False,
                memory_before=0,
                memory_after=0,
                memory_growth=0,
                execution_time=0,
                error_message=f"测试文件不存在: {test_file_path}"
            )
        
        # 运行激进清理
        self.run_aggressive_cleanup()
        
        # 记录测试前内存
        memory_before = self.get_memory_usage()
        start_time = time.time()
        
        try:
            # 运行测试
            cmd = [
                sys.executable, "-m", "pytest", 
                str(test_file_path),
                "-v",
                "--tb=short",
                "--disable-warnings",
                "--no-header",
                "--no-summary"
            ]
            
            print(f"   执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            execution_time = time.time() - start_time
            
            # 记录测试后内存
            memory_after = self.get_memory_usage()
            memory_growth = memory_after - memory_before
            
            # 判断是否成功
            success = result.returncode == 0
            
            # 判断是否有内存泄漏（增长超过50MB）
            leak_detected = memory_growth > 50
            
            error_message = ""
            if not success:
                error_message = result.stderr if result.stderr else result.stdout
            
            print(f"   ✅ 测试完成")
            print(f"   内存变化: {memory_before:.2f}MB -> {memory_after:.2f}MB (增长: {memory_growth:.2f}MB)")
            print(f"   执行时间: {execution_time:.2f}秒")
            print(f"   成功: {success}")
            print(f"   内存泄漏: {'是' if leak_detected else '否'}")
            
            if leak_detected:
                print(f"   ⚠️  检测到内存泄漏！")
            
            return ModuleTestResult(
                module_name=module_name,
                test_file=test_file,
                success=success,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_growth=memory_growth,
                execution_time=execution_time,
                error_message=error_message,
                leak_detected=leak_detected
            )
            
        except subprocess.TimeoutExpired:
            return ModuleTestResult(
                module_name=module_name,
                test_file=test_file,
                success=False,
                memory_before=memory_before,
                memory_after=self.get_memory_usage(),
                memory_growth=self.get_memory_usage() - memory_before,
                execution_time=300,
                error_message="测试超时（5分钟）",
                leak_detected=True
            )
        except Exception as e:
            return ModuleTestResult(
                module_name=module_name,
                test_file=test_file,
                success=False,
                memory_before=memory_before,
                memory_after=self.get_memory_usage(),
                memory_growth=self.get_memory_usage() - memory_before,
                execution_time=time.time() - start_time,
                error_message=str(e),
                leak_detected=True
            )
    
    def run_all_tests(self) -> List[ModuleTestResult]:
        """运行所有模块测试"""
        print("🚀 开始基础设施层模块逐个测试")
        print("=" * 60)
        
        for module_name, test_file in self.module_test_mapping.items():
            result = self.test_single_module(module_name, test_file)
            self.results.append(result)
            
            # 如果检测到内存泄漏，立即停止
            if result.leak_detected:
                print(f"\n❌ 检测到内存泄漏，停止测试")
                print(f"   泄漏模块: {module_name}")
                print(f"   内存增长: {result.memory_growth:.2f}MB")
                break
        
        return self.results
    
    def generate_report(self) -> str:
        """生成测试报告"""
        if not self.results:
            return "没有测试结果"
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        leak_tests = sum(1 for r in self.results if r.leak_detected)
        
        total_memory_growth = sum(r.memory_growth for r in self.results)
        avg_execution_time = sum(r.execution_time for r in self.results) / total_tests
        
        report = f"""
📊 基础设施层模块测试报告
{'=' * 60}

📈 总体统计:
   总测试数: {total_tests}
   成功测试: {successful_tests}
   失败测试: {failed_tests}
   内存泄漏: {leak_tests}
   总内存增长: {total_memory_growth:.2f}MB
   平均执行时间: {avg_execution_time:.2f}秒

📋 详细结果:
"""
        
        for result in self.results:
            status = "✅" if result.success else "❌"
            leak_status = "⚠️" if result.leak_detected else "✅"
            
            report += f"""
{status} {result.module_name} ({result.test_file})
   成功: {result.success}
   内存泄漏: {leak_status} ({result.memory_growth:.2f}MB)
   执行时间: {result.execution_time:.2f}秒
"""
            
            if result.error_message:
                report += f"   错误: {result.error_message}\n"
        
        # 内存泄漏详情
        leak_results = [r for r in self.results if r.leak_detected]
        if leak_results:
            report += f"\n🚨 内存泄漏详情:\n"
            for result in leak_results:
                report += f"   - {result.module_name}: {result.memory_growth:.2f}MB\n"
        
        return report


def main():
    """主函数"""
    project_root = os.getcwd()
    
    print("🔧 初始化模块逐个测试器...")
    tester = ModuleByModuleTester(project_root)
    
    # 运行所有测试
    results = tester.run_all_tests()
    
    # 生成报告
    report = tester.generate_report()
    print(report)
    
    # 保存报告
    report_file = Path(project_root) / "reports" / "module_by_module_test_report.md"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 报告已保存: {report_file}")
    
    # 返回状态码
    leak_count = sum(1 for r in results if r.leak_detected)
    if leak_count > 0:
        print(f"\n❌ 检测到 {leak_count} 个模块存在内存泄漏")
        return 1
    else:
        print(f"\n✅ 所有模块测试通过，未检测到内存泄漏")
        return 0


if __name__ == "__main__":
    sys.exit(main())

