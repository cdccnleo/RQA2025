#!/usr/bin/env python3
"""
快速基础设施层模块测试脚本
直接测试核心模块，验证内存泄漏问题
"""

import os
import sys
import time
import psutil
import gc
import importlib
from typing import List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QuickTestResult:
    """快速测试结果"""
    module_name: str
    success: bool
    memory_before: float
    memory_after: float
    memory_growth: float
    execution_time: float
    error_message: str = ""
    leak_detected: bool = False


class QuickModuleTester:
    """快速模块测试器"""
    
    def __init__(self):
        self.results: List[QuickTestResult] = []
        
        # 定义要测试的核心模块（只测试最重要的几个）
        self.core_modules = [
            "src.infrastructure.init_infrastructure",
            "src.infrastructure.monitoring.application_monitor",
            "src.infrastructure.monitoring.system_monitor", 
            "src.infrastructure.config.unified_config_manager",
        ]
    
    def get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def run_cleanup(self):
        """运行内存清理"""
        print("🧹 运行内存清理...")
        
        # 强制垃圾回收
        for i in range(3):
            collected = gc.collect()
            if collected > 0:
                print(f"✅ 第{i+1}次垃圾回收: 清理了 {collected} 个对象")
        
        # 清理模块缓存
        modules_to_clear = []
        for name in list(sys.modules.keys()):
            if any(keyword in name.lower() for keyword in ['infrastructure', 'monitoring', 'config']):
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
    
    def test_single_module(self, module_name: str) -> QuickTestResult:
        """测试单个模块"""
        print(f"\n🔍 测试模块: {module_name}")
        
        # 运行清理
        self.run_cleanup()
        
        # 记录测试前内存
        memory_before = self.get_memory_usage()
        start_time = time.time()
        
        try:
            # 导入模块
            print(f"   导入模块: {module_name}")
            module = importlib.import_module(module_name)
            
            # 尝试创建实例（如果模块有主要类）
            instances = []
            for attr_name in dir(module):
                attr = getattr(module, attr_name, None)
                if (isinstance(attr, type) and 
                    hasattr(attr, '__init__') and 
                    not attr_name.startswith('_')):
                    try:
                        # 尝试创建实例
                        instance = attr()
                        instances.append(instance)
                        print(f"   创建实例: {attr_name}")
                    except Exception as e:
                        print(f"   跳过实例创建 {attr_name}: {e}")
            
            execution_time = time.time() - start_time
            
            # 记录测试后内存
            memory_after = self.get_memory_usage()
            memory_growth = memory_after - memory_before
            
            # 判断是否有内存泄漏（增长超过20MB）
            leak_detected = memory_growth > 20
            
            print(f"   ✅ 模块测试完成")
            print(f"   内存变化: {memory_before:.2f}MB -> {memory_after:.2f}MB (增长: {memory_growth:.2f}MB)")
            print(f"   执行时间: {execution_time:.2f}秒")
            print(f"   内存泄漏: {'是' if leak_detected else '否'}")
            
            if leak_detected:
                print(f"   ⚠️  检测到内存泄漏！")
            
            return QuickTestResult(
                module_name=module_name,
                success=True,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_growth=memory_growth,
                execution_time=execution_time,
                leak_detected=leak_detected
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            memory_after = self.get_memory_usage()
            memory_growth = memory_after - memory_before
            
            print(f"   ❌ 模块测试失败: {e}")
            print(f"   内存变化: {memory_before:.2f}MB -> {memory_after:.2f}MB (增长: {memory_growth:.2f}MB)")
            print(f"   执行时间: {execution_time:.2f}秒")
            
            return QuickTestResult(
                module_name=module_name,
                success=False,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_growth=memory_growth,
                execution_time=execution_time,
                error_message=str(e),
                leak_detected=memory_growth > 20
            )
    
    def run_all_tests(self) -> List[QuickTestResult]:
        """运行所有模块测试"""
        print("🚀 开始基础设施层快速模块测试")
        print("=" * 60)
        
        for module_name in self.core_modules:
            result = self.test_single_module(module_name)
            self.results.append(result)
            
            # 如果检测到内存泄漏，记录但继续测试
            if result.leak_detected:
                print(f"   ⚠️  模块 {module_name} 存在内存泄漏")
        
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
📊 基础设施层快速模块测试报告
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
{status} {result.module_name}
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
    print("🔧 初始化快速模块测试器...")
    tester = QuickModuleTester()
    
    # 运行所有测试
    results = tester.run_all_tests()
    
    # 生成报告
    report = tester.generate_report()
    print(report)
    
    # 保存报告
    report_file = Path("reports") / "quick_module_test_report.md"
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
