#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据层内存优化增强脚本
实现自动内存监控告警、建立内存使用基线标准、优化数据结构清理策略
"""

import os
import sys
import gc
import psutil
import time
import json
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.logger import get_logger
    from src.data.cache.cache_manager import CacheManager
    from src.data.quality.data_quality_monitor import DataQualityMonitor
except ImportError as e:
    print(f"导入模块失败: {e}")
    # 创建模拟类用于测试

    class MockLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def warning(self, msg): print(f"[WARNING] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
        def debug(self, msg): print(f"[DEBUG] {msg}")

    class MockCacheManager:
        def clear_expired_cache(self): pass
        def get_cache_size(self): return 0

    class MockDataQualityMonitor:
        pass

    def get_logger(name):
        return MockLogger()
    CacheManager = MockCacheManager
    DataQualityMonitor = MockDataQualityMonitor


@dataclass
class MemoryBaseline:
    """内存使用基线数据"""
    timestamp: str
    memory_usage_mb: float
    memory_percent: float
    available_memory_mb: float
    cache_size_mb: float
    gc_objects: int
    module_count: int


@dataclass
class MemoryAlert:
    """内存告警数据"""
    timestamp: str
    alert_type: str
    current_value: float
    threshold_value: float
    message: str
    severity: str


class EnhancedMemoryOptimizer:
    """
    增强版内存优化器
    实现自动内存监控告警、建立内存使用基线标准、优化数据结构清理策略
    """

    def __init__(self,
                 memory_threshold_percent: float = 80.0,
                 cache_threshold_mb: float = 500.0,
                 gc_threshold_objects: int = 10000,
                 monitoring_interval: int = 30):
        """
        初始化内存优化器

        Args:
            memory_threshold_percent: 内存使用率告警阈值
            cache_threshold_mb: 缓存大小告警阈值(MB)
            gc_threshold_objects: GC对象数量告警阈值
            monitoring_interval: 监控间隔(秒)
        """
        self.logger = get_logger("enhanced_memory_optimizer")
        self.memory_threshold_percent = memory_threshold_percent
        self.cache_threshold_mb = cache_threshold_mb
        self.gc_threshold_objects = gc_threshold_objects
        self.monitoring_interval = monitoring_interval

        # 初始化组件
        try:
            from src.data.cache.cache_manager import CacheConfig
            cache_config = CacheConfig(
                max_size=1000,
                ttl=3600,
                enable_disk_cache=True,
                disk_cache_dir='cache',
                compression=False,
                encryption=False,
                enable_stats=True,
                cleanup_interval=300,
                max_file_size=10 * 1024 * 1024,
                backup_enabled=False,
                backup_interval=3600
            )
            self.cache_manager = CacheManager(config=cache_config)
            self.quality_monitor = DataQualityMonitor()
        except Exception as e:
            self.logger.warning(f"初始化组件失败，使用模拟组件: {e}")
            # 使用模拟组件

            class MockCacheManager:
                def clear_expired_cache(self): pass
                def get_cache_size(self): return 0

            class MockDataQualityMonitor:
                pass

            self.cache_manager = MockCacheManager()
            self.quality_monitor = MockDataQualityMonitor()

        # 内存基线数据
        self.baseline_data: List[MemoryBaseline] = []
        self.alert_history: List[MemoryAlert] = []

        # 监控状态
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # 性能统计
        self.optimization_stats = {
            'total_optimizations': 0,
            'memory_reduced_mb': 0.0,
            'cache_cleared_count': 0,
            'gc_collections': 0,
            'alerts_generated': 0
        }

        self.logger.info("增强版内存优化器初始化完成")

    def establish_memory_baseline(self) -> MemoryBaseline:
        """
        建立内存使用基线标准

        Returns:
            MemoryBaseline: 内存基线数据
        """
        self.logger.info("开始建立内存使用基线标准...")

        # 获取当前内存状态
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

        # 获取系统内存信息
        system_memory = psutil.virtual_memory()
        available_memory_mb = system_memory.available / (1024 * 1024)

        # 获取缓存大小
        cache_size_mb = self._get_cache_size_mb()

        # 获取GC对象数量
        gc_objects = len(gc.get_objects())

        # 统计模块数量
        module_count = len([name for name in sys.modules.keys() if name.startswith('src.data')])

        baseline = MemoryBaseline(
            timestamp=datetime.now().isoformat(),
            memory_usage_mb=memory_info.rss / (1024 * 1024),
            memory_percent=memory_percent,
            available_memory_mb=available_memory_mb,
            cache_size_mb=cache_size_mb,
            gc_objects=gc_objects,
            module_count=module_count
        )

        self.baseline_data.append(baseline)
        self.logger.info(
            f"内存基线建立完成: 使用{baseline.memory_usage_mb:.2f}MB, 占比{baseline.memory_percent:.2f}%")

        return baseline

    def _get_cache_size_mb(self) -> float:
        """获取缓存大小(MB)"""
        try:
            return self.cache_manager.get_cache_size() / (1024 * 1024)
        except:
            return 0.0

    def start_auto_monitoring(self) -> None:
        """启动自动内存监控告警"""
        if self.is_monitoring:
            self.logger.warning("内存监控已在运行中")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitor_thread.start()
        self.logger.info(f"自动内存监控已启动，监控间隔: {self.monitoring_interval}秒")

    def stop_auto_monitoring(self) -> None:
        """停止自动内存监控告警"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("自动内存监控已停止")

    def _monitor_memory(self) -> None:
        """内存监控循环"""
        while self.is_monitoring:
            try:
                self._check_memory_status()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"内存监控异常: {e}")
                time.sleep(5)

    def _check_memory_status(self) -> None:
        """检查内存状态并生成告警"""
        process = psutil.Process()
        memory_percent = process.memory_percent()
        cache_size_mb = self._get_cache_size_mb()
        gc_objects = len(gc.get_objects())

        # 检查内存使用率
        if memory_percent > self.memory_threshold_percent:
            alert = MemoryAlert(
                timestamp=datetime.now().isoformat(),
                alert_type="MEMORY_HIGH_USAGE",
                current_value=memory_percent,
                threshold_value=self.memory_threshold_percent,
                message=f"内存使用率过高: {memory_percent:.2f}% > {self.memory_threshold_percent}%",
                severity="WARNING"
            )
            self._handle_alert(alert)

        # 检查缓存大小
        if cache_size_mb > self.cache_threshold_mb:
            alert = MemoryAlert(
                timestamp=datetime.now().isoformat(),
                alert_type="CACHE_LARGE_SIZE",
                current_value=cache_size_mb,
                threshold_value=self.cache_threshold_mb,
                message=f"缓存大小过大: {cache_size_mb:.2f}MB > {self.cache_threshold_mb}MB",
                severity="WARNING"
            )
            self._handle_alert(alert)

        # 检查GC对象数量
        if gc_objects > self.gc_threshold_objects:
            alert = MemoryAlert(
                timestamp=datetime.now().isoformat(),
                alert_type="GC_TOO_MANY_OBJECTS",
                current_value=gc_objects,
                threshold_value=self.gc_threshold_objects,
                message=f"GC对象数量过多: {gc_objects} > {self.gc_threshold_objects}",
                severity="WARNING"
            )
            self._handle_alert(alert)

    def _handle_alert(self, alert: MemoryAlert) -> None:
        """处理内存告警"""
        self.alert_history.append(alert)
        self.optimization_stats['alerts_generated'] += 1

        self.logger.warning(f"内存告警: {alert.message}")

        # 根据告警类型自动优化
        if alert.alert_type == "MEMORY_HIGH_USAGE":
            self._optimize_memory_usage()
        elif alert.alert_type == "CACHE_LARGE_SIZE":
            self._optimize_cache_usage()
        elif alert.alert_type == "GC_TOO_MANY_OBJECTS":
            self._optimize_gc_objects()

    def _optimize_memory_usage(self) -> None:
        """优化内存使用"""
        self.logger.info("执行内存使用优化...")

        # 记录优化前内存
        before_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        # 执行优化策略
        self._enhanced_data_structure_cleanup()
        self._optimize_module_cache()
        self._force_garbage_collection()

        # 记录优化后内存
        after_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_reduced = before_memory - after_memory

        self.optimization_stats['total_optimizations'] += 1
        self.optimization_stats['memory_reduced_mb'] += memory_reduced

        self.logger.info(f"内存优化完成，减少: {memory_reduced:.2f}MB")

    def _optimize_cache_usage(self) -> None:
        """优化缓存使用"""
        self.logger.info("执行缓存优化...")

        try:
            self.cache_manager.clear_expired_cache()
            self.optimization_stats['cache_cleared_count'] += 1
            self.logger.info("缓存清理完成")
        except Exception as e:
            self.logger.error(f"缓存清理失败: {e}")

    def _optimize_gc_objects(self) -> None:
        """优化GC对象"""
        self.logger.info("执行GC对象优化...")

        # 强制垃圾回收
        collected = gc.collect()
        self.optimization_stats['gc_collections'] += 1

        self.logger.info(f"GC回收完成，回收对象: {collected}")

    def _enhanced_data_structure_cleanup(self) -> None:
        """
        增强版数据结构清理策略
        进一步优化数据结构清理，避免内存泄漏
        """
        self.logger.debug("执行增强版数据结构清理...")

        # 清理src.data模块的缓存
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('src.data'):
                try:
                    module = sys.modules[module_name]
                    if hasattr(module, '__dict__'):
                        # 安全清理非内置属性
                        for attr_name in list(module.__dict__.keys()):
                            if not attr_name.startswith('__'):
                                try:
                                    delattr(module, attr_name)
                                except (AttributeError, TypeError):
                                    pass
                except Exception as e:
                    self.logger.debug(f"清理模块 {module_name} 失败: {e}")

        # 清理全局变量
        self._cleanup_global_variables()

        # 清理弱引用
        self._cleanup_weak_references()

    def _cleanup_global_variables(self) -> None:
        """清理全局变量"""
        try:
            # 清理常见的全局变量
            global_vars_to_clean = [
                '__builtins__', '__cached__', '__doc__', '__file__',
                '__loader__', '__name__', '__package__', '__spec__'
            ]

            for var_name in global_vars_to_clean:
                if var_name in globals():
                    try:
                        del globals()[var_name]
                    except:
                        pass
        except Exception as e:
            self.logger.debug(f"清理全局变量失败: {e}")

    def _cleanup_weak_references(self) -> None:
        """清理弱引用"""
        try:
            pass

            # 清理弱引用字典
            for obj in gc.get_objects():
                if hasattr(obj, '__weakref__'):
                    try:
                        obj.__weakref__ = None
                    except:
                        pass
        except Exception as e:
            self.logger.debug(f"清理弱引用失败: {e}")

    def _optimize_module_cache(self) -> None:
        """优化模块缓存"""
        try:
            # 清理importlib缓存
            import importlib
            importlib.invalidate_caches()

            # 清理sys.modules中的临时模块
            temp_modules = [name for name in sys.modules.keys()
                            if name.startswith('_') or name.startswith('temp')]
            for module_name in temp_modules:
                try:
                    del sys.modules[module_name]
                except:
                    pass
        except Exception as e:
            self.logger.debug(f"优化模块缓存失败: {e}")

    def _force_garbage_collection(self) -> None:
        """强制垃圾回收"""
        try:
            # 设置GC参数
            gc.set_threshold(700, 10, 10)

            # 强制回收
            collected = gc.collect()

            # 重置GC参数
            gc.set_threshold(700, 10, 10)

            self.logger.debug(f"强制GC回收完成，回收对象: {collected}")
        except Exception as e:
            self.logger.debug(f"强制GC失败: {e}")

    def get_memory_status(self) -> Dict[str, Any]:
        """获取当前内存状态"""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()

        return {
            'process_memory_mb': memory_info.rss / (1024 * 1024),
            'memory_percent': process.memory_percent(),
            'available_memory_mb': system_memory.available / (1024 * 1024),
            'cache_size_mb': self._get_cache_size_mb(),
            'gc_objects': len(gc.get_objects()),
            'module_count': len([name for name in sys.modules.keys() if name.startswith('src.data')]),
            'optimization_stats': self.optimization_stats.copy(),
            'baseline_count': len(self.baseline_data),
            'alert_count': len(self.alert_history),
            'is_monitoring': self.is_monitoring
        }

    def generate_memory_report(self) -> Dict[str, Any]:
        """生成内存优化报告"""
        current_status = self.get_memory_status()

        # 计算基线统计
        baseline_stats = {}
        if self.baseline_data:
            memory_values = [b.memory_usage_mb for b in self.baseline_data]
            baseline_stats = {
                'baseline_count': len(self.baseline_data),
                'avg_memory_usage_mb': sum(memory_values) / len(memory_values),
                'min_memory_usage_mb': min(memory_values),
                'max_memory_usage_mb': max(memory_values),
                'latest_baseline': asdict(self.baseline_data[-1]) if self.baseline_data else None
            }

        # 计算告警统计
        alert_stats = {}
        if self.alert_history:
            alert_types = [alert.alert_type for alert in self.alert_history]
            alert_stats = {
                'total_alerts': len(self.alert_history),
                'alert_types': list(set(alert_types)),
                'alert_type_counts': {alert_type: alert_types.count(alert_type) for alert_type in set(alert_types)},
                'latest_alerts': [asdict(alert) for alert in self.alert_history[-5:]]  # 最近5个告警
            }

        report = {
            'timestamp': datetime.now().isoformat(),
            'current_status': current_status,
            'baseline_stats': baseline_stats,
            'alert_stats': alert_stats,
            'optimization_stats': self.optimization_stats.copy(),
            'summary': {
                'memory_optimization_active': self.is_monitoring,
                'total_optimizations': self.optimization_stats['total_optimizations'],
                'total_memory_reduced_mb': self.optimization_stats['memory_reduced_mb'],
                'total_alerts': len(self.alert_history)
            }
        }

        return report


def main():
    """主函数"""
    print("=" * 60)
    print("数据层内存优化增强脚本")
    print("=" * 60)

    # 创建内存优化器
    optimizer = EnhancedMemoryOptimizer(
        memory_threshold_percent=75.0,
        cache_threshold_mb=300.0,
        gc_threshold_objects=8000,
        monitoring_interval=20
    )

    try:
        # 1. 建立内存基线
        print("\n1. 建立内存使用基线标准...")
        baseline = optimizer.establish_memory_baseline()
        print(f"   基线建立完成: 内存使用 {baseline.memory_usage_mb:.2f}MB")

        # 2. 启动自动监控
        print("\n2. 启动自动内存监控告警...")
        optimizer.start_auto_monitoring()
        print("   自动监控已启动")

        # 3. 模拟内存压力测试
        print("\n3. 执行内存压力测试...")
        for i in range(5):
            print(f"   第{i+1}轮测试...")

            # 模拟内存使用
            large_list = [f"test_data_{j}" * 1000 for j in range(1000)]

            # 获取当前状态
            status = optimizer.get_memory_status()
            print(
                f"   当前内存: {status['process_memory_mb']:.2f}MB, 使用率: {status['memory_percent']:.2f}%")

            # 清理测试数据
            del large_list
            time.sleep(2)

        # 4. 停止监控
        print("\n4. 停止自动监控...")
        optimizer.stop_auto_monitoring()
        print("   监控已停止")

        # 5. 生成报告
        print("\n5. 生成内存优化报告...")
        report = optimizer.generate_memory_report()

        # 保存报告
        report_file = "reports/enhanced_memory_optimization_report.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"   报告已保存: {report_file}")

        # 6. 显示结果摘要
        print("\n" + "=" * 60)
        print("内存优化增强完成")
        print("=" * 60)
        print(f"总优化次数: {report['summary']['total_optimizations']}")
        print(f"总内存减少: {report['summary']['total_memory_reduced_mb']:.2f}MB")
        print(f"总告警次数: {report['summary']['total_alerts']}")
        print(f"基线数据点: {report['baseline_stats']['baseline_count']}")

        current_status = report['current_status']
        print(f"\n当前状态:")
        print(f"  内存使用: {current_status['process_memory_mb']:.2f}MB")
        print(f"  内存使用率: {current_status['memory_percent']:.2f}%")
        print(f"  缓存大小: {current_status['cache_size_mb']:.2f}MB")
        print(f"  GC对象数: {current_status['gc_objects']}")

    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n脚本执行完成!")


if __name__ == "__main__":
    main()
