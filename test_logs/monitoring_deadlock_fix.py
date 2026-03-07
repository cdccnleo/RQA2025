#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控系统死锁修复方案

修复 SimpleMonitoringSystem 中的死锁问题
"""

import threading
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
import time

# ============================================================================
# 方案1：使用可重入锁（快速修复）
# ============================================================================

class SimpleMonitoringSystem_Fixed_v1:
    """简化的监控系统 - 方案1：使用RLock修复死锁"""

    def __init__(self, name: str = "monitoring_system"):
        self.name = name
        self._alerts: Dict[str, Any] = {}
        self._metrics: Dict[str, Any] = {}
        self._alert_handlers: List[Callable] = []
        
        # ✅ 修复：使用可重入锁代替普通锁
        self._lock = threading.RLock()  # ← 从 Lock() 改为 RLock()

    def get_active_alerts(self) -> Dict[str, Any]:
        """获取活跃告警"""
        with self._lock:  # ✅ 现在可以安全重入
            return {aid: alert for aid, alert in self._alerts.items()
                   if alert.get('status') == 'ACTIVE'}

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        with self._lock:  # ✅ 第一次获取锁
            active_alerts = len(self.get_active_alerts())  # ✅ 可以安全调用（重入）
            total_alerts = len(self._alerts)
            total_metrics = len(self._metrics)

            # 计算告警严重程度
            alert_levels = {}
            for alert in self._alerts.values():
                level = alert.get('level', 'unknown')
                alert_levels[level] = alert_levels.get(level, 0) + 1

            return {
                "system_name": self.name,
                "timestamp": time.time(),
                "alerts": {
                    "total": total_alerts,
                    "active": active_alerts,
                    "by_level": alert_levels
                },
                "metrics": {
                    "total": total_metrics
                },
                "health_score": max(0, 100 - (active_alerts * 10))
            }


# ============================================================================
# 方案2：重构避免嵌套锁（更优方案）
# ============================================================================

class SimpleMonitoringSystem_Fixed_v2:
    """简化的监控系统 - 方案2：重构避免嵌套锁调用"""

    def __init__(self, name: str = "monitoring_system"):
        self.name = name
        self._alerts: Dict[str, Any] = {}
        self._metrics: Dict[str, Any] = {}
        self._alert_handlers: List[Callable] = []
        self._lock = threading.Lock()  # 保持使用普通锁

    def get_active_alerts(self) -> Dict[str, Any]:
        """获取活跃告警"""
        with self._lock:
            return {aid: alert for aid, alert in self._alerts.items()
                   if alert.get('status') == 'ACTIVE'}

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态 - 重构版本"""
        with self._lock:
            # ✅ 直接计算，不调用其他需要锁的方法
            active_alerts_count = sum(
                1 for alert in self._alerts.values()
                if alert.get('status') == 'ACTIVE'
            )
            total_alerts = len(self._alerts)
            total_metrics = len(self._metrics)

            # 计算告警严重程度
            alert_levels = {}
            for alert in self._alerts.values():
                level = alert.get('level', 'unknown')
                alert_levels[level] = alert_levels.get(level, 0) + 1

            return {
                "system_name": self.name,
                "timestamp": time.time(),
                "alerts": {
                    "total": total_alerts,
                    "active": active_alerts_count,  # ✅ 使用直接计算的值
                    "by_level": alert_levels
                },
                "metrics": {
                    "total": total_metrics
                },
                "health_score": max(0, 100 - (active_alerts_count * 10))
            }


# ============================================================================
# 方案3：细粒度锁（长期优化方案）
# ============================================================================

class SimpleMonitoringSystem_Fixed_v3:
    """简化的监控系统 - 方案3：细粒度锁优化"""

    def __init__(self, name: str = "monitoring_system"):
        self.name = name
        self._alerts: Dict[str, Any] = {}
        self._metrics: Dict[str, Any] = {}
        self._alert_handlers: List[Callable] = []
        
        # ✅ 分离的锁，减少锁竞争
        self._alerts_lock = threading.RLock()
        self._metrics_lock = threading.RLock()
        self._handlers_lock = threading.RLock()

    def get_active_alerts(self) -> Dict[str, Any]:
        """获取活跃告警"""
        with self._alerts_lock:  # ✅ 只锁定告警数据
            return {aid: alert for aid, alert in self._alerts.items()
                   if alert.get('status') == 'ACTIVE'}

    def get_metric(self, name: str) -> Optional[Any]:
        """获取指标"""
        with self._metrics_lock:  # ✅ 只锁定指标数据
            return self._metrics.get(name)

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态 - 细粒度锁版本"""
        # ✅ 分别获取告警和指标数据
        with self._alerts_lock:
            active_alerts_count = sum(
                1 for alert in self._alerts.values()
                if alert.get('status') == 'ACTIVE'
            )
            total_alerts = len(self._alerts)
            
            # 计算告警严重程度
            alert_levels = {}
            for alert in self._alerts.values():
                level = alert.get('level', 'unknown')
                alert_levels[level] = alert_levels.get(level, 0) + 1

        with self._metrics_lock:
            total_metrics = len(self._metrics)

        return {
            "system_name": self.name,
            "timestamp": time.time(),
            "alerts": {
                "total": total_alerts,
                "active": active_alerts_count,
                "by_level": alert_levels
            },
            "metrics": {
                "total": total_metrics
            },
            "health_score": max(0, 100 - (active_alerts_count * 10))
        }


# ============================================================================
# 死锁验证测试
# ============================================================================

def test_deadlock_verification():
    """验证修复后不再出现死锁"""
    import concurrent.futures
    
    print("\n" + "="*80)
    print("死锁验证测试")
    print("="*80)
    
    test_cases = [
        ("方案1: RLock", SimpleMonitoringSystem_Fixed_v1),
        ("方案2: 重构", SimpleMonitoringSystem_Fixed_v2),
        ("方案3: 细粒度锁", SimpleMonitoringSystem_Fixed_v3),
    ]
    
    for case_name, MonitoringClass in test_cases:
        print(f"\n测试 {case_name}...")
        
        monitoring = MonitoringClass("deadlock_test")
        
        # 创建测试数据
        for i in range(10):
            monitoring._alerts[f"alert_{i}"] = {
                'status': 'ACTIVE',
                'level': 'INFO',
                'message': f'Alert {i}'
            }
            monitoring._metrics[f"metric_{i}"] = {
                'value': i,
                'type': 'COUNTER'
            }
        
        results = []
        errors = []
        
        def query_status():
            """并发查询状态"""
            try:
                start = time.time()
                status = monitoring.get_system_status()
                duration = time.time() - start
                return (status, duration)
            except Exception as e:
                errors.append(str(e))
                raise
        
        # 并发执行状态查询
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(query_status) for _ in range(50)]
            
            try:
                for future in concurrent.futures.as_completed(futures, timeout=5.0):
                    result = future.result()
                    results.append(result)
            except concurrent.futures.TimeoutError:
                print(f"  ❌ {case_name} - 检测到死锁：状态查询超时")
                continue
            except Exception as e:
                print(f"  ❌ {case_name} - 执行错误: {e}")
                continue
        
        total_time = time.time() - start_time
        
        # 验证结果
        if len(results) == 50:
            avg_duration = sum(r[1] for r in results) / len(results)
            max_duration = max(r[1] for r in results)
            
            print(f"  ✅ {case_name} - 测试通过")
            print(f"     - 查询数量: {len(results)}")
            print(f"     - 总耗时: {total_time:.3f}s")
            print(f"     - 平均查询时间: {avg_duration*1000:.2f}ms")
            print(f"     - 最大查询时间: {max_duration*1000:.2f}ms")
        else:
            print(f"  ❌ {case_name} - 部分查询失败")
            print(f"     - 成功查询: {len(results)}/50")
            if errors:
                print(f"     - 错误: {errors[0]}")


# ============================================================================
# 性能对比测试
# ============================================================================

def test_performance_comparison():
    """性能对比测试"""
    import concurrent.futures
    
    print("\n" + "="*80)
    print("性能对比测试")
    print("="*80)
    
    test_cases = [
        ("方案1: RLock", SimpleMonitoringSystem_Fixed_v1),
        ("方案2: 重构", SimpleMonitoringSystem_Fixed_v2),
        ("方案3: 细粒度锁", SimpleMonitoringSystem_Fixed_v3),
    ]
    
    for case_name, MonitoringClass in test_cases:
        print(f"\n测试 {case_name}...")
        
        monitoring = MonitoringClass("perf_test")
        
        # 创建大量测试数据
        for i in range(100):
            monitoring._alerts[f"alert_{i}"] = {
                'status': 'ACTIVE' if i % 2 == 0 else 'RESOLVED',
                'level': 'INFO',
                'message': f'Alert {i}'
            }
            monitoring._metrics[f"metric_{i}"] = {
                'value': i,
                'type': 'COUNTER'
            }
        
        # 测试高并发场景
        iterations = 1000
        
        def benchmark():
            monitoring.get_system_status()
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(benchmark) for _ in range(iterations)]
            concurrent.futures.wait(futures, timeout=30.0)
        
        total_time = time.time() - start_time
        ops_per_sec = iterations / total_time
        
        print(f"  📊 性能指标:")
        print(f"     - 总查询数: {iterations}")
        print(f"     - 总耗时: {total_time:.3f}s")
        print(f"     - 吞吐量: {ops_per_sec:.0f} ops/s")
        print(f"     - 平均延迟: {(total_time/iterations)*1000:.2f}ms")


if __name__ == '__main__':
    print("\n🔧 监控系统死锁修复验证")
    print("="*80)
    
    # 运行验证测试
    test_deadlock_verification()
    
    # 运行性能对比测试
    test_performance_comparison()
    
    print("\n" + "="*80)
    print("✅ 所有测试完成")
    print("="*80)

