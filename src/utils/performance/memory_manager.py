#!/usr/bin/env python3
"""
动态内存管理器
实现智能内存监控、GC优化和内存泄漏检测
"""

import gc
import sys
import psutil
import threading
import weakref
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """内存统计信息"""
    timestamp: datetime
    used_mb: float
    available_mb: float
    percent: float
    gc_collections: List[int]
    large_objects: int


class DynamicMemoryManager:
    """
    动态内存管理器
    
    特性：
    1. 动态内存阈值监控
    2. 智能GC触发策略
    3. 大对象追踪
    4. 内存泄漏检测
    """
    
    def __init__(
        self,
        memory_threshold: float = 0.8,  # 80%内存使用率触发
        gc_thresholds: tuple = (700, 10, 10),  # 分代GC阈值
        monitoring_interval: int = 5,  # 监控间隔（秒）
        enable_auto_gc: bool = True
    ):
        self.memory_threshold = memory_threshold
        self.gc_thresholds = gc_thresholds
        self.monitoring_interval = monitoring_interval
        self.enable_auto_gc = enable_auto_gc
        
        # 设置GC阈值
        gc.set_threshold(*self.gc_thresholds)
        
        # 监控状态
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
        
        # 统计信息
        self._stats_history: List[MemoryStats] = []
        self._max_history_size = 1000
        
        # 大对象追踪
        self._large_objects: Dict[int, Any] = {}
        self._large_object_threshold = 10 * 1024 * 1024  # 10MB
        
        # 内存泄漏检测
        self._object_snapshots: Dict[str, List[int]] = {}
        
        logger.info("DynamicMemoryManager initialized")
    
    def start_monitoring(self):
        """启动内存监控"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """停止内存监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """监控循环"""
        while self._monitoring:
            try:
                self._check_memory()
                threading.Event().wait(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
    
    def _check_memory(self):
        """检查内存状态"""
        memory = psutil.virtual_memory()
        
        # 记录统计信息
        stats = MemoryStats(
            timestamp=datetime.now(),
            used_mb=memory.used / (1024 * 1024),
            available_mb=memory.available / (1024 * 1024),
            percent=memory.percent,
            gc_collections=list(gc.get_count()),
            large_objects=len(self._large_objects)
        )
        
        self._stats_history.append(stats)
        if len(self._stats_history) > self._max_history_size:
            self._stats_history.pop(0)
        
        # 检查内存阈值
        if memory.percent > self.memory_threshold * 100:
            logger.warning(f"Memory usage {memory.percent}% exceeds threshold {self.memory_threshold * 100}%")
            self._handle_high_memory()
        
        # 触发回调
        for callback in self._callbacks:
            try:
                callback(stats)
            except Exception as e:
                logger.error(f"Memory callback error: {e}")
    
    def _handle_high_memory(self):
        """处理高内存使用"""
        logger.info("Triggering emergency GC due to high memory usage")
        
        # 强制垃圾回收
        if self.enable_auto_gc:
            self.emergency_gc()
        
        # 清理大对象
        self._cleanup_large_objects()
    
    def emergency_gc(self):
        """紧急垃圾回收"""
        # 收集所有代
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
        
        # 记录GC后内存状态
        memory = psutil.virtual_memory()
        logger.info(f"Emergency GC completed. Memory usage: {memory.percent}%")
    
    def register_callback(self, callback: Callable):
        """注册内存状态回调"""
        self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable):
        """注销内存状态回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def track_large_object(self, obj: Any, name: str = ""):
        """追踪大对象"""
        size = sys.getsizeof(obj)
        if size > self._large_object_threshold:
            obj_id = id(obj)
            self._large_objects[obj_id] = {
                'object': weakref.ref(obj),
                'size': size,
                'name': name,
                'created_at': datetime.now()
            }
            logger.debug(f"Tracking large object: {name} ({size / (1024 * 1024):.2f} MB)")
    
    def _cleanup_large_objects(self):
        """清理大对象"""
        current_time = datetime.now()
        expired_objects = []
        
        for obj_id, info in list(self._large_objects.items()):
            # 检查对象是否还存在
            if info['object']() is None:
                expired_objects.append(obj_id)
            # 检查是否过期（超过1小时）
            elif current_time - info['created_at'] > timedelta(hours=1):
                expired_objects.append(obj_id)
        
        for obj_id in expired_objects:
            del self._large_objects[obj_id]
        
        if expired_objects:
            logger.info(f"Cleaned up {len(expired_objects)} large objects")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        memory = psutil.virtual_memory()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_mb': memory.total / (1024 * 1024),
            'available_mb': memory.available / (1024 * 1024),
            'used_mb': memory.used / (1024 * 1024),
            'percent': memory.percent,
            'gc_collections': gc.get_count(),
            'gc_thresholds': gc.get_threshold(),
            'large_objects_count': len(self._large_objects),
            'large_objects_total_mb': sum(
                info['size'] for info in self._large_objects.values()
            ) / (1024 * 1024)
        }
    
    def get_stats_history(self, limit: int = 100) -> List[MemoryStats]:
        """获取历史统计信息"""
        return self._stats_history[-limit:]
    
    def detect_memory_leak(self, class_name: str, duration_minutes: int = 10) -> bool:
        """
        检测内存泄漏
        
        Args:
            class_name: 要检测的类名
            duration_minutes: 检测持续时间
            
        Returns:
            是否检测到泄漏
        """
        # 获取当前对象数量
        current_count = len([
            obj for obj in gc.get_objects()
            if obj.__class__.__name__ == class_name
        ])
        
        # 记录快照
        if class_name not in self._object_snapshots:
            self._object_snapshots[class_name] = []
        
        self._object_snapshots[class_name].append(current_count)
        
        # 保留最近10个快照
        if len(self._object_snapshots[class_name]) > 10:
            self._object_snapshots[class_name].pop(0)
        
        # 检测持续增长趋势
        snapshots = self._object_snapshots[class_name]
        if len(snapshots) >= 5:
            # 简单线性回归检测增长趋势
            n = len(snapshots)
            x_mean = sum(range(n)) / n
            y_mean = sum(snapshots) / n
            
            numerator = sum((i - x_mean) * (snapshots[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            
            if denominator > 0:
                slope = numerator / denominator
                # 如果斜率为正且显著，可能存在泄漏
                if slope > 0.1:  # 每个周期增长超过0.1个对象
                    logger.warning(f"Potential memory leak detected for {class_name}: slope={slope:.2f}")
                    return True
        
        return False


class MemoryPool:
    """
    内存池
    预分配和复用对象，减少GC压力
    """
    
    def __init__(self, max_size: int = 100, auto_expand: bool = True):
        self.max_size = max_size
        self.auto_expand = auto_expand
        self._pool: List[Any] = []
        self._in_use: set = set()
        self._lock = threading.Lock()
    
    def acquire(self) -> Optional[Any]:
        """从池中获取对象"""
        with self._lock:
            if self._pool:
                obj = self._pool.pop()
                self._in_use.add(id(obj))
                return obj
            return None
    
    def release(self, obj: Any):
        """释放对象回池"""
        with self._lock:
            obj_id = id(obj)
            if obj_id in self._in_use:
                self._in_use.remove(obj_id)
                if len(self._pool) < self.max_size:
                    self._pool.append(obj)
    
    def resize(self, new_size: int):
        """调整池大小"""
        with self._lock:
            self.max_size = new_size
            # 如果新大小更小，移除多余对象
            while len(self._pool) > self.max_size:
                self._pool.pop()


# 全局内存管理器实例
_memory_manager: Optional[DynamicMemoryManager] = None


def get_memory_manager() -> DynamicMemoryManager:
    """获取全局内存管理器实例"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = DynamicMemoryManager()
    return _memory_manager


def setup_memory_monitoring():
    """设置内存监控"""
    manager = get_memory_manager()
    manager.start_monitoring()
    
    # 注册内存告警回调
    def memory_alert(stats: MemoryStats):
        if stats.percent > 90:
            logger.critical(f"CRITICAL: Memory usage at {stats.percent}%!")
    
    manager.register_callback(memory_alert)


# 示例用法
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建内存管理器
    manager = DynamicMemoryManager(
        memory_threshold=0.7,  # 70%阈值
        monitoring_interval=2   # 2秒监控间隔
    )
    
    # 启动监控
    manager.start_monitoring()
    
    print("=== 内存管理器测试 ===\n")
    
    # 测试1: 获取内存统计
    stats = manager.get_memory_stats()
    print(f"当前内存使用: {stats['percent']}%")
    print(f"已用内存: {stats['used_mb']:.2f} MB")
    print(f"可用内存: {stats['available_mb']:.2f} MB")
    print()
    
    # 测试2: 追踪大对象
    large_list = [0] * (10 * 1024 * 1024 // 8)  # 10MB列表
    manager.track_large_object(large_list, "test_large_list")
    print(f"大对象数量: {stats['large_objects_count']}")
    print()
    
    # 测试3: 内存泄漏检测
    class TestObject:
        pass
    
    # 创建一些对象
    objects = [TestObject() for _ in range(100)]
    
    for i in range(5):
        # 模拟泄漏：不断创建新对象
        objects.extend([TestObject() for _ in range(10)])
        is_leak = manager.detect_memory_leak("TestObject")
        print(f"检测 {i+1}: 泄漏检测={is_leak}, 对象数={len(objects)}")
    
    print("\n=== 测试完成 ===")
    
    # 停止监控
    manager.stop_monitoring()
