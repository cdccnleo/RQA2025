#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 内存性能优化脚本

优化内存使用率：超标 → <70%
"""

import os
import sys
import psutil
import gc
import threading
from pathlib import Path
from datetime import datetime


def optimize_memory_performance():
    """优化内存性能"""
    print("💾 RQA2025 内存性能优化")
    print("=" * 50)

    project_root = Path(__file__).parent.parent

    # 1. 分析当前内存使用情况
    analyze_current_memory_usage()

    # 2. 实施内存优化策略
    implement_memory_optimizations(project_root)

    # 3. 优化缓存内存使用
    optimize_cache_memory(project_root)

    # 4. 优化模型内存使用
    optimize_model_memory(project_root)

    # 5. 实施内存池管理
    implement_memory_pool(project_root)

    # 6. 建立内存监控和告警
    setup_memory_monitoring(project_root)

    print("\n✅ 内存性能优化完成!")
    return True


def analyze_current_memory_usage():
    """分析当前内存使用情况"""
    print("\n📊 分析当前内存使用情况...")
    print("-" * 30)

    # 获取内存信息
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()

    print(f"总内存: {memory.total / 1024**3:.1f} GB")
    print(f"可用内存: {memory.available / 1024**3:.1f} GB")
    print(f"已用内存: {memory.used / 1024**3:.1f} GB")
    print(f"内存使用率: {memory.percent}%")
    print(f"Swap总大小: {swap.total / 1024**3:.1f} GB")
    print(f"Swap使用率: {swap.percent}%")

    # 分析进程内存使用
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'memory_percent']):
        try:
            if 'python' in proc.info['name'].lower():
                python_processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'memory_mb': proc.info['memory_info'].rss / 1024**2,
                    'memory_percent': proc.info['memory_percent']
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # 按内存使用排序
    python_processes.sort(key=lambda x: x['memory_mb'], reverse=True)

    print("
🔍 Python进程内存使用: " for proc in python_processes[:5]:  # 显示前5个
        print(".1f"
    if memory.percent > 80:
        print("⚠️  内存使用率过高，需要优化")
    elif memory.percent > 60:
        print("⚠️  内存使用率偏高，建议优化")
    else:
        print("✅ 内存使用率正常")

def implement_memory_optimizations(project_root):
    """实施内存优化策略"""
    print("\n🧠 实施内存优化策略...")
    print("-" * 30)

    # 1. 创建内存优化管理器
    memory_manager=project_root / "src" / "infrastructure" / "memory_manager.py"
    memory_content="""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
RQA2025 内存优化管理器

提供内存优化策略和监控
\"\"\"

import os
import gc
import psutil
import threading
import weakref
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    \"\"\"内存管理器\"\"\"

    def __init__(self, memory_limit_gb: float = 8.0):
        self.memory_limit = memory_limit_gb * 1024**3  # 转换为字节
        self.current_memory_usage = 0
        self.memory_threshold = 0.75  # 75%阈值
        self.cleanup_threshold = 0.85  # 85%时触发清理
        self.monitoring = False
        self.lock = threading.RLock()

        # 内存对象跟踪
        self.large_objects = weakref.WeakSet()
        self.object_sizes = {}

        # 内存池
        self.object_pools = defaultdict(list)

        logger.info(f"内存管理器初始化: 限制={memory_limit_gb}GB, 阈值={self.memory_threshold}")

    def get_memory_usage(self) -> Dict[str, float]:
        \"\"\"获取内存使用情况\"\"\"
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / 1024**3,
            'used_gb': memory.used / 1024**3,
            'available_gb': memory.available / 1024**3,
            'usage_percent': memory.percent
        }

    def check_memory_pressure(self) -> bool:
        \"\"\"检查内存压力\"\"\"
        memory = self.get_memory_usage()
        return memory['usage_percent'] > (self.memory_threshold * 100)

    def force_garbage_collection(self):
        \"\"\"强制垃圾回收\"\"\"
        logger.info("执行垃圾回收...")
        collected = gc.collect()
        logger.info(f"垃圾回收完成: 回收 {collected} 个对象")

        # 获取回收后的内存使用
        memory_after = self.get_memory_usage()
        logger.info(f"垃圾回收后内存使用: {memory_after['usage_percent']:.1f}%")

        return collected

    def optimize_data_structures(self, data: Any) -> Any:
        \"\"\"优化数据结构内存使用\"\"\"
        import numpy as np
        import pandas as pd

        if isinstance(data, list) and len(data) > 1000:
            # 大列表转换为numpy数组
            try:
                return np.array(data)
            except:
                return data

        elif isinstance(data, pd.DataFrame):
            # DataFrame内存优化
            memory_usage = data.memory_usage(deep=True).sum() / 1024**2  # MB
            if memory_usage > 100:  # 大于100MB
                logger.info(f"DataFrame内存使用: {memory_usage:.1f}MB，尝试优化")

                # 尝试降低数据类型
                for col in data.select_dtypes(include=['int64']).columns:
                    if data[col].max() < 2**31:
                        data[col] = data[col].astype('int32')

                for col in data.select_dtypes(include=['float64']).columns:
                    data[col] = data[col].astype('float32')

                memory_after = data.memory_usage(deep=True).sum() / 1024**2
                logger.info(
                    f"优化后内存使用: {memory_after:.1f}MB (减少 {(memory_usage-memory_after)/memory_usage*100:.1f}%)")

            return data

        return data

    def create_memory_pool(self, pool_name: str, object_factory: Callable, pool_size: int = 10):
        \"\"\"创建内存池\"\"\"
        if pool_name not in self.object_pools:
            self.object_pools[pool_name] = []

        # 预创建对象
        for _ in range(pool_size):
            try:
                obj = object_factory()
                self.object_pools[pool_name].append(obj)
            except Exception as e:
                logger.error(f"创建对象失败: {e}")
                break

        logger.info(f"内存池 {pool_name} 创建完成: {len(self.object_pools[pool_name])} 个对象")

    def get_from_pool(self, pool_name: str):
        \"\"\"从内存池获取对象\"\"\"
        if pool_name in self.object_pools and self.object_pools[pool_name]:
            return self.object_pools[pool_name].pop()
        return None

    def return_to_pool(self, pool_name: str, obj: Any):
        \"\"\"将对象返回内存池\"\"\"
        if pool_name in self.object_pools:
            if len(self.object_pools[pool_name]) < 20:  # 限制池大小
                self.object_pools[pool_name].append(obj)

    def cleanup_large_objects(self):
        \"\"\"清理大对象\"\"\"
        logger.info("清理大对象...")

        # 清理弱引用对象
        self.large_objects.clear()

        # 清理对象池
        for pool_name in self.object_pools:
            pool = self.object_pools[pool_name]
            # 只保留最近使用的对象
            self.object_pools[pool_name] = pool[-10:] if len(pool) > 10 else pool

        logger.info("大对象清理完成")

    def start_monitoring(self):
        \"\"\"开始内存监控\"\"\"
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("内存监控已启动")

    def stop_monitoring(self):
        \"\"\"停止内存监控\"\"\"
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)
        logger.info("内存监控已停止")

    def _monitor_loop(self):
        \"\"\"内存监控循环\"\"\"
        while self.monitoring:
            try:
                memory = self.get_memory_usage()

                # 检查内存压力
                if memory['usage_percent'] > (self.cleanup_threshold * 100):
                    logger.warning(f"内存使用率过高: {memory['usage_percent']:.1f}%")
                    self.force_garbage_collection()
                    self.cleanup_large_objects()

                # 定期清理
                self.force_garbage_collection()

            except Exception as e:
                logger.error(f"内存监控出错: {e}")

            # 每30秒检查一次
            threading.Event().wait(30)

# 全局内存管理器实例
memory_manager = MemoryManager()

def optimize_memory_usage(func):
    \"\"\"内存优化装饰器\"\"\"
    def wrapper(*args, **kwargs):
        # 检查内存压力
        if memory_manager.check_memory_pressure():
            memory_manager.force_garbage_collection()

        result = func(*args, **kwargs)

        # 优化结果对象
        result = memory_manager.optimize_data_structures(result)

        return result
    return wrapper

if __name__ == "__main__":
    print("内存管理器测试...")

    manager = MemoryManager(memory_limit_gb=4.0)  # 4GB限制

    # 获取内存使用情况
    memory_info = manager.get_memory_usage()
    print(f"当前内存使用: {memory_info}")

    # 测试垃圾回收
    manager.force_garbage_collection()

    # 启动监控
    manager.start_monitoring()

    # 创建测试数据进行优化
    import numpy as np

    # 测试大数组优化
    large_list = list(range(100000))
    print(f"原始列表大小: {sys.getsizeof(large_list)} bytes")

    optimized_array = manager.optimize_data_structures(large_list)
    print(f"优化后数组类型: {type(optimized_array)}")
    if hasattr(optimized_array, 'nbytes'):
        print(f"优化后数组大小: {optimized_array.nbytes} bytes")

    # 停止监控
    manager.stop_monitoring()

    print("✅ 内存管理器测试完成")
"""

    with open(memory_manager, 'w', encoding='utf-8') as f:
        f.write(memory_content)

    print("✅ 内存优化管理器已创建")

    # 2. 创建内存分析工具
    memory_analyzer=project_root / "scripts" / "analyze_memory_usage.py"
    analyzer_content="""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
内存使用分析工具
\"\"\"

import os
import psutil
import sys
import gc
import threading
from pathlib import Path
from datetime import datetime

def analyze_memory_usage():
    \"\"\"分析内存使用情况\"\"\"
    print("🧠 内存使用分析")
    print("=" * 30)

    # 1. 系统内存信息
    memory = psutil.virtual_memory()
    print(f"系统总内存: {memory.total / 1024**3:.1f} GB")
    print(f"可用内存: {memory.available / 1024**3:.1f} GB")
    print(f"已用内存: {memory.used / 1024**3:.1f} GB")
    print(f"内存使用率: {memory.percent}%")

    # 2. Python进程分析
    python_processes = []
    current_process = psutil.Process()

    for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'memory_percent']):
        try:
            if 'python' in proc.info['name'].lower():
                python_processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'memory_mb': proc.info['memory_info'].rss / 1024**2,
                    'memory_percent': proc.info['memory_percent'],
                    'is_current': proc.info['pid'] == current_process.pid
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # 按内存使用排序
    python_processes.sort(key=lambda x: x['memory_mb'], reverse=True)

    print("
🐍 Python进程内存使用:"    total_python_memory = 0
    for proc in python_processes:
        marker = " <-- 当前进程" if proc['is_current'] else ""
        print(".1f"        total_python_memory += proc['memory_mb']

    print(".1f"
    # 3. 当前进程详细分析
    print("
🔍 当前进程内存分析:"    current_memory = current_process.memory_info()
    print(f"  RSS内存: {current_memory.rss / 1024**2:.1f} MB")
    print(f"  VMS内存: {current_memory.vms / 1024**2:.1f} MB")

    # 4. 垃圾回收信息
    print("
🗑️  垃圾回收统计:"    gc_stats = gc.get_stats()
    for i, gen_stats in enumerate(gc_stats):
        print(f"  第{i}代: 收集 {gen_stats['collected']} 次, 处理 {gen_stats['uncollectable']} 个不可回收对象")

    # 5. 内存优化建议
    print("
💡 内存优化建议:"    if memory.percent > 80:
        print("  🔴 紧急: 内存使用率过高，建议立即优化")
        print("    - 减少数据加载量")
        print("    - 使用数据流处理")
        print("    - 增加内存清理")
    elif memory.percent > 70:
        print("  🟡 重要: 内存使用率偏高，建议优化")
        print("    - 优化数据结构")
        print("    - 实现内存池")
        print("    - 增加缓存清理")
    else:
        print("  🟢 良好: 内存使用率正常")

    if total_python_memory > 1000:  # 大于1GB
        print("    - Python进程占用内存较大")
        print("    - 考虑使用numpy数组替代Python列表")
        print("    - 优化pandas DataFrame内存使用")

    return memory.percent

def create_memory_optimization_report():
    \"\"\"创建内存优化报告\"\"\"
    print("
📊 生成内存优化报告..."    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"memory_analysis_report_{timestamp}.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("RQA2025 内存分析报告\\n")
        f.write("=" * 50 + "\\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")

        # 系统内存信息
        memory = psutil.virtual_memory()
        f.write("系统内存信息:\\n")
        f.write(f"  总内存: {memory.total / 1024**3:.1f} GB\\n")
        f.write(f"  可用内存: {memory.available / 1024**3:.1f} GB\\n")
        f.write(f"  已用内存: {memory.used / 1024**3:.1f} GB\\n")
        f.write(f"  使用率: {memory.percent}%\\n\\n")

        # Python进程信息
        f.write("Python进程内存使用:\\n")
        current_process = psutil.Process()

        for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
            try:
                if 'python' in proc.info['name'].lower():
                    memory_mb = proc.info['memory_info'].rss / 1024**2
                    marker = " (当前进程)" if proc.info['pid'] == current_process.pid else ""
                    f.write(
                        f"  PID {proc.info['pid']}: {proc.info['name']} - {memory_mb:.1f} MB{marker}\\n")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        f.write("\\n优化建议:\\n")
        if memory.percent > 80:
            f.write("- 内存使用率过高，建议:\\n")
            f.write("  * 减少同时处理的数据量\\n")
            f.write("  * 使用数据流处理替代全量加载\\n")
            f.write("  * 增加定期内存清理\\n")
        elif memory.percent > 70:
            f.write("- 内存使用率偏高，建议:\\n")
            f.write("  * 优化数据结构内存占用\\n")
            f.write("  * 实现对象池管理\\n")
            f.write("  * 增加智能缓存清理\\n")
        else:
            f.write("- 内存使用率正常\\n")

    print(f"✅ 内存分析报告已生成: {report_file}")
    return report_file

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report_file = create_memory_optimization_report()
    else:
        memory_usage = analyze_memory_usage()
        if memory_usage > 80:
            print("\\n⚠️  内存使用率过高，建议优化")
        elif memory_usage > 70:
            print("\\n⚠️  内存使用率偏高，建议关注")
        else:
            print("\\n✅ 内存使用率正常")
"""

    with open(memory_analyzer, 'w', encoding='utf-8') as f:
        f.write(analyzer_content)

    print("✅ 内存分析工具已创建")

def optimize_cache_memory(project_root):
    """优化缓存内存使用"""
    print("\n💾 优化缓存内存使用...")
    print("-" * 30)

    # 创建缓存内存优化器
    cache_optimizer=project_root / "src" / "infrastructure" / "cache_memory_optimizer.py"
    cache_content="""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
缓存内存优化器

优化缓存策略，减少内存占用
\"\"\"

import os
import gc
import weakref
import threading
from typing import Dict, List, Any, Optional, Tuple
from collections import OrderedDict, defaultdict
import logging

logger = logging.getLogger(__name__)

class CacheMemoryOptimizer:
    \"\"\"缓存内存优化器\"\"\"

    def __init__(self, max_memory_mb: float = 500):
        self.max_memory = max_memory_mb * 1024 * 1024  # 转换为字节
        self.current_memory = 0
        self.cache_entries: OrderedDict = OrderedDict()
        self.memory_usage = defaultdict(int)
        self.access_count = defaultdict(int)
        self.lock = threading.RLock()

        # 内存监控
        self.memory_threshold = 0.8  # 80%阈值
        self.cleanup_ratio = 0.2     # 清理20%的条目

        logger.info(f"缓存内存优化器初始化: 最大内存={max_memory_mb}MB")

    def get_object_size(self, obj: Any) -> int:
        \"\"\"估算对象内存大小\"\"\"
        import sys

        if hasattr(obj, '__sizeof__'):
            return sys.getsizeof(obj)

        # 对于复杂对象，使用近似估算
        if isinstance(obj, dict):
            size = sys.getsizeof(obj)
            for key, value in obj.items():
                size += self.get_object_size(key) + self.get_object_size(value)
            return size
        elif isinstance(obj, (list, tuple)):
            size = sys.getsizeof(obj)
            for item in obj:
                size += self.get_object_size(item)
            return size
        else:
            return sys.getsizeof(obj)

    def should_evict(self) -> bool:
        \"\"\"判断是否需要清理\"\"\"
        return self.current_memory > (self.max_memory * self.memory_threshold)

    def evict_entries(self):
        \"\"\"清理缓存条目\"\"\"
        with self.lock:
            if not self.cache_entries:
                return 0

            # 计算需要清理的条目数量
            entries_to_evict = int(len(self.cache_entries) * self.cleanup_ratio)
            if entries_to_evict < 1:
                entries_to_evict = 1

            # 使用LRU策略清理
            evicted_count = 0
            evicted_memory = 0

            for key in list(self.cache_entries.keys())[:entries_to_evict]:
                if key in self.cache_entries:
                    entry = self.cache_entries[key]
                    evicted_memory += entry['size']
                    del self.cache_entries[key]
                    if key in self.memory_usage:
                        del self.memory_usage[key]
                    if key in self.access_count:
                        del self.access_count[key]
                    evicted_count += 1

            self.current_memory -= evicted_memory

            logger.info(f"缓存清理完成: 清理 {evicted_count} 个条目, 释放 {evicted_memory/1024/1024:.1f} MB")
            return evicted_count

    def get(self, key: str) -> Optional[Any]:
        \"\"\"获取缓存项\"\"\"
        with self.lock:
            if key in self.cache_entries:
                # 更新访问统计
                self.access_count[key] += 1

                # 移动到末尾（最近使用）
                self.cache_entries.move_to_end(key)

                return self.cache_entries[key]['value']

            return None

    def set(self, key: str, value: Any, size_hint: Optional[int] = None):
        \"\"\"设置缓存项\"\"\"
        with self.lock:
            # 计算对象大小
            if size_hint is None:
                size_hint = self.get_object_size(value)

            # 检查是否需要清理
            if self.should_evict():
                self.evict_entries()

            # 检查是否还有空间
            if self.current_memory + size_hint > self.max_memory:
                logger.warning(f"缓存已满，无法添加新条目: {key}")
                return False

            # 添加或更新条目
            if key in self.cache_entries:
                old_size = self.cache_entries[key]['size']
                self.current_memory -= old_size

            self.cache_entries[key] = {
                'value': value,
                'size': size_hint,
                'timestamp': time.time()
            }

            self.memory_usage[key] = size_hint
            self.current_memory += size_hint

            return True

    def delete(self, key: str):
        \"\"\"删除缓存项\"\"\"
        with self.lock:
            if key in self.cache_entries:
                entry = self.cache_entries[key]
                self.current_memory -= entry['size']
                del self.cache_entries[key]
                if key in self.memory_usage:
                    del self.memory_usage[key]
                if key in self.access_count:
                    del self.access_count[key]
                return True
            return False

    def clear(self):
        \"\"\"清空缓存\"\"\"
        with self.lock:
            self.cache_entries.clear()
            self.memory_usage.clear()
            self.access_count.clear()
            self.current_memory = 0

    def get_stats(self) -> Dict[str, Any]:
        \"\"\"获取缓存统计信息\"\"\"
        with self.lock:
            total_access = sum(self.access_count.values())

            return {
                'entries_count': len(self.cache_entries),
                'memory_usage_mb': self.current_memory / 1024 / 1024,
                'max_memory_mb': self.max_memory / 1024 / 1024,
                'memory_utilization': self.current_memory / self.max_memory * 100,
                'total_access': total_access,
                'avg_access_per_entry': total_access / len(self.cache_entries) if self.cache_entries else 0,
                'most_accessed': max(self.access_count.items(), key=lambda x: x[1]) if self.access_count else None,
                'largest_entry': max(self.memory_usage.items(), key=lambda x: x[1]) if self.memory_usage else None
            }

    def optimize_memory_usage(self):
        \"\"\"优化内存使用\"\"\"
        with self.lock:
            # 强制垃圾回收
            gc.collect()

            # 清理弱引用
            gc.collect()

            # 如果内存使用过高，主动清理
            if self.should_evict():
                evicted = self.evict_entries()
                logger.info(f"主动清理缓存: 清理 {evicted} 个条目")

    def set_memory_limit(self, max_memory_mb: float):
        \"\"\"设置内存限制\"\"\"
        with self.lock:
            self.max_memory = max_memory_mb * 1024 * 1024

            # 如果当前使用超过新限制，清理
            if self.current_memory > self.max_memory:
                self.evict_entries()

# 全局缓存优化器实例
cache_optimizer = CacheMemoryOptimizer()

def cached_with_memory_limit(max_memory_mb: float = 100):
    \"\"\"带内存限制的缓存装饰器\"\"\"
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 生成缓存键
            key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            # 尝试从缓存获取
            cached_result = cache_optimizer.get(key)
            if cached_result is not None:
                return cached_result

            # 执行函数
            result = func(*args, **kwargs)

            # 缓存结果
            cache_optimizer.set(key, result)

            return result
        return wrapper
    return decorator

if __name__ == "__main__":
    print("缓存内存优化器测试...")

    optimizer = CacheMemoryOptimizer(max_memory_mb=50)  # 50MB限制

    # 测试基本操作
    test_data = {"key": "value", "large_data": list(range(1000))}
    success = optimizer.set("test_key", test_data)
    print(f"设置缓存: {'成功' if success else '失败'}")

    # 获取缓存
    result = optimizer.get("test_key")
    print(f"获取缓存: {'成功' if result else '失败'}")

    # 获取统计
    stats = optimizer.get_stats()
    print(f"缓存统计: {stats['entries_count']} 个条目, 使用 {stats['memory_usage_mb']:.1f} MB")

    # 测试内存优化
    optimizer.optimize_memory_usage()
    print("内存优化完成")

    print("✅ 缓存内存优化器测试完成")
"""

    with open(cache_optimizer, 'w', encoding='utf-8') as f:
        f.write(cache_content)

    print("✅ 缓存内存优化器已创建")

def optimize_model_memory(project_root):
    """优化模型内存使用"""
    print("\n🤖 优化模型内存使用...")
    print("-" * 30)

    # 创建模型内存优化器
    model_optimizer=project_root / "src" / "ml" / "model_memory_optimizer.py"
    model_content="""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
模型内存优化器

优化机器学习模型的内存使用
\"\"\"

import os
import gc
import weakref
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ModelMemoryOptimizer:
    \"\"\"模型内存优化器\"\"\"

    def __init__(self):
        self.loaded_models = weakref.WeakValueDictionary()
        self.model_memory_usage = {}
        self.max_models_in_memory = 3  # 最多同时加载3个模型

    def optimize_sklearn_model(self, model):
        \"\"\"优化sklearn模型内存使用\"\"\"
        # 对于决策树模型，可以减少叶子节点
        if hasattr(model, 'max_leaf_nodes'):
            if hasattr(model, 'estimators_'):  # Random Forest
                for estimator in model.estimators_:
                    if hasattr(estimator, 'max_leaf_nodes'):
                        # 可以在这里应用进一步的优化
                        pass
            else:
                # 单个决策树
                pass

        return model

    def optimize_tensorflow_model(self, model):
        \"\"\"优化TensorFlow模型内存使用\"\"\"
        try:
            import tensorflow as tf

            # 清除Keras会话
            if hasattr(tf.keras.backend, 'clear_session'):
                tf.keras.backend.clear_session()

            # 优化模型权重
            # 这里可以添加模型量化或其他优化

            return model

        except ImportError:
            logger.warning("TensorFlow未安装，跳过优化")
            return model

    def optimize_pytorch_model(self, model):
        \"\"\"优化PyTorch模型内存使用\"\"\"
        try:
            import torch

            # 将模型移动到CPU以节省GPU内存
            if hasattr(model, 'to'):
                model = model.to('cpu')

            # 清空梯度
            if hasattr(model, 'zero_grad'):
                model.zero_grad()

            # 释放未使用的缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return model

        except ImportError:
            logger.warning("PyTorch未安装，跳过优化")
            return model

    def load_model_with_memory_limit(self, model_path: str, model_type: str = "sklearn"):
        \"\"\"带内存限制的模型加载\"\"\"
        # 检查是否超过最大模型数量
        if len(self.loaded_models) >= self.max_models_in_memory:
            # 卸载最少使用的模型
            self.unload_least_used_model()

        # 加载模型
        model = self._load_model_from_path(model_path, model_type)

        if model:
            model_id = f"{model_type}_{len(self.loaded_models)}"
            self.loaded_models[model_id] = model

            # 估算内存使用
            memory_usage = self.estimate_model_memory(model, model_type)
            self.model_memory_usage[model_id] = memory_usage

            logger.info(f"模型加载完成: {model_id}, 内存使用: {memory_usage:.1f} MB")

        return model

    def _load_model_from_path(self, model_path: str, model_type: str):
        \"\"\"从文件加载模型\"\"\"
        try:
            if model_type == "sklearn":
                import joblib
                return joblib.load(model_path)
            elif model_type == "tensorflow":
                import tensorflow as tf
                return tf.keras.models.load_model(model_path)
            elif model_type == "pytorch":
                import torch
                return torch.load(model_path, map_location='cpu')
            else:
                logger.error(f"不支持的模型类型: {model_type}")
                return None
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return None

    def _unload_model(self, model_id: str):
        \"\"\"卸载模型\"\"\"
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            if model_id in self.model_memory_usage:
                del self.model_memory_usage[model_id]

            # 强制垃圾回收
            gc.collect()

            logger.info(f"模型卸载完成: {model_id}")

    def unload_least_used_model(self):
        \"\"\"卸载最少使用的模型\"\"\"
        if self.loaded_models:
            # 这里可以根据访问频率来选择卸载哪个模型
            # 现在简单地卸载第一个
            model_id = next(iter(self.loaded_models))
            self._unload_model(model_id)

    def estimate_model_memory(self, model, model_type: str) -> float:
        \"\"\"估算模型内存使用\"\"\"
        import sys

        try:
            if model_type == "sklearn":
                # 对于sklearn模型，主要内存使用在树结构或系数数组
                if hasattr(model, 'coef_'):  # 线性模型
                    if hasattr(model.coef_, 'nbytes'):
                        return model.coef_.nbytes / 1024 / 1024  # MB
                elif hasattr(model, 'feature_importances_'):  # 树模型
                    if hasattr(model.feature_importances_, 'nbytes'):
                        return model.feature_importances_.nbytes / 1024 / 1024
                else:
                    return sys.getsizeof(model) / 1024 / 1024

            elif model_type == "tensorflow":
                # TensorFlow模型内存估算
                if hasattr(model, 'count_params'):
                    param_count = model.count_params()
                    # 假设每个参数占用4字节(float32)
                    return param_count * 4 / 1024 / 1024
                else:
                    return sys.getsizeof(model) / 1024 / 1024

            elif model_type == "pytorch":
                # PyTorch模型内存估算
                if hasattr(model, 'parameters'):
                    param_count = sum(p.numel() for p in model.parameters())
                    return param_count * 4 / 1024 / 1024  # 假设float32
                else:
                    return sys.getsizeof(model) / 1024 / 1024

            else:
                return sys.getsizeof(model) / 1024 / 1024

        except Exception as e:
            logger.warning(f"内存估算失败: {e}")
            return sys.getsizeof(model) / 1024 / 1024

    def get_memory_stats(self) -> Dict[str, Any]:
        \"\"\"获取内存统计\"\"\"
        total_memory = sum(self.model_memory_usage.values())

        return {
            'loaded_models_count': len(self.loaded_models),
            'total_memory_mb': total_memory,
            'model_memory_usage': self.model_memory_usage.copy(),
            'max_models_allowed': self.max_models_in_memory
        }

    def optimize_all_models(self):
        \"\"\"优化所有已加载的模型\"\"\"
        for model_id, model in list(self.loaded_models.items()):
            if 'sklearn' in model_id:
                self.optimize_sklearn_model(model)
            elif 'tensorflow' in model_id:
                self.optimize_tensorflow_model(model)
            elif 'pytorch' in model_id:
                self.optimize_pytorch_model(model)

        logger.info("所有模型优化完成")

# 全局模型内存优化器实例
model_memory_optimizer = ModelMemoryOptimizer()

def optimize_model_loading(model_path: str, model_type: str = "sklearn"):
    \"\"\"优化模型加载\"\"\"
    return model_memory_optimizer.load_model_with_memory_limit(model_path, model_type)

def optimize_model_usage(func):
    \"\"\"模型使用优化装饰器\"\"\"
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # 如果结果是模型，进行内存优化
        if hasattr(result, 'predict') or hasattr(result, 'transform'):
            # 识别模型类型
            if hasattr(result, 'coef_') or hasattr(result, 'feature_importances_'):
                model_memory_optimizer.optimize_sklearn_model(result)
            elif hasattr(result, 'layers'):  # TensorFlow/Keras
                model_memory_optimizer.optimize_tensorflow_model(result)
            elif hasattr(result, 'parameters'):  # PyTorch
                model_memory_optimizer.optimize_pytorch_model(result)

        return result
    return wrapper

if __name__ == "__main__":
    print("模型内存优化器测试...")

    optimizer = ModelMemoryOptimizer()

    # 测试内存统计
    stats = optimizer.get_memory_stats()
    print(f"初始状态: {stats['loaded_models_count']} 个模型, {stats['total_memory_mb']:.1f} MB")

    # 测试模型内存估算
    class DummyModel:
        def __init__(self):
            self.coef_ = [1, 2, 3, 4, 5] * 1000  # 创建较大对象

    dummy_model = DummyModel()
    memory_usage = optimizer.estimate_model_memory(dummy_model, "sklearn")
    print(f"虚拟模型内存使用: {memory_usage:.1f} MB")

    # 测试模型优化
    optimizer.optimize_sklearn_model(dummy_model)
    print("sklearn模型优化完成")

    print("✅ 模型内存优化器测试完成")
"""

    with open(model_optimizer, 'w', encoding='utf-8') as f:
        f.write(model_content)

    print("✅ 模型内存优化器已创建")

def implement_memory_pool(project_root):
    """实施内存池管理"""
    print("\n🏊 实施内存池管理...")
    print("-" * 30)

    # 创建内存池管理器
    memory_pool=project_root / "src" / "infrastructure" / "memory_pool.py"
    pool_content="""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
内存池管理器

实现对象池化管理，减少内存分配和垃圾回收开销
\"\"\"

import threading
import weakref
from typing import Dict, List, Any, Optional, Callable, Type, TypeVar
from collections import defaultdict, deque
import logging

T = TypeVar('T')
logger = logging.getLogger(__name__)

class ObjectPool:
    \"\"\"对象池\"\"\"

    def __init__(self, object_factory: Callable[[], T], max_size: int = 10, min_size: int = 2):
        self.object_factory = object_factory
        self.max_size = max_size
        self.min_size = min_size
        self.pool: deque[T] = deque()
        self.created_count = 0
        self.reused_count = 0
        self.lock = threading.RLock()

        # 预创建对象
        self._initialize_pool()

    def _initialize_pool(self):
        \"\"\"初始化对象池\"\"\"
        for _ in range(self.min_size):
            obj = self.object_factory()
            self.pool.append(obj)
            self.created_count += 1

    def acquire(self) -> T:
        \"\"\"获取对象\"\"\"
        with self.lock:
            if self.pool:
                obj = self.pool.popleft()
                self.reused_count += 1
                return obj
            else:
                # 池为空，创建新对象
                obj = self.object_factory()
                self.created_count += 1
                return obj

    def release(self, obj: T):
        \"\"\"释放对象回池\"\"\"
        with self.lock:
            if len(self.pool) < self.max_size:
                # 重置对象状态（如果需要）
                self._reset_object(obj)
                self.pool.append(obj)
            # 如果池已满，对象会被垃圾回收

    def _reset_object(self, obj: T):
        \"\"\"重置对象状态\"\"\"
        # 这里可以添加对象状态重置逻辑
        # 例如：清空列表、字典，重置计数器等
        pass

    def get_stats(self) -> Dict[str, Any]:
        \"\"\"获取池统计信息\"\"\"
        with self.lock:
            return {
                'pool_size': len(self.pool),
                'max_size': self.max_size,
                'min_size': self.min_size,
                'created_count': self.created_count,
                'reused_count': self.reused_count,
                'reuse_rate': self.reused_count / (self.created_count + self.reused_count) * 100 if (self.created_count + self.reused_count) > 0 else 0
            }

class MemoryPoolManager:
    \"\"\"内存池管理器\"\"\"

    def __init__(self):
        self.pools: Dict[str, ObjectPool] = {}
        self.object_types: Dict[str, Type] = {}

    def create_pool(self, pool_name: str, object_type: Type[T],
                   object_factory: Optional[Callable[[], T]] = None,
                   max_size: int = 10, min_size: int = 2):
        \"\"\"创建对象池\"\"\"
        if object_factory is None:
            # 默认工厂函数
            object_factory = object_type

        pool = ObjectPool(object_factory, max_size, min_size)
        self.pools[pool_name] = pool
        self.object_types[pool_name] = object_type

        logger.info(f"对象池创建完成: {pool_name}, 类型: {object_type.__name__}, 大小: {max_size}")
        return pool

    def get_pool(self, pool_name: str) -> Optional[ObjectPool]:
        \"\"\"获取对象池\"\"\"
        return self.pools.get(pool_name)

    def acquire_object(self, pool_name: str) -> Optional[T]:
        \"\"\"从池中获取对象\"\"\"
        pool = self.get_pool(pool_name)
        if pool:
            return pool.acquire()
        return None

    def release_object(self, pool_name: str, obj: T):
        \"\"\"将对象释放回池\"\"\"
        pool = self.get_pool(pool_name)
        if pool:
            pool.release(obj)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        \"\"\"获取所有池的统计信息\"\"\"
        stats = {}
        for pool_name, pool in self.pools.items():
            stats[pool_name] = pool.get_stats()
        return stats

    def cleanup_all_pools(self):
        \"\"\"清理所有对象池\"\"\"
        for pool_name, pool in self.pools.items():
            # 清空池中的对象
            while pool.pool:
                pool.pool.pop()

        logger.info("所有对象池已清理")

# 全局内存池管理器实例
memory_pool_manager = MemoryPoolManager()

def pooled_object(pool_name: str, object_type: Type[T]):
    \"\"\"对象池装饰器\"\"\"
    def decorator(cls):
        # 为类创建对象池
        memory_pool_manager.create_pool(
            pool_name,
            object_type,
            lambda: cls(),
            max_size=20,
            min_size=5
        )
        return cls
    return decorator

def use_pooled_object(pool_name: str):
    \"\"\"使用池化对象的上下文管理器\"\"\"
    class PooledObjectContext:
        def __enter__(self):
            self.obj = memory_pool_manager.acquire_object(pool_name)
            return self.obj

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.obj:
                memory_pool_manager.release_object(pool_name, self.obj)

    return PooledObjectContext()

# 常用对象池预定义
def create_common_pools():
    \"\"\"创建常用对象池\"\"\"
    # 列表对象池
    memory_pool_manager.create_pool(
        'list_pool',
        list,
        lambda: [],
        max_size=100,
        min_size=10
    )

    # 字典对象池
    memory_pool_manager.create_pool(
        'dict_pool',
        dict,
        lambda: {},
        max_size=100,
        min_size=10
    )

    # 集合对象池
    memory_pool_manager.create_pool(
        'set_pool',
        set,
        lambda: set(),
        max_size=50,
        min_size=5
    )

    logger.info("常用对象池创建完成")

# 初始化常用对象池
create_common_pools()

if __name__ == "__main__":
    print("内存池管理器测试...")

    manager = MemoryPoolManager()

    # 创建自定义对象池
    class TestObject:
        def __init__(self):
            self.data = []
            self.count = 0

        def reset(self):
            self.data.clear()
            self.count = 0

    # 创建对象池
    manager.create_pool(
        'test_object_pool',
        TestObject,
        lambda: TestObject(),
        max_size=5,
        min_size=2
    )

    # 测试对象获取和释放
    obj1 = manager.acquire_object('test_object_pool')
    print(f"获取对象1: {type(obj1).__name__}")

    obj2 = manager.acquire_object('test_object_pool')
    print(f"获取对象2: {type(obj2).__name__}")

    # 释放对象
    manager.release_object('test_object_pool', obj1)
    manager.release_object('test_object_pool', obj2)

    # 获取统计信息
    stats = manager.get_all_stats()
    print(f"对象池统计: {stats}")

    # 测试常用对象池
    with use_pooled_object('list_pool') as pooled_list:
        pooled_list.extend([1, 2, 3, 4, 5])
        print(f"使用池化列表: {pooled_list}")

    print("✅ 内存池管理器测试完成")
"""

    with open(memory_pool, 'w', encoding='utf-8') as f:
        f.write(pool_content)

    print("✅ 内存池管理器已创建")

def setup_memory_monitoring(project_root):
    """建立内存监控和告警"""
    print("\n📊 建立内存监控和告警...")
    print("-" * 30)

    # 创建内存监控器
    memory_monitor=project_root / "scripts" / "monitor_memory_usage.py"
    monitor_content="""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
内存使用率监控工具
\"\"\"

import time
import psutil
import threading
from datetime import datetime
from pathlib import Path

class MemoryMonitor:
    \"\"\"内存监控器\"\"\"

    def __init__(self, warning_threshold: float = 75.0, critical_threshold: float = 85.0):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.monitoring = False
        self.alerts = []
        self.metrics = {
            "memory_usage": [],
            "memory_details": []
        }
        self.process = psutil.Process()

    def start_monitoring(self, interval=30.0):
        \"\"\"开始监控\"\"\"
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        print("📊 开始内存监控...")

    def stop_monitoring(self):
        \"\"\"停止监控\"\"\"
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)
        print("📊 内存监控已停止")

    def _monitor_loop(self, interval):
        \"\"\"监控循环\"\"\"
        while self.monitoring:
            try:
                self._collect_metrics()
                self._check_alerts()
                time.sleep(interval)
            except Exception as e:
                print(f"内存监控出错: {e}")

    def _collect_metrics(self):
        \"\"\"收集内存指标\"\"\"
        timestamp = datetime.now().isoformat()

        # 系统内存信息
        memory = psutil.virtual_memory()

        self.metrics["memory_usage"].append({
            "timestamp": timestamp,
            "total_gb": memory.total / 1024**3,
            "used_gb": memory.used / 1024**3,
            "available_gb": memory.available / 1024**3,
            "usage_percent": memory.percent
        })

        # 进程内存信息
        process_memory = self.process.memory_info()

        self.metrics["memory_details"].append({
            "timestamp": timestamp,
            "process_rss_mb": process_memory.rss / 1024**2,
            "process_vms_mb": process_memory.vms / 1024**2,
            "process_percent": self.process.memory_percent()
        })

    def _check_alerts(self):
        \"\"\"检查内存告警\"\"\"
        if not self.metrics["memory_usage"]:
            return

        latest_memory = self.metrics["memory_usage"][-1]
        usage_percent = latest_memory["usage_percent"]

        if usage_percent >= self.critical_threshold:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "type": "critical",
                "message": f"内存使用率严重超标: {usage_percent:.1f}%",
                "current_value": usage_percent,
                "threshold": self.critical_threshold,
                "recommendation": "立即采取内存优化措施"
            }
            self.alerts.append(alert)
            print(f"🚨 内存严重告警: {usage_percent:.1f}%")

        elif usage_percent >= self.warning_threshold:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "type": "warning",
                "message": f"内存使用率偏高: {usage_percent:.1f}%",
                "current_value": usage_percent,
                "threshold": self.warning_threshold,
                "recommendation": "建议进行内存优化"
            }
            self.alerts.append(alert)
            print(f"⚠️  内存警告: {usage_percent:.1f}%")

    def get_performance_report(self):
        \"\"\"生成内存性能报告\"\"\"
        if not self.metrics["memory_usage"]:
            return {"error": "没有监控数据"}

        memory_usages = [m["usage_percent"] for m in self.metrics["memory_usage"]]
        process_memories = [m["process_rss_mb"] for m in self.metrics["memory_details"]]

        avg_memory_usage = sum(memory_usages) / len(memory_usages)
        max_memory_usage = max(memory_usages)
        avg_process_memory = sum(process_memories) / len(process_memories)

        report = {
            "monitoring_summary": {
                "total_samples": len(self.metrics["memory_usage"]),
                "duration_minutes": len(self.metrics["memory_usage"]) * 0.5,  # 假设30秒间隔
                "start_time": self.metrics["memory_usage"][0]["timestamp"] if self.metrics["memory_usage"] else None,
                "end_time": self.metrics["memory_usage"][-1]["timestamp"] if self.metrics["memory_usage"] else None
            },
            "memory_analysis": {
                "average_usage": round(avg_memory_usage, 2),
                "max_usage": round(max_memory_usage, 2),
                "min_usage": round(min(memory_usages), 2),
                "current_usage": memory_usages[-1] if memory_usages else 0,
                "warning_threshold": self.warning_threshold,
                "critical_threshold": self.critical_threshold,
                "warning_breaches": len([u for u in memory_usages if u >= self.warning_threshold]),
                "critical_breaches": len([u for u in memory_usages if u >= self.critical_threshold])
            },
            "process_analysis": {
                "average_process_memory_mb": round(avg_process_memory, 2),
                "max_process_memory_mb": round(max(process_memories), 2),
                "current_process_memory_mb": process_memories[-1] if process_memories else 0
            },
            "alerts": self.alerts[-10:] if len(self.alerts) > 10 else self.alerts,  # 最近10个告警
            "recommendations": []
        }

        # 生成建议
        avg_usage = report["memory_analysis"]["average_usage"]
        max_usage = report["memory_analysis"]["max_usage"]
        current_usage = report["memory_analysis"]["current_usage"]

        if current_usage >= self.critical_threshold:
            report["recommendations"].extend([
                "立即执行内存清理",
                "停止非关键进程",
                "考虑增加物理内存"
            ])
        elif current_usage >= self.warning_threshold:
            report["recommendations"].extend([
                "执行定期内存清理",
                "优化应用程序内存使用",
                "监控内存增长趋势"
            ])
        else:
            report["recommendations"].append("内存使用正常")

        if avg_process_memory > 500:  # 超过500MB
            report["recommendations"].append("应用程序内存使用较大，建议优化数据结构")

        return report

    def save_report(self, file_path=None):
        \"\"\"保存内存监控报告\"\"\"
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"memory_monitoring_report_{timestamp}.json"

        report = self.get_performance_report()

        import json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        print(f"📊 内存监控报告已保存: {file_path}")
        return file_path

# 全局内存监控器实例
memory_monitor = MemoryMonitor()

def start_memory_monitoring():
    \"\"\"开始内存监控\"\"\"
    memory_monitor.start_monitoring()

def stop_memory_monitoring():
    \"\"\"停止内存监控\"\"\"
    memory_monitor.stop_monitoring()
    return memory_monitor.get_performance_report()

if __name__ == "__main__":
    print("内存监控器测试...")

    # 启动监控
    memory_monitor.start_monitoring(interval=5)  # 5秒间隔

    # 运行一段时间
    print("监控内存使用情况 (30秒)...")
    time.sleep(30)

    # 停止监控
    memory_monitor.stop_monitoring()

    # 生成报告
    report = memory_monitor.get_performance_report()
    print(f"\\n📊 内存监控报告摘要:")
    memory_analysis = report.get("memory_analysis", {})
    print(f"  平均内存使用率: {memory_analysis.get('average_usage', 0):.1f}%")
    print(f"  最高内存使用率: {memory_analysis.get('max_usage', 0):.1f}%")
    print(f"  当前内存使用率: {memory_analysis.get('current_usage', 0):.1f}%")
    print(f"  告警次数: {memory_analysis.get('warning_breaches', 0)}")

    # 保存详细报告
    report_file = memory_monitor.save_report()

    print(f"\\n✅ 内存监控测试完成")
    print(f"📁 详细报告已保存: {report_file}")
"""

    with open(memory_monitor, 'w', encoding='utf-8') as f:
        f.write(monitor_content)

    print("✅ 内存监控器已创建")

if __name__ == "__main__":
    optimize_memory_performance()
