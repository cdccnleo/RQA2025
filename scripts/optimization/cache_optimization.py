#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据缓存机制优化脚本
实现高效的数据缓存和性能优化功能
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from collections import OrderedDict
import pickle
import hashlib


@dataclass
class CacheConfig:
    """缓存配置"""
    max_size: int = 1000  # 最大缓存条目数
    ttl_seconds: int = 3600  # 缓存生存时间(秒)
    cleanup_interval: int = 300  # 清理间隔(秒)
    enable_persistence: bool = True  # 启用持久化
    compression_enabled: bool = True  # 启用压缩
    memory_limit_mb: int = 100  # 内存限制(MB)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0


class LRUCache:
    """LRU缓存实现"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            if key in self.cache:
                # 移动到末尾(最近使用)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None

    def put(self, key: str, value: Any) -> None:
        """放入缓存"""
        with self.lock:
            if key in self.cache:
                # 更新现有条目
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # 移除最久未使用的条目
                self.cache.popitem(last=False)

            self.cache[key] = value

    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)

    def clear(self) -> None:
        """清空缓存"""
        with self.lock:
            self.cache.clear()


class DataCacheManager:
    """数据缓存管理器"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = LRUCache(config.max_size)
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
        self.cleanup_thread = None
        self.running = False

    def start(self):
        """启动缓存管理器"""
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()

    def stop(self):
        """停止缓存管理器"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        self.stats["total_requests"] += 1

        # 生成缓存键
        cache_key = self._generate_cache_key(key)

        # 尝试从缓存获取
        cached_entry = self.cache.get(cache_key)
        if cached_entry:
            # 检查是否过期
            if time.time() - cached_entry.timestamp < self.config.ttl_seconds:
                self.stats["hits"] += 1
                cached_entry.access_count += 1
                return cached_entry.value
            else:
                # 过期，移除
                self.cache.cache.pop(cache_key, None)

        self.stats["misses"] += 1
        return None

    def put(self, key: str, value: Any) -> None:
        """放入缓存数据"""
        cache_key = self._generate_cache_key(key)

        # 计算数据大小
        size_bytes = self._calculate_size(value)

        # 创建缓存条目
        entry = CacheEntry(
            key=cache_key,
            value=value,
            timestamp=time.time(),
            access_count=1,
            size_bytes=size_bytes
        )

        # 检查内存限制
        if self._check_memory_limit(size_bytes):
            self.cache.put(cache_key, entry)
        else:
            self.stats["evictions"] += 1

    def _generate_cache_key(self, key: str) -> str:
        """生成缓存键"""
        return hashlib.md5(key.encode()).hexdigest()

    def _calculate_size(self, value: Any) -> int:
        """计算数据大小"""
        try:
            return len(pickle.dumps(value))
        except:
            return 1024  # 默认1KB

    def _check_memory_limit(self, new_size: int) -> bool:
        """检查内存限制"""
        total_size = sum(entry.size_bytes for entry in self.cache.cache.values())
        return (total_size + new_size) <= (self.config.memory_limit_mb * 1024 * 1024)

    def _cleanup_worker(self):
        """清理工作线程"""
        while self.running:
            time.sleep(self.config.cleanup_interval)
            self._cleanup_expired()

    def _cleanup_expired(self):
        """清理过期条目"""
        current_time = time.time()
        expired_keys = []

        for key, entry in self.cache.cache.items():
            if current_time - entry.timestamp > self.config.ttl_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            self.cache.cache.pop(key, None)
            self.stats["evictions"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        hit_rate = 0
        if self.stats["total_requests"] > 0:
            hit_rate = self.stats["hits"] / self.stats["total_requests"]

        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "total_requests": self.stats["total_requests"],
            "hit_rate": hit_rate,
            "evictions": self.stats["evictions"],
            "current_size": self.cache.size(),
            "max_size": self.config.max_size
        }


class CacheOptimizer:
    """缓存优化器"""

    def __init__(self, cache_manager: DataCacheManager):
        self.cache_manager = cache_manager
        self.optimization_history = []

    def optimize_cache_config(self, usage_patterns: Dict[str, Any]) -> CacheConfig:
        """优化缓存配置"""
        # 基于使用模式调整配置
        hit_rate = usage_patterns.get("hit_rate", 0.5)
        avg_request_rate = usage_patterns.get("avg_request_rate", 100)
        memory_usage = usage_patterns.get("memory_usage", 50)

        # 调整缓存大小
        if hit_rate < 0.3:
            # 低命中率，增加缓存大小
            new_max_size = min(self.cache_manager.config.max_size * 2, 5000)
        elif hit_rate > 0.8:
            # 高命中率，可以减少缓存大小
            new_max_size = max(self.cache_manager.config.max_size // 2, 500)
        else:
            new_max_size = self.cache_manager.config.max_size

        # 调整TTL
        if avg_request_rate > 1000:
            # 高请求率，减少TTL
            new_ttl = max(self.cache_manager.config.ttl_seconds // 2, 1800)
        else:
            new_ttl = self.cache_manager.config.ttl_seconds

        # 调整内存限制
        if memory_usage > 80:
            # 高内存使用，减少限制
            new_memory_limit = max(self.cache_manager.config.memory_limit_mb // 2, 50)
        else:
            new_memory_limit = self.cache_manager.config.memory_limit_mb

        optimized_config = CacheConfig(
            max_size=new_max_size,
            ttl_seconds=new_ttl,
            cleanup_interval=self.cache_manager.config.cleanup_interval,
            enable_persistence=self.cache_manager.config.enable_persistence,
            compression_enabled=self.cache_manager.config.compression_enabled,
            memory_limit_mb=new_memory_limit
        )

        # 记录优化历史
        self.optimization_history.append({
            "timestamp": time.time(),
            "original_config": asdict(self.cache_manager.config),
            "optimized_config": asdict(optimized_config),
            "usage_patterns": usage_patterns
        })

        return optimized_config

    def generate_optimization_report(self) -> Dict[str, Any]:
        """生成优化报告"""
        return {
            "cache_stats": self.cache_manager.get_stats(),
            "optimization_history": self.optimization_history,
            "recommendations": self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        stats = self.cache_manager.get_stats()
        recommendations = []

        if stats["hit_rate"] < 0.5:
            recommendations.append("缓存命中率较低，建议增加缓存大小或调整TTL")

        if stats["evictions"] > stats["hits"] * 0.1:
            recommendations.append("缓存淘汰率较高，建议增加内存限制")

        if stats["current_size"] < stats["max_size"] * 0.3:
            recommendations.append("缓存利用率较低，可以考虑减少缓存大小")

        return recommendations


class CachePerformanceMonitor:
    """缓存性能监控器"""

    def __init__(self):
        self.metrics = {
            "response_times": [],
            "memory_usage": [],
            "throughput": []
        }

    def record_response_time(self, response_time: float):
        """记录响应时间"""
        self.metrics["response_times"].append(response_time)
        if len(self.metrics["response_times"]) > 1000:
            self.metrics["response_times"] = self.metrics["response_times"][-1000:]

    def record_memory_usage(self, memory_mb: float):
        """记录内存使用"""
        self.metrics["memory_usage"].append(memory_mb)
        if len(self.metrics["memory_usage"]) > 1000:
            self.metrics["memory_usage"] = self.metrics["memory_usage"][-1000:]

    def record_throughput(self, requests_per_second: float):
        """记录吞吐量"""
        self.metrics["throughput"].append(requests_per_second)
        if len(self.metrics["throughput"]) > 1000:
            self.metrics["throughput"] = self.metrics["throughput"][-1000:]

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        response_times = self.metrics["response_times"]
        memory_usage = self.metrics["memory_usage"]
        throughput = self.metrics["throughput"]

        return {
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "avg_memory_usage": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            "max_memory_usage": max(memory_usage) if memory_usage else 0,
            "avg_throughput": sum(throughput) / len(throughput) if throughput else 0,
            "max_throughput": max(throughput) if throughput else 0
        }


def main():
    """主函数"""
    print("🔧 启动数据缓存机制优化...")

    # 创建缓存配置
    config = CacheConfig(
        max_size=1000,
        ttl_seconds=3600,
        cleanup_interval=300,
        enable_persistence=True,
        compression_enabled=True,
        memory_limit_mb=100
    )

    # 创建缓存管理器
    cache_manager = DataCacheManager(config)
    cache_manager.start()

    # 创建缓存优化器
    optimizer = CacheOptimizer(cache_manager)

    # 创建性能监控器
    monitor = CachePerformanceMonitor()

    # 模拟数据操作
    print("📊 模拟缓存操作...")

    # 模拟市场数据
    market_data = {
        "timestamp": "2025-07-27 12:00:00",
        "volatility": 0.18,
        "trading_volume": {
            "average_volume": 2500000
        },
        "market_conditions": {
            "stress_index": 0.45
        }
    }

    # 测试缓存操作
    start_time = time.time()

    # 放入缓存
    cache_manager.put("market_data", market_data)

    # 获取缓存
    cached_data = cache_manager.get("market_data")

    response_time = (time.time() - start_time) * 1000  # 转换为毫秒
    monitor.record_response_time(response_time)
    monitor.record_memory_usage(50.5)  # 模拟内存使用
    monitor.record_throughput(1000)  # 模拟吞吐量

    # 获取统计信息
    cache_stats = cache_manager.get_stats()
    performance_stats = monitor.get_performance_stats()

    # 优化缓存配置
    usage_patterns = {
        "hit_rate": cache_stats["hit_rate"],
        "avg_request_rate": 1000,
        "memory_usage": 60
    }

    optimized_config = optimizer.optimize_cache_config(usage_patterns)

    # 生成优化报告
    optimization_report = optimizer.generate_optimization_report()

    # 停止缓存管理器
    cache_manager.stop()

    print("✅ 缓存优化完成!")

    # 打印结果
    print("\n" + "="*50)
    print("🎯 缓存性能统计:")
    print("="*50)
    print(f"缓存命中率: {cache_stats['hit_rate']:.2%}")
    print(f"缓存大小: {cache_stats['current_size']}/{cache_stats['max_size']}")
    print(f"平均响应时间: {performance_stats['avg_response_time']:.3f}ms")
    print(f"平均内存使用: {performance_stats['avg_memory_usage']:.1f}MB")
    print(f"平均吞吐量: {performance_stats['avg_throughput']:.0f} req/s")
    print("="*50)

    # 保存优化报告
    output_dir = Path("reports/optimization/")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_report_file = output_dir / "cache_optimization_report.json"
    with open(cache_report_file, 'w', encoding='utf-8') as f:
        json.dump(optimization_report, f, ensure_ascii=False, indent=2)

    print(f"📄 优化报告已保存: {cache_report_file}")


if __name__ == "__main__":
    main()
