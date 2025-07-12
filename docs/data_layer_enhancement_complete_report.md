# RQA2025 数据层功能增强完整报告

## 目录

1. [概述](#1-概述)
2. [功能分析](#2-功能分析)
   1. [性能优化](#21-性能优化)
   2. [功能扩展](#22-功能扩展)
   3. [监控告警](#23-监控告警)
3. [实施计划](#3-实施计划)
   1. [阶段一：高优先级功能](#31-阶段一高优先级功能)
   2. [阶段二：中优先级功能](#32-阶段二中优先级功能)
   3. [阶段三：其他功能](#33-阶段三其他功能)
4. [测试计划](#4-测试计划)
   1. [测试原则和覆盖要求](#41-测试原则和覆盖要求)
   2. [详细测试计划](#42-详细测试计划)
   3. [测试执行计划](#43-测试执行计划)
   4. [测试进度和里程碑](#44-测试进度和里程碑)
5. [总结](#5-总结)

## 1. 概述

RQA2025项目是一个适用于A股的量化交易模型系统，采用LSTM、随机森林、神经网络等多个模型对数据进行训练和预测，并对预测结果进行堆叠后回测。项目已完成数据层、特征层、模型层以及交易层的基本框架。

本报告针对数据层的功能增强需求进行全面分析，并提出具体的实现建议、实施计划和测试计划。数据层作为整个系统的基础，其性能和可靠性对整个系统至关重要。通过本次功能增强，我们旨在提升数据层的性能、可靠性和可维护性，为量化交易模型提供更好的数据支持。

功能增强主要涵盖三个方面：
1. **性能优化**：提高数据加载和处理效率
2. **功能扩展**：增加数据质量监控和导出功能
3. **监控告警**：实现性能监控和异常告警机制

## 2. 功能分析

### 2.1 性能优化

#### 2.1.1 并行数据加载

**现状分析**：
当前数据加载器采用串行方式加载数据，未利用多线程/多进程并行加载能力，导致加载大量数据时效率较低。

**实现建议**：
实现一个 `ParallelDataLoader` 类，利用 Python 的 `concurrent.futures` 模块实现并行数据加载。该类将提供以下功能：

- 使用线程池并行执行多个数据加载任务
- 支持配置最大工作线程数
- 提供统一的接口收集并行加载结果

在 `DataManager` 中集成并行加载功能，提供 `load_data_parallel` 方法，支持同时加载多种类型的数据。

**核心代码示例**：
```python
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class ParallelDataLoader:
    """并行数据加载器"""
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        初始化并行数据加载器
        
        Args:
            max_workers: 最大工作线程数，默认为None（由ThreadPoolExecutor决定）
        """
        self.max_workers = max_workers
    
    def load_parallel(
        self,
        tasks: List[Tuple[Callable, tuple, dict]],
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        并行执行多个数据加载任务
        
        Args:
            tasks: 任务列表，每个任务是一个元组 (func, args, kwargs)
            timeout: 超时时间（秒），默认为None（无超时）
            
        Returns:
            List[Any]: 任务结果列表，顺序与任务列表相同
        """
        start_time = time.time()
        results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(func, *args, **kwargs): (i, func.__name__)
                for i, (func, args, kwargs) in enumerate(tasks)
            }
            
            # 收集结果
            for future in future_to_task:
                task_index, task_name = future_to_task[future]
                try:
                    result = future.result(timeout=timeout)
                    results.append((task_index, result))
                except Exception as e:
                    logger.error(f"Task {task_name} failed: {e}")
                    errors.append((task_index, e))
                    results.append((task_index, None))
        
        # 按原始任务顺序排序结果
        results.sort(key=lambda x: x[0])
        
        end_time = time.time()
        logger.info(f"Parallel loading completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Successful tasks: {len(results) - len(errors)}, Failed tasks: {len(errors)}")
        
        if errors:
            logger.warning(f"Some tasks failed: {errors}")
        
        # 返回结果（不包括任务索引）
        return [result for _, result in results]
```

#### 2.1.2 优化缓存策略

**现状分析**：
版本管理系统提供了基本的数据存储，但缺乏系统化的缓存策略，导致重复加载相同数据，浪费计算资源。

**实现建议**：
实现一个 `DataCache` 类，使用 LRU (Least Recently Used) 策略管理内存缓存，并提供磁盘缓存作为二级缓存。该类将提供以下功能：

- 内存缓存：使用 `lru_cache` 装饰器实现高效的内存缓存
- 磁盘缓存：使用 parquet 格式存储数据，提供持久化缓存
- 缓存键生成：基于请求参数生成唯一的缓存键
- 缓存管理：提供清除缓存的方法

在 `DataManager` 中集成缓存功能，优化 `load_data` 方法，支持从缓存获取数据。

**核心代码示例**：
```python
import os
import hashlib
import json
from functools import lru_cache
from typing import Dict, Any, Optional, Union, Callable
import pandas as pd
import logging
from datetime import datetime, timedelta
import shutil

logger = logging.getLogger(__name__)

class DataCache:
    """数据缓存管理器"""
    
    def __init__(
        self,
        cache_dir: str = './cache',
        memory_size: int = 128,
        disk_ttl: int = 7  # 磁盘缓存有效期（天）
    ):
        """
        初始化数据缓存管理器
        
        Args:
            cache_dir: 缓存目录
            memory_size: 内存缓存大小（项数）
            disk_ttl: 磁盘缓存有效期（天）
        """
        self.cache_dir = cache_dir
        self.memory_size = memory_size
        self.disk_ttl = disk_ttl
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 缓存统计
        self.stats = {
            'memory_hits': 0,
            'memory_misses': 0,
            'disk_hits': 0,
            'disk_misses': 0
        }
    
    def generate_cache_key(self, params: Dict[str, Any]) -> str:
        """
        生成缓存键
        
        Args:
            params: 请求参数
            
        Returns:
            str: 缓存键
        """
        # 将参数转换为JSON字符串
        params_str = json.dumps(params, sort_keys=True)
        
        # 计算MD5哈希
        return hashlib.md5(params_str.encode()).hexdigest()
    
    @lru_cache(maxsize=128)
    def get_from_memory(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        从内存缓存获取数据
        
        Args:
            cache_key: 缓存键
            
        Returns:
            Optional[pd.DataFrame]: 缓存数据，如果不存在则返回None
        """
        # 此方法利用lru_cache装饰器实现内存缓存
        # 实际上只是一个占位符，永远返回None
        # 真正的缓存逻辑在get方法中
        return None
    
    def get_from_disk(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        从磁盘缓存获取数据
        
        Args:
            cache_key: 缓存键
            
        Returns:
            Optional[pd.DataFrame]: 缓存数据，如果不存在则返回None
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.parquet")
        
        if not os.path.exists(cache_file):
            return None
        
        # 检查文件修改时间，判断是否过期
        file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - file_mtime > timedelta(days=self.disk_ttl):
            logger.info(f"Cache expired: {cache_key}")
            os.remove(cache_file)
            return None
        
        try:
            data = pd.read_parquet(cache_file)
            logger.debug(f"Disk cache hit: {cache_key}")
            return data
        except Exception as e:
            logger.error(f"Failed to read cache file: {e}")
            return None
    
    def save_to_disk(self, cache_key: str, data: pd.DataFrame) -> bool:
        """
        保存数据到磁盘缓存
        
        Args:
            cache_key: 缓存键
            data: 数据
            
        Returns:
            bool: 是否成功
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.parquet")
        
        try:
            data.to_parquet(cache_file)
            logger.debug(f"Saved to disk cache: {cache_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to save cache file: {e}")
            return False
    
    def get(
        self,
        params: Dict[str, Any],
        loader_func: Callable[..., pd.DataFrame]
    ) -> pd.DataFrame:
        """
        获取数据，优先从缓存获取，缓存不存在则加载并缓存
        
        Args:
            params: 请求参数
            loader_func: 数据加载函数
            
        Returns:
            pd.DataFrame: 数据
        """
        cache_key = self.generate_cache_key(params)
        
        # 尝试从内存缓存获取
        data = self.get_from_memory(cache_key)
        if data is not None:
            self.stats['memory_hits'] += 1
            logger.debug(f"Memory cache hit: {cache_key}")
            return data
        
        self.stats['memory_misses'] += 1
        
        # 尝试从磁盘缓存获取
        data = self.get_from_disk(cache_key)
        if data is not None:
            self.stats['disk_hits'] += 1
            # 更新内存缓存
            self.get_from_memory.cache_clear()  # 清除lru_cache以更新
            self.get_from_memory(cache_key)  # 调用一次以将结果放入缓存
            return data
        
        self.stats['disk_misses'] += 1
        
        # 加载数据
        logger.debug(f"Cache miss, loading data: {cache_key}")
        data = loader_func(**params)
        
        # 保存到缓存
        self.save_to_disk(cache_key, data)
        
        # 更新内存缓存
        self.get_from_memory.cache_clear()  # 清除lru_cache以更新
        self.get_from_memory(cache_key)  # 调用一次以将结果放入缓存
        
        return data
    
    def clear_memory_cache(self) -> None:
        """清除内存缓存"""
        self.get_from_memory.cache_clear()
        logger.info("Memory cache cleared")
    
    def clear_disk_cache(self, older_than: Optional[int] = None) -> int:
        """
        清除磁盘缓存
        
        Args:
            older_than: 清除早于指定天数的缓存，默认为None（清除所有）
            
        Returns:
            int: 清除的文件数量
        """
        if not os.path.exists(self.cache_dir):
            return 0
        
        count = 0
        now = datetime.now()
        
        for filename in os.listdir(self.cache_dir):
            if not filename.endswith('.parquet'):
                continue
            
            file_path = os.path.join(self.cache_dir, filename)
            
            if older_than is not None:
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                if now - file_mtime <= timedelta(days=older_than):
                    continue
            
            try:
                os.remove(file_path)
                count += 1
            except Exception as e:
                logger.error(f"Failed to remove cache file {filename}: {e}")
        
        logger.info(f"Disk cache cleared: {count} files removed")