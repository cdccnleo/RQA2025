# 缓存管理器使用指南

**文档版本**: 1.0  
**更新日期**: 2026-03-25  
**作者**: RQA2025开发团队

---

## 一、概述

缓存管理器 (`CacheManager`) 提供统一的缓存管理功能，支持TTL过期策略、LRU淘汰策略、缓存统计监控、缓存预热和缓存穿透保护。

## 二、核心特性

- **TTL过期策略**: 支持基于时间的缓存过期
- **LRU淘汰策略**: 当缓存满时自动淘汰最近最少使用的项
- **线程安全**: 使用RLock确保多线程安全
- **缓存统计**: 提供命中率、访问时间等统计信息
- **缓存穿透保护**: 缓存空值防止重复查询
- **缓存预热**: 支持启动时预热指定键
- **装饰器支持**: 提供便捷的缓存装饰器

## 三、基本使用

### 3.1 创建缓存管理器

```python
from strategy.persistence.cache_manager import CacheManager, CacheConfig, CacheStrategy

# 使用默认配置
manager = CacheManager()

# 使用自定义配置
config = CacheConfig(
    strategy=CacheStrategy.TTL,
    max_size=1000,
    ttl_seconds=3600.0,
    enable_stats=True
)
manager = CacheManager(config=config, cache_name="my_cache")
```

### 3.2 基本操作

```python
# 设置缓存
manager.set("key1", "value1")

# 获取缓存
value = manager.get("key1")  # 返回 "value1"

# 检查存在
exists = manager.exists("key1")  # 返回 True

# 删除缓存
manager.delete("key1")

# 清空缓存
manager.clear()
```

### 3.3 使用装饰器

```python
# 基本装饰器
@manager.cache_decorator()
def expensive_function(x, y):
    # 耗时操作
    return x + y

# 自定义键生成
@manager.cache_decorator(key_func=lambda x, y: f"sum_{x}_{y}")
def custom_key_function(x, y):
    return x + y
```

## 四、高级功能

### 4.1 缓存统计

```python
stats = manager.get_stats()
print(f"命中率: {stats.hit_rate:.2%}")
print(f"命中次数: {stats.hits}")
print(f"未命中次数: {stats.misses}")
print(f"当前大小: {stats.size}")
print(f"平均访问时间: {stats.avg_access_time:.3f}s")
```

### 4.2 全局缓存管理器

```python
from strategy.persistence.cache_manager import get_cache_manager, get_all_cache_stats

# 获取全局实例
manager = get_cache_manager("default")

# 查看所有缓存统计
all_stats = get_all_cache_stats()
```

## 五、配置说明

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| strategy | CacheStrategy | TTL | 缓存策略 |
| max_size | int | 1000 | 最大缓存条目数 |
| ttl_seconds | float | 3600.0 | TTL过期时间（秒） |
| cleanup_interval | float | 300.0 | 清理间隔（秒） |
| enable_stats | bool | True | 启用统计 |
| enable_warmup | bool | False | 启用预热 |
| warmup_keys | List[str] | [] | 预热键列表 |

## 六、最佳实践

1. **合理设置TTL**: 根据数据更新频率设置合适的过期时间
2. **监控命中率**: 命中率低于50%时需要优化缓存策略
3. **设置大小限制**: 防止内存溢出
4. **使用上下文管理器**: 确保资源正确释放

---

**相关文档**:
- [事务管理使用指南](transaction_manager_guide.md)
- [错误码参考手册](error_codes_reference.md)
