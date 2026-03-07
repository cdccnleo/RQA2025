# 增强缓存策略指南

## 功能概述

新版缓存系统提供以下增强功能：

1. **动态配置**：支持按数据类型配置不同缓存策略
2. **多级存储**：可组合内存+磁盘存储
3. **数据压缩**：支持LZ4等压缩算法
4. **版本控制**：关键数据的版本管理
5. **智能策略**：支持多种缓存更新策略

## 配置参考

### 基础配置示例

```python
from src.data.cache.cache_manager import CacheManager

cache = CacheManager()
cache.configure({
    '行情数据': {
        'strategy': 'aggressive',
        'ttl': 86400,
        'compression': 'lz4',
        'storage': ['memory', 'disk']
    },
    '龙虎榜数据': {
        'strategy': 'incremental',
        'ttl': 3600,
        'version_control': True
    }
})
```

### 配置项说明

| 参数 | 类型 | 说明 |
|------|------|------|
| strategy | str | 缓存策略：`default`/`aggressive`/`incremental` |
| ttl | int | 缓存存活时间(秒) |
| compression | str | 压缩算法：`None`/`lz4` |
| storage | list | 存储位置：`memory`/`disk` |
| version_control | bool | 是否启用版本控制 |

## 使用示例

### 基本使用

```python
# 设置缓存
cache.set('key', large_data, '行情数据')

# 获取缓存
data = cache.get('key', '行情数据')
```

### 版本控制

```python
# 第一次设置
cache.set('stock_600000', v1_data, '龙虎榜数据')

# 更新版本
cache.set('stock_600000', v2_data, '龙虎榜数据')

# 获取最新版本
latest = cache.get('stock_600000', '龙虎榜数据')
```

## 最佳实践

1. **高频小数据**：
   ```python
   {
       'strategy': 'aggressive',
       'storage': ['memory'],
       'ttl': 300  # 5分钟
   }
   ```

2. **低频大数据**：
   ```python
   {
       'strategy': 'default',
       'storage': ['disk'],
       'compression': 'lz4',
       'ttl': 86400  # 1天
   }
   ```

3. **关键历史数据**：
   ```python
   {
       'strategy': 'incremental',
       'storage': ['disk'],
       'version_control': True,
       'ttl': 604800  # 1周
   }
   ```

## 性能建议

1. 对大于1MB的数据启用压缩
2. 高频访问数据优先使用内存存储
3. 版本控制会增加约10%的存储开销
4. 增量策略适合变化率<20%的数据
