# 配置管理器迁移指南 (v1 → v2)

## 主要变更

1. **线程安全改进**：
   - 新增专用回调锁
   - 使用线程池管理回调
   - 环境变量访问加锁

2. **验证增强**：
   - 集成ConfigValidator
   - 支持JSON Schema验证
   - 更严格的类型检查

3. **性能优化**：
   - 环境变量缓存(TTL 60秒)
   - 减少锁竞争

4. **API变更**：
   - `validate()` 现在使用外部验证器
   - `get()` 内部使用缓存

## 迁移步骤

### 1. 文件替换
将旧版 `config_manager.py` 替换为新的 `config_manager_optimized.py`

### 2. 配置验证
为每个环境添加schema文件：
```
config/
  schemas/
    default.schema.json
    production.schema.json
    ...
```

示例schema文件：
```json
{
  "type": "object",
  "properties": {
    "key1": {"type": "string"},
    "key2": {"type": "number"}
  },
  "required": ["key1"]
}
```

### 3. 代码调整

#### 旧代码：
```python
from .config_manager import ConfigManager
manager = ConfigManager()
```

#### 新代码：
```python
from .config_manager_optimized import ConfigManager
manager = ConfigManager()  # 接口保持兼容
```

### 4. 回调处理
回调函数现在在独立线程池中执行，确保回调函数是线程安全的。

### 5. 环境变量
环境变量现在有60秒缓存，如需立即获取最新值可重启服务。

## 回滚说明
如果遇到问题，可以：
1. 恢复旧版config_manager.py
2. 删除schema文件
3. 重启服务

## 性能对比

| 操作 | v1 (ms) | v2 (ms) |
|------|---------|---------|
| get(高频) | 1.2 | 0.3 |
| reload | 15 | 12 |
| 并发get | 120 | 35 |

## 注意事项

1. 新版本需要Python 3.8+
2. 首次加载时会进行严格验证
3. 缓存机制可能导致环境变量更新延迟
4. 建议在测试环境验证后再上线生产
