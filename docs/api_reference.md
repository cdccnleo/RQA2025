# API接口参考文档

## 1. 交易接口

### 1.1 提交盘后委托
- **端点**: `POST /api/v1/after_hours/orders`
- **请求**:
  ```json
  {
    "symbol": "688001",
    "direction": "buy",
    "quantity": 1000,
    "client_id": "client123"
  }
  ```
- **响应**:
  ```json
  {
    "status": "success",
    "order_id": "AH202406301505001",
    "price": 98.50,
    "msg": "委托已接收"
  }
  ```

### 1.2 查询订单状态
- **端点**: `GET /api/v1/orders/{order_id}`
- **参数**:
  - `order_id`: 盘后订单ID
- **响应**:
  ```json
  {
    "status": "filled",
    "price": 98.50,
    "quantity": 1000,
    "executed": 1000,
    "settlement_date": "2024-07-01"
  }
  ```

## 2. 数据接口

### 2.1 获取LSTM预测信号
- **端点**: `POST /api/v1/predict/lstm`
- **请求**:
  ```json
  {
    "symbol": "600000",
    "features": [[0.1, 0.2, ...], ...] // 20x10数组
  }
  ```
- **响应**:
  ```json
  {
    "signals": [1, 0, 1], // 买入信号
    "probabilities": [0.85, 0.45, 0.92] // 原始概率
  }
  ```

### 2.2 数据质量监控

#### 质量报告获取
- **端点**: `GET /api/v1/data/quality/report`
- **参数**:
  - `symbol`: 股票代码
  - `start`: 开始日期(YYYY-MM-DD)
  - `end`: 结束日期(YYYY-MM-DD)
  - `metrics`: 可选指标列表(逗号分隔)
- **响应**:
  ```json
  {
    "symbol": "600000",
    "period": "2023-01",
    "metrics": {
      "price_gap": 0.02,
      "volume_spike": 3,
      "missing_rate": 0.001
    },
    "anomalies": [
      {
        "timestamp": "2023-01-15T14:30:00Z",
        "type": "price_gap", 
        "value": 0.15,
        "severity": "high"
      }
    ]
  }
  ```

#### 实时质量指标订阅
- **端点**: `WS /api/v1/data/quality/stream`
- **消息格式**:
  ```json
  {
    "symbol": "600000",
    "timestamp": "2023-01-15T14:30:00Z",
    "metric": "price_gap",
    "value": 0.15,
    "status": "warning"
  }
  ```

### 2.3 龙虎榜增量查询
- **端点**: `GET /api/v1/dragon_board/updates`
- **参数**:
  - `since`: 时间戳(ISO格式)
  - `limit`: 返回条数
- **响应**:
  ```json
  {
    "updates": [
      {
        "symbol": "688001",
        "buyer": "机构A",
        "amount": 5000000,
        "timestamp": "2024-06-30T15:30:00Z"
      }
    ],
    "next_since": "2024-06-30T15:35:00Z"
  }
  ```

## 3. 监控接口

### 3.1 行为告警订阅
- **端点**: `WS /api/v1/alerts/subscribe`
- **消息格式**:
  ```json
  {
    "alert_id": "ALERT-202406301505",
    "pattern": "fat_finger",
    "order_id": "ORDER123",
    "timestamp": "2024-06-30T15:05:00Z"
  }
  ```

### 3.2 混沌测试触发
- **端点**: `POST /api/v1/chaos/trigger`
- **请求**:
  ```json
  {
    "scenario": "network_partition",
    "duration": 60,
    "intensity": 0.7
  }
  ```
- **响应**:
  ```json
  {
    "status": "running",
    "test_id": "CHAOS-001"
  }
  ```

## 4. 日志系统

### 4.1 LogManager 接口

#### 初始化日志管理器
```python
class LogManager:
    def __init__(self,
                 name: str = "default",
                 level: int = logging.INFO,
                 sampling_rate: float = 1.0,
                 sampling_severity: int = logging.WARNING):
        """
        参数:
            name: 日志器名称
            level: 日志级别(DEBUG/INFO/WARNING/ERROR/CRITICAL)
            sampling_rate: 采样率(0.0-1.0)
            sampling_severity: 采样级别阈值
        """
```

#### 记录日志
```python
def log(self, level: int, msg: str, **kwargs):
    """
    参数:
        level: 日志级别
        msg: 日志消息
        kwargs: 额外上下文信息
    """
```

### 4.2 LogSampler 接口

#### 采样决策
```python
class LogSampler:
    def filter(self, record: logging.LogRecord) -> bool:
        """
        决定是否记录当前日志
        
        参数:
            record: 日志记录对象
            
        返回:
            bool: True表示记录日志，False表示丢弃
        """
```

#### 动态配置
```python
def configure(self, 
             sampling_rate: float = None,
             sampling_severity: int = None):
    """
    动态更新采样配置
    
    参数:
        sampling_rate: 新采样率
        sampling_severity: 新采样级别阈值
    """
```

## 5. 错误码

| 代码 | 含义 | 解决方案 |
|------|------|----------|
| 4001 | 非科创板股票 | 检查股票代码(688开头) |
| 4002 | 非交易时段 | 检查盘后交易时间(15:05-15:30) |
| 5001 | FPGA超载 | 等待恢复或切换软件路径 |
| 5002 | 风控拦截 | 联系合规部门 |

## 5. 速率限制
- 交易API: 100次/分钟
- 数据API: 1000次/分钟
- 监控API: 连接数限制50/实例

## 6. 基础设施层API

### 6.1 配置管理 (`ConfigManager`)
```python
class ConfigManager:
    def get(self, key: str, default=None) -> Any:
        """获取配置值
        Args:
            key: 配置键，支持点分隔符(如'database.host')
            default: 默认值(当键不存在时返回)
        Returns:
            配置值或默认值
        """
        
    def update(self, config: Dict[str, Any]) -> None:
        """批量更新配置
        Args:
            config: 配置字典
        """
        
    def watch(self, key: str, callback: Callable[[Any], None]) -> None:
        """监听配置变更
        Args:
            key: 要监听的配置键
            callback: 变更回调函数
        """
```

### 6.2 错误处理 (`ErrorHandler`)
```python
class ErrorHandler:
    def handle(self, error: Exception, context: Dict[str, Any]) -> None:
        """处理异常并记录上下文
        Args:
            error: 异常对象
            context: 错误上下文信息
        """
        
    def add_retry_strategy(self, strategy: RetryStrategy) -> None:
        """添加自定义重试策略
        Args:
            strategy: 重试策略实例
        """
```

### 6.3 监控系统 (`Monitor`)
```python
class Monitor:
    @contextmanager
    def track(self, name: str) -> Generator[None, None, None]:
        """跟踪代码块执行
        Args:
            name: 监控点名称
        """
        
    def record_metric(self, name: str, value: float, tags: Dict[str, str]) -> None:
        """记录自定义指标
        Args:
            name: 指标名称
            value: 指标值
            tags: 标签字典
        """
```

## 7. 单元测试示例

### 7.1 配置管理测试
```python
def test_config_manager():
    """测试配置管理基础功能"""
    config = ConfigManager()
    
    # 测试默认值
    assert config.get("nonexistent.key", "default") == "default"
    
    # 测试更新和获取
    config.update({"database": {"host": "localhost"}})
    assert config.get("database.host") == "localhost"
    
    # 测试监听
    callback_called = False
    def callback(value):
        nonlocal callback_called
        callback_called = True
    
    config.watch("database.host", callback)
    config.update({"database": {"host": "127.0.0.1"}})
    assert callback_called
```

### 7.2 错误处理测试
```python
def test_error_handler():
    """测试错误处理流程"""
    handler = ErrorHandler()
    error = ValueError("test error")
    
    # 测试基本错误处理
    with patch('logging.Logger.error') as mock_log:
        handler.handle(error, {"context": "test"})
        mock_log.assert_called_once()
    
    # 测试重试策略
    strategy = MagicMock(spec=RetryStrategy)
    handler.add_retry_strategy(strategy)
    handler.handle(error, {"context": "retry"})
    strategy.apply.assert_called_once()
```

### 7.3 性能测试示例
```python
def test_config_manager_performance(benchmark):
    """测试配置管理器性能"""
    config = ConfigManager()
    config.update({
        "database": {
            "host": "localhost",
            "port": 5432
        }
    })
    
    def get_config():
        return config.get("database.host")
    
    benchmark(get_config)
    assert benchmark.stats.stats.mean < 0.001  # 平均延迟应小于1ms
```

## 8. 测试最佳实践

1. **测试夹具管理**:
```python
# conftest.py
@pytest.fixture
def config_manager():
    """配置管理器测试夹具"""
    manager = ConfigManager()
    manager.update(DEFAULT_CONFIG)
    yield manager
    manager.clear()
```

2. **参数化测试**:
```python
@pytest.mark.parametrize("input,expected", [
    ("valid_data.csv", True),
    ("invalid_data.csv", False)
])
def test_data_validation(input, expected):
    adapter = DataAdapter()
    assert adapter.validate(input) == expected
```

3. **异步测试**:
```python
@pytest.mark.asyncio
async def test_async_config_update():
    config = AsyncConfigManager()
    await config.update_async({"key": "value"})
    assert await config.get_async("key") == "value"
```

4. **性能测试建议**:
- 使用`pytest-benchmark`插件
- 关键路径延迟应<10ms
- 批量操作吞吐量应>1000次/秒

## 9. 版本历史
- v1.0 (2024-07-02): 初始版本
- v1.1 (2024-07-03): 添加混沌测试接口
- v1.2 (2024-07-05): 新增基础设施层API和单元测试示例
