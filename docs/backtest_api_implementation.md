# 策略回测API实现总结

## 实现时间
2025年1月7日

## 实现内容

### 1. 回测服务层 (`src/gateway/web/backtest_service.py`)

**功能**:
- 封装后端回测引擎 (`BacktestEngine`)
- 提供统一的回测接口
- 支持降级方案（当后端组件不可用时返回模拟数据）

**主要函数**:
- `run_backtest()`: 运行策略回测
- `get_backtest_result()`: 获取回测结果
- `list_backtests()`: 列出回测任务

### 2. 回测API路由 (`src/gateway/web/backtest_routes.py`)

**API端点**:
- `POST /api/v1/backtest/run` - 运行策略回测
- `GET /api/v1/backtest/{backtest_id}` - 获取回测结果
- `GET /api/v1/backtest` - 列出回测任务

**请求模型**:
```python
class BacktestRequest:
    strategy_id: str          # 策略ID
    start_date: str           # 开始日期 (YYYY-MM-DD)
    end_date: str             # 结束日期 (YYYY-MM-DD)
    initial_capital: float    # 初始资金 (默认: 100000.0)
    commission_rate: float    # 手续费率 (默认: 0.001)
```

**响应模型**:
```python
class BacktestResponse:
    backtest_id: str          # 回测ID
    strategy_id: str          # 策略ID
    status: str               # 状态
    start_date: str           # 开始日期
    end_date: str             # 结束日期
    initial_capital: float    # 初始资金
    final_capital: float      # 最终资金
    total_return: float       # 总收益率
    annualized_return: float  # 年化收益率
    sharpe_ratio: float       # 夏普比率
    max_drawdown: float       # 最大回撤
    win_rate: float           # 胜率
    total_trades: int         # 总交易次数
    equity_curve: list        # 收益曲线
    trades: list              # 交易记录
    metrics: dict             # 其他指标
    created_at: str           # 创建时间
```

### 3. 前端页面更新 (`web-static/strategy-backtest.html`)

**更新内容**:
- 取消注释回测API调用代码
- 实现真实的API请求
- 添加错误处理
- 回测成功后自动刷新数据

**API调用示例**:
```javascript
const response = await fetch('/api/v1/backtest/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        strategy_id: strategyId,
        start_date: startDate,
        end_date: endDate,
        initial_capital: initialCapital,
        commission_rate: 0.001
    })
});
```

### 4. 路由注册 (`src/gateway/web/api.py`)

**更新内容**:
- 导入回测路由模块
- 注册回测路由器到FastAPI应用
- 添加错误处理和日志

## 使用说明

### 运行回测

1. 在策略回测页面 (`strategy-backtest.html`) 选择策略
2. 配置回测参数：
   - 回测时间范围（开始日期、结束日期）
   - 初始资金
   - 交易费用
3. 点击"开始回测"按钮
4. 系统将调用 `/api/v1/backtest/run` API执行回测
5. 回测完成后自动刷新页面数据

### API调用示例

```bash
# 运行回测
curl -X POST "http://localhost:8080/api/v1/backtest/run" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": "trend_following_123",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100000.0,
    "commission_rate": 0.001
  }'

# 获取回测结果
curl "http://localhost:8080/api/v1/backtest/backtest_trend_following_123_1234567890"

# 列出回测任务
curl "http://localhost:8080/api/v1/backtest?strategy_id=trend_following_123"
```

## 降级方案

当后端回测引擎不可用时，系统会自动使用模拟数据：
- 生成模拟的收益曲线
- 计算模拟的性能指标
- 确保前端页面能够正常显示

## 后续优化建议

1. **对接实际回测引擎**: 完善 `_execute_backtest()` 函数，对接实际的 `BacktestEngine` 接口
2. **持久化存储**: 实现回测结果的持久化存储，支持查询历史回测
3. **异步执行**: 对于长时间运行的回测，实现异步执行和进度查询
4. **WebSocket支持**: 添加WebSocket支持，实时推送回测进度
5. **批量回测**: 支持批量运行多个策略的回测

## 测试验证

### API端点测试

```bash
# 测试回测API
python -m pytest tests/dashboard_verification/test_api_endpoints.py::test_backtest_api -v
```

### 前端功能测试

1. 打开 `http://localhost:8080/strategy-backtest`
2. 选择策略
3. 配置回测参数
4. 点击"开始回测"
5. 验证回测结果是否正确显示

## 相关文件

- `src/gateway/web/backtest_service.py` - 回测服务层
- `src/gateway/web/backtest_routes.py` - 回测API路由
- `src/gateway/web/api.py` - API应用主文件
- `web-static/strategy-backtest.html` - 策略回测前端页面
- `src/strategy/backtest/backtest_engine.py` - 回测引擎实现
- `src/strategy/backtest/backtest_service.py` - 回测服务实现

