# RQA2025 用户指南

## 目录

1. [快速开始](#快速开始)
2. [策略开发](#策略开发)
3. [回测分析](#回测分析)
4. [参数优化](#参数优化)
5. [系统监控](#系统监控)
6. [Web界面使用](#web界面使用)
7. [故障排除](#故障排除)
8. [最佳实践](#最佳实践)

## 快速开始

### 1. 环境准备

确保您的系统满足以下要求：

- **操作系统**: Windows 10+ / macOS 10.15+ / Ubuntu 18.04+
- **Python版本**: 3.8 或更高版本
- **内存**: 至少 4GB RAM
- **磁盘空间**: 至少 2GB 可用空间

### 2. 安装和启动

```bash
# 1. 下载项目
git clone https://github.com/your-org/rqa2025.git
cd rqa2025

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动服务
python scripts/start_workspace.py
```

### 3. 访问系统

- **Web界面**: http://localhost:8000
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

## 策略开发

### 策略类型

RQA2025 支持多种内置策略类型：

#### 动量策略 (Momentum)
基于价格动量的趋势跟随策略。

```python
from src.strategy.strategies import create_momentum_strategy

strategy = create_momentum_strategy(
    lookback_period=20,      # 回溯周期
    momentum_threshold=0.05, # 动量阈值
    volume_threshold=1.5     # 成交量阈值
)
```

#### 均值回归策略 (Mean Reversion)
基于价格均值回归的反转策略。

```python
from src.strategy.strategies import create_mean_reversion_strategy

strategy = create_mean_reversion_strategy(
    lookback_period=20,       # 回溯周期
    std_threshold=2.0,        # 标准差阈值
    profit_target=0.05,       # 止盈目标
    stop_loss=-0.03          # 止损线
)
```

### 自定义策略开发

```python
from src.strategy.strategies.base_strategy import BaseStrategy
from src.strategy.interfaces.strategy_interfaces import StrategyConfig, StrategySignal

class MyCustomStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        # 自定义初始化逻辑

    def generate_signals(self, market_data, context=None):
        signals = []

        for symbol, data in market_data.items():
            # 自定义信号生成逻辑
            if self._custom_condition(data):
                signal = StrategySignal(
                    symbol=symbol,
                    action='BUY',
                    quantity=100,
                    price=data[-1]['close'],
                    confidence=0.8,
                    strategy_id=self.strategy_id
                )
                signals.append(signal)

        return signals

    def _custom_condition(self, data):
        # 自定义判断逻辑
        return True
```

## 回测分析

### 执行回测

```python
from src.strategy.workspace.web_api import StrategyWorkspaceAPI

# 创建API客户端
api = StrategyWorkspaceAPI()

# 配置回测参数
backtest_config = {
    "strategy_id": "your_strategy_id",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 100000.0,
    "commission": 0.0003,
    "slippage": 0.0001
}

# 执行回测
backtest_result = await api.create_backtest(backtest_config)
```

### 回测结果分析

回测完成后，您可以分析以下关键指标：

- **总收益率**: 策略的总收益表现
- **年化收益率**: 年度化的收益水平
- **夏普比率**: 风险调整后的收益
- **最大回撤**: 策略的最大亏损幅度
- **胜率**: 盈利交易占比

### 可视化分析

使用内置的可视化功能：

```python
# 生成回测分析图表
charts = await api.get_backtest_analysis_charts(backtest_id)

# 图表类型包括：
# - 收益率曲线图
# - 收益分布直方图
# - 回撤分析图
# - 月度收益热力图
```

## 参数优化

### 优化算法选择

RQA2025 支持多种优化算法：

1. **网格搜索**: 系统的参数空间搜索
2. **随机搜索**: 基于概率的参数采样
3. **贝叶斯优化**: 智能的参数优化
4. **遗传算法**: 基于进化论的优化

### 执行优化

```python
# 配置优化参数
optimization_config = {
    "strategy_id": "your_strategy_id",
    "algorithm": "bayesian_optimization",
    "parameter_ranges": {
        "lookback_period": [10, 20, 30, 50, 100],
        "momentum_threshold": [0.01, 0.05, 0.1, 0.15],
        "position_size": [100, 500, 1000, 2000]
    },
    "max_iterations": 50,
    "target": "sharpe_ratio"  # 优化目标
}

# 执行优化
optimization_result = await api.create_optimization(optimization_config)
```

### 优化结果分析

```python
# 获取最优参数
best_params = optimization_result["best_parameters"]
best_score = optimization_result["best_score"]

print(f"最优参数: {best_params}")
print(f"最优得分: {best_score}")

# 查看优化历史
convergence_history = optimization_result["convergence_history"]
```

## 系统监控

### 监控指标

RQA2025 提供全面的系统监控：

- **系统指标**: CPU使用率、内存使用率、磁盘空间
- **应用指标**: 响应时间、吞吐量、错误率
- **业务指标**: 策略表现、交易执行情况

### 设置告警

```python
# 创建告警规则
alert_rule = {
    "strategy_id": "your_strategy_id",
    "metric_name": "cpu_usage",
    "condition": ">",
    "threshold": 80.0,
    "level": "WARNING",
    "description": "CPU使用率过高告警"
}

await api.create_alert_rule(alert_rule)
```

### 查看监控数据

```python
# 获取监控指标
metrics = await api.get_monitoring_dashboard(time_range="1h")

# 查看告警信息
alerts = await api.get_active_alerts()
```

## Web界面使用

### 登录系统

1. 打开浏览器访问 http://localhost:8000
2. 点击"登录"或"注册"按钮
3. 输入用户名和密码
4. 点击"登录"进入系统

### 仪表板

仪表板显示系统的整体状态：

- **系统状态**: 显示系统运行状态
- **策略数量**: 当前创建的策略总数
- **回测任务**: 正在运行的回测任务数
- **活跃告警**: 当前活跃的告警数量

### 策略管理

1. 点击"策略管理"菜单
2. 点击"创建策略"按钮
3. 选择策略类型并填写参数
4. 点击"创建"完成策略创建
5. 在策略列表中可以查看、编辑或删除策略

### 回测分析

1. 点击"回测分析"菜单
2. 选择要回测的策略
3. 设置回测参数（时间范围、初始资金等）
4. 点击"开始回测"
5. 查看回测结果和可视化图表

### 参数优化

1. 点击"参数优化"菜单
2. 选择要优化的策略
3. 配置优化参数和算法
4. 设置参数范围和优化目标
5. 点击"开始优化"
6. 查看优化过程和结果

## 故障排除

### 常见问题

#### 1. 服务启动失败

**问题**: `python scripts/start_workspace.py` 启动失败

**解决方案**:
```bash
# 检查Python版本
python --version

# 检查依赖是否安装
pip list

# 重新安装依赖
pip install -r requirements.txt

# 检查端口是否被占用
netstat -an | findstr :8000
```

#### 2. 数据库连接失败

**问题**: 系统提示数据库连接失败

**解决方案**:
```bash
# 检查数据库服务状态
sudo systemctl status postgresql

# 检查连接配置
cat config/database.yaml

# 测试数据库连接
python -c "import psycopg2; psycopg2.connect(...)"
```

#### 3. 内存不足

**问题**: 系统运行时内存不足

**解决方案**:
- 增加系统内存
- 优化策略参数
- 减少并发任务数
- 使用数据分页加载

#### 4. API调用失败

**问题**: API请求返回错误

**解决方案**:
```bash
# 检查服务状态
curl http://localhost:8000/health

# 查看API日志
tail -f logs/api.log

# 检查请求格式
curl -X POST http://localhost:8000/api/strategies \
  -H "Content-Type: application/json" \
  -d '{"strategy_name": "test"}'
```

### 日志分析

系统日志位置：
- 主日志: `logs/workspace.log`
- API日志: `logs/api.log`
- 错误日志: `logs/error.log`

### 性能优化

1. **数据库优化**
   - 添加适当的索引
   - 优化查询语句
   - 使用连接池

2. **缓存策略**
   - 使用Redis缓存热点数据
   - 实施应用级缓存
   - 配置CDN加速

3. **异步处理**
   - 使用异步任务队列
   - 实现后台处理
   - 避免阻塞操作

## 最佳实践

### 策略开发

1. **参数验证**: 始终验证输入参数的有效性
2. **错误处理**: 实现完善的异常处理机制
3. **日志记录**: 添加详细的日志记录
4. **单元测试**: 为关键函数编写测试用例

### 回测分析

1. **数据质量**: 使用高质量的历史数据
2. **样本外测试**: 保留部分数据用于验证
3. **多市场测试**: 在不同市场条件下测试
4. **风险管理**: 实施适当的风险控制措施

### 生产部署

1. **环境分离**: 开发/测试/生产环境分离
2. **监控告警**: 配置完善的监控和告警
3. **备份策略**: 定期备份重要数据
4. **安全加固**: 实施安全最佳实践

### 性能优化

1. **代码优化**: 优化算法复杂度
2. **内存管理**: 避免内存泄漏
3. **并发控制**: 合理使用并发资源
4. **缓存利用**: 最大化缓存效果

## 支持与帮助

### 获取帮助

- **文档**: https://rqa2025.readthedocs.io/
- **问题跟踪**: https://github.com/your-org/rqa2025/issues
- **社区论坛**: https://community.rqa2025.com/
- **技术支持**: support@rqa2025.com

### 培训资源

- **入门教程**: 基础概念和快速开始
- **进阶指南**: 高级特性和最佳实践
- **视频教程**: 详细的操作演示
- **API示例**: 完整的代码示例

---

*版本: v1.0.0*
*最后更新: {datetime.now().strftime('%Y-%m-%d')}*
