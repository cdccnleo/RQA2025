# 持续测试覆盖率监控系统

## 概述

持续测试覆盖率监控系统是一个自动化工具，用于定期检查和报告项目中各层的测试覆盖率状态，帮助团队持续跟踪代码质量并及时发现覆盖率下降的问题。

## 功能特性

### 🔍 自动化监控
- 定期执行测试覆盖率分析
- 支持多种报告格式（HTML、JSON、XML）
- 实时告警机制

### 📊 多维度分析
- 整体项目覆盖率统计
- 分层覆盖率分析
- 文件级覆盖率详情
- 覆盖率趋势图表

### 🚨 智能告警
- 可配置的告警阈值
- 分层告警机制
- 多种通知渠道

### 💾 数据持久化
- SQLite数据库存储历史数据
- 基准覆盖率设置
- 趋势分析功能

## 快速开始

### 1. 启动监控系统

```bash
# 启动持续监控
./scripts/start_coverage_monitor.sh

# 或者直接使用Python命令
python scripts/coverage_continuous_monitor.py --command start
```

### 2. 执行一次性监控

```bash
python scripts/coverage_continuous_monitor.py --command report --once
```

### 3. 查看监控状态

```bash
python scripts/coverage_continuous_monitor.py --command status
```

## 配置说明

### 监控配置

编辑 `config/coverage_monitor_config.json` 文件来调整监控参数：

```json
{
  "monitoring": {
    "interval_seconds": 3600,      // 监控间隔（秒）
    "timeout_seconds": 300,        // 超时时间（秒）
    "max_retries": 3              // 最大重试次数
  },
  "alerts": {
    "overall_coverage_threshold": 70.0,    // 整体覆盖率阈值
    "layer_coverage_threshold": 60.0,      // 层覆盖率阈值
    "file_coverage_threshold": 50.0        // 文件覆盖率阈值
  }
}
```

### 层配置

系统支持以下层级的监控：

- **infrastructure**: 基础设施层
- **data**: 数据层
- **strategy**: 策略层
- **trading**: 交易层
- **risk**: 风险控制层

## 使用命令

### 基本命令

```bash
# 启动监控
python scripts/coverage_continuous_monitor.py --command start

# 停止监控
python scripts/coverage_continuous_monitor.py --command stop

# 查看状态
python scripts/coverage_continuous_monitor.py --command status

# 生成报告
python scripts/coverage_continuous_monitor.py --command report

# 执行一次性监控
python scripts/coverage_continuous_monitor.py --command report --once
```

### 基准管理

```bash
# 设置基准覆盖率
python scripts/coverage_continuous_monitor.py --command baseline --layer infrastructure --coverage 75.0

# 查看基准覆盖率
python scripts/coverage_continuous_monitor.py --command baseline --layer infrastructure
```

## 报告和输出

### 报告位置
- 报告文件保存在：`reports/coverage_monitoring/`
- 数据库文件：`data/coverage_monitor.db`
- 日志文件：`logs/coverage_monitor.log`

### 报告格式
1. **JSON报告**: 包含详细的覆盖率数据和分析结果
2. **HTML报告**: 可视化界面，方便查看覆盖率趋势
3. **控制台输出**: 实时显示告警信息

### 数据库表结构

#### coverage_history 表
```sql
CREATE TABLE coverage_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    layer TEXT NOT NULL,
    coverage REAL NOT NULL,
    statements INTEGER,
    missed INTEGER,
    branches INTEGER,
    partial INTEGER
);
```

#### alerts 表
```sql
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    message TEXT NOT NULL,
    severity TEXT NOT NULL,
    layer TEXT,
    file_path TEXT
);
```

#### baselines 表
```sql
CREATE TABLE baselines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    layer TEXT NOT NULL UNIQUE,
    baseline_coverage REAL NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

## 告警机制

### 告警类型
1. **overall_coverage_low**: 整体覆盖率低于阈值
2. **layer_coverage_low**: 层覆盖率低于阈值
3. **file_coverage_low**: 文件覆盖率低于阈值

### 告警级别
- 🔴 **high**: 高优先级，需要立即处理
- 🟡 **medium**: 中优先级，需要关注
- 🟢 **low**: 低优先级，可选处理

## 最佳实践

### 1. 设置合理的阈值
根据项目实际情况设置覆盖率阈值：
- 新项目：60-70%
- 成熟项目：80-90%
- 关键组件：90%+

### 2. 定期审查基准
- 每季度调整基准覆盖率
- 根据项目进展调整阈值
- 关注覆盖率趋势而非绝对值

### 3. 持续改进
- 优先覆盖核心业务逻辑
- 关注边界条件和异常处理
- 定期代码审查和重构

## 故障排除

### 常见问题

1. **监控无法启动**
   - 检查Python环境和依赖
   - 确认数据库文件权限
   - 查看日志文件中的错误信息

2. **覆盖率数据不准确**
   - 确认pytest和coverage配置正确
   - 检查测试文件路径和命名规范
   - 验证覆盖率工具版本兼容性

3. **告警频繁触发**
   - 调整告警阈值
   - 检查测试用例质量
   - 确认代码变更影响范围

### 日志分析
查看 `logs/coverage_monitor.log` 文件来诊断问题：

```bash
tail -f logs/coverage_monitor.log
```

## 集成到CI/CD

### GitHub Actions 示例

```yaml
name: Coverage Monitoring
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  coverage-monitor:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run coverage monitoring
      run: |
        python scripts/coverage_continuous_monitor.py --command report --once
    - name: Upload coverage reports
      uses: actions/upload-artifact@v2
      with:
        name: coverage-reports
        path: reports/coverage_monitoring/
```

## 扩展开发

### 添加新的监控指标
1. 在 `ContinuousCoverageMonitor` 类中添加新的分析方法
2. 更新数据库表结构
3. 修改报告生成逻辑
4. 添加相应的告警检查

### 自定义通知渠道
1. 实现新的通知类
2. 在配置文件中添加配置项
3. 更新 `send_notifications` 方法

## 版本历史

### v1.0.0
- 初始版本发布
- 基础覆盖率监控功能
- SQLite数据库存储
- JSON/HTML报告生成
- 基础告警机制

## 贡献指南

欢迎提交Issue和Pull Request来改进这个监控系统！

## 许可证

本项目采用MIT许可证。