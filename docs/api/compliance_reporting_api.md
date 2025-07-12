# 合规报告接口文档

## IReportGenerator 接口

### 1. 接口概述
提供合规报告生成的标准接口，所有报告生成器必须实现此接口。

### 2. 方法说明

#### generate_daily_report()
```python
def generate_daily_report(self) -> str
```
**功能**：生成每日合规报告

**参数**：无

**返回值**：
- `str`: 生成的报告文件路径

**异常**：
- `ReportGenerationError`: 报告生成失败时抛出
- `DataUnavailableError`: 数据不可用时抛出

**示例**：
```python
try:
    report_path = generator.generate_daily_report()
    print(f"报告生成成功: {report_path}")
except ReportGenerationError as e:
    print(f"报告生成失败: {str(e)}")
```

#### get_today_trades()
```python
def get_today_trades(self) -> List[Dict]
```
**功能**：获取当日交易记录

**返回值**：
- `List[Dict]`: 交易记录列表，每个字典包含：
  - `trade_id`: 交易ID
  - `symbol`: 标的代码
  - `price`: 成交价格
  - `quantity`: 数量
  - `timestamp`: 时间戳
  - `account`: 账户
  - `broker`: 券商

**示例**：
```python
trades = generator.get_today_trades()
print(f"今日交易数: {len(trades)}")
```

#### run_compliance_checks()
```python
def run_compliance_checks(self, trades: List[Dict]) -> List[Dict]
```
**功能**：执行合规检查

**参数**：
- `trades`: 要检查的交易记录列表

**返回值**：
- `List[Dict]`: 违规记录列表，每个字典包含：
  - `rule_id`: 规则ID
  - `description`: 违规描述
  - `severity`: 严重程度
  - `trade_ids`: 相关交易ID
  - `timestamp`: 检测时间

**示例**：
```python
violations = generator.run_compliance_checks(trades)
print(f"发现违规: {len(violations)}条")
```

## RegulatoryReporter 服务

### 1. 初始化
```python
def __init__(self, 
             report_generator: IReportGenerator,
             notification_service: Optional[NotificationService] = None)
```
**参数**：
- `report_generator`: 报告生成器实现
- `notification_service`: 通知服务(可选)

### 2. 主要方法

#### generate_and_send_report()
```python
def generate_and_send_report(self) -> bool
```
**返回值**：
- `bool`: 是否成功

#### get_last_report_status()
```python
def get_last_report_status(self) -> Dict
```
**返回值**：
```python
{
    "last_report_time": datetime,  # 最后报告时间
    "status": str  # "SUCCESS"或"PENDING"
}
```

## 实现检查清单
- [ ] 实现所有抽象方法
- [ ] 添加适当的日志记录
- [ ] 处理所有可能的异常
- [ ] 编写单元测试
- [ ] 性能优化(如需要)
