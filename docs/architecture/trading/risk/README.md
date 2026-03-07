# 风控与合规（risk/compliance）架构设计说明

## 1. 模块定位
risk/compliance模块为RQA2025系统所有交易、订单、执行等主流程提供统一、灵活、可扩展的风控与合规能力，是保障业务合规与风险可控的核心。

## 2. 主要子系统
- **统一风控入口**：RiskController、ChinaRiskController 统一管理各类风控规则，支持A股特有规则。
- **A股合规与风控**：T+1限制、涨跌停、科创板规则、熔断机制、持仓限制、融资融券、北向资金流、政策风险等。
- **动态风控与FPGA加速**：支持动态风险参数、市场监控、FPGA加速风控检查。
- **批量风控与联动**：支持批量订单风控、与交易、监控、告警等模块联动。
- **审计与追踪**：风控日志、违规记录、审计报告等。

## 3. 典型用法
### 单笔风控校验
```python
from src.trading.risk.risk_controller import RiskController
rc = RiskController(feature_engine)
result = rc.check_order(order, market_data)
```

### A股合规校验
```python
from src.trading.risk.china.risk_controller import ChinaRiskController
crc = ChinaRiskController(config)
result = crc.check(order)
```

### 批量风控与动态风控
```python
rc.batch_check(orders, market_data_list)
rc.dynamic_params.update_params(market_data)
```

## 4. 在主流程中的地位
- 为trading、order、execution等提供合规校验与风险控制，保障业务合规与风险可控。
- 支持A股、科创板、融资融券等多市场、多规则、多场景风控，提升系统安全性与合规性。
- 接口抽象与注册机制，便于扩展新风控规则、适配新市场、Mock测试等。

## 5. 测试与质量保障
- 已实现高质量pytest单元测试，覆盖统一风控入口、A股合规、动态风控、批量风控、审计追踪等主要功能和边界。
- 测试用例见：tests/unit/trading/risk/ 目录下相关文件。 