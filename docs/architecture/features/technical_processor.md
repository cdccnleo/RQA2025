# 特征工程-技术指标处理器设计与测试说明

## 1. 模块定位
- 统一RQA2025项目技术指标计算主流程，支持主流与A股扩展指标。
- 兼容多种输入类型与批量/单指标计算，支持高性能与灵活扩展。

## 2. 核心接口说明
- TechnicalProcessor(data=None, register_func=None): 初始化，支持DataFrame/Series等。
- calculate_ma/calc_ma: 计算移动平均线，支持自定义窗口。
- calculate_rsi/calc_rsi: 计算RSI，支持自定义窗口。
- calculate_macd/calc_macd: 计算MACD，支持多参数。
- calculate_bollinger/calc_bollinger: 计算布林带。
- calc_indicators/calculate_indicators: 批量计算，兼容FeatureEngineer。
- 支持A股专用指标扩展（如LIMIT_STRENGTH）。

## 3. 典型用法示例
```python
from src.features.technical.technical_processor import TechnicalProcessor
processor = TechnicalProcessor(data=df)
ma = processor.calculate_ma(window=5)
rsi = processor.calculate_rsi(window=14)
result = processor.calc_indicators(df, ['ma', 'rsi', 'macd'])
```

## 4. 测试覆盖与质量保障
- 单元测试：覆盖主流指标、批量计算、异常分支、极端行情、性能极限、A股扩展等。
- 并发/异常/Mock分支：覆盖多线程、依赖异常、文件IO等。
- 所有测试用例均import本模块，确保主流程一致性。

## 5. 优化与扩展建议
- 新增指标、性能优化、A股扩展等均以本模块为唯一入口。
- 持续合并冗余实现，提升可维护性。
- 建议定期回归专项测试，关注极端行情与大数据性能。

## 6. 相关文档
- [专项优化报告](../../progress/technical_debt/technical_processor_unification_report.md)
- [技术债务治理计划](../../progress/technical_debt/technical_debt_governance_plan.md) 