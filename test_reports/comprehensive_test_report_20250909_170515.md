# 综合测试报告

**生成时间**: 2025-09-09T17:05:15.578600

## 📊 总体状态

**验收状态**: ❌ FAILED

## 🧪 测试执行结果

❌ 测试执行失败
- **总测试数**: 0
- **通过**: 0
- **失败**: 0
- **跳过**: 0
- **执行时间**: 0.00秒

## 📈 覆盖率分析

❌ **总体覆盖率**: 2.7% (未达标，需要≥80%)
- **覆盖文件数**: 88561
- **总文件数**: 1133
- **覆盖行数**: 88561
- **总行数**: 0

## 🎯 验收标准评估

❌ **coverage_target**: 2.6605445136683854 (要求: 80.0)
❌ **test_success_rate**: 0.0 (要求: 95.0)
❌ **critical_modules_covered**: False (要求: True)

## 💡 改进建议

- 覆盖率未达标：当前 2.7%，需要达到80.0%
- 测试成功率未达标：当前 0.0%，需要达到95.0%
- 发现 1210 个未测试的关键模块

## 🔍 测试缺口分析

### 未测试模块 (1210 个)

- aliases.py
- main.py
- adapters\base_adapter.py
- adapters\market_adapters.py
- adapters\minqmt.py
- adapters\professional_data_adapters.py
- adapters\qmt_adapter.py
- async\components\health_checker.py
- async\components\infra_processor.py
- async\components\monitoring_processor.py
- ... 还有 1200 个

### 低覆盖率文件 (1108 个)

- src\adapters\base_adapter.py: 22.1%
- src\adapters\market_adapters.py: 0.0%
- src\adapters\minqmt.py: 12.6%
- src\adapters\professional_data_adapters.py: 0.0%
- src\adapters\qmt_adapter.py: 0.0%
- src\aliases.py: 0.0%
- src\async\components\health_checker.py: 0.0%
- src\async\components\infra_processor.py: 0.0%
- src\async\components\monitoring_processor.py: 0.0%
- src\async\components\system_processor.py: 0.0%
