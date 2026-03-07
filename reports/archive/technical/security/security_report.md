# RQA2025 代码安全审查报告

## 📊 审查摘要

**审查时间**: 2025-07-21 21:16:37
**总文件数**: 30
**通过文件**: 30
**失败文件**: 0
**平均安全评分**: 85.7/100

## 🔒 安全规则

### 禁止的导入
- os, sys, subprocess, eval, exec, globals, locals
- input, raw_input, compile, reload, __import__

### 禁止的函数
- eval(), exec(), compile(), input()
- os.system(), os.popen(), subprocess.call()

### 复杂度限制
- 最大圈复杂度: 10
- 最大嵌套深度: 5
- 最大代码行数: 500

## 📋 详细结果


### src/infrastructure/async_inference_engine.py
- **状态**: ✅ 通过
- **安全评分**: 100/100
- **问题数**: 0
- **警告数**: 0


### src/infrastructure/auto_recovery.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: invalid syntax (<unknown>, line 23) (行 23, 严重度 8)
- AST解析错误: invalid syntax (<unknown>, line 23) (行 23, 严重度 8)
**警告**:
- 复杂度检查错误: invalid syntax (<unknown>, line 23) (严重度 5)

### src/infrastructure/cache/redis_cache.py
- **状态**: ✅ 通过
- **安全评分**: 73/100
- **问题数**: 3
- **警告数**: 1

**问题**:
- 语法错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (行 1, 严重度 8)
- 危险导入: sys (行 1, 严重度 9)
- AST解析错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (行 1, 严重度 8)
**警告**:
- 复杂度检查错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (严重度 5)
**建议**:
- 移除危险导入，使用安全的替代方案

### src/infrastructure/cache/thread_safe_cache.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: invalid syntax (<unknown>, line 1) (行 1, 严重度 8)
- AST解析错误: invalid syntax (<unknown>, line 1) (行 1, 严重度 8)
**警告**:
- 复杂度检查错误: invalid syntax (<unknown>, line 1) (严重度 5)

### src/infrastructure/circuit_breaker.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: unexpected indent (<unknown>, line 9) (行 9, 严重度 8)
- AST解析错误: unexpected indent (<unknown>, line 9) (行 9, 严重度 8)
**警告**:
- 复杂度检查错误: unexpected indent (<unknown>, line 9) (严重度 5)

### src/data/adapters/base_adapter.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: unexpected indent (<unknown>, line 18) (行 18, 严重度 8)
- AST解析错误: unexpected indent (<unknown>, line 18) (行 18, 严重度 8)
**警告**:
- 复杂度检查错误: unexpected indent (<unknown>, line 18) (严重度 5)

### src/data/adapters/china/adapter.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (行 1, 严重度 8)
- AST解析错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (行 1, 严重度 8)
**警告**:
- 复杂度检查错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (严重度 5)

### src/data/adapters/china/dragon_board.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (行 1, 严重度 8)
- AST解析错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (行 1, 严重度 8)
**警告**:
- 复杂度检查错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (严重度 5)

### src/data/adapters/china/financial_adapter.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: unexpected indent (<unknown>, line 9) (行 9, 严重度 8)
- AST解析错误: unexpected indent (<unknown>, line 9) (行 9, 严重度 8)
**警告**:
- 复杂度检查错误: unexpected indent (<unknown>, line 9) (严重度 5)

### src/data/adapters/china/index_adapter.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: invalid syntax (<unknown>, line 55) (行 55, 严重度 8)
- AST解析错误: invalid syntax (<unknown>, line 55) (行 55, 严重度 8)
**警告**:
- 复杂度检查错误: invalid syntax (<unknown>, line 55) (严重度 5)

### src/features/config.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (行 1, 严重度 8)
- AST解析错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (行 1, 严重度 8)
**警告**:
- 复杂度检查错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (严重度 5)

### src/features/feature_config.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (行 1, 严重度 8)
- AST解析错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (行 1, 严重度 8)
**警告**:
- 复杂度检查错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (严重度 5)

### src/features/feature_engine.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: invalid syntax (<unknown>, line 7) (行 7, 严重度 8)
- AST解析错误: invalid syntax (<unknown>, line 7) (行 7, 严重度 8)
**警告**:
- 复杂度检查错误: invalid syntax (<unknown>, line 7) (严重度 5)

### src/features/feature_engineer.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: invalid syntax (<unknown>, line 10) (行 10, 严重度 8)
- AST解析错误: invalid syntax (<unknown>, line 10) (行 10, 严重度 8)
**警告**:
- 复杂度检查错误: invalid syntax (<unknown>, line 10) (严重度 5)

### src/features/feature_importance.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: invalid syntax (<unknown>, line 15) (行 15, 严重度 8)
- AST解析错误: invalid syntax (<unknown>, line 15) (行 15, 严重度 8)
**警告**:
- 复杂度检查错误: invalid syntax (<unknown>, line 15) (严重度 5)

### src/models/api/monitoring.py
- **状态**: ✅ 通过
- **安全评分**: 100/100
- **问题数**: 0
- **警告数**: 0


### src/models/api/rest_api.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: unindent does not match any outer indentation level (<unknown>, line 13) (行 13, 严重度 8)
- AST解析错误: unindent does not match any outer indentation level (<unknown>, line 13) (行 13, 严重度 8)
**警告**:
- 复杂度检查错误: unindent does not match any outer indentation level (<unknown>, line 13) (严重度 5)

### src/models/api/sdk_client.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: invalid syntax (<unknown>, line 10) (行 10, 严重度 8)
- AST解析错误: invalid syntax (<unknown>, line 10) (行 10, 严重度 8)
**警告**:
- 复杂度检查错误: invalid syntax (<unknown>, line 10) (严重度 5)

### src/models/api/websocket_api.py
- **状态**: ✅ 通过
- **安全评分**: 99/100
- **问题数**: 0
- **警告数**: 1

**警告**:
- 嵌套深度过高: 9 (严重度 3)
**建议**:
- 减少嵌套层级，提高代码可读性

### src/models/base_model.py
- **状态**: ✅ 通过
- **安全评分**: 96/100
- **问题数**: 1
- **警告数**: 0

**问题**:
- 风险模式: 文件操作 (行 4, 严重度 4)
**建议**:
- 检查并移除风险代码模式

### src/trading/backtest_analyzer.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: invalid syntax (<unknown>, line 11) (行 11, 严重度 8)
- AST解析错误: invalid syntax (<unknown>, line 11) (行 11, 严重度 8)
**警告**:
- 复杂度检查错误: invalid syntax (<unknown>, line 11) (严重度 5)

### src/trading/backtester.py
- **状态**: ✅ 通过
- **安全评分**: 100/100
- **问题数**: 0
- **警告数**: 0


### src/trading/broker_adapter.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: invalid character '，' (U+FF0C) (<unknown>, line 16) (行 16, 严重度 8)
- AST解析错误: invalid character '，' (U+FF0C) (<unknown>, line 16) (行 16, 严重度 8)
**警告**:
- 复杂度检查错误: invalid character '，' (U+FF0C) (<unknown>, line 16) (严重度 5)

### src/trading/execution/execution_algorithm.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: unmatched ')' (<unknown>, line 8) (行 8, 严重度 8)
- AST解析错误: unmatched ')' (<unknown>, line 8) (行 8, 严重度 8)
**警告**:
- 复杂度检查错误: unmatched ')' (<unknown>, line 8) (严重度 5)

### src/trading/execution/execution_engine.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (行 1, 严重度 8)
- AST解析错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (行 1, 严重度 8)
**警告**:
- 复杂度检查错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (严重度 5)

### src/backtest/analyzer.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (行 1, 严重度 8)
- AST解析错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (行 1, 严重度 8)
**警告**:
- 复杂度检查错误: invalid character '：' (U+FF1A) (<unknown>, line 1) (严重度 5)

### src/backtest/backtest_engine.py
- **状态**: ✅ 通过
- **安全评分**: 100/100
- **问题数**: 0
- **警告数**: 0


### src/backtest/data_loader.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: invalid syntax (<unknown>, line 8) (行 8, 严重度 8)
- AST解析错误: invalid syntax (<unknown>, line 8) (行 8, 严重度 8)
**警告**:
- 复杂度检查错误: invalid syntax (<unknown>, line 8) (严重度 5)

### src/backtest/engine.py
- **状态**: ✅ 通过
- **安全评分**: 82/100
- **问题数**: 2
- **警告数**: 1

**问题**:
- 语法错误: unexpected indent (<unknown>, line 16) (行 16, 严重度 8)
- AST解析错误: unexpected indent (<unknown>, line 16) (行 16, 严重度 8)
**警告**:
- 复杂度检查错误: unexpected indent (<unknown>, line 16) (严重度 5)

### src/backtest/evaluation/model_evaluator.py
- **状态**: ✅ 通过
- **安全评分**: 99/100
- **问题数**: 0
- **警告数**: 1

**警告**:
- 嵌套深度过高: 8 (严重度 3)
**建议**:
- 减少嵌套层级，提高代码可读性

## 🚀 安全建议

1. **定期审查**: 定期运行安全审查
2. **培训团队**: 提高团队安全意识
3. **自动化检查**: 集成到CI/CD流程
4. **持续监控**: 监控安全评分变化

---
**报告版本**: v1.0
**审查时间**: 2025-07-21 21:16:37
