# 测试日志目录说明

本目录用于存储测试覆盖率报告和测试统计日志。

## 📁 文件说明

### 覆盖率报告文件

1. **`coverage_report_YYYYMMDD_HHMMSS.log`**
   - 完整的测试执行日志和覆盖率详情
   - 包含所有测试输出和错误信息
   - 按时间戳命名，保留历史记录

2. **`coverage_report_latest.log`**
   - 最新生成的完整覆盖率报告
   - 每次运行都会更新

3. **`coverage_summary_YYYYMMDD_HHMMSS.txt`**
   - 测试统计和覆盖率摘要
   - 按覆盖率排序的模块列表
   - 便于快速查看关键指标

4. **`coverage_summary_latest.txt`**
   - 最新的覆盖率摘要报告
   - 每次运行都会更新

5. **`quick_coverage_check_latest.log`**
   - 快速检查的简要日志
   - 只包含测试通过率和总覆盖率

## 🚀 使用方法

### 生成完整覆盖率报告

```bash
# 运行完整测试并生成详细报告（需要3-5分钟）
python scripts/generate_coverage_report.py
```

生成的文件：
- `test_logs/coverage_report_YYYYMMDD_HHMMSS.log` - 详细日志
- `test_logs/coverage_summary_YYYYMMDD_HHMMSS.txt` - 摘要报告
- `test_logs/coverage_report_latest.log` - 最新详细日志
- `test_logs/coverage_summary_latest.txt` - 最新摘要

### 快速检查（推荐日常使用）

```bash
# 快速检查测试通过率和总覆盖率（需要3-4分钟）
python scripts/quick_coverage_check.py
```

生成的文件：
- `test_logs/quick_coverage_check_latest.log` - 简要日志

### 查看最新报告

```bash
# 查看最新摘要
cat test_logs/coverage_summary_latest.txt

# 查看最新详细日志
cat test_logs/coverage_report_latest.log

# 查看快速检查结果
cat test_logs/quick_coverage_check_latest.log
```

## 📊 报告内容说明

### 覆盖率摘要报告包含：

1. **测试统计**
   - 通过/失败/跳过测试数量
   - 测试执行时间

2. **总覆盖率**
   - 总体代码覆盖率百分比
   - 已覆盖/未覆盖行数

3. **模块覆盖率详情**
   - 按覆盖率降序排列
   - 显示每个模块的覆盖率百分比
   - 显示未覆盖的行号

### 关键指标

- **测试通过率**: 应保持100%
- **总覆盖率**: 目标80%+
- **核心模块覆盖率**: 应达到80%+

## 💡 最佳实践

1. **日常开发**: 使用 `quick_coverage_check.py` 快速验证测试通过率
2. **提交前检查**: 运行 `generate_coverage_report.py` 生成完整报告
3. **定期审查**: 查看 `coverage_summary_latest.txt` 识别低覆盖率模块
4. **历史对比**: 保留带时间戳的报告文件，便于追踪覆盖率变化趋势

## 📝 注意事项

- 覆盖率报告生成需要运行完整测试套件，可能需要3-5分钟
- 建议在提交代码前运行完整报告
- 日常开发可以使用快速检查来验证测试通过率
- 日志文件会自动保存在 `test_logs/` 目录
