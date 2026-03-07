# 测试统计日志保存说明

## 📝 快速保存测试统计

### 方法1: 保存完整测试输出（推荐）

```bash
# 保存完整测试输出和覆盖率信息
conda run -n rqa pytest tests/unit/features -n auto --cov=src.features --cov-report=term-missing --tb=line -q >> test_logs/test_stats_latest.log 2>&1

# 查看最后几行（测试统计和总覆盖率）
tail -n 50 test_logs/test_stats_latest.log
```

### 方法2: 只保存测试统计和覆盖率摘要

```bash
# 只保存关键信息（测试通过率和总覆盖率）
conda run -n rqa pytest tests/unit/features -n auto --cov=src.features --cov-report=term-missing --tb=line -q 2>&1 | Select-String -Pattern "(passed|failed|TOTAL)" >> test_logs/test_stats_summary.log

# 查看摘要
cat test_logs/test_stats_summary.log
```

### 方法3: 保存详细覆盖率报告（包含所有模块）

```bash
# 保存完整覆盖率报告（包含所有模块的覆盖率）
conda run -n rqa pytest tests/unit/features -n auto --cov=src.features --cov-report=term-missing --tb=line -q 2>&1 | Select-String -Pattern "(src\\features|TOTAL)" >> test_logs/coverage_details.log

# 查看低覆盖率模块（<70%）
cat test_logs/coverage_details.log | Select-String -Pattern "(6[0-9]%|5[0-9]%|4[0-9]%|3[0-9]%|2[0-9]%|1[0-9]%|[0-9]%)"
```

## 🔍 检索低覆盖率模块

### 查找覆盖率低于70%的模块

```powershell
# PowerShell
Get-Content test_logs/test_stats_latest.log | Select-String -Pattern "(src\\features.*[0-9]+%.*[0-9]+%)" | Select-String -Pattern "(6[0-9]%|5[0-9]%|4[0-9]%|3[0-9]%|2[0-9]%|1[0-9]%|[0-9]%)" | Select-Object -First 20
```

### 查找测试失败的模块

```powershell
# PowerShell
Get-Content test_logs/test_stats_latest.log | Select-String -Pattern "(FAILED|ERROR)" | Select-Object -First 20
```

## 📊 常用命令组合

### 完整测试并保存日志

```powershell
# 运行测试并保存完整日志
conda run -n rqa pytest tests/unit/features -n auto --cov=src.features --cov-report=term-missing --tb=line -q >> test_logs/test_stats_$(Get-Date -Format 'yyyyMMdd_HHmmss').log 2>&1

# 同时显示最后几行
conda run -n rqa pytest tests/unit/features -n auto --cov=src.features --cov-report=term-missing --tb=line -q 2>&1 | Tee-Object -FilePath test_logs/test_stats_latest.log | Select-Object -Last 10
```

### 快速检查并保存摘要

```powershell
# 快速检查测试通过率和总覆盖率
conda run -n rqa pytest tests/unit/features -n auto --tb=line -q 2>&1 | Select-String -Pattern "(passed|failed)" | Select-Object -Last 1 >> test_logs/test_pass_rate.log

# 查看测试通过率
cat test_logs/test_pass_rate.log
```

## 📁 日志文件说明

- `test_logs/test_stats_latest.log` - 最新完整测试统计日志
- `test_logs/test_stats_summary.log` - 测试统计摘要（仅通过率和总覆盖率）
- `test_logs/coverage_details.log` - 详细覆盖率信息（所有模块）
- `test_logs/test_pass_rate.log` - 测试通过率记录

## 💡 使用建议

1. **日常开发**: 使用 `>> test_logs/test_stats_latest.log` 保存完整日志
2. **快速检查**: 使用 `Select-String` 过滤关键信息后保存
3. **定期清理**: 定期清理旧日志文件，保留最新的几个版本
4. **检索低覆盖率**: 使用 `Select-String` 过滤低覆盖率模块

## 🔄 自动化示例

### 创建批处理脚本（可选）

创建 `scripts/run_tests_and_save.bat`:

```batch
@echo off
set TIMESTAMP=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

conda run -n rqa pytest tests/unit/features -n auto --cov=src.features --cov-report=term-missing --tb=line -q >> test_logs\test_stats_%TIMESTAMP%.log 2>&1

echo.
echo 测试统计已保存到: test_logs\test_stats_%TIMESTAMP%.log
echo.
echo 最后10行:
tail -n 10 test_logs\test_stats_%TIMESTAMP%.log
```

### 创建PowerShell脚本（可选）

创建 `scripts/run_tests_and_save.ps1`:

```powershell
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = "test_logs\test_stats_$timestamp.log"

conda run -n rqa pytest tests/unit/features -n auto --cov=src.features --cov-report=term-missing --tb=line -q >> $logFile 2>&1

Write-Host "`n测试统计已保存到: $logFile`n"
Write-Host "最后10行:"
Get-Content $logFile -Tail 10
```

