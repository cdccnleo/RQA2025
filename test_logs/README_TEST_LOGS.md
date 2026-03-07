# 测试统计日志说明

## 📋 日志文件说明

测试统计日志保存在 `test_logs/` 目录下，文件名格式为：
- `test_statistics_YYYYMMDD_HHMMSS.log` - 完整的测试执行日志（包含覆盖率信息）

## 🔍 如何查看日志

### 查看最新日志
```powershell
# 查看最新的测试日志
Get-Content test_logs/test_statistics_*.log | Select-Object -Last 50

# 或者直接打开最新文件
Get-ChildItem test_logs/test_statistics_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { notepad $_.FullName }
```

### 检索失败测试
```powershell
# 查找失败的测试
Select-String -Path test_logs/test_statistics_*.log -Pattern "FAILED|failed|ERROR|error"

# 查找特定模块的测试结果
Select-String -Path test_logs/test_statistics_*.log -Pattern "plugins|processors|monitoring"
```

### 检索覆盖率信息
```powershell
# 查找覆盖率信息
Select-String -Path test_logs/test_statistics_*.log -Pattern "TOTAL|coverage|Coverage"

# 查找低覆盖率模块（<50%）
Select-String -Path test_logs/test_statistics_*.log -Pattern "\d+%|\d+\s+\d+\s+\d+%" | Where-Object { $_.Line -match "\d+%" -and [int]($_.Line -replace '.*?(\d+)%.*', '$1') -lt 50 }
```

### 检索跳过的测试
```powershell
# 查找跳过的测试
Select-String -Path test_logs/test_statistics_*.log -Pattern "SKIPPED|skipped"
```

## 📊 生成测试统计

### 运行测试并保存日志
```powershell
# 方法1: 使用PowerShell脚本（推荐）
.\test_logs\run_tests_simple.ps1

# 方法2: 直接使用命令（使用Out-File，更可靠）
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = "test_logs/test_statistics_$timestamp.log"
pytest tests/unit/features/ -n auto -k "not e2e" --tb=line -q --cov=src.features --cov-report=term-missing 2>&1 | Out-File -FilePath $logFile -Encoding UTF8
Write-Host "测试日志已保存到: $logFile"
```

### 快速运行测试（不保存日志）
```powershell
pytest tests/unit/features/ -n auto -k "not e2e" --tb=line -q
```

## 📈 日志内容说明

测试日志包含以下信息：
1. **测试执行结果**
   - 通过的测试数量
   - 失败的测试数量
   - 跳过的测试数量
   - 错误数量

2. **覆盖率信息**
   - 总体覆盖率
   - 各模块覆盖率
   - 未覆盖的代码行

3. **测试详情**
   - 每个测试的执行状态
   - 失败测试的错误信息
   - 跳过的测试原因

## 🔧 常用检索命令

### 查找所有失败的测试
```powershell
Get-ChildItem test_logs/test_statistics_*.log | ForEach-Object {
    Write-Host "`n=== $($_.Name) ==="
    Select-String -Path $_.FullName -Pattern "FAILED" | Select-Object -First 10
}
```

### 统计测试通过率
```powershell
Get-ChildItem test_logs/test_statistics_*.log | ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    if ($content -match "(\d+)\s+passed") {
        $passed = $matches[1]
    }
    if ($content -match "(\d+)\s+failed") {
        $failed = $matches[1]
    }
    Write-Host "$($_.Name): Passed=$passed, Failed=$failed"
}
```

### 查找低覆盖率文件
```powershell
Get-ChildItem test_logs/test_statistics_*.log | ForEach-Object {
    Write-Host "`n=== $($_.Name) ==="
    Select-String -Path $_.FullName -Pattern "^\s+\S+\s+\d+\s+\d+\s+\d+%" | 
        Where-Object { 
            $line = $_.Line.Trim()
            if ($line -match "(\d+)%$") {
                $coverage = [int]$matches[1]
                $coverage -lt 50
            } else {
                $false
            }
        } | 
        Select-Object -First 20
}
```

## 📝 注意事项

1. **日志文件大小**: 测试日志可能较大，建议定期清理旧日志
2. **日志格式**: 日志使用UTF-8编码，确保编辑器支持
3. **时间戳**: 每次运行测试都会生成新的日志文件，不会覆盖旧文件
4. **并行执行**: 使用 `-n auto` 参数时，日志输出可能交错，但不影响统计信息

## 🗑️ 清理旧日志

```powershell
# 删除7天前的日志
Get-ChildItem test_logs/test_statistics_*.log | 
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } | 
    Remove-Item

# 只保留最新的10个日志文件
Get-ChildItem test_logs/test_statistics_*.log | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -Skip 10 | 
    Remove-Item
```

