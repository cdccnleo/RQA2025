# 快速测试命令参考

## 🚀 运行测试并保存日志

### 方法1: 使用PowerShell脚本（推荐）
```powershell
.\test_logs\run_tests_with_logging.ps1
```

### 方法2: 直接使用命令
```powershell
# 生成带时间戳的测试日志（使用Out-File，更可靠）
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = "test_logs/test_statistics_$timestamp.log"
pytest tests/unit/features/ -n auto -k "not e2e" --tb=line -q --cov=src.features --cov-report=term-missing 2>&1 | Out-File -FilePath $logFile -Encoding UTF8
Write-Host "测试日志已保存到: $logFile"
```

### 方法3: 简化版本（不包含覆盖率）
```powershell
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = "test_logs/test_statistics_$timestamp.log"
pytest tests/unit/features/ -n auto -k "not e2e" --tb=line -q 2>&1 | Out-File -FilePath $logFile -Encoding UTF8
Write-Host "测试日志已保存到: $logFile"
```

## 📊 查看日志

### 查看最新日志的最后几行
```powershell
Get-Content test_logs/test_statistics_*.log | Select-Object -Last 20
```

### 查看最新日志文件
```powershell
Get-ChildItem test_logs/test_statistics_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | Get-Content
```

### 查找失败的测试
```powershell
Select-String -Path test_logs/test_statistics_*.log -Pattern "FAILED"
```

### 查找覆盖率信息
```powershell
Select-String -Path test_logs/test_statistics_*.log -Pattern "TOTAL"
```

## 🔍 检索低覆盖率文件

### 查找覆盖率低于50%的文件
```powershell
Get-ChildItem test_logs/test_statistics_*.log | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 1 | 
    ForEach-Object {
        Get-Content $_.FullName | 
            Select-String -Pattern "^\s+\S+\s+\d+\s+\d+\s+\d+%" | 
            Where-Object { 
                if ($_.Line -match "(\d+)%$") {
                    [int]$matches[1] -lt 50
                }
            }
    }
```

## 📝 快速运行（不保存日志）

```powershell
pytest tests/unit/features/ -n auto -k "not e2e" --tb=line -q
```

## 🗑️ 清理旧日志

```powershell
# 删除7天前的日志
Get-ChildItem test_logs/test_statistics_*.log | 
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } | 
    Remove-Item
```

