# 流处理层测试运行脚本
# 自动保存测试日志到 test_logs 目录

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "test_logs\streaming_test_results_$timestamp.log"
$summaryFile = "test_logs\streaming_test_summary_$timestamp.txt"

Write-Host "开始运行流处理层测试..." -ForegroundColor Green
Write-Host "日志文件: $logFile" -ForegroundColor Yellow
Write-Host "摘要文件: $summaryFile" -ForegroundColor Yellow

# 确保日志目录存在
if (-not (Test-Path "test_logs")) {
    New-Item -ItemType Directory -Path "test_logs" | Out-Null
}

# 运行测试并保存完整日志
pytest tests/unit/streaming --cov=src.streaming --cov-report=term-missing -k "not e2e" --tb=no -q 2>&1 | Tee-Object -FilePath $logFile

# 提取关键信息到摘要文件
$summary = @"
流处理层测试运行摘要
====================
运行时间: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
日志文件: $logFile

测试结果:
"@

# 从日志中提取关键信息
$testResults = Get-Content $logFile | Select-String -Pattern "TOTAL|failed|passed|ERROR|WARNING" | Select-Object -Last 10
$summary += "`n" + ($testResults -join "`n")

# 提取覆盖率信息
$coverageInfo = Get-Content $logFile | Select-String -Pattern "TOTAL|Name.*Stmts.*Miss.*Cover" | Select-Object -Last 5
if ($coverageInfo) {
    $summary += "`n`n覆盖率信息:`n"
    $summary += ($coverageInfo -join "`n")
}

# 保存摘要
$summary | Out-File -FilePath $summaryFile -Encoding UTF8

Write-Host "`n测试完成！" -ForegroundColor Green
Write-Host "完整日志: $logFile" -ForegroundColor Cyan
Write-Host "测试摘要: $summaryFile" -ForegroundColor Cyan

# 显示最后的关键信息
Write-Host "`n最后的关键信息:" -ForegroundColor Yellow
Get-Content $logFile | Select-String -Pattern "TOTAL|failed|passed" | Select-Object -Last 3


