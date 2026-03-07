# 分析风险测试日志
# 使用方式: .\analyze_risk_test_log.ps1 [日志文件路径]
# 默认使用: test_logs/pytest_risk_latest.log

param(
    [string]$logFile = "test_logs\pytest_risk_latest.log"
)

if (-not (Test-Path $logFile)) {
    Write-Host "错误: 日志文件不存在: $logFile" -ForegroundColor Red
    exit 1
}

Write-Host "分析日志文件: $logFile" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan

# 提取测试摘要
Write-Host "`n📊 测试摘要:" -ForegroundColor Yellow
$summary = Get-Content $logFile | Select-String -Pattern "(passed|failed|skipped|error|===)" | Select-Object -Last 3
$summary

# 提取失败的测试
Write-Host "`n❌ 失败的测试:" -ForegroundColor Red
$failed = Get-Content $logFile | Select-String -Pattern "FAILED" | Select-Object -First 20
if ($failed) {
    $failed
} else {
    Write-Host "  无失败测试 ✅" -ForegroundColor Green
}

# 提取错误的测试
Write-Host "`n⚠️ 错误的测试:" -ForegroundColor Yellow
$errors = Get-Content $logFile | Select-String -Pattern "ERROR" | Select-Object -First 10
if ($errors) {
    $errors
} else {
    Write-Host "  无错误测试 ✅" -ForegroundColor Green
}

# 提取跳过的测试（按模块分类）
Write-Host "`n⏭️ 跳过的测试（按模块分类）:" -ForegroundColor Cyan
$skipped = Get-Content $logFile | Select-String -Pattern "SKIPPED"
$skippedCount = ($skipped | Measure-Object).Count
Write-Host "  总计跳过: $skippedCount 个" -ForegroundColor Yellow

Write-Host "`n  RiskManager相关跳过:" -ForegroundColor Yellow
$skipped | Select-String -Pattern "RiskManager" | Select-Object -First 10

Write-Host "`n  RiskCalculationEngine相关跳过:" -ForegroundColor Yellow
$skipped | Select-String -Pattern "RiskCalculationEngine" | Select-Object -First 10

Write-Host "`n  RealTimeRiskMonitor相关跳过:" -ForegroundColor Yellow
$skipped | Select-String -Pattern "RealTimeRiskMonitor|RealtimeRiskMonitor" | Select-Object -First 10

Write-Host "`n  AlertSystem相关跳过:" -ForegroundColor Yellow
$skipped | Select-String -Pattern "AlertSystem" | Select-Object -First 10

# 统计跳过原因
Write-Host "`n📋 跳过原因统计:" -ForegroundColor Cyan
$skipped | Select-String -Pattern "不可用|not available" | Group-Object | Sort-Object Count -Descending | Select-Object -First 10 | ForEach-Object {
    Write-Host "  $($_.Count) 个: $($_.Name)" -ForegroundColor Yellow
}

Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan

