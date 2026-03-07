# 简化的测试运行脚本（避免语法错误）
# 使用方法: .\test_logs\run_tests_simple.ps1

# 设置工作目录
Set-Location $PSScriptRoot\..

# 生成时间戳
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = "test_logs\test_statistics_$timestamp.log"

Write-Host "开始运行测试..." -ForegroundColor Green
Write-Host "日志将保存到: $logFile" -ForegroundColor Cyan

# 运行测试并保存日志
pytest tests/unit/features/ -n auto -k "not e2e" --tb=line -q --cov=src.features --cov-report=term-missing 2>&1 | Out-File -FilePath $logFile -Encoding UTF8

# 检查文件是否创建
if (Test-Path $logFile) {
    # 显示最后几行结果
    Write-Host "`n测试结果摘要:" -ForegroundColor Green
    Get-Content $logFile | Select-Object -Last 5 | ForEach-Object { Write-Host $_ }
    
    Write-Host "`n完整日志: $logFile" -ForegroundColor Cyan
} else {
    Write-Host "警告: 日志文件未创建" -ForegroundColor Yellow
}


