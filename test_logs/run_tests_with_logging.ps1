# 运行测试并保存日志到文件
# 使用方法: .\test_logs\run_tests_with_logging.ps1

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = "test_logs\test_statistics_$timestamp.log"

Write-Host "开始运行测试，日志将保存到: $logFile" -ForegroundColor Green

# 运行测试并将输出保存到文件
# 使用 Out-File 而不是重定向，这样更可靠
pytest tests/unit/features/ -n auto -k "not e2e" --tb=line -q --cov=src.features --cov-report=term-missing 2>&1 | Out-File -FilePath $logFile -Encoding UTF8

# 检查测试结果
if (Test-Path $logFile) {
    $testResult = Get-Content $logFile | Select-Object -Last 5
    
    Write-Host "`n测试完成！结果摘要:" -ForegroundColor Green
    $testResult | ForEach-Object { Write-Host $_ }
    
    # 显示日志文件位置
    Write-Host "`n完整日志已保存到: $logFile" -ForegroundColor Cyan
    
    # 询问是否打开日志文件
    $open = Read-Host "`n是否打开日志文件? (y/n)"
    if ($open -eq 'y' -or $open -eq 'Y') {
        notepad $logFile
    }
} else {
    Write-Host "错误：日志文件未创建！" -ForegroundColor Red
}

