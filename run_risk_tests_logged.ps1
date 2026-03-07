#!/usr/bin/env pwsh
# -*- coding: utf-8 -*-
"""
运行风险测试并保存日志
使用方式: .\run_risk_tests_logged.ps1
"""

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = "test_logs\pytest_risk_$timestamp.log"
$latestLogFile = "test_logs\pytest_risk_latest.log"
$summaryFile = "test_logs\pytest_risk_summary_$timestamp.txt"

Write-Host "开始运行风险测试，日志将保存到: $logFile" -ForegroundColor Green

# 运行测试并保存完整日志（使用*>重定向所有输出）
conda run -n rqa pytest tests/unit/risk -v --tb=line *> $logFile

# 同时保存到latest.log
Copy-Item $logFile $latestLogFile -Force

# 提取摘要信息
Write-Host "`n提取测试摘要..." -ForegroundColor Yellow
$summary = Get-Content $logFile | Select-String -Pattern "(passed|failed|skipped|error|===)" | Select-Object -Last 5
$summary | Out-File -FilePath $summaryFile -Encoding UTF8

Write-Host "`n测试完成！" -ForegroundColor Green
Write-Host "完整日志: $logFile" -ForegroundColor Cyan
Write-Host "最新日志: $latestLogFile" -ForegroundColor Cyan
Write-Host "摘要文件: $summaryFile" -ForegroundColor Cyan
Write-Host "`n最后5行摘要:" -ForegroundColor Yellow
$summary

