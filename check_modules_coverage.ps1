Write-Host "🚀 基础设施层16个子模块测试覆盖率检查" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan
Write-Host ""

Set-Location "C:\PythonProject\RQA2025"

$modules = @("config", "distributed", "versioning", "resource", "logging", "ops", "monitoring", "health", "security", "constants", "interfaces", "optimization", "core", "utils", "cache", "error")
$totalModules = 0
$passedModules = 0

foreach ($module in $modules) {
    $totalModules++
    Write-Host ""
    Write-Host "🔍 检查模块: $module" -ForegroundColor Yellow
    Write-Host ("─" * 20) -ForegroundColor Yellow

    try {
        $output = & python -m pytest --cov="src/infrastructure/$module" --cov-report=term --cov-fail-under=0 "tests/unit/infrastructure/$module/" -q --tb=no 2>$null
        $totalLine = $output | Select-String "TOTAL" | Select-Object -Last 1

        if ($totalLine) {
            Write-Host $totalLine -ForegroundColor White
            # 提取覆盖率百分比
            $parts = $totalLine -split '\s+'
            if ($parts.Length -ge 4) {
                $coverageStr = $parts[3].Trim('%')
                try {
                    $coverage = [int]$coverageStr
                    if ($coverage -ge 80) {
                        Write-Host "✅ 达标 (80%+)" -ForegroundColor Green
                        $passedModules++
                    } else {
                        Write-Host "❌ 未达标" -ForegroundColor Red
                    }
                } catch {
                    Write-Host "❌ 无法解析覆盖率数据" -ForegroundColor Red
                }
            } else {
                Write-Host "❌ 覆盖率数据格式异常" -ForegroundColor Red
            }
        } else {
            Write-Host "❌ 无覆盖率数据或测试失败" -ForegroundColor Red
        }
    } catch {
        Write-Host "❌ 执行失败: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "📊 总结报告" -ForegroundColor Cyan
Write-Host "=" * 10 -ForegroundColor Cyan
Write-Host "检查模块总数: $totalModules" -ForegroundColor White
Write-Host "达标模块数量: $passedModules" -ForegroundColor $(if ($passedModules -ge 12) { "Green" } else { "Red" })
$notPassed = $totalModules - $passedModules
Write-Host "未达标模块数: $notPassed" -ForegroundColor $(if ($notPassed -gt 4) { "Red" } else { "Yellow" })

if ($passedModules -ge 12) {
    Write-Host ""
    Write-Host "🎉 基础设施层测试覆盖率已达标投产要求！" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "⚠️ 基础设施层测试覆盖率未达标，需要继续改进" -ForegroundColor Yellow
}

Write-Host ""
Read-Host "按任意键退出"

























