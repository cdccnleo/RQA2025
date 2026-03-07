# RQA2025 Conda环境测试运行脚本
# 仅支持conda run方式，不再支持conda activate

param(
    [string]$Layer = "all",
    [switch]$InstallDeps,
    [switch]$GenerateReport
)

Write-Host "RQA2025 Conda环境测试运行器" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""

# 检查conda环境
Write-Host "[1] 检查conda环境..." -ForegroundColor Yellow
try {
    $condaInfo = conda info --envs 2>&1
    if ($condaInfo -match "test") {
        Write-Host "✓ 检测到test环境" -ForegroundColor Green
    } else {
        Write-Host "⚠️  未检测到test环境，请先创建: conda create -n test python=3.9" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "✗ Conda检查失败" -ForegroundColor Red
    exit 1
}

# 检查Python和pytest
Write-Host "[2/5] 检查Python和pytest..." -ForegroundColor Yellow
try {
    $pythonVersion = conda run -n test python --version 2>&1
    Write-Host "✓ $pythonVersion" -ForegroundColor Green
    
    $pytestVersion = conda run -n test python -m pytest --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Pytest已安装" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Pytest未安装，正在安装..." -ForegroundColor Yellow
        if ($InstallDeps) {
            conda run -n test pip install pytest pytest-cov
            Write-Host "✓ Pytest安装完成" -ForegroundColor Green
        } else {
            Write-Host "请运行: conda run -n test pip install pytest pytest-cov" -ForegroundColor Red
            exit 1
        }
    }
} catch {
    Write-Host "✗ Python检查失败" -ForegroundColor Red
    exit 1
}

# 设置环境变量
Write-Host "[3/5] 设置环境变量..." -ForegroundColor Yellow
$env:PYTHONPATH = Get-Location
$env:RQA_ENV = "conda_test"
$env:TEST_MODE = "layered"

# 创建报告目录
if (!(Test-Path "reports")) { New-Item -ItemType Directory -Path "reports" }
if (!(Test-Path "docs")) { New-Item -ItemType Directory -Path "docs" }

Write-Host "[4/5] 开始分层测试执行..." -ForegroundColor Yellow

$testLayers = @{
    "infrastructure" = @{
        test_path = "tests/unit/infrastructure/"
        src_path = "src/infrastructure"
        target_coverage = 90
    }
    "data" = @{
        test_path = "tests/unit/data/"
        src_path = "src/data"
        target_coverage = 80
    }
    "features" = @{
        test_path = "tests/unit/features/"
        src_path = "src/features"
        target_coverage = 80
    }
    "models" = @{
        test_path = "tests/unit/models/"
        src_path = "src/models"
        target_coverage = 80
    }
    "trading" = @{
        test_path = "tests/unit/trading/"
        src_path = "src/trading"
        target_coverage = 80
    }
    "backtest" = @{
        test_path = "tests/unit/backtest/"
        src_path = "src/backtest"
        target_coverage = 80
    }
}

function Run-LayerTest {
    param($layerName, $layerConfig)
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host "执行$layerName 层测试..." -ForegroundColor Cyan
    Write-Host "=========================================" -ForegroundColor Cyan
    $cmd = @(
        "conda", "run", "-n", "test", "python", "-m", "pytest",
        $layerConfig.test_path,
        "--cov=$($layerConfig.src_path)",
        "--cov-report=html:reports/$($layerName)_coverage.html",
        "--cov-report=term-missing",
        "--cov-report=json:reports/$($layerName)_coverage.json",
        "-v",
        "--tb=short"
    )
    try {
        $result = & $cmd[0] $cmd[1..($cmd.Length-1)]
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) {
            Write-Host "✓ $layerName 层测试成功" -ForegroundColor Green
        } else {
            Write-Host "✗ $layerName 层测试失败" -ForegroundColor Red
        }
        return @{
            layer = $layerName
            success = ($exitCode -eq 0)
            exit_code = $exitCode
            output = $result
        }
    } catch {
        Write-Host "✗ $layerName 层测试异常: $_" -ForegroundColor Red
        return @{
            layer = $layerName
            success = $false
            exit_code = -1
            output = $_.Exception.Message
        }
    }
}

$results = @()

if ($Layer -eq "all") {
    foreach ($layerName in $testLayers.Keys) {
        $layerConfig = $testLayers[$layerName]
        $result = Run-LayerTest -layerName $layerName -layerConfig $layerConfig
        $results += $result
    }
} else {
    if ($testLayers.ContainsKey($Layer)) {
        $layerConfig = $testLayers[$Layer]
        $result = Run-LayerTest -layerName $Layer -layerConfig $layerConfig
        $results += $result
    } else {
        Write-Host "✗ 未知的测试层: $Layer" -ForegroundColor Red
        Write-Host "可用层: $($testLayers.Keys -join ", ")" -ForegroundColor Yellow
        exit 1
    }
}

if ($GenerateReport) {
    Write-Host "生成汇总报告..." -ForegroundColor Yellow
    try {
        conda run -n test python scripts/conda_test_runner.py
        Write-Host "✓ 报告生成完成" -ForegroundColor Green
    } catch {
        Write-Host "✗ 报告生成失败" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "测试执行结果摘要" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green

$successCount = ($results | Where-Object { $_.success }).Count
$totalCount = $results.Count
foreach ($result in $results) {
    $status = if ($result.success) { "✓ 成功" } else { "✗ 失败" }
    Write-Host "$($result.layer)层: $status (退出码: $($result.exit_code))" -ForegroundColor $(if ($result.success) { "Green" } else { "Red" })
}
Write-Host "总体结果: " -ForegroundColor Cyan
Write-Host "- 总层数: $totalCount" -ForegroundColor White
Write-Host "- 成功层数: $successCount" -ForegroundColor Green
Write-Host "- 失败层数: $($totalCount - $successCount)" -ForegroundColor Red
if ($successCount -eq $totalCount) {
    Write-Host "🎉 所有层测试执行成功！" -ForegroundColor Green
} else {
    Write-Host "⚠️  部分层测试存在问题，请查看详细报告" -ForegroundColor Yellow
}
Write-Host "报告文件位置: " -ForegroundColor Cyan
Write-Host "- 覆盖率报告: reports/" -ForegroundColor White
Write-Host "- 测试报告: docs/conda_test_report.md" -ForegroundColor White
Write-Host "- 测试数据: docs/conda_test_results.json" -ForegroundColor White
Write-Host "脚本执行完成！" -ForegroundColor Green 