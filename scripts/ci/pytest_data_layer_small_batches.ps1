Param(
    [switch]$FailFast,
    [int]$CovFailUnder = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path "test_logs")) {
    New-Item -ItemType Directory -Path "test_logs" | Out-Null
}

function Invoke-Batch {
    param(
        [string]$Name,
        [string]$TestPath,
        [string]$CovTarget
    )
    Write-Host "Running batch: $Name" -ForegroundColor Cyan
    $junit = "test_logs/coverage-$Name-latest.junit.xml"
    $covXml = "test_logs/coverage-$Name-latest.xml"
    $logFile = "test_logs/pytest-$Name-latest.log"

    $common = @(
        "-q",
        "-n", "auto",
        "--maxfail=1",
        "--disable-warnings",
        "--cov=$CovTarget",
        "--cov-report=term-missing",
        "--cov-report=xml:$covXml",
        "--cov-branch",
        "--cov-fail-under=$CovFailUnder",
        "--junitxml=$junit",
        "--log-cli-level=INFO",
        "--log-file=$logFile"
    )

    $cmd = @("pytest", $TestPath) + $common
    Write-Host "Command: $($cmd -join ' ')" -ForegroundColor DarkGray
    if ($cmd.Length -gt 1) {
        & $cmd[0] $cmd[1..($cmd.Length-1)]
    } else {
        & $cmd[0]
    }
}

try {
    # 激活 conda 环境（如果在 CI 中未预激活）
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        conda activate rqa | Out-Null
    }
} catch {
    Write-Warning "Conda 未可用，跳过激活。"
}

# 分层小批：按模块拆分，均使用并行执行与统一输出目录
$batches = @(
    @{ Name = "validation";      TestPath = "tests/unit/data/validation";      Cov = "src/data/validation" },
    @{ Name = "transformers";    TestPath = "tests/unit/data/transformers";    Cov = "src/data/transformers" },
    @{ Name = "version_control"; TestPath = "tests/unit/data/version_control"; Cov = "src/data/version_control" },
    @{ Name = "monitoring";      TestPath = "tests/unit/data/monitoring";      Cov = "src/data/monitoring" },
    @{ Name = "distributed";     TestPath = "tests/unit/data/distributed";     Cov = "src/data/distributed" },
    @{ Name = "cache";           TestPath = "tests/unit/data/cache";           Cov = "src/data/cache" },
    @{ Name = "core";            TestPath = "tests/unit/data/core";            Cov = "src/data/core" },
    @{ Name = "sources";         TestPath = "tests/unit/data/sources";         Cov = "src/data/sources" },
    @{ Name = "preload";         TestPath = "tests/unit/data/preload";         Cov = "src/data/preload" }
)

foreach ($b in $batches) {
    try {
        Invoke-Batch -Name $b.Name -TestPath $b.TestPath -CovTarget $b.Cov
    } catch {
        Write-Error "Batch '$($b.Name)' failed: $($_.Exception.Message)"
        if ($FailFast) {
            exit 1
        }
    }
}

Write-Host "All batches executed. Coverage XMLs are available under test_logs." -ForegroundColor Green


