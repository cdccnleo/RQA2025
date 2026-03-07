Param(
    [string]$PytestArgs = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path "test_logs")) { New-Item -ItemType Directory -Path "test_logs" | Out-Null }
if (-not (Test-Path "reports")) { New-Item -ItemType Directory -Path "reports" | Out-Null }

function Run-Pytest {
    param(
        [Parameter(Mandatory=$true)][string]$TestPath,
        [Parameter(Mandatory=$true)][string]$CovTarget,
        [Parameter(Mandatory=$true)][string]$CovXml,
        [Parameter(Mandatory=$true)][string]$LogFile
    )
    $cmd = @(
        "pytest -q $TestPath -n auto --maxfail=1 --disable-warnings",
        "--cov=$CovTarget",
        "--cov-report=term-missing",
        "--cov-report=xml:$CovXml",
        "--log-cli-level=INFO",
        "--log-file=$LogFile",
        $PytestArgs
    ) -join " "
    Write-Host "Running: $cmd"
    pwsh -NoProfile -Command $cmd
}

# 分域并行执行（串行调度避免日志竞争）
Run-Pytest "tests/unit/data/interfaces"   "src/data/interfaces"   "test_logs/coverage-interfaces-latest.xml"   "test_logs/pytest-interfaces-latest.log"
Run-Pytest "tests/unit/data/monitoring"   "src/data/monitoring"   "test_logs/coverage-monitoring-latest.xml"   "test_logs/pytest-monitoring-latest.log"
Run-Pytest "tests/unit/data/export"       "src/data/export"       "test_logs/coverage-export-latest.xml"       "test_logs/pytest-export-latest.log"
Run-Pytest "tests/unit/data/cache"        "src/data/cache"        "test_logs/coverage-cache-latest.xml"        "test_logs/pytest-cache-latest.log"
Run-Pytest "tests/unit/data/distributed"  "src/data/distributed"  "test_logs/coverage-distributed-latest.xml"  "test_logs/pytest-distributed-latest.log"
Run-Pytest "tests/unit/data/governance"   "src/data/governance"   "test_logs/coverage-governance-latest.xml"   "test_logs/pytest-governance-latest.log"
Run-Pytest "tests/unit/data/transformers" "src/data/transformers" "test_logs/coverage-transformers-latest.xml" "test_logs/pytest-transformers-latest.log"
Run-Pytest "tests/unit/data/sources"      "src/data/sources"      "test_logs/coverage-sources-latest.xml"      "test_logs/pytest-sources-latest.log"
Run-Pytest "tests/unit/data/quality"      "src/data/quality"      "test_logs/coverage-quality-latest.xml"      "test_logs/pytest-quality-latest.log"

# 打包评审包
$zipPath = "test_logs/data-layer-review-package.zip"
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Compress-Archive -Path 'test_logs/coverage-*-latest.xml','test_logs/pytest-*-latest.log' -DestinationPath $zipPath -Force
Write-Host "Review package created: $zipPath"


