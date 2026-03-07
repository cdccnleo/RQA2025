# 时间同步脚本 - Windows PowerShell
# 检查并同步系统时间

Write-Host "=== RQA2025 时间同步检查工具 ===" -ForegroundColor Cyan
Write-Host ""

# 检查当前时间
Write-Host "当前系统时间:" -ForegroundColor Yellow
Get-Date
Write-Host ""

# 检查时间同步状态
Write-Host "检查时间同步服务状态:" -ForegroundColor Yellow
try {
    $timeStatus = w32tm /query /status
    Write-Host $timeStatus
} catch {
    Write-Host "无法查询时间同步状态" -ForegroundColor Red
}
Write-Host ""

# 同步时间
Write-Host "正在同步系统时间..." -ForegroundColor Yellow
try {
    w32tm /resync /force
    Write-Host "时间同步完成!" -ForegroundColor Green
} catch {
    Write-Host "时间同步失败，请手动检查网络连接" -ForegroundColor Red
}
Write-Host ""

# 再次显示时间
Write-Host "同步后的系统时间:" -ForegroundColor Yellow
Get-Date
Write-Host ""

# 检查Docker容器时间（如果运行中）
Write-Host "检查Docker容器状态:" -ForegroundColor Yellow
try {
    $containers = docker ps --format "table {{.Names}}\t{{.Status}}"
    if ($containers) {
        Write-Host "运行中的容器:" -ForegroundColor Green
        Write-Host $containers
        Write-Host ""

        # 检查容器时间
        Write-Host "检查容器时间..." -ForegroundColor Yellow
        $containerNames = docker ps --format "{{.Names}}"
        foreach ($name in $containerNames) {
            Write-Host "容器 $name 的时间:" -ForegroundColor Cyan
            try {
                docker exec $name date 2>$null
            } catch {
                Write-Host "  无法获取容器时间" -ForegroundColor Red
            }
        }
    } else {
        Write-Host "没有运行中的Docker容器" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "启动容器命令:" -ForegroundColor Cyan
        Write-Host "  docker-compose up -d"
    }
} catch {
    Write-Host "Docker未运行或未安装" -ForegroundColor Red
    Write-Host "请确保Docker Desktop正在运行" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== 时间同步检查完成 ===" -ForegroundColor Cyan