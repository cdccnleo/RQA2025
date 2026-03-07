# RQA2025 Nginx重启脚本
# 用于在配置更新后重启nginx容器

Write-Host "正在重启RQA2025 nginx容器..." -ForegroundColor Green

# 检查docker是否可用
try {
    $dockerVersion = docker --version 2>$null
    if (-not $dockerVersion) {
        Write-Host "错误: Docker未安装或不可用" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "错误: Docker未安装或不可用" -ForegroundColor Red
    exit 1
}

# 检查nginx容器是否正在运行
$nginxContainer = docker ps --filter "name=rqa2025-nginx" --format "{{.Names}}"
if ($nginxContainer) {
    Write-Host "发现运行中的nginx容器，正在重启..." -ForegroundColor Yellow
    docker restart rqa2025-nginx

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ nginx容器重启成功" -ForegroundColor Green
        Write-Host "等待nginx完全启动..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5

        # 检查nginx健康状态
        try {
            $healthResponse = Invoke-WebRequest -Uri "http://localhost/health" -TimeoutSec 10 -ErrorAction Stop
            if ($healthResponse.StatusCode -eq 200) {
                Write-Host "✅ nginx健康检查通过" -ForegroundColor Green
            } else {
                Write-Host "⚠️ nginx健康检查失败，状态码: $($healthResponse.StatusCode)" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "⚠️ nginx健康检查失败: $($_.Exception.Message)" -ForegroundColor Yellow
            Write-Host "请检查容器日志: docker logs rqa2025-nginx" -ForegroundColor Yellow
        }
    } else {
        Write-Host "❌ nginx容器重启失败" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "未发现运行中的nginx容器，请先启动完整的RQA2025系统" -ForegroundColor Yellow
    Write-Host "运行命令: docker-compose -f docker-compose.prod.yml up -d" -ForegroundColor Cyan
    exit 1
}

Write-Host "nginx重启完成" -ForegroundColor Green