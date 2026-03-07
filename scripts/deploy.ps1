# RQA2025 PowerShell部署脚本
# Windows环境下的部署脚本

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("development", "staging", "production")]
    [string]$Environment = "development"
)

# 日志函数
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# 检查依赖
function Test-Dependencies {
    Write-Info "检查系统依赖..."

    if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker 未安装，请先安装 Docker"
        exit 1
    }

    if (!(Get-Command docker-compose -ErrorAction SilentlyContinue)) {
        Write-Error "Docker Compose 未安装，请先安装 Docker Compose"
        exit 1
    }

    Write-Success "系统依赖检查通过"
}

# 质量检查
function Test-Quality {
    Write-Info "运行质量检查..."

    try {
        & python scripts/quality_monitor_simple.py
        if ($LASTEXITCODE -ne 0) {
            Write-Error "质量检查失败，不允许部署"
            exit 1
        }
    }
    catch {
        Write-Error "质量检查执行失败: $($_.Exception.Message)"
        exit 1
    }

    Write-Success "质量检查通过"
}

# 构建镜像
function Build-Images {
    Write-Info "构建 Docker 镜像..."

    docker-compose build --no-cache

    if ($LASTEXITCODE -ne 0) {
        Write-Error "镜像构建失败"
        exit 1
    }

    Write-Success "镜像构建完成"
}

# 部署服务
function Deploy-Services {
    param([string]$Environment)

    Write-Info "部署到 $Environment 环境..."

    switch ($Environment) {
        "development" {
            docker-compose --profile testing up -d
        }
        "staging" {
            docker-compose --profile monitoring up -d redis postgres
        }
        "production" {
            docker-compose up -d
        }
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Error "服务部署失败"
        exit 1
    }

    Write-Success "服务部署完成"
}

# 健康检查
function Test-Health {
    Write-Info "执行健康检查..."

    # 等待服务启动
    Start-Sleep -Seconds 30

    # 检查容器状态
    $containers = docker-compose ps
    if ($containers -match "Up") {
        Write-Success "容器状态检查通过"
    }
    else {
        Write-Error "容器状态检查失败"
        exit 1
    }

    # 尝试连接主服务
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 10 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Success "主服务健康检查通过"
        }
    }
    catch {
        Write-Warning "主服务健康检查失败，服务可能仍在启动中"
    }
}

# 显示部署信息
function Show-DeploymentInfo {
    Write-Info "部署信息:"

    Write-Host ""
    Write-Host "服务状态:" -ForegroundColor Cyan
    docker-compose ps

    Write-Host ""
    Write-Host "服务端口映射:" -ForegroundColor Cyan
    Write-Host "  RQA2025 App: http://localhost:8000"
    Write-Host "  API Gateway: http://localhost:8080"
    Write-Host "  Redis: localhost:6379"
    Write-Host "  PostgreSQL: localhost:5432"

    Write-Host ""
    Write-Host "常用命令:" -ForegroundColor Cyan
    Write-Host "  查看日志: docker-compose logs -f rqa2025-app"
    Write-Host "  停止服务: docker-compose down"
    Write-Host "  重启服务: docker-compose restart"
}

# 清理函数
function Clear-Deployment {
    Write-Info "清理部署环境..."
    docker-compose down -v 2>$null
    Write-Success "清理完成"
}

# 主函数
function Main {
    Write-Info "RQA2025 PowerShell部署脚本启动"
    Write-Info "目标环境: $Environment"
    Write-Host ""

    # 设置错误处理
    $ErrorActionPreference = "Stop"

    try {
        # 执行部署步骤
        Test-Dependencies
        Test-Quality
        Build-Images
        Deploy-Services -Environment $Environment
        Test-Health
        Show-DeploymentInfo

        Write-Success "RQA2025 部署成功! 🎉"
        Write-Info "使用 'docker-compose logs -f' 查看日志"
        Write-Info "使用 'docker-compose down' 停止服务"
    }
    catch {
        Write-Error "部署失败: $($_.Exception.Message)"
        exit 1
    }
    finally {
        # 注册清理函数
        Register-EngineEvent PowerShell.Exiting -Action {
            Clear-Deployment
        } | Out-Null
    }
}

# 执行主函数
Main
