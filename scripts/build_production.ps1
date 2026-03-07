# RQA2025 生产环境构建脚本
# 使用 BuildKit 启用缓存以加速构建

param(
    [switch]$NoCache
)

Write-Host "🚀 构建 RQA2025 生产环境镜像..." -ForegroundColor Green

# 启用 Docker BuildKit 以使用缓存
$env:DOCKER_BUILDKIT = "1"

# 构建参数
$buildArgs = @()

if ($NoCache) {
    $buildArgs += "--no-cache"
}

# 构建镜像
$buildCommand = "docker build -t rqa2025-app:latest"
if ($buildArgs.Count -gt 0) {
    $buildCommand += " " + ($buildArgs -join " ")
}

Write-Host "执行命令: $buildCommand" -ForegroundColor Yellow
Invoke-Expression $buildCommand

Write-Host "✅ 镜像构建完成！" -ForegroundColor Green
Write-Host "💡 提示：使用 DOCKER_BUILDKIT=1 可以启用 pip 缓存以加速后续构建" -ForegroundColor Cyan
