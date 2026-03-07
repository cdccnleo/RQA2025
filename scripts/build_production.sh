#!/bin/bash
# RQA2025 生产环境构建脚本
# 使用 BuildKit 启用缓存以加速构建

set -e

echo "🚀 构建 RQA2025 生产环境镜像..."

# 构建前清理容器
echo "🧹 构建前清理容器..."
./scripts/manage_containers.sh pre-build

# 启用 Docker BuildKit 以使用缓存
export DOCKER_BUILDKIT=1

# 构建镜像
echo "🏗️  正在构建 rqa2025-app 镜像..."
docker build -t rqa2025-app:latest .

echo "✅ 镜像构建完成！"
echo "💡 提示：使用 DOCKER_BUILDKIT=1 可以启用 pip 缓存以加速后续构建"
