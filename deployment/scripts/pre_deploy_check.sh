#!/bin/bash
# RQA2025 部署前检查脚本

echo "🔍 执行部署前检查..."

# 检查系统要求
echo "📊 检查系统资源..."
CPU_CORES=$(nproc)
MEMORY_GB=$(free -g | awk 'NR==2{printf "%.0f", $2}')
DISK_GB=$(df / | awk 'NR==2{printf "%.0f", $4/1024/1024}')

echo "CPU核心数: $CPU_CORES (需要: ≥4)"
echo "内存大小: ${MEMORY_GB}GB (需要: ≥8GB)"
echo "磁盘空间: ${DISK_GB}GB (需要: ≥50GB)"

# 检查端口占用
echo "🔌 检查端口占用..."
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "❌ 端口8000已被占用"
    exit 1
else
    echo "✅ 端口8000可用"
fi

# 检查Python版本
echo "🐍 检查Python版本..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $PYTHON_VERSION"

# 检查依赖
echo "📦 检查Python依赖..."
if [ -f "requirements.txt" ]; then
    python3 -c "
import pkg_resources
import sys

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

missing = []
for req in requirements:
    if req.strip() and not req.startswith('#'):
        try:
            pkg_resources.require(req)
        except:
            missing.append(req)

if missing:
    echo '❌ 缺少依赖包:'
    for pkg in missing:
        echo "  - $pkg"
    exit 1
else:
    echo '✅ 所有依赖包已安装'
fi
"
else
    echo "⚠️ 未找到requirements.txt文件"
fi

echo "🎉 部署前检查完成"
