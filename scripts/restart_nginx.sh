#!/bin/bash
# RQA2025 Nginx重启脚本
# 用于在配置更新后重启nginx容器

echo "正在重启RQA2025 nginx容器..."

# 检查docker是否可用
if ! command -v docker &> /dev/null; then
    echo "错误: Docker未安装或不可用"
    exit 1
fi

# 检查nginx容器是否正在运行
if docker ps | grep -q rqa2025-nginx; then
    echo "发现运行中的nginx容器，正在重启..."
    docker restart rqa2025-nginx

    if [ $? -eq 0 ]; then
        echo "✅ nginx容器重启成功"
        echo "等待nginx完全启动..."
        sleep 5

        # 检查nginx健康状态
        if curl -f http://localhost/health &> /dev/null; then
            echo "✅ nginx健康检查通过"
        else
            echo "⚠️ nginx健康检查失败，请检查容器日志"
        fi
    else
        echo "❌ nginx容器重启失败"
        exit 1
    fi
else
    echo "未发现运行中的nginx容器，请先启动完整的RQA2025系统"
    echo "运行命令: docker-compose -f docker-compose.prod.yml up -d"
    exit 1
fi

echo "nginx重启完成"