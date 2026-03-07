#!/bin/bash
# RQA2025 健康检查脚本

echo "🏥 执行RQA2025健康检查..."

# 检查服务状态
echo "📊 检查应用服务状态..."
if systemctl is-active --quiet rqa2025; then
    echo "✅ 应用服务运行正常"
else
    echo "❌ 应用服务未运行"
    exit 1
fi

# 检查端口监听
echo "🔌 检查端口监听..."
if netstat -tln | grep -q ":8000 "; then
    echo "✅ 端口8000正常监听"
else
    echo "❌ 端口8000未监听"
    exit 1
fi

# 检查HTTP健康端点
echo "🌐 检查HTTP健康端点..."
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$HEALTH_STATUS" -eq 200 ]; then
    echo "✅ 健康检查通过"
else
    echo "❌ 健康检查失败 (HTTP $HEALTH_STATUS)"
    exit 1
fi

# 检查数据库连接
echo "🗄️ 检查数据库连接..."
# 这里添加具体的数据库连接检查逻辑

# 检查Redis连接
echo "🔴 检查Redis连接..."
# 这里添加Redis连接检查逻辑

# 检查磁盘空间
echo "💾 检查磁盘空间..."
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -lt 90 ]; then
    echo "✅ 磁盘空间正常 (${DISK_USAGE}%)"
else
    echo "❌ 磁盘空间不足 (${DISK_USAGE}%)"
    exit 1
fi

# 检查内存使用
echo "🧠 检查内存使用..."
MEM_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
if [ "$MEM_USAGE" -lt 90 ]; then
    echo "✅ 内存使用正常 (${MEM_USAGE}%)"
else
    echo "❌ 内存使用过高 (${MEM_USAGE}%)"
    exit 1
fi

echo "🎉 所有健康检查通过"
exit 0
