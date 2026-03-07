#!/bin/bash
# RQA2025 回滚脚本

echo "🔄 开始执行RQA2025回滚..."

# 停止服务
echo "🛑 停止当前服务..."
sudo systemctl stop rqa2025 || true

# 备份当前版本
BACKUP_DIR="/opt/rqa2025_backup_$(date +%Y%m%d_%H%M%S)"
echo "💾 备份当前版本到: $BACKUP_DIR"
sudo cp -r /opt/rqa2025 $BACKUP_DIR

# 恢复上一版本
if [ -d "/opt/rqa2025_previous" ]; then
    echo "🔄 恢复上一版本..."
    sudo rm -rf /opt/rqa2025
    sudo mv /opt/rqa2025_previous /opt/rqa2025
else
    echo "❌ 未找到上一版本，无法回滚"
    exit 1
fi

# 重启服务
echo "🚀 重启服务..."
sudo systemctl start rqa2025

# 验证服务状态
echo "🔍 验证服务状态..."
sleep 5
if sudo systemctl is-active --quiet rqa2025; then
    echo "✅ 服务回滚成功"
else
    echo "❌ 服务回滚失败，请手动检查"
    exit 1
fi

echo "🎉 回滚完成"
