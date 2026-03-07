#!/bin/bash
# RQA2025 Kubernetes生产环境回滚脚本

set -e

echo "🔄 RQA2025 Kubernetes生产环境回滚开始"
echo "=========================================="

# 确认回滚操作
read -p "⚠️ 确定要回滚到上一个版本吗? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 回滚操作已取消"
    exit 1
fi

# 回滚应用部署
echo "🔄 回滚应用部署..."
kubectl rollout undo deployment/rqa2025-app -n rqa2025-app

# 等待回滚完成
echo "⏳ 等待回滚完成..."
kubectl rollout status deployment/rqa2025-app -n rqa2025-app

# 验证回滚结果
echo "✅ 验证回滚结果..."
kubectl get pods -n rqa2025-app
kubectl logs -n rqa2025-app deployment/rqa2025-app --tail=10

echo "✅ RQA2025 Kubernetes生产环境回滚完成！"
echo "=========================================="




