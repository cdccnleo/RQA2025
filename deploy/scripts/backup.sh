#!/bin/bash
# RQA2025 自动备份脚本
# 支持数据库、配置、模型等定时备份

set -e

BACKUP_ROOT="/backup/rqa2025"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/$DATE"

mkdir -p "$BACKUP_DIR"

# 备份数据库（PostgreSQL）
echo "备份数据库..."
sudo -u postgres pg_dump rqa2025 > "$BACKUP_DIR/db_rqa2025.sql"

# 备份配置文件
echo "备份配置文件..."
cp -r /etc/rqa2025 "$BACKUP_DIR/config"

# 备份模型文件
echo "备份模型文件..."
cp -r /var/lib/rqa2025/models "$BACKUP_DIR/models"

# 备份日志（可选）
echo "备份日志..."
cp -r /var/log/rqa2025 "$BACKUP_DIR/logs"

echo "备份完成，存储于: $BACKUP_DIR" 