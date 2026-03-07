#!/bin/bash
# 备份脚本

echo "开始备份基础设施层健康管理系统..."

# 创建备份目录
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# 备份源代码
echo "备份源代码..."
cp -r src "$BACKUP_DIR/"
cp -r tests "$BACKUP_DIR/"
cp -r deployment "$BACKUP_DIR/"

# 备份配置文件
echo "备份配置文件..."
cp *.py "$BACKUP_DIR/" 2>/dev/null || true
cp requirements*.txt "$BACKUP_DIR/" 2>/dev/null || true
cp pytest.ini "$BACKUP_DIR/" 2>/dev/null || true
cp .gitignore "$BACKUP_DIR/" 2>/dev/null || true

# 备份报告
echo "备份报告..."
cp *.json "$BACKUP_DIR/" 2>/dev/null || true
cp *.md "$BACKUP_DIR/" 2>/dev/null || true

echo "备份完成: $BACKUP_DIR"
echo "备份大小: $(du -sh "$BACKUP_DIR" | cut -f1)"
