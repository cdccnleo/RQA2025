# 生产环境备份和恢复计划

## 概述

本文档详细说明RQA2025量化交易系统的备份策略、恢复流程和灾难恢复计划，确保系统数据的安全性和业务连续性。

## 备份策略

### 1. 备份分类

#### 1.1 数据备份

##### 数据库备份
- **备份类型**: 完整备份 + 增量备份
- **备份频率**:
  - 完整备份: 每周日 02:00
  - 增量备份: 每日 02:00
- **保留策略**:
  - 完整备份: 保留30天
  - 增量备份: 保留7天
- **备份位置**: 本地 + 云存储
- **加密**: AES256加密

##### 配置文件备份
- **备份类型**: 完整备份
- **备份频率**: 每次配置变更后
- **保留策略**: 保留所有版本，90天后归档
- **备份位置**: Git仓库 + 本地

#### 1.2 应用备份

##### 应用代码备份
- **备份类型**: Git标签 + 发布包
- **备份频率**: 每次发布后
- **保留策略**: 保留所有发布版本
- **备份位置**: Git仓库 + 制品库

##### 依赖包备份
- **备份类型**: Python包 + 系统包
- **备份频率**: 每月更新
- **保留策略**: 保留3个月
- **备份位置**: 本地私有仓库

#### 1.3 系统备份

##### 操作系统备份
- **备份类型**: 系统镜像
- **备份频率**: 每月一次
- **保留策略**: 保留6个月
- **备份位置**: 云存储

##### 容器镜像备份
- **备份类型**: Docker镜像
- **备份频率**: 每次构建后
- **保留策略**: 保留30天
- **备份位置**: 私有镜像仓库

### 2. 备份实施

#### 2.1 数据库备份脚本

```bash
#!/bin/bash
# database_backup.sh

# 备份配置
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="rqa2025"
DB_USER="rqa2025"
BACKUP_DIR="/data/backup/database"
RETENTION_DAYS=30

# 创建备份目录
mkdir -p $BACKUP_DIR

# 生成备份文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/rqa2025_full_backup_$TIMESTAMP.sql.gz"

# 执行完整备份
pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME \
        --no-password --compress=9 --format=custom \
        --verbose --blobs --clean --if-exists \
        --exclude-table-data='*_temp_*' \
        | gzip > $BACKUP_FILE

# 验证备份文件
if [ $? -eq 0 ] && [ -f $BACKUP_FILE ]; then
    echo "数据库备份成功: $BACKUP_FILE"

    # 计算文件大小
    FILE_SIZE=$(du -h $BACKUP_FILE | cut -f1)
    echo "备份文件大小: $FILE_SIZE"

    # 上传到云存储
    aws s3 cp $BACKUP_FILE s3://rqa2025-backup/database/

    # 清理本地旧备份
    find $BACKUP_DIR -name "*.sql.gz" -type f -mtime +$RETENTION_DAYS -delete

    echo "备份完成并上传到云存储"
else
    echo "数据库备份失败"
    exit 1
fi
```

#### 2.2 Redis备份脚本

```bash
#!/bin/bash
# redis_backup.sh

# 备份配置
REDIS_HOST="localhost"
REDIS_PORT="6379"
BACKUP_DIR="/data/backup/redis"
RETENTION_DAYS=7

# 创建备份目录
mkdir -p $BACKUP_DIR

# 生成备份文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/redis_backup_$TIMESTAMP.rdb"

# 执行Redis备份
redis-cli -h $REDIS_HOST -p $REDIS_PORT --rdb $BACKUP_FILE

# 验证备份文件
if [ $? -eq 0 ] && [ -f $BACKUP_FILE ]; then
    echo "Redis备份成功: $BACKUP_FILE"

    # 压缩备份文件
    gzip $BACKUP_FILE
    COMPRESSED_FILE="$BACKUP_FILE.gz"

    # 上传到云存储
    aws s3 cp $COMPRESSED_FILE s3://rqa2025-backup/redis/

    # 清理本地旧备份
    find $BACKUP_DIR -name "*.rdb.gz" -type f -mtime +$RETENTION_DAYS -delete

    echo "Redis备份完成并上传到云存储"
else
    echo "Redis备份失败"
    exit 1
fi
```

#### 2.3 配置文件备份脚本

```bash
#!/bin/bash
# config_backup.sh

# 备份配置
CONFIG_DIRS=(
    "/etc/rqa2025"
    "/etc/nginx/sites-available"
    "/etc/systemd/system"
    "/opt/rqa2025/config"
)
BACKUP_DIR="/data/backup/config"
RETENTION_DAYS=90

# 创建备份目录
mkdir -p $BACKUP_DIR

# 生成备份文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/config_backup_$TIMESTAMP.tar.gz"

# 创建配置文件归档
tar -czf $BACKUP_FILE \
    --exclude='*.log' \
    --exclude='*.tmp' \
    --exclude='*.cache' \
    "${CONFIG_DIRS[@]}"

# 验证备份文件
if [ $? -eq 0 ] && [ -f $BACKUP_FILE ]; then
    echo "配置文件备份成功: $BACKUP_FILE"

    # 计算文件大小
    FILE_SIZE=$(du -h $BACKUP_FILE | cut -f1)
    echo "备份文件大小: $FILE_SIZE"

    # 上传到云存储
    aws s3 cp $BACKUP_FILE s3://rqa2025-backup/config/

    # 清理本地旧备份
    find $BACKUP_DIR -name "*.tar.gz" -type f -mtime +$RETENTION_DAYS -delete

    echo "配置文件备份完成"
else
    echo "配置文件备份失败"
    exit 1
fi
```

## 恢复流程

### 1. 数据库恢复

#### 1.1 完整恢复流程

```bash
#!/bin/bash
# database_restore.sh

# 恢复配置
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="rqa2025"
DB_USER="rqa2025"
BACKUP_FILE="$1"  # 备份文件路径

# 检查备份文件是否存在
if [ ! -f "$BACKUP_FILE" ]; then
    echo "备份文件不存在: $BACKUP_FILE"
    exit 1
fi

# 确认恢复操作
echo "警告: 此操作将覆盖现有数据库！"
read -p "确认要恢复数据库吗? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "操作已取消"
    exit 0
fi

# 停止应用服务
echo "停止应用服务..."
sudo systemctl stop rqa2025

# 创建恢复前备份
echo "创建当前数据库备份..."
pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME \
        --no-password --compress=9 --format=custom \
        --verbose --blobs --clean \
        > pre_restore_backup.sql

# 恢复数据库
echo "开始恢复数据库..."
pg_restore -h $DB_HOST -p $DB_PORT -U $DB_USER \
           -d $DB_NAME --no-password --clean --if-exists \
           --verbose $BACKUP_FILE

# 验证恢复结果
if [ $? -eq 0 ]; then
    echo "数据库恢复成功"

    # 执行数据一致性检查
    psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "
    -- 检查表是否存在
    SELECT schemaname, tablename FROM pg_tables WHERE schemaname = 'public';

    -- 检查数据完整性
    SELECT COUNT(*) FROM orders;
    SELECT COUNT(*) FROM users;
    "

    # 重新启动应用服务
    echo "重新启动应用服务..."
    sudo systemctl start rqa2025

    echo "数据库恢复完成"
else
    echo "数据库恢复失败"
    exit 1
fi
```

#### 1.2 增量恢复流程

```bash
#!/bin/bash
# incremental_restore.sh

# 使用WAL归档进行时间点恢复
RESTORE_POINT="$1"  # 恢复时间点
BACKUP_FILE="$2"    # 基础备份文件

# 停止PostgreSQL服务
sudo systemctl stop postgresql

# 清理数据目录
sudo rm -rf /var/lib/postgresql/13/main/*

# 恢复基础备份
pg_restore -h localhost -p 5432 -U postgres \
           -d postgres --clean --if-exists \
           --verbose $BACKUP_FILE

# 恢复WAL日志到指定时间点
sudo -u postgres pg_waldump --start-stop $RESTORE_POINT /var/lib/postgresql/13/main/pg_wal/*

# 启动PostgreSQL服务
sudo systemctl start postgresql

# 验证恢复
psql -h localhost -U rqa2025 -d rqa2025 -c "
SELECT NOW();
SELECT COUNT(*) FROM orders WHERE created_at <= '$RESTORE_POINT';
"
```

### 2. Redis恢复

```bash
#!/bin/bash
# redis_restore.sh

# 恢复配置
REDIS_HOST="localhost"
REDIS_PORT="6379"
BACKUP_FILE="$1"  # 备份文件路径

# 检查备份文件
if [ ! -f "$BACKUP_FILE" ]; then
    echo "备份文件不存在: $BACKUP_FILE"
    exit 1
fi

# 停止Redis服务
sudo systemctl stop redis

# 备份当前数据
cp /var/lib/redis/dump.rdb /var/lib/redis/dump.rdb.backup

# 恢复备份文件
cp $BACKUP_FILE /var/lib/redis/dump.rdb

# 设置正确的权限
chown redis:redis /var/lib/redis/dump.rdb
chmod 660 /var/lib/redis/dump.rdb

# 启动Redis服务
sudo systemctl start redis

# 验证恢复
redis-cli ping
redis-cli dbsize

echo "Redis恢复完成"
```

### 3. 应用恢复

#### 3.1 代码回滚

```bash
#!/bin/bash
# application_rollback.sh

# 回滚配置
TARGET_VERSION="$1"  # 目标版本
DEPLOY_DIR="/opt/rqa2025"
BACKUP_DIR="/data/backup/application"

# 检查目标版本是否存在
if [ ! -f "$BACKUP_DIR/rqa2025_$TARGET_VERSION.tar.gz" ]; then
    echo "目标版本不存在: $TARGET_VERSION"
    exit 1
fi

# 停止应用服务
sudo systemctl stop rqa2025

# 创建当前版本备份
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
tar -czf $BACKUP_DIR/rqa2025_pre_rollback_$TIMESTAMP.tar.gz -C $DEPLOY_DIR .

# 恢复目标版本
tar -xzf $BACKUP_DIR/rqa2025_$TARGET_VERSION.tar.gz -C $DEPLOY_DIR

# 恢复配置文件
cp $BACKUP_DIR/config_$TARGET_VERSION.tar.gz /tmp/
tar -xzf /tmp/config_$TARGET_VERSION.tar.gz -C /etc/rqa2025

# 重新启动应用服务
sudo systemctl start rqa2025

# 验证回滚结果
curl http://localhost:8000/health
curl http://localhost:8000/version

echo "应用回滚完成"
```

#### 3.2 配置恢复

```bash
#!/bin/bash
# config_restore.sh

# 恢复配置
BACKUP_FILE="$1"
RESTORE_DIR="/etc/rqa2025"

# 创建当前配置备份
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
tar -czf /data/backup/config_pre_restore_$TIMESTAMP.tar.gz -C $RESTORE_DIR .

# 恢复配置文件
tar -xzf $BACKUP_FILE -C $RESTORE_DIR

# 验证配置文件
# 检查必需的配置文件是否存在
REQUIRED_FILES=(
    "config.yaml"
    "database.yaml"
    "redis.yaml"
    "logging.yaml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$RESTORE_DIR/$file" ]; then
        echo "缺少必需的配置文件: $file"
        exit 1
    fi
done

# 重新加载服务配置
sudo systemctl reload rqa2025

echo "配置文件恢复完成"
```

## 灾难恢复计划

### 1. 灾难恢复策略

#### 1.1 灾难分级

| 级别 | 描述 | RTO | RPO | 恢复策略 |
|------|------|-----|-----|----------|
| P1 | 完全服务中断 | 1小时 | 1分钟 | 多区域切换 |
| P2 | 主要功能失效 | 4小时 | 5分钟 | 部分服务切换 |
| P3 | 次要功能失效 | 8小时 | 15分钟 | 本地修复 |
| P4 | 性能问题 | 24小时 | 1小时 | 性能优化 |

#### 1.2 灾难恢复流程

##### 阶段1: 灾难确认
```bash
# 1. 确认灾难范围和影响
echo "=== 灾难确认阶段 ==="
echo "1. 检查服务状态"
sudo systemctl status rqa2025

echo "2. 检查系统资源"
top -b -n1 | head -5

echo "3. 检查网络连接"
ping -c 3 google.com

echo "4. 检查数据库连接"
pg_isready -h localhost -p 5432

echo "5. 确认影响范围"
# 检查受影响的用户和服务
```

##### 阶段2: 灾难隔离
```bash
# 1. 隔离受损服务
echo "=== 灾难隔离阶段 ==="
echo "1. 停止受损服务"
sudo systemctl stop rqa2025

echo "2. 断开受损数据库连接"
# 终止数据库连接
psql -h localhost -U rqa2025 -d rqa2025 -c "
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'active' AND pid <> pg_backend_pid();
"

echo "3. 隔离受损缓存"
redis-cli flushall

echo "4. 切换到备用系统（如果有）"
# 切换到灾备环境
```

##### 阶段3: 数据恢复
```bash
# 1. 确定恢复点
echo "=== 数据恢复阶段 ==="
LATEST_BACKUP=$(ls -t /data/backup/database/*.sql.gz | head -1)
echo "使用最新备份: $LATEST_BACKUP"

# 2. 执行数据恢复
./database_restore.sh $LATEST_BACKUP

# 3. 验证数据完整性
psql -h localhost -U rqa2025 -d rqa2025 -c "
-- 检查关键表的数据量
SELECT 'orders' as table_name, COUNT(*) as count FROM orders
UNION ALL
SELECT 'users', COUNT(*) FROM users
UNION ALL
SELECT 'trades', COUNT(*) FROM trades;
"

# 4. 重新同步增量数据（如果需要）
# 根据业务需求，从其他数据源同步增量数据
```

##### 阶段4: 服务恢复
```bash
# 1. 启动恢复后的服务
echo "=== 服务恢复阶段 ==="

# 启动数据库相关服务
sudo systemctl start postgresql
sudo systemctl start redis

# 启动应用服务
sudo systemctl start rqa2025

# 等待服务完全启动
sleep 30

# 2. 验证服务健康状态
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health

# 3. 验证关键功能
curl http://localhost:8000/api/orders
curl http://localhost:8000/api/market

# 4. 恢复监控和告警
sudo systemctl restart prometheus
sudo systemctl restart grafana
sudo systemctl restart alertmanager
```

##### 阶段5: 业务验证
```bash
# 1. 业务功能验证
echo "=== 业务验证阶段 ==="

# 验证用户登录
curl -X POST http://localhost:8000/api/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username": "test", "password": "test"}'

# 验证交易功能
curl -X POST http://localhost:8000/api/trading/execute \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <token>" \
     -d '{"symbol": "AAPL", "quantity": 100, "price": 150.0}'

# 验证查询功能
curl http://localhost:8000/api/portfolio \
     -H "Authorization: Bearer <token>"

# 2. 性能验证
# 检查响应时间
# 检查系统资源使用
# 检查并发处理能力
```

### 2. 灾难恢复演练

#### 2.1 定期演练计划

| 演练类型 | 频率 | 范围 | 参与人员 |
|----------|------|------|----------|
| 数据库恢复 | 每季度 | 完整恢复流程 | DBA + 运维 |
| 应用回滚 | 每季度 | 代码回滚流程 | 开发 + 运维 |
| 完整灾难恢复 | 每半年 | 端到端恢复 | 全技术团队 |
| 网络故障 | 每季度 | 网络切换 | 网络 + 运维 |

#### 2.2 演练执行步骤

```bash
#!/bin/bash
# disaster_recovery_drill.sh

# 演练配置
DRILL_TYPE="$1"  # database, application, full
DRILL_DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/var/log/drill/drill_$DRILL_DATE.log"

# 创建演练日志
exec > >(tee -a $LOG_FILE) 2>&1

echo "=== 灾难恢复演练开始 ==="
echo "演练类型: $DRILL_TYPE"
echo "开始时间: $(date)"

case $DRILL_TYPE in
    "database")
        echo "执行数据库恢复演练"
        # 1. 备份当前数据库
        ./database_backup.sh

        # 2. 模拟数据库故障
        sudo systemctl stop postgresql
        rm -rf /var/lib/postgresql/13/main/*

        # 3. 执行恢复
        LATEST_BACKUP=$(ls -t /data/backup/database/*.sql.gz | head -1)
        ./database_restore.sh $LATEST_BACKUP

        # 4. 验证恢复结果
        pg_isready -h localhost -p 5432
        ;;

    "application")
        echo "执行应用回滚演练"
        # 1. 模拟应用故障
        sudo systemctl stop rqa2025

        # 2. 执行回滚
        PREVIOUS_VERSION=$(git tag --sort=-version:refname | head -2 | tail -1)
        ./application_rollback.sh $PREVIOUS_VERSION

        # 3. 验证回滚结果
        curl http://localhost:8000/health
        curl http://localhost:8000/version
        ;;

    "full")
        echo "执行完整灾难恢复演练"
        # 执行完整的灾难恢复流程
        # 包括数据库恢复、应用恢复、配置恢复等
        ;;
esac

echo "演练完成时间: $(date)"
echo "=== 灾难恢复演练结束 ==="
```

## 备份监控和报告

### 1. 备份监控

#### 1.1 备份状态监控

```bash
#!/bin/bash
# backup_monitor.sh

# 检查备份状态
BACKUP_DIRS=(
    "/data/backup/database"
    "/data/backup/redis"
    "/data/backup/config"
)

for dir in "${BACKUP_DIRS[@]}"; do
    echo "检查备份目录: $dir"

    # 检查最新备份文件
    LATEST_BACKUP=$(ls -t $dir/* 2>/dev/null | head -1)
    if [ -n "$LATEST_BACKUP" ]; then
        BACKUP_AGE=$(( $(date +%s) - $(stat -c %Y "$LATEST_BACKUP") ))
        BACKUP_HOURS=$(( BACKUP_AGE / 3600 ))

        echo "  最新备份: $LATEST_BACKUP"
        echo "  备份时间: $BACKUP_HOURS 小时前"

        # 检查备份是否过旧
        if [ $BACKUP_HOURS -gt 25 ]; then
            echo "  ⚠️  警告: 备份文件过旧!"
        fi
    else
        echo "  ❌ 错误: 没有找到备份文件!"
    fi

    echo ""
done

# 检查备份文件大小
echo "备份文件大小统计:"
for dir in "${BACKUP_DIRS[@]}"; do
    echo "$dir: $(du -sh $dir 2>/dev/null | cut -f1 || echo 'N/A')"
done
```

#### 1.2 备份完整性检查

```bash
#!/bin/bash
# backup_integrity_check.sh

# 检查备份文件完整性
BACKUP_FILE="$1"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "备份文件不存在: $BACKUP_FILE"
    exit 1
fi

# 检查文件大小
FILE_SIZE=$(stat -c %s "$BACKUP_FILE")
if [ $FILE_SIZE -lt 1024 ]; then
    echo "警告: 备份文件过小 ($FILE_SIZE bytes)"
fi

# 对于压缩文件，检查解压是否正常
if [[ "$BACKUP_FILE" == *.gz ]]; then
    gunzip -t "$BACKUP_FILE"
    if [ $? -ne 0 ]; then
        echo "错误: 备份文件损坏，无法解压"
        exit 1
    fi
fi

# 对于数据库备份，检查基本结构
if [[ "$BACKUP_FILE" == *database* ]]; then
    # 解压并检查SQL文件
    zcat "$BACKUP_FILE" | head -20 | grep -q "PostgreSQL"
    if [ $? -ne 0 ]; then
        echo "错误: 数据库备份文件格式不正确"
        exit 1
    fi
fi

echo "备份文件完整性检查通过"
```

### 2. 备份报告

#### 2.1 每日备份报告

```bash
#!/bin/bash
# daily_backup_report.sh

REPORT_DATE=$(date +%Y-%m-%d)
REPORT_FILE="/var/log/backup/daily_report_${REPORT_DATE}.md"

cat > $REPORT_FILE << EOF
# RQA2025 备份日报 - ${REPORT_DATE}

## 备份状态总览

### 数据库备份
- **状态**: $(ls -t /data/backup/database/*.sql.gz 2>/dev/null | head -1 | xargs -I {} echo "✅ 成功" || echo "❌ 失败")
- **最新备份**: $(ls -t /data/backup/database/*.sql.gz 2>/dev/null | head -1 | xargs -I {} basename {} || echo "无")
- **备份大小**: $(ls -t /data/backup/database/*.sql.gz 2>/dev/null | head -1 | xargs -I {} du -h {} | cut -f1 || echo "N/A")
- **备份时间**: $(ls -t /data/backup/database/*.sql.gz 2>/dev/null | head -1 | xargs -I {} stat -c '%y' {} | cut -d'.' -f1 || echo "N/A")

### Redis备份
- **状态**: $(ls -t /data/backup/redis/*.rdb.gz 2>/dev/null | head -1 | xargs -I {} echo "✅ 成功" || echo "❌ 失败")
- **最新备份**: $(ls -t /data/backup/redis/*.rdb.gz 2>/dev/null | head -1 | xargs -I {} basename {} || echo "无")
- **备份大小**: $(ls -t /data/backup/redis/*.rdb.gz 2>/dev/null | head -1 | xargs -I {} du -h {} | cut -f1 || echo "N/A")

### 配置备份
- **状态**: $(ls -t /data/backup/config/*.tar.gz 2>/dev/null | head -1 | xargs -I {} echo "✅ 成功" || echo "❌ 失败")
- **最新备份**: $(ls -t /data/backup/config/*.tar.gz 2>/dev/null | head -1 | xargs -I {} basename {} || echo "无")
- **备份大小**: $(ls -t /data/backup/config/*.tar.gz 2>/dev/null | head -1 | xargs -I {} du -h {} | cut -f1 || echo "N/A")

## 存储使用情况

### 本地存储
- **数据库备份**: $(du -sh /data/backup/database 2>/dev/null | cut -f1 || echo "N/A")
- **Redis备份**: $(du -sh /data/backup/redis 2>/dev/null | cut -f1 || echo "N/A")
- **配置备份**: $(du -sh /data/backup/config 2>/dev/null | cut -f1 || echo "N/A")
- **总计**: $(du -sh /data/backup 2>/dev/null | cut -f1 || echo "N/A")

### 云存储
- **上传状态**: $(aws s3 ls s3://rqa2025-backup/ 2>/dev/null | wc -l | xargs -I {} echo "{} 个对象" || echo "检查失败")
- **最近上传**: $(aws s3 ls s3://rqa2025-backup/ 2>/dev/null | sort | tail -1 | awk '{print $1, $2}' || echo "检查失败")

## 问题和异常

$(cat /var/log/backup/error.log 2>/dev/null || echo "无错误记录")

## 建议和改进

- [ ] 定期检查备份文件完整性
- [ ] 优化备份存储策略
- [ ] 改进备份监控告警
- [ ] 增加备份性能测试

EOF

echo "备份日报已生成: $REPORT_FILE"
```

#### 2.2 备份合规性报告

```bash
#!/bin/bash
# backup_compliance_report.sh

REPORT_DATE=$(date +%Y-%m-%d)
REPORT_FILE="/var/log/backup/compliance_report_${REPORT_DATE}.md"

cat > $REPORT_FILE << EOF
# RQA2025 备份合规性报告 - ${REPORT_DATE}

## 合规性检查结果

### 备份频率检查
$(# 检查数据库备份频率
DB_BACKUPS=$(find /data/backup/database -name "*.sql.gz" -mtime -1 2>/dev/null | wc -l)
if [ $DB_BACKUPS -gt 0 ]; then
    echo "- ✅ 数据库备份: 符合每日备份要求"
else
    echo "- ❌ 数据库备份: 缺少今日备份"
fi

# 检查Redis备份频率
REDIS_BACKUPS=$(find /data/backup/redis -name "*.rdb.gz" -mtime -1 2>/dev/null | wc -l)
if [ $REDIS_BACKUPS -gt 0 ]; then
    echo "- ✅ Redis备份: 符合每日备份要求"
else
    echo "- ❌ Redis备份: 缺少今日备份"
fi
)

### 备份保留期检查
$(# 检查备份保留期
OLD_DB_BACKUPS=$(find /data/backup/database -name "*.sql.gz" -mtime +30 2>/dev/null | wc -l)
if [ $OLD_DB_BACKUPS -eq 0 ]; then
    echo "- ✅ 数据库备份: 符合30天保留期要求"
else
    echo "- ❌ 数据库备份: 存在超过30天的旧备份文件"
fi
)

### 备份完整性检查
$(# 检查备份文件完整性
INTEGRITY_ERRORS=0
for backup in /data/backup/database/*.sql.gz /data/backup/redis/*.rdb.gz; do
    if [ -f "$backup" ]; then
        if [[ "$backup" == *.gz ]]; then
            gunzip -t "$backup" 2>/dev/null || ((INTEGRITY_ERRORS++))
        fi
    fi
done

if [ $INTEGRITY_ERRORS -eq 0 ]; then
    echo "- ✅ 备份完整性: 所有备份文件完整"
else
    echo "- ❌ 备份完整性: 发现 $INTEGRITY_ERRORS 个损坏的备份文件"
fi
)

### 加密检查
$(# 检查备份文件是否加密
ENCRYPTED_BACKUPS=$(find /data/backup -name "*.enc" 2>/dev/null | wc -l)
TOTAL_BACKUPS=$(find /data/backup -type f \( -name "*.sql.gz" -o -name "*.rdb.gz" -o -name "*.tar.gz" \) 2>/dev/null | wc -l)

if [ $ENCRYPTED_BACKUPS -eq $TOTAL_BACKUPS ]; then
    echo "- ✅ 备份加密: 所有备份文件已加密"
else
    echo "- ❌ 备份加密: 只有 $ENCRYPTED_BACKUPS/$TOTAL_BACKUPS 个备份文件已加密"
fi
)

## 合规性状态

### 总体合规性
- **合规状态**: $( # 计算总体合规性
COMPLIANCE_SCORE=0
[ $DB_BACKUPS -gt 0 ] && ((COMPLIANCE_SCORE+=25))
[ $REDIS_BACKUPS -gt 0 ] && ((COMPLIANCE_SCORE+=25))
[ $OLD_DB_BACKUPS -eq 0 ] && ((COMPLIANCE_SCORE+=25))
[ $INTEGRITY_ERRORS -eq 0 ] && ((COMPLIANCE_SCORE+=25))
echo "${COMPLIANCE_SCORE}%"
)

### 监管要求对照
- [ ] **数据保护法规**: GDPR, PDPA等
- [ ] **金融监管要求**: 交易数据保留要求
- [ ] **行业标准**: ISO 27001, NIST等
- [ ] **内部政策**: 公司备份策略

## 改进建议

1. **自动化改进**
   - 实施自动化备份完整性检查
   - 增加备份失败的自动重试机制
   - 完善备份监控和告警

2. **安全加固**
   - 实施备份文件加密
   - 加强备份存储访问控制
   - 定期进行备份安全审计

3. **效率优化**
   - 优化备份文件压缩算法
   - 实施增量备份策略
   - 优化备份传输带宽使用

4. **合规完善**
   - 建立备份恢复演练计划
   - 完善备份文档和流程
   - 建立备份审计机制

EOF

echo "备份合规性报告已生成: $REPORT_FILE"
```

## 总结

完善的备份和恢复体系包括：

1. **多层次备份策略**: 数据库、应用、系统多层次备份
2. **标准化的恢复流程**: 详细的恢复步骤和验证方法
3. **灾难恢复计划**: 完整的灾难恢复流程和演练机制
4. **监控和报告**: 全面的备份监控和合规性报告
5. **持续改进**: 通过定期演练和报告不断优化备份策略

备份不仅仅是数据的保险，更是业务连续性的基石。
