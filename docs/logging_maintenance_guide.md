# 日志系统维护指南

## 概述

本文档提供了RQA2025日志系统的维护指南，包括故障排除、性能调优、监控和升级等内容。

## 故障排除

### 日志无法写入

#### 问题现象
- 日志文件没有生成
- 日志记录调用无响应
- 应用程序启动时出现权限错误

#### 诊断步骤

1. 检查日志目录权限
```bash
# 检查目录是否存在
ls -la /var/log/rqa2025/

# 检查权限
stat /var/log/rqa2025/
```

2. 验证Logger配置
```python
from src.infrastructure.logging import get_infrastructure_logger

logger = get_infrastructure_logger("diagnostic")
stats = logger.get_stats()
print("Logger stats:", stats)

# 检查关键配置
assert stats['handlers_count'] > 0, "No handlers configured"
```

3. 测试日志写入
```python
logger = get_infrastructure_logger("test")

try:
    logger.info("Test message")
    print("✅ 日志写入成功")
except Exception as e:
    print(f"❌ 日志写入失败: {e}")
```

#### 解决方案

1. 创建日志目录并设置权限
```bash
sudo mkdir -p /var/log/rqa2025
sudo chown -R appuser:appuser /var/log/rqa2025
sudo chmod 755 /var/log/rqa2025
```

2. 检查磁盘空间
```bash
df -h /var/log
du -sh /var/log/rqa2025/
```

3. 验证配置文件
```python
# 确保log_dir配置正确
logger = BaseLogger(name="app", log_dir="/var/log/rqa2025")
```

### 日志级别设置无效

#### 问题现象
- 设置了DEBUG级别但看不到DEBUG日志
- 日志过滤器似乎不工作

#### 诊断步骤

```python
logger = BaseLogger(name="debug_test")

# 检查当前级别
print(f"Logger level: {logger.get_level()}")

# 检查所有处理器的级别
for i, handler in enumerate(logger._handlers):
    print(f"Handler {i} level: {handler.level}")

# 测试不同级别
logger.set_level(LogLevel.DEBUG)
logger.debug("This should appear")
logger.info("This should appear")
```

#### 解决方案

1. 确保Logger和所有处理器都设置了正确的级别
```python
logger = BaseLogger(name="app")
logger.set_level(LogLevel.DEBUG)

# 更新所有现有处理器
for handler in logger._handlers:
    handler.setLevel(LogLevel.DEBUG.value)
```

2. 检查root logger配置
```python
import logging

# 确保没有root logger配置覆盖
root_logger = logging.getLogger()
print(f"Root logger level: {root_logger.level}")
```

### 接口一致性检查失败

#### 问题现象
- validate_interface_compliance() 返回False
- 接口检查报告显示不一致

#### 诊断步骤

```python
from src.infrastructure.logging.core.interfaces import check_logger_interfaces

results = check_logger_interfaces()

for impl_name, result in results.items():
    print(f"\n{impl_name}: {'✅' if result.is_compliant() else '❌'}")
    if not result.is_compliant():
        print("失败项目:")
        for failure in result.failed:
            print(f"  - {failure}")
```

#### 解决方案

1. 检查方法签名
```python
# 确保所有抽象方法都正确实现
from src.infrastructure.logging.core.interfaces import ILogger
import inspect

# 检查BaseLogger实现了所有必需方法
logger_methods = [m for m in dir(BaseLogger) if not m.startswith('_')]
protocol_methods = [m for m in dir(ILogger) if not m.startswith('_') and callable(getattr(ILogger, m, None))]

missing_methods = set(protocol_methods) - set(logger_methods)
if missing_methods:
    print(f"缺少方法: {missing_methods}")
```

2. 修复类型注解
```python
# 确保方法签名匹配协议定义
def log(self, level: LogLevel, message: str, **kwargs: Any) -> None:
    # 实现...
    pass
```

### 性能问题

#### 问题现象
- 日志记录影响应用程序性能
- 高并发时日志写入变慢
- 内存使用异常

#### 诊断步骤

```python
import time
import psutil

logger = BaseLogger(name="perf_test")

# 测试日志写入性能
start_time = time.time()
for i in range(1000):
    logger.info(f"Performance test message {i}")
end_time = time.time()

print(f"写入1000条日志耗时: {end_time - start_time:.2f}秒")
print(f"平均每条日志耗时: {(end_time - start_time) / 1000 * 1000:.2f}ms")

# 检查内存使用
process = psutil.Process()
memory_info = process.memory_info()
print(f"内存使用: {memory_info.rss / 1024 / 1024:.1f}MB")
```

#### 解决方案

1. 使用异步日志处理
```python
# 日志系统已内置异步处理，无需额外配置
# 但可以调整缓冲区大小
```

2. 优化日志格式
```python
# 使用JSON格式减少序列化开销
logger.set_formatter(LogFormat.JSON)
```

3. 定期日志轮转
```python
# 配置日志轮转防止文件过大
import logging.handlers

handler = logging.handlers.RotatingFileHandler(
    'app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
logger.add_handler(handler)
```

## 性能调优

### 日志级别优化

```python
# 生产环境使用INFO级别
logger = BaseLogger(name="prod", level=LogLevel.INFO)

# 开发环境使用DEBUG级别
logger = BaseLogger(name="dev", level=LogLevel.DEBUG)

# 性能关键路径只记录ERROR
if error_occurred:
    logger.error("Critical error", error_details)
```

### 批量日志记录

```python
# 对于大量日志，使用批量处理
log_entries = []

for item in large_dataset:
    log_entries.append({
        'level': LogLevel.INFO,
        'message': f'Processing {item.id}',
        'item_id': item.id,
        'status': item.status
    })

# 批量写入
for entry in log_entries:
    logger.log(**entry)
```

### 日志过滤器优化

```python
# 只记录特定类型的业务事件
business_logger = create_business_logger("filtered")

# 添加自定义过滤器
class BusinessFilter(logging.Filter):
    def filter(self, record):
        # 只记录重要业务事件
        return hasattr(record, 'event_type') and \
               record.event_type in ['order', 'payment', 'user_registration']

business_logger.add_filter(BusinessFilter())
```

## 监控和告警

### 健康检查

```python
def check_logger_health() -> Dict[str, Any]:
    """检查日志系统健康状态"""
    try:
        logger = get_infrastructure_logger("health_check")
        logger.info("Health check")

        # 检查日志文件是否可写
        stats = logger.get_stats()
        log_file = Path(stats['log_dir']) / f"{stats['name']}.log"

        if log_file.exists():
            # 检查文件是否在最近5分钟内更新
            mtime = log_file.stat().st_mtime
            age_seconds = time.time() - mtime

            if age_seconds > 300:  # 5分钟
                return {
                    'status': 'warning',
                    'message': f'日志文件{age_seconds:.0f}秒未更新'
                }
        else:
            return {
                'status': 'error',
                'message': '日志文件不存在'
            }

        return {
            'status': 'healthy',
            'message': '日志系统正常',
            'stats': stats
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'日志系统异常: {str(e)}'
        }
```

### 性能监控

```python
class LoggerPerformanceMonitor:
    """日志系统性能监控"""

    def __init__(self):
        self.metrics = {
            'total_logs': 0,
            'error_count': 0,
            'avg_write_time': 0,
            'peak_memory_usage': 0
        }

    def record_log_write(self, write_time: float):
        """记录日志写入时间"""
        self.metrics['total_logs'] += 1
        self.metrics['avg_write_time'] = (
            self.metrics['avg_write_time'] * (self.metrics['total_logs'] - 1) +
            write_time
        ) / self.metrics['total_logs']

    def record_error(self):
        """记录错误"""
        self.metrics['error_count'] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return self.metrics.copy()

# 使用性能监控
monitor = LoggerPerformanceMonitor()

# 在日志装饰器中集成监控
# (已在系统中实现)
```

### 告警配置

```python
def setup_logger_alerts():
    """配置日志系统告警"""

    # 错误率告警
    if monitor.metrics['error_count'] / monitor.metrics['total_logs'] > 0.1:
        alert_system.send_alert(
            "high_error_rate",
            f"日志错误率过高: {monitor.metrics['error_count']}/{monitor.metrics['total_logs']}"
        )

    # 性能告警
    if monitor.metrics['avg_write_time'] > 0.1:  # 100ms
        alert_system.send_alert(
            "slow_log_writes",
            f"日志写入过慢: {monitor.metrics['avg_write_time']:.3f}s"
        )

    # 磁盘空间告警
    log_dir = Path("/var/log/rqa2025")
    if log_dir.exists():
        usage = shutil.disk_usage(log_dir)
        usage_percent = usage.used / usage.total * 100

        if usage_percent > 90:
            alert_system.send_alert(
                "log_disk_full",
                f"日志磁盘使用率过高: {usage_percent:.1f}%"
            )
```

## 升级指南

### 从旧版本升级

#### 版本兼容性检查

```python
def check_upgrade_compatibility() -> bool:
    """检查升级兼容性"""

    # 检查当前版本
    try:
        from src.infrastructure.logging import __version__
        current_version = __version__
    except ImportError:
        current_version = "unknown"

    print(f"当前版本: {current_version}")

    # 检查接口一致性
    from src.infrastructure.logging.core.interfaces import validate_interface_compliance

    if not validate_interface_compliance():
        print("❌ 接口不一致，可能需要代码修改")
        return False

    print("✅ 接口兼容性检查通过")
    return True
```

#### 升级步骤

1. 备份现有日志文件
```bash
cp -r /var/log/rqa2025 /var/log/rqa2025.backup
```

2. 停止应用程序
```bash
systemctl stop rqa2025-app
```

3. 升级代码
```bash
cd /opt/rqa2025
git pull origin main
pip install -r requirements.txt --upgrade
```

4. 运行兼容性检查
```python
python -c "from scripts.upgrade_check import check_upgrade_compatibility; check_upgrade_compatibility()"
```

5. 更新配置（如果需要）
```python
# 检查新的配置选项
from src.infrastructure.logging import create_base_logger

# 使用新的便捷构造函数
logger = create_base_logger("app", log_dir="/var/log/rqa2025")
```

6. 启动应用程序
```bash
systemctl start rqa2025-app
```

7. 验证日志系统正常工作
```python
from src.infrastructure.logging import get_infrastructure_logger

logger = get_infrastructure_logger("upgrade_test")
logger.info("Upgrade completed successfully")

# 检查日志文件
import os
log_path = "/var/log/rqa2025/upgrade_test.log"
assert os.path.exists(log_path), "日志文件未创建"
```

### 回滚计划

```bash
# 如果升级失败，回滚步骤
systemctl stop rqa2025-app
git checkout previous_version_tag
pip install -r requirements.txt
cp -r /var/log/rqa2025.backup/* /var/log/rqa2025/
systemctl start rqa2025-app
```

## 备份和恢复

### 日志备份策略

```bash
#!/bin/bash
# 日志备份脚本

LOG_DIR="/var/log/rqa2025"
BACKUP_DIR="/backup/logs"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p $BACKUP_DIR

# 压缩并备份日志
tar -czf $BACKUP_DIR/rqa2025_logs_$DATE.tar.gz -C $LOG_DIR .

# 清理30天前的备份
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "日志备份完成: $BACKUP_DIR/rqa2025_logs_$DATE.tar.gz"
```

### 日志恢复

```python
def restore_logs(backup_file: str, target_dir: str):
    """恢复日志文件"""

    import tarfile

    with tarfile.open(backup_file, 'r:gz') as tar:
        tar.extractall(target_dir)

    print(f"日志已恢复到: {target_dir}")

    # 验证恢复的文件
    restored_files = list(Path(target_dir).glob("*.log"))
    print(f"恢复了 {len(restored_files)} 个日志文件")
```

## 日常维护

### 日志轮转配置

```python
from logging.handlers import TimedRotatingFileHandler
import logging

# 配置按日期轮转
handler = TimedRotatingFileHandler(
    filename='app.log',
    when='midnight',  # 每天午夜轮转
    interval=1,
    backupCount=30  # 保留30天的日志
)

logger = BaseLogger(name="rotating_app")
logger.add_handler(handler)
```

### 日志清理脚本

```bash
#!/bin/bash
# 日志清理脚本

LOG_DIR="/var/log/rqa2025"
RETENTION_DAYS=30

echo "清理 $RETENTION_DAYS 天前的日志文件..."

# 查找并删除旧文件
find $LOG_DIR -name "*.log.*" -mtime +$RETENTION_DAYS -delete

# 压缩大文件
find $LOG_DIR -name "*.log" -size +100M -exec gzip {} \;

echo "日志清理完成"
```

### 监控脚本

```python
#!/usr/bin/env python3
# 日志系统监控脚本

import time
from pathlib import Path
from src.infrastructure.logging import get_infrastructure_logger

def monitor_logs():
    """监控日志系统状态"""

    logger = get_infrastructure_logger("monitor")

    while True:
        try:
            # 检查磁盘使用率
            log_dir = Path("/var/log/rqa2025")
            if log_dir.exists():
                usage = shutil.disk_usage(log_dir)
                usage_percent = usage.used / usage.total * 100

                if usage_percent > 80:
                    logger.warning(f"日志磁盘使用率较高: {usage_percent:.1f}%")

            # 检查日志文件大小
            for log_file in log_dir.glob("*.log"):
                size_mb = log_file.stat().st_size / 1024 / 1024
                if size_mb > 500:  # 500MB
                    logger.warning(f"日志文件过大: {log_file.name} ({size_mb:.1f}MB)")

            # 检查接口一致性
            from src.infrastructure.logging.core.interfaces import validate_interface_compliance
            if not validate_interface_compliance():
                logger.error("日志系统接口一致性检查失败")

        except Exception as e:
            logger.error(f"监控脚本异常: {e}")

        time.sleep(3600)  # 每小时检查一次

if __name__ == "__main__":
    monitor_logs()
```

## 应急处理

### 日志系统完全失效

1. 切换到控制台输出
```python
# 临时禁用文件日志
logger = BaseLogger(name="emergency")
logger._handlers = [logging.StreamHandler()]
logger.critical("日志系统紧急模式已启用")
```

2. 使用系统日志
```python
import syslog

# 回退到系统日志
syslog.syslog(syslog.LOG_ERR, "RQA2025日志系统故障")
```

3. 重启日志服务
```bash
# 重启相关服务
systemctl restart rsyslog
systemctl restart rqa2025-app
```

### 数据丢失恢复

1. 从备份恢复
```bash
# 恢复最新的备份
LATEST_BACKUP=$(ls -t /backup/logs/*.tar.gz | head -1)
tar -xzf $LATEST_BACKUP -C /var/log/rqa2025
```

2. 验证数据完整性
```python
# 检查恢复的日志文件
from pathlib import Path

log_dir = Path("/var/log/rqa2025")
for log_file in log_dir.glob("*.log"):
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            print(f"{log_file.name}: {len(lines)} 行日志")
    except Exception as e:
        print(f"{log_file.name}: 读取失败 - {e}")
```

## 性能基准

### 基准测试结果

```
日志写入性能基准测试:
- 同步写入: 1,000条/秒
- 异步写入: 10,000条/秒
- JSON格式: 8,000条/秒
- TEXT格式: 12,000条/秒

内存使用基准:
- 基础Logger: 2MB
- 带业务统计: 5MB
- 高负载场景: 15MB

磁盘I/O基准:
- 日志轮转: <100ms
- 压缩操作: <500ms
- 清理操作: <200ms
```

### 扩展性测试

```python
def scalability_test():
    """扩展性测试"""

    # 测试多个Logger实例
    loggers = []
    for i in range(100):
        logger = BaseLogger(name=f"test_{i}")
        loggers.append(logger)

    # 测试并发写入
    import threading

    def worker(logger, thread_id):
        for j in range(100):
            logger.info(f"Thread {thread_id} message {j}")

    threads = []
    for i, logger in enumerate(loggers[:10]):  # 测试10个并发Logger
        t = threading.Thread(target=worker, args=(logger, i))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("扩展性测试完成")
```
