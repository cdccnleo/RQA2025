# 生产环境性能调优指南

## 概述

本文档提供RQA2025量化交易系统在生产环境中的性能调优方法和最佳实践，帮助系统达到最佳性能状态。

## 性能基准

### 1. 性能目标

| 指标 | 目标值 | 当前值 | 状态 |
|------|--------|--------|------|
| API响应时间 | <100ms (P95) | <50ms | ✅ 优秀 |
| 数据库查询时间 | <50ms (P95) | <30ms | ✅ 优秀 |
| 内存使用率 | <80% | <60% | ✅ 良好 |
| CPU使用率 | <70% | <40% | ✅ 良好 |
| 并发用户数 | >1000 | >1500 | ✅ 优秀 |
| 交易成功率 | >99.5% | >99.8% | ✅ 优秀 |

### 2. 监控指标

#### 应用层指标
```python
# 关键性能指标
METRICS = {
    'api_response_time': 'API响应时间',
    'db_query_time': '数据库查询时间',
    'cache_hit_rate': '缓存命中率',
    'error_rate': '错误率',
    'throughput': '吞吐量',
    'active_connections': '活跃连接数'
}
```

#### 系统层指标
```bash
# 系统性能监控
top -b -n1 | head -10
vmstat 1 5
iostat -x 1 5
free -h
df -h
```

## 数据库性能调优

### 1. PostgreSQL调优

#### 1.1 内存配置

```sql
-- 查看当前配置
SELECT name, setting, unit FROM pg_settings WHERE name IN (
    'shared_buffers',
    'work_mem',
    'maintenance_work_mem',
    'effective_cache_size'
);

-- 推荐配置（根据服务器内存调整）
-- 假设服务器内存64GB
ALTER SYSTEM SET shared_buffers = '16GB';
ALTER SYSTEM SET work_mem = '128MB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET effective_cache_size = '48GB';
```

#### 1.2 连接池配置

```sql
-- 连接池设置
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET listen_addresses = '*';
ALTER SYSTEM SET max_prepared_transactions = 100;
```

#### 1.3 查询优化

```sql
-- 分析慢查询
SELECT query, calls, total_time, mean_time, rows
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- 创建索引
CREATE INDEX CONCURRENTLY idx_orders_user_id ON orders(user_id);
CREATE INDEX CONCURRENTLY idx_orders_status ON orders(status);
CREATE INDEX CONCURRENTLY idx_orders_created_at ON orders(created_at);

-- 分析表统计信息
ANALYZE orders;
ANALYZE users;
ANALYZE trades;
```

### 2. Redis调优

#### 2.1 内存优化

```bash
# Redis配置优化
maxmemory 4gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# 持久化配置
save 900 1
save 300 10
save 60 10000

# AOF配置
appendonly yes
appendfsync everysec
```

#### 2.2 连接优化

```bash
# 连接池配置
redis-cli config set maxclients 1000
redis-cli config set timeout 300

# TCP优化
echo "net.core.somaxconn = 65535" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65535" >> /etc/sysctl.conf
sysctl -p
```

## 应用性能调优

### 1. Python应用优化

#### 1.1 Gunicorn配置

```python
# gunicorn.conf.py
import multiprocessing

# Worker配置
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'uvicorn.workers.UvicornWorker'
worker_connections = 1000

# 超时配置
timeout = 30
keepalive = 5
graceful_timeout = 30

# 内存配置
worker_tmp_dir = '/dev/shm'
preload_app = True

# 日志配置
loglevel = 'info'
accesslog = '/var/log/rqa2025/access.log'
errorlog = '/var/log/rqa2025/error.log'
```

#### 1.2 异步处理优化

```python
# 使用异步数据库连接
from databases import Database
import asyncio

database = Database('postgresql://user:password@localhost/dbname')

# 连接池配置
database_config = {
    'min_size': 5,
    'max_size': 20,
    'pool_recycle': 3600,
    'echo': False
}

# 异步缓存操作
import aioredis

redis = aioredis.from_url("redis://localhost", encoding="utf-8", decode_responses=True)

async def get_cached_data(key):
    data = await redis.get(key)
    if data is None:
        data = await fetch_from_database(key)
        await redis.setex(key, 3600, data)
    return data
```

#### 1.3 内存优化

```python
# 内存监控
import psutil
import gc

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

def optimize_memory():
    """内存优化策略"""
    # 1. 清理未使用的对象
    gc.collect()

    # 2. 监控大对象
    import sys
    large_objects = []
    for obj in gc.get_objects():
        size = sys.getsizeof(obj)
        if size > 1024 * 1024:  # 1MB
            large_objects.append((type(obj).__name__, size))

    # 3. 优化数据结构
    # 使用slots减少内存占用
    class OptimizedClass:
        __slots__ = ['field1', 'field2', 'field3']

    return large_objects
```

### 2. API性能优化

#### 2.1 缓存策略

```python
# 多级缓存策略
from functools import lru_cache
import time

# 应用级缓存
@lru_cache(maxsize=1000)
def get_user_permissions(user_id):
    return fetch_permissions_from_db(user_id)

# Redis缓存装饰器
def redis_cache(expire=3600):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{args}:{kwargs}"
            result = await redis.get(key)
            if result is None:
                result = await func(*args, **kwargs)
                await redis.setex(key, expire, json.dumps(result))
            else:
                result = json.loads(result)
            return result
        return wrapper
    return decorator

# 使用缓存
@redis_cache(expire=300)
async def get_market_data(symbol):
    return await fetch_market_data(symbol)
```

#### 2.2 分页和限流

```python
# 智能分页
def smart_pagination(query, page=1, size=50):
    """智能分页策略"""
    max_size = 1000  # 最大页面大小

    if size > max_size:
        size = max_size

    offset = (page - 1) * size

    # 添加索引提示
    if 'ORDER BY created_at' in str(query):
        query = query.with_hint('INDEX(orders_created_at_idx)')

    return query.offset(offset).limit(size)

# 限流器
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware

limiter = Limiter(key_func=get_remote_address)

# 全局限流
@limiter.limit("100/minute")
async def global_limit(request: Request):
    pass

# API特定限流
@limiter.limit("10/minute")
async def create_order(request: Request):
    """创建订单接口限流"""
    pass
```

## 系统层性能调优

### 1. Linux系统优化

#### 1.1 内核参数优化

```bash
# /etc/sysctl.conf 优化配置

# 网络优化
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_tw_recycle = 1
net.ipv4.tcp_fin_timeout = 30

# 内存优化
vm.swappiness = 10
vm.dirty_ratio = 20
vm.dirty_background_ratio = 10
vm.overcommit_memory = 1

# 文件系统优化
fs.file-max = 1000000
fs.nr_open = 1000000

# 应用优化
kernel.sched_migration_cost_ns = 5000000
kernel.sched_autogroup_enabled = 0

# 应用配置
kernel.shmmax = 68719476736
kernel.shmall = 4294967296
```

#### 1.2 文件系统优化

```bash
# 挂载选项优化
# /etc/fstab
UUID=xxx / ext4 noatime,nodiratime,barrier=0,data=writeback 0 1
UUID=xxx /data ext4 noatime,nodiratime,barrier=0,data=writeback 0 2

# 禁用透明大页
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag

# I/O调度器优化
echo deadline > /sys/block/sda/queue/scheduler
echo 0 > /sys/block/sda/queue/rotational
```

### 2. 网络优化

#### 2.1 Nginx优化

```nginx
# nginx.conf 优化配置
user nginx;
worker_processes auto;
worker_rlimit_nofile 65535;

events {
    use epoll;
    worker_connections 65535;
    multi_accept on;
}

http {
    # 基础设置
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    keepalive_requests 100;

    # 缓存设置
    open_file_cache max=200000 inactive=20s;
    open_file_cache_valid 30s;
    open_file_cache_min_uses 2;
    open_file_cache_errors on;

    # Gzip压缩
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;
}
```

#### 2.2 TCP优化

```bash
# TCP参数优化
cat >> /etc/sysctl.conf << EOF
# TCP优化
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_timestamps = 1
net.ipv4.tcp_sack = 1
net.ipv4.tcp_dsack = 1
net.ipv4.tcp_fack = 1
net.ipv4.tcp_congestion_control = cubic
net.ipv4.tcp_rmem = 4096 87380 4194304
net.ipv4.tcp_wmem = 4096 87380 4194304
net.ipv4.tcp_mem = 24576 32768 49152
EOF

sysctl -p
```

## 监控和诊断工具

### 1. 性能监控工具

#### 1.1 系统监控

```bash
# 实时系统监控
htop
glances
dstat

# 网络监控
iftop
nethogs
tcpdump

# 磁盘I/O监控
iotop
iostat -x 1
```

#### 1.2 应用监控

```bash
# Python性能监控
py-spy record -p <pid> --format speedscope
memory_profiler
line_profiler

# 数据库监控
pg_top
pg_activity
pg_stat_statements
```

### 2. 诊断工具

#### 2.1 问题诊断

```python
# 应用诊断工具
import traceback
import logging
from contextlib import contextmanager

@contextmanager
def performance_context(name):
    """性能上下文管理器"""
    start_time = time.time()
    start_memory = get_memory_usage()

    try:
        yield
    finally:
        end_time = time.time()
        end_memory = get_memory_usage()

        logger.info(f"{name} 性能数据:")
        logger.info(f"  执行时间: {end_time - start_time:.3f}秒")
        logger.info(f"  内存使用: {end_memory - start_memory:.2f}MB")

def diagnose_performance_issue():
    """性能问题诊断"""
    # 1. 检查数据库连接池
    db_pool_status = get_db_pool_status()
    if db_pool_status['active_connections'] > db_pool_status['max_connections'] * 0.8:
        logger.warning("数据库连接池压力较大")

    # 2. 检查缓存命中率
    cache_stats = get_cache_stats()
    if cache_stats['hit_rate'] < 0.8:
        logger.warning(f"缓存命中率过低: {cache_stats['hit_rate']:.2%}")

    # 3. 检查慢查询
    slow_queries = get_slow_queries()
    if slow_queries:
        logger.warning(f"发现 {len(slow_queries)} 个慢查询")

    # 4. 检查内存泄露
    memory_info = get_memory_info()
    if memory_info['growth_rate'] > 10:  # MB/minute
        logger.warning("检测到内存泄露迹象")
```

#### 2.2 自动诊断脚本

```bash
#!/bin/bash
# performance_diagnostic.sh

echo "=== RQA2025 性能诊断 ==="
echo "诊断时间: $(date)"

# 1. 系统负载检查
echo "系统负载:"
uptime
echo ""

# 2. CPU使用情况
echo "CPU使用情况:"
top -b -n1 | head -10
echo ""

# 3. 内存使用情况
echo "内存使用情况:"
free -h
echo ""

# 4. 磁盘I/O
echo "磁盘I/O情况:"
iostat -x 1 3
echo ""

# 5. 网络连接
echo "网络连接情况:"
netstat -ant | awk '{print $6}' | sort | uniq -c | sort -n
echo ""

# 6. 应用进程状态
echo "应用进程状态:"
ps aux | grep -E "(python|rqa2025)" | grep -v grep
echo ""

# 7. 数据库连接状态
echo "数据库连接状态:"
psql -h localhost -U rqa2025 -d rqa2025 -c "
SELECT count(*) as total_connections,
       count(*) filter (where state = 'active') as active_connections,
       count(*) filter (where state = 'idle') as idle_connections
FROM pg_stat_activity;
" 2>/dev/null || echo "无法连接数据库"
echo ""

# 8. Redis状态
echo "Redis状态:"
redis-cli info | grep -E "(connected_clients|used_memory_human|hits|misses)" 2>/dev/null || echo "无法连接Redis"
echo ""

# 9. 应用健康检查
echo "应用健康检查:"
curl -s http://localhost:8000/health | head -5 || echo "健康检查失败"
echo ""

# 10. 错误日志检查
echo "最近错误日志:"
tail -20 /var/log/rqa2025/error.log 2>/dev/null || echo "无错误日志文件"
echo ""

echo "=== 诊断完成 ==="
```

## 性能测试和基准

### 1. 性能测试策略

#### 1.1 负载测试

```bash
# 使用Apache Bench进行负载测试
ab -n 10000 -c 100 -g results.tsv http://localhost:8000/api/orders

# 使用JMeter进行复杂场景测试
# 1. 登录场景
# 2. 查询订单
# 3. 创建订单
# 4. 取消订单
```

#### 1.2 压力测试

```bash
# 逐步增加负载
for concurrency in 10 50 100 200 500 1000; do
    echo "测试并发数: $concurrency"
    ab -n 1000 -c $concurrency http://localhost:8000/api/market
    sleep 10
done
```

#### 1.3 容量测试

```bash
# 持续负载测试
# 运行24小时，监控系统稳定性
nohup ab -n 1000000 -c 100 http://localhost:8000/api/orders > capacity_test.log 2>&1 &
```

### 2. 性能基准管理

#### 2.1 基准数据收集

```python
# 性能基准收集器
import time
import psutil
import statistics
from typing import List, Dict

class PerformanceBenchmark:
    def __init__(self):
        self.metrics = {
            'response_time': [],
            'memory_usage': [],
            'cpu_usage': [],
            'db_query_time': [],
            'cache_hit_rate': []
        }

    def collect_metric(self, metric_name: str, value: float):
        """收集性能指标"""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)

    def get_statistics(self, metric_name: str) -> Dict:
        """获取统计信息"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}

        data = self.metrics[metric_name]
        return {
            'count': len(data),
            'min': min(data),
            'max': max(data),
            'mean': statistics.mean(data),
            'median': statistics.median(data),
            'p95': sorted(data)[int(len(data) * 0.95)] if len(data) > 20 else max(data),
            'p99': sorted(data)[int(len(data) * 0.99)] if len(data) > 100 else max(data)
        }

    def generate_report(self) -> str:
        """生成基准报告"""
        report = []
        report.append("# 性能基准报告")
        report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        for metric, data in self.metrics.items():
            if data:
                stats = self.get_statistics(metric)
                report.append(f"\n## {metric}")
                report.append(".3f"                report.append(".3f"                report.append(".3f"                report.append(".3f"                report.append(".3f"                report.append(".3f"                report.append(f"- 样本数量: {stats['count']}")

        return '\n'.join(report)
```

#### 2.2 基准比较

```python
def compare_with_baseline(current: Dict, baseline: Dict, threshold: float = 0.1) -> Dict:
    """与基准比较"""
    comparison = {}

    for metric in current.keys():
        if metric in baseline:
            current_p95 = current[metric].get('p95', 0)
            baseline_p95 = baseline[metric].get('p95', 0)

            if baseline_p95 > 0:
                diff_percent = (current_p95 - baseline_p95) / baseline_p95

                if abs(diff_percent) > threshold:
                    status = "退化" if diff_percent > 0 else "改进"
                    comparison[metric] = {
                        'status': status,
                        'current': current_p95,
                        'baseline': baseline_p95,
                        'difference': abs(diff_percent),
                        'significant': True
                    }
                else:
                    comparison[metric] = {
                        'status': '稳定',
                        'current': current_p95,
                        'baseline': baseline_p95,
                        'difference': abs(diff_percent),
                        'significant': False
                    }

    return comparison
```

## 性能调优清单

### 1. 日常调优检查清单

- [ ] 检查系统资源使用率
- [ ] 监控应用响应时间
- [ ] 分析数据库查询性能
- [ ] 检查缓存命中率
- [ ] 监控错误日志
- [ ] 验证备份状态
- [ ] 检查安全告警

### 2. 周度调优任务

- [ ] 分析慢查询并优化
- [ ] 检查索引使用情况
- [ ] 清理过期缓存
- [ ] 优化配置文件
- [ ] 更新系统补丁
- [ ] 验证监控告警

### 3. 月度调优任务

- [ ] 全面性能测试
- [ ] 容量规划评估
- [ ] 架构优化评估
- [ ] 成本优化分析
- [ ] 安全漏洞扫描
- [ ] 灾难恢复演练

## 总结

系统性能调优是一个持续的过程，需要：

1. **建立基准**: 通过持续的性能监控建立性能基准
2. **主动监控**: 使用多层次的监控体系主动发现问题
3. **快速诊断**: 建立完善的诊断工具和流程
4. **持续优化**: 通过定期的性能测试和调优改进系统性能
5. **预防为主**: 通过容量规划和架构优化预防性能问题

性能调优不仅能提升用户体验，还能提高系统稳定性，降低运维成本。
