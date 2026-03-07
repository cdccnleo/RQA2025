# RQA2025 基础设施层故障排除指南

## 目录
1. [故障分类与严重程度](#故障分类与严重程度)
2. [常见故障诊断](#常见故障诊断)
3. [系统级故障](#系统级故障)
4. [应用级故障](#应用级故障)
5. [网络与连接故障](#网络与连接故障)
6. [数据与存储故障](#数据与存储故障)
7. [性能问题诊断](#性能问题诊断)
8. [故障恢复流程](#故障恢复流程)
9. [预防措施](#预防措施)
10. [工具与命令参考](#工具与命令参考)

## 故障分类与严重程度

### 故障严重程度分级
- **P0 (严重)**: 系统完全不可用，影响核心业务
- **P1 (高)**: 主要功能不可用，影响大部分用户
- **P2 (中)**: 部分功能不可用，影响部分用户
- **P3 (低)**: 功能可用但性能下降，影响用户体验

### 故障响应时间要求
- **P0**: 15分钟内响应，2小时内恢复
- **P1**: 30分钟内响应，4小时内恢复
- **P2**: 2小时内响应，24小时内恢复
- **P3**: 24小时内响应，72小时内恢复

## 常见故障诊断

### 诊断流程
```mermaid
graph TD
    A[故障报告] --> B[问题分类]
    B --> C[影响评估]
    C --> D[紧急程度判断]
    D --> E[立即响应]
    E --> F[问题诊断]
    F --> G[解决方案实施]
    G --> H[验证测试]
    H --> I[恢复完成]
    I --> J[事后分析]
    J --> K[预防措施]
```

### 诊断工具清单
```bash
# 系统监控工具
htop, iotop, netstat, ss, lsof
# 日志分析工具
tail, grep, awk, sed, jq
# 网络诊断工具
ping, traceroute, telnet, curl, wget
# 数据库工具
psql, redis-cli, mysql
# 性能分析工具
strace, ltrace, perf, valgrind
```

## 系统级故障

### 1. 系统资源耗尽

#### 症状表现
- 系统响应缓慢或无响应
- 内存不足错误
- 磁盘空间不足
- CPU使用率持续100%

#### 诊断方法
```bash
# 检查系统资源
free -h                    # 内存使用情况
df -h                      # 磁盘使用情况
top                        # CPU和进程状态
iostat -x 1               # 磁盘I/O状态

# 检查进程资源占用
ps aux --sort=-%mem | head -10    # 内存占用最高的进程
ps aux --sort=-%cpu | head -10    # CPU占用最高的进程
lsof | wc -l                      # 打开文件数量
```

#### 解决方案
```bash
# 内存问题
# 1. 清理缓存
sync && echo 3 > /proc/sys/vm/drop_caches

# 2. 杀死占用内存过多的进程
kill -9 $(ps aux --sort=-%mem | head -2 | tail -1 | awk '{print $2}')

# 3. 调整内存参数
echo 'vm.swappiness=10' >> /etc/sysctl.conf
sysctl -p

# 磁盘问题
# 1. 清理日志文件
find /var/log -name "*.log" -mtime +7 -delete
find /var/log -name "*.gz" -mtime +30 -delete

# 2. 清理临时文件
rm -rf /tmp/*
rm -rf /var/tmp/*

# 3. 清理缓存
yum clean all  # CentOS/RHEL
apt clean      # Ubuntu/Debian
```

### 2. 系统服务异常

#### 症状表现
- 服务无法启动
- 服务频繁重启
- 服务响应超时
- 端口被占用

#### 诊断方法
```bash
# 检查服务状态
systemctl status <service_name>
journalctl -u <service_name> -f

# 检查端口占用
netstat -tulpn | grep :<port>
lsof -i :<port>
ss -tulpn | grep :<port>

# 检查服务依赖
systemctl list-dependencies <service_name>
```

#### 解决方案
```bash
# 服务重启
systemctl restart <service_name>

# 强制重启
systemctl stop <service_name>
pkill -f <service_name>
systemctl start <service_name>

# 端口冲突解决
# 1. 找到占用端口的进程
lsof -i :<port>
# 2. 杀死进程
kill -9 <PID>
# 3. 重新启动服务
systemctl start <service_name>
```

## 应用级故障

### 1. 应用启动失败

#### 症状表现
- 应用进程不存在
- 启动日志显示错误
- 端口监听失败
- 依赖服务连接失败

#### 诊断方法
```bash
# 检查应用进程
ps aux | grep <app_name>
pgrep -f <app_name>

# 检查启动日志
tail -f logs/app.log
tail -f logs/error.log

# 检查端口监听
netstat -tulpn | grep :<port>
ss -tulpn | grep :<port>

# 检查依赖服务
redis-cli ping
psql -h <host> -U <user> -d <db> -c "SELECT 1"
```

#### 解决方案
```python
# 常见启动问题及解决方案

# 1. 配置文件错误
# 检查配置文件语法
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# 2. 环境变量缺失
# 检查必需环境变量
required_envs = ['DB_URL', 'REDIS_URL', 'SECRET_KEY']
missing_envs = [env for env in required_envs if not os.getenv(env)]
if missing_envs:
    raise ValueError(f"Missing environment variables: {missing_envs}")

# 3. 依赖服务连接失败
# 实现重试机制
import time
from functools import wraps

def retry_on_failure(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry_on_failure(max_retries=5, delay=2)
def connect_database():
    # 数据库连接逻辑
    pass
```

### 2. 应用性能问题

#### 症状表现
- 响应时间过长
- 吞吐量下降
- 内存泄漏
- CPU使用率异常

#### 诊断方法
```bash
# 性能监控
# 1. 应用性能指标
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/health"

# 2. 系统性能指标
vmstat 1                    # 虚拟内存统计
iostat -x 1                 # 磁盘I/O统计
netstat -i                  # 网络接口统计

# 3. 进程性能分析
strace -p <PID> -o trace.log    # 系统调用跟踪
perf record -g -p <PID>          # 性能分析
```

#### 解决方案
```python
# 性能优化示例

# 1. 内存泄漏检测
import tracemalloc
import gc

def detect_memory_leak():
    tracemalloc.start()
    # 执行可疑操作
    snapshot1 = tracemalloc.take_snapshot()
    
    # 执行垃圾回收
    gc.collect()
    
    snapshot2 = tracemalloc.take_snapshot()
    
    # 比较快照
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    for stat in top_stats[:10]:
        print(stat)

# 2. 性能瓶颈分析
import cProfile
import pstats

def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    return result

# 3. 异步优化
import asyncio
import aiohttp

async def async_request(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [await resp.text() for resp in responses]
```

## 网络与连接故障

### 1. 网络连接问题

#### 症状表现
- 网络延迟高
- 连接超时
- 网络丢包
- DNS解析失败

#### 诊断方法
```bash
# 网络连通性测试
ping <host>
traceroute <host>
telnet <host> <port>

# 网络性能测试
iperf3 -c <server>          # 带宽测试
iperf3 -c <server> -u       # UDP测试
iperf3 -c <server> -R       # 反向测试

# DNS解析测试
nslookup <domain>
dig <domain>
host <domain>

# 网络接口状态
ip addr show
ip link show
ethtool <interface>
```

#### 解决方案
```bash
# 网络配置优化

# 1. TCP参数调优
echo 'net.core.somaxconn = 65535' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_max_syn_backlog = 65535' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_fin_timeout = 30' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_keepalive_time = 1200' >> /etc/sysctl.conf
sysctl -p

# 2. 网络接口调优
ethtool -G <interface> rx 4096 tx 4096
ethtool -C <interface> adaptive-rx on adaptive-tx on

# 3. 防火墙规则优化
iptables -A INPUT -p tcp --dport <port> -m limit --limit 100/second --limit-burst 200 -j ACCEPT
```

### 2. 负载均衡问题

#### 症状表现
- 部分节点无响应
- 流量分配不均
- 健康检查失败
- 会话保持问题

#### 诊断方法
```bash
# 负载均衡器状态检查
# Nginx
nginx -t                    # 配置语法检查
nginx -s reload            # 重新加载配置
nginx -s status            # 状态检查

# HAProxy
haproxy -c -f haproxy.cfg  # 配置检查
systemctl status haproxy   # 服务状态
```

#### 解决方案
```nginx
# Nginx负载均衡配置优化
upstream backend {
    least_conn;                    # 最少连接数算法
    server backend1:8000 max_fails=3 fail_timeout=30s;
    server backend2:8000 max_fails=3 fail_timeout=30s;
    server backend3:8000 max_fails=3 fail_timeout=30s;
    
    # 健康检查
    health_check interval=5s fails=3 passes=2;
}

server {
    listen 80;
    server_name example.com;
    
    location / {
        proxy_pass http://backend;
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;
    }
}
```

## 数据与存储故障

### 1. 数据库连接问题

#### 症状表现
- 数据库连接超时
- 连接池耗尽
- 查询执行缓慢
- 事务死锁

#### 诊断方法
```sql
-- PostgreSQL诊断查询
-- 1. 连接状态
SELECT * FROM pg_stat_activity WHERE state = 'active';

-- 2. 锁等待情况
SELECT 
    l.pid,
    l.mode,
    l.granted,
    a.usename,
    a.query
FROM pg_locks l
JOIN pg_stat_activity a ON l.pid = a.pid
WHERE NOT l.granted;

-- 3. 慢查询
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- 4. 表大小和索引使用
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE tablename = 'your_table_name';
```

#### 解决方案
```python
# 数据库连接优化

# 1. 连接池配置
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:pass@host:port/db',
    poolclass=QueuePool,
    pool_size=20,                    # 连接池大小
    max_overflow=30,                 # 最大溢出连接数
    pool_timeout=30,                 # 连接超时时间
    pool_recycle=3600,               # 连接回收时间
    pool_pre_ping=True               # 连接前ping检查
)

# 2. 查询优化
# 使用索引提示
query = """
SELECT /*+ INDEX(table_name index_name) */ 
    column1, column2
FROM table_name
WHERE condition = 'value'
"""

# 3. 批量操作优化
def batch_insert(data, batch_size=1000):
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        # 执行批量插入
        execute_batch(batch)
```

### 2. 缓存问题

#### 症状表现
- 缓存命中率低
- 缓存数据过期
- 内存使用过高
- 缓存穿透/雪崩

#### 诊断方法
```bash
# Redis诊断命令
redis-cli ping                    # 连接测试
redis-cli info memory            # 内存使用情况
redis-cli info stats             # 统计信息
redis-cli info keyspace          # 键空间信息
redis-cli monitor                # 实时监控命令

# 缓存命中率计算
keyspace_hits=$(redis-cli info stats | grep keyspace_hits | cut -d: -f2)
keyspace_misses=$(redis-cli info stats | grep keyspace_misses | cut -d: -f2)
hit_rate=$(echo "scale=2; $keyspace_hits / ($keyspace_hits + $keyspace_misses) * 100" | bc)
echo "Cache hit rate: ${hit_rate}%"
```

#### 解决方案
```python
# 缓存优化策略

# 1. 缓存穿透防护
import hashlib
import time

class CacheProtection:
    def __init__(self, cache_client):
        self.cache = cache_client
        self.bloom_filter = set()
    
    def get_with_protection(self, key, default=None):
        # 布隆过滤器检查
        if key not in self.bloom_filter:
            return default
        
        # 缓存查询
        value = self.cache.get(key)
        if value is None:
            # 设置空值缓存，防止穿透
            self.cache.setex(key, 300, "NULL_VALUE")
            return default
        
        return value if value != "NULL_VALUE" else default

# 2. 缓存雪崩防护
import random

def get_with_random_ttl(key, value, base_ttl=3600):
    # 添加随机TTL，避免同时过期
    random_ttl = base_ttl + random.randint(-300, 300)
    cache.setex(key, random_ttl, value)

# 3. 缓存预热
def cache_warmup():
    hot_keys = get_hot_keys_from_analytics()
    for key in hot_keys:
        value = fetch_from_database(key)
        cache.set(key, value, ex=3600)
```

## 性能问题诊断

### 1. 响应时间分析

#### 诊断方法
```bash
# HTTP响应时间测试
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/api/endpoint"

# 创建curl-format.txt文件
cat > curl-format.txt << EOF
     time_namelookup:  %{time_namelookup}\\n
        time_connect:  %{time_connect}\\n
     time_appconnect:  %{time_appconnect}\\n
    time_pretransfer:  %{time_pretransfer}\\n
       time_redirect:  %{time_redirect}\\n
  time_starttransfer:  %{time_starttransfer}\\n
                     ----------\\n
          time_total:  %{time_total}\\n
EOF

# 批量性能测试
ab -n 1000 -c 10 http://localhost:8000/api/endpoint
wrk -t12 -c400 -d30s http://localhost:8000/api/endpoint
```

#### 解决方案
```python
# 性能优化示例

# 1. 异步处理
import asyncio
import aiohttp

async def async_api_call(urls):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            task = asyncio.create_task(session.get(url))
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return [await resp.json() for resp in responses]

# 2. 连接池优化
import aiohttp
import asyncio

async def create_session_pool():
    connector = aiohttp.TCPConnector(
        limit=100,                    # 总连接数限制
        limit_per_host=30,            # 每个主机连接数限制
        ttl_dns_cache=300,            # DNS缓存TTL
        use_dns_cache=True,           # 启用DNS缓存
        keepalive_timeout=30,         # 保活超时
        enable_cleanup_closed=True    # 清理关闭连接
    )
    
    timeout = aiohttp.ClientTimeout(
        total=30,                     # 总超时
        connect=10,                   # 连接超时
        sock_read=30                  # 读取超时
    )
    
    return aiohttp.ClientSession(
        connector=connector,
        timeout=timeout
    )

# 3. 缓存优化
from functools import lru_cache
import time

class TimedCache:
    def __init__(self, ttl=3600):
        self.ttl = ttl
        self.cache = {}
    
    def get(self, key):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = (value, time.time())
```

### 2. 资源使用分析

#### 诊断方法
```bash
# 系统资源监控
# 1. CPU使用率
top -p $(pgrep -f app_name)
htop -p $(pgrep -f app_name)

# 2. 内存使用
ps -o pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -10
cat /proc/$(pgrep -f app_name)/status | grep -E "VmSize|VmRSS|VmPeak"

# 3. 磁盘I/O
iotop -p $(pgrep -f app_name)
iostat -x 1

# 4. 网络I/O
iftop -i eth0
nethogs eth0
```

#### 解决方案
```python
# 资源优化示例

# 1. 内存优化
import gc
import weakref

class MemoryOptimizer:
    def __init__(self):
        self._cache = weakref.WeakValueDictionary()
    
    def get(self, key):
        return self._cache.get(key)
    
    def set(self, key, value):
        self._cache[key] = value
    
    def cleanup(self):
        gc.collect()

# 2. 文件I/O优化
import aiofiles
import asyncio

async def async_file_operations():
    async with aiofiles.open('large_file.txt', 'r') as f:
        # 分块读取，避免内存溢出
        chunk_size = 1024 * 1024  # 1MB
        while True:
            chunk = await f.read(chunk_size)
            if not chunk:
                break
            # 处理数据块
            process_chunk(chunk)

# 3. 网络I/O优化
import socket

def optimize_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # 设置socket选项
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    # 设置缓冲区大小
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
    
    return sock
```

## 故障恢复流程

### 1. 紧急恢复流程

```mermaid
graph TD
    A[故障发生] --> B[告警触发]
    B --> C[紧急响应团队]
    C --> D[问题快速评估]
    D --> E[立即缓解措施]
    E --> F[服务恢复]
    F --> G[功能验证]
    G --> H[用户通知]
    H --> I[根本原因分析]
    I --> J[长期解决方案]
    J --> K[预防措施实施]
```

### 2. 恢复检查清单

```bash
#!/bin/bash
# recovery_checklist.sh

echo "=== 故障恢复检查清单 ==="

# 1. 服务状态检查
echo "1. 检查核心服务状态..."
systemctl status rqa-app
systemctl status redis
systemctl status postgresql

# 2. 网络连通性检查
echo "2. 检查网络连通性..."
ping -c 3 localhost
curl -f http://localhost:8000/health

# 3. 数据库连接检查
echo "3. 检查数据库连接..."
psql -h localhost -U rqa_user -d rqa -c "SELECT 1"

# 4. 缓存服务检查
echo "4. 检查缓存服务..."
redis-cli ping

# 5. 日志检查
echo "5. 检查错误日志..."
tail -n 50 logs/error.log | grep -i error

# 6. 性能指标检查
echo "6. 检查性能指标..."
curl -s http://localhost:8000/metrics | grep -E "(http_requests_total|http_request_duration_seconds)"

echo "=== 检查完成 ==="
```

### 3. 回滚策略

```yaml
# rollback-strategy.yaml
rollback:
  triggers:
    - error_rate_threshold: 0.05    # 错误率超过5%
    - response_time_threshold: 2.0  # 响应时间超过2秒
    - health_check_failures: 3      # 健康检查连续失败3次
  
  actions:
    - name: "自动回滚"
      type: "deployment_rollback"
      target_version: "previous_stable"
      
    - name: "流量切换"
      type: "traffic_switch"
      target: "blue_deployment"
      
    - name: "服务降级"
      type: "feature_toggle"
      disabled_features: ["advanced_analytics", "real_time_processing"]
  
  verification:
    - health_check_interval: 30s
    - metrics_collection: 60s
    - user_impact_assessment: true
```

## 预防措施

### 1. 监控告警配置

```yaml
# monitoring-rules.yaml
groups:
  - name: "预防性告警"
    rules:
      - alert: "高内存使用率预警"
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.8
        for: 5m
        labels:
          severity: "warning"
        annotations:
          summary: "内存使用率超过80%"
          description: "建议检查内存使用情况，防止内存不足"
      
      - alert: "磁盘空间预警"
        expr: (node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes > 0.85
        for: 5m
        labels:
          severity: "warning"
        annotations:
          summary: "磁盘空间使用率超过85%"
          description: "建议清理磁盘空间，防止磁盘满"
      
      - alert: "连接池使用率预警"
        expr: database_connections_active / database_connections_max > 0.8
        for: 2m
        labels:
          severity: "warning"
        annotations:
          summary: "数据库连接池使用率超过80%"
          description: "建议检查连接泄漏或增加连接池大小"
```

### 2. 定期维护计划

```bash
#!/bin/bash
# maintenance-schedule.sh

# 每日维护任务
daily_maintenance() {
    echo "执行每日维护任务..."
    
    # 1. 日志轮转
    logrotate /etc/logrotate.d/rqa
    
    # 2. 临时文件清理
    find /tmp -mtime +1 -delete
    find /var/tmp -mtime +1 -delete
    
    # 3. 缓存清理
    redis-cli FLUSHDB
    
    # 4. 健康检查
    curl -f http://localhost:8000/health
}

# 每周维护任务
weekly_maintenance() {
    echo "执行每周维护任务..."
    
    # 1. 数据库维护
    psql -h localhost -U rqa_user -d rqa -c "VACUUM ANALYZE;"
    
    # 2. 日志归档
    tar -czf logs/archive/logs-$(date +%Y%m%d).tar.gz logs/*.log
    rm logs/*.log
    
    # 3. 性能分析
    ./scripts/performance_analysis.sh
    
    # 4. 安全扫描
    ./scripts/security_scan.sh
}

# 每月维护任务
monthly_maintenance() {
    echo "执行每月维护任务..."
    
    # 1. 系统更新
    yum update -y  # CentOS/RHEL
    # apt update && apt upgrade -y  # Ubuntu/Debian
    
    # 2. 备份验证
    ./scripts/backup_verification.sh
    
    # 3. 容量规划
    ./scripts/capacity_planning.sh
    
    # 4. 文档更新
    ./scripts/documentation_update.sh
}

# 主函数
main() {
    case $1 in
        "daily")
            daily_maintenance
            ;;
        "weekly")
            weekly_maintenance
            ;;
        "monthly")
            monthly_maintenance
            ;;
        *)
            echo "Usage: $0 {daily|weekly|monthly}"
            exit 1
            ;;
    esac
}

main "$@"
```

### 3. 自动化测试

```python
# automated_testing.py
import unittest
import requests
import time
import subprocess

class AutomatedTesting:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
    
    def run_health_check(self):
        """健康检查测试"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            assert response.status_code == 200
            self.test_results.append(("健康检查", "通过"))
        except Exception as e:
            self.test_results.append(("健康检查", f"失败: {e}"))
    
    def run_performance_test(self):
        """性能测试"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/api/endpoint", timeout=10)
            end_time = time.time()
            
            response_time = end_time - start_time
            assert response_time < 1.0  # 响应时间应小于1秒
            self.test_results.append(("性能测试", f"通过: {response_time:.2f}s"))
        except Exception as e:
            self.test_results.append(("性能测试", f"失败: {e}"))
    
    def run_load_test(self):
        """负载测试"""
        try:
            # 使用ab进行负载测试
            cmd = f"ab -n 100 -c 10 {self.base_url}/api/endpoint"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.test_results.append(("负载测试", "通过"))
            else:
                self.test_results.append(("负载测试", f"失败: {result.stderr}"))
        except Exception as e:
            self.test_results.append(("负载测试", f"失败: {e}"))
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始自动化测试...")
        
        self.run_health_check()
        self.run_performance_test()
        self.run_load_test()
        
        # 输出测试结果
        print("\n=== 测试结果 ===")
        for test_name, result in self.test_results:
            print(f"{test_name}: {result}")
        
        # 检查是否有失败的测试
        failed_tests = [test for test, result in self.test_results if "失败" in result]
        if failed_tests:
            print(f"\n❌ 有 {len(failed_tests)} 个测试失败")
            return False
        else:
            print("\n✅ 所有测试通过")
            return True

if __name__ == "__main__":
    tester = AutomatedTesting()
    success = tester.run_all_tests()
    exit(0 if success else 1)
```

## 工具与命令参考

### 1. 系统诊断工具

```bash
# 系统信息
uname -a                    # 系统版本信息
cat /etc/os-release        # 操作系统信息
lscpu                      # CPU信息
free -h                    # 内存信息
df -h                      # 磁盘信息

# 进程管理
ps aux                     # 所有进程
top                        # 实时进程监控
htop                       # 交互式进程监控
iotop                      # I/O监控
nethogs                    # 网络使用监控

# 网络诊断
ip addr show               # 网络接口信息
ip route show              # 路由表
ss -tulpn                  # 网络连接状态
netstat -i                 # 网络接口统计
tcpdump -i eth0            # 网络包捕获

# 日志分析
journalctl -f              # 系统日志实时查看
journalctl -u service      # 特定服务日志
tail -f /var/log/messages  # 系统消息日志
grep -i error /var/log/*   # 搜索错误日志
```

### 2. 应用诊断工具

```bash
# 应用状态检查
systemctl status service   # 服务状态
systemctl is-active service # 服务是否活跃
systemctl is-enabled service # 服务是否启用

# 端口检查
lsof -i :port              # 端口占用情况
netstat -tulpn | grep port # 端口监听状态
ss -tulpn | grep port      # 现代端口检查

# 性能分析
strace -p PID              # 系统调用跟踪
ltrace -p PID              # 库函数调用跟踪
perf record -g -p PID      # 性能分析
valgrind --tool=memcheck   # 内存检查

# 日志分析
tail -f logfile            # 实时日志查看
grep -i error logfile      # 错误日志搜索
awk '/ERROR/ {print}' logfile # 错误行提取
sed -n '/ERROR/p' logfile  # 错误行显示
```

### 3. 数据库诊断工具

```sql
-- PostgreSQL诊断查询
-- 连接状态
SELECT 
    pid,
    usename,
    application_name,
    client_addr,
    state,
    query_start,
    query
FROM pg_stat_activity
WHERE state != 'idle';

-- 锁等待情况
SELECT 
    l.pid,
    l.mode,
    l.granted,
    a.usename,
    a.query
FROM pg_locks l
JOIN pg_stat_activity a ON l.pid = a.pid
WHERE NOT l.granted;

-- 慢查询统计
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;

-- 表大小和索引
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 20;
```

### 4. 缓存诊断工具

```bash
# Redis诊断命令
redis-cli ping                    # 连接测试
redis-cli info                    # 服务器信息
redis-cli info memory            # 内存使用情况
redis-cli info stats             # 统计信息
redis-cli info keyspace          # 键空间信息
redis-cli info replication       # 复制信息
redis-cli info clients           # 客户端信息

# Redis性能测试
redis-benchmark -h localhost -p 6379 -n 100000 -c 50

# Redis监控
redis-cli monitor                # 实时命令监控
redis-cli slowlog get 10         # 慢查询日志
redis-cli client list            # 客户端列表

# 缓存命中率计算
keyspace_hits=$(redis-cli info stats | grep keyspace_hits | cut -d: -f2)
keyspace_misses=$(redis-cli info stats | grep keyspace_misses | cut -d: -f2)
hit_rate=$(echo "scale=2; $keyspace_hits / ($keyspace_hits + $keyspace_misses) * 100" | bc)
echo "Cache hit rate: ${hit_rate}%"
```

## 总结

本故障排除指南提供了RQA2025基础设施层的全面故障诊断和解决方案，包括：

1. **故障分类与响应**: 明确的故障分级和响应时间要求
2. **系统级故障**: 资源耗尽、服务异常等系统问题的诊断和解决
3. **应用级故障**: 启动失败、性能问题等应用问题的诊断和解决
4. **网络与连接故障**: 网络问题、负载均衡等网络问题的诊断和解决
5. **数据与存储故障**: 数据库连接、缓存问题等存储问题的诊断和解决
6. **性能问题诊断**: 响应时间、资源使用等性能问题的分析和优化
7. **故障恢复流程**: 标准化的故障恢复流程和检查清单
8. **预防措施**: 监控告警、定期维护、自动化测试等预防性措施
9. **工具与命令参考**: 常用的诊断工具和命令参考

通过遵循本指南，运维团队可以快速定位和解决各种故障，确保系统的稳定性和可靠性。同时，预防措施的实施可以有效减少故障的发生，提高系统的整体质量。
