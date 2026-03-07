# 生产环境故障排查指南

## 概述

本文档提供RQA2025量化交易系统在生产环境中的常见故障排查方法和解决方案，帮助运维人员快速定位和解决问题。

## 故障分类

### 1. 系统层故障

#### 1.1 服务器宕机

**现象**:
- 无法访问应用
- 系统无响应
- 监控指标全部丢失

**排查步骤**:
```bash
# 1. 检查服务器状态
ping <server_ip>

# 2. 检查SSH连接
ssh -i <key_file> user@<server_ip>

# 3. 检查系统进程
ps aux | grep -E "(python|java|nginx|docker)"

# 4. 检查系统日志
tail -f /var/log/syslog
tail -f /var/log/messages

# 5. 检查系统资源
top
free -h
df -h
```

**解决方案**:
```bash
# 1. 重启服务器（如果可以远程访问）
sudo reboot

# 2. 联系云服务商重启服务器（如果无法访问）

# 3. 检查启动脚本
ls -la /etc/init.d/
ls -la /lib/systemd/system/

# 4. 重启关键服务
sudo systemctl restart nginx
sudo systemctl restart docker
sudo systemctl restart rqa2025
```

#### 1.2 高CPU使用率

**现象**:
- 应用响应缓慢
- CPU使用率持续超过90%
- 监控告警触发

**排查步骤**:
```bash
# 1. 查看CPU使用情况
top -b -n1 | head -20

# 2. 分析进程CPU使用
ps aux --sort=-%cpu | head -10

# 3. 查看系统负载
uptime
cat /proc/loadavg

# 4. 分析线程状态
ps -eLf | grep python | wc -l
```

**解决方案**:
```bash
# 1. 重启应用服务
sudo systemctl restart rqa2025

# 2. 如果是特定进程导致，杀掉进程
kill -9 <PID>

# 3. 优化应用配置
# 减少工作线程数
# 增加缓存配置
# 优化数据库查询

# 4. 检查是否有死锁或无限循环
# 查看应用日志中的错误信息
tail -f /var/log/rqa2025/application.log | grep -i error
```

#### 1.3 内存不足

**现象**:
- 应用频繁重启
- OutOfMemory错误
- 系统响应缓慢

**排查步骤**:
```bash
# 1. 查看内存使用情况
free -h
cat /proc/meminfo

# 2. 分析内存使用进程
ps aux --sort=-%mem | head -10

# 3. 检查内存泄露
# 查看Java/Python垃圾回收日志
# 分析堆转储文件

# 4. 检查系统内存配置
cat /proc/sys/vm/overcommit_memory
sysctl vm.swappiness
```

**解决方案**:
```bash
# 1. 增加系统内存（如果可能）
# 升级服务器配置

# 2. 优化应用内存配置
# 减少缓存大小
# 调整JVM/Python内存参数

# 3. 重启应用服务
sudo systemctl restart rqa2025

# 4. 清理系统缓存
echo 3 > /proc/sys/vm/drop_caches

# 5. 检查是否有内存泄露
# 分析应用代码
# 使用内存分析工具
```

### 2. 网络层故障

#### 2.1 服务端口无法访问

**现象**:
- 特定端口无法访问
- 连接超时
- 服务启动失败

**排查步骤**:
```bash
# 1. 检查端口监听状态
netstat -tlnp | grep -E ":(8000|8001|8002)"
ss -tlnp | grep -E ":(8000|8001|8002)"

# 2. 检查防火墙配置
sudo ufw status
sudo iptables -L -n

# 3. 检查服务进程
ps aux | grep -E "(python|java)" | grep -v grep

# 4. 检查服务日志
tail -f /var/log/rqa2025/application.log
tail -f /var/log/nginx/error.log

# 5. 测试端口连接
telnet localhost 8000
curl http://localhost:8000/health
```

**解决方案**:
```bash
# 1. 重启服务
sudo systemctl restart rqa2025

# 2. 检查配置文件的端口设置
grep -r "8000\|8001\|8002" /etc/rqa2025/

# 3. 修复防火墙规则
sudo ufw allow 8000
sudo ufw allow 8001
sudo ufw allow 8002

# 4. 检查nginx配置
sudo nginx -t
sudo systemctl reload nginx
```

#### 2.2 网络延迟高

**现象**:
- API响应时间超过预期
- 网络请求超时
- 用户反馈访问慢

**排查步骤**:
```bash
# 1. 测试网络延迟
ping -c 10 <target_server>
traceroute <target_server>

# 2. 检查网络带宽
iftop -i eth0
nload eth0

# 3. 分析网络连接数
netstat -ant | awk '{print $6}' | sort | uniq -c | sort -n
ss -ant | awk '{print $1}' | sort | uniq -c

# 4. 检查网络丢包
netstat -s | grep -i "packet"
```

**解决方案**:
```bash
# 1. 优化网络配置
# 调整TCP参数
sudo sysctl -w net.ipv4.tcp_tw_reuse=1
sudo sysctl -w net.ipv4.tcp_tw_recycle=1

# 2. 增加带宽
# 升级网络配置

# 3. 优化应用配置
# 减少网络请求
# 增加连接池大小
# 使用异步请求

# 4. 检查是否有网络攻击
# 查看网络安全日志
tail -f /var/log/ufw.log
```

### 3. 应用层故障

#### 3.1 应用启动失败

**现象**:
- 应用无法启动
- 服务状态为failed
- 启动日志显示错误

**排查步骤**:
```bash
# 1. 检查服务状态
sudo systemctl status rqa2025
sudo systemctl is-active rqa2025

# 2. 查看启动日志
sudo journalctl -u rqa2025 -f
tail -f /var/log/rqa2025/startup.log

# 3. 检查配置文件
ls -la /etc/rqa2025/
cat /etc/rqa2025/config.yaml

# 4. 检查依赖服务
sudo systemctl status postgresql
sudo systemctl status redis
sudo systemctl status nginx

# 5. 检查端口冲突
netstat -tlnp | grep -E ":(8000|8001|8002)"
```

**解决方案**:
```bash
# 1. 修复配置文件错误
sudo nano /etc/rqa2025/config.yaml

# 2. 解决依赖问题
sudo systemctl restart postgresql
sudo systemctl restart redis

# 3. 解决端口冲突
sudo netstat -tlnp | grep 8000
sudo kill -9 <conflicting_pid>

# 4. 重新启动服务
sudo systemctl daemon-reload
sudo systemctl start rqa2025
sudo systemctl enable rqa2025
```

#### 3.2 数据库连接失败

**现象**:
- 数据库连接错误
- 查询超时
- 连接池耗尽

**排查步骤**:
```bash
# 1. 检查数据库服务状态
sudo systemctl status postgresql
pg_isready -h localhost -p 5432

# 2. 检查数据库连接数
psql -h localhost -U rqa2025 -d rqa2025 -c "SELECT count(*) FROM pg_stat_activity;"

# 3. 检查数据库配置
cat /etc/postgresql/13/main/postgresql.conf | grep -E "(max_connections|shared_buffers|work_mem)"

# 4. 检查数据库日志
tail -f /var/log/postgresql/postgresql-13-main.log

# 5. 测试数据库连接
psql -h localhost -U rqa2025 -d rqa2025 -c "SELECT 1;"
```

**解决方案**:
```bash
# 1. 重启数据库服务
sudo systemctl restart postgresql

# 2. 优化数据库配置
sudo nano /etc/postgresql/13/main/postgresql.conf
# 增加 max_connections
# 调整 shared_buffers
# 优化 work_mem

# 3. 清理连接
psql -h localhost -U rqa2025 -d rqa2025 -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle';"

# 4. 检查数据库健康
psql -h localhost -U rqa2025 -d rqa2025 -c "VACUUM ANALYZE;"
psql -h localhost -U rqa2025 -d rqa2025 -c "REINDEX DATABASE rqa2025;"

# 5. 检查应用连接池配置
# 调整连接池大小
# 增加连接超时时间
# 优化连接重试机制
```

#### 3.3 Redis连接失败

**现象**:
- 缓存读取失败
- 会话丢失
- 性能下降

**排查步骤**:
```bash
# 1. 检查Redis服务状态
sudo systemctl status redis
redis-cli ping

# 2. 检查Redis内存使用
redis-cli info memory
redis-cli info stats

# 3. 检查Redis配置
cat /etc/redis/redis.conf | grep -E "(maxmemory|maxclients|timeout)"

# 4. 检查Redis日志
tail -f /var/log/redis/redis-server.log

# 5. 测试Redis连接
redis-cli set test_key "test_value"
redis-cli get test_key
```

**解决方案**:
```bash
# 1. 重启Redis服务
sudo systemctl restart redis

# 2. 清理Redis内存
redis-cli flushall  # 清空所有数据（谨慎使用）
redis-cli flushdb   # 清空当前数据库

# 3. 优化Redis配置
sudo nano /etc/redis/redis.conf
# 增加 maxmemory
# 调整 maxclients
# 优化 timeout

# 4. 检查Redis持久化
redis-cli bgsave
redis-cli lastsave

# 5. 检查应用缓存配置
# 调整缓存过期时间
# 优化缓存策略
# 增加缓存命中率监控
```

### 4. 业务层故障

#### 4.1 交易执行失败

**现象**:
- 交易订单无法提交
- 交易状态异常
- 交易数据不一致

**排查步骤**:
```bash
# 1. 检查交易服务日志
tail -f /var/log/rqa2025/trading.log | grep -i error

# 2. 检查订单状态
# 查询数据库中的订单状态
psql -h localhost -U rqa2025 -d rqa2025 -c "SELECT id, status, created_at FROM orders WHERE created_at > NOW() - INTERVAL '1 hour' ORDER BY created_at DESC LIMIT 10;"

# 3. 检查交易队列
# 检查消息队列状态
redis-cli llen trading_queue
redis-cli lrange trading_queue 0 10

# 4. 检查风控服务
# 查看风控日志
tail -f /var/log/rqa2025/risk.log | grep -i error

# 5. 检查市场数据
# 验证市场数据连接
curl http://localhost:8001/market/status
```

**解决方案**:
```bash
# 1. 重启交易服务
sudo systemctl restart rqa2025-trading

# 2. 清理交易队列
redis-cli del trading_queue

# 3. 修复数据一致性
# 检查并修复不一致的订单状态
psql -h localhost -U rqa2025 -d rqa2025 -c "
UPDATE orders SET status = 'cancelled'
WHERE status = 'pending' AND created_at < NOW() - INTERVAL '24 hours';
"

# 4. 检查风控配置
# 验证风控参数
# 调整风控阈值

# 5. 验证市场连接
# 检查市场数据源连接
# 重启市场数据服务
```

#### 4.2 策略执行异常

**现象**:
- 策略无法执行
- 策略结果错误
- 策略性能下降

**排查步骤**:
```bash
# 1. 检查策略服务日志
tail -f /var/log/rqa2025/strategy.log | grep -i error

# 2. 检查策略状态
psql -h localhost -U rqa2025 -d rqa2025 -c "SELECT id, name, status, updated_at FROM strategies WHERE status != 'active';"

# 3. 检查策略参数
psql -h localhost -U rqa2025 -d rqa2025 -c "SELECT strategy_id, parameter_name, parameter_value FROM strategy_parameters WHERE updated_at > NOW() - INTERVAL '1 hour';"

# 4. 检查数据源
# 验证策略所需的数据源是否正常
curl http://localhost:8001/data/status

# 5. 检查计算资源
# 验证是否有足够的计算资源运行策略
top -b -n1 | grep -E "(python|java)"
```

**解决方案**:
```bash
# 1. 重启策略服务
sudo systemctl restart rqa2025-strategy

# 2. 修复策略配置
# 检查策略配置文件
# 验证策略参数有效性

# 3. 清理策略缓存
redis-cli keys "strategy:*" | xargs redis-cli del

# 4. 重置策略状态
psql -h localhost -U rqa2025 -d rqa2025 -c "
UPDATE strategies SET status = 'active', error_message = NULL
WHERE status = 'error' AND updated_at < NOW() - INTERVAL '1 hour';
"

# 5. 检查数据源连接
# 验证数据源连接
# 重启数据服务
```

## 应急处理流程

### 1. 告警响应流程

#### 1.1 告警接收
- 通过邮件/钉钉/微信接收告警通知
- 记录告警接收时间
- 确认告警级别和影响范围

#### 1.2 问题确认
```bash
# 登录监控系统
open http://grafana.company.com

# 查看详细指标
# 确认问题范围和严重程度
```

#### 1.3 问题定位
```bash
# 使用本指南中的排查步骤
# 根据告警类型选择相应章节
# 记录排查过程和发现的问题
```

#### 1.4 问题解决
```bash
# 执行相应的解决方案
# 记录执行的命令和操作
# 验证问题是否解决
```

#### 1.5 验证恢复
```bash
# 确认系统正常运行
# 验证业务功能正常
# 确认监控指标恢复正常
```

#### 1.6 记录总结
```markdown
# 故障处理报告

## 基本信息
- 告警时间: 2025-01-15 14:30:00
- 恢复时间: 2025-01-15 15:00:00
- 故障时长: 30分钟
- 影响范围: 交易API响应延迟

## 问题描述
CPU使用率持续超过90%，导致API响应时间增加

## 排查过程
1. 收到CPU高使用率告警
2. 登录服务器查看系统状态
3. 使用top命令发现Python进程CPU占用过高
4. 检查应用日志发现死锁问题

## 解决方案
1. 重启应用服务
2. 清理Redis缓存
3. 优化数据库查询

## 预防措施
1. 增加CPU监控阈值
2. 优化应用代码，防止死锁
3. 增加自动重启机制
```

### 2. 紧急联系方式

#### 技术团队联系方式
- **技术负责人**: tech-leader@company.com (138-0000-0000)
- **后端开发**: backend-team@company.com
- **前端开发**: frontend-team@company.com
- **运维团队**: ops-team@company.com
- **安全团队**: security-team@company.com

#### 外部支持联系方式
- **云服务商**: support@cloud-provider.com
- **数据库专家**: dba@company.com
- **网络专家**: network@company.com

### 3. 灾难恢复流程

#### 3.1 灾难声明
- 确认灾难级别（P1/P2/P3/P4）
- 启动应急响应团队
- 通知相关利益方

#### 3.2 灾难评估
```bash
# 评估系统损坏程度
# 检查数据完整性
# 评估业务影响
```

#### 3.3 灾难恢复
```bash
# 1. 停止受损服务
sudo systemctl stop rqa2025

# 2. 备份当前数据
# 立即备份数据库和日志
pg_dump -h localhost -U rqa2025 -d rqa2025 > emergency_backup.sql

# 3. 从备份恢复
psql -h localhost -U rqa2025 -d rqa2025 < latest_backup.sql

# 4. 验证数据完整性
psql -h localhost -U rqa2025 -d rqa2025 -c "SELECT COUNT(*) FROM orders;"
psql -h localhost -U rqa2025 -d rqa2025 -c "SELECT COUNT(*) FROM users;"

# 5. 重新启动服务
sudo systemctl start rqa2025
```

#### 3.4 业务恢复验证
```bash
# 1. 验证API可用性
curl http://localhost:8000/health

# 2. 验证数据库连接
psql -h localhost -U rqa2025 -d rqa2025 -c "SELECT 1;"

# 3. 验证缓存服务
redis-cli ping

# 4. 验证业务功能
curl http://localhost:8000/api/orders
```

## 故障排查工具

### 1. 系统工具

```bash
# 1. 系统状态检查
htop                    # 实时系统监控
iotop                   # I/O监控
nmon                    # 综合性能监控

# 2. 网络工具
tcpdump                 # 网络包分析
wireshark               # 网络协议分析
netstat -antp           # 网络连接状态

# 3. 数据库工具
pg_top                  # PostgreSQL实时监控
pgBadger                # PostgreSQL日志分析
pg_stat_statements      # SQL性能分析
```

### 2. 应用工具

```bash
# 1. Python应用监控
py-spy                  # Python程序性能分析
memory_profiler         # 内存使用分析
line_profiler           # 代码行性能分析

# 2. 日志分析工具
grep                    # 日志过滤
awk                     # 日志处理
sed                     # 日志格式化
logrotate               # 日志轮转
```

### 3. 监控工具

```bash
# 1. Prometheus工具
promtool                # 配置检查和验证
amtool                  # AlertManager管理

# 2. Grafana工具
grafana-cli             # Grafana命令行工具
```

## 总结

通过系统化的故障排查方法，可以：

1. **快速定位问题**: 使用分层排查方法，快速定位故障根因
2. **规范处理流程**: 遵循标准化的应急响应流程
3. **减少影响时间**: 通过预案和工具，减少故障恢复时间
4. **积累经验**: 通过故障总结，不断完善监控和告警体系
5. **预防未来问题**: 通过问题分析，改进系统设计和运维策略

故障排查是一个持续改进的过程，需要运维人员不断学习和积累经验。
