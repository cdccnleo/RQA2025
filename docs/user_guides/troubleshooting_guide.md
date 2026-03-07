# 故障排除指南

## 概述

本文档提供了RQA2025数据层的故障排除指南，帮助快速定位和解决常见问题。

## 问题分类

### 1. 启动问题

#### 问题: 应用无法启动

**症状**:
- 启动时出现错误信息
- 端口被占用
- 依赖缺失

**排查步骤**:

1. 检查Python版本
   ```bash
   python --version
   # 确保版本 >= 3.8
   ```

2. 检查依赖安装
   ```bash
   pip list | grep -E "(fastapi|uvicorn|aiohttp)"
   ```

3. 检查端口占用
   ```bash
   netstat -tlnp | grep 8000
   ```

4. 检查配置文件
   ```bash
   # 检查配置文件是否存在
   ls -la config/
   
   # 检查配置文件格式
   python -c "import yaml; yaml.safe_load(open('config/app.yaml'))"
   ```

**解决方案**:
- 升级Python到3.8+
- 重新安装依赖: `pip install -r requirements.txt`
- 更换端口或杀死占用进程
- 修复配置文件格式错误

#### 问题: 数据库连接失败

**症状**:
- 启动时数据库连接错误
- 数据加载失败

**排查步骤**:

1. 检查数据库服务
   ```bash
   systemctl status postgresql
   # 或
   systemctl status mysql
   ```

2. 检查连接配置
   ```bash
   # 检查环境变量
   echo $DATABASE_URL
   
   # 测试连接
   psql $DATABASE_URL -c "SELECT 1"
   ```

3. 检查网络连接
   ```bash
   telnet localhost 5432
   ```

**解决方案**:
- 启动数据库服务
- 检查数据库用户权限
- 修复连接字符串
- 检查防火墙设置

### 2. 性能问题

#### 问题: 响应时间过长

**症状**:
- API响应时间 > 1秒
- 数据加载缓慢
- 系统卡顿

**排查步骤**:

1. 检查系统资源
   ```bash
   # CPU使用率
   top
   
   # 内存使用
   free -h
   
   # 磁盘使用
   df -h
   ```

2. 检查应用性能
   ```bash
   # 查看慢查询
   grep "slow" logs/app.log
   
   # 检查缓存命中率
   curl http://localhost:8000/api/v1/data/cache/stats
   ```

3. 检查网络延迟
   ```bash
   # 测试网络延迟
   ping api.coingecko.com
   ```

**解决方案**:
- 增加系统资源
- 优化数据库查询
- 调整缓存配置
- 使用CDN加速

#### 问题: 内存使用过高

**症状**:
- 内存使用率 > 90%
- 系统开始使用交换分区
- 应用响应变慢

**排查步骤**:

1. 检查内存使用
   ```bash
   # 查看内存使用详情
   cat /proc/meminfo
   
   # 查看进程内存使用
   ps aux --sort=-%mem | head -10
   ```

2. 检查内存泄漏
   ```bash
   # 使用memory_profiler
   python -m memory_profiler your_script.py
   ```

**解决方案**:
- 增加系统内存
- 优化代码减少内存使用
- 调整缓存大小
- 重启应用释放内存

### 3. 数据问题

#### 问题: 数据加载失败

**症状**:
- 数据源连接失败
- 数据格式错误
- 数据不完整

**排查步骤**:

1. 检查数据源状态
   ```bash
   # 测试API连接
   curl -I https://api.coingecko.com/api/v3/ping
   ```

2. 检查API密钥
   ```bash
   # 检查环境变量
   echo $COINGECKO_API_KEY
   ```

3. 检查数据格式
   ```bash
   # 查看错误日志
   grep "data" logs/app.log | grep ERROR
   ```

**解决方案**:
- 检查网络连接
- 验证API密钥
- 修复数据格式
- 实现重试机制

#### 问题: 数据质量差

**症状**:
- 数据不准确
- 数据缺失
- 数据重复

**排查步骤**:

1. 检查数据质量报告
   ```bash
   curl http://localhost:8000/api/v1/data/quality/report
   ```

2. 检查数据验证规则
   ```bash
   # 查看验证配置
   cat config/validation.yaml
   ```

**解决方案**:
- 修复数据源问题
- 调整验证规则
- 实现数据修复
- 联系数据提供商

### 4. 监控问题

#### 问题: 监控指标异常

**症状**:
- 监控面板显示异常
- 告警频繁触发
- 指标数据缺失

**排查步骤**:

1. 检查监控服务
   ```bash
   # 检查Prometheus
   curl http://localhost:9090/-/healthy
   
   # 检查Grafana
   curl http://localhost:3000/api/health
   ```

2. 检查指标收集
   ```bash
   # 查看应用指标
   curl http://localhost:8000/metrics
   ```

**解决方案**:
- 重启监控服务
- 检查指标配置
- 修复数据收集
- 调整告警阈值

## 日志分析

### 日志位置

```bash
# 应用日志
logs/app.log

# 错误日志
logs/error.log

# 访问日志
logs/access.log

# 性能日志
logs/performance.log
```

### 常用日志命令

```bash
# 查看最新日志
tail -f logs/app.log

# 查看错误日志
grep ERROR logs/app.log

# 查看特定时间段的日志
sed -n '/2024-01-01 10:00/,/2024-01-01 11:00/p' logs/app.log

# 统计错误次数
grep -c ERROR logs/app.log
```

### 日志级别

- **DEBUG**: 调试信息
- **INFO**: 一般信息
- **WARNING**: 警告信息
- **ERROR**: 错误信息
- **CRITICAL**: 严重错误

## 调试工具

### 1. 应用调试

```python
# 启用调试模式
import logging
logging.basicConfig(level=logging.DEBUG)

# 使用pdb调试
import pdb
pdb.set_trace()
```

### 2. 性能分析

```bash
# 使用cProfile分析性能
python -m cProfile -o profile.stats your_script.py

# 分析结果
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"
```

### 3. 内存分析

```bash
# 使用memory_profiler
pip install memory_profiler
python -m memory_profiler your_script.py
```

## 预防措施

### 1. 定期维护

- 每日检查系统状态
- 每周清理日志文件
- 每月更新依赖包
- 每季度进行安全更新

### 2. 监控告警

- 设置合理的告警阈值
- 配置多渠道通知
- 定期检查监控状态
- 及时响应告警信息

### 3. 备份策略

- 定期备份数据库
- 备份配置文件
- 备份重要数据
- 测试恢复流程

## 联系支持

如果问题无法通过本指南解决，请联系技术支持：

- **邮箱**: support@rqa2025.com
- **文档**: https://docs.rqa2025.com
- **GitHub**: https://github.com/your-org/rqa2025/issues
