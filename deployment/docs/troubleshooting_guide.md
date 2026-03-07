# RQA2025 故障排除指南

## 启动问题

### 服务无法启动

**症状**: `systemctl start rqa2025` 失败

**检查步骤**:
1. 查看详细错误日志：
   ```bash
   sudo journalctl -u rqa2025 -n 50
   ```

2. 检查配置文件语法：
   ```bash
   python -m py_compile config/production.json
   ```

3. 验证数据库连接：
   ```bash
   python -c "
   import psycopg2
   conn = psycopg2.connect(os.environ['RQA2025_DATABASE_URL'])
   print('数据库连接正常')
   "
   ```

4. 检查端口占用：
   ```bash
   sudo lsof -i :8000
   ```

**解决方案**:
- 修复配置文件错误
- 重新安装缺失的依赖
- 重启相关服务（PostgreSQL、Redis）

### 应用崩溃

**症状**: 服务启动后不久崩溃

**检查步骤**:
1. 查看应用日志：
   ```bash
   tail -f /var/log/rqa2025/app.log
   ```

2. 检查内存使用：
   ```bash
   ps aux | grep rqa2025
   ```

3. 验证Python环境：
   ```bash
   /opt/rqa2025/venv/bin/python --version
   ```

**解决方案**:
- 增加内存限制
- 检查代码中的资源泄漏
- 优化启动脚本

## 性能问题

### 响应缓慢

**检查步骤**:
1. 监控系统资源：
   ```bash
   top
   iostat -x 1
   ```

2. 检查数据库查询：
   ```sql
   SELECT * FROM pg_stat_activity WHERE state = 'active';
   ```

3. 分析应用指标：
   ```bash
   curl http://localhost:8000/metrics
   ```

**解决方案**:
- 优化数据库查询
- 增加缓存
- 扩展系统资源

### 内存泄漏

**检查步骤**:
1. 监控内存使用趋势：
   ```bash
   ps aux --no-headers -o pmem,pid,cmd | sort -nr | head -10
   ```

2. 使用内存分析工具：
   ```bash
   python -c "
   import tracemalloc
   tracemalloc.start()
   # 运行一段时间后
   snapshot = tracemalloc.take_snapshot()
   top_stats = snapshot.statistics('traceback')
   for stat in top_stats[:10]:
       print(stat)
   "
   ```

**解决方案**:
- 重启服务释放内存
- 修复代码中的内存泄漏
- 优化对象生命周期管理

## 数据库问题

### 连接失败

**检查步骤**:
1. 验证数据库服务状态：
   ```bash
   sudo systemctl status postgresql
   ```

2. 检查连接配置：
   ```bash
   psql -h localhost -U rqa2025_user -d rqa2025
   ```

3. 查看连接池状态：
   ```bash
   SELECT * FROM pg_stat_activity;
   ```

**解决方案**:
- 重启数据库服务
- 检查网络配置
- 调整连接池大小

### 查询缓慢

**检查步骤**:
1. 识别慢查询：
   ```sql
   SELECT query, total_time, calls, mean_time
   FROM pg_stat_statements
   ORDER BY mean_time DESC
   LIMIT 10;
   ```

2. 分析查询计划：
   ```sql
   EXPLAIN ANALYZE SELECT * FROM large_table WHERE conditions;
   ```

**解决方案**:
- 添加适当的索引
- 重写查询语句
- 优化数据库配置

## 网络问题

### 无法访问API

**检查步骤**:
1. 验证服务状态：
   ```bash
   curl http://localhost:8000/health
   ```

2. 检查防火墙：
   ```bash
   sudo ufw status
   ```

3. 验证反向代理配置：
   ```bash
   sudo nginx -t
   ```

**解决方案**:
- 配置防火墙规则
- 检查Nginx配置
- 验证SSL证书

## 日志分析

### 错误日志模式

**常见错误模式**:
- `Connection refused`: 数据库或外部服务连接问题
- `MemoryError`: 内存不足
- `TimeoutError`: 请求超时
- `ImportError`: 依赖缺失

### 日志轮转

```bash
# 配置logrotate
sudo tee /etc/logrotate.d/rqa2025 > /dev/null <<EOF
/var/log/rqa2025/*.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
    create 644 rqa2025 rqa2025
    postrotate
        systemctl reload rqa2025
    endscript
}
EOF
```

## 预防措施

### 定期维护

- 每周检查系统资源使用
- 每月更新依赖包
- 每季度进行安全审计

### 监控告警

- 设置关键指标阈值
- 配置多渠道告警通知
- 建立应急响应流程

### 备份策略

- 每日数据库备份
- 每周完整应用备份
- 关键配置实时同步

---
*本文档由RQA2025部署准备系统自动生成*
