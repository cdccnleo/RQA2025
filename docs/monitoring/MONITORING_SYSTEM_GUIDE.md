# 监控系统使用指南

## 系统概述

RQA2025监控系统是一个多层次、智能化的监控解决方案，包含以下核心组件：

### 1. 监控核心组件
- **增强版监控系统**: `scripts/monitoring/enhanced_deployment_monitor.py`
- **Web仪表板**: 三个版本（基础版、增强版、高级版）
- **数据库存储**: SQLite数据库持久化
- **告警系统**: 智能告警通知机制

### 2. 技术特性
- ✅ 实时监控数据收集
- ✅ 多线程异步处理
- ✅ WebSocket实时更新
- ✅ 多种图表类型（柱状图、折线图、饼图、雷达图）
- ✅ 智能告警通知
- ✅ 数据持久化存储

## 快速开始

### 1. 启动监控系统

```bash
# 启动增强版监控系统
python scripts/monitoring/enhanced_deployment_monitor.py

# 启动Web仪表板（选择其中一个）
python scripts/monitoring/simple_web_dashboard.py      # 端口5000
python scripts/monitoring/enhanced_web_dashboard.py    # 端口5001
python scripts/monitoring/advanced_web_dashboard.py    # 端口5002
```

### 2. 访问Web仪表板

- **基础版**: http://localhost:5000
- **增强版**: http://localhost:5001
- **高级版**: http://localhost:5002

### 3. API接口

```bash
# 状态接口
GET http://localhost:5000/api/status
GET http://localhost:5001/api/status
GET http://localhost:5002/api/status

# 分布数据接口（高级版）
GET http://localhost:5002/api/distribution
```

## 系统配置

### 1. 数据库配置

监控数据存储在 `data/monitoring.db`，包含以下表：
- `monitoring_data`: 监控数据记录
- `alert_history`: 告警历史记录
- `system_metrics`: 系统指标数据

### 2. 告警配置

告警系统支持以下配置：
- 告警阈值设置
- 通知方式（邮件、Webhook）
- 告警级别（INFO、WARNING、ERROR、CRITICAL）

### 3. 监控指标

系统监控以下指标：
- CPU使用率
- 内存使用率
- 磁盘使用率
- 网络流量
- 应用性能指标

## 故障排除

### 1. 常见问题

**问题**: Web仪表板无法访问
**解决方案**: 
1. 检查端口是否被占用
2. 确认防火墙设置
3. 重启Web服务

**问题**: 数据库连接失败
**解决方案**:
1. 检查数据库文件权限
2. 确认数据库路径正确
3. 重新初始化数据库

**问题**: 监控数据不更新
**解决方案**:
1. 检查监控进程是否运行
2. 确认数据收集配置
3. 查看日志文件

### 2. 日志文件

监控系统日志位置：
- 应用日志: `logs/app.log`
- 错误日志: `logs/error.log`
- 监控日志: `logs/monitoring.log`

### 3. 性能优化

- 调整数据收集频率
- 优化数据库查询
- 配置缓存策略
- 启用数据压缩

## 开发指南

### 1. 添加新的监控指标

```python
# 在 enhanced_deployment_monitor.py 中添加新指标
def collect_custom_metric(self):
    """收集自定义指标"""
    metric_value = self.get_custom_value()
    self.record_metric('custom_metric', metric_value)
```

### 2. 扩展告警规则

```python
# 在 AlertEvaluator 中添加新规则
def check_custom_alert(self, metric_value):
    """检查自定义告警"""
    if metric_value > self.threshold:
        self.trigger_alert('custom_alert', metric_value)
```

### 3. 自定义Web仪表板

```python
# 在Web仪表板中添加新页面
@app.route('/custom')
def custom_page():
    return render_template('custom.html')
```

## 最佳实践

### 1. 监控策略
- 设置合理的告警阈值
- 定期检查监控数据
- 及时响应告警通知
- 保持监控系统更新

### 2. 数据管理
- 定期备份监控数据
- 清理过期数据
- 优化数据库性能
- 监控存储空间

### 3. 安全考虑
- 限制访问权限
- 加密敏感数据
- 定期更新依赖
- 监控异常访问

## 下一步开发计划

### 短期目标 (1-2周)
1. **完善Web仪表板**
   - 实现环境切换功能
   - 优化移动端显示
   - 添加更多图表类型

2. **实现真正的邮件发送功能**
   - 集成SMTP邮件服务
   - 配置邮件模板
   - 支持多种告警级别

3. **添加用户认证和权限管理**
   - 用户登录系统
   - 角色权限控制
   - 操作日志记录

### 中期目标 (1-2个月)
1. **支持更多监控指标**
   - CPU、内存、磁盘、网络详细指标
   - 应用性能监控
   - 业务指标监控

2. **实现监控数据聚合和分析**
   - 数据统计分析
   - 趋势预测
   - 异常检测

3. **集成Prometheus/Grafana**
   - 数据导出到Prometheus
   - Grafana仪表板集成
   - 告警规则同步

### 长期目标 (3-6个月)
1. **引入AI驱动的异常检测**
   - 机器学习异常检测
   - 智能根因分析
   - 预测性维护

2. **支持多云环境监控**
   - 多云资源监控
   - 统一监控平台
   - 跨云告警管理

3. **实现智能运维**
   - 自动化故障修复
   - 智能容量规划
   - 性能自动优化

## 联系支持

如有问题或建议，请联系开发团队或查看项目文档：
- 项目文档: `docs/`
- 测试报告: `reports/`
- 变更日志: `CHANGELOG.md` 