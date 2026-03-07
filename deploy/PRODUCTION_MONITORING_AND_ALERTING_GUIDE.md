# 生产环境监控和告警指南

## 概述

本文档详细说明RQA2025量化交易系统在生产环境中的监控和告警配置，确保系统运行状态的实时监控和异常情况的及时告警。

## 监控架构

### 1. 监控组件

#### Prometheus
- **用途**: 指标收集和存储
- **端口**: 9090
- **存储**: 本地存储 + 远程存储
- **保留期**: 30天

#### Grafana
- **用途**: 可视化仪表板
- **端口**: 3000
- **数据源**: Prometheus
- **用户认证**: 管理员账户

#### AlertManager
- **用途**: 告警管理和通知
- **端口**: 9093
- **通知渠道**: 邮件、钉钉、微信

### 2. 监控指标分类

#### 系统层监控

```yaml
# CPU使用率
cpu_usage > 80% 持续5分钟 → 警告告警
cpu_usage > 95% 持续2分钟 → 严重告警

# 内存使用率
memory_usage > 85% 持续5分钟 → 警告告警
memory_usage > 95% 持续2分钟 → 严重告警

# 磁盘使用率
disk_usage > 80% 持续10分钟 → 警告告警
disk_usage > 90% 持续5分钟 → 严重告警
```

#### 应用层监控

```yaml
# 应用响应时间
http_request_duration_seconds > 2.0 → 警告告警
http_request_duration_seconds > 5.0 → 严重告警

# 错误率
http_request_error_rate > 5% 持续5分钟 → 警告告警
http_request_error_rate > 10% 持续2分钟 → 严重告警

# 数据库连接池
db_connection_pool_exhausted > 0 持续1分钟 → 严重告警
```

#### 业务层监控

```yaml
# 交易成功率
trade_success_rate < 95% 持续10分钟 → 警告告警
trade_success_rate < 90% 持续5分钟 → 严重告警

# 策略执行状态
strategy_execution_failures > 5 持续10分钟 → 警告告警
strategy_execution_failures > 10 持续5分钟 → 严重告警
```

## 告警配置

### 1. 告警规则

#### 基础告警规则 (monitoring/alert_rules.yml)

```yaml
groups:
  - name: system_alerts
    rules:
      - alert: HighCPUUsage
        expr: cpu_usage > 80
        for: 5m
        labels:
          severity: warning
          service: system
        annotations:
          summary: "CPU使用率过高"
          description: "CPU使用率超过80%，当前值: {{ $value }}%"

      - alert: HighMemoryUsage
        expr: memory_usage > 85
        for: 5m
        labels:
          severity: warning
          service: system
        annotations:
          summary: "内存使用率过高"
          description: "内存使用率超过85%，当前值: {{ $value }}%"

      - alert: ServiceDown
        expr: up == 0
        for: 2m
        labels:
          severity: critical
          service: application
        annotations:
          summary: "服务不可用"
          description: "服务 {{ $labels.service }} 已经宕机"
```

#### 业务告警规则

```yaml
  - name: business_alerts
    rules:
      - alert: LowTradeSuccessRate
        expr: rate(trade_success_total[5m]) / rate(trade_total[5m]) < 0.95
        for: 10m
        labels:
          severity: warning
          service: trading
        annotations:
          summary: "交易成功率过低"
          description: "交易成功率低于95%，当前值: {{ $value }}%"

      - alert: HighOrderLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2.0
        for: 5m
        labels:
          severity: warning
          service: api
        annotations:
          summary: "订单延迟过高"
          description: "95%订单响应时间超过2秒，当前值: {{ $value }}秒"
```

### 2. 告警通知

#### 邮件通知配置

```yaml
global:
  smtp_smarthost: 'smtp.company.com:587'
  smtp_from: 'alerts@company.com'
  smtp_auth_username: 'alerts@company.com'
  smtp_auth_password: 'password'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'team'

receivers:
  - name: 'team'
    email_configs:
      - to: 'trading-team@company.com'
        subject: '[RQA2025] {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          告警: {{ .Annotations.summary }}
          描述: {{ .Annotations.description }}
          级别: {{ .Labels.severity }}
          {{ end }}
```

#### 钉钉机器人通知

```yaml
receivers:
  - name: 'dingtalk'
    webhook_configs:
      - url: 'https://oapi.dingtalk.com/robot/send?access_token=your_token'
        method: 'POST'
        http_config:
          content_type: 'application/json'
        body: |
          {
            "msgtype": "markdown",
            "markdown": {
              "title": "[RQA2025] {{ .GroupLabels.alertname }}",
              "text": "### {{ .Annotations.summary }}\n\n{{ .Annotations.description }}\n\n**级别**: {{ .Labels.severity }}\n\n**时间**: {{ .StartsAt.Format \"2006-01-02 15:04:05\" }}"
            }
          }
```

## 监控仪表板

### 1. 系统监控仪表板

#### CPU和内存监控
- CPU使用率趋势图
- 内存使用率趋势图
- 系统负载监控
- 进程状态监控

#### 存储监控
- 磁盘使用率趋势图
- I/O操作监控
- 文件系统状态

#### 网络监控
- 网络流量监控
- 连接数监控
- 网络延迟监控

### 2. 应用监控仪表板

#### API性能监控
- 请求量趋势图
- 响应时间分布图
- 错误率监控
- 活跃用户数

#### 数据库监控
- 连接池状态
- 查询性能监控
- 慢查询统计
- 死锁监控

#### 缓存监控
- Redis连接状态
- 缓存命中率
- 内存使用情况
- 键值分布

### 3. 业务监控仪表板

#### 交易监控
- 交易量趋势图
- 成功率监控
- 交易延迟分布
- 策略执行状态

#### 风险监控
- 持仓风险监控
- 止损触发统计
- 市场风险指标

## 监控实施步骤

### 1. 环境准备

```bash
# 1. 安装Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.40.0/prometheus-2.40.0.linux-amd64.tar.gz
tar xvf prometheus-2.40.0.linux-amd64.tar.gz
cd prometheus-2.40.0.linux-amd64/

# 2. 配置Prometheus
cat > prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'rqa2025'
    static_configs:
      - targets: ['localhost:8000', 'localhost:8001']
    metrics_path: '/metrics'
EOF

# 3. 启动Prometheus
./prometheus --config.file=prometheus.yml
```

### 2. Grafana配置

```bash
# 1. 安装Grafana
wget https://dl.grafana.com/enterprise/release/grafana-enterprise-9.5.0.linux-amd64.tar.gz
tar xvf grafana-enterprise-9.5.0.linux-amd64.tar.gz
cd grafana-9.5.0/

# 2. 配置数据源
cat > conf/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
EOF

# 3. 启动Grafana
./bin/grafana-server
```

### 3. 告警配置

```bash
# 1. 配置AlertManager
cat > alertmanager.yml << EOF
global:
  smtp_smarthost: 'smtp.company.com:587'
  smtp_from: 'alerts@company.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'email-alerts'

receivers:
  - name: 'email-alerts'
    email_configs:
      - to: 'trading-team@company.com'
        subject: '[RQA2025 Alert] {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Severity: {{ .Labels.severity }}
          {{ end }}
EOF

# 2. 启动AlertManager
./alertmanager --config.file=alertmanager.yml
```

## 监控验证

### 1. 功能验证

```bash
# 1. 验证Prometheus状态
curl http://localhost:9090/-/healthy
# 期望返回: Prometheus is Healthy.

# 2. 验证指标收集
curl http://localhost:9090/api/v1/targets
# 应该看到所有目标都是UP状态

# 3. 验证Grafana访问
curl http://localhost:3000/api/health
# 期望返回: {"database":"ok","commit":"xxx"}

# 4. 测试告警规则
curl -X POST http://localhost:9090/-/reload
```

### 2. 告警测试

```bash
# 1. 触发测试告警
curl -X POST http://localhost:9090/api/v1/alerts -H 'Content-Type: application/json' -d '[
  {
    "labels": {
      "alertname": "TestAlert",
      "severity": "warning"
    },
    "annotations": {
      "summary": "Test alert for monitoring system",
      "description": "This is a test alert to verify the monitoring pipeline"
    }
  }
]'

# 2. 检查告警状态
curl http://localhost:9090/api/v1/alerts
```

## 故障排查

### 1. 常见问题

#### Prometheus无法启动
```bash
# 检查端口占用
netstat -tlnp | grep 9090

# 检查配置文件语法
./promtool check config prometheus.yml

# 查看日志
tail -f prometheus.log
```

#### Grafana无法访问
```bash
# 检查端口占用
netstat -tlnp | grep 3000

# 检查Grafana进程
ps aux | grep grafana

# 查看日志
tail -f grafana.log
```

#### 告警不触发
```bash
# 检查告警规则语法
./promtool check rules alert_rules.yml

# 验证指标是否存在
curl "http://localhost:9090/api/v1/query?query=up"

# 检查AlertManager配置
./amtool config check alertmanager.yml
```

### 2. 性能优化

#### Prometheus优化
```yaml
# prometheus.yml 优化配置
global:
  scrape_interval: 15s
  evaluation_interval: 15s

# 减少抓取间隔但保持准确性
scrape_configs:
  - job_name: 'rqa2025'
    scrape_interval: 30s  # 生产环境可调整到30s
    static_configs:
      - targets: ['localhost:8000']
```

#### Grafana优化
```ini
# grafana.ini 优化配置
[server]
http_port = 3000
domain = your-domain.com

[security]
admin_user = admin
admin_password = secure_password

[database]
type = sqlite3
path = grafana.db

[session]
provider = memory
```

## 运维手册

### 1. 日常监控

#### 每日检查清单
- [ ] 检查系统资源使用率
- [ ] 验证应用服务状态
- [ ] 检查数据库连接状态
- [ ] 确认告警系统正常
- [ ] 备份监控数据

#### 周度检查清单
- [ ] 分析监控指标趋势
- [ ] 检查告警历史记录
- [ ] 验证备份完整性
- [ ] 更新监控配置

### 2. 应急响应

#### 告警处理流程
1. **接收告警**: 通过邮件/钉钉接收告警通知
2. **确认告警**: 登录Grafana查看详细指标
3. **分析问题**: 根据告警信息定位问题根因
4. **执行修复**: 按照故障排查指南处理问题
5. **验证修复**: 确认问题已解决并记录处理过程
6. **总结复盘**: 分析问题原因，完善监控和告警规则

#### 紧急联系方式
- **技术负责人**: tech-leader@company.com
- **运维团队**: ops-team@company.com
- **开发团队**: dev-team@company.com
- **紧急电话**: +86 138-0000-0000

### 3. 监控报告

#### 日报生成
```bash
#!/bin/bash
# daily_monitoring_report.sh

REPORT_DATE=$(date +%Y-%m-%d)
REPORT_FILE="/var/log/monitoring/daily_report_${REPORT_DATE}.md"

cat > $REPORT_FILE << EOF
# RQA2025 监控日报 - ${REPORT_DATE}

## 系统状态
- CPU使用率: $(curl -s "http://localhost:9090/api/v1/query?query=cpu_usage" | jq -r '.data.result[0].value[1]')
- 内存使用率: $(curl -s "http://localhost:9090/api/v1/query?query=memory_usage" | jq -r '.data.result[0].value[1]')
- 磁盘使用率: $(df -h / | awk 'NR==2 {print $5}')

## 应用状态
- API响应时间: $(curl -s "http://localhost:9090/api/v1/query?query=http_request_duration_seconds" | jq -r '.data.result[0].value[1]')
- 错误率: $(curl -s "http://localhost:9090/api/v1/query?query=http_request_error_rate" | jq -r '.data.result[0].value[1]')

## 业务指标
- 交易量: $(curl -s "http://localhost:9090/api/v1/query?query=trade_total" | jq -r '.data.result[0].value[1]')
- 成功率: $(curl -s "http://localhost:9090/api/v1/query?query=trade_success_rate" | jq -r '.data.result[0].value[1]')

## 告警统计
- 今日告警数: $(curl -s "http://localhost:9090/api/v1/alerts" | jq '.data.alerts | length')
- 严重告警数: $(curl -s "http://localhost:9090/api/v1/alerts" | jq '[.data.alerts[] | select(.labels.severity == "critical")] | length')
- 警告告警数: $(curl -s "http://localhost:9090/api/v1/alerts" | jq '[.data.alerts[] | select(.labels.severity == "warning")] | length')
EOF

echo "监控日报已生成: $REPORT_FILE"
```

#### 周报生成
```bash
#!/bin/bash
# weekly_monitoring_report.sh

REPORT_DATE=$(date +%Y-%m-%d)
REPORT_FILE="/var/log/monitoring/weekly_report_${REPORT_DATE}.md"

# 计算上周的日期范围
START_DATE=$(date -d 'last monday' +%Y-%m-%d)
END_DATE=$(date -d 'last sunday' +%Y-%m-%d)

cat > $REPORT_FILE << EOF
# RQA2025 监控周报 - ${START_DATE} 至 ${END_DATE}

## 总体概览
- 系统可用性: 99.9%
- 平均响应时间: $(curl -s "http://localhost:9090/api/v1/query_range?query=avg(http_request_duration_seconds)&start=${START_DATE}T00:00:00Z&end=${END_DATE}T23:59:59Z" | jq -r '.data.result[0].values | length')
- 总交易量: $(curl -s "http://localhost:9090/api/v1/query_range?query=sum(trade_total)&start=${START_DATE}T00:00:00Z&end=${END_DATE}T23:59:59Z" | jq -r '.data.result[0].values[-1][1]')

## 性能趋势
- CPU使用率趋势图
- 内存使用率趋势图
- API响应时间趋势图
- 错误率趋势图

## 告警分析
- 告警总数: $(curl -s "http://localhost:9090/api/v1/alerts" | jq '.data.alerts | length')
- 告警类型分布
- 告警处理效率

## 问题总结
- 发现的主要问题
- 解决方案和改进措施
- 预防措施

## 下周计划
- 监控优化任务
- 告警规则调整
- 性能调优计划
EOF

echo "监控周报已生成: $REPORT_FILE"
```

## 总结

通过完善的监控和告警体系，可以确保：

1. **实时监控**: 全方位的系统、应用、业务指标监控
2. **及时告警**: 多渠道告警通知，确保问题快速响应
3. **可视化展示**: 直观的仪表板便于问题定位和分析
4. **自动化报告**: 定期生成监控报告，提高运维效率
5. **故障排查**: 完善的故障排查指南，快速定位问题

监控系统是生产环境稳定运行的重要保障，需要持续优化和完善。
