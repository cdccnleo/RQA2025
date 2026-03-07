# RQA2025 生产环境监控体系搭建指南

## 📊 监控架构概览

生产环境监控采用多层次、全方位的监控策略，确保系统稳定运行和快速问题定位。

### 监控层次
```
应用层监控 (Application Monitoring)
├── 业务指标监控 (Business Metrics)
├── 应用性能监控 (APM)
└── 错误跟踪监控 (Error Tracking)

系统层监控 (System Monitoring)
├── 基础设施监控 (Infrastructure)
├── 容器监控 (Container)
└── 网络监控 (Network)

业务层监控 (Business Monitoring)
├── 用户体验监控 (UX Monitoring)
├── 业务流程监控 (Process Monitoring)
└── SLA监控 (SLA Monitoring)
```

## 🛠️ 监控工具栈

### 核心监控工具
- **Prometheus**: 指标收集和存储
- **Grafana**: 可视化仪表板
- **ELK Stack**: 日志聚合和分析
- **Jaeger**: 分布式链路追踪
- **AlertManager**: 告警管理

### 部署配置
```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  elasticsearch:
    image: elasticsearch:7.10.0
    environment:
      - discovery.type=single-node

  logstash:
    image: logstash:7.10.0
    volumes:
      - ./monitoring/logstash.conf:/usr/share/logstash/pipeline/logstash.conf

  kibana:
    image: kibana:7.10.0
    ports:
      - "5601:5601"
```

## 📈 核心监控指标

### 应用性能指标 (APM)

#### HTTP请求监控
```python
from prometheus_client import Counter, Histogram, Gauge

# 请求计数器
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code']
)

# 请求延迟直方图
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

# 活跃连接数
ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections'
)
```

#### 业务指标监控
```python
# 策略服务指标
STRATEGY_CALCULATION_COUNT = Counter(
    'strategy_calculation_total',
    'Total number of strategy calculations'
)

STRATEGY_CALCULATION_DURATION = Histogram(
    'strategy_calculation_duration_seconds',
    'Strategy calculation duration'
)

# 交易引擎指标
ORDER_EXECUTION_COUNT = Counter(
    'order_execution_total',
    'Total number of order executions',
    ['status']
)

ORDER_EXECUTION_DURATION = Histogram(
    'order_execution_duration_seconds',
    'Order execution duration'
)

# 风险管理指标
RISK_ASSESSMENT_COUNT = Counter(
    'risk_assessment_total',
    'Total number of risk assessments'
)

RISK_VIOLATION_COUNT = Counter(
    'risk_violation_total',
    'Total number of risk violations',
    ['severity']
)
```

### 系统资源监控

#### CPU和内存监控
```bash
# Prometheus Node Exporter配置
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'rqa2025'
    static_configs:
      - targets: ['localhost:8000']
```

#### 磁盘和网络监控
```yaml
# 磁盘使用率告警规则
groups:
  - name: disk_alerts
    rules:
      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Disk space low"
          description: "Disk space is below 10% on {{ $labels.instance }}"
```

## 🎯 告警规则配置

### 业务告警规则
```yaml
# 策略服务告警
groups:
  - name: strategy_alerts
    rules:
      - alert: StrategyCalculationSlow
        expr: histogram_quantile(0.95, rate(strategy_calculation_duration_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Strategy calculation is slow"
          description: "95th percentile of strategy calculation duration > 1s"

      - alert: StrategyCalculationFailed
        expr: rate(strategy_calculation_total{status="error"}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Strategy calculation failure rate high"
          description: "Strategy calculation error rate > 10%"
```

### 系统告警规则
```yaml
# 交易引擎告警
groups:
  - name: trading_alerts
    rules:
      - alert: OrderExecutionFailed
        expr: rate(order_execution_total{status="failed"}[5m]) / rate(order_execution_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Order execution failure rate high"
          description: "Order execution failure rate > 5%"

      - alert: TradingEngineDown
        expr: up{job="rqa2025-trading"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Trading engine is down"
          description: "Trading engine service is not responding"
```

## 📊 Grafana仪表板

### 核心业务仪表板
```json
{
  "dashboard": {
    "title": "RQA2025 - 核心业务监控",
    "tags": ["rqa2025", "business"],
    "panels": [
      {
        "title": "策略服务性能",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(strategy_calculation_total[5m])",
            "legendFormat": "策略计算QPS"
          },
          {
            "expr": "histogram_quantile(0.95, rate(strategy_calculation_duration_seconds_bucket[5m]))",
            "legendFormat": "95%响应时间"
          }
        ]
      },
      {
        "title": "交易引擎状态",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(order_execution_total{status=\"success\"}[5m]) / rate(order_execution_total[5m]) * 100",
            "legendFormat": "成功率"
          }
        ]
      }
    ]
  }
}
```

### 系统资源仪表板
```json
{
  "dashboard": {
    "title": "RQA2025 - 系统资源监控",
    "tags": ["rqa2025", "system"],
    "panels": [
      {
        "title": "CPU使用率",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg by(instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "CPU使用率"
          }
        ]
      },
      {
        "title": "内存使用率",
        "type": "graph",
        "targets": [
          {
            "expr": "(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100",
            "legendFormat": "内存使用率"
          }
        ]
      }
    ]
  }
}
```

## 🔍 日志聚合配置

### Logstash配置
```conf
# logstash.conf
input {
  file {
    path => "/var/log/rqa2025/*.log"
    start_position => "beginning"
  }
}

filter {
  json {
    source => "message"
  }

  date {
    match => ["timestamp", "ISO8601"]
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "rqa2025-%{+YYYY.MM.dd}"
  }
}
```

### 应用日志配置
```python
import logging
import sys
from pythonjsonlogger import jsonlogger

# 配置JSON日志格式
logger = logging.getLogger('rqa2025')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = jsonlogger.JsonFormatter(
    fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# 业务日志记录
logger.info('Strategy calculation completed', extra={
    'strategy_id': 'strat_001',
    'duration_ms': 150,
    'success': True
})
```

## 🚨 告警通知配置

### AlertManager配置
```yaml
# alertmanager.yml
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
  routes:
    - match:
        severity: critical
      receiver: 'critical'

receivers:
  - name: 'team'
    email_configs:
      - to: 'devops@company.com'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/...'

  - name: 'critical'
    email_configs:
      - to: 'oncall@company.com'
    pagerduty_configs:
      - service_key: 'pagerduty_key'
```

## 📋 监控验收清单

### 部署前检查
- [ ] Prometheus配置正确
- [ ] Grafana仪表板创建完成
- [ ] ELK Stack部署成功
- [ ] 告警规则配置完成

### 部署中验证
- [ ] 指标数据正常收集
- [ ] 仪表板显示正确
- [ ] 日志聚合正常工作
- [ ] 告警通知测试成功

### 部署后监控
- [ ] 业务指标监控正常
- [ ] 系统资源监控完整
- [ ] 日志分析功能可用
- [ ] 告警响应及时有效

---

**监控覆盖率**: 100% 核心指标
**告警响应时间**: P0级 <5分钟
**数据保留期**: 30天指标，90天日志
