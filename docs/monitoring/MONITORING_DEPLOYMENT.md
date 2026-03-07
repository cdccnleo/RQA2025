# RQA2025 监控系统部署指南

## 📋 监控概述

**版本**：V1.0
**更新日期**：2024年12月
**监控目标**：全栈监控、实时告警、智能分析
**覆盖范围**：应用、服务、基础设施、业务指标

---

## 🏗️ 监控架构

### 1. 监控层次架构

```
RQA2025监控体系架构
==============================================
┌─────────────────────────────────────────┐
│              业务监控层                  │
│  ┌─────────────────────────────────┐   │
│  │     交易量、成功率、收益等     │   │
│  └─────────────────────────────────┘   │
├─────────────────────────────────────────┤
│              应用监控层                  │
│  ┌─────────────────────────────────┐   │
│  │   API响应时间、错误率、QPS     │   │
│  └─────────────────────────────────┘   │
├─────────────────────────────────────────┤
│              服务监控层                  │
│  ┌─────────────────────────────────┐   │
│  │   微服务健康、依赖调用、队列   │   │
│  └─────────────────────────────────┘   │
├─────────────────────────────────────────┤
│              系统监控层                  │
│  ┌─────────────────────────────────┐   │
│  │     CPU、内存、磁盘、网络      │   │
│  └─────────────────────────────────┘   │
├─────────────────────────────────────────┤
│              数据收集层                  │
│  ┌─────────────────────────────────┐   │
│  │   Prometheus + Node Exporter   │   │
│  │   Fluent Bit + Elasticsearch   │   │
│  │   Jaeger + OpenTelemetry       │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### 2. 监控组件

#### 核心监控组件
- **Prometheus**：指标收集和存储
- **Grafana**：可视化监控面板
- **AlertManager**：告警管理和路由
- **Node Exporter**：节点资源监控
- **cAdvisor**：容器资源监控

#### 日志监控组件
- **Fluent Bit**：日志收集器
- **Elasticsearch**：日志存储和搜索
- **Kibana**：日志可视化分析

#### 链路追踪组件
- **Jaeger**：分布式链路追踪
- **OpenTelemetry**：观测数据收集

---

## 📊 指标体系

### 1. 业务指标

#### 核心业务指标

| 指标名称 | 指标类型 | 采集频率 | 告警阈值 | 说明 |
|---------|---------|---------|---------|------|
| **交易成功率** | 百分比 | 1分钟 | <99.5% | 交易执行成功率 |
| **订单响应时间** | 毫秒 | 1分钟 | >1000ms | 订单处理响应时间 |
| **策略执行频率** | 次/秒 | 1分钟 | <10 | 策略执行频率 |
| **风控触发频率** | 次/分钟 | 1分钟 | >50 | 风险控制触发频率 |
| **用户活跃度** | 在线用户数 | 5分钟 | - | 当前在线用户数量 |

#### 量化交易指标

```prometheus
# 交易成功率
rate(rqa2025_trades_total{status="success"}[5m])
/
rate(rqa2025_trades_total[5m])

# 订单响应时间分位数
histogram_quantile(0.95,
  rate(rqa2025_order_duration_bucket[5m]))

# 策略收益指标
rqa2025_strategy_pnl_total
```

### 2. 应用性能指标

#### API性能指标

| 指标名称 | 采集方式 | 告警阈值 | 说明 |
|---------|---------|---------|------|
| **请求QPS** | Prometheus | >1000 | 每秒请求数 |
| **响应时间P95** | Prometheus | >1000ms | 95%请求响应时间 |
| **错误率** | Prometheus | >1% | 请求错误率 |
| **活跃连接数** | Prometheus | >1000 | 并发连接数 |
| **内存使用率** | Prometheus | >85% | 应用内存使用 |

#### 数据库性能指标

```prometheus
# 数据库连接池使用率
rqa2025_db_connections_active
/ rqa2025_db_connections_max

# 查询响应时间
histogram_quantile(0.95,
  rate(rqa2025_db_query_duration_bucket[5m]))

# 慢查询数量
increase(rqa2025_db_slow_queries_total[5m])
```

### 3. 系统资源指标

#### 计算资源指标

| 指标名称 | 采集工具 | 告警阈值 | 说明 |
|---------|---------|---------|------|
| **CPU使用率** | Node Exporter | >80% | 节点CPU使用率 |
| **内存使用率** | Node Exporter | >85% | 节点内存使用率 |
| **磁盘使用率** | Node Exporter | >85% | 节点磁盘使用率 |
| **网络带宽** | Node Exporter | >80% | 网络带宽使用率 |
| **磁盘IOPS** | Node Exporter | >1000 | 磁盘I/O操作 |

#### 容器资源指标

```prometheus
# 容器CPU使用率
rate(container_cpu_usage_seconds_total{pod=~"rqa2025-.*"}[5m])
/ container_spec_cpu_quota{pod=~"rqa2025-.*"} / 100000

# 容器内存使用率
container_memory_usage_bytes{pod=~"rqa2025-.*"}
/ container_spec_memory_limit_bytes{pod=~"rqa2025-.*"}
```

---

## 🚀 部署步骤

### 1. 创建监控命名空间

```bash
# 创建监控命名空间
kubectl create namespace monitoring

# 设置命名空间标签
kubectl label namespace monitoring name=monitoring
```

### 2. 部署Prometheus

#### Prometheus配置

```yaml
# Prometheus ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s

    rule_files:
      - /etc/prometheus/rules/*.yml

    alerting:
      alertmanagers:
      - static_configs:
        - targets:
          - alertmanager.monitoring.svc.cluster.local:9093

    scrape_configs:
    - job_name: 'kubernetes-apiservers'
      kubernetes_sd_configs:
      - role: endpoints
      scheme: https
      tls_config:
        ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        insecure_skip_verify: true
      bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
      relabel_configs:
      - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
        action: keep
        regex: default;kubernetes;https

    - job_name: 'kubernetes-nodes'
      kubernetes_sd_configs:
      - role: node
      relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
      - target_label: __address__
        replacement: kubernetes.default.svc.cluster.local:443
      - source_labels: [__meta_kubernetes_node_name]
        regex: (.+)
        target_label: __metrics_path__
        replacement: /api/v1/nodes/$1/proxy/metrics

    - job_name: 'kubernetes-pods'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: kubernetes_pod_name

    - job_name: 'rqa2025-services'
      kubernetes_sd_configs:
      - role: pod
      namespaces:
        names:
        - rqa2025
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: rqa2025-.*
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: instance
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: namespace
```

#### Prometheus部署

```yaml
# Prometheus StatefulSet
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: prometheus
  namespace: monitoring
spec:
  serviceName: prometheus
  replicas: 2
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      serviceAccountName: prometheus
      containers:
      - name: prometheus
        image: prom/prometheus:v2.45.0
        ports:
        - containerPort: 9090
          name: http
        args:
        - '--config.file=/etc/prometheus/prometheus.yml'
        - '--storage.tsdb.path=/prometheus'
        - '--storage.tsdb.retention.time=30d'
        - '--web.console.libraries=/etc/prometheus/console_libraries'
        - '--web.console.templates=/etc/prometheus/consoles'
        - '--web.enable-lifecycle'
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
        - name: data
          mountPath: /prometheus
        resources:
          requests:
            cpu: 1
            memory: 2Gi
          limits:
            cpu: 2
            memory: 4Gi
        readinessProbe:
          httpGet:
            path: /-/ready
            port: 9090
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /-/healthy
            port: 9090
          initialDelaySeconds: 30
          periodSeconds: 30
      volumes:
      - name: config
        configMap:
          name: prometheus-config
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
```

### 3. 部署Grafana

#### Grafana配置

```yaml
# Grafana部署
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:10.1.0
        ports:
        - containerPort: 3000
          name: http
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          valueFrom:
            secretKeyRef:
              name: grafana-secret
              key: admin-password
        - name: GF_USERS_ALLOW_SIGN_UP
          value: "false"
        volumeMounts:
        - name: data
          mountPath: /var/lib/grafana
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1
            memory: 2Gi
        readinessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 60
          periodSeconds: 30
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: grafana-data
```

#### Grafana持久化存储

```yaml
# Grafana PVC
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grafana-data
  namespace: monitoring
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 50Gi
```

### 4. 部署AlertManager

#### AlertManager配置

```yaml
# AlertManager配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: monitoring
data:
  config.yml: |
    global:
      smtp_smarthost: 'smtp.gmail.com:587'
      smtp_from: 'alerts@rqa2025.com'
      smtp_auth_username: 'alerts@rqa2025.com'
      smtp_auth_password: 'your-smtp-password'

    route:
      group_by: ['alertname']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 1h
      receiver: 'rqa2025-team'
      routes:
      - match:
          severity: critical
        receiver: 'rqa2025-critical'
      - match:
          severity: warning
        receiver: 'rqa2025-warning'

    receivers:
    - name: 'rqa2025-critical'
      email_configs:
      - to: 'emergency@rqa2025.com'
        subject: 'RQA2025 紧急告警: {{ .GroupLabels.alertname }}'
        body: |
          告警级别: CRITICAL
          告警名称: {{ .GroupLabels.alertname }}
          告警详情: {{ .Annotations.description }}
          实例信息: {{ .Labels.instance }}
          开始时间: {{ .StartsAt }}
    - name: 'rqa2025-warning'
      email_configs:
      - to: 'alerts@rqa2025.com'
        subject: 'RQA2025 告警: {{ .GroupLabels.alertname }}'
    - name: 'rqa2025-team'
      slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#rqa2025-alerts'
        title: 'RQA2025告警'
        text: |
          告警: {{ .GroupLabels.alertname }}
          详情: {{ .Annotations.description }}
          实例: {{ .Labels.instance }}
```

#### AlertManager部署

```yaml
# AlertManager部署
apiVersion: apps/v1
kind: Deployment
metadata:
  name: alertmanager
  namespace: monitoring
spec:
  replicas: 2
  selector:
    matchLabels:
      app: alertmanager
  template:
    metadata:
      labels:
        app: alertmanager
    spec:
      containers:
      - name: alertmanager
        image: prom/alertmanager:v0.26.0
        ports:
        - containerPort: 9093
        args:
        - '--config.file=/etc/alertmanager/config.yml'
        - '--storage.path=/alertmanager'
        volumeMounts:
        - name: config
          mountPath: /etc/alertmanager
        - name: data
          mountPath: /alertmanager
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 200m
            memory: 512Mi
      volumes:
      - name: config
        configMap:
          name: alertmanager-config
      - name: data
        emptyDir: {}
```

### 5. 部署日志系统

#### Elasticsearch部署

```yaml
# Elasticsearch StatefulSet
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: elasticsearch
  namespace: monitoring
spec:
  serviceName: elasticsearch
  replicas: 3
  selector:
    matchLabels:
      app: elasticsearch
  template:
    metadata:
      labels:
        app: elasticsearch
    spec:
      containers:
      - name: elasticsearch
        image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
        ports:
        - containerPort: 9200
          name: http
        - containerPort: 9300
          name: transport
        env:
        - name: discovery.seed_hosts
          value: elasticsearch-0.elasticsearch,elasticsearch-1.elasticsearch,elasticsearch-2.elasticsearch
        - name: cluster.initial_master_nodes
          value: elasticsearch-0,elasticsearch-1,elasticsearch-2
        - name: node.name
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: cluster.name
          value: rqa2025-monitoring
        volumeMounts:
        - name: data
          mountPath: /usr/share/elasticsearch/data
        resources:
          requests:
            cpu: 1
            memory: 2Gi
          limits:
            cpu: 2
            memory: 4Gi
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
```

#### Fluent Bit部署

```yaml
# Fluent Bit配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
  namespace: monitoring
data:
  fluent-bit.conf: |
    [SERVICE]
        Flush         5
        Log_Level     info
        Daemon        off

    [INPUT]
        Name              tail
        Path              /var/log/containers/*rqa2025*.log
        Parser            docker
        Tag               kube.*
        Refresh_Interval  5

    [INPUT]
        Name              systemd
        Tag               host.*
        Systemd_Filter    _SYSTEMD_UNIT=kubelet.service

    [FILTER]
        Name                kubernetes
        Match               kube.*
        Kube_URL            https://kubernetes.default.svc.cluster.local:443
        Kube_CA_File        /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        Kube_Token_File     /var/run/secrets/kubernetes.io/serviceaccount/token
        Kube_Tag_Prefix     kube.var.log.containers.
        Merge_Log           On
        Merge_Log_Key       log_processed
        K8S-Logging.Parser  On
        K8S-Logging.Exclude On

    [OUTPUT]
        Name  es
        Match kube.*
        Host  elasticsearch
        Port  9200
        Index kube
        Type  _doc

    [OUTPUT]
        Name  stdout
        Match host.*
```

---

## 📊 监控面板配置

### 1. 核心业务监控面板

#### 交易监控面板

```json
{
  "dashboard": {
    "title": "RQA2025 交易监控",
    "tags": ["rqa2025", "trading"],
    "panels": [
      {
        "title": "交易成功率",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(rqa2025_trades_total{status=\"success\"}[5m]) / rate(rqa2025_trades_total[5m]) * 100",
            "legendFormat": "成功率"
          }
        ]
      },
      {
        "title": "订单响应时间",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(rqa2025_order_duration_bucket[5m]))",
            "legendFormat": "P95响应时间"
          }
        ]
      },
      {
        "title": "策略执行状态",
        "type": "bargauge",
        "targets": [
          {
            "expr": "rqa2025_strategy_status",
            "legendFormat": "{{strategy}}"
          }
        ]
      }
    ]
  }
}
```

### 2. 系统性能监控面板

#### 资源使用监控

```json
{
  "dashboard": {
    "title": "RQA2025 系统资源监控",
    "panels": [
      {
        "title": "CPU使用率",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg by (instance) (rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "title": "内存使用率",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - ((node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100)",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "title": "磁盘使用率",
        "type": "bargauge",
        "targets": [
          {
            "expr": "100 - ((node_filesystem_avail_bytes{mountpoint=\"/\"} / node_filesystem_size_bytes{mountpoint=\"/\"}) * 100)",
            "legendFormat": "{{instance}}"
          }
        ]
      }
    ]
  }
}
```

### 3. 应用性能监控面板

#### API性能监控

```json
{
  "dashboard": {
    "title": "RQA2025 API性能监控",
    "panels": [
      {
        "title": "请求QPS",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "QPS"
          }
        ]
      },
      {
        "title": "响应时间分位数",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.5, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P99"
          }
        ]
      },
      {
        "title": "错误率",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m]) * 100",
            "legendFormat": "错误率"
          }
        ]
      }
    ]
  }
}
```

---

## 🚨 告警规则配置

### 1. 业务告警规则

```yaml
# 业务告警规则
groups:
- name: rqa2025-business-alerts
  rules:
  - alert: TradingSuccessRateLow
    expr: rate(rqa2025_trades_total{status="success"}[5m]) / rate(rqa2025_trades_total[5m]) * 100 < 99.5
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "交易成功率过低"
      description: "交易成功率低于99.5%，当前值: {{ $value }}%"

  - alert: HighOrderLatency
    expr: histogram_quantile(0.95, rate(rqa2025_order_duration_bucket[5m])) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "订单处理延迟高"
      description: "95%订单处理时间超过1秒，当前值: {{ $value }}秒"

  - alert: StrategyExecutionFailed
    expr: rqa2025_strategy_status == 0
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "策略执行失败"
      description: "策略 {{ $labels.strategy }} 执行失败"
```

### 2. 系统告警规则

```yaml
# 系统告警规则
groups:
- name: rqa2025-system-alerts
  rules:
  - alert: HighCPUUsage
    expr: 100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "CPU使用率过高"
      description: "节点 {{ $labels.instance }} CPU使用率超过80%，当前值: {{ $value }}%"

  - alert: HighMemoryUsage
    expr: 100 - ((node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100) > 85
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "内存使用率过高"
      description: "节点 {{ $labels.instance }} 内存使用率超过85%，当前值: {{ $value }}%"

  - alert: DiskSpaceLow
    expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"} * 100) < 15
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "磁盘空间不足"
      description: "节点 {{ $labels.instance }} 根分区剩余空间不足15%"
```

### 3. 服务告警规则

```yaml
# 服务告警规则
groups:
- name: rqa2025-service-alerts
  rules:
  - alert: ServiceDown
    expr: up{job="rqa2025-services"} == 0
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "服务不可用"
      description: "服务 {{ $labels.pod }} 已停止运行超过2分钟"

  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m]) * 100 > 5
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "服务错误率过高"
      description: "服务 {{ $labels.service }} 错误率超过5%，当前值: {{ $value }}%"

  - alert: SlowResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "响应时间过慢"
      description: "服务 {{ $labels.service }} 95%响应时间超过2秒"
```

---

## 📋 部署验证

### 1. 监控系统验证

```bash
# 检查监控组件状态
kubectl get pods -n monitoring

# 验证Prometheus状态
curl http://prometheus.monitoring.svc.cluster.local:9090/-/ready

# 验证Grafana状态
curl http://grafana.monitoring.svc.cluster.local:3000/api/health

# 检查指标收集
curl http://prometheus.monitoring.svc.cluster.local:9090/api/v1/targets
```

### 2. 告警系统验证

```bash
# 检查告警规则
curl http://prometheus.monitoring.svc.cluster.local:9090/api/v1/rules

# 验证告警路由
curl http://alertmanager.monitoring.svc.cluster.local:9093/api/v2/status

# 测试告警触发
kubectl exec -it deployment/rqa2025-core -n rqa2025 -- curl -X POST http://localhost:8080/test-alert
```

### 3. 日志系统验证

```bash
# 检查日志收集
curl http://elasticsearch.monitoring.svc.cluster.local:9200/_cluster/health

# 验证日志索引
curl http://elasticsearch.monitoring.svc.cluster.local:9200/_cat/indices

# 检查日志流
kubectl logs -f deployment/fluent-bit -n monitoring
```

---

## 📊 监控指标定义

### 1. 应用指标

```python
# Python应用指标定义
from prometheus_client import Counter, Histogram, Gauge

# 交易指标
trades_total = Counter('rqa2025_trades_total', 'Total number of trades', ['status', 'symbol'])
order_duration = Histogram('rqa2025_order_duration', 'Order processing duration', ['strategy'])
strategy_status = Gauge('rqa2025_strategy_status', 'Strategy execution status', ['strategy'])

# 业务指标
active_users = Gauge('rqa2025_active_users', 'Number of active users')
portfolio_value = Gauge('rqa2025_portfolio_value', 'Total portfolio value')
strategy_pnl = Counter('rqa2025_strategy_pnl_total', 'Strategy profit and loss', ['strategy'])

# 系统指标
memory_usage = Gauge('rqa2025_memory_usage', 'Memory usage percentage')
cpu_usage = Gauge('rqa2025_cpu_usage', 'CPU usage percentage')
disk_usage = Gauge('rqa2025_disk_usage', 'Disk usage percentage')
```

### 2. 基础设施指标

```yaml
# Kubernetes指标
- node_cpu_usage_seconds_total
- node_memory_MemTotal_bytes
- node_memory_MemAvailable_bytes
- node_filesystem_size_bytes
- node_filesystem_avail_bytes
- node_network_receive_bytes_total
- node_network_transmit_bytes_total

# 容器指标
- container_cpu_usage_seconds_total
- container_memory_usage_bytes
- container_spec_cpu_quota
- container_spec_memory_limit_bytes
- container_network_receive_bytes_total
- container_network_transmit_bytes_total
```

---

## 📞 联系支持

### 监控支持
- **监控团队**：monitoring@rqa2025.com
- **技术支持**：tech-support@rqa2025.com
- **紧急联系**：emergency@rqa2025.com

### 相关文档
- [系统架构文档](../../ARCHITECTURE.md)
- [运维手册](../operations/OPERATIONS_MANUAL.md)
- [应急响应手册](../operations/EMERGENCY_RESPONSE_MANUAL.md)
- [Kubernetes部署指南](../deployment/KUBERNETES_DEPLOYMENT.md)

---

**文档维护人**：RQA2025监控团队
**最后更新**：2024年12月
**版本**：V1.0

*监控系统的稳定运行对量化交易系统至关重要，请严格按照本指南执行部署和配置*
