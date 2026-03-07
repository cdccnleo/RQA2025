# 基础设施层增强实施指南

## 1. 环境准备

### 1.1 硬件要求
| 组件              | 最小配置                     | 生产环境推荐           |
|-------------------|-----------------------------|-----------------------|
| 控制节点          | 4核CPU/8GB内存/100GB存储    | 8核CPU/32GB内存/500GB |
| 监控节点          | 4核CPU/16GB内存/200GB存储   | 16核CPU/64GB内存/1TB  |
| 灾备节点          | 同生产环境配置               | 同生产环境+热备       |

### 1.2 软件依赖
```bash
# 基础依赖安装
sudo apt-get install -y \
    docker-ce \
    python3-pip \
    prometheus \
    grafana

# Python库安装
pip install \
    scikit-learn==1.2.2 \
    prometheus-client==0.16.0 \
    hashicorp-vault==1.0.0
```

## 2. 智能容灾系统部署

### 2.1 多中心配置
```yaml
# config/disaster_recovery.yaml
centers:
  primary:
    endpoint: "https://primary.example.com"
    weight: 100
  backup1:
    endpoint: "https://backup1.example.com" 
    weight: 80
  backup2:
    endpoint: "https://backup2.example.com"
    weight: 60

sync:
  interval: "5s"
  timeout: "10s"
```

### 2.2 部署步骤
1. 初始化状态同步服务
```bash
./disaster_recovery init \
    --config=config/disaster_recovery.yaml \
    --storage=etcd://localhost:2379
```

2. 启动健康监测
```bash
./disaster_recovery monitor \
    --interval=5s \
    --threshold=3
```

3. 验证容灾切换
```bash
# 模拟主中心故障
curl -X POST http://localhost:8080/failover/test

# 检查切换状态
curl http://localhost:8080/status
```

## 3. 智能监控体系配置

### 3.1 Prometheus配置示例
```yaml
# prometheus/config.yml
scrape_configs:
  - job_name: 'infra_metrics'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
    
  - job_name: 'business_metrics'
    scrape_interval: 30s
    metrics_path: '/metrics'
    static_configs:
      - targets: ['app-server:8080']
```

### 3.2 异常检测规则
```yaml
# monitoring/alerts.yml
groups:
- name: infrastructure
  rules:
  - alert: HighErrorRate
    expr: rate(infra_errors_total[5m]) > 10
    for: 10m
    labels:
      severity: 'critical'
    annotations:
      summary: "High error rate detected"
```

### 3.3 部署验证
```bash
# 启动Prometheus
prometheus --config.file=./config.yml

# 验证指标采集
curl http://localhost:9090/api/v1/targets
```

## 4. 自适应资源管理

### 4.1 动态策略配置
```json
{
  "strategies": {
    "high_frequency": {
      "base_cpu": 30,
      "base_memory": "4Gi",
      "volatility_factor": 0.5,
      "liquidity_factor": 0.3
    }
  }
}
```

### 4.2 资源调节器部署
```bash
# 启动资源管理器
./resource_manager start \
    --config=config/resource_policy.json \
    --market-data-source=kafka://market-feed

# 验证调节效果
watch -n 1 "curl http://localhost:8080/quota/high_frequency"
```

## 5. 安全架构实施

### 5.1 RBAC策略定义
```yaml
# security/roles.yml
roles:
  admin:
    permissions:
      - "config:*"
      - "security:*"
  operator:
    permissions:
      - "monitoring:*"
      - "log:read"
```

### 5.2 密钥轮换流程
```mermaid
sequenceDiagram
    participant A as KeyManager
    participant B as Vault
    participant C as Applications
    
    A->>B: 生成新密钥
    B->>C: 通知密钥更新
    C->>B: 获取新密钥
    B->>A: 确认轮换完成
```

## 6. 混沌工程实施

### 6.1 实验定义
```yaml
# chaos/network_failure.yml
experiment:
  name: "network-partition"
  steps:
    - action: "network-loss"
      target: "payment-service"
      params:
        latency: "500ms"
        duration: "2m"
    - action: "monitor"
      metric: "transaction_success_rate"
      threshold: "95%"
```

### 6.2 执行验证
```bash
# 运行实验
chaos run chaos/network_failure.yml

# 生成报告
chaos report --export=html
```

## 7. 验证与验收

### 7.1 验收测试用例
| 测试项           | 方法                      | 预期结果              |
|------------------|--------------------------|----------------------|
| 容灾自动切换     | 停止主节点服务            | 30秒内完成切换       |
| 监控指标采集     | 生成测试负载              | 所有指标正常上报     |
| 资源动态调整     | 模拟市场波动              | CPU配额自动增减±20%  |
| 安全审计         | 尝试越权操作              | 操作被拒绝并记录日志 |

### 7.2 性能基准
```bash
# 运行压力测试
./load_test \
    --threads=100 \
    --duration=1h \
    --scenario=full

# 生成性能报告
python analyze_results.py load_test.log
```

## 8. 运维管理

### 8.1 日常维护命令
```bash
# 检查系统状态
./infra_tool status --all

# 执行密钥轮换
./security_tool rotate_keys --service=database

# 清理监控数据
./monitoring_tool purge --older-than=30d
```

### 8.2 故障排查指南
1. 容灾切换失败：
   - 检查网络连通性：`ping backup1.example.com`
   - 验证状态同步：`etcdctl get /infra/sync/last`
   
2. 监控数据缺失：
   - 检查Prometheus目标：`curl http://localhost:9090/targets`
   - 验证指标暴露端点：`curl http://app-server:8080/metrics`
```

# MiniQMT监控指标集成

MiniQMTAdapter采集并上报如下监控指标，建议对接Prometheus等监控系统：

- miniqmt_connection_status (Gauge): 连接状态，标签type区分数据/交易通道
- miniqmt_data_latency (Histogram): 数据延迟，标签data_type区分行情/订单等
- miniqmt_order_count (Counter): 订单数量，标签order_type区分买/卖/撤单等
- miniqmt_cache_hit_rate (Gauge): 缓存命中率
- miniqmt_reconnects (Counter): 重连次数

## 采集方式
在MiniQMTAdapter各操作点调用metrics接口实时采集并上报。

## 告警建议
- 连接状态异常（断开超30秒）触发告警
- 数据延迟超过阈值（如1秒）触发告警
- 重连次数异常增长触发运维通知
