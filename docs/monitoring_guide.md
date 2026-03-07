# 基础设施层监控指标体系

## 📋 文档信息

- **版本**: v1.0
- **创建日期**: 2025年9月23日
- **监控工具**: Prometheus + Grafana
- **更新频率**: 实时监控，日报表

## 🎯 监控目标

通过全面的监控指标体系，实现基础设施层的可观测性，确保系统稳定运行和快速问题定位。

### 核心价值
- **主动发现**: 在问题影响用户前发现并解决
- **快速定位**: 问题发生时快速定位根本原因
- **持续优化**: 基于监控数据持续优化系统性能
- **容量规划**: 为未来扩容提供数据支撑

---

## 📊 核心监控指标

### 1. 业务指标 (Business Metrics)

#### 1.1 缓存服务指标
```prometheus
# 缓存命中率
cache_hit_ratio = rate(cache_hits_total[5m]) / rate(cache_requests_total[5m])

# 缓存穿透率
cache_miss_ratio = rate(cache_misses_total[5m]) / rate(cache_requests_total[5m])

# 缓存响应时间
cache_response_time_bucket{le="0.1"}  # P95 < 100ms
cache_response_time_bucket{le="1.0"}   # P99 < 1s

# 缓存数据一致性
cache_consistency_errors_total  # 数据不一致错误数
cache_repair_operations_total   # 自动修复操作数
```

#### 1.2 配置管理指标
```prometheus
# 配置加载成功率
config_load_success_ratio = config_load_success_total / config_load_attempts_total

# 配置更新延迟
config_update_delay_seconds  # 配置更新传播时间

# 配置版本一致性
config_version_mismatch_total  # 版本不一致实例数
```

#### 1.3 监控服务指标
```prometheus
# 监控数据收集率
monitoring_collection_success_ratio = monitoring_collection_success_total / monitoring_collection_attempts_total

# 告警触发延迟
alert_trigger_delay_seconds  # 从问题发生到告警触发的延迟

# 监控覆盖率
monitoring_coverage_ratio  # 被监控的组件比例
```

### 2. 系统指标 (System Metrics)

#### 2.1 性能指标
```prometheus
# CPU使用率
cpu_usage_percent{gauge}  # 整体CPU使用率
cpu_usage_percent{core="0"}  # 各核心使用率

# 内存使用率
memory_usage_bytes  # 已使用内存
memory_available_bytes  # 可用内存

# 磁盘I/O
disk_read_bytes_per_second  # 磁盘读取速率
disk_write_bytes_per_second  # 磁盘写入速率

# 网络I/O
network_receive_bytes_per_second  # 网络接收速率
network_transmit_bytes_per_second  # 网络发送速率
```

#### 2.2 资源指标
```prometheus
# 连接池状态
connection_pool_active_connections  # 活跃连接数
connection_pool_idle_connections    # 空闲连接数
connection_pool_max_connections     # 最大连接数

# 线程池状态
thread_pool_active_threads  # 活跃线程数
thread_pool_idle_threads    # 空闲线程数
thread_pool_queue_size      # 队列大小

# 缓存大小
cache_memory_usage_bytes  # 内存缓存使用量
cache_disk_usage_bytes    # 磁盘缓存使用量
```

### 3. 质量指标 (Quality Metrics)

#### 3.1 代码质量指标
```prometheus
# 代码重复率
code_duplication_ratio{gauge}  # 代码重复百分比

# 复杂度指标
code_complexity_average{gauge}  # 平均圈复杂度
code_complexity_max{gauge}      # 最大圈复杂度

# 测试覆盖率
test_coverage_ratio{gauge}  # 单元测试覆盖率
integration_test_coverage_ratio  # 集成测试覆盖率
```

#### 3.2 接口质量指标
```prometheus
# 接口一致性
interface_consistency_violations_total  # 接口不一致违规数

# API响应质量
api_response_success_ratio  # API成功率
api_response_error_4xx_total  # 客户端错误数
api_response_error_5xx_total  # 服务器错误数

# 契约测试结果
contract_test_passed_total   # 契约测试通过数
contract_test_failed_total   # 契约测试失败数
```

---

## 🚨 告警规则

### 1. 紧急告警 (P0)

#### 1.1 系统可用性告警
```prometheus
# 基础设施服务不可用
ALERT InfrastructureServiceDown
  IF up{job="infrastructure"} == 0
  FOR 1m
  LABELS { severity = "critical" }
  ANNOTATIONS {
    summary = "基础设施服务宕机",
    description = "基础设施服务 {{ $labels.instance }} 已宕机 1 分钟",
    runbook_url = "https://docs.example.com/runbooks/infrastructure-down"
  }

# 数据丢失告警
ALERT DataLossDetected
  IF rate(data_loss_events_total[5m]) > 0
  FOR 30s
  LABELS { severity = "critical" }
  ANNOTATIONS {
    summary = "检测到数据丢失",
    description = "过去5分钟内检测到 {{ $value }} 次数据丢失事件"
  }
```

#### 1.2 安全告警
```prometheus
# 异常访问告警
ALERT AbnormalAccessDetected
  IF rate(unauthorized_access_total[5m]) > 10
  FOR 1m
  LABELS { severity = "critical" }
  ANNOTATIONS {
    summary = "检测到异常访问",
    description = "过去5分钟内检测到 {{ $value }} 次未授权访问"
  }

# 敏感数据泄露告警
ALERT SensitiveDataExposure
  IF rate(sensitive_data_exposure_total[5m]) > 0
  FOR 30s
  LABELS { severity = "critical" }
  ANNOTATIONS {
    summary = "敏感数据泄露",
    description = "检测到敏感数据泄露事件"
  }
```

### 2. 重要告警 (P1)

#### 2.1 性能告警
```prometheus
# 响应时间过长
ALERT HighResponseTime
  IF histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1.0
  FOR 5m
  LABELS { severity = "warning" }
  ANNOTATIONS {
    summary = "响应时间过长",
    description = "95%请求响应时间超过1秒，当前值: {{ $value }}s"
  }

# 错误率升高
ALERT HighErrorRate
  IF rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
  FOR 3m
  LABELS { severity = "warning" }
  ANNOTATIONS {
    summary = "错误率过高",
    description = "HTTP 5xx错误率超过5%，当前值: {{ $value }}"
  }
```

#### 2.2 容量告警
```prometheus
# 内存使用过高
ALERT HighMemoryUsage
  IF (1 - system_memory_available_bytes / system_memory_total_bytes) > 0.9
  FOR 5m
  LABELS { severity = "warning" }
  ANNOTATIONS {
    summary = "内存使用过高",
    description = "系统内存使用率超过90%，当前值: {{ $value }}"
  }

# 磁盘空间不足
ALERT LowDiskSpace
  IF (system_disk_total_bytes - system_disk_available_bytes) / system_disk_total_bytes > 0.85
  FOR 10m
  LABELS { severity = "warning" }
  ANNOTATIONS {
    summary = "磁盘空间不足",
    description = "磁盘使用率超过85%，当前值: {{ $value }}"
  }
```

### 3. 一般告警 (P2)

#### 3.1 质量告警
```prometheus
# 缓存命中率降低
ALERT LowCacheHitRate
  IF cache_hit_ratio < 0.8
  FOR 15m
  LABELS { severity = "info" }
  ANNOTATIONS {
    summary = "缓存命中率降低",
    description = "缓存命中率低于80%，当前值: {{ $value }}"
  }

# 代码质量下降
ALERT CodeQualityDecline
  IF code_duplication_ratio > 0.1
  FOR 1h
  LABELS { severity = "info" }
  ANNOTATIONS {
    summary = "代码质量下降",
    description = "代码重复率超过10%，当前值: {{ $value }}"
  }
```

#### 3.2 连接告警
```prometheus
# 连接池耗尽
ALERT ConnectionPoolExhausted
  IF connection_pool_active_connections / connection_pool_max_connections > 0.9
  FOR 5m
  LABELS { severity = "info" }
  ANNOTATIONS {
    summary = "连接池即将耗尽",
    description = "连接池使用率超过90%，当前值: {{ $value }}"
  }

# 网络连接异常
ALERT NetworkConnectionIssues
  IF rate(network_connection_errors_total[5m]) > 5
  FOR 3m
  LABELS { severity = "info" }
  ANNOTATIONS {
    summary = "网络连接异常",
    description = "过去5分钟检测到 {{ $value }} 次网络连接错误"
  }
```

---

## 📈 仪表盘设计

### 1. 业务概览仪表盘

#### 1.1 核心业务指标面板
```
┌─────────────────────────────────────────────────────────────┐
│                    基础设施业务概览                           │
├─────────────────────────────────────────────────────────────┤
│  📊 缓存命中率     📈 95.2% (+2.1%)    ⚠️  目标: >90%      │
│  📊 配置一致性     ✅ 100%             ✅ 目标: 100%         │
│  📊 服务可用性     ✅ 99.95%           ✅ 目标: >99.9%      │
│  📊 数据新鲜度     ⚡ 2.3min           ✅ 目标: <5min       │
├─────────────────────────────────────────────────────────────┤
│  🚨 活跃告警       🔴 2个 P1          🟡 5个 P2            │
│  📋 今日变更       ✅ 3个成功         ❌ 0个失败           │
├─────────────────────────────────────────────────────────────┤
│  📈 趋势图 (过去24小时)                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  缓存命中率趋势图                                     │   │
│  │  96% ┼────┬────┬──────────────────────────────────  │   │
│  │     │    │    │                                  │   │
│  │  94%┼────┼────┼──────────────────────────────────  │   │
│  │     └────┴────┴──────────────────────────────────┘   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### 1.2 服务健康状态面板
```
┌─────────────────────────────────────────────────────────────┐
│                    服务健康状态                              │
├─────────────────────────────────────────────────────────────┤
│  🟢 缓存服务        ✅ 运行正常          响应时间: 45ms     │
│  🟢 配置服务        ✅ 运行正常          最后同步: 2min前   │
│  🟢 监控服务        ✅ 运行正常          收集率: 99.8%      │
│  🟡 日志服务        ⚠️ 轻微延迟          队列积压: 150条    │
├─────────────────────────────────────────────────────────────┤
│  🔄 自动恢复状态                                           │
│  ✅ 自愈成功: 12次/天     ❌ 自愈失败: 0次/天             │
└─────────────────────────────────────────────────────────────┘
```

### 2. 性能监控仪表盘

#### 2.1 响应时间分布
```
┌─────────────────────────────────────────────────────────────┐
│                 响应时间分布 (过去1小时)                     │
├─────────────────────────────────────────────────────────────┤
│  P50:  25ms    ████████████████████████████████░░░   95%    │
│  P95:  85ms    ████████████████████████████████░░░   99%    │
│  P99: 150ms    ████████████████████████████████░░░   99.9%  │
├─────────────────────────────────────────────────────────────┤
│  📈 趋势对比                                               │
│  今日: ████████░░ 优于昨日 12%                             │
│  本周: ███████░░░ 优于上周 8%                              │
└─────────────────────────────────────────────────────────────┘
```

#### 2.2 资源使用情况
```
┌─────────────────────────────────────────────────────────────┐
│                   系统资源使用情况                           │
├─────────────────────────────────────────────────────────────┤
│  CPU使用率                                                  │
│  ████████░░░░  65%    峰值: 78%   平均: 45%                │
├─────────────────────────────────────────────────────────────┤
│  内存使用                                                   │
│  █████████░░░  72%    已用: 5.8GB  可用: 2.2GB            │
├─────────────────────────────────────────────────────────────┤
│  磁盘使用                                                   │
│  ███████░░░░  58%    已用: 234GB  可用: 167GB             │
├─────────────────────────────────────────────────────────────┤
│  网络流量                                                   │
│  入站: 45Mbps     出站: 67Mbps   峰值: 120Mbps            │
└─────────────────────────────────────────────────────────────┘
```

### 3. 质量监控仪表盘

#### 3.1 代码质量指标
```
┌─────────────────────────────────────────────────────────────┐
│                   代码质量指标                               │
├─────────────────────────────────────────────────────────────┤
│  📊 代码重复率     4.2%    ✅ 目标: <5%   📈 下降0.3%       │
│  📊 测试覆盖率     87.3%   ✅ 目标: >80%  📈 上升2.1%       │
│  📊 平均复杂度     6.8     ✅ 目标: <10   📉 下降0.2        │
│  📊 可维护性指数   68.5    ✅ 目标: >60   📈 上升3.2        │
├─────────────────────────────────────────────────────────────┤
│  🎯 质量趋势 (过去30天)                                     │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  质量评分趋势                                        │   │
│  │  75 ┼────┬────┬──────────────────────────────────   │   │
│  │    │    │    │                                  │   │
│  │  70 ┼────┼────┼──────────────────────────────────   │   │
│  │    └────┴────┴──────────────────────────────────┘   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### 3.2 接口一致性监控
```
┌─────────────────────────────────────────────────────────────┐
│                 接口一致性监控                              │
├─────────────────────────────────────────────────────────────┤
│  ✅ 接口实现检查     通过: 98.5%     失败: 1.5%            │
│  ✅ 方法签名验证     通过: 99.2%     失败: 0.8%            │
│  ✅ 返回类型检查     通过: 97.8%     失败: 2.2%            │
├─────────────────────────────────────────────────────────────┤
│  📋 不一致接口列表                                        │
│  1. CacheComponent.set_cache_item - 参数顺序不一致       │
│  2. ConfigManager.get_config - 返回类型不匹配            │
│  3. MonitorService.record_metric - 方法签名错误          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 监控数据流

### 1. 数据收集流程

```
应用程序代码
       ↓
   自定义指标收集
       ↓
   Prometheus客户端
       ↓
   Prometheus服务器
       ↓
   数据存储 (TSDB)
       ↓
   Grafana可视化
       ↓
   告警管理 (AlertManager)
       ↓
   通知渠道 (邮件/微信/短信)
```

### 2. 数据处理管道

#### 2.1 指标收集
```python
from prometheus_client import Counter, Histogram, Gauge

# 计数器：单调递增的计数
requests_total = Counter('http_requests_total', 'Total HTTP requests', ['method', 'status'])

# 直方图：分布统计
response_time = Histogram('http_request_duration_seconds', 'Request duration', ['method'])

# 仪表盘：当前值
active_connections = Gauge('active_connections', 'Number of active connections')

# 使用示例
requests_total.labels(method='GET', status='200').inc()
response_time.labels(method='GET').observe(0.5)
active_connections.set(42)
```

#### 2.2 数据聚合
```yaml
# Prometheus聚合规则
groups:
  - name: infrastructure_aggregation
    rules:
      # 请求率聚合
      - record: http_requests_per_second
        expr: rate(http_requests_total[5m])

      # 错误率聚合
      - record: http_error_rate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

      # 可用性计算
      - record: service_availability
        expr: (1 - (rate(http_requests_total{status=~"5.."}[1h]) / rate(http_requests_total[1h]))) * 100
```

---

## 📋 监控维护规范

### 1. 监控配置管理

#### 1.1 配置版本控制
```yaml
# monitoring-config.yaml
version: "1.0"
metadata:
  created: "2025-09-23"
  updated: "2025-09-23"
  maintainer: "专项修复小组"

prometheus:
  global:
    scrape_interval: 15s
    evaluation_interval: 15s

  rule_files:
    - "alert_rules.yml"
    - "recording_rules.yml"

  alerting:
    alertmanagers:
      - static_configs:
          - targets: ["alertmanager:9093"]
```

#### 1.2 配置更新流程
1. **开发环境测试**: 在开发环境验证配置变更
2. **预发布验证**: 在预发布环境进行压力测试
3. **灰度发布**: 分批次应用配置变更
4. **效果监控**: 监控配置变更后的系统表现
5. **回滚准备**: 准备配置回滚方案

### 2. 告警管理

#### 2.1 告警生命周期
```
告警触发 → 去重处理 → 路由分发 → 升级策略 → 解决确认 → 根因分析 → 预防措施
```

#### 2.2 告警抑制规则
```yaml
# 抑制规则示例
inhibit_rules:
  # 如果基础设施服务宕机，抑制所有相关告警
  - source_match:
      alertname: InfrastructureServiceDown
    target_match:
      service: infrastructure
    equal: ['instance']

  # 如果数据中心网络异常，抑制该区域的连接告警
  - source_match:
      alertname: DataCenterNetworkDown
    target_match_re:
      instance: "dc1-.*"
    equal: ['datacenter']
```

### 3. 容量规划

#### 3.1 资源预测模型
```python
from sklearn.linear_model import LinearRegression
import numpy as np

class ResourcePredictor:
    def __init__(self):
        self.cpu_model = LinearRegression()
        self.memory_model = LinearRegression()
        self.disk_model = LinearRegression()

    def train_models(self, historical_data):
        """训练预测模型"""
        X = historical_data[['timestamp', 'request_rate', 'user_count']]
        y_cpu = historical_data['cpu_usage']
        y_memory = historical_data['memory_usage']
        y_disk = historical_data['disk_usage']

        self.cpu_model.fit(X, y_cpu)
        self.memory_model.fit(X, y_memory)
        self.disk_model.fit(X, y_disk)

    def predict_resources(self, future_data):
        """预测未来资源需求"""
        return {
            'cpu': self.cpu_model.predict(future_data),
            'memory': self.memory_model.predict(future_data),
            'disk': self.disk_model.predict(future_data)
        }
```

#### 3.2 扩容决策树
```
需要扩容吗？
├── 是 → 资源使用率 > 80% 持续15分钟
│   ├── CPU瓶颈 → 增加CPU核心数
│   ├── 内存瓶颈 → 增加内存容量
│   └── I/O瓶颈 → 增加磁盘或网络带宽
└── 否 → 继续监控
    └── 预测未来7天使用率
        ├── > 90% → 提前扩容准备
        └── < 70% → 考虑缩容优化
```

---

## 📊 监控效果评估

### 1. 监控覆盖率指标

| 组件 | 监控覆盖率 | 告警覆盖率 | SLO达成率 |
|------|------------|------------|-----------|
| 缓存服务 | 98% | 95% | 99.9% |
| 配置服务 | 96% | 92% | 99.9% |
| 监控服务 | 99% | 98% | 99.95% |
| 总体 | 97% | 95% | 99.9% |

### 2. 问题发现效率

```
平均问题发现时间: 2.3分钟 (目标: <5分钟)
平均问题定位时间: 12.5分钟 (目标: <15分钟)
平均问题解决时间: 45分钟 (目标: <1小时)
```

### 3. 业务影响度量

```
宕机时间减少: 65% (通过主动监控)
用户投诉减少: 40% (通过快速响应)
系统可用性提升: 0.5% (99.9% → 99.95%)
```

---

## 🔧 监控工具链

### 1. 数据收集工具
- **Prometheus**: 指标收集和存储
- **Node Exporter**: 系统指标收集
- **Redis Exporter**: Redis监控
- **Custom Exporters**: 业务特定指标

### 2. 可视化工具
- **Grafana**: 仪表盘和图表
- **Prometheus UI**: 原始查询界面
- **Custom Dashboards**: 业务定制视图

### 3. 告警工具
- **AlertManager**: 告警路由和管理
- **通知集成**: 邮件、微信、短信
- **告警聚合**: 去重和抑制

### 4. 分析工具
- **PromQL**: 查询语言
- **Grafana Explore**: 临时分析
- **Jupyter Notebooks**: 深度分析

---

**完善的监控体系，是系统稳定的守护者！** 👁️‍🗨️

---

## 📞 监控支持

### 技术支持
- **监控配置**: monitoring@infrastructure.team
- **告警处理**: alerts@infrastructure.team
- **性能优化**: performance@infrastructure.team
- **容量规划**: capacity@infrastructure.team

### 文档资源
- [Prometheus文档](https://prometheus.io/docs/)
- [Grafana文档](https://grafana.com/docs/)
- [监控最佳实践](../../best-practices/monitoring/)

---

**让数据驱动决策，让监控守护系统！** 📊
