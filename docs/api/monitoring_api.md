# 监控面板API文档

## 概述

性能监控面板提供全面的企业级监控功能，包括系统资源监控、内存泄漏检测、连接池监控、缓存效率分析和综合健康评估。该面板支持实时监控和历史数据分析。

## 架构概览

```
performance_monitor_dashboard.py
├── PerformanceMonitorDashboard          # 主监控面板类
│   ├── core: PerformanceMonitorDashboardCore  # 核心监控功能
│   ├── anomaly_detector: AnomalyDetector     # 异常检测器
│   ├── trend_analyzer: TrendAnalyzer         # 趋势分析器
│   └── performance_predictor: PerformancePredictor # 性能预测器

核心监控指标:
├── 系统健康状态       # CPU、内存、磁盘监控
├── 内存泄漏检测       # GC统计、内存增长趋势
├── 连接池监控         # 数据库、Redis连接池
├── 缓存效率分析       # 命中率、内存使用
├── 业务指标监控       # 请求量、错误率、用户会话
├── 安全监控指标       # 认证、授权、数据保护
└── 综合健康报告       # 整体健康评分和建议
```

## 主监控面板API

### PerformanceMonitorDashboard 类

```python
from infrastructure.config.monitoring.performance_monitor_dashboard import PerformanceMonitorDashboard

# 创建监控面板
monitor = PerformanceMonitorDashboard(
    storage_path="config/performance",  # 存储路径
    retention_days=30,                  # 数据保留天数
    enable_system_monitoring=True       # 启用系统监控
)

# 启动监控
monitor.start_monitoring()

# 停止监控
monitor.stop_monitoring()
```

### 核心监控方法

#### 系统健康监控

```python
# 获取系统健康状态
health_status = monitor.get_system_health_status()
print(f"系统状态: {health_status['status']}")
print(f"健康评分: {health_status['health_score']}/100")
print(f"CPU使用率: {health_status['cpu_usage']}%")
print(f"内存使用率: {health_status['memory_usage']}%")
```

**返回数据结构：**
```json
{
    "status": "healthy|warning|critical",
    "health_score": 85,
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "memory_used_gb": 5.4,
    "memory_total_gb": 8.0,
    "disk_usage": 72.3,
    "disk_free_gb": 25.1,
    "process_count": 1250,
    "timestamp": "2025-09-23T10:30:00"
}
```

#### 内存泄漏检测

```python
# 获取内存泄漏检测结果
memory_info = monitor.get_memory_leak_detection()
print(f"GC收集次数: {memory_info['gc_collections']}")
print(f"当前内存使用: {memory_info['current_memory_mb']} MB")
print(f"内存增长率: {memory_info['memory_growth_rate']}")
```

**返回数据结构：**
```json
{
    "gc_collections": [120, 45, 12],
    "gc_stats": [...],
    "current_memory_mb": 256.7,
    "memory_growth_rate": 0.02,
    "potential_leaks": 3,
    "timestamp": "2025-09-23T10:30:00"
}
```

#### 连接池监控

```python
# 获取连接池监控指标
connection_pools = monitor.get_connection_pool_metrics()
print(f"数据库连接池效率: {connection_pools['connection_pool_efficiency']}")
print(f"活跃数据库连接: {connection_pools['database_connections']['active']}")
```

**返回数据结构：**
```json
{
    "database_connections": {
        "active": 5,
        "idle": 10,
        "total": 15,
        "waiting": 0
    },
    "redis_connections": {
        "active": 3,
        "idle": 7,
        "total": 10,
        "waiting": 1
    },
    "connection_pool_efficiency": 0.95,
    "timestamp": "2025-09-23T10:30:00"
}
```

#### 缓存效率分析

```python
# 获取缓存效率监控指标
cache_efficiency = monitor.get_cache_efficiency_metrics()
print(f"内存缓存命中率: {cache_efficiency['memory_cache']['hit_rate']}")
print(f"Redis缓存命中率: {cache_efficiency['redis_cache']['hit_rate']}")
print(f"整体缓存效率: {cache_efficiency['overall_cache_efficiency']}")
```

**返回数据结构：**
```json
{
    "memory_cache": {
        "hit_rate": 0.85,
        "miss_rate": 0.15,
        "size_mb": 50,
        "entries": 1000
    },
    "redis_cache": {
        "hit_rate": 0.92,
        "miss_rate": 0.08,
        "size_mb": 200,
        "keys": 5000
    },
    "overall_cache_efficiency": 0.88,
    "cache_memory_usage_percent": 15.5,
    "timestamp": "2025-09-23T10:30:00"
}
```

#### 业务指标监控

```python
# 获取业务指标
business_metrics = monitor.get_business_metrics()
print(f"每秒请求数: {business_metrics['request_throughput']['per_second']}")
print(f"错误率: {business_metrics['error_rates']['error_rate_percent']}%")
```

**返回数据结构：**
```json
{
    "request_throughput": {
        "per_second": 150,
        "per_minute": 9000,
        "peak_hourly": 500000
    },
    "error_rates": {
        "total_errors": 25,
        "error_rate_percent": 0.0025,
        "critical_errors": 2
    },
    "user_sessions": {
        "active_sessions": 1250,
        "total_sessions_today": 15000,
        "avg_session_duration_minutes": 12.5
    },
    "data_processing": {
        "records_processed": 2500000,
        "processing_rate_per_second": 2500,
        "data_quality_score": 0.98
    },
    "timestamp": "2025-09-23T10:30:00"
}
```

#### 安全监控指标

```python
# 获取安全监控指标
security_metrics = monitor.get_security_metrics()
print(f"安全评分: {security_metrics['security_score']}/100")
print(f"失败登录尝试: {security_metrics['authentication']['failed_login_attempts']}")
```

**返回数据结构：**
```json
{
    "authentication": {
        "successful_logins": 1250,
        "failed_login_attempts": 15,
        "blocked_ips": 3,
        "suspicious_activities": 8
    },
    "authorization": {
        "access_denied_events": 12,
        "privilege_escalation_attempts": 0,
        "unauthorized_access_attempts": 5
    },
    "data_protection": {
        "encrypted_connections": 98.5,
        "data_encryption_coverage": 100.0,
        "audit_log_completeness": 100.0
    },
    "security_score": 95.2,
    "timestamp": "2025-09-23T10:30:00"
}
```

#### 综合健康报告

```python
# 获取综合健康报告
comprehensive_report = monitor.get_comprehensive_health_report()
print(f"整体健康评分: {comprehensive_report['overall_health_score']}/100")
print(f"系统状态: {comprehensive_report['status']}")

print("健康建议:")
for recommendation in comprehensive_report['recommendations']:
    print(f"- {recommendation}")

print("组件健康状态:")
for component, status in comprehensive_report['components'].items():
    print(f"- {component}: {status.get('status', 'unknown')}")
```

**返回数据结构：**
```json
{
    "overall_health_score": 87.5,
    "status": "healthy",
    "components": {
        "system": {...},
        "memory": {...},
        "connections": {...},
        "cache": {...},
        "business": {...},
        "security": {...}
    },
    "recommendations": [
        "系统运行状况良好，继续保持当前性能水平"
    ],
    "generated_at": "2025-09-23T10:30:00",
    "report_version": "2.0"
}
```

### 增强监控统计

```python
# 获取增强的监控统计信息
enhanced_stats = monitor.get_enhanced_monitoring_stats()

# 包含所有监控指标的完整统计
print("系统健康:", enhanced_stats['system_health']['status'])
print("内存泄漏:", "检测到" if enhanced_stats['memory_leaks']['potential_leaks'] > 0 else "正常")
print("缓存效率:", f"{enhanced_stats['cache_efficiency']['overall_cache_efficiency']:.1%}")
print("综合健康:", enhanced_stats['comprehensive_health']['status'])
```

## 操作记录和统计

### 记录操作

```python
# 记录操作执行
monitor.record_operation(
    operation="user_login",     # 操作名称
    duration=0.125,             # 执行时间(秒)
    success=True               # 是否成功
)

# 获取操作统计
operation_stats = monitor.get_operation_stats()
print(f"总操作数: {operation_stats.get('total_operations', 0)}")
print(f"平均响应时间: {operation_stats.get('avg_duration', 0):.3f}秒")
```

### 性能摘要

```python
# 获取性能摘要 (指定时间范围)
performance_summary = monitor.get_performance_summary(time_range_hours=24)

print(f"时间范围: {performance_summary['time_range_hours']}小时")
print(f"总操作数: {performance_summary['total_operations']}")
print(f"成功率: {performance_summary['success_rate']:.1%}")
print(f"平均响应时间: {performance_summary['avg_response_time']:.3f}秒")
```

## 高级监控功能

### 异常检测

```python
# 检测异常
anomalies = monitor.detect_anomalies(metric_name="response_time")
if anomalies:
    print("检测到性能异常:")
    for anomaly in anomalies:
        print(f"- {anomaly['description']}")
```

### 趋势分析

```python
# 分析趋势
trends = monitor.analyze_trends(metric_name="cpu_usage")
if trends:
    print("CPU使用率趋势:")
    print(f"- 趋势方向: {trends.get('direction', 'stable')}")
    print(f"- 变化率: {trends.get('change_rate', 0):.2f}%")
```

### 性能预测

```python
# 预测性能
predictions = monitor.predict_performance(
    metric_name="memory_usage",
    hours_ahead=24
)

if predictions:
    print("内存使用率预测:")
    print(f"- 24小时后预测值: {predictions.get('predicted_value', 0):.1f}%")
    print(f"- 置信度: {predictions.get('confidence', 0):.1f}%")
```

## 配置和初始化

### 高级配置

```python
from infrastructure.config.monitoring.performance_monitor_dashboard import PerformanceMonitorDashboard

# 高级配置初始化
monitor = PerformanceMonitorDashboard(
    storage_path="data/monitoring",     # 监控数据存储路径
    retention_days=90,                  # 数据保留90天
    enable_system_monitoring=True,      # 启用系统监控
    system_monitoring_interval=30,      # 系统监控间隔(秒)
    anomaly_detection_enabled=True,     # 启用异常检测
    predictive_analytics_enabled=True,  # 启用预测分析
    alert_thresholds={                  # 告警阈值配置
        "cpu_usage_percent": 80,
        "memory_usage_percent": 85,
        "disk_usage_percent": 90,
        "error_rate_percent": 1.0
    }
)
```

### 自定义监控指标

```python
class CustomMonitor(PerformanceMonitorDashboard):
    """自定义监控面板"""

    def get_custom_metrics(self):
        """获取自定义业务指标"""
        return {
            "business_kpis": {
                "conversion_rate": 0.035,
                "customer_satisfaction": 4.2,
                "revenue_per_user": 125.50
            },
            "technical_metrics": {
                "api_response_time_p95": 245,
                "database_query_efficiency": 0.89,
                "cache_hit_ratio": 0.94
            },
            "timestamp": datetime.now().isoformat()
        }

    def get_comprehensive_health_report(self):
        """重写综合健康报告，包含自定义指标"""
        base_report = super().get_comprehensive_health_report()
        custom_metrics = self.get_custom_metrics()

        # 集成自定义指标到健康评分
        business_score = custom_metrics['business_kpis']['conversion_rate'] * 1000  # 标准化
        technical_score = custom_metrics['technical_metrics']['cache_hit_ratio'] * 100

        # 重新计算整体健康评分
        component_scores = [
            base_report['overall_health_score'],
            min(business_score, 100),  # 限制在0-100
            technical_score
        ]

        base_report['overall_health_score'] = sum(component_scores) / len(component_scores)
        base_report['components']['custom'] = custom_metrics

        return base_report
```

## 监控数据存储

### 数据持久化

监控面板自动处理数据存储：

```python
# 数据自动保存到配置的存储路径
# 支持JSON格式存储，带压缩和轮转

# 手动触发数据保存
monitor.core.save_metrics_to_disk()

# 清理过期数据
monitor.core.cleanup_old_data(days_to_keep=30)
```

### 数据导出

```python
# 导出监控数据为JSON
export_data = monitor.core.export_metrics(
    start_date="2025-09-01",
    end_date="2025-09-23",
    metrics=["cpu_usage", "memory_usage", "response_time"]
)

# 保存到文件
import json
with open("monitoring_export.json", "w") as f:
    json.dump(export_data, f, indent=2)
```

## 告警和通知

### 告警配置

```python
# 配置告警回调
def alert_callback(alert_data):
    """告警回调函数"""
    severity = alert_data['severity']
    message = alert_data['message']

    if severity == 'critical':
        # 发送短信告警
        send_sms_alert(message)
    elif severity == 'warning':
        # 发送邮件告警
        send_email_alert(message)

    # 记录告警日志
    monitor.logger.warning(f"监控告警: {message}", extra=alert_data)

# 注册告警回调
monitor.add_alert_callback(alert_callback)

# 启用告警系统
monitor.enable_alerts(thresholds={
    'cpu_usage': 90,      # CPU使用率超过90%告警
    'memory_usage': 95,   # 内存使用率超过95%告警
    'error_rate': 5.0     # 错误率超过5%告警
})
```

## 最佳实践

### 1. 定期健康检查

```python
import schedule
import time

def scheduled_health_check():
    """定时健康检查"""
    report = monitor.get_comprehensive_health_report()

    if report['overall_health_score'] < 70:
        # 发送告警
        alert_system.send_alert(
            title="系统健康评分下降",
            message=f"当前健康评分: {report['overall_health_score']}",
            severity="warning"
        )

    # 记录健康日志
    monitor.logger.info("定期健康检查完成", extra={
        "health_score": report['overall_health_score'],
        "status": report['status']
    })

# 每5分钟执行一次健康检查
schedule.every(5).minutes.do(scheduled_health_check)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### 2. 性能基准监控

```python
class PerformanceBaselineMonitor:
    """性能基准监控器"""

    def __init__(self, monitor):
        self.monitor = monitor
        self.baselines = {}

    def establish_baselines(self, observation_period_days=7):
        """建立性能基准"""
        # 收集一周的数据作为基准
        metrics_data = self.monitor.core.get_metrics_history(days=observation_period_days)

        for metric_name in ['response_time', 'cpu_usage', 'memory_usage']:
            if metric_name in metrics_data:
                values = metrics_data[metric_name]
                self.baselines[metric_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values),
                    'p95': statistics.quantiles(values, n=20)[18],  # 95th percentile
                    'p99': statistics.quantiles(values, n=100)[98]  # 99th percentile
                }

    def check_performance_against_baseline(self):
        """检查当前性能是否偏离基准"""
        current_metrics = self.monitor.get_system_health_status()
        alerts = []

        for metric_name, baseline in self.baselines.items():
            if metric_name in current_metrics:
                current_value = current_metrics[metric_name]
                mean = baseline['mean']
                std = baseline['std']

                # 检查是否偏离2个标准差
                if abs(current_value - mean) > 2 * std:
                    alerts.append({
                        'metric': metric_name,
                        'current': current_value,
                        'baseline': mean,
                        'deviation': abs(current_value - mean) / std,
                        'direction': 'above' if current_value > mean else 'below'
                    })

        return alerts
```

### 3. 监控面板集成

```python
class ApplicationMonitor:
    """应用监控集成器"""

    def __init__(self):
        self.monitor = PerformanceMonitorDashboard()
        self.monitor.start_monitoring()

        # 配置自定义告警
        self.setup_custom_alerts()

    def setup_custom_alerts(self):
        """设置自定义告警规则"""

        @self.monitor.alert_handler
        def handle_high_error_rate(alert):
            """处理高错误率告警"""
            if alert['metric'] == 'error_rate' and alert['value'] > 5.0:
                # 触发自动扩容或降级
                self.trigger_auto_scaling()

        @self.monitor.alert_handler
        def handle_memory_leak(alert):
            """处理内存泄漏告警"""
            if 'memory_leak' in alert['type']:
                # 触发GC和内存分析
                self.perform_memory_analysis()

    def record_business_operation(self, operation_name, **context):
        """记录业务操作"""
        start_time = time.time()

        try:
            # 执行业务操作
            result = self.execute_business_logic(operation_name, **context)

            # 记录成功操作
            duration = time.time() - start_time
            self.monitor.record_operation(
                operation=f"business.{operation_name}",
                duration=duration,
                success=True
            )

            return result

        except Exception as e:
            # 记录失败操作
            duration = time.time() - start_time
            self.monitor.record_operation(
                operation=f"business.{operation_name}",
                duration=duration,
                success=False
            )

            # 记录错误指标
            self.monitor.record_error(
                error_type=type(e).__name__,
                operation=operation_name,
                context=context
            )

            raise

    def get_application_health_dashboard(self):
        """获取应用健康仪表板"""
        health_report = self.monitor.get_comprehensive_health_report()

        # 添加应用特定指标
        health_report['application_metrics'] = {
            'uptime_hours': self.get_application_uptime(),
            'active_connections': self.get_active_connections(),
            'queued_requests': self.get_queued_requests(),
            'feature_usage': self.get_feature_usage_stats()
        }

        return health_report
```

## 性能优化建议

### 监控频率调整

```python
# 根据环境调整监控频率
environments = {
    'development': {
        'monitoring_interval': 5,      # 5秒
        'data_retention_days': 7,
        'enable_detailed_logging': True
    },
    'staging': {
        'monitoring_interval': 30,     # 30秒
        'data_retention_days': 30,
        'enable_detailed_logging': True
    },
    'production': {
        'monitoring_interval': 60,     # 1分钟
        'data_retention_days': 90,
        'enable_detailed_logging': False
    }
}

# 根据环境配置监控面板
env = os.getenv('ENVIRONMENT', 'development')
config = environments[env]

monitor = PerformanceMonitorDashboard(
    monitoring_interval=config['monitoring_interval'],
    retention_days=config['data_retention_days'],
    enable_detailed_logging=config['enable_detailed_logging']
)
```

### 数据采样优化

```python
class SmartSampler:
    """智能采样器"""

    def __init__(self, monitor):
        self.monitor = monitor
        self.sample_counts = {}

    def should_sample(self, metric_name, value, threshold=0.1):
        """决定是否采样数据"""
        # 对于变化小的指标，减少采样频率
        if metric_name not in self.sample_counts:
            self.sample_counts[metric_name] = {'count': 0, 'last_value': value}

        last_value = self.sample_counts[metric_name]['last_value']
        change_percent = abs(value - last_value) / last_value if last_value != 0 else 1

        if change_percent > threshold:
            # 重要变化，立即采样
            self.sample_counts[metric_name]['last_value'] = value
            self.sample_counts[metric_name]['count'] = 0
            return True
        else:
            # 小变化，降低采样频率
            self.sample_counts[metric_name]['count'] += 1
            # 每10次采样一次
            return self.sample_counts[metric_name]['count'] % 10 == 0
```

## 故障排查

### 常见问题

1. **监控数据不准确**
   ```python
   # 检查系统权限
   import psutil
   try:
       psutil.cpu_percent()
       print("系统监控权限正常")
   except Exception as e:
       print(f"系统监控权限不足: {e}")
   ```

2. **性能开销过高**
   ```python
   # 降低监控频率
   monitor.set_monitoring_interval(120)  # 2分钟间隔

   # 禁用详细日志
   monitor.disable_detailed_logging()
   ```

3. **磁盘空间不足**
   ```python
   # 清理旧数据
   monitor.cleanup_old_data(days_to_keep=7)

   # 启用数据压缩
   monitor.enable_data_compression()
   ```

---

**版本**: v2.0
**更新日期**: 2025-09-23
**兼容性**: Python 3.8+
