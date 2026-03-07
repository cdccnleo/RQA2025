# RQA2025 日志分析平台标准格式示例

本目录包含RQA2025基础设施层日志系统对主流日志分析平台标准格式输出的示例代码。

## 支持的平台格式

### 🔍 ELK Stack (Elasticsearch, Logstash, Kibana)
- **格式**: JSON with ECS (Elastic Common Schema)
- **特性**: 结构化字段、ECS兼容、批量索引支持
- **用途**: 企业级日志聚合和可视化

### 🔎 Splunk
- **格式**: Splunk HTTP Event Collector (HEC) JSON
- **特性**: 时间戳精度控制、索引路由、字段提取
- **用途**: 企业安全信息和事件管理

### 🐶 Datadog
- **格式**: Datadog Log Management API JSON
- **特性**: 标签系统、分布式追踪集成、状态映射
- **用途**: 云原生应用监控和可观测性

### 🔧 New Relic
- **格式**: New Relic Logs API JSON
- **特性**: 实体关联、APM集成、属性扩展
- **用途**: 应用性能监控和业务洞察

### 📊 Loki (Prometheus)
- **格式**: Loki Push API with LogQL labels
- **特性**: 标签驱动查询、流聚合、多租户支持
- **用途**: 云原生日志聚合

### 🌫️ Graylog
- **格式**: Graylog Extended Log Format (GELF) JSON
- **特性**: 结构化字段、严重程度映射、字段扩展
- **用途**: 集中式日志管理

### 🔥 Fluentd
- **格式**: Fluentd Forward Protocol (MessagePack/JSON)
- **特性**: 标签系统、时间戳精度、插件生态
- **用途**: 统一日志收集管道

## 核心特性

### 📋 标准化日志条目
所有格式都基于统一的`StandardLogEntry`数据结构，包含：
- 时间戳和时区信息
- 日志级别和类别
- 分布式追踪上下文
- 用户和会话信息
- 元数据和标签
- 自定义扩展字段

### 🔄 统一转换接口
```python
from infrastructure.logging.standards import StandardFormatter

formatter = StandardFormatter()

# 单个条目转换
result = formatter.format_log_entry(entry, StandardFormatType.ELK)

# 批量转换
results = formatter.format_batch(entries, StandardFormatType.SPLUNK)
```

### ⚙️ 配置化管理
```python
from infrastructure.logging.standards import StandardFormatManager

manager = StandardFormatManager()

# 注册输出配置
config = StandardOutputConfig(
    format_type=StandardFormatType.DATADOG,
    endpoint="https://http-intake.logs.datadoghq.com/v1/input",
    api_key="your-api-key",
    batch_size=100
)
manager.register_config("datadog-prod", config)

# 发送日志
result = await manager.send_to_target(entries, "datadog-prod")
```

## 使用示例

### 基本使用
```bash
cd examples/logging/standards
python standard_formats_example.py
```

### 集成到现有代码
```python
# 从现有日志记录转换为标准格式
from infrastructure.logging.standards import StandardFormatter

formatter = StandardFormatter()

# 转换内部日志格式
standard_entry = formatter.convert_from_internal_format(internal_record)

# 输出到多个平台
elk_output = formatter.format_log_entry(standard_entry, StandardFormatType.ELK)
splunk_output = formatter.format_log_entry(standard_entry, StandardFormatType.SPLUNK)
```

## 架构优势

1. **标准化**: 统一的日志数据模型
2. **可扩展**: 易于添加新的日志平台支持
3. **性能优化**: 批量处理和异步发送
4. **容错性**: 重试机制和降级处理
5. **配置化**: 灵活的输出配置管理

## 生产部署建议

### 配置模板
```yaml
# config/standard_outputs.yaml
outputs:
  elk-production:
    format_type: elk
    endpoint: https://elasticsearch.prod.company.com:9200/_bulk
    batch_size: 100
    compression: true
    timeout: 30

  datadog-monitoring:
    format_type: datadog
    endpoint: https://http-intake.logs.datadoghq.com/v1/input
    api_key: ${DATADOG_API_KEY}
    batch_size: 200
    async_mode: true
```

### 监控指标
- 输出成功率
- 批处理延迟
- 错误重试次数
- 队列积压情况

## 扩展开发

### 添加新平台支持
1. 继承`BaseStandardFormat`抽象类
2. 实现格式化方法
3. 定义平台特定的字段映射
4. 在`StandardFormatType`枚举中注册
5. 更新`StandardFormatter`工厂方法

### 自定义字段映射
```python
class CustomFormat(BaseStandardFormat):
    def format_log_entry(self, entry: StandardLogEntry) -> Dict[str, Any]:
        return {
            "custom_timestamp": entry.timestamp.isoformat(),
            "custom_level": self.convert_log_level(entry.level),
            "custom_message": entry.message,
            # 添加自定义映射逻辑
        }
```

## 故障排除

### 常见问题
1. **格式验证失败**: 检查`StandardLogEntry`字段完整性
2. **网络超时**: 调整`timeout`配置或启用压缩
3. **API密钥错误**: 验证凭据配置
4. **批量大小过大**: 降低`batch_size`参数

### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用详细日志
manager = StandardFormatManager()
# 所有操作都会输出调试信息
```

## 许可证

本代码遵循RQA2025项目许可证。
