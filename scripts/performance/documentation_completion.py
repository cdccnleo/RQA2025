#!/usr/bin/env python3
"""
文档完善脚本 - 补充技术文档和用户指南
完善数据层的各种文档
"""

import asyncio
import logging
import json
import os
import sys
from typing import List
from dataclasses import dataclass, field
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logger = logging.getLogger(__name__)


@dataclass
class DocumentationTask:
    """文档任务"""
    name: str
    description: str
    doc_type: str  # 'technical', 'user_guide', 'api_doc'
    status: str  # 'pending', 'completed', 'failed'
    priority: str  # 'high', 'medium', 'low'
    file_path: str
    content_template: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class DocumentationCompleter:
    """文档完善器"""

    def __init__(self):
        self.documentation_tasks: List[DocumentationTask] = []

        # 初始化文档任务列表
        self._initialize_documentation_tasks()

        logger.info("DocumentationCompleter initialized")

    def _initialize_documentation_tasks(self):
        """初始化文档任务列表"""
        tasks = [
            DocumentationTask(
                name="数据层架构文档",
                description="完善数据层架构设计文档",
                doc_type="technical",
                status="pending",
                priority="high",
                file_path="docs/architecture/data_layer_architecture.md",
                content_template="data_layer_architecture"
            ),
            DocumentationTask(
                name="API使用指南",
                description="创建API使用指南",
                doc_type="user_guide",
                status="pending",
                priority="high",
                file_path="docs/user_guides/api_usage_guide.md",
                content_template="api_usage_guide"
            ),
            DocumentationTask(
                name="性能优化指南",
                description="创建性能优化指南",
                doc_type="user_guide",
                status="pending",
                priority="medium",
                file_path="docs/user_guides/performance_optimization_guide.md",
                content_template="performance_optimization_guide"
            ),
            DocumentationTask(
                name="数据质量监控文档",
                description="创建数据质量监控文档",
                doc_type="technical",
                status="pending",
                priority="medium",
                file_path="docs/technical/data_quality_monitoring.md",
                content_template="data_quality_monitoring"
            ),
            DocumentationTask(
                name="部署指南",
                description="创建部署指南",
                doc_type="user_guide",
                status="pending",
                priority="high",
                file_path="docs/user_guides/deployment_guide.md",
                content_template="deployment_guide"
            ),
            DocumentationTask(
                name="故障排除指南",
                description="创建故障排除指南",
                doc_type="user_guide",
                status="pending",
                priority="medium",
                file_path="docs/user_guides/troubleshooting_guide.md",
                content_template="troubleshooting_guide"
            )
        ]

        self.documentation_tasks = tasks

    async def run_documentation_completion(self):
        """运行文档完善"""
        logger.info("开始文档完善...")

        # 按优先级排序
        high_priority = [t for t in self.documentation_tasks if t.priority == 'high']
        medium_priority = [t for t in self.documentation_tasks if t.priority == 'medium']
        low_priority = [t for t in self.documentation_tasks if t.priority == 'low']

        # 按优先级执行
        for task in high_priority + medium_priority + low_priority:
            await self._complete_documentation(task)

        # 生成完成报告
        await self._generate_completion_report()

        logger.info("文档完善完成")

    async def _complete_documentation(self, task: DocumentationTask):
        """完成单个文档任务"""
        logger.info(f"开始完善文档: {task.name}")

        try:
            # 创建目录
            os.makedirs(os.path.dirname(task.file_path), exist_ok=True)

            # 生成文档内容
            content = await self._generate_document_content(task)

            # 写入文件
            with open(task.file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            task.status = "completed"
            logger.info(f"文档完善成功: {task.name} -> {task.file_path}")

        except Exception as e:
            task.status = "failed"
            logger.error(f"文档完善失败: {task.name}, 错误: {e}")

    async def _generate_document_content(self, task: DocumentationTask) -> str:
        """生成文档内容"""
        if task.content_template == "data_layer_architecture":
            return self._generate_data_layer_architecture()
        elif task.content_template == "api_usage_guide":
            return self._generate_api_usage_guide()
        elif task.content_template == "performance_optimization_guide":
            return self._generate_performance_optimization_guide()
        elif task.content_template == "data_quality_monitoring":
            return self._generate_data_quality_monitoring()
        elif task.content_template == "deployment_guide":
            return self._generate_deployment_guide()
        elif task.content_template == "troubleshooting_guide":
            return self._generate_troubleshooting_guide()
        else:
            return f"# {task.name}\n\n{task.description}"

    def _generate_data_layer_architecture(self) -> str:
        """生成数据层架构文档"""
        return """# 数据层架构设计文档

## 概述

数据层是RQA2025系统的核心组件，负责数据的获取、处理、存储和分发。本文档详细描述了数据层的架构设计、组件关系和实现细节。

## 架构概览

### 核心组件

1. **数据加载器 (Data Loaders)**
   - 支持多种数据源：加密货币、宏观经济、期权、债券、商品、外汇
   - 统一的加载接口和错误处理机制
   - 异步加载和并发处理

2. **缓存管理器 (Cache Manager)**
   - 多级缓存策略：内存缓存、磁盘缓存
   - 智能缓存失效和预热机制
   - 缓存统计和性能监控

3. **数据验证器 (Data Validator)**
   - 数据格式验证和完整性检查
   - 自定义验证规则支持
   - 数据质量评估和报告

4. **性能监控器 (Performance Monitor)**
   - 实时性能指标收集
   - 告警机制和阈值管理
   - 性能报告和趋势分析

5. **数据质量监控器 (Quality Monitor)**
   - 多维度数据质量评估
   - 质量修复建议和自动修复
   - 质量报告和可视化

## 数据流

```
数据源 -> 数据加载器 -> 缓存管理器 -> 数据验证器 -> 应用层
                ↓
        性能监控器 + 质量监控器
```

## 技术栈

- **语言**: Python 3.8+
- **异步框架**: asyncio
- **缓存**: 内存缓存 + 磁盘缓存
- **监控**: 自定义监控系统
- **API**: FastAPI + WebSocket

## 性能指标

- 缓存命中率: >95%
- API响应时间: <100ms
- 数据加载时间: <5s
- 系统可用性: >99.9%

## 扩展性

- 插件化数据加载器
- 可配置的缓存策略
- 可扩展的监控指标
- 模块化架构设计
"""

    def _generate_api_usage_guide(self) -> str:
        """生成API使用指南"""
        return """# API使用指南

## 概述

RQA2025数据层提供了REST API和WebSocket API两种接口，支持数据的查询、监控和管理。

## REST API

### 基础信息

- **基础URL**: `http://localhost:8000/api/v1`
- **认证**: 暂不需要
- **格式**: JSON

### 主要端点

#### 1. 健康检查

```bash
GET /api/v1/data/health
```

响应示例:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0"
}
```

#### 2. 数据源列表

```bash
GET /api/v1/data/sources
```

响应示例:
```json
{
  "sources": [
    {
      "name": "crypto",
      "description": "加密货币数据",
      "status": "active"
    },
    {
      "name": "macro",
      "description": "宏观经济数据", 
      "status": "active"
    }
  ]
}
```

#### 3. 加载数据

```bash
POST /api/v1/data/load
Content-Type: application/json

{
  "source": "crypto",
  "symbol": "BTC",
  "timeframe": "1d"
}
```

#### 4. 性能指标

```bash
GET /api/v1/data/performance
```

#### 5. 数据质量检查

```bash
GET /api/v1/data/quality
```

## WebSocket API

### 连接

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/market_data');
```

### 订阅频道

- `market_data`: 实时市场数据
- `quality_monitor`: 数据质量监控
- `performance_monitor`: 性能监控
- `alerts`: 告警信息

### 消息格式

```json
{
  "channel": "market_data",
  "data": {
    "symbol": "BTC",
    "price": 50000,
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

## 客户端SDK

### Python SDK

```python
from src.engine.web.client_sdk import RQA2025DataClient

# 创建客户端
client = RQA2025DataClient("http://localhost:8000")

# 获取数据源列表
sources = await client.list_data_sources()

# 加载数据
data = await client.load_data("crypto", "BTC")

# 获取性能指标
metrics = await client.get_performance_metrics()
```

## 错误处理

### HTTP状态码

- `200`: 成功
- `400`: 请求参数错误
- `404`: 资源不存在
- `500`: 服务器内部错误

### 错误响应格式

```json
{
  "error": "错误描述",
  "code": "ERROR_CODE",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## 最佳实践

1. **缓存策略**: 合理使用缓存减少请求
2. **错误处理**: 实现重试机制和降级策略
3. **监控**: 监控API调用性能和错误率
4. **限流**: 遵守API调用频率限制
"""

    def _generate_performance_optimization_guide(self) -> str:
        """生成性能优化指南"""
        return """# 性能优化指南

## 概述

本文档提供了RQA2025数据层的性能优化指南，包括缓存优化、内存管理、并发处理等方面的最佳实践。

## 缓存优化

### 缓存策略

1. **LRU (Least Recently Used)**
   - 适用场景: 访问模式相对均匀
   - 优点: 实现简单，内存效率高
   - 缺点: 可能淘汰热点数据

2. **LFU (Least Frequently Used)**
   - 适用场景: 访问频率差异较大
   - 优点: 保留热点数据
   - 缺点: 实现复杂，内存开销大

3. **TTL (Time To Live)**
   - 适用场景: 数据有明确的生命周期
   - 优点: 自动过期，内存管理简单
   - 缺点: 可能过早过期

### 缓存配置建议

```python
# 高性能配置
cache_config = CacheConfig(
    max_size=5000,
    ttl=1800,  # 30分钟
    enable_disk_cache=True,
    compression=True
)

# 内存优化配置
cache_config = CacheConfig(
    max_size=1000,
    ttl=300,   # 5分钟
    enable_disk_cache=False,
    compression=False
)
```

## 内存管理

### 内存监控

```python
import psutil

# 监控内存使用
memory = psutil.virtual_memory()
print(f"内存使用率: {memory.percent}%")
print(f"可用内存: {memory.available / 1024 / 1024:.2f} MB")
```

### 内存优化策略

1. **数据压缩**
   - 使用gzip压缩大数据
   - 压缩比可达70-80%

2. **分页加载**
   - 避免一次性加载大量数据
   - 使用游标分页

3. **对象池**
   - 重用对象减少GC压力
   - 适用于频繁创建的对象

## 并发处理

### 异步编程

```python
import asyncio

async def load_data_concurrent(sources):
    tasks = []
    for source in sources:
        task = asyncio.create_task(load_single_source(source))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

### 并发控制

```python
# 限制并发数
semaphore = asyncio.Semaphore(10)

async def controlled_load(source):
    async with semaphore:
        return await load_single_source(source)
```

## 数据库优化

### 索引优化

1. **复合索引**
   ```sql
   CREATE INDEX idx_symbol_timestamp ON market_data(symbol, timestamp);
   ```

2. **覆盖索引**
   ```sql
   CREATE INDEX idx_symbol_price ON market_data(symbol, price, timestamp);
   ```

### 查询优化

1. **分页查询**
   ```sql
   SELECT * FROM market_data 
   WHERE symbol = 'BTC' 
   ORDER BY timestamp DESC 
   LIMIT 100 OFFSET 0;
   ```

2. **批量操作**
   ```python
   # 批量插入
   await db.execute_many(
       "INSERT INTO market_data VALUES (?, ?, ?)",
       data_batch
   )
   ```

## 网络优化

### 连接池

```python
import aiohttp

# 使用连接池
async with aiohttp.ClientSession(
    connector=aiohttp.TCPConnector(
        limit=100,
        limit_per_host=10
    )
) as session:
    # 使用session进行请求
    pass
```

### 超时设置

```python
# 设置合理的超时时间
timeout = aiohttp.ClientTimeout(total=30, connect=10)
```

## 监控和调优

### 性能指标

1. **响应时间**: <100ms
2. **吞吐量**: >1000 req/s
3. **错误率**: <1%
4. **内存使用**: <80%

### 监控工具

1. **Prometheus**: 指标收集
2. **Grafana**: 可视化监控
3. **Jaeger**: 分布式追踪

### 调优步骤

1. **基准测试**: 确定性能基线
2. **瓶颈分析**: 识别性能瓶颈
3. **优化实施**: 实施优化措施
4. **效果验证**: 验证优化效果
5. **持续监控**: 持续监控性能

## 最佳实践

1. **缓存优先**: 优先使用缓存减少计算
2. **异步处理**: 使用异步提高并发性能
3. **批量操作**: 批量处理提高效率
4. **资源复用**: 复用连接和对象
5. **监控告警**: 实时监控性能指标
"""

    def _generate_data_quality_monitoring(self) -> str:
        """生成数据质量监控文档"""
        return """# 数据质量监控文档

## 概述

数据质量监控是RQA2025数据层的核心功能，通过多维度评估确保数据的准确性、完整性和可靠性。

## 质量维度

### 1. 完整性 (Completeness)

评估数据是否完整，包括：
- 必需字段是否存在
- 数据覆盖率
- 缺失值比例

```python
# 完整性检查示例
def check_completeness(data):
    required_fields = ['price', 'volume', 'timestamp']
    missing_fields = [field for field in required_fields if field not in data]
    completeness_score = 1 - len(missing_fields) / len(required_fields)
    return completeness_score
```

### 2. 准确性 (Accuracy)

评估数据的准确性，包括：
- 数值范围检查
- 逻辑一致性
- 异常值检测

```python
# 准确性检查示例
def check_accuracy(data):
    issues = []
    
    # 价格检查
    if data.get('price', 0) <= 0:
        issues.append("价格无效")
    
    # 交易量检查
    if data.get('volume', 0) < 0:
        issues.append("交易量为负")
    
    accuracy_score = 1 - len(issues) / 2
    return accuracy_score
```

### 3. 一致性 (Consistency)

评估数据的一致性，包括：
- 格式一致性
- 时间序列一致性
- 跨数据源一致性

### 4. 时效性 (Timeliness)

评估数据的时效性，包括：
- 数据延迟
- 更新频率
- 实时性要求

### 5. 有效性 (Validity)

评估数据的有效性，包括：
- 格式验证
- 类型检查
- 约束验证

## 监控指标

### 质量分数

```python
# 综合质量分数计算
def calculate_quality_score(data):
    scores = {
        'completeness': check_completeness(data),
        'accuracy': check_accuracy(data),
        'consistency': check_consistency(data),
        'timeliness': check_timeliness(data),
        'validity': check_validity(data)
    }
    
    # 加权平均
    weights = {
        'completeness': 0.2,
        'accuracy': 0.3,
        'consistency': 0.2,
        'timeliness': 0.15,
        'validity': 0.15
    }
    
    total_score = sum(scores[k] * weights[k] for k in scores)
    return total_score
```

### 告警阈值

```python
# 告警配置
alert_thresholds = {
    'completeness': 0.9,
    'accuracy': 0.95,
    'consistency': 0.85,
    'timeliness': 0.8,
    'validity': 0.9
}
```

## 质量修复

### 自动修复

```python
# 数据修复示例
def repair_data(data):
    repaired_data = data.copy()
    
    # 修复空值
    if repaired_data.get('price') is None:
        repaired_data['price'] = 0
    
    # 修复负值
    if repaired_data.get('volume', 0) < 0:
        repaired_data['volume'] = abs(repaired_data['volume'])
    
    # 修复时间戳
    if repaired_data.get('timestamp') is None:
        repaired_data['timestamp'] = time.time()
    
    return repaired_data
```

### 修复策略

1. **默认值填充**: 使用合理的默认值
2. **插值修复**: 使用前后数据插值
3. **删除异常**: 删除无法修复的数据
4. **人工审核**: 标记需要人工处理的数据

## 监控报告

### 实时监控

```python
# 实时质量监控
async def monitor_quality():
    while True:
        quality_metrics = await collect_quality_metrics()
        
        # 检查告警
        for metric, value in quality_metrics.items():
            if value < alert_thresholds.get(metric, 0.8):
                await send_alert(metric, value)
        
        await asyncio.sleep(60)  # 每分钟检查一次
```

### 定期报告

```python
# 生成质量报告
def generate_quality_report():
    report = {
        'timestamp': datetime.now().isoformat(),
        'overall_score': calculate_overall_score(),
        'dimension_scores': get_dimension_scores(),
        'issues': get_quality_issues(),
        'recommendations': get_recommendations()
    }
    return report
```

## 可视化

### 质量仪表板

1. **实时质量分数**: 显示当前质量分数
2. **质量趋势**: 显示质量变化趋势
3. **问题分布**: 显示各维度问题分布
4. **修复效果**: 显示修复措施的效果

### 告警通知

1. **邮件通知**: 质量分数低于阈值时发送邮件
2. **Webhook**: 集成第三方监控系统
3. **Slack通知**: 发送到Slack频道

## 最佳实践

1. **持续监控**: 建立持续的质量监控机制
2. **及时修复**: 发现问题及时修复
3. **预防为主**: 通过监控预防质量问题
4. **持续改进**: 根据监控结果持续改进
5. **文档记录**: 记录质量问题和解决方案
"""

    def _generate_deployment_guide(self) -> str:
        """生成部署指南"""
        return """# 部署指南

## 概述

本文档提供了RQA2025数据层的详细部署指南，包括环境准备、安装配置、部署验证等步骤。

## 环境要求

### 系统要求

- **操作系统**: Linux (Ubuntu 18.04+) / Windows 10+ / macOS 10.15+
- **Python**: 3.8+
- **内存**: 最少4GB，推荐8GB+
- **磁盘**: 最少10GB可用空间
- **网络**: 稳定的互联网连接

### 依赖软件

- **Docker**: 20.10+ (可选)
- **Docker Compose**: 1.29+ (可选)
- **Git**: 2.20+

## 安装步骤

### 1. 克隆代码

```bash
git clone https://github.com/your-org/rqa2025.git
cd rqa2025
```

### 2. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/macOS
source venv/bin/activate
# Windows
venv\\Scripts\\activate
```

### 3. 安装依赖

```bash
# 升级pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量
nano .env
```

环境变量配置示例:
```bash
# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/rqa2025

# Redis配置
REDIS_URL=redis://localhost:6379

# API配置
API_HOST=0.0.0.0
API_PORT=8000

# 监控配置
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
```

## 部署方式

### 方式一: 直接部署

```bash
# 启动应用
python -m src.infrastructure.web.app_factory

# 或者使用uvicorn
uvicorn src.infrastructure.web.app_factory:app --host 0.0.0.0 --port 8000
```

### 方式二: Docker部署

```bash
# 构建镜像
docker build -t rqa2025:latest .

# 运行容器
docker run -d -p 8000:8000 --name rqa2025 rqa2025:latest
```

### 方式三: Docker Compose部署

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

## 配置说明

### 应用配置

```yaml
# config/app.yaml
app:
  name: RQA2025
  version: 1.0.0
  debug: false

server:
  host: 0.0.0.0
  port: 8000
  workers: 4

database:
  url: postgresql://user:password@localhost:5432/rqa2025
  pool_size: 10
  max_overflow: 20

cache:
  type: redis
  url: redis://localhost:6379
  ttl: 3600

monitoring:
  enabled: true
  prometheus_port: 9090
  grafana_port: 3000
```

### 数据源配置

```yaml
# config/data_sources.yaml
crypto:
  enabled: true
  sources:
    - name: coingecko
      api_key: your_api_key
      rate_limit: 100
    - name: binance
      api_key: your_api_key
      secret: your_secret

macro:
  enabled: true
  sources:
    - name: fred
      api_key: your_api_key
    - name: worldbank
      api_key: your_api_key
```

## 验证部署

### 1. 健康检查

```bash
# 检查API健康状态
curl http://localhost:8000/api/v1/data/health

# 预期响应
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0"
}
```

### 2. 功能测试

```bash
# 测试数据源列表
curl http://localhost:8000/api/v1/data/sources

# 测试数据加载
curl -X POST http://localhost:8000/api/v1/data/load \
  -H "Content-Type: application/json" \
  -d '{"source": "crypto", "symbol": "BTC"}'
```

### 3. 监控验证

```bash
# 检查Prometheus指标
curl http://localhost:9090/metrics

# 检查Grafana (如果启用)
# 访问 http://localhost:3000
```

## 性能调优

### 系统调优

```bash
# 增加文件描述符限制
echo "* soft nofile 65536" >> /etc/security/limits.conf
echo "* hard nofile 65536" >> /etc/security/limits.conf

# 优化内核参数
echo "net.core.somaxconn = 65535" >> /etc/sysctl.conf
echo "net.ipv4.tcp_max_syn_backlog = 65535" >> /etc/sysctl.conf
sysctl -p
```

### 应用调优

```python
# 优化配置示例
app_config = {
    'workers': 4,  # CPU核心数
    'max_connections': 1000,
    'timeout': 30,
    'keepalive': 5
}
```

## 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 检查端口占用
   netstat -tlnp | grep 8000
   
   # 杀死进程
   kill -9 <pid>
   ```

2. **数据库连接失败**
   ```bash
   # 检查数据库状态
   systemctl status postgresql
   
   # 检查连接
   psql -h localhost -U user -d rqa2025
   ```

3. **内存不足**
   ```bash
   # 检查内存使用
   free -h
   
   # 优化内存配置
   export PYTHONMALLOC=malloc
   ```

### 日志分析

```bash
# 查看应用日志
tail -f logs/app.log

# 查看错误日志
grep ERROR logs/app.log

# 查看性能日志
grep "response_time" logs/app.log
```

## 维护

### 备份策略

```bash
# 数据库备份
pg_dump rqa2025 > backup_$(date +%Y%m%d).sql

# 配置文件备份
tar -czf config_backup_$(date +%Y%m%d).tar.gz config/
```

### 更新部署

```bash
# 停止服务
docker-compose down

# 拉取最新代码
git pull origin main

# 重新构建
docker-compose build

# 启动服务
docker-compose up -d
```

### 监控维护

```bash
# 检查服务状态
docker-compose ps

# 查看资源使用
docker stats

# 清理日志
docker system prune -f
```
"""

    def _generate_troubleshooting_guide(self) -> str:
        """生成故障排除指南"""
        return """# 故障排除指南

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
"""

    async def _generate_completion_report(self):
        """生成完成报告"""
        logger.info("生成文档完善报告...")

        completed_tasks = [t for t in self.documentation_tasks if t.status == 'completed']
        failed_tasks = [t for t in self.documentation_tasks if t.status == 'failed']

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tasks': len(self.documentation_tasks),
                'completed': len(completed_tasks),
                'failed': len(failed_tasks),
                'completion_rate': len(completed_tasks) / len(self.documentation_tasks) * 100
            },
            'completed_tasks': [
                {
                    'name': t.name,
                    'description': t.description,
                    'doc_type': t.doc_type,
                    'priority': t.priority,
                    'file_path': t.file_path
                }
                for t in completed_tasks
            ]
        }

        # 保存报告
        report_file = f"reports/documentation_completion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"文档完善报告已保存: {report_file}")

        # 打印摘要
        print("\n=== 文档完善摘要 ===")
        print(f"总任务数: {report['summary']['total_tasks']}")
        print(f"完成数: {report['summary']['completed']}")
        print(f"完成率: {report['summary']['completion_rate']:.1f}%")

        if completed_tasks:
            print("\n✅ 完成的文档:")
            for task in completed_tasks:
                print(f"  - {task.name} ({task.doc_type}) -> {task.file_path}")


async def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建文档完善器
    completer = DocumentationCompleter()

    try:
        # 运行文档完善
        await completer.run_documentation_completion()

        print("\n✅ 文档完善完成!")

    except Exception as e:
        logger.error(f"文档完善失败: {e}")
        print(f"\n❌ 文档完善失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
