# 资源管理模块指南

## 1. 模块概述
资源管理模块负责统一管理系统硬件资源分配和监控，包括：
- GPU资源分配和负载均衡
- 内存使用监控和配额管理
- 线程/进程资源分配
- 网络带宽控制

## 2. 核心文件结构
```
src/infrastructure/resource/
├── gpu_manager.py       # GPU资源管理
├── quota_manager.py     # 资源配额控制  
├── resource_manager.py  # 统一资源接口
└── docs/
    └── module_guide.md  # 本说明文档
```

## 3. 核心功能说明

### 3.1 GPU资源管理 (`gpu_manager.py`)
```python
class GPUManager:
    def allocate_gpu(self, task_id, mem_required):
        """智能分配GPU资源
        参数:
            task_id: 任务唯一标识
            mem_required: 需要的内存大小(GB)
        返回:
            分配的GPU设备列表
        """
        
    def release_gpu(self, task_id):
        """释放GPU资源"""
        
    def monitor_utilization(self):
        """实时监控GPU使用率
        返回:
            {
                "gpu0": {"utilization": 0.65, "mem_used": "8GB"},
                "gpu1": {"utilization": 0.32, "mem_used": "4GB"}
            }
        """
```

### 3.2 配额管理 (`quota_manager.py`)
```python
class QuotaManager:
    def set_quota(self, resource_type, limits):
        """设置资源配额
        参数:
            resource_type: 资源类型(gpu/cpu/mem/network)
            limits: 配额限制
                gpu: {'count': 2, 'mem_per_gpu': '16GB'}
                cpu: {'cores': 8}
                mem: {'total': '64GB'}
                network: {'bandwidth': '1Gbps'}
        """
        
    def check_quota(self, task_id):
        """检查任务资源使用是否超限
        返回:
            (bool, str): (是否超限, 超限描述)
        """
```

### 3.3 统一资源接口 (`resource_manager.py`)
```python
class ResourceManager:
    def request_resources(self, spec):
        """统一资源申请接口
        参数:
            spec = {
                'gpu': {'count': 2, 'mem': '16GB'},
                'cpu': {'cores': 4},
                'mem': '32GB',
                'network': {'bandwidth': '500Mbps'}
            }
        返回:
            分配的资源句柄
        """
        
    def get_usage_report(self):
        """生成资源使用报告
        返回:
            {
                "total": {...},
                "per_task": {...},
                "trends": {...}
            }
        """
```

## 4. 使用示例

### 4.1 基础使用
```python
from infrastructure.resource import ResourceManager

# 初始化
res_mgr = ResourceManager()

# 申请资源
resource_spec = {
    'gpu': {'count': 1, 'mem': '8GB'},
    'cpu': {'cores': 2},
    'mem': '16GB'
}
handle = res_mgr.request_resources(resource_spec)

# 使用资源...
# 释放资源
res_mgr.release_resources(handle)
```

### 4.2 高级配置
```python
# 设置配额限制
from infrastructure.resource import QuotaManager
QuotaManager().set_quota(
    'gpu', 
    {'max_count': 4, 'mem_per_gpu': '16GB'}
)

# 获取实时监控数据
usage = ResourceManager().get_usage_report()
```

## 5. 性能指标
| 操作 | 延迟 | 吞吐量 | 适用场景 |
|------|------|--------|----------|
| GPU分配 | ≤10ms | 1000次/秒 | 模型训练 |
| 配额检查 | ≤2ms | 5000次/秒 | 实时交易 |
| 资源监控 | ≤5ms | 1000次/秒 | 系统监控 |

## 6. 设计原则
1. **统一接口**：通过ResourceManager提供一致访问方式
2. **智能调度**：基于负载的动态资源分配
3. **安全隔离**：严格的配额限制防止资源抢占
4. **实时监控**：秒级资源使用监控

## 7. 版本历史
- v1.0 (2023-10-01): 初始版本
- v1.1 (2023-11-15): 增加网络带宽控制
