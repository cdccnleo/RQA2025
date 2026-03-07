# RQA2025 引擎层统一Web管理界面

## 概述

RQA2025引擎层统一Web管理界面是系统的核心Web服务组件，提供统一的Web管理入口，整合所有业务模块的Web管理功能。该界面采用现代化的模块化设计，支持实时数据推送和响应式用户界面。

## 架构设计

### 目录结构

```
src/engine/web/
├── __init__.py                    # 引擎层Web包初始化
├── unified_dashboard.py           # 统一Web管理界面主程序
├── modules/                       # 模块化组件目录
│   ├── __init__.py               # 模块包初始化
│   ├── base_module.py            # 基础模块抽象类
│   ├── module_registry.py        # 模块注册管理器
│   ├── module_factory.py         # 模块工厂
│   ├── config_module.py          # 配置管理模块
│   ├── fpga_module.py            # FPGA监控模块
│   ├── resource_module.py        # 资源监控模块
│   └── features_module.py        # 特征监控模块
├── templates/                     # 前端模板目录
│   └── dashboard.html            # 主仪表板模板
└── static/                       # 静态资源目录
    ├── css/                      # 样式文件
    │   └── dashboard.css        # 主样式文件
    └── js/                       # JavaScript文件
        └── dashboard.js         # 主脚本文件
```

### 核心组件

#### 1. UnifiedDashboard (统一Web管理界面)
- **功能**: 主程序入口，管理所有Web服务
- **特性**: FastAPI框架、WebSocket支持、模块化架构
- **位置**: `src/engine/web/unified_dashboard.py`

#### 2. ModuleRegistry (模块注册表)
- **功能**: 管理所有注册的模块
- **特性**: 动态注册、依赖管理、状态监控
- **位置**: `src/engine/web/modules/module_registry.py`

#### 3. BaseModule (基础模块抽象)
- **功能**: 所有模块的基类
- **特性**: 标准化接口、路由管理、配置支持
- **位置**: `src/engine/web/modules/base_module.py`

## 技术栈

### 后端技术
- **FastAPI**: 现代化Python Web框架
- **WebSocket**: 实时双向通信
- **Pydantic**: 数据验证和序列化
- **Uvicorn**: ASGI服务器

### 前端技术
- **HTML5 + CSS3**: 现代化响应式设计
- **Tailwind CSS**: 实用优先的CSS框架
- **JavaScript (ES6+)**: 交互逻辑
- **Chart.js**: 数据可视化
- **WebSocket API**: 实时数据更新

### 数据通信
- **RESTful API**: 标准HTTP接口
- **WebSocket**: 实时数据推送
- **JSON**: 数据交换格式

## 功能特性

### 1. 统一入口
- 单一访问地址，整合所有Web管理功能
- 统一的用户界面和交互体验
- 集中的权限管理和认证

### 2. 模块化设计
- 支持动态模块注册和卸载
- 模块间松耦合，独立开发
- 可扩展的插件架构

### 3. 实时更新
- WebSocket实时数据推送
- 自动刷新和状态同步
- 实时告警和通知

### 4. 现代化界面
- 响应式设计，支持多设备
- 直观的数据可视化
- 用户友好的交互体验

## 快速开始

### 1. 启动服务

```bash
# 使用引擎层启动脚本
conda activate test
python scripts/engine/web/start_dashboard.py
```

### 2. 快速访问

```bash
# 自动打开浏览器访问
python scripts/engine/web/access_dashboard.py
```

### 3. 访问地址

- **主界面**: http://127.0.0.1:8081
- **API文档**: http://127.0.0.1:8081/api/docs
- **WebSocket**: ws://127.0.0.1:8081/ws

## API接口

### RESTful API端点

```
GET  /api/modules              # 获取所有模块
GET  /api/modules/{name}       # 获取特定模块
GET  /api/system/overview      # 系统概览
GET  /api/config              # 配置信息
GET  /api/resources           # 资源信息
GET  /api/health             # 健康检查
```

### 模块API

```
GET  /api/modules/config     # 配置管理API
GET  /api/modules/fpga_monitoring    # FPGA监控API
GET  /api/modules/resource_monitoring # 资源监控API
GET  /api/modules/features_monitoring # 特征监控API
```

### WebSocket端点

```
WS   /ws                     # WebSocket连接
```

## 已集成模块

### 1. 配置管理模块 (config)
- **功能**: 统一配置管理和热重载
- **路由**: `/api/modules/config`
- **特性**: 配置验证、热重载、加密支持

### 2. FPGA监控模块 (fpga_monitoring)
- **功能**: FPGA性能监控和告警管理
- **路由**: `/api/modules/fpga_monitoring`
- **特性**: 性能监控、告警管理、状态跟踪

### 3. 资源监控模块 (resource_monitoring)
- **功能**: 系统资源使用情况监控
- **路由**: `/api/modules/resource_monitoring`
- **特性**: CPU、内存、磁盘监控

### 4. 特征监控模块 (features_monitoring)
- **功能**: 特征工程性能和数据质量监控
- **路由**: `/api/modules/features_monitoring`
- **特性**: 特征质量监控、性能分析

## 开发指南

### 添加新模块

1. **创建模块类**
```python
from src.engine.web.modules.base_module import BaseModule

class MyModule(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.setup_routes()
    
    def setup_routes(self):
        @self.router.get("/my-endpoint")
        async def my_endpoint():
            return {"message": "Hello from MyModule"}
    
    def get_module_data(self):
        return {"status": "online", "data": "..."}
    
    def get_module_status(self):
        return "online"
```

2. **注册模块**
```python
# 在unified_dashboard.py中注册
self.module_registry.register_module(MyModule, config)
```

### 自定义API

1. **在模块中定义路由**
```python
@self.router.get("/custom-api")
async def custom_api():
    return {"data": "custom response"}
```

2. **实现业务逻辑**
```python
def process_data(self, data):
    # 实现业务逻辑
    return processed_data
```

3. **添加数据验证**
```python
from pydantic import BaseModel

class MyDataModel(BaseModel):
    field1: str
    field2: int
```

### 前端开发

1. **创建HTML模板**
```html
<!-- templates/my_module.html -->
<div class="module-card">
    <h3>我的模块</h3>
    <div id="module-content"></div>
</div>
```

2. **编写JavaScript逻辑**
```javascript
// static/js/my_module.js
class MyModuleManager {
    constructor() {
        this.init();
    }
    
    async init() {
        await this.loadData();
        this.setupEventListeners();
    }
}
```

3. **设计CSS样式**
```css
/* static/css/my_module.css */
.my-module {
    background: var(--bg-secondary);
    border-radius: 0.75rem;
    padding: 1.5rem;
}
```

## 配置管理

### 主配置文件
- **位置**: `config/web_dashboard_config.json`
- **功能**: 统一管理所有配置

### 配置结构
```json
{
    "dashboard": {
        "title": "RQA2025 统一管理平台",
        "version": "1.0.0",
        "theme": "modern"
    },
    "modules": {
        "config": {
            "enabled": true,
            "refresh_interval": 30
        }
    },
    "security": {
        "enable_auth": false,
        "allowed_hosts": ["127.0.0.1"]
    }
}
```

## 部署指南

### 开发环境
```bash
# 启动开发服务器
python scripts/engine/web/start_dashboard.py
```

### 生产环境
```bash
# 使用生产配置启动
RQA_ENV=production python scripts/engine/web/start_dashboard.py
```

### Docker部署
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["python", "scripts/engine/web/start_dashboard.py"]
```

## 监控和日志

### 日志配置
- **位置**: `logs/web/`
- **级别**: INFO, WARNING, ERROR
- **格式**: JSON结构化日志

### 监控指标
- **服务健康检查**: `/api/health`
- **性能监控**: CPU、内存、响应时间
- **模块状态**: 各模块运行状态

## 故障排除

### 常见问题

1. **端口冲突**
```bash
# 检查端口占用
netstat -ano | findstr :8080

# 终止占用进程
taskkill /PID <PID> /F
```

2. **依赖问题**
```bash
# 安装缺失依赖
pip install fastapi uvicorn websockets psutil
```

3. **编码问题**
```bash
# 设置环境变量
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
```

### 调试模式
```bash
# 启用调试模式
RQA_DEBUG=true python scripts/engine/web/start_dashboard.py
```

## 更新日志

### v1.0.0 (2025-08-05)
- ✅ 完成引擎层迁移
- ✅ 实现核心架构和模块化设计
- ✅ 集成4个核心监控模块
- ✅ 创建现代化前端界面
- ✅ 实现WebSocket实时通信
- ✅ 优化启动脚本和配置管理

## 贡献指南

1. **代码规范**: 遵循PEP 8规范
2. **文档更新**: 及时更新相关文档
3. **测试覆盖**: 为新功能添加测试用例
4. **提交规范**: 使用清晰的提交信息

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请联系开发团队。 