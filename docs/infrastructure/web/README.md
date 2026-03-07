# RQA2025 统一Web管理界面

## 概述

RQA2025统一Web管理界面是系统的核心管理平台，整合了所有模块的Web管理功能，提供统一的访问入口和现代化的用户界面。

## 功能特性

### 🎯 核心功能
- **统一入口**: 所有模块的Web管理功能通过单一平台访问
- **现代化界面**: 采用响应式设计和现代化UI框架
- **模块化架构**: 支持动态注册和管理功能模块
- **实时更新**: 通过WebSocket实现实时数据更新
- **权限控制**: 统一的用户认证和权限管理

### 📊 管理模块
1. **系统概览** - 系统整体状态监控
2. **配置管理** - 统一配置管理界面
3. **策略管理** - 策略配置和监控
4. **数据管理** - 数据源和数据集管理
5. **回测管理** - 回测配置和结果查看
6. **监控告警** - 系统监控和告警管理
7. **资源管理** - 计算资源监控和管理
8. **用户管理** - 用户权限和访问控制

## 快速开始

### 1. 启动服务

#### 使用启动脚本
```bash
# 开发模式启动
python scripts/web/start_unified_dashboard.py --reload

# 生产模式启动
python scripts/web/start_unified_dashboard.py --env production --port 8080

# 自定义配置启动
python scripts/web/start_unified_dashboard.py --config config/dashboard.json
```

#### 直接启动
```bash
# 进入项目目录
cd RQA2025

# 启动统一Web管理界面
python src/infrastructure/web/unified_dashboard.py
```

### 2. 访问界面

启动成功后，可以通过以下地址访问：

- **主界面**: http://localhost:8080
- **API文档**: http://localhost:8080/api/docs
- **ReDoc文档**: http://localhost:8080/api/redoc

### 3. 命令行参数

```bash
python scripts/web/start_unified_dashboard.py [选项]

选项:
  --host HOST          监听主机地址 (默认: 0.0.0.0)
  --port PORT          监听端口 (默认: 8080)
  --reload             启用自动重载 (开发模式)
  --workers WORKERS    工作进程数 (默认: 1)
  --log-level LEVEL    日志级别 (debug/info/warning/error)
  --config CONFIG      配置文件路径
  --env ENV            运行环境 (development/testing/production)
```

## 架构设计

### 模块化架构

```
统一Web管理界面
├── 前端层 (HTML/CSS/JavaScript)
│   ├── 主界面 (dashboard.html)
│   ├── 模块页面 (各模块特定页面)
│   └── 组件库 (Tailwind CSS + Chart.js)
├── API层 (FastAPI)
│   ├── 统一仪表板 (UnifiedDashboard)
│   ├── 模块路由 (各模块API)
│   └── WebSocket (实时通信)
├── 模块层 (模块化组件)
│   ├── 基础模块 (BaseModule)
│   ├── 配置管理 (ConfigModule)
│   ├── 策略管理 (StrategyModule)
│   ├── 数据管理 (DataModule)
│   ├── 回测管理 (BacktestModule)
│   ├── 监控告警 (MonitoringModule)
│   ├── 资源管理 (ResourceModule)
│   └── 用户管理 (UserModule)
└── 服务层 (业务逻辑)
    ├── 配置管理服务 (UnifiedConfigManager)
    ├── 监控服务 (ApplicationMonitor)
    ├── 资源管理服务 (ResourceManager)
    └── 健康检查服务 (HealthCheck)
```

### 核心组件

#### 1. 统一仪表板 (UnifiedDashboard)
- 系统核心管理界面
- 模块注册和生命周期管理
- WebSocket实时通信
- 统一路由管理

#### 2. 基础模块 (BaseModule)
- 模块化组件的基础类
- 标准化的模块接口
- 统一的配置和状态管理
- 权限验证机制

#### 3. 模块注册表 (ModuleRegistry)
- 模块注册和注销
- 依赖关系管理
- 模块生命周期控制
- 模块状态监控

#### 4. 模块工厂 (ModuleFactory)
- 动态模块创建
- 模块发现机制
- 配置管理
- 模块信息查询

## API接口

### RESTful API

#### 系统接口
```
GET  /api/modules              # 获取模块列表
GET  /api/system/overview      # 获取系统概览
GET  /api/health              # 健康检查
```

#### 配置管理接口
```
GET  /api/config              # 获取配置信息
GET  /api/config/categories   # 获取配置分类
GET  /api/config/category/{category_name}  # 获取分类配置
GET  /api/config/item/{config_key}         # 获取配置项
PUT  /api/config/item/{config_key}         # 更新配置项
POST /api/config/validate     # 验证配置
POST /api/config/export       # 导出配置
POST /api/config/import       # 导入配置
POST /api/config/reload       # 重新加载配置
```

#### 模块特定接口
每个模块提供标准化的API接口：
```
GET  /api/{module}/api/data   # 获取模块数据
GET  /api/{module}/api/status # 获取模块状态
GET  /api/{module}/api/config # 获取模块配置
```

### WebSocket API

```
WS  /ws                       # WebSocket连接端点
```

#### WebSocket消息格式
```json
{
  "type": "subscribe|get_system_info|get_metrics",
  "data": "消息数据"
}
```

## 配置管理

### 配置文件格式

```json
{
  "title": "RQA2025 统一管理平台",
  "version": "1.0.0",
  "theme": "modern",
  "refresh_interval": 30,
  "max_connections": 100,
  "enable_websocket": true,
  "enable_real_time": true
}
```

### 环境配置

#### 开发环境
```bash
export RQA_ENV=development
export LOG_LEVEL=info
export DEBUG=true
```

#### 生产环境
```bash
export RQA_ENV=production
export LOG_LEVEL=warning
export DEBUG=false
```

## 模块开发

### 创建新模块

1. **继承基础模块类**
```python
from src.engine.web.modules.base_module import BaseModule, ModuleConfig

class MyModule(BaseModule):
    def _register_routes(self):
        # 注册模块路由
        pass
    
    async def get_module_data(self):
        # 返回模块数据
        return {"data": "value"}
    
    async def get_module_status(self):
        # 返回模块状态
        return {"status": "running"}
    
    async def validate_permissions(self, user_permissions):
        # 验证用户权限
        return "read" in user_permissions
    
    async def _initialize_module(self):
        # 初始化模块
        pass
    
    async def _start_module(self):
        # 启动模块
        pass
    
    async def _stop_module(self):
        # 停止模块
        pass
    
    @classmethod
    def get_default_config(cls) -> ModuleConfig:
        return ModuleConfig(
            name="my_module",
            display_name="我的模块",
            description="模块描述",
            icon="icon",
            route="/my_module",
            permissions=["read", "write"]
        )
```

2. **注册模块**
```python
from src.engine.web.modules import ModuleRegistry

registry = ModuleRegistry()
config = MyModule.get_default_config()
registry.register_module(MyModule, config)
```

### 模块生命周期

1. **注册**: 模块注册到注册表
2. **初始化**: 模块内部资源初始化
3. **启动**: 模块开始运行
4. **运行**: 模块正常工作
5. **停止**: 模块停止运行
6. **注销**: 模块从注册表移除

## 前端开发

### 界面组件

#### 主界面 (dashboard.html)
- 系统概览
- 模块导航
- 实时监控
- 最近活动

#### 模块页面
每个模块可以有自己的特定页面，通过模板系统渲染。

### 实时数据更新

```javascript
// 建立WebSocket连接
const ws = new WebSocket(`ws://${window.location.host}/ws`);

// 订阅数据
ws.send(JSON.stringify({
    type: 'subscribe',
    data: 'system_metrics'
}));

// 接收数据
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    updateDashboard(data);
};
```

## 部署指南

### 开发环境部署

1. **安装依赖**
```bash
pip install fastapi uvicorn jinja2 pydantic
```

2. **启动服务**
```bash
python scripts/web/start_unified_dashboard.py --reload
```

### 生产环境部署

1. **使用Gunicorn**
```bash
gunicorn src.infrastructure.web.unified_dashboard:create_dashboard() \
    --bind 0.0.0.0:8080 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker
```

2. **使用Docker**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["python", "scripts/web/start_unified_dashboard.py", "--env", "production"]
```

### Nginx配置

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## 监控和日志

### 日志配置

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dashboard.log'),
        logging.StreamHandler()
    ]
)
```

### 健康检查

```bash
# 检查服务状态
curl http://localhost:8080/api/health

# 检查模块状态
curl http://localhost:8080/api/modules
```

## 故障排除

### 常见问题

1. **端口被占用**
```bash
# 查看端口占用
netstat -tulpn | grep 8080

# 杀死进程
kill -9 <PID>
```

2. **模块加载失败**
- 检查模块依赖是否正确安装
- 查看日志文件获取详细错误信息
- 验证模块配置文件格式

3. **WebSocket连接失败**
- 检查防火墙设置
- 验证代理配置
- 确认WebSocket支持

### 调试模式

```bash
# 启用调试模式
python scripts/web/start_unified_dashboard.py --log-level debug --reload
```

## 测试

### 运行单元测试

```bash
# 运行所有测试
pytest tests/unit/infrastructure/web/ -v

# 运行特定测试
pytest tests/unit/infrastructure/web/test_unified_dashboard.py -v

# 生成覆盖率报告
pytest tests/unit/infrastructure/web/ --cov=src.infrastructure.web --cov-report=html
```

### 集成测试

```bash
# 启动测试服务
python scripts/web/start_unified_dashboard.py --env testing

# 运行集成测试
pytest tests/integration/web/ -v
```

## 贡献指南

### 开发流程

1. **Fork项目**
2. **创建功能分支**
3. **编写代码和测试**
4. **提交Pull Request**

### 代码规范

- 遵循PEP 8代码风格
- 编写完整的文档字符串
- 添加适当的类型注解
- 确保测试覆盖率

### 提交规范

```
feat: 添加新功能
fix: 修复bug
docs: 更新文档
style: 代码格式调整
refactor: 代码重构
test: 添加测试
chore: 构建过程或辅助工具的变动
```

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 邮箱: your-email@example.com
- 文档: [项目文档](https://your-docs-url.com) 