# RQA2025 引擎层Web管理界面文档索引

## 概述

本文档索引列出了RQA2025引擎层统一Web管理界面的所有相关文档和脚本，方便开发者快速查找和使用。

## 文档结构

### 📁 架构文档
```
docs/architecture/engine/web/
├── README.md                           # 主要使用指南
├── INDEX.md                           # 本文档索引
├── web_engine_architecture.md         # 引擎层Web架构设计
└── unified_dashboard_progress_2025.md # 推进报告
```

### 📁 脚本文件
```
scripts/engine/web/
├── start_dashboard.py                  # 启动脚本
└── access_dashboard.py                # 快速访问脚本
```

### 📁 源代码
```
src/engine/web/
├── unified_dashboard.py               # 主程序
├── modules/                           # 模块化组件
│   ├── base_module.py                # 基础模块抽象
│   ├── module_registry.py            # 模块注册表
│   ├── module_factory.py             # 模块工厂
│   ├── config_module.py              # 配置管理模块
│   ├── fpga_module.py                # FPGA监控模块
│   ├── resource_module.py            # 资源监控模块
│   └── features_module.py            # 特征监控模块
├── templates/                         # 前端模板
│   └── dashboard.html                # 主仪表板模板
└── static/                           # 静态资源
    ├── css/
    │   └── dashboard.css             # 主样式文件
    └── js/
        └── dashboard.js              # 主脚本文件
```

## 快速导航

### 🚀 快速开始
1. **启动服务**: `python scripts/engine/web/start_dashboard.py`
2. **访问界面**: `python scripts/engine/web/access_dashboard.py`
3. **查看文档**: [README.md](README.md)

### 📖 详细文档
- **[README.md](README.md)**: 完整的使用指南和开发文档
- **[web_engine_architecture.md](web_engine_architecture.md)**: 详细的架构设计文档
- **[unified_dashboard_progress_2025.md](unified_dashboard_progress_2025.md)**: 推进报告和状态更新

### 🔧 开发资源
- **API文档**: http://127.0.0.1:8081/api/docs
- **主界面**: http://127.0.0.1:8081
- **WebSocket**: ws://127.0.0.1:8081/ws

## 功能模块

### 已集成模块
1. **配置管理模块** (`config`)
   - 路由: `/api/modules/config`
   - 功能: 统一配置管理和热重载

2. **FPGA监控模块** (`fpga_monitoring`)
   - 路由: `/api/modules/fpga_monitoring`
   - 功能: FPGA性能监控和告警管理

3. **资源监控模块** (`resource_monitoring`)
   - 路由: `/api/modules/resource_monitoring`
   - 功能: 系统资源使用情况监控

4. **特征监控模块** (`features_monitoring`)
   - 路由: `/api/modules/features_monitoring`
   - 功能: 特征工程性能和数据质量监控

## 技术特性

### 后端特性
- ✅ FastAPI现代化Web框架
- ✅ WebSocket实时通信
- ✅ 模块化架构设计
- ✅ 自动API文档生成
- ✅ 数据验证和序列化

### 前端特性
- ✅ 响应式设计
- ✅ 实时数据可视化
- ✅ 现代化UI组件
- ✅ WebSocket实时更新
- ✅ 模块化JavaScript

### 部署特性
- ✅ 端口自动检测
- ✅ 进程管理
- ✅ 编码问题修复
- ✅ 依赖自动检查
- ✅ 环境配置管理

## 开发指南

### 添加新模块
1. 继承 `BaseModule` 基类
2. 实现必要的抽象方法
3. 在 `ModuleRegistry` 中注册
4. 配置模块路由

### 自定义API
1. 在模块中定义路由
2. 实现业务逻辑
3. 添加数据验证
4. 编写测试用例

### 前端开发
1. 创建HTML模板
2. 编写JavaScript逻辑
3. 设计CSS样式
4. 集成数据可视化

## 配置管理

### 主配置文件
- **位置**: `config/web_dashboard_config.json`
- **功能**: 统一管理所有配置

### 环境变量
- `RQA_ENV`: 运行环境 (development/production)
- `RQA_DASHBOARD_PORT`: 服务端口
- `PYTHONIOENCODING`: 编码设置
- `PYTHONUTF8`: UTF-8编码支持

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
1. **端口冲突**: 使用 `netstat -ano | findstr :8080` 检查
2. **依赖问题**: 运行 `pip install fastapi uvicorn websockets psutil`
3. **编码问题**: 设置 `PYTHONIOENCODING=utf-8` 和 `PYTHONUTF8=1`

### 调试模式
```bash
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

## 相关链接

- **项目主页**: [RQA2025项目](https://github.com/your-org/rqa2025)
- **API文档**: http://127.0.0.1:8081/api/docs
- **问题反馈**: [Issues](https://github.com/your-org/rqa2025/issues)
- **贡献指南**: [CONTRIBUTING.md](CONTRIBUTING.md)

## 联系方式

如有问题或建议，请联系开发团队：
- **邮箱**: dev-team@rqa2025.com
- **文档**: [docs/architecture/engine/web/](.)
- **源码**: [src/engine/web/](../../../src/engine/web/) 