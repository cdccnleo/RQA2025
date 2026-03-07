# RQA2025 本地开发环境说明

## 🚀 服务架构

RQA2025采用前后端分离架构：

### 后端服务 (FastAPI)
- **端口**: 8000
- **功能**: RESTful API、数据处理、业务逻辑
- **启动命令**: `python scripts/start_production.py`
- **状态**: ✅ 已启动 (PID: 20724)

### 前端服务 (静态文件服务器)
- **端口**: 8080
- **功能**: HTML/CSS/JS 静态文件服务
- **启动命令**: `cd web-static && python -m http.server 8080`
- **状态**: ✅ 已启动 (PID: 31448)

## 🌐 访问地址

### 前端页面
- **主页**: http://localhost:8080
- **数据源配置**: http://localhost:8080/data-sources-config.html
- **系统仪表板**: http://localhost:8080/rqa2025-dashboard.html
- **策略构思**: http://localhost:8080/strategy-conception.html
- **策略回测**: http://localhost:8080/strategy-backtest.html
- **交易执行**: http://localhost:8080/trading-execution.html

### API 端点
- **直接访问**: http://localhost:8000/api/v1/*
- **前端代理**: http://localhost:8080/api/v1/* (⚠️ 本地环境不支持代理)

## 🔧 本地开发注意事项

### API 调用方式

由于本地环境使用Python简单HTTP服务器，不支持代理功能，前端页面需要直接调用后端API。

#### 1. 修改前端代码中的API地址

在前端JavaScript代码中，将API调用地址从相对路径改为绝对路径：

```javascript
// 从 (相对路径 - 用于代理环境)
const apiUrl = '/api/v1/data/sources';

// 改为 (绝对路径 - 用于本地开发)
const apiUrl = window.location.protocol === 'file:'
    ? `http://localhost:8000/api/v1/data/sources`
    : '/api/v1/data/sources';
```

#### 2. 已修复的页面
- ✅ `data-sources-config.html` - 数据源配置页面
- ✅ 其他页面可能需要类似修改

### 数据源配置

当前系统配置了3个数据源：
1. **宏观经济数据** (ID: macrodata) - 财经新闻类型
2. **新浪财经** (ID: sinafinance) - 财经新闻类型
3. **测试数据源** (ID: testsource001) - 财经新闻类型

配置文件位置: `data/data_sources_config.json`

## 📊 服务状态监控

### 健康检查
- **后端健康检查**: http://localhost:8000/health
- **前端健康检查**: http://localhost:8080/health (静态响应)

### 进程监控
```bash
# 查看运行进程
tasklist | findstr python
tasklist | findstr node

# 查看端口占用
netstat -ano | findstr :8000
netstat -ano | findstr :8080
```

## 🔄 开发工作流

### 1. 启动服务
```bash
# 启动后端API服务
python scripts/start_production.py

# 启动前端静态服务器
cd web-static && python -m http.server 8080
```

### 2. 开发调试
```bash
# 修改后端代码后重启服务
# Ctrl+C 停止后端服务，然后重新启动

# 修改前端代码后刷新浏览器
# 前端文件修改后直接刷新页面即可
```

### 3. API测试
```bash
# 测试数据源API
curl http://localhost:8000/api/v1/data/sources

# 测试健康检查
curl http://localhost:8000/health
```

## 🐛 常见问题

### 端口冲突
如果端口被占用，停止相关进程：
```bash
# 查找占用进程
netstat -ano | findstr :8000
netstat -ano | findstr :8080

# 停止进程
taskkill /f /pid <PID>
```

### API调用失败
- 确保后端服务正在运行 (端口8000)
- 检查前端代码中的API地址配置
- 查看后端日志输出

### 前端页面无法加载
- 确保前端服务正在运行 (端口8080)
- 检查浏览器控制台的错误信息
- 确认静态文件路径正确

## 📁 项目结构

```
RQA2025/
├── scripts/                 # 启动脚本
│   └── start_production.py  # 后端服务启动脚本
├── src/                     # 后端源代码
├── web-static/              # 前端静态文件
│   ├── *.html              # 页面文件
│   ├── nginx.conf          # Nginx配置（容器环境）
│   └── ...                 # 其他静态资源
├── data/                    # 数据文件
│   └── data_sources_config.json  # 数据源配置
└── README_LOCAL_DEVELOPMENT.md   # 本文档
```

## 🎯 部署建议

### 生产环境
使用Nginx + Gunicorn/Uvicorn：
- Nginx作为反向代理和静态文件服务器
- Gunicorn/Uvicorn运行FastAPI应用
- 支持API代理和WebSocket

### 开发环境
继续使用当前设置：
- Python http.server提供静态文件
- 直接调用后端API（不使用代理）

## 📞 技术支持

如遇到问题，请检查：
1. 服务启动状态
2. 端口占用情况
3. 日志输出信息
4. 浏览器开发者工具控制台
