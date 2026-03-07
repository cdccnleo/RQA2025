# Web管理界面使用指南

## 概述

Web管理界面是RQA2025项目基础设施层的可视化管理系统，提供直观的Web界面来监控系统状态、管理告警规则、配置系统参数和跟踪测试执行。该系统基于Flask框架构建，支持实时数据更新和响应式设计。

## 核心功能

### 1. 系统概览
- **实时监控**: 显示CPU、内存、磁盘、网络等系统资源使用情况
- **状态指示**: 监控系统运行状态、活跃测试数量、告警数量等关键指标
- **快速操作**: 提供系统启动/停止、测试注册等常用功能

### 2. 监控仪表板
- **性能图表**: 实时显示CPU、内存使用率趋势图
- **资源监控**: 系统资源使用情况的环形图展示
- **指标详情**: 详细的性能指标统计表格，包含当前值、平均值、最大值、最小值
- **实时更新**: 每5秒自动刷新数据，支持手动刷新

### 3. 告警管理
- **告警统计**: 显示活跃告警、严重告警、警告告警、已解决告警的数量
- **告警列表**: 详细的告警信息表格，包含类型、级别、消息、来源、时间等
- **规则管理**: 添加、编辑、删除告警规则，支持多种告警类型和级别
- **告警处理**: 支持告警解决、状态更新等操作

### 4. 配置管理
- **系统配置**: 性能监控间隔、历史数据保存、网络检测等参数设置
- **通知配置**: 邮件和Webhook通知的详细配置
- **日志配置**: 日志级别、文件路径、备份策略等设置
- **配置导入导出**: 支持JSON和YAML格式的配置文件导入导出

### 5. 测试管理
- **测试注册**: 注册新的测试用例，指定测试ID和名称
- **活跃测试**: 显示当前正在执行的测试，支持状态更新
- **测试历史**: 查看已完成的测试记录，包含执行时间和结果
- **统计图表**: 测试执行统计趋势图和状态分布饼图

## 系统架构

### 技术栈
- **后端框架**: Flask + Flask-SocketIO
- **前端技术**: HTML5 + CSS3 + JavaScript + Bootstrap 5
- **图表库**: Chart.js
- **实时通信**: WebSocket (Socket.IO)
- **模板引擎**: Jinja2

### 组件结构
```
WebManagementInterface
├── Flask应用 (Flask App)
├── SocketIO服务器 (Real-time updates)
├── 路由注册 (Route registration)
├── API接口 (RESTful APIs)
├── 后台更新线程 (Background update thread)
└── 模板渲染 (Template rendering)
```

### 数据流
1. **监控系统** → **后台线程** → **SocketIO** → **前端实时更新**
2. **用户操作** → **前端JavaScript** → **API接口** → **监控系统**
3. **系统状态** → **API接口** → **前端显示**

## 安装和部署

### 环境要求
- Python 3.8+
- Flask 2.0+
- Flask-SocketIO 5.0+
- 监控告警系统组件

### 安装依赖
```bash
pip install flask flask-socketio
```

### 快速启动
```python
from src.infrastructure.performance import create_web_interface

# 创建Web界面实例
interface = create_web_interface()

# 启动服务器
interface.run(host='0.0.0.0', port=5000, debug=True)
```

### 生产部署
```python
# 生产环境配置
interface.run(
    host='0.0.0.0',
    port=5000,
    debug=False
)
```

## 使用说明

### 1. 访问系统
- 打开浏览器访问 `http://localhost:5000`
- 系统会自动加载当前监控状态和性能数据

### 2. 导航菜单
- **首页**: 系统概览和快速操作
- **监控仪表板**: 详细的性能监控图表
- **告警管理**: 告警查看和规则配置
- **配置管理**: 系统参数设置
- **测试管理**: 测试用例管理

### 3. 实时监控
- 系统状态每5秒自动更新
- 性能图表实时显示最新数据
- 告警信息实时推送
- 支持手动刷新数据

### 4. 告警配置
- 点击"添加规则"按钮创建新规则
- 选择告警类型和级别
- 设置触发条件和阈值
- 配置冷却时间和通知方式

### 5. 测试管理
- 在"测试管理"页面注册新测试
- 监控测试执行状态
- 更新测试结果和错误信息
- 查看测试执行历史

## API接口

### 系统状态
```http
GET /api/status
```
返回系统运行状态、活跃测试数量、告警数量等信息。

### 性能数据
```http
GET /api/performance?minutes=60
```
返回指定时间范围内的性能指标数据。

### 告警管理
```http
GET /api/alerts                    # 获取告警列表
POST /api/alerts/<id>/resolve      # 解决告警
GET /api/alerts/rules              # 获取告警规则
POST /api/alerts/rules             # 添加告警规则
```

### 测试管理
```http
GET /api/tests                     # 获取测试信息
POST /api/tests/register           # 注册新测试
PUT /api/tests/<id>/status         # 更新测试状态
```

### 系统控制
```http
POST /api/system/start             # 启动系统
POST /api/system/stop              # 停止系统
```

### 配置管理
```http
GET /api/config                    # 获取配置信息
POST /api/config/system            # 更新系统配置
POST /api/config/notifications     # 更新通知配置
POST /api/config/logging           # 更新日志配置
GET /api/config/export             # 导出配置
POST /api/config/import            # 导入配置
```

## 配置选项

### 系统配置
- `update_interval`: 性能监控更新间隔(秒)
- `metrics_history_size`: 历史指标保存数量
- `network_latency_check_url`: 网络延迟检测URL
- `network_latency_timeout`: 网络延迟检测超时时间
- `alert_check_interval`: 告警检查间隔
- `performance_report_interval`: 性能报告生成间隔

### 通知配置
- **邮件通知**: SMTP服务器、端口、TLS设置、发件人、收件人
- **Webhook通知**: 启用状态、回调URL

### 日志配置
- `level`: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `file`: 日志文件路径
- `max_size`: 日志文件最大大小(MB)
- `backup_count`: 日志备份数量
- `format`: 日志格式

## 最佳实践

### 1. 性能优化
- 合理设置监控更新间隔，避免过于频繁的数据采集
- 定期清理历史数据，控制内存使用
- 使用生产环境的Web服务器(如Gunicorn)部署

### 2. 安全配置
- 在生产环境中设置强密码和访问控制
- 启用HTTPS加密传输
- 限制API访问权限和频率

### 3. 监控告警
- 设置合理的告警阈值，避免误报
- 配置多种通知方式，确保告警及时送达
- 定期检查和优化告警规则

### 4. 数据备份
- 定期导出配置文件
- 备份重要的监控数据和告警规则
- 建立配置版本管理机制

## 故障排除

### 常见问题

#### 1. 页面无法访问
- 检查Flask服务是否正常启动
- 确认端口是否被占用
- 检查防火墙设置

#### 2. 实时数据不更新
- 检查后台更新线程是否正常运行
- 确认SocketIO连接是否建立
- 查看浏览器控制台错误信息

#### 3. 告警规则不生效
- 验证规则配置是否正确
- 检查告警检查间隔设置
- 确认监控系统是否正常运行

#### 4. 性能数据异常
- 检查系统资源使用情况
- 验证监控组件是否正常工作
- 查看系统日志错误信息

### 日志分析
```python
import logging

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

# 查看Web界面日志
logger = logging.getLogger('src.infrastructure.performance.web_management_interface')
```

### 调试模式
```python
# 启用调试模式
interface.run(debug=True)

# 查看详细错误信息
# 启用Flask调试工具栏
```

## 扩展开发

### 添加新页面
1. 在`templates`目录创建HTML模板
2. 在`_register_routes`方法中添加路由
3. 实现对应的API接口
4. 更新导航菜单

### 自定义API
```python
@self.app.route('/api/custom')
def api_custom():
    try:
        # 实现自定义逻辑
        result = self.custom_function()
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

### 添加实时功能
```python
# 在后台更新线程中发送自定义事件
self.socketio.emit('custom_event', custom_data)

# 在前端接收事件
socket.on('custom_event', function(data) {
    // 处理自定义数据
});
```

## 总结

Web管理界面为RQA2025项目提供了完整的可视化监控和管理解决方案。通过直观的Web界面，用户可以：

- 实时监控系统性能和资源使用情况
- 高效管理告警规则和通知配置
- 便捷地跟踪和管理测试执行
- 灵活配置系统参数和监控策略

系统采用现代化的Web技术栈，支持实时数据更新和响应式设计，为运维人员提供了强大的监控和管理工具。通过合理的配置和最佳实践，可以构建稳定、高效的监控管理系统。
