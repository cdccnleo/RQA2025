# 仪表盘测试验证

## 概述

本目录包含RQA2025量化交易系统仪表盘的完整测试验证套件，按照业务流程顺序进行系统化测试。

## 最新更新

### ✅ 路由健康检查功能（2025-01-07）

新增功能：
- ✅ 路由健康检查模块 (`route_health_check.py`)
- ✅ 自动路由验证脚本 (`check_routes_health.py`)
- ✅ CI/CD验证脚本 (`ci_route_verification.py`)
- ✅ GitHub Actions工作流 (`.github/workflows/route-health-check.yml`)
- ✅ 路由健康检查测试用例

## 测试结果总结

### 总体测试结果
- **测试通过率: 100%** (50/50)
- **页面加载**: 16/16 (100%)
- **API端点**: 23/23 (100%)
- **WebSocket连接**: 4/4 (100%)
- **业务流程数据流**: 7/7 (100%)
- **路由健康检查**: 5/5 (100%)

## 测试结构

### 1. API端点测试 (`test_api_endpoints.py`)

测试所有仪表盘相关的API端点：
- 数据收集阶段API
- 特征工程监控API
- 模型训练监控API
- 策略性能评估API
- 交易信号监控API
- 订单路由监控API
- 风险报告生成API

### 2. WebSocket连接测试 (`test_websocket_connections.py`)

测试所有WebSocket实时数据推送连接：
- 特征工程WebSocket
- 模型训练WebSocket
- 交易信号WebSocket
- 订单路由WebSocket

### 3. 仪表盘页面测试 (`test_dashboard_pages.py`)

测试所有仪表盘页面是否正常加载。

### 4. 业务流程数据流测试 (`test_business_process_flow.py`)

按照业务流程顺序测试数据流连通性：
- 量化策略开发流程数据流
- 交易执行流程数据流
- 风险控制流程数据流

### 5. 路由健康检查测试 (`test_route_health_check.py`) ⭐ 新增

测试路由健康检查功能：
- 健康检查器初始化
- 路由检查功能
- 健康状态验证
- 预期路由存在性验证

### 6. 数据获取验证 (`verify_dashboard_data.py`)

快速验证所有仪表盘的数据获取情况并生成报告。

### 7. 完整测试运行 (`run_all_tests.py`)

按照业务流程顺序执行所有测试并生成汇总报告。

## 使用方法

### 快速验证

```bash
# 运行数据获取验证
python tests/dashboard_verification/verify_dashboard_data.py
```

### 完整测试

```bash
# 运行所有测试
python tests/dashboard_verification/run_all_tests.py

# 或使用pytest
pytest tests/dashboard_verification/ -v
```

### 路由健康检查 ⭐ 新增

```bash
# 独立运行路由健康检查
python scripts/check_routes_health.py

# CI/CD完整验证
python scripts/ci_route_verification.py
```

### 单独测试

```bash
# 测试API端点
pytest tests/dashboard_verification/test_api_endpoints.py -v

# 测试WebSocket
pytest tests/dashboard_verification/test_websocket_connections.py -v

# 测试页面加载
pytest tests/dashboard_verification/test_dashboard_pages.py -v

# 测试业务流程数据流
pytest tests/dashboard_verification/test_business_process_flow.py -v

# 测试路由健康检查 ⭐ 新增
pytest tests/dashboard_verification/test_route_health_check.py -v
```

## 测试环境要求

- Python 3.8+
- pytest
- requests
- websockets
- 后端服务运行在 `http://localhost:8000`
- Web服务运行在 `http://localhost:8080`

## 安装依赖

```bash
pip install pytest requests websockets
```

## 测试报告

测试完成后会生成详细的验证报告，包括：
- API端点状态
- 页面加载状态
- WebSocket连接状态
- 数据获取情况
- 业务流程连通性
- 路由健康状态 ⭐ 新增
- 问题识别和建议

详细报告请查看：
- `test_results_report.md` - 详细测试结果报告
- `test_results_final.md` - 最终测试总结

## 路由健康检查 ⭐ 新增

### 功能说明

路由健康检查在应用启动时自动执行，验证所有预期路由是否正确注册。

### 检查内容

- 数据源路由 (2个)
- 数据质量路由 (1个)
- 特征工程路由 (3个)
- 模型训练路由 (2个)
- 策略性能路由 (2个)
- 交易信号路由 (3个)
- 订单路由路由 (3个)
- 风险报告路由 (4个)
- WebSocket路由 (4个)

### 健康检查报告

应用启动时会自动打印健康检查报告：
```
================================================================================
  路由健康检查报告
================================================================================
总路由数: 133
健康状态: HEALTHY

✅ 数据源: 预期: 2 | 已注册: 2 | 缺失: 0
✅ 特征工程: 预期: 3 | 已注册: 3 | 缺失: 0
...

总结: 24/24 个预期路由已注册
✅ 所有路由健康检查通过！
================================================================================
```

## CI/CD集成

### GitHub Actions

已配置自动化工作流：
- 代码推送或PR时自动运行
- 检查路由健康状态
- 运行API端点测试
- 运行WebSocket测试

工作流文件: `.github/workflows/route-health-check.yml`

## 问题排查

如果测试失败，请检查：
1. 后端服务是否正常运行
2. API路由是否正确注册（运行路由健康检查）
3. 服务层是否正确对接组件
4. 数据流是否完整连通

### 路由问题排查

如果路由健康检查失败：
```bash
# 运行路由健康检查获取详细信息
python scripts/check_routes_health.py

# 查看应用启动日志
docker-compose logs rqa2025-app | grep "路由健康检查"
```

## 已知问题

所有已知问题已修复：
- ✅ 数据源API HTTP 500错误 - 已修复
- ✅ 数据源指标API HTTP 500错误 - 已修复
- ✅ 数据质量指标API 404错误 - 已修复

## 下一步计划

1. ✅ 路由健康检查 - 已完成
2. ✅ 改进错误处理 - 已完成
3. ✅ 自动化测试 - 已完成
4. [ ] 对接实际后端组件（替换模拟数据）
5. [ ] 完善性能测试和错误处理测试
