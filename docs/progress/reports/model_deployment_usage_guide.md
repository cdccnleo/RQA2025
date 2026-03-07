# RQA2025 模型落地使用指南

## 📋 概述

本指南详细说明如何使用RQA2025项目的自动化脚本进行模型落地部署。整个部署过程分为四个阶段，每个阶段都有相应的自动化脚本支持。

## 🚀 快速开始

### 1. 一键完整部署

```bash
# 执行完整的模型落地部署流程
python scripts/model_deployment_controller.py --full
```

这将自动执行所有四个阶段：
- 第一阶段：基础设施完善
- 第二阶段：模型服务化
- 第三阶段：系统集成
- 第四阶段：生产部署

### 2. 分阶段部署

```bash
# 执行第一阶段：基础设施完善
python scripts/model_deployment_controller.py --phase phase1

# 执行第二阶段：模型服务化
python scripts/model_deployment_controller.py --phase phase2

# 执行第三阶段：系统集成
python scripts/model_deployment_controller.py --phase phase3

# 执行第四阶段：生产部署
python scripts/model_deployment_controller.py --phase phase4
```

## 📊 进度监控

### 1. 更新进度数据

```bash
# 更新测试覆盖率和环境检查数据
python scripts/progress_monitor.py --update
```

### 2. 生成进度报告

```bash
# 生成详细的进度报告
python scripts/progress_monitor.py --report
```

### 3. 查看当前进度

```bash
# 查看当前进度状态
python scripts/progress_monitor.py
```

## 🔍 环境检查

### 1. 生产环境检查

```bash
# 检查生产环境配置
python scripts/environment_checker.py --env production
```

### 2. 测试覆盖率分析

```bash
# 分析各层测试覆盖率
python scripts/test_coverage_analyzer.py --target 80

# 生成HTML报告
python scripts/test_coverage_analyzer.py --target 80 --html
```

## 🚀 自动化部署

### 1. 一键部署

```bash
# 自动部署到生产环境
python scripts/auto_deployment.py --env production

# 禁用自动回滚
python scripts/auto_deployment.py --env production --no-rollback
```

### 2. 部署验证

```bash
# 验证部署结果
python scripts/deployment_validator.py --comprehensive
```

## 📋 详细使用说明

### 第一阶段：基础设施完善

#### 目标
- 各层测试覆盖率达到目标值（基础设施层90%，其他层80%）
- 环境配置自动化
- 依赖管理完善

#### 执行步骤

1. **环境检查**
```bash
python scripts/environment_checker.py --env production
```

2. **测试覆盖率分析**
```bash
python scripts/test_coverage_analyzer.py --target 80
```

3. **修复测试问题**
```bash
# 根据分析结果手动修复测试问题
# 或使用自动修复脚本（如果可用）
```

4. **环境配置自动化**
```bash
# 生成生产环境配置
python scripts/config_generator.py --env production

# 检查依赖
python scripts/dependency_checker.py --strict
```

### 第二阶段：模型服务化

#### 目标
- 将模型封装为RESTful API服务
- 将特征工程封装为独立服务
- 实现服务健康检查和性能优化

#### 执行步骤

1. **模型推理服务部署**
```bash
python scripts/model_servicizer.py --model lstm_v1 --port 8000
```

2. **特征处理服务部署**
```bash
python scripts/feature_service_deploy.py --workers 4
```

3. **服务健康检查**
```bash
python scripts/service_health_check.py --service model_api
```

4. **性能基准测试**
```bash
python scripts/performance_benchmark.py --model lstm_v1
```

### 第三阶段：系统集成

#### 目标
- 实现完整的端到端业务流程
- 建立监控和告警体系
- 完成集成性能测试

#### 执行步骤

1. **端到端集成测试**
```bash
python scripts/e2e_integration_test.py --scenarios all
```

2. **业务流程验证**
```bash
python scripts/business_flow_validator.py --strict
```

3. **监控系统部署**
```bash
python scripts/monitoring_deploy.py --components all
```

4. **告警规则配置**
```bash
python scripts/alert_rules_config.py --rules production
```

5. **集成性能测试**
```bash
python scripts/integration_performance_test.py --load 1000
```

### 第四阶段：生产部署

#### 目标
- 实现生产环境的一键部署
- 完成部署验证和性能测试
- 配置生产环境监控和运维自动化

#### 执行步骤

1. **生产环境部署**
```bash
python scripts/auto_deployment.py --env production --auto
```

2. **部署验证**
```bash
python scripts/deployment_validator.py --comprehensive
```

3. **性能测试**
```bash
python scripts/production_performance_test.py --duration 1h
```

4. **监控配置**
```bash
python scripts/grafana_dashboard_import.py --dashboards all
```

5. **运维自动化**
```bash
python scripts/ops_automation.py --tasks backup,monitor,cleanup
```

## 📈 监控和报告

### 1. 实时监控

```bash
# 启动监控系统
python scripts/monitoring_deploy.py --components all

# 查看服务状态
docker-compose -f deploy/docker-compose.yml ps

# 查看服务日志
docker-compose -f deploy/docker-compose.yml logs -f
```

### 2. 性能监控

```bash
# 运行性能测试
python scripts/performance_test.py --duration 30m

# 生成性能报告
python scripts/performance_report.py --output reports/performance_report.html
```

### 3. 健康检查

```bash
# 检查API服务健康状态
curl http://localhost:8000/health

# 检查推理服务健康状态
curl http://localhost:8001/health

# 检查监控系统状态
curl http://localhost:9090/-/healthy
```

## 🔧 故障排除

### 1. 常见问题

#### 环境检查失败
```bash
# 检查Python环境
python --version

# 检查conda环境
conda info --envs

# 检查依赖包
pip list | grep -E "(numpy|pandas|scikit-learn)"
```

#### 测试覆盖率过低
```bash
# 运行特定层的测试
python -m pytest tests/unit/infrastructure/ --cov=src/infrastructure

# 生成详细覆盖率报告
python -m pytest --cov=src --cov-report=html
```

#### 服务启动失败
```bash
# 检查Docker服务状态
docker --version
docker-compose --version

# 检查端口占用
netstat -tlnp | grep -E ':(8000|8001|9090)'

# 查看服务日志
docker-compose -f deploy/docker-compose.yml logs
```

### 2. 回滚操作

```bash
# 自动回滚到上一个版本
python scripts/auto_rollback.py

# 手动回滚
docker-compose -f deploy/docker-compose.yml down
docker tag rqa2025:backup rqa2025:latest
docker-compose -f deploy/docker-compose.yml up -d
```

### 3. 日志分析

```bash
# 查看应用日志
tail -f logs/rqa2025.log

# 查看错误日志
tail -f logs/rqa2025_error.log

# 分析日志
python scripts/log_analyzer.py --real_time
```

## 📊 报告和文档

### 1. 生成报告

```bash
# 生成测试覆盖率报告
python scripts/test_coverage_analyzer.py --html

# 生成环境检查报告
python scripts/environment_checker.py --output reports/env_check.html

# 生成部署报告
python scripts/auto_deployment.py --output reports/deployment_report.json
```

### 2. 查看报告

- **测试覆盖率报告**: `reports/coverage_report_*.html`
- **环境检查报告**: `reports/environment_check_*.json`
- **部署报告**: `reports/deployment_report_*.json`
- **进度报告**: `reports/progress_report_*.html`

## 🔄 持续集成

### 1. 自动化流水线

```yaml
# .github/workflows/model_deployment.yml
name: Model Deployment Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Tests
        run: python scripts/test_coverage_analyzer.py --target 80
      - name: Environment Check
        run: python scripts/environment_checker.py --env production

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Production
        run: python scripts/model_deployment_controller.py --full
```

### 2. 定时任务

```bash
# 每天更新进度
0 9 * * * python scripts/progress_monitor.py --update

# 每周生成报告
0 10 * * 1 python scripts/progress_monitor.py --report

# 每天健康检查
0 */6 * * * python scripts/health_check.py --timeout 300
```

## 📞 技术支持

### 1. 获取帮助

```bash
# 查看脚本帮助
python scripts/model_deployment_controller.py --help
python scripts/environment_checker.py --help
python scripts/test_coverage_analyzer.py --help
```

### 2. 联系支持

- **文档**: 查看 `docs/progress/reports/` 目录下的详细文档
- **日志**: 检查 `logs/` 目录下的日志文件
- **报告**: 查看 `reports/` 目录下的各种报告

### 3. 反馈和建议

如有问题或建议，请：
1. 查看相关日志文件
2. 生成详细的错误报告
3. 联系技术支持团队

---

**最后更新**: 2025-01-19  
**版本**: v1.0  
**维护状态**: ✅ 活跃维护中 