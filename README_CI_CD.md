# RQA2025 CI/CD 流水线指南

## 概述

RQA2025 系统采用现代化的CI/CD实践，通过GitHub Actions实现自动化构建、测试、部署和监控。本文档介绍完整的CI/CD流程和使用方法。

## 🏗️ CI/CD 架构

### 流水线阶段

```mermaid
graph LR
    A[代码提交] --> B[代码质量检查]
    B --> C[单元测试]
    C --> D[集成测试]
    D --> E[端到端测试]
    E --> F[性能测试]
    F --> G[安全扫描]
    G --> H[质量报告]
    H --> I{部署检查}
    I --> J[测试环境部署]
    J --> K[生产环境部署]
```

### 质量门禁

系统设置了严格的质量门禁标准：

- **测试覆盖率**: ≥70% (核心模块 ≥75%)
- **测试成功率**: ≥80%
- **响应时间**: <2秒
- **内存使用**: <85%
- **安全问题**: 0个严重/高风险问题

## 🚀 快速开始

### 1. 本地开发环境

```bash
# 安装依赖
pip install -r requirements.txt

# 运行代码质量检查
flake8 src/ tests/
black --check src/ tests/
isort --check-only src/ tests/

# 运行测试
pytest tests/unit/ --cov=src --cov-report=html

# 运行质量门禁检查
python scripts/quality_gate.py
```

### 2. 提交代码

```bash
# 添加文件
git add .

# 提交代码 (会触发CI)
git commit -m "feat: 添加新功能"

# 推送到主分支
git push origin main
```

## 📋 CI/CD 流水线详解

### 🔍 代码质量检查 (Code Quality)

**触发条件**: 每次推送和PR

**检查内容**:
- 代码风格检查 (flake8)
- 代码格式检查 (black)
- 导入排序检查 (isort)
- 类型检查 (mypy)

**失败处理**: 代码质量不合格，阻止后续流程

### 🧪 单元测试 (Unit Tests)

**触发条件**: 代码质量检查通过

**测试范围**:
- 所有Python版本 (3.8, 3.9, 3.10)
- 核心业务逻辑
- 工具函数和帮助方法

**产物**:
- 覆盖率报告 (`coverage.xml`, HTML报告)
- 测试结果 (`unit-tests.html`)

### 🔗 集成测试 (Integration Tests)

**触发条件**: 单元测试通过

**测试范围**:
- 模块间接口
- 数据库连接
- 缓存系统
- 外部服务集成

**产物**:
- 集成测试报告 (`integration-tests.html`)

### 🌐 端到端测试 (E2E Tests)

**触发条件**: 集成测试通过

**测试范围**:
- 完整业务流程
- 用户界面交互
- 跨系统集成
- 并发负载测试

**产物**:
- E2E测试报告 (`e2e-tests.html`)

### ⚡ 性能测试 (Performance Tests)

**触发条件**: E2E测试通过

**测试内容**:
- 响应时间基准测试
- 内存使用分析
- CPU使用监控
- 并发处理能力

**产物**:
- 性能基准报告 (`benchmark.json`)
- 性能测试报告 (`performance-tests.html`)

### 🔒 安全扫描 (Security Scan)

**触发条件**: 代码质量检查通过

**扫描工具**: Bandit

**检查内容**:
- 常见安全漏洞
- 硬编码凭据
- SQL注入风险
- XSS漏洞

**产物**:
- 安全扫描报告 (`security-results.json`)

### 📊 质量报告生成 (Quality Report)

**触发条件**: 所有测试阶段完成

**报告内容**:
- 测试覆盖率趋势图
- 质量指标雷达图
- 详细的质量分析
- 改进建议

**产物**:
- 综合质量报告 (Markdown)
- 可视化图表 (PNG)

### 🎯 部署检查 (Deployment Check)

**触发条件**: 推送到main分支且所有检查通过

**检查内容**:
- 质量门禁验证
- 部署就绪确认
- 风险评估

**产物**:
- 部署批准通知

### 🚀 自动部署 (Auto Deployment)

**触发条件**: 部署检查通过

**部署流程**:
1. **测试环境部署**
   - 构建部署包
   - 上传到测试服务器
   - 重启服务
   - 健康检查

2. **生产环境部署**
   - 金丝雀部署策略
   - 分批上线
   - 监控验证
   - 自动回滚机制

## 🛠️ 本地工具使用

### 质量门禁检查

```bash
# 运行质量门禁
python scripts/quality_gate.py --project-root . --output quality-report.md

# 仅检查覆盖率
python scripts/quality_gate.py --json-output coverage-check.json
```

### 部署工具

```bash
# 部署到测试环境
python scripts/deploy.py --environment staging

# 部署到生产环境
python scripts/deploy.py --environment production

# 回滚操作
python scripts/deploy.py --rollback previous-version

# 仅执行健康检查
python scripts/deploy.py --health-check
```

### 系统监控

```bash
# 启动监控服务
python scripts/monitor.py

# 执行一次性检查
python scripts/monitor.py --once --report

# 测试告警功能
python scripts/monitor.py --alert-test
```

## 📊 监控和告警

### 系统监控指标

- **CPU使用率**: 实时监控，阈值85%
- **内存使用率**: 实时监控，阈值90%
- **磁盘使用率**: 实时监控，阈值95%
- **服务健康**: HTTP状态检查，响应时间监控
- **业务指标**: 交易成功率，风险覆盖率

### 告警通知

支持多种告警渠道：
- **邮件告警**: SMTP配置
- **Slack告警**: Webhook集成
- **日志记录**: 本地文件记录

### 配置告警规则

```json
{
  "alerts": {
    "email": {
      "enabled": true,
      "smtp_server": "smtp.company.com",
      "sender_email": "alerts@rqa2025.com",
      "receiver_emails": ["team@company.com"]
    },
    "slack": {
      "enabled": true,
      "webhook_url": "https://hooks.slack.com/...",
      "channel": "#rqa2025-alerts"
    }
  }
}
```

## 🔧 配置管理

### 流水线配置

主要配置文件：
- `.github/workflows/ci-cd-pipeline.yml`: GitHub Actions流水线
- `.quality-gate.json`: 质量门禁配置
- `requirements.txt`: Python依赖
- `deploy-config.json`: 部署配置 (可选)

### 环境变量

```bash
# 数据库配置
DATABASE_URL=postgresql://user:pass@host:port/db

# 缓存配置
REDIS_URL=redis://host:port

# 监控配置
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
SMTP_PASSWORD=your_smtp_password

# 部署配置
DEPLOY_HOST=your-server.com
DEPLOY_USER=deploy
```

## 🚨 故障排除

### 常见问题

#### 1. 测试失败
```bash
# 查看详细测试输出
pytest tests/ -v --tb=long

# 运行特定测试
pytest tests/unit/trading/ -k "test_trading_engine"
```

#### 2. 覆盖率不足
```bash
# 查看覆盖率详情
pytest --cov=src --cov-report=html
# 在htmlcov/index.html中查看详情
```

#### 3. 部署失败
```bash
# 检查部署日志
python scripts/deploy.py --dry-run

# 手动部署检查
python scripts/deploy.py --health-check
```

#### 4. 监控告警
```bash
# 检查监控状态
python scripts/monitor.py --once --report

# 查看监控日志
tail -f logs/monitor_$(date +%Y%m%d).log
```

## 📈 持续改进

### 质量指标跟踪

系统会自动跟踪以下指标：
- 测试覆盖率趋势
- 构建成功率
- 部署频率
- 故障恢复时间 (MTTR)
- 变更失败率

### 优化建议

1. **测试策略优化**
   - 增加集成测试覆盖率
   - 完善端到端测试场景
   - 引入契约测试

2. **性能优化**
   - 优化CI/CD运行时间
   - 改进并行测试执行
   - 缓存依赖安装

3. **可靠性提升**
   - 增加重试机制
   - 完善回滚策略
   - 加强监控覆盖

## 📞 支持

### 联系方式
- **开发团队**: dev@rqa2025.com
- **运维团队**: ops@rqa2025.com
- **项目文档**: https://docs.rqa2025.com

### 紧急联系
- **生产故障**: +86 400-123-4567
- **安全事件**: security@rqa2025.com

---

**最后更新**: 2025年12月6日
**版本**: v1.0.0
