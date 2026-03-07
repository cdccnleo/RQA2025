# RQA2025 文档索引

## 文档概览

本文档索引列出了 RQA2025 项目的完整文档集合，按类别和重要性排序。

## 核心文档

### 用户文档
- [**README.md**](README.md) - 项目介绍、安装和快速开始指南
- [**USER_GUIDE.md**](USER_GUIDE.md) - 用户使用指南，包含详细的操作说明
- [**API.md**](API.md) - REST API 文档，包含所有接口说明

### 开发者文档
- [**DEVELOPER_GUIDE.md**](DEVELOPER_GUIDE.md) - 开发者指南，包含架构和扩展说明
- [**TESTING.md**](TESTING.md) - 测试文档，包含测试策略和执行指南

### 架构设计文档
- [**BUSINESS_PROCESS_DRIVEN_ARCHITECTURE.md**](architecture/BUSINESS_PROCESS_DRIVEN_ARCHITECTURE.md) - 业务流程驱动架构设计
- [**strategy_layer_architecture_design.md**](architecture/strategy_layer_architecture_design.md) - 策略服务层架构设计
- [**ARCHITECTURE.md**](ARCHITECTURE.md) - 系统架构设计文档

### 部署文档
- [**DEPLOYMENT_GUIDE.md**](DEPLOYMENT_GUIDE.md) - 部署指南，包含多种部署方式
- [**DOCKER_DEPLOYMENT.md**](DOCKER_DEPLOYMENT.md) - Docker 部署指南
- [**KUBERNETES_DEPLOYMENT.md**](KUBERNETES_DEPLOYMENT.md) - Kubernetes 部署指南

## Phase 文档

### Phase 3: 核心服务迁移
- [**PHASE3_CORE_SERVICES_MIGRATION_COMPLETION.md**](PHASE3_CORE_SERVICES_MIGRATION_COMPLETION.md) - Phase 3 完成报告

### Phase 4: 工作空间集成
- [**PHASE4_WORKSPACE_INTEGRATION_COMPLETION.md**](PHASE4_WORKSPACE_INTEGRATION_COMPLETION.md) - Phase 4 完成报告

### Phase 5: 测试和验证
- [**PHASE5_TESTING_VALIDATION_COMPLETION.md**](PHASE5_TESTING_VALIDATION_COMPLETION.md) - Phase 5 完成报告

## 目录结构

```
docs/
├── README.md                              # 项目介绍
├── USER_GUIDE.md                          # 用户指南
├── DEVELOPER_GUIDE.md                     # 开发者指南
├── API.md                                 # API 文档
├── ARCHITECTURE.md                        # 系统架构设计
├── TESTING.md                             # 测试文档
├── DEPLOYMENT_GUIDE.md                    # 部署指南
├── architecture/                          # 架构设计文档
│   ├── BUSINESS_PROCESS_DRIVEN_ARCHITECTURE.md      # 业务流程驱动架构
│   └── strategy_layer_architecture_design.md       # 策略服务层架构设计
├── strategy/                              # 策略相关文档
│   ├── STRATEGY_SERVICE_LAYER_ARCHITECTURE.md      # 策略服务架构（旧版）
│   ├── PHASE3_CORE_SERVICES_MIGRATION_COMPLETION.md
│   ├── PHASE4_WORKSPACE_INTEGRATION_COMPLETION.md
│   └── PHASE5_TESTING_VALIDATION_COMPLETION.md
├── api/                                   # API 相关文档
├── deployment/                            # 部署相关文档
└── DEVELOPMENT_INDEX.md                   # 文档索引（本文件）
```

## 文档更新日志

### v1.0.0 (2024-01-27)
- ✅ 完成所有核心文档编写
- ✅ 建立完整的文档体系
- ✅ 创建文档索引和导航
- ✅ 集成自动化文档生成

### 近期更新
- **2025-08-31**: 重构文档结构，建立统一的架构设计目录
- **2025-08-31**: 移动策略服务层架构设计文档到 architecture/ 目录
- **2024-01-27**: 完成 Phase 5 测试和验证文档
- **2024-01-27**: 完成 Phase 4 工作空间集成文档
- **2024-01-27**: 完成 Phase 3 核心服务迁移文档
- **2024-01-27**: 建立文档自动化生成系统

## 文档维护

### 更新频率
- **核心文档**: 每次主版本发布时更新
- **用户指南**: 功能变更时及时更新
- **API文档**: API变更时立即更新
- **部署文档**: 部署方式变更时更新

### 贡献指南
1. 遵循现有的文档结构和格式
2. 使用 Markdown 格式编写
3. 包含必要的代码示例
4. 更新本文档索引

### 文档生成
使用自动化脚本生成和更新文档：

```bash
# 生成所有文档
python scripts/generate_docs.py

# 生成特定类型文档
python scripts/generate_docs.py --type api
python scripts/generate_docs.py --type user-guide
```

## 相关链接

- **项目主页**: https://github.com/your-org/rqa2025
- **问题跟踪**: https://github.com/your-org/rqa2025/issues
- **讨论论坛**: https://community.rqa2025.com/
- **在线文档**: https://docs.rqa2025.com/

## 技术支持

### 联系方式
- **技术支持**: support@rqa2025.com
- **商务合作**: business@rqa2025.com
- **媒体联系**: press@rqa2025.com

### 社区资源
- **GitHub**: https://github.com/your-org/rqa2025
- **Stack Overflow**: 使用标签 `rqa2025`
- **Reddit**: r/rqa2025
- **Discord**: https://discord.gg/rqa2025

---

**文档版本**: v1.1.0
**最后更新**: 2025-08-31 17:15:00
**维护者**: RQA2025 Team
