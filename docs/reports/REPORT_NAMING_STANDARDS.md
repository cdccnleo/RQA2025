# RQA2025 报告命名规范

## 📋 规范概览

本文档定义了RQA2025项目的报告命名规范，确保报告的一致性和可维护性。

## 🏗️ 命名规范

### 1. 基本命名格式

```
{category}_{type}_{subject}.{extension}
```

#### 命名组件说明

| 组件 | 说明 | 示例 |
|------|------|------|
| category | 报告类别 | project, technical, business, operational, research |
| type | 报告类型 | progress, completion, analysis, test, audit |
| subject | 报告主题 | deployment, performance, security |
| extension | 文件扩展名 | .md, .json, .html |

**注意**: 由于报告具有时效性，基本格式不包含日期和版本信息，始终保持报告为最新版本。版本控制通过内容更新实现，而非文件名。

### 2. 类别定义 (category)

#### 项目报告 (project)
- **progress**: 进度报告、里程碑报告、状态更新
- **completion**: 完成报告、最终报告、总结报告
- **architecture**: 架构报告、设计报告、结构分析
- **deployment**: 部署报告、上线报告、环境配置

#### 技术报告 (technical)
- **testing**: 测试报告、测试分析、测试结果
- **performance**: 性能报告、性能分析、性能优化
- **security**: 安全报告、安全审计、风险评估
- **quality**: 质量报告、代码质量、技术债务
- **optimization**: 优化报告、改进报告、增强报告

#### 业务报告 (business)
- **analytics**: 分析报告、数据分析、趋势分析
- **trading**: 交易报告、交易分析、策略报告
- **backtest**: 回测报告、回测分析、回测结果
- **compliance**: 合规报告、监管报告、合规审计

#### 运维报告 (operational)
- **monitoring**: 监控报告、监控分析、告警报告
- **deployment**: 部署报告、部署分析、环境报告
- **notification**: 通知报告、通知分析、沟通报告
- **maintenance**: 维护报告、维护分析、支持报告

#### 研究报告 (research)
- **ml_integration**: 机器学习集成报告
- **deep_learning**: 深度学习报告
- **reinforcement_learning**: 强化学习报告
- **continuous_optimization**: 持续优化报告

### 3. 类型定义 (type)

#### 进度类型
- **progress**: 进度报告、状态更新
- **milestone**: 里程碑报告、关键节点
- **status**: 状态报告、当前状态

#### 完成类型
- **completion**: 完成报告、最终报告
- **final**: 最终版本、最终总结
- **summary**: 总结报告、汇总报告

#### 分析类型
- **analysis**: 分析报告、详细分析
- **audit**: 审计报告、审查报告
- **review**: 审查报告、评估报告

#### 测试类型
- **test**: 测试报告、测试结果
- **coverage**: 覆盖率报告、测试覆盖
- **integration**: 集成测试报告

### 4. 主题定义 (subject)

#### 技术主题
- **deployment**: 部署相关
- **performance**: 性能相关
- **security**: 安全相关
- **quality**: 质量相关
- **optimization**: 优化相关

#### 业务主题
- **trading**: 交易相关
- **backtest**: 回测相关
- **analytics**: 分析相关
- **compliance**: 合规相关

#### 运维主题
- **monitoring**: 监控相关
- **notification**: 通知相关
- **maintenance**: 维护相关

## 📝 命名示例

### 项目报告示例
```
project_progress_deployment.md
project_completion_final.md
project_architecture_design.md
project_deployment_environment.md
```

### 技术报告示例
```
technical_test_performance.json
technical_analysis_security.md
technical_audit_quality.md
technical_optimization_improvement.md
```

### 业务报告示例
```
business_analytics_trend.md
business_trading_strategy.md
business_backtest_result.md
business_compliance_audit.md
```

### 运维报告示例
```
operational_monitoring_system.md
operational_deployment_blue_green.json
operational_notification_alert.md
operational_maintenance_support.md
```

### 研究报告示例
```
research_ml_integration_model.md
research_deep_learning_neural.md
research_reinforcement_learning_rl.md
research_continuous_optimization_auto.md
```

## 🔧 特殊命名规则

### 1. 版本控制（内容级别）
- 版本控制通过报告内容内的版本信息实现
- 在报告内容中标注版本号和更新时间
- 不依赖文件名进行版本管理

### 2. 时间标识（内容级别）
- 在报告内容中记录生成时间和更新时间
- 使用报告元数据记录时间信息
- 文件名不包含时间标识

### 3. 状态标识
- **pending**: 待处理
- **in_progress**: 进行中
- **completed**: 已完成
- **failed**: 失败
- **cancelled**: 已取消

### 4. 优先级标识
- **high**: 高优先级
- **medium**: 中优先级
- **low**: 低优先级
- **urgent**: 紧急

## 📊 文件组织规范

### 1. 目录结构
```
reports/
├── project/
│   ├── progress/
│   ├── completion/
│   ├── architecture/
│   └── deployment/
├── technical/
│   ├── testing/
│   ├── performance/
│   ├── security/
│   ├── quality/
│   └── optimization/
├── business/
│   ├── analytics/
│   ├── trading/
│   ├── backtest/
│   └── compliance/
├── operational/
│   ├── monitoring/
│   ├── deployment/
│   ├── notification/
│   └── maintenance/
└── research/
    ├── ml_integration/
    ├── deep_learning/
    ├── reinforcement_learning/
    └── continuous_optimization/
```

### 2. 文件排序
- 按修改时间排序（最新的在前）
- 按文件名字母顺序排序
- 按状态排序（completed, in_progress, pending）

### 3. 归档规则
- 年度归档：每年底将旧报告移至archive/YYYY/
- 内容归档：基于报告内容状态进行归档
- 废弃归档：不再使用的报告移至archive/deprecated/

## 🚀 最佳实践

### 1. 命名原则
- **清晰性**: 文件名应清楚表达报告内容
- **一致性**: 遵循统一的命名格式
- **可读性**: 使用有意义的词汇和缩写
- **可搜索性**: 包含关键信息便于搜索

### 2. 内容要求
- **完整性**: 包含必要的元数据
- **准确性**: 数据准确，分析合理
- **及时性**: 重要事件24小时内生成报告
- **规范性**: 遵循模板和格式要求

### 3. 维护要求
- **定期更新**: 进度报告定期更新
- **内容版本控制**: 在报告内容中维护版本信息
- **索引维护**: 保持索引文件同步
- **清理归档**: 定期清理过期报告

## 📋 检查清单

### 创建新报告时
- [ ] 使用正确的命名格式
- [ ] 包含必要的元数据
- [ ] 遵循模板规范
- [ ] 更新相关索引
- [ ] 设置正确的权限

### 维护现有报告时
- [ ] 检查命名一致性
- [ ] 更新过时信息
- [ ] 维护内容版本信息
- [ ] 清理重复文件
- [ ] 更新索引链接

### 归档报告时
- [ ] 确认报告状态
- [ ] 选择合适的归档位置
- [ ] 更新索引引用
- [ ] 保留必要的访问权限
- [ ] 记录归档原因

## 🔗 相关文档

- [报告组织规范](README.md)
- [报告模板库](templates/README.md)
- [报告索引](INDEX.md)
- [文档维护指南](../DOCUMENT_MAINTENANCE_GUIDE.md)

---

**最后更新**: 2025-01-27  
**维护者**: 项目团队  
**状态**: ✅ 活跃维护 