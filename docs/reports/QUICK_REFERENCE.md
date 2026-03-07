# RQA2025 报告快速参考指南

## 📋 快速导航

### 🏗️ 项目报告
- **进度报告**: `reports/project/progress/` - 项目里程碑、任务完成情况
- **完成报告**: `reports/project/completion/` - 项目或功能模块完成总结
- **架构报告**: `reports/project/architecture/` - 系统架构设计和变更报告
- **部署报告**: `reports/project/deployment/` - 系统部署和上线报告

### 🔧 技术报告
- **测试报告**: `reports/technical/testing/` - 单元测试、集成测试、性能测试结果
- **性能报告**: `reports/technical/performance/` - 系统性能分析和优化报告
- **安全报告**: `reports/technical/security/` - 安全审计、漏洞扫描、风险评估
- **质量报告**: `reports/technical/quality/` - 代码质量、技术债务分析
- **优化报告**: `reports/technical/optimization/` - 性能优化、架构优化总结

### 💼 业务报告
- **分析报告**: `reports/business/analytics/` - 业务数据分析、趋势分析
- **交易报告**: `reports/business/trading/` - 交易策略、交易执行分析
- **回测报告**: `reports/business/backtest/` - 策略回测结果、回测一致性验证
- **合规报告**: `reports/business/compliance/` - 监管合规、风险控制报告

### 🚀 运维报告
- **监控报告**: `reports/operational/monitoring/` - 系统监控、告警分析
- **部署报告**: `reports/operational/deployment/` - 环境部署、配置变更
- **通知报告**: `reports/operational/notification/` - 系统通知、团队沟通
- **维护报告**: `reports/operational/maintenance/` - 系统维护、故障处理

### 🔬 研究报告
- **机器学习集成**: `reports/research/ml_integration/` - ML模型集成、AI功能开发
- **深度学习**: `reports/research/deep_learning/` - 神经网络、深度学习应用
- **强化学习**: `reports/research/reinforcement_learning/` - 强化学习算法、策略优化
- **持续优化**: `reports/research/continuous_optimization/` - 自动化优化、智能调优

## 📝 命名规范

### 基本格式
```
{category}_{type}_{subject}.{extension}
```

### 命名示例
```
project_progress_deployment.md
technical_test_performance.json
business_analysis_trading.md
operational_monitoring_system.md
research_ml_integration.md
```

**注意**: 文件名不包含日期和版本信息，版本控制通过报告内容实现。

## 🔍 快速查找

### 按功能查找
- **项目进度**: `reports/project/progress/`
- **测试结果**: `reports/technical/testing/`
- **性能分析**: `reports/technical/performance/`
- **安全审计**: `reports/technical/security/`
- **业务分析**: `reports/business/analytics/`
- **交易策略**: `reports/business/trading/`
- **系统监控**: `reports/operational/monitoring/`

### 按时间查找
- **最新报告**: 按文件修改时间排序
- **历史报告**: `reports/archive/`
- **废弃报告**: `reports/archive/deprecated/`

### 按类型查找
- **Markdown报告**: `*.md` 文件
- **JSON报告**: `*.json` 文件
- **HTML报告**: `*.html` 文件

## 📊 常用报告

### 项目进度
- `project_progress_deployment.md` - 部署进度报告
- `project_progress_milestone.md` - 里程碑报告
- `project_progress_status.md` - 状态更新报告

### 技术测试
- `technical_test_performance.json` - 性能测试报告
- `technical_test_integration.md` - 集成测试报告
- `technical_test_coverage.md` - 测试覆盖率报告

### 业务分析
- `business_analytics_trend.md` - 趋势分析报告
- `business_trading_strategy.md` - 交易策略报告
- `business_backtest_result.md` - 回测结果报告

### 运维监控
- `operational_monitoring_system.md` - 系统监控报告
- `operational_deployment_blue_green.json` - 蓝绿部署报告
- `operational_notification_alert.md` - 告警通知报告

## 🚀 最佳实践

### 创建新报告
1. 确定报告类别和类型
2. 使用标准命名格式
3. 选择合适的模板
4. 更新相关索引

### 查找报告
1. 使用INDEX.md快速定位
2. 按功能目录查找
3. 使用文件名搜索
4. 查看README文件

### 维护报告
1. 定期更新进度报告
2. 及时归档历史报告
3. 保持索引同步
4. 清理过期文件

## 📋 检查清单

### 创建报告时
- [ ] 使用正确的命名格式
- [ ] 包含必要的元数据
- [ ] 遵循模板规范
- [ ] 更新相关索引
- [ ] 设置正确的权限

### 查找报告时
- [ ] 确定报告类别
- [ ] 选择合适目录
- [ ] 使用搜索功能
- [ ] 查看README文件

### 维护报告时
- [ ] 检查命名一致性
- [ ] 更新过时信息
- [ ] 维护版本控制
- [ ] 清理重复文件

## 🔗 相关链接

- [完整报告索引](../reports/INDEX.md)
- [报告组织规范](../reports/README.md)
- [命名规范详情](REPORT_NAMING_STANDARDS.md)
- [重新组织总结](../reports/REORGANIZATION_SUMMARY.md)

---

**最后更新**: 2025-01-27  
**维护者**: 项目团队  
**状态**: ✅ 活跃维护 