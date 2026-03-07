# Scripts目录快速索引

## 📋 脚本清单概览

本索引提供scripts目录的快速查找功能，按功能模块和优先级分类，帮助您快速定位所需脚本。

## 🚀 核心脚本 (优先使用)

### 测试运行核心
| 脚本 | 路径 | 功能 | 优先级 |
|------|------|------|--------|
| `run_tests.py` | `testing/` | 主测试运行器 | ⭐⭐⭐⭐⭐ |
| `run_focused_tests.py` | `testing/` | 聚焦测试运行器 | ⭐⭐⭐⭐ |
| `run_e2e_tests.py` | `testing/` | 端到端测试运行器 | ⭐⭐⭐⭐ |
| `run_infrastructure_tests.py` | `testing/` | 基础设施测试运行器 | ⭐⭐⭐⭐ |
| `verify_core_modules.py` | `testing/` | 核心模块验证器 | ⭐⭐⭐⭐ |
| `enhance_test_coverage_plan.py` | `testing/` | 测试覆盖率提升计划，自动分析当前覆盖率、识别优先模块、生成测试用例和报告 | ⭐⭐⭐⭐ |

### 配置管理核心
| 脚本 | 路径 | 功能 | 优先级 |
|------|------|------|--------|
| `config_validation_test.py` | `testing/` | 配置验证测试 | ⭐⭐⭐⭐⭐ |
| `config_sync_test.py` | `testing/` | 配置同步测试 | ⭐⭐⭐⭐ |
| `config_web_test.py` | `testing/` | 配置Web管理测试 | ⭐⭐⭐⭐ |
| `config_performance_test.py` | `testing/` | 配置性能测试 | ⭐⭐⭐⭐ |

### 环境管理核心
| 脚本 | 路径 | 功能 | 优先级 |
|------|------|------|--------|
| `health_check.py` | `deployment/environment/` | 健康检查 | ⭐⭐⭐⭐⭐ |
| `quick_start.bat` | `deployment/environment/` | 快速启动脚本 | ⭐⭐⭐⭐⭐ |
| `run_conda_tests.bat` | `deployment/environment/` | Conda测试运行 | ⭐⭐⭐⭐ |

### 部署核心
| 脚本 | 路径 | 功能 | 优先级 |
|------|------|------|--------|
| `auto_deployment.py` | `deployment/` | 自动部署工具 | ⭐⭐⭐⭐⭐ |
| `production_deploy.py` | `deployment/` | 生产环境部署 | ⭐⭐⭐⭐⭐ |
| `deployment_preparation.py` | `deployment/` | 部署准备 | ⭐⭐⭐⭐ |

## 🔧 开发工具 (常用)

### 代码优化
| 脚本 | 路径 | 功能 | 使用频率 |
|------|------|------|----------|
| `optimize_imports.py` | `development/` | 优化导入 | 高频 |
| `smart_fix_engine.py` | `development/` | 智能修复引擎 | 高频 |
| `fix_filename_issues.py` | `development/` | 修复文件名问题 | 中频 |

### 测试修复
| 脚本 | 路径 | 功能 | 使用频率 |
|------|------|------|----------|
| `auto_fix_tests.py` | `testing/fixes/` | 自动修复测试 | 高频 |
| `update_test_imports.py` | `testing/fixes/` | 更新测试导入 | 高频 |
| `fix_infrastructure_tests.py` | `testing/fixes/` | 修复基础设施测试 | 中频 |

### 覆盖率分析
| 脚本 | 路径 | 功能 | 使用频率 |
|------|------|------|----------|
| `test_coverage_analyzer.py` | `testing/optimization/` | 测试覆盖率分析器 | 高频 |
| `boost_infrastructure_coverage.py` | `testing/optimization/` | 提升基础设施覆盖率 | 中频 |
| `analyze_infrastructure_coverage.py` | `testing/optimization/` | 分析基础设施覆盖率 | 中频 |

## 📊 监控和报告 (定期使用)

### 进度监控
| 脚本 | 路径 | 功能 | 使用频率 |
|------|------|------|----------|
| `progress_monitor.py` | `monitoring/` | 进度监控器 | 定期 |
| `progress_tracker.py` | `monitoring/` | 进度跟踪器 | 定期 |

### 报告生成
| 脚本 | 路径 | 功能 | 使用频率 |
|------|------|------|----------|
| `generate_test_reports.py` | `testing/tools/` | 生成测试报告 | 定期 |
| `optimize_reports_filenames.py` | `reports/` | 优化报告文件名 | 低频 |
| `verify_core_modules.py` | `testing/` | 核心模块验证器 | 定期 |

## 🧠 模型相关 (按需使用)

### 模型部署
| 脚本 | 路径 | 功能 | 使用场景 |
|------|------|------|----------|
| `model_deployment_controller.py` | `models/` | 模型部署控制器 | 模型部署时 |
| `auto_model_landing.py` | `models/` | 自动模型落地 | 模型部署时 |
| `auto_model_landing_conda.py` | `models/` | Conda环境模型落地 | 模型部署时 |

### 模型演示
| 脚本 | 路径 | 功能 | 使用场景 |
|------|------|------|----------|
| `pretrained_models_demo.py` | `models/demos/` | 预训练模型演示 | 演示时 |
| `optimized_pretrained_models_demo.py` | `models/demos/` | 优化预训练模型演示 | 演示时 |

## 🎯 工作流脚本 (按需使用)

### 主流程
| 脚本 | 路径 | 功能 | 使用场景 |
|------|------|------|----------|
| `minimal_e2e_main_flow.py` | `workflows/` | 最小化端到端主流程 | 端到端测试 |
| `minimal_infra_main_flow.py` | `workflows/` | 最小化基础设施主流程 | 基础设施测试 |
| `minimal_model_main_flow.py` | `workflows/` | 最小化模型主流程 | 模型测试 |

## 🧪 压力测试 (性能测试时)

| 脚本 | 路径 | 功能 | 使用场景 |
|------|------|------|----------|
| `run_stress_test.py` | `stress_testing/` | 运行压力测试 | 性能测试 |
| `run_optimized_stress_test.py` | `stress_testing/` | 优化压力测试 | 性能测试 |
| `run_stable_infrastructure_tests.py` | `stress_testing/` | 稳定基础设施测试 | 稳定性测试 |

## 🔧 基础设施工具 (维护时)

### 系统优化
| 脚本 | 路径 | 功能 | 使用场景 |
|------|------|------|----------|
| `technical_debt_manager.py` | `infrastructure/optimization/` | 技术债务管理器 | 技术债务管理 |
| `optimize_system.py` | `infrastructure/optimization/` | 系统优化 | 系统优化 |
| `ops_optimizer.py` | `infrastructure/optimization/` | 运维优化器 | 运维优化 |

### 验证工具
| 脚本 | 路径 | 功能 | 使用场景 |
|------|------|------|----------|
| `verify_fpga_modules.py` | `infrastructure/validation/` | 验证FPGA模块 | FPGA验证 |
| `update_fpga_test_imports.py` | `infrastructure/validation/` | 更新FPGA测试导入 | FPGA测试 |

## 💼 交易和回测 (交易相关)

### 交易系统
| 脚本 | 路径 | 功能 | 使用场景 |
|------|------|------|----------|
| `minimal_trading_main_flow.py` | `trading/` | 最小化交易主流程 | 交易测试 |
| `minimal_risk_main_flow.py` | `trading/risk/` | 最小化风控主流程 | 风控测试 |

### 回测工具
| 脚本 | 路径 | 功能 | 使用场景 |
|------|------|------|----------|
| `backtest_optimizer.py` | `backtest/` | 回测优化器 | 回测优化 |
| `portfolio_optimizer.py` | `backtest/` | 投资组合优化器 | 投资组合优化 |

## 🔧 API和集成 (集成时)

### API工具
| 脚本 | 路径 | 功能 | 使用场景 |
|------|------|------|----------|
| `api_sdk_demo.py` | `api/` | API SDK演示 | API演示 |
| `simple_api_server.py` | `api/` | 简单API服务器 | API测试 |
| `optimized_api_server.py` | `api/` | 优化API服务器 | API测试 |

### 集成测试
| 脚本 | 路径 | 功能 | 使用场景 |
|------|------|------|----------|
| `integration_test.py` | `integration/` | 集成测试 | 集成测试 |
| `run_complete_e2e_test.py` | `integration/` | 完整端到端测试 | 端到端测试 |

## 🏁 项目收尾 (项目结束时)

| 脚本 | 路径 | 功能 | 使用场景 |
|------|------|------|----------|
| `project_finalizer.py` | `project/` | 项目完成器 | 项目完成 |
| `project_closure.py` | `project/` | 项目关闭器 | 项目关闭 |

## 📋 快速查找表

### 按功能快速查找

#### 🚀 日常开发
- **运行测试**: `testing/run_tests.py`
- **覆盖率提升计划**: `testing/enhance_test_coverage_plan.py`
- **环境检查**: `deployment/environment/health_check.py`
- **代码优化**: `development/optimize_imports.py`
- **测试修复**: `testing/fixes/auto_fix_tests.py`

#### 📊 监控报告
- **进度监控**: `monitoring/progress_monitor.py`
- **覆盖率分析**: `testing/optimization/test_coverage_analyzer.py`
- **生成报告**: `testing/tools/generate_test_reports.py`

#### 🚀 部署运维
- **快速启动**: `deployment/environment/quick_start.bat`
- **自动部署**: `deployment/auto_deployment.py`
- **生产部署**: `deployment/production_deploy.py`

#### 🧠 模型相关
- **模型部署**: `models/model_deployment_controller.py`
- **模型演示**: `models/demos/pretrained_models_demo.py`

#### 🧪 性能测试
- **压力测试**: `stress_testing/run_stress_test.py`
- **优化测试**: `stress_testing/run_optimized_stress_test.py`

### 按使用频率分类

#### ⭐⭐⭐⭐⭐ 核心脚本 (每日使用)
1. `testing/run_tests.py` - 主测试运行器
2. `deployment/environment/health_check.py` - 健康检查
3. `deployment/environment/quick_start.bat` - 快速启动
4. `deployment/auto_deployment.py` - 自动部署
5. `development/optimize_imports.py` - 优化导入

#### ⭐⭐⭐⭐ 重要脚本 (每周使用)
1. `testing/run_focused_tests.py` - 聚焦测试
2. `testing/run_e2e_tests.py` - 端到端测试
3. `testing/fixes/auto_fix_tests.py` - 自动修复测试
4. `testing/optimization/test_coverage_analyzer.py` - 覆盖率分析
5. `testing/enhance_test_coverage_plan.py` - 测试覆盖率提升计划
6. `monitoring/progress_monitor.py` - 进度监控

#### ⭐⭐⭐ 常用脚本 (每月使用)
1. `development/smart_fix_engine.py` - 智能修复引擎
2. `testing/optimization/boost_infrastructure_coverage.py` - 提升覆盖率
3. `deployment/production_deploy.py` - 生产部署
4. `models/model_deployment_controller.py` - 模型部署
5. `stress_testing/run_stress_test.py` - 压力测试

#### ⭐⭐ 专用脚本 (按需使用)
1. `models/demos/` - 模型演示脚本
2. `trading/` - 交易相关脚本
3. `backtest/` - 回测工具
4. `api/` - API工具
5. `infrastructure/` - 基础设施工具

#### ⭐ 特殊脚本 (项目特定)
1. `project/` - 项目收尾脚本
2. `workflows/demos/` - 工作流演示
3. `integration/` - 集成测试

## 🚫 避免脚本膨胀的建议

### 1. 脚本合并原则
- **相似功能合并**: 将功能相似的脚本合并为一个
- **参数化设计**: 使用命令行参数控制不同功能
- **模块化设计**: 将复杂脚本拆分为多个模块

### 2. 脚本清理策略
- **定期清理**: 每月清理未使用的脚本
- **版本控制**: 保留重要版本，删除过时版本
- **文档同步**: 及时更新脚本文档

### 3. 脚本标准化
- **命名规范**: 使用统一的命名规范
- **目录结构**: 保持清晰的目录结构
- **文档完整**: 每个脚本都有完整文档

### 4. 优先级管理
- **核心脚本**: 优先维护核心脚本
- **专用脚本**: 按需创建专用脚本
- **临时脚本**: 及时清理临时脚本

## 📝 使用建议

### 日常开发流程
1. **环境检查**: `deployment/environment/health_check.py`
2. **运行测试**: `testing/run_tests.py`
3. **代码优化**: `development/optimize_imports.py`
4. **测试修复**: `testing/fixes/auto_fix_tests.py`
5. **进度监控**: `monitoring/progress_monitor.py`

### 部署流程
1. **部署准备**: `deployment/deployment_preparation.py`
2. **自动部署**: `deployment/auto_deployment.py`
3. **生产部署**: `deployment/production_deploy.py`

### 性能测试流程
1. **压力测试**: `stress_testing/run_stress_test.py`
2. **优化测试**: `stress_testing/run_optimized_stress_test.py`
3. **稳定性测试**: `stress_testing/run_stable_infrastructure_tests.py`

## 🧹 脚本清理和维护

### 脚本清理工具
| 脚本 | 路径 | 功能 | 使用频率 |
|------|------|------|----------|
| `cleanup_unused_scripts.py` | `scripts/` | 脚本清理工具 | 每月 |
| `SCRIPT_USAGE_ANALYSIS.md` | `scripts/` | 脚本使用分析 | 定期 |

### 清理流程
```bash
# 1. 预览清理结果
python scripts/cleanup_unused_scripts.py --dry-run

# 2. 执行清理
python scripts/cleanup_unused_scripts.py --execute

# 3. 检查清理报告
cat scripts_cleanup_report.md
```

### 维护建议
1. **每月清理**: 定期运行清理工具
2. **功能检查**: 清理后运行测试确保功能正常
3. **文档更新**: 及时更新脚本索引和使用指南
4. **备份恢复**: 保留重要脚本的备份

### 生产环境影响评估
- ✅ **不会直接影响生产环境发布**
- ✅ **部署脚本与开发脚本完全分离**
- ✅ **生产环境使用独立的部署配置**
- ⚠️ **建议定期清理以降低维护负担**

---

**索引版本**: v1.1  
**最后更新**: 2025-08-06  
**维护状态**: ✅ 活跃维护中  
**清理状态**: ✅ 已建立清理机制 