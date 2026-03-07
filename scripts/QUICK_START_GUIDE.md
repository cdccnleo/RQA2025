# Scripts快速使用指南

## 🚀 快速开始

### 第一步：环境检查
```bash
# 检查环境状态
python scripts/deployment/environment/health_check.py

# 快速启动
scripts/deployment/environment/quick_start.bat
```

### 第二步：运行测试
```bash
# 运行所有测试
python scripts/testing/run_tests.py

# 运行聚焦测试
python scripts/testing/run_focused_tests.py

# 运行端到端测试
python scripts/testing/run_e2e_tests.py
```

### 第三步：代码优化
```bash
# 优化导入
python scripts/development/optimize_imports.py

# 修复测试
python scripts/testing/fixes/auto_fix_tests.py
```

## 📋 常用脚本速查

### 🧪 测试相关
| 命令 | 功能 | 使用场景 |
|------|------|----------|
| `python scripts/testing/run_tests.py` | 运行所有测试 | 日常测试 |
| `python scripts/testing/run_focused_tests.py` | 运行聚焦测试 | 快速验证 |
| `python scripts/testing/run_e2e_tests.py` | 运行端到端测试 | 完整测试 |
| `python scripts/testing/run_infrastructure_tests.py` | 运行基础设施测试 | 基础设施验证 |
| `python scripts/testing/run_data_tests.py` | 运行数据层测试 | 数据验证 |
| `python scripts/testing/run_chaos_test.py` | 运行混沌测试 | 稳定性测试 |

### 🔧 开发工具
| 命令 | 功能 | 使用场景 |
|------|------|----------|
| `python scripts/development/optimize_imports.py` | 优化导入 | 代码清理 |
| `python scripts/development/smart_fix_engine.py` | 智能修复引擎 | 自动修复 |
| `python scripts/development/fix_filename_issues.py` | 修复文件名问题 | 文件整理 |
| `python scripts/testing/fixes/auto_fix_tests.py` | 自动修复测试 | 测试修复 |
| `python scripts/testing/fixes/update_test_imports.py` | 更新测试导入 | 导入更新 |

### 📊 监控和报告
| 命令 | 功能 | 使用场景 |
|------|------|----------|
| `python scripts/monitoring/progress_monitor.py` | 进度监控 | 项目监控 |
| `python scripts/monitoring/progress_tracker.py` | 进度跟踪 | 进度跟踪 |
| `python scripts/testing/tools/generate_test_reports.py` | 生成测试报告 | 报告生成 |
| `python scripts/testing/optimization/test_coverage_analyzer.py` | 覆盖率分析 | 覆盖率检查 |

### 🚀 部署相关
| 命令 | 功能 | 使用场景 |
|------|------|----------|
| `python scripts/deployment/auto_deployment.py` | 自动部署 | 自动部署 |
| `python scripts/deployment/production_deploy.py` | 生产部署 | 生产环境 |
| `python scripts/deployment/deployment_preparation.py` | 部署准备 | 部署前准备 |
| `python scripts/deployment/environment/health_check.py` | 健康检查 | 环境检查 |

### 🧠 模型相关
| 命令 | 功能 | 使用场景 |
|------|------|----------|
| `python scripts/models/model_deployment_controller.py` | 模型部署控制器 | 模型部署 |
| `python scripts/models/auto_model_landing.py` | 自动模型落地 | 模型落地 |
| `python scripts/models/auto_model_landing_conda.py` | Conda环境模型落地 | Conda环境 |
| `python scripts/models/demos/pretrained_models_demo.py` | 预训练模型演示 | 模型演示 |

### 🧪 压力测试
| 命令 | 功能 | 使用场景 |
|------|------|----------|
| `python scripts/stress_testing/run_stress_test.py` | 运行压力测试 | 性能测试 |
| `python scripts/stress_testing/run_optimized_stress_test.py` | 优化压力测试 | 优化测试 |
| `python scripts/stress_testing/run_simple_stress_test.py` | 简单压力测试 | 快速测试 |
| `python scripts/stress_testing/run_stable_infrastructure_tests.py` | 稳定基础设施测试 | 稳定性测试 |

## 🎯 按场景使用

### 日常开发流程
```bash
# 1. 环境检查
python scripts/deployment/environment/health_check.py

# 2. 运行测试
python scripts/testing/run_tests.py

# 3. 代码优化
python scripts/development/optimize_imports.py

# 4. 测试修复
python scripts/testing/fixes/auto_fix_tests.py

# 5. 进度监控
python scripts/monitoring/progress_monitor.py
```

### 部署流程
```bash
# 1. 部署准备
python scripts/deployment/deployment_preparation.py

# 2. 自动部署
python scripts/deployment/auto_deployment.py

# 3. 生产部署
python scripts/deployment/production_deploy.py
```

### 性能测试流程
```bash
# 1. 压力测试
python scripts/stress_testing/run_stress_test.py

# 2. 优化测试
python scripts/stress_testing/run_optimized_stress_test.py

# 3. 稳定性测试
python scripts/stress_testing/run_stable_infrastructure_tests.py
```

### 模型部署流程
```bash
# 1. 模型部署
python scripts/models/model_deployment_controller.py

# 2. 自动落地
python scripts/models/auto_model_landing.py

# 3. 演示验证
python scripts/models/demos/pretrained_models_demo.py
```

## 📊 脚本优先级

### ⭐⭐⭐⭐⭐ 核心脚本 (每日必用)
1. `testing/run_tests.py` - 主测试运行器
2. `deployment/environment/health_check.py` - 健康检查
3. `deployment/environment/quick_start.bat` - 快速启动
4. `development/optimize_imports.py` - 优化导入

### ⭐⭐⭐⭐ 重要脚本 (每周使用)
1. `testing/run_focused_tests.py` - 聚焦测试
2. `testing/fixes/auto_fix_tests.py` - 自动修复测试
3. `testing/optimization/test_coverage_analyzer.py` - 覆盖率分析
4. `monitoring/progress_monitor.py` - 进度监控

### ⭐⭐⭐ 常用脚本 (每月使用)
1. `development/smart_fix_engine.py` - 智能修复引擎
2. `deployment/auto_deployment.py` - 自动部署
3. `models/model_deployment_controller.py` - 模型部署
4. `stress_testing/run_stress_test.py` - 压力测试

## 🚫 避免脚本膨胀

### 脚本选择原则
1. **优先使用核心脚本**: 优先选择核心脚本，避免创建重复功能
2. **参数化使用**: 使用命令行参数控制脚本的不同功能
3. **模块化设计**: 将复杂功能拆分为多个模块
4. **定期清理**: 每月清理未使用的脚本

### 脚本创建规范
1. **功能明确**: 每个脚本功能单一明确
2. **命名规范**: 使用描述性文件名
3. **文档完整**: 每个脚本都有完整文档
4. **版本控制**: 重要脚本需要版本记录

### 脚本维护策略
1. **核心脚本**: 优先维护核心脚本
2. **专用脚本**: 按需创建专用脚本
3. **临时脚本**: 及时清理临时脚本
4. **文档同步**: 及时更新脚本文档

## 📝 使用技巧

### 快速查找脚本
```bash
# 查找测试相关脚本
find scripts/testing -name "*.py"

# 查找部署相关脚本
find scripts/deployment -name "*.py"

# 查找开发工具脚本
find scripts/development -name "*.py"
```

### 脚本参数使用
```bash
# 带参数的脚本使用
python scripts/testing/run_tests.py --focus=infrastructure
python scripts/development/optimize_imports.py --dry-run
python scripts/monitoring/progress_monitor.py --verbose
```

### 脚本组合使用
```bash
# 组合多个脚本
python scripts/deployment/environment/health_check.py && \
python scripts/testing/run_tests.py && \
python scripts/monitoring/progress_monitor.py
```

## 🔄 定期维护

### 每周维护
- 检查核心脚本功能
- 更新脚本文档
- 清理临时脚本

### 每月维护
- 分析脚本使用情况
- 合并相似功能脚本
- 优化脚本性能

### 每季度维护
- 全面清理未使用脚本
- 更新脚本索引
- 优化目录结构

---

**指南版本**: v1.0  
**最后更新**: 2025-07-19  
**维护状态**: ✅ 活跃维护中 