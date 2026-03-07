# RQA2025 AI增强测试覆盖率自动化指南

## 📋 概述

本指南介绍如何使用AI增强的测试覆盖率自动化系统，该系统集成了Deepseek大模型，能够智能生成测试用例并持续提升项目覆盖率。

## 🏗️ 系统架构

### 核心组件

1. **AI连接器** (`DeepseekAIConnector`)
   - 连接本地Deepseek大模型服务
   - 智能生成测试代码
   - 本地缓存机制

2. **自动化执行器** (`AICoverageAutomation`)
   - 分析覆盖率差距
   - 生成AI测试用例
   - 运行和评估测试结果

3. **持续运行器** (`ContinuousAICoverageRunner`)
   - 定时执行自动化任务
   - 健康检查和错误处理
   - 执行历史记录

4. **启动脚本** (`start_ai_coverage_automation.py`)
   - 便捷的启动和管理功能
   - 环境检查和状态监控

## 🚀 快速开始

### 1. 环境准备

确保已激活conda rqa环境（推荐，项目默认）：

```bash
conda activate rqa
```

# 如需base环境
# conda activate base

### 2. 安装依赖

```bash
pip install aiohttp schedule
```

### 3. 启动Deepseek服务

确保本地Deepseek服务正在运行：

```bash
# 假设使用Ollama运行Deepseek
ollama run deepseek-coder
```

### 4. 检查环境

```bash
python scripts/testing/start_ai_coverage_automation.py check
```

### 5. 单次执行

```bash
python scripts/testing/start_ai_coverage_automation.py once
```

### 6. 启动持续自动化

```bash
python scripts/testing/start_ai_coverage_automation.py start --mode continuous
```

## 📊 配置说明

### 配置文件位置

`scripts/testing/ai_coverage_config.json`

### 主要配置项

```json
{
  "ai_config": {
    "api_base": "http://localhost:11434",  // AI服务地址
    "model": "deepseek-coder",             // 使用的模型
    "temperature": 0.3,                    // 生成温度
    "max_tokens": 4000,                    // 最大token数
    "timeout": 120                         // 超时时间
  },
  "coverage_targets": {
    "infrastructure": 90.0,                // 基础设施层目标
    "data": 85.0,                         // 数据层目标
    "features": 85.0,                     // 特征层目标
    "models": 85.0,                       // 模型层目标
    "trading": 85.0,                      // 交易层目标
    "backtest": 85.0                      // 回测层目标
  },
  "automation": {
    "schedule_time": "02:00",             // 定时执行时间
    "run_immediately": false,             // 是否立即执行
    "max_modules_per_layer": 5,           // 每层最大模块数
    "cache_enabled": true,                // 是否启用缓存
    "health_check_interval": 60           // 健康检查间隔
  }
}
```

## 🎯 使用场景

### 1. 开发阶段

在开发新功能时，使用AI自动化快速生成测试用例：

```bash
# 针对特定层级生成测试
python scripts/testing/ai_enhanced_coverage_automation.py \
  --layers infrastructure data \
  --target 90.0
```

### 2. 持续集成

在CI/CD流程中集成AI自动化：

```bash
# 在CI脚本中添加
python scripts/testing/start_ai_coverage_automation.py once
```

### 3. 覆盖率提升

当覆盖率不达标时，使用AI自动化提升：

```bash
# 分析并提升覆盖率
python scripts/testing/ai_enhanced_coverage_automation.py \
  --target 85.0 \
  --layers infrastructure data features trading
```

### 4. 持续监控

设置定时任务持续监控和提升覆盖率：

```bash
# 启动持续自动化
python scripts/testing/start_ai_coverage_automation.py start --mode continuous
```

## 📈 监控和报告

### 1. 查看状态

```bash
python scripts/testing/start_ai_coverage_automation.py status
```

### 2. 生成报告

```bash
python scripts/testing/start_ai_coverage_automation.py report
```

### 3. 查看日志

```bash
# 查看AI自动化日志
tail -f logs/ai_coverage_automation.log

# 查看持续运行日志
tail -f logs/continuous_ai_coverage.log
```

## 🔧 高级功能

### 1. 自定义AI提示

修改 `ai_enhanced_coverage_automation.py` 中的 `_build_test_generation_prompt` 方法来自定义AI提示词。

### 2. 优先级模块配置

在配置文件中设置模块优先级：

```json
{
  "priority_modules": {
    "critical": ["config_manager.py", "logger.py"],
    "high": ["data_loader.py", "trading_engine.py"],
    "medium": ["utils.py", "helpers.py"]
  }
}
```

### 3. 缓存管理

AI生成的测试代码会缓存在 `cache/ai_test_generation/` 目录中，可以手动清理：

```bash
rm -rf cache/ai_test_generation/*
```

### 4. 自定义调度

修改配置文件中的调度时间：

```json
{
  "automation": {
    "schedule_time": "03:00"  // 改为凌晨3点执行
  }
}
```

## 🐛 故障排除

### 1. AI服务连接失败

**症状**: 出现 "AI连接失败" 错误

**解决方案**:
- 检查Deepseek服务是否运行
- 验证API地址是否正确
- 检查网络连接

```bash
# 测试AI服务连接
curl http://localhost:11434/v1/models
```

### 2. 测试生成失败

**症状**: AI生成的测试无法运行

**解决方案**:
- 检查模块导入路径
- 验证依赖包是否安装
- 查看生成的测试代码语法

### 3. 覆盖率分析失败

**症状**: 无法获取覆盖率数据

**解决方案**:
- 确保pytest和pytest-cov已安装
- 检查测试目录结构
- 验证源代码路径

### 4. 持续运行停止

**症状**: 持续自动化意外停止

**解决方案**:
- 检查日志文件
- 验证进程状态
- 重启服务

```bash
# 检查进程状态
python scripts/testing/start_ai_coverage_automation.py status

# 重启服务
python scripts/testing/start_ai_coverage_automation.py stop
python scripts/testing/start_ai_coverage_automation.py start --mode continuous
```

## 📊 性能指标

### 目标指标

- **总体覆盖率**: ≥ 85%
- **AI测试通过率**: ≥ 90%
- **核心模块覆盖率**: ≥ 95%
- **自动化程度**: ≥ 100%

### 监控指标

- 执行成功率
- 平均覆盖率
- 测试通过率
- 生成文件数量
- 执行时间

## 🔄 最佳实践

### 1. 渐进式提升

不要一次性设置过高的覆盖率目标，建议：

1. 先设置50%的目标
2. 逐步提升到70%
3. 最终达到85%以上

### 2. 优先级管理

按重要性分配测试资源：

1. **关键模块**: 基础设施、核心功能
2. **重要模块**: 数据处理、交易引擎
3. **一般模块**: 工具类、辅助功能

### 3. 定期维护

- 每周检查覆盖率报告
- 每月清理缓存文件
- 每季度更新AI提示词

### 4. 团队协作

- 将AI生成的测试代码纳入代码审查
- 定期分享覆盖率提升经验
- 建立测试质量反馈机制

## 📚 相关文档

- [测试覆盖率提升计划](enhance_test_coverage_plan.py)
- [测试框架文档](../testing/README.md)
- [项目架构文档](../../architecture/README.md)

## 🤝 贡献指南

欢迎提交问题和改进建议：

1. 报告AI生成测试的问题
2. 优化AI提示词
3. 改进覆盖率分析算法
4. 添加新的自动化功能

---

**版本**: v1.0  
**最后更新**: 2025-01-20  
**维护者**: RQA2025测试团队 