# 智能测试运行器功能说明文档

## 概述

`scripts/testing/run_tests.py` 是一个功能强大的智能测试运行器，支持多种测试模式、覆盖率分析、结果持久化和智能错误处理。

## 主要功能特性

### 1. 测试优先级分组
- **STABLE**: 稳定测试 - 核心功能
- **MODERATE**: 中等测试 - 重要功能  
- **EXPERIMENTAL**: 实验性测试 - 新功能
- **FEATURES**: 特征处理层测试 - 专门分组

### 2. 多种运行模式

#### 2.1 优先级模式
```bash
# 运行所有测试（按优先级顺序）
python scripts/testing/run_tests.py --enable-coverage --save-results

# 只运行稳定测试
python scripts/testing/run_tests.py --priority stable --enable-coverage --save-results

# 只运行特征处理层测试
python scripts/testing/run_tests.py --priority features --enable-coverage --save-results
```

#### 2.2 模块分析模式
```bash
# 分析特定模块的测试覆盖情况
python scripts/testing/run_tests.py --module feature_manager --enable-coverage --save-results

# 分析GPU处理模块
python scripts/testing/run_tests.py --module gpu_processor --enable-coverage --save-results
```

#### 2.3 路径指定模式 ⭐ 新功能
```bash
# 运行单个测试文件
python scripts/testing/run_tests.py --path tests/unit/features/test_feature_manager.py --enable-coverage --save-results

# 运行整个目录的测试
python scripts/testing/run_tests.py --path tests/unit/features/ --enable-coverage --save-results

# 使用通配符模式
python scripts/testing/run_tests.py --path "tests/unit/features/test_*manager*.py" --enable-coverage --save-results

# 运行多个路径
python scripts/testing/run_tests.py --path tests/unit/features/test_feature_manager.py tests/unit/features/test_gpu_technical_processor.py --enable-coverage --save-results
```

### 3. 路径参数功能详解

#### 3.1 支持的文件类型
- 单个 `.py` 测试文件
- 目录路径（自动递归查找测试文件）
- 通配符模式（支持 `*` 和 `?`）
- 相对路径和绝对路径

#### 3.2 路径验证规则
- 文件必须以 `.py` 结尾
- 文件名必须包含 `test_` 前缀
- 自动过滤无效路径
- 支持模糊匹配

#### 3.3 通配符示例
```bash
# 所有manager相关测试
test_*manager*.py

# 所有GPU相关测试
test_gpu*.py

# 特定模块的增强版本测试
test_*_enhanced.py

# 覆盖率增强版本测试
test_*_coverage_enhanced.py
```

### 4. 覆盖率分析功能

#### 4.1 覆盖率收集
- 使用 `pytest-cov` 收集覆盖率数据
- 支持多种覆盖率报告格式
- 自动生成HTML覆盖率报告

#### 4.2 覆盖率统计
- 整体覆盖率统计
- 按优先级分组统计
- 按架构层分组统计
- 模块级详细统计

#### 4.3 覆盖率分布分析
- 优秀(90%+): 覆盖率分布
- 良好(70-89%): 覆盖率分布
- 一般(50-69%): 覆盖率分布
- 较差(30-49%): 覆盖率分布
- 很差(<30%): 覆盖率分布

### 5. 结果持久化

#### 5.1 SQLite数据库存储
- 测试结果历史记录
- 覆盖率趋势分析
- 性能指标统计
- 层级别分析数据

#### 5.2 自动数据库升级
- 检测现有数据库结构
- 自动添加新列
- 向后兼容性保证
- 数据完整性保护

### 6. 智能错误处理

#### 6.1 测试失败处理
- 允许部分测试失败
- 智能重试机制
- 详细错误信息记录
- 失败原因分析

#### 6.2 性能监控
- 测试执行时间统计
- 超时检测和处理
- 资源使用监控
- 性能瓶颈识别

## 使用示例

### 示例1: 快速验证特定功能
```bash
# 只测试特征管理器相关功能
python scripts/testing/run_tests.py --path "tests/unit/features/test_feature_manager*.py" --enable-coverage
```

### 示例2: 分析模块测试质量
```bash
# 分析插件系统的测试覆盖情况
python scripts/testing/run_tests.py --module plugin_system --enable-coverage --save-results
```

### 示例3: 批量测试多个模块
```bash
# 测试所有核心管理相关模块
python scripts/testing/run_tests.py --path "tests/unit/features/test_*manager*.py" --enable-coverage --save-results
```

### 示例4: 完整测试套件
```bash
# 运行所有测试，生成完整报告
python scripts/testing/run_tests.py --enable-coverage --save-results --verbose
```

## 配置选项

### 基本参数
- `--max-workers`: 最大并行数（默认4）
- `--verbose`: 详细输出模式
- `--enable-coverage`: 启用覆盖率检查
- `--save-results`: 保存结果到数据库

### 运行模式参数
- `--priority`: 指定测试优先级
- `--module`: 指定要分析的模块
- `--path`: 指定要运行的测试路径

## 输出报告

### 1. 测试执行报告
- 测试文件执行状态
- 通过/失败/错误统计
- 执行时间统计
- 覆盖率统计

### 2. 模块分析报告
- 模块概览统计
- 文件级详细统计
- 覆盖率分布分析
- 性能指标分析
- 改进建议

### 3. 系统健康度评分
- 综合评分（0-100）
- 测试通过率权重
- 稳定性权重
- 覆盖率权重

## 最佳实践

### 1. 日常开发
- 使用 `--path` 参数快速测试修改的模块
- 结合 `--enable-coverage` 监控代码覆盖率
- 定期运行完整测试套件验证系统稳定性

### 2. 持续集成
- 使用 `--priority stable` 确保核心功能稳定
- 结合 `--save-results` 跟踪测试趋势
- 设置覆盖率阈值告警

### 3. 性能优化
- 使用 `--max-workers` 调整并行度
- 监控测试执行时间
- 识别性能瓶颈模块

## 故障排除

### 常见问题

#### 1. 路径参数无效
- 检查文件路径是否正确
- 确认文件名包含 `test_` 前缀
- 验证文件扩展名为 `.py`

#### 2. 覆盖率数据异常
- 检查 `pytest-cov` 是否正确安装
- 确认源代码路径配置正确
- 验证覆盖率报告格式

#### 3. 数据库错误
- 检查数据库文件权限
- 确认SQLite版本兼容性
- 尝试删除数据库文件重新创建

### 调试技巧
- 使用 `--verbose` 参数获取详细输出
- 检查生成的覆盖率报告文件
- 查看数据库中的测试结果记录

## 更新日志

### v2.0.0 (2025-01-28)
- ✅ 新增 `--path` 参数支持
- ✅ 支持目录路径自动展开
- ✅ 支持通配符模式匹配
- ✅ 数据库结构自动升级
- ✅ 路径验证和错误处理完善

### v1.5.0 (2025-01-28)
- ✅ 新增 `--module` 参数支持
- ✅ 特征处理层覆盖率分析
- ✅ 模块级详细统计
- ✅ 智能测试分组

### v1.0.0 (2025-01-27)
- ✅ 基础测试运行器功能
- ✅ 优先级分组测试
- ✅ 覆盖率收集和统计
- ✅ 结果持久化存储

## 技术支持

如有问题或建议，请：
1. 查看本文档的故障排除部分
2. 检查测试输出日志
3. 联系测试团队获取支持
