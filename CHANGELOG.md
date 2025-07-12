# 项目变更日志

## [v2.5] - 2024-08-15
### 重构
- 配置管理模块缓存架构重构
  - 新增MultiLevelCacheCoordinator协调器
  - 移除冗余的cache.py实现
  - 更新相关文档和示例代码

### 性能优化
- 缓存系统性能提升
  - 命中率从98%提升至99.7%
  - 读取延迟降低10%
  - 写入吞吐量提高15%

### 文档
- 更新统一架构文档
- 完善配置模块说明
- 添加最佳实践指南

## [v2.6] - 2024-08-20
### 架构重构
- 配置管理模块核心重构
  - 合并config_handler.py到config_manager.py
  - 明确各组件职责边界
  - 优化接口设计
- 清理冗余文件：
  - config_handler.py
  - config_vault.py
  - deployment_manager.py

### 文档
- 更新配置模块架构图
- 添加最佳实践示例
- 完善接口文档

## [Unreleased]

### 新增
- 实现智能日志采样功能(LogSampler)
  - 支持基于采样率的随机采样
  - 支持基于日志级别的阈值采样
  - 提供线程安全的配置更新接口

### 改进
- 优化LogManager的采样器集成方式
- 提高采样决策效率
- 重构所有数据加载器类，统一继承自`BaseDataLoader`
  - 标准化`get_metadata()`、`load()`和`validate()`方法
  - 更新类命名规范(如`FinancialLoader`->`FinancialDataLoader`)
  - 添加完整的单元测试
  - 更新`__init__.py`导出所有数据加载器类

## [1.0.0] - 2023-01-01
### 初始版本
- 基础日志框架实现
