"""
RQA2025 统一Web管理界面 - 模块化组件架构

模块化设计原则：
1. 单一职责 - 每个模块负责特定功能领域
2. 松耦合 - 模块间通过接口通信，减少依赖
3. 高内聚 - 相关功能集中在同一模块内
4. 可扩展 - 支持动态注册新模块
5. 可配置 - 模块行为通过配置控制

模块分类：
- 核心模块：系统概览、配置管理
- 业务模块：策略管理、数据管理、回测管理
- 运维模块：监控告警、资源管理
- 管理模块：用户管理、权限控制
"""

from .base_module import BaseModule
from .module_registry import ModuleRegistry
from .module_factory import ModuleFactory

__all__ = [
    'BaseModule',
    'ModuleRegistry',
    'ModuleFactory'
]
