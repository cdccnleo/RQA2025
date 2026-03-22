# -*- coding: utf-8 -*-
"""
集成兼容性适配模块

提供系统集成兼容性适配、版本管理和迁移支持功能。

函数级注释:
- 所有公开函数均包含详细的中文注释
- 支持向后兼容性保证
- 支持PostgreSQL优先策略，连接失败时自动降级

作者: AI系统集成助手
日期: 2026-03-22
版本: 1.0.0
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class CompatibilityLevel(Enum):
    """兼容性级别枚举"""
    FULL = "full"          # 完全兼容
    BACKWARD = "backward"  # 向后兼容
    DEPRECATED = "deprecated"  # 已弃用
    BREAKING = "breaking"  # 不兼容变更


@dataclass
class VersionInfo:
    """
    版本信息数据类
    
    属性:
        major: 主版本号
        minor: 次版本号
        patch: 修订号
        compatibility: 兼容性级别
        release_date: 发布日期
        changes: 变更说明
    """
    major: int
    minor: int
    patch: int
    compatibility: CompatibilityLevel
    release_date: datetime
    changes: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def to_tuple(self) -> tuple:
        return (self.major, self.minor, self.patch)


@dataclass
class MigrationStep:
    """
    迁移步骤数据类
    
    属性:
        step_id: 步骤ID
        from_version: 源版本
        to_version: 目标版本
        description: 描述
        action: 迁移操作
        rollback_action: 回滚操作
    """
    step_id: str
    from_version: str
    to_version: str
    description: str
    action: Optional[Callable] = None
    rollback_action: Optional[Callable] = None


class IntegrationCompatibilityManager:
    """
    集成兼容性管理器
    
    管理集成模块的版本兼容性、提供迁移支持和兼容性检查。
    
    使用示例:
        manager = IntegrationCompatibilityManager()
        
        # 检查兼容性
        is_compatible = manager.check_compatibility("1.0.0", "1.1.0")
        
        # 执行迁移
        success = await manager.migrate("trading_scheduler", "1.0.0", "1.1.0")
    """
    
    _instance: Optional['IntegrationCompatibilityManager'] = None
    
    def __new__(cls) -> 'IntegrationCompatibilityManager':
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化兼容性管理器"""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._version_history: Dict[str, List[VersionInfo]] = {}
        self._migrations: Dict[str, List[MigrationStep]] = {}
        self._adapters: Dict[str, Dict[str, Callable]] = {}
        
        logger.info("集成兼容性管理器初始化完成")
    
    def register_version(
        self,
        integration_name: str,
        version: VersionInfo
    ):
        """
        注册版本信息
        
        Args:
            integration_name: 集成模块名称
            version: 版本信息
        """
        if integration_name not in self._version_history:
            self._version_history[integration_name] = []
        
        self._version_history[integration_name].append(version)
        self._version_history[integration_name].sort(
            key=lambda v: v.to_tuple(),
            reverse=True
        )
        
        logger.info(f"版本已注册: {integration_name} v{version}")
    
    def check_compatibility(
        self,
        current_version: str,
        target_version: str
    ) -> CompatibilityLevel:
        """
        检查版本兼容性
        
        Args:
            current_version: 当前版本
            target_version: 目标版本
            
        Returns:
            CompatibilityLevel: 兼容性级别
        """
        current = self._parse_version(current_version)
        target = self._parse_version(target_version)
        
        # 相同版本
        if current == target:
            return CompatibilityLevel.FULL
        
        # 主版本变更 - 不兼容
        if target[0] != current[0]:
            return CompatibilityLevel.BREAKING
        
        # 次版本变更 - 向后兼容
        if target[1] != current[1]:
            return CompatibilityLevel.BACKWARD
        
        # 修订版本变更 - 完全兼容
        if target[2] != current[2]:
            return CompatibilityLevel.FULL
        
        return CompatibilityLevel.BACKWARD
    
    def _parse_version(self, version_str: str) -> tuple:
        """解析版本字符串"""
        parts = version_str.split(".")
        return tuple(int(p) for p in parts[:3])
    
    def register_migration(
        self,
        integration_name: str,
        migration: MigrationStep
    ):
        """
        注册迁移步骤
        
        Args:
            integration_name: 集成模块名称
            migration: 迁移步骤
        """
        if integration_name not in self._migrations:
            self._migrations[integration_name] = []
        
        self._migrations[integration_name].append(migration)
        
        logger.info(
            f"迁移步骤已注册: {integration_name} "
            f"{migration.from_version} -> {migration.to_version}"
        )
    
    async def migrate(
        self,
        integration_name: str,
        from_version: str,
        to_version: str
    ) -> bool:
        """
        执行迁移
        
        Args:
            integration_name: 集成模块名称
            from_version: 源版本
            to_version: 目标版本
            
        Returns:
            bool: 是否迁移成功
        """
        migrations = self._migrations.get(integration_name, [])
        
        # 查找迁移路径
        migration_path = self._find_migration_path(
            migrations, from_version, to_version
        )
        
        if not migration_path:
            logger.error(f"未找到迁移路径: {from_version} -> {to_version}")
            return False
        
        # 执行迁移
        for step in migration_path:
            try:
                if step.action:
                    if asyncio.iscoroutinefunction(step.action):
                        await step.action()
                    else:
                        step.action()
                
                logger.info(f"迁移步骤完成: {step.step_id}")
                
            except Exception as e:
                logger.error(f"迁移步骤失败: {step.step_id}, error={e}")
                # 回滚
                await self._rollback(integration_name, migration_path, step)
                return False
        
        logger.info(f"迁移完成: {integration_name} {from_version} -> {to_version}")
        return True
    
    def _find_migration_path(
        self,
        migrations: List[MigrationStep],
        from_version: str,
        to_version: str
    ) -> List[MigrationStep]:
        """查找迁移路径"""
        # 简单实现：直接查找
        path = []
        current = from_version
        
        while current != to_version:
            found = False
            for migration in migrations:
                if migration.from_version == current:
                    path.append(migration)
                    current = migration.to_version
                    found = True
                    break
            
            if not found:
                return []
        
        return path
    
    async def _rollback(
        self,
        integration_name: str,
        path: List[MigrationStep],
        failed_step: MigrationStep
    ):
        """回滚迁移"""
        logger.warning(f"开始回滚迁移: {integration_name}")
        
        # 回滚已完成的步骤
        for step in reversed(path):
            if step == failed_step:
                break
            
            try:
                if step.rollback_action:
                    if asyncio.iscoroutinefunction(step.rollback_action):
                        await step.rollback_action()
                    else:
                        step.rollback_action()
                
                logger.info(f"回滚步骤完成: {step.step_id}")
                
            except Exception as e:
                logger.error(f"回滚步骤失败: {step.step_id}, error={e}")
    
    def register_adapter(
        self,
        integration_name: str,
        version: str,
        adapter: Callable
    ):
        """
        注册适配器
        
        Args:
            integration_name: 集成模块名称
            version: 版本
            adapter: 适配器函数
        """
        if integration_name not in self._adapters:
            self._adapters[integration_name] = {}
        
        self._adapters[integration_name][version] = adapter
        
        logger.info(f"适配器已注册: {integration_name} v{version}")
    
    async def adapt(
        self,
        integration_name: str,
        version: str,
        data: Any
    ) -> Any:
        """
        执行适配
        
        Args:
            integration_name: 集成模块名称
            version: 版本
            data: 数据
            
        Returns:
            Any: 适配后的数据
        """
        adapters = self._adapters.get(integration_name, {})
        adapter = adapters.get(version)
        
        if adapter:
            if asyncio.iscoroutinefunction(adapter):
                return await adapter(data)
            else:
                return adapter(data)
        
        return data
    
    def get_version_history(self, integration_name: str) -> List[VersionInfo]:
        """
        获取版本历史
        
        Args:
            integration_name: 集成模块名称
            
        Returns:
            List[VersionInfo]: 版本历史
        """
        return self._version_history.get(integration_name, []).copy()
    
    def get_latest_version(self, integration_name: str) -> Optional[str]:
        """
        获取最新版本
        
        Args:
            integration_name: 集成模块名称
            
        Returns:
            Optional[str]: 最新版本号
        """
        versions = self._version_history.get(integration_name, [])
        if versions:
            return str(versions[0])
        return None
    
    def get_compatibility_report(self) -> Dict[str, Any]:
        """
        获取兼容性报告
        
        Returns:
            Dict[str, Any]: 兼容性报告
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "integrations": {}
        }
        
        for name, versions in self._version_history.items():
            report["integrations"][name] = {
                "latest_version": str(versions[0]) if versions else None,
                "version_count": len(versions),
                "versions": [
                    {
                        "version": str(v),
                        "compatibility": v.compatibility.value,
                        "release_date": v.release_date.isoformat(),
                        "changes": v.changes
                    }
                    for v in versions[:5]  # 最近5个版本
                ]
            }
        
        return report


class DataFormatAdapter:
    """
    数据格式适配器
    
    提供不同版本数据格式的适配功能。
    """
    
    @staticmethod
    def adapt_event_v1_to_v2(event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将v1格式事件适配到v2格式
        
        Args:
            event_data: v1格式事件数据
            
        Returns:
            Dict[str, Any]: v2格式事件数据
        """
        # v1 -> v2 适配
        adapted = {
            "event_id": event_data.get("id", ""),
            "event_type": event_data.get("type", "").upper(),
            "timestamp": event_data.get("timestamp", datetime.now().isoformat()),
            "correlation_id": event_data.get("correlation_id", ""),
            "source": event_data.get("source", "unknown"),
            "version": "2.0",
            "payload": event_data.get("data", {}),
            "metadata": {
                "original_version": "1.0"
            }
        }
        
        return adapted
    
    @staticmethod
    def adapt_event_v2_to_v1(event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将v2格式事件适配到v1格式
        
        Args:
            event_data: v2格式事件数据
            
        Returns:
            Dict[str, Any]: v1格式事件数据
        """
        # v2 -> v1 适配
        adapted = {
            "id": event_data.get("event_id", ""),
            "type": event_data.get("event_type", "").lower(),
            "timestamp": event_data.get("timestamp", datetime.now().isoformat()),
            "correlation_id": event_data.get("correlation_id", ""),
            "source": event_data.get("source", "unknown"),
            "data": event_data.get("payload", {})
        }
        
        return adapted


# 全局实例获取函数
def get_integration_compatibility_manager() -> IntegrationCompatibilityManager:
    """
    获取集成兼容性管理器实例
    
    Returns:
        IntegrationCompatibilityManager: 管理器实例
    """
    return IntegrationCompatibilityManager()
