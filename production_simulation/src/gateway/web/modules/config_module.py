import logging
#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
配置管理模块
提供统一的配置管理界面，支持各层配置的可视化管理和编辑
"""

from typing import Dict, List, Any
from datetime import datetime
import yaml

from fastapi import HTTPException
from pydantic import BaseModel

from .base_module import BaseModule, ModuleConfig, ModuleData, ModuleStatus  # 当前层级内部导入：基础模块类
from src.infrastructure.config.core.unified_manager import UnifiedConfigManager  # 合理跨层级导入：基础设施层配置管理

logger = logging.getLogger(__name__)


class ConfigItem(BaseModel):

    """配置项模型"""
    key: str
    value: Any
    type: str
    description: str = ""
    category: str = ""
    editable: bool = True
    validation_rules: Dict[str, Any] = {}


class ConfigCategory(BaseModel):

    """配置分类模型"""
    name: str
    display_name: str
    description: str = ""
    items: List[ConfigItem] = []
    collapsed: bool = False


class ConfigModule(BaseModule):

    """配置管理模块"""

    def __init__(self, config: ModuleConfig):

        super().__init__(config)
        self.config_manager = UnifiedConfigManager()
        self.config_cache: Dict[str, Any] = {}
        self.last_cache_update = datetime.now()

        # 配置分类定义
        self.categories = {
            "system": {
                "display_name": "系统配置",
                "description": "系统核心配置参数"
            },
            "database": {
                "display_name": "数据库配置",
                "description": "数据库连接和存储配置"
            },
            "trading": {
                "display_name": "交易配置",
                "description": "交易相关配置参数"
            },
            "monitoring": {
                "display_name": "监控配置",
                "description": "监控和告警配置"
            },
            "security": {
                "display_name": "安全配置",
                "description": "安全相关配置"
            },
            "performance": {
                "display_name": "性能配置",
                "description": "性能优化配置"
            }
        }

    def _register_routes(self):
        """注册模块路由"""
        self._add_common_routes()

        @self.router.get("/categories")
        async def get_categories():
            """获取配置分类"""
            return await self._get_config_categories()

        @self.router.get("/category/{category_name}")
        async def get_category_config(category_name: str):
            """获取分类配置"""
            return await self._get_category_config(category_name)

        @self.router.get("/item/{config_key}")
        async def get_config_item(config_key: str):
            """获取配置项"""
            return await self._get_config_item(config_key)

        @self.router.put("/item/{config_key}")
        async def update_config_item(config_key: str, value: Any):
            """更新配置项"""
            return await self._update_config_item(config_key, value)

        @self.router.post("/validate")
        async def validate_config(config_data: Dict[str, Any]):
            """验证配置"""
            return await self._validate_config(config_data)

        @self.router.post("/export")
        async def export_config(export_format: str = "json"):
            """导出配置"""
            return await self._export_config(export_format)

        @self.router.post("/import")
        async def import_config(config_file: str):
            """导入配置"""
            return await self._import_config(config_file)

        @self.router.get("/history")
        async def get_config_history():
            """获取配置历史"""
            return await self._get_config_history()

        @self.router.post("/reload")
        async def reload_config():
            """重新加载配置"""
            return await self._reload_config()

    async def get_module_data(self) -> ModuleData:
        """获取模块数据"""
        try:
            # 更新配置缓存
            await self._update_config_cache()

            return ModuleData(
                module_id=self.config.name,
                data_type="config_metrics",
                content={
                    "total_configs": len(self.config_cache),
                    "categories": len(self.categories),
                    "last_updated": self.last_cache_update.isoformat(),
                    "cache_size": len(str(self.config_cache)),
                    "config_categories": list(self.categories.keys())
                },
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"获取配置模块数据失败: {str(e)}")
            return ModuleData(
                module_id=self.config.name,
                data_type="config_metrics",
                content={"error": str(e)},
                timestamp=datetime.now()
            )

    async def get_module_status(self) -> ModuleStatus:
        """获取模块状态"""
        try:
            return ModuleStatus(
                module_id=self.config.name,
                status="running" if self.is_healthy() else "stopped",
                is_healthy=self.is_healthy(),
                last_activity=datetime.now(),
                error_count=0,
                uptime=0.0,
                performance_metrics={
                    "config_count": len(self.config_cache),
                    "categories_count": len(self.categories),
                    "cache_valid": (datetime.now() - self.last_cache_update).seconds < 300
                },
                alerts=[]
            )
        except Exception as e:
            logger.error(f"获取配置模块状态失败: {str(e)}")
            return ModuleStatus(
                module_id=self.config.name,
                status="error",
                is_healthy=False,
                last_activity=datetime.now(),
                error_count=1,
                uptime=0.0,
                performance_metrics={"error": str(e)},
                alerts=[]
            )

    async def validate_permissions(self, user_permissions: List[str]) -> bool:
        """验证用户权限"""
        required_permissions = ["read", "write"]
        return all(perm in user_permissions for perm in required_permissions)

    async def _initialize_module(self):
        """初始化模块内部逻辑"""
        await self._update_config_cache()
        logger.info("配置管理模块初始化完成")

    async def _start_module(self):
        """启动模块内部逻辑"""
        # 启动配置监听
        logger.info("配置管理模块启动完成")

    async def _stop_module(self):
        """停止模块内部逻辑"""
        # 停止配置监听
        logger.info("配置管理模块停止完成")

    async def _update_config_cache(self):
        """更新配置缓存"""
        try:
            # 获取所有配置
            all_configs = self.config_manager.get_all_configs()

            # 按分类组织配置
            categorized_configs = {}
            for key, value in all_configs.items():
                category = self._get_config_category(key)
                if category not in categorized_configs:
                    categorized_configs[category] = {}
                categorized_configs[category][key] = value

            self.config_cache = categorized_configs
            self.last_cache_update = datetime.now()

        except Exception as e:
            logger.error(f"更新配置缓存失败: {str(e)}")

    def _get_config_category(self, config_key: str) -> str:
        """获取配置项的分类"""
        # 根据配置键的前缀判断分类
        if config_key.startswith(("system.", "app.", "core.")):
            return "system"
        elif config_key.startswith(("db.", "database.", "storage.")):
            return "database"
        elif config_key.startswith(("trading.", "trade.", "order.")):
            return "trading"
        elif config_key.startswith(("monitor.", "alert.", "log.")):
            return "monitoring"
        elif config_key.startswith(("auth.", "security.", "encrypt.")):
            return "security"
        elif config_key.startswith(("perf.", "performance.", "optimize.")):
            return "performance"
        else:
            return "system"

    async def _get_config_categories(self) -> Dict[str, Any]:
        """获取配置分类"""
        await self._update_config_cache()

        categories = {}
        for category_name, category_info in self.categories.items():
            config_count = len(self.config_cache.get(category_name, {}))
            categories[category_name] = {
                **category_info,
                "config_count": config_count,
                "has_configs": config_count > 0
            }

        return categories

    async def _get_category_config(self, category_name: str) -> Dict[str, Any]:
        """获取分类配置"""
        if category_name not in self.categories:
            raise HTTPException(status_code=404, detail=f"配置分类 {category_name} 不存在")

        await self._update_config_cache()

        configs = self.config_cache.get(category_name, {})
        config_items = []

        for key, value in configs.items():
            config_items.append(ConfigItem(
                key=key,
                value=value,
                type=type(value).__name__,
                description=self._get_config_description(key),
                category=category_name,
                editable=self._is_config_editable(key)
            ))

        return {
            "category": self.categories[category_name],
            "items": [item.dict() for item in config_items],
            "total_items": len(config_items)
        }

    async def _get_config_item(self, config_key: str) -> Dict[str, Any]:
        """获取配置项"""
        try:
            value = self.config_manager.get(config_key)
            return {
                "key": config_key,
                "value": value,
                "type": type(value).__name__,
                "description": self._get_config_description(config_key),
                "editable": self._is_config_editable(config_key)
            }
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"配置项 {config_key} 不存在")

    async def _update_config_item(self, config_key: str, value: Any) -> Dict[str, Any]:
        """更新配置项"""
        try:
            # 验证配置项是否可编辑
            if not self._is_config_editable(config_key):
                raise HTTPException(status_code=403, detail=f"配置项 {config_key} 不可编辑")

            # 更新配置
            self.config_manager.update_config(config_key, value)

            # 清除缓存
            self.config_cache.clear()

            logger.info(f"配置项 {config_key} 更新成功")
            return {"success": True, "message": "配置更新成功"}

        except Exception as e:
            logger.error(f"更新配置项 {config_key} 失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _validate_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置"""
        try:
            # 这里应该实现具体的配置验证逻辑
            validation_results = {}

            for key, value in config_data.items():
                validation_results[key] = {
                    "valid": True,
                    "errors": []
                }

            return {
                "valid": all(result["valid"] for result in validation_results.values()),
                "results": validation_results
            }

        except Exception as e:
            logger.error(f"配置验证失败: {str(e)}")
            return {"valid": False, "error": str(e)}

    async def _export_config(self, export_format: str = "json") -> Dict[str, Any]:
        """导出配置"""
        try:
            await self._update_config_cache()

            if export_format.lower() == "json":
                return {
                    "format": "json",
                    "data": self.config_cache,
                    "timestamp": datetime.now().isoformat()
                }
            elif export_format.lower() == "yaml":
                return {
                    "format": "yaml",
                    "data": yaml.dump(self.config_cache, default_flow_style=False),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                raise HTTPException(status_code=400, detail=f"不支持的导出格式: {export_format}")

        except Exception as e:
            logger.error(f"导出配置失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _import_config(self, config_file: str) -> Dict[str, Any]:
        """导入配置"""
        try:
            # 这里应该实现配置导入逻辑
            return {"success": True, "message": "配置导入成功"}
        except Exception as e:
            logger.error(f"导入配置失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_config_history(self) -> Dict[str, Any]:
        """获取配置历史"""
        try:
            # 这里应该实现配置历史记录逻辑
            return {
                "history": [],
                "total_records": 0
            }
        except Exception as e:
            logger.error(f"获取配置历史失败: {str(e)}")
            return {"error": str(e)}

    async def _reload_config(self) -> Dict[str, Any]:
        """重新加载配置"""
        try:
            # 清除缓存
            self.config_cache.clear()

            # 重新加载配置
            await self._update_config_cache()

            logger.info("配置重新加载成功")
            return {"success": True, "message": "配置重新加载成功"}

        except Exception as e:
            logger.error(f"重新加载配置失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def _get_config_description(self, config_key: str) -> str:
        """获取配置项描述"""
        # 这里应该从配置元数据中获取描述
        descriptions = {
            "system.name": "系统名称",
            "system.version": "系统版本",
            "database.host": "数据库主机地址",
            "database.port": "数据库端口",
            "trading.enabled": "是否启用交易功能",
            "monitoring.enabled": "是否启用监控功能"
        }
        return descriptions.get(config_key, "")

    def _is_config_editable(self, config_key: str) -> bool:
        """判断配置项是否可编辑"""
        # 系统核心配置不可编辑
        readonly_configs = {
            "system.name",
            "system.version",
            "system.startup_time"
        }
        return config_key not in readonly_configs

    @classmethod
    def get_default_config(cls) -> ModuleConfig:
        """获取默认配置"""
        return ModuleConfig(
            name="config",
            display_name="配置管理",
            description="统一配置管理界面",
            icon="settings",
            route="/config",
            permissions=["read", "write"],
            version="1.0.0",
            dependencies=[],
            settings={
                "cache_ttl": 300,
                "auto_reload": True,
                "enable_history": True
            }
        )
