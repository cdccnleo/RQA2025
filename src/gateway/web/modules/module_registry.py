import logging
#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
模块注册表
管理所有模块的注册、生命周期和依赖关系
"""

from typing import Dict, List, Optional, Type
import asyncio

from .base_module import BaseModule, ModuleConfig

logger = logging.getLogger(__name__)


class ModuleRegistry:

    """模块注册表"""

    def __init__(self):

        self.modules: Dict[str, BaseModule] = {}
        self.module_classes: Dict[str, Type[BaseModule]] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.initialization_order: List[str] = []

        logger.info("模块注册表初始化完成")

    def register_module(self, module_class: Type[BaseModule], config: ModuleConfig):
        """注册模块"""
        try:
            # 检查模块是否已注册
            if config.name in self.modules:
                logger.warning(f"模块 {config.name} 已存在，跳过注册")
                return False

            # 创建模块实例
            module_instance = module_class(config)

            # 注册模块
            self.modules[config.name] = module_instance
            self.module_classes[config.name] = module_class
            self.dependencies[config.name] = config.dependencies

            logger.info(f"模块 {config.name} 注册成功")
            return True

        except Exception as e:
            logger.error(f"模块 {config.name} 注册失败: {str(e)}")
            return False

    def unregister_module(self, module_name: str) -> bool:
        """注销模块"""
        try:
            if module_name in self.modules:
                module = self.modules[module_name]

                # 停止模块
                asyncio.create_task(module.stop())

                # 移除模块
                del self.modules[module_name]
                del self.module_classes[module_name]
                del self.dependencies[module_name]

                logger.info(f"模块 {module_name} 注销成功")
                return True
            else:
                logger.warning(f"模块 {module_name} 不存在")
                return False

        except Exception as e:
            logger.error(f"模块 {module_name} 注销失败: {str(e)}")
            return False

    def get_module(self, module_name: str) -> Optional[BaseModule]:
        """获取模块实例"""
        return self.modules.get(module_name)

    def get_all_modules(self) -> Dict[str, BaseModule]:
        """获取所有模块"""
        return self.modules.copy()

    def get_enabled_modules(self) -> Dict[str, BaseModule]:
        """获取启用的模块"""
        return {
            name: module
            for name, module in self.modules.items()
            if module.config.enabled
        }

    def get_module_configs(self) -> List[ModuleConfig]:
        """获取所有模块配置"""
        return [module.config for module in self.modules.values()]

    async def initialize_all_modules(self) -> Dict[str, bool]:
        """初始化所有模块"""
        results = {}

        # 计算初始化顺序（考虑依赖关系）
        init_order = self._calculate_initialization_order()

        for module_name in init_order:
            if module_name in self.modules:
                module = self.modules[module_name]
                if module.config.enabled:
                    try:
                        success = await module.initialize()
                        results[module_name] = success

                        if success:
                            logger.info(f"模块 {module_name} 初始化成功")
                        else:
                            logger.error(f"模块 {module_name} 初始化失败")

                    except Exception as e:
                        logger.error(f"模块 {module_name} 初始化异常: {str(e)}")
                        results[module_name] = False
                else:
                    logger.info(f"模块 {module_name} 已禁用，跳过初始化")
                    results[module_name] = True

        return results

    async def start_all_modules(self) -> Dict[str, bool]:
        """启动所有模块"""
        results = {}

        for module_name, module in self.modules.items():
            if module.config.enabled and module.is_initialized:
                try:
                    success = await module.start()
                    results[module_name] = success

                    if success:
                        logger.info(f"模块 {module_name} 启动成功")
                    else:
                        logger.error(f"模块 {module_name} 启动失败")

                except Exception as e:
                    logger.error(f"模块 {module_name} 启动异常: {str(e)}")
                    results[module_name] = False
            else:
                logger.info(f"模块 {module_name} 未启用或未初始化，跳过启动")
                results[module_name] = True

        return results

    async def stop_all_modules(self) -> Dict[str, bool]:
        """停止所有模块"""
        results = {}

        # 按依赖关系的逆序停止
        stop_order = list(reversed(self._calculate_initialization_order()))

        for module_name in stop_order:
            if module_name in self.modules:
                module = self.modules[module_name]
                try:
                    success = await module.stop()
                    results[module_name] = success

                    if success:
                        logger.info(f"模块 {module_name} 停止成功")
                    else:
                        logger.error(f"模块 {module_name} 停止失败")

                except Exception as e:
                    logger.error(f"模块 {module_name} 停止异常: {str(e)}")
                    results[module_name] = False

        return results

    def get_module_status(self) -> Dict[str, Dict]:
        """获取所有模块状态"""
        status = {}

        for module_name, module in self.modules.items():
            status[module_name] = {
                "name": module.config.name,
                "display_name": module.config.display_name,
                "enabled": module.config.enabled,
                "initialized": module.is_initialized,
                "running": module.is_running,
                "healthy": module.is_healthy(),
                "last_activity": module.last_activity.isoformat(),
                "dependencies": module.config.dependencies
            }

        return status

    def get_healthy_modules(self) -> List[str]:
        """获取健康的模块列表"""
        return [
            name for name, module in self.modules.items()
            if module.is_healthy()
        ]

    def get_failed_modules(self) -> List[str]:
        """获取失败的模块列表"""
        return [
            name for name, module in self.modules.items()
            if not module.is_healthy()
        ]

    def _calculate_initialization_order(self) -> List[str]:
        """计算模块初始化顺序（拓扑排序）"""
        # 简单的拓扑排序实现
        visited = set()
        temp_visited = set()
        order = []

        def dfs(module_name: str):

            if module_name in temp_visited:
                # 检测到循环依赖
                logger.warning(f"检测到循环依赖: {module_name}")
                return
            if module_name in visited:
                return

            temp_visited.add(module_name)

            # 先处理依赖
            for dep in self.dependencies.get(module_name, []):
                if dep in self.modules:
                    dfs(dep)

            temp_visited.remove(module_name)
            visited.add(module_name)
            order.append(module_name)

        # 对所有模块进行DFS
        for module_name in self.modules.keys():
            if module_name not in visited:
                dfs(module_name)

        return order

    def validate_dependencies(self) -> Dict[str, List[str]]:
        """验证模块依赖关系"""
        issues = {}

        for module_name, deps in self.dependencies.items():
            missing_deps = []
            for dep in deps:
                if dep not in self.modules:
                    missing_deps.append(dep)

            if missing_deps:
                issues[module_name] = missing_deps

        return issues

    def get_module_routers(self) -> Dict[str, any]:
        """获取所有模块的路由器"""
        return {
            name: module.get_router()
            for name, module in self.modules.items()
            if module.config.enabled
        }
