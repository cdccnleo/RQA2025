import logging
#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
模块工厂类
用于动态创建和管理模块实例
"""

from typing import Dict, List, Optional, Type
from pathlib import Path
import importlib
import inspect

from .base_module import BaseModule, ModuleConfig

logger = logging.getLogger(__name__)


class ModuleFactory:

    """模块工厂类"""

    def __init__(self):

        self.module_classes: Dict[str, Type[BaseModule]] = {}
        self.module_configs: Dict[str, ModuleConfig] = {}
        self.discovered_modules: List[str] = []

        logger.info("模块工厂初始化完成")

    def register_module_class(self, name: str, module_class: Type[BaseModule]):
        """注册模块类"""
        try:
            if not issubclass(module_class, BaseModule):
                raise ValueError(f"模块类 {name} 必须继承自 BaseModule")

            self.module_classes[name] = module_class
            logger.info(f"模块类 {name} 注册成功")
            return True

        except Exception as e:
            logger.error(f"模块类 {name} 注册失败: {str(e)}")
            return False

    def register_module_config(self, name: str, config: ModuleConfig):
        """注册模块配置"""
        try:
            self.module_configs[name] = config
            logger.info(f"模块配置 {name} 注册成功")
            return True

        except Exception as e:
            logger.error(f"模块配置 {name} 注册失败: {str(e)}")
            return False

    def create_module(self, name: str) -> Optional[BaseModule]:
        """创建模块实例"""
        try:
            if name not in self.module_classes:
                logger.error(f"模块类 {name} 未注册")
                return None

            if name not in self.module_configs:
                logger.error(f"模块配置 {name} 未注册")
                return None

            module_class = self.module_classes[name]
            config = self.module_configs[name]

            module_instance = module_class(config)
            logger.info(f"模块 {name} 创建成功")
            return module_instance

        except Exception as e:
            logger.error(f"模块 {name} 创建失败: {str(e)}")
            return None

    def discover_modules(self, modules_dir: str = None) -> List[str]:
        """自动发现模块"""
        if modules_dir is None:
            modules_dir = Path(__file__).parent

        discovered = []

        try:
            for module_file in Path(modules_dir).glob("*.py"):
                if module_file.name.startswith("__"):
                    continue

                module_name = module_file.stem
                if module_name in ["base_module", "module_registry", "module_factory"]:
                    continue

                try:
                    # 动态导入模块
                    module_path = f"src.infrastructure.web.modules.{module_name}"
                    module = importlib.import_module(module_path)

                    # 查找模块类
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj)
                            and issubclass(obj, BaseModule)
                                and obj != BaseModule):

                            # 检查是否有默认配置
                            if hasattr(obj, 'get_default_config'):
                                config = obj.get_default_config()
                                self.register_module_config(module_name, config)

                            self.register_module_class(module_name, obj)
                            discovered.append(module_name)
                            logger.info(f"发现模块: {module_name}")
                            break

                except Exception as e:
                    logger.warning(f"发现模块 {module_name} 失败: {str(e)}")

        except Exception as e:
            logger.error(f"模块发现过程失败: {str(e)}")

        self.discovered_modules = discovered
        return discovered

    def get_available_modules(self) -> List[str]:
        """获取可用模块列表"""
        return list(self.module_classes.keys())

    def get_module_info(self, name: str) -> Optional[Dict]:
        """获取模块信息"""
        if name not in self.module_classes:
            return None

        module_class = self.module_classes[name]
        config = self.module_configs.get(name)

        return {
            "name": name,
            "class": module_class.__name__,
            "config": config.dict() if config else None,
            "discovered": name in self.discovered_modules
        }

    def get_all_module_info(self) -> Dict[str, Dict]:
        """获取所有模块信息"""
        return {
            name: self.get_module_info(name)
            for name in self.module_classes.keys()
        }
