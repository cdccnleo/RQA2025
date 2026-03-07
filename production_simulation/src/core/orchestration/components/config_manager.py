"""
流程配置管理组件

职责:
- 流程配置的CRUD操作
- 配置文件的持久化
- 配置验证
"""

import os
import json
import logging
from typing import Dict, List, Optional
from dataclasses import asdict

from ..models.process_models import ProcessConfig

logger = logging.getLogger(__name__)


class ProcessConfigManager:
    """
    流程配置管理组件

    管理流程配置的存储和检索
    """

    def __init__(self, config: 'ConfigManagerConfig'):
        """
        初始化配置管理器

        Args:
            config: 配置管理器配置
        """
        self.config = config
        self.configs: Dict[str, ProcessConfig] = {}
        self._load_configs()

        logger.info(f"流程配置管理器初始化完成 (目录: {config.config_dir})")

    def get_config(self, process_id: str) -> Optional[ProcessConfig]:
        """获取配置"""
        return self.configs.get(process_id)

    def save_config(self, config: ProcessConfig) -> bool:
        """保存配置"""
        try:
            if self.config.enable_validation:
                errors = self.validate_config(config)
                if errors:
                    logger.error(f"配置验证失败: {errors}")
                    return False

            # 备份旧配置
            if self.config.backup_configs and config.process_id in self.configs:
                self._backup_config(config.process_id)

            # 保存到文件
            if self.config.auto_save:
                config_path = os.path.join(self.config.config_dir, f"{config.process_id}.json")
                os.makedirs(self.config.config_dir, exist_ok=True)

                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(asdict(config), f, indent=2, ensure_ascii=False)

            # 更新内存缓存
            self.configs[config.process_id] = config
            logger.info(f"配置已保存: {config.process_id}")
            return True

        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False

    def delete_config(self, process_id: str) -> bool:
        """删除配置"""
        try:
            if process_id in self.configs:
                del self.configs[process_id]

                # 删除文件
                config_path = os.path.join(self.config.config_dir, f"{process_id}.json")
                if os.path.exists(config_path):
                    os.remove(config_path)

                logger.info(f"配置已删除: {process_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"删除配置失败: {e}")
            return False

    def list_configs(self) -> List[ProcessConfig]:
        """列出所有配置"""
        return list(self.configs.values())

    def validate_config(self, config: ProcessConfig) -> List[str]:
        """验证配置"""
        errors = []

        if not config.process_id:
            errors.append("process_id不能为空")
        if not config.name:
            errors.append("name不能为空")
        if config.max_retries < 0:
            errors.append("max_retries不能为负数")
        if config.timeout <= 0:
            errors.append("timeout必须大于0")

        return errors

    def get_status(self) -> Dict:
        """获取管理器状态"""
        return {
            'total_configs': len(self.configs),
            'config_dir': self.config.config_dir,
            'auto_save': self.config.auto_save
        }

    def _load_configs(self):
        """加载配置"""
        try:
            if not os.path.exists(self.config.config_dir):
                os.makedirs(self.config.config_dir, exist_ok=True)
                return

            for filename in os.listdir(self.config.config_dir):
                if filename.endswith('.json'):
                    config_path = os.path.join(self.config.config_dir, filename)
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                            config = ProcessConfig(**config_data)
                            self.configs[config.process_id] = config
                    except Exception as e:
                        logger.error(f"加载配置失败: {filename}, {e}")
        except Exception as e:
            logger.error(f"加载配置目录失败: {e}")

    def _backup_config(self, process_id: str):
        """备份配置"""
        # 简化实现
        pass


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..configs.orchestrator_configs import ConfigManagerConfig
