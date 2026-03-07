"""
业务流程配置管理
提供流程配置的创建、验证、存储和管理功能
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import hashlib

from src.core.constants import DEFAULT_TEST_TIMEOUT

from ..models.models import ProcessConfig

logger = logging.getLogger(__name__)


class ProcessConfigManager:

    """流程配置管理器 - 增强版"""

    def __init__(self, config_dir: str = "./process_configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._config_cache: Dict[str, ProcessConfig] = {}
        self._config_versions: Dict[str, List[str]] = {}

        # 加载现有配置
        self._load_existing_configs()

    def _load_existing_configs(self) -> None:
        """加载现有配置"""
        try:
            for config_file in self.config_dir.glob("*.json"):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)

                    # 验证配置
                    if self._validate_config_data(config_data):
                        config = self._deserialize_config(config_data)
                        self._config_cache[config.process_id] = config

                        # 记录版本信息
                        version_key = f"{config.process_id}_{config.version}"
                        if config.process_id not in self._config_versions:
                            self._config_versions[config.process_id] = []
                        if version_key not in self._config_versions[config.process_id]:
                            self._config_versions[config.process_id].append(version_key)

                except Exception as e:
                    logger.error(f"加载配置文件失败 {config_file}: {e}")

        except Exception as e:
            logger.error(f"加载现有配置失败: {e}")

    def create_config(self, process_id: str, process_name: str,
                      description: Optional[str] = None, **kwargs) -> ProcessConfig:
        """创建新配置"""
        config = ProcessConfig(
            process_id=process_id,
            process_name=process_name,
            description=description,
            **kwargs
        )

        # 验证配置
        if not self._validate_config(config):
            raise ValueError(f"配置验证失败: {process_id}")

        # 保存配置
        self.save_config(config)
        self._config_cache[process_id] = config

        logger.info(f"创建流程配置: {process_id}")
        return config

    def get_config(self, process_id: str, version: Optional[str] = None) -> Optional[ProcessConfig]:
        """获取配置"""
        if version:
            # 获取指定版本
            return self._load_config_version(process_id, version)
        else:
            # 获取最新版本
            return self._config_cache.get(process_id)

    def update_config(self, process_id: str, updates: Dict[str, Any]) -> Optional[ProcessConfig]:
        """更新配置"""
        config = self.get_config(process_id)
        if not config:
            return None

        # 创建新版本
        new_version = self._increment_version(config.version)
        updated_config = ProcessConfig(
            process_id=config.process_id,
            process_name=config.process_name,
            description=config.description,
            version=new_version,
            updated_at=datetime.now(),
            **{**self._serialize_config(config), **updates}
        )

        # 验证并保存
        if self._validate_config(updated_config):
            self.save_config(updated_config)
            self._config_cache[process_id] = updated_config

            # 记录版本
            version_key = f"{process_id}_{new_version}"
            if process_id not in self._config_versions:
                self._config_versions[process_id] = []
            self._config_versions[process_id].append(version_key)

            logger.info(f"更新流程配置: {process_id} -> {new_version}")
            return updated_config

        return None

    def delete_config(self, process_id: str) -> bool:
        """删除配置"""
        try:
            # 删除缓存
            if process_id in self._config_cache:
                del self._config_cache[process_id]

            # 删除文件
            config_file = self.config_dir / f"{process_id}.json"
            if config_file.exists():
                config_file.unlink()

            # 删除版本记录
            if process_id in self._config_versions:
                del self._config_versions[process_id]

            logger.info(f"删除流程配置: {process_id}")
            return True

        except Exception as e:
            logger.error(f"删除配置失败: {e}")
            return False

    def list_configs(self) -> List[Dict[str, Any]]:
        """列出所有配置"""
        configs = []
        for config in self._config_cache.values():
            configs.append({
                'process_id': config.process_id,
                'process_name': config.process_name,
                'description': config.description,
                'version': config.version,
                'created_at': config.created_at.isoformat(),
                'updated_at': config.updated_at.isoformat()
            })
        return configs

    def get_config_versions(self, process_id: str) -> List[str]:
        """获取配置版本列表"""
        return self._config_versions.get(process_id, [])

    def save_config(self, config: ProcessConfig) -> bool:
        """保存配置到文件"""
        try:
            config_data = self._serialize_config(config)
            config_file = self.config_dir / f"{config.process_id}.json"

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, default=str)

            return True

        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False

    def _serialize_config(self, config: ProcessConfig) -> Dict[str, Any]:
        """序列化配置"""
        return {
            'process_id': config.process_id,
            'process_name': config.process_name,
            'description': config.description,
            'data_sources': config.data_sources,
            'data_quality_rules': config.data_quality_rules,
            'feature_config': config.feature_config,
            'gpu_acceleration_enabled': config.gpu_acceleration_enabled,
            'model_configs': config.model_configs,
            'ensemble_method': config.ensemble_method,
            'strategy_configs': config.strategy_configs,
            'risk_check_configs': config.risk_check_configs,
            'trading_configs': config.trading_configs,
            'monitoring_configs': config.monitoring_configs,
            'timeout_seconds': config.timeout_seconds,
            'retry_count': config.retry_count,
            'created_at': config.created_at.isoformat(),
            'updated_at': config.updated_at.isoformat(),
            'version': config.version
        }

    def _deserialize_config(self, data: Dict[str, Any]) -> ProcessConfig:
        """反序列化配置"""
        # 处理时间字段
        created_at = datetime.fromisoformat(
            data['created_at']) if 'created_at' in data else datetime.now()
        updated_at = datetime.fromisoformat(
            data['updated_at']) if 'updated_at' in data else datetime.now()

        return ProcessConfig(
            process_id=data['process_id'],
            process_name=data['process_name'],
            description=data.get('description'),
            data_sources=data.get('data_sources', []),
            data_quality_rules=data.get('data_quality_rules', {}),
            feature_config=data.get('feature_config', {}),
            gpu_acceleration_enabled=data.get('gpu_acceleration_enabled', False),
            model_configs=data.get('model_configs', []),
            ensemble_method=data.get('ensemble_method'),
            strategy_configs=data.get('strategy_configs', []),
            risk_check_configs=data.get('risk_check_configs', {}),
            trading_configs=data.get('trading_configs', {}),
            monitoring_configs=data.get('monitoring_configs', {}),
            timeout_seconds=data.get('timeout_seconds', DEFAULT_TEST_TIMEOUT),
            retry_count=data.get('retry_count', 3),
            created_at=created_at,
            updated_at=updated_at,
            version=data.get('version', '1.0.0')
        )

    def _load_config_version(self, process_id: str, version: str) -> Optional[ProcessConfig]:
        """加载指定版本的配置"""
        try:
            config_file = self.config_dir / f"{process_id}_{version}.json"
            if not config_file.exists():
                config_file = self.config_dir / f"{process_id}.json"

            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                return self._deserialize_config(config_data)

        except Exception as e:
            logger.error(f"加载配置版本失败 {process_id}@{version}: {e}")

        return None

    def _validate_config(self, config: ProcessConfig) -> bool:
        """验证配置"""
        try:
            # 基本字段验证
            if not config.process_id or not config.process_name:
                return False

            # 数据源验证
            if config.data_sources:
                for source in config.data_sources:
                    if not isinstance(source, dict) or 'type' not in source:
                        return False

            # 模型配置验证
            if config.model_configs:
                for model_config in config.model_configs:
                    if not isinstance(model_config, dict):
                        return False

            # 超时配置验证
            if config.timeout_seconds <= 0:
                return False

            return True

        except Exception as e:
            logger.error(f"配置验证异常: {e}")
            return False

    def _validate_config_data(self, data: Dict[str, Any]) -> bool:
        """验证配置数据"""
        required_fields = ['process_id', 'process_name']
        return all(field in data for field in required_fields)

    def _increment_version(self, current_version: str) -> str:
        """递增版本号"""
        try:
            parts = current_version.split('.')
            if len(parts) >= 3:
                major, minor, patch = map(int, parts[:3])
                return f"{major}.{minor}.{patch + 1}"
            else:
                # 简单递增
                return str(float(current_version) + 0.1)
        except:
            # 如果版本格式不正确，返回默认新版本
            return "1.0.1"

    def get_config_hash(self, config: ProcessConfig) -> str:
        """获取配置哈希值"""
        config_str = json.dumps(self._serialize_config(config), sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()

    def compare_configs(self, config1: ProcessConfig, config2: ProcessConfig) -> Dict[str, Any]:
        """比较两个配置的差异"""
        diff = {
            'changed_fields': [],
            'added_fields': [],
            'removed_fields': [],
            'hash_changed': self.get_config_hash(config1) != self.get_config_hash(config2)
        }

        # 这里可以实现更详细的字段比较逻辑
        # 暂时返回基本信息
        return diff
