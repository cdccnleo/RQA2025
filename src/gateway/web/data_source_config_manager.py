#!/usr/bin/env python3
"""
数据源配置管理器

集成基础设施层配置管理模块，实现数据源配置的统一管理：
1. 使用基础设施层配置管理器进行配置加载和存储（适配器模式）
2. 支持环境隔离和配置热更新
3. 提供配置验证和监控
4. 支持多格式配置（JSON、YAML、TOML等）

架构设计说明：
- 采用适配器模式集成基础设施层 UnifiedConfigManager
- 通过适配器封装，为业务层提供统一的配置管理接口
- 支持多种配置后端（文件系统、PostgreSQL等）的透明切换
- 实现配置管理的统一抽象，符合依赖倒置原则
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from src.infrastructure.config.core.unified_manager_enhanced import UnifiedConfigManager
from src.infrastructure.logging.core.unified_logger import get_unified_logger
from src.infrastructure.utils.optimization.concurrency_controller import ConcurrencyController

logger = get_unified_logger(__name__)

# 全局并发控制器实例（单例模式）
_config_update_controller = None


def get_config_update_controller() -> ConcurrencyController:
    """
    获取配置更新并发控制器实例（单例模式）
    
    Returns:
        ConcurrencyController: 并发控制器实例
    """
    global _config_update_controller
    if _config_update_controller is None:
        _config_update_controller = ConcurrencyController()
    return _config_update_controller


class DataSourceConfigManager:
    """数据源配置管理器

    集成基础设施层配置管理，实现数据源配置的统一管理
    
    架构设计说明：
    - 采用适配器模式集成基础设施层 UnifiedConfigManager（适配器模式）
    - 通过适配器封装，为业务层提供统一的配置管理接口
    - 支持多种配置后端（文件系统、PostgreSQL等）的透明切换
    - 实现配置管理的统一抽象，符合依赖倒置原则
    - 使用统一日志系统 (get_unified_logger) 符合基础设施层规范
    """

    def __init__(self, config_dir: str = "data", auto_save: bool = True):
        """初始化数据源配置管理器

        Args:
            config_dir: 配置目录
            auto_save: 是否自动保存
        """
        # #region agent log
        import json as json_module
        import time
        try:
            import os

            debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"data_source_config_manager.py:__init__:43","message":"DataSourceConfigManager被实例化","data":{"config_dir":config_dir,"auto_save":auto_save},"timestamp":int(time.time()*1000)})+'\n')
        except: pass
        # #endregion
        
        self.config_dir = config_dir
        self.auto_save = auto_save

        # 获取环境信息
        self.env = os.getenv("RQA_ENV", "development").lower()
        
        # #region agent log
        try:
            import os

            debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"data_source_config_manager.py:__init__:54","message":"环境变量检查","data":{"RQA_ENV":os.getenv("RQA_ENV"),"env":self.env},"timestamp":int(time.time()*1000)})+'\n')
        except: pass
        # #endregion

        # 初始化基础设施层配置管理器
        self.config_manager = UnifiedConfigManager()

        # 初始化配置修改标志
        self._config_was_modified = False

        # 缓存
        self._cache = {}
        self._last_modified = {}

        # 初始化配置结构
        self._initialize_config_structure()

        # 加载配置
        self.load_config()

        logger.info(f"数据源配置管理器初始化完成，环境: {self.env}")

    def _initialize_config_structure(self):
        """初始化配置结构"""
        # 基础配置
        self.config_manager.set('data_sources.core.data_sources', [])
        self.config_manager.set('data_sources.metadata.last_updated', datetime.now().isoformat())
        self.config_manager.set('data_sources.metadata.version', '1.0.0')
        self.config_manager.set('data_sources.metadata.environment', self.env)

        # 环境特定配置
        if self.env == 'production':
            self.config_manager.set('data_sources.production.auto_backup', True)
            self.config_manager.set('data_sources.production.validation_strict', True)
        else:
            self.config_manager.set('data_sources.development.auto_backup', False)
            self.config_manager.set('data_sources.development.validation_strict', False)

    def _get_config_file_path(self, format_type: str = 'json') -> str:
        """获取配置文件路径（与config_manager.py保持一致）"""
        if self.env == "production":
            # 生产环境使用主配置文件，确保与config_manager.py一致
            config_file = f"{self.config_dir}/data_sources_config.{format_type}"
        elif self.env == "testing":
            config_file = f"{self.config_dir}/testing/data_sources_config.{format_type}"
        else:
            # 开发环境使用默认目录
            config_file = f"{self.config_dir}/data_sources_config.{format_type}"

        return config_file

    def load_config(self) -> bool:
        """加载配置（优先从PostgreSQL，然后主配置文件，最后环境特定文件）"""
        # #region agent log
        import json as json_module
        import time
        try:
            import os

            debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"data_source_config_manager.py:load_config:124","message":"开始load_config方法","data":{"env":self.env},"timestamp":int(time.time()*1000)})+'\n')
        except: pass
        # #endregion
        
        try:
            # 步骤1: 优先从PostgreSQL加载配置（与config_manager.py保持一致）
            logger.info("优先从PostgreSQL加载数据源配置")
            pg_config = self._load_from_postgresql()
            
            if pg_config:
                    # 从PostgreSQL加载成功，更新配置管理器
                    data_sources = pg_config.get('data_sources', [])
                    self.config_manager.set('data_sources.core.data_sources', data_sources)
                    if 'metadata' in pg_config:
                        metadata = pg_config['metadata']
                        self.config_manager.set('data_sources.metadata.last_updated', metadata.get('last_updated', datetime.now().isoformat()))
                        self.config_manager.set('data_sources.metadata.version', metadata.get('version', '1.0.0'))
                    logger.info(f"从PostgreSQL加载数据源配置成功，数据源数量: {len(data_sources)}")
                    
                    # 更新缓存，确保内存数据与PostgreSQL一致
                    self._cache_config_data(pg_config)
                    logger.info("从PostgreSQL加载配置后，更新了内存缓存")
                    
                    # 同步更新本地文件，确保文件与数据库一致
                    try:
                        config_file = self._get_config_file_path('json')
                        os.makedirs(os.path.dirname(config_file), exist_ok=True)
                        
                        # 检查现有文件格式
                        file_format_is_list = False
                        if os.path.exists(config_file):
                            try:
                                with open(config_file, 'r', encoding='utf-8') as f:
                                    existing_data = json.load(f)
                                    file_format_is_list = isinstance(existing_data, list)
                            except:
                                pass
                        
                        # 保存配置到本地文件
                        if file_format_is_list:
                            with open(config_file, 'w', encoding='utf-8') as f:
                                json.dump(data_sources, f, ensure_ascii=False, indent=2)
                        else:
                            with open(config_file, 'w', encoding='utf-8') as f:
                                json.dump(pg_config, f, ensure_ascii=False, indent=2)
                        logger.info(f"从PostgreSQL加载配置后，同步更新了本地文件: {config_file}")
                    except Exception as e:
                        logger.warning(f"同步更新本地文件失败: {e}")
                    
                    return True
            
            # 步骤2: 如果PostgreSQL加载失败，尝试从文件加载
            logger.info("PostgreSQL加载失败，尝试从文件加载配置")
            
            # 尝试加载JSON配置文件
            config_files_to_try = [
                f"{self.config_dir}/data_sources_config.json",  # 主配置文件
                self._get_config_file_path('json'),  # 环境特定配置文件
            ]

            config_loaded = False
            for config_file in config_files_to_try:
                if os.path.exists(config_file):
                    logger.info(f"尝试从 {config_file} 加载数据源配置")
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            raw_config_data = json.load(f)

                        # 处理不同格式的配置文件
                        if isinstance(raw_config_data, list):
                            # 数组格式：直接使用数据源列表
                            config_data = {"data_sources": raw_config_data}
                            logger.info(f"从 {config_file} 读取到数组格式配置，包含 {len(raw_config_data)} 个数据源")
                            # 记录enabled状态用于调试
                            for i, source in enumerate(raw_config_data[:3]):  # 只记录前3个
                                logger.debug(f"  数据源 {i}: id={source.get('id')}, enabled={source.get('enabled')}")
                            # 检查是否包含Baostock数据源
                            baostock_count = len([s for s in raw_config_data if 'baostock' in s.get('id', '').lower()])
                            logger.info(f"配置文件中包含 {baostock_count} 个Baostock数据源")
                        elif isinstance(raw_config_data, dict):
                            # 字典格式：检查是否已有data_sources键
                            if 'data_sources' not in raw_config_data:
                                raw_config_data['data_sources'] = []
                            config_data = raw_config_data
                            data_sources = config_data.get('data_sources', [])
                            logger.info(f"从 {config_file} 读取到字典格式配置，包含 {len(data_sources)} 个数据源")
                            # 记录enabled状态用于调试
                            for i, source in enumerate(data_sources[:3]):  # 只记录前3个
                                logger.debug(f"  数据源 {i}: id={source.get('id')}, enabled={source.get('enabled')}")
                            # 检查是否包含Baostock数据源
                            baostock_count = len([s for s in data_sources if 'baostock' in s.get('id', '').lower()])
                            logger.info(f"配置文件中包含 {baostock_count} 个Baostock数据源")
                        else:
                            logger.error(f"不支持的配置文件格式: {type(raw_config_data)}")
                            continue

                        # 验证并修复配置
                        if self._validate_and_fix_config(config_data):
                            # 如果配置被修复，保存修复后的配置回文件
                            if self._config_was_modified:
                                logger.info(f"配置已被修改，保存修复后的配置到 {config_file}")
                                try:
                                    with open(config_file, 'w', encoding='utf-8') as f:
                                        json.dump(config_data, f, ensure_ascii=False, indent=2)
                                    logger.info(f"修复后的配置已保存到 {config_file}")
                                except Exception as e:
                                    logger.error(f"保存修复后的配置失败: {e}")

                            # 加载到配置管理器
                            self._load_config_to_manager(config_data)
                            self._cache_config_data(config_data)
                            logger.info(f"成功从 {config_file} 加载配置，数据源数量: {len(config_data.get('data_sources', []))}")
                            config_loaded = True
                            break
                        else:
                            logger.warning(f"配置验证失败: {config_file}")
                    except Exception as e:
                        logger.warning(f"加载配置文件失败 {config_file}: {e}")
                else:
                    logger.debug(f"配置文件不存在: {config_file}")

            # 如果文件加载失败，使用默认配置
            if not config_loaded:
                # 如果PostgreSQL也不可用，使用默认配置
                logger.warning("未找到有效配置文件且PostgreSQL不可用，使用默认配置")
                default_config = self._get_default_config()
                self._load_config_to_manager(default_config)
                self._cache_config_data(default_config)
                # 检查默认配置中是否包含Baostock数据源
                baostock_count = len([s for s in default_config.get('data_sources', []) if 'baostock' in s.get('id', '').lower()])
                logger.info(f"默认配置中包含 {baostock_count} 个Baostock数据源")
            else:
                logger.info("配置文件加载成功")

            # 在非生产环境下保存默认配置
            if self.env != 'production':
                self.save_config()

            return True

        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return False

    def _validate_and_fix_config(self, config_data: Dict[str, Any]) -> bool:
        """验证配置数据并自动修复问题"""
        try:
            # 基础结构验证
            if not isinstance(config_data, dict):
                logger.error("配置数据必须是字典格式")
                return False

            if 'data_sources' not in config_data:
                logger.error("配置中缺少 data_sources 字段")
                return False

            data_sources = config_data['data_sources']
            if not isinstance(data_sources, list):
                logger.error("data_sources 必须是列表格式")
                return False

            # 数据源验证和修复
            valid_sources = []
            config_modified = False

            for i, source in enumerate(data_sources):
                if self._validate_data_source(source, i):
                    valid_sources.append(source)
                elif self._fix_data_source(source, i):
                    # 如果能够修复，添加到有效列表
                    valid_sources.append(source)
                    config_modified = True
                    logger.info(f"已修复数据源 {i}: {source.get('name')}")
                else:
                    logger.warning(f"跳过无效数据源 {i}: {source.get('name')}")

            # 更新配置中的数据源列表
            config_data['data_sources'] = valid_sources

            # 设置修改标志
            self._config_was_modified = config_modified
            if config_modified:
                logger.info(f"配置已被修改，修复了数据源ID")

            logger.info(f"配置验证完成，有效数据源数量: {len(valid_sources)}")

            return len(valid_sources) > 0

        except Exception as e:
            logger.error(f"配置验证异常: {e}")
            return False

    def _fix_data_source(self, source: Dict[str, Any], index: int) -> bool:
        """尝试修复单个数据源配置"""
        try:
            # 修复null或空的ID
            if not source.get('id') or str(source.get('id')).lower() == 'null':
                # 基于名称生成ID
                name = source.get('name', f'datasource_{index}')
                # 清理名称，移除特殊字符
                import re
                clean_name = re.sub(r'[^\w\s-]', '', name).strip().lower()
                source_id = clean_name.replace(' ', '_') + f'_{int(time.time())}'
                source['id'] = source_id
                logger.info(f"修复数据源ID: {name} -> {source_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"修复数据源 {index} 失败: {e}")
            return False

    def _validate_data_source(self, source: Dict[str, Any], index: int) -> bool:
        """验证单个数据源配置"""
        required_fields = ['id', 'name', 'type', 'url']

        for field in required_fields:
            if field not in source:
                logger.error(f"数据源 {index} 缺少必需字段: {field}")
                return False

        # 类型验证
        if not isinstance(source['id'], str) or not source['id'].strip():
            logger.error(f"数据源 {index} ID 必须是非空字符串")
            return False

        if not isinstance(source['name'], str) or not source['name'].strip():
            logger.error(f"数据源 {index} 名称必须是非空字符串")
            return False

        # 允许的数据源类型（包含实际使用的所有类型）
        valid_types = [
            '财经新闻', '交易接口', '宏观经济', '市场指数', '加密货币',
            '股票数据', '指数数据', '债券数据', '期货数据', '外汇数据', '财务报告'
        ]
        if source['type'] not in valid_types:
            logger.error(f"数据源 {index} 类型无效: {source['type']}")
            return False

        # URL格式验证
        if not isinstance(source['url'], str) or not source['url'].strip():
            logger.error(f"数据源 {index} URL 必须是非空字符串")
            return False

        return True

    def _load_config_to_manager(self, config_data: Dict[str, Any]):
        """将配置数据加载到配置管理器"""
        logger.info(f"加载配置数据到管理器，数据源数量: {len(config_data.get('data_sources', []))}")

        # 核心配置
        if 'data_sources' in config_data:
            data_sources = config_data['data_sources']
            logger.info(f"设置数据源配置: {len(data_sources)} 个数据源")
            for i, source in enumerate(data_sources):
                logger.debug(f"数据源 {i}: id={source.get('id')}, name={source.get('name')}")
            self.config_manager.set('data_sources.core.data_sources', data_sources)

        # 元数据
        if 'metadata' in config_data:
            metadata = config_data['metadata']
            for key, value in metadata.items():
                self.config_manager.set(f'data_sources.metadata.{key}', value)

        # 环境特定配置
        env_config = config_data.get(self.env, {})
        for key, value in env_config.items():
            self.config_manager.set(f'data_sources.{self.env}.{key}', value)

    def _cache_config_data(self, config_data: Dict[str, Any]):
        """缓存配置数据"""
        self._cache = config_data.copy()
        self._last_modified = datetime.now()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "metadata": {
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "environment": self.env
            },
            "data_sources": [
                {
                    "id": "sinafinance",
                    "name": "新浪财经",
                    "type": "财经新闻",
                    "url": "https://finance.sina.com.cn",
                    "rate_limit": "10次/分钟",
                    "enabled": True,
                    "last_test": None,
                    "status": "未测试",
                    "last_collection": None,
                    "collection_status": "未采集",
                    "collection_count": 0,
                    "collection_errors": 0,
                    "total_collections": 0,
                    "total_records": 0
                },
                {
                    "id": "miniqmt",
                    "name": "MiniQMT交易接口",
                    "type": "交易接口",
                    "url": "127.0.0.1:8888",
                    "rate_limit": "按协议",
                    "enabled": True,
                    "last_test": None,
                    "status": "未测试",
                    "last_collection": None,
                    "collection_status": "未采集",
                    "collection_count": 0,
                    "collection_errors": 0,
                    "total_collections": 0,
                    "total_records": 0
                },
                {
                    "id": "macrodata",
                    "name": "宏观经济数据",
                    "type": "宏观经济",
                    "url": "https://api.macrodata.com",
                    "rate_limit": "100次/分钟",
                    "enabled": True,
                    "last_test": None,
                    "status": "未测试",
                    "last_collection": None,
                    "collection_status": "未采集",
                    "collection_count": 0,
                    "collection_errors": 0,
                    "total_collections": 0,
                    "total_records": 0
                },
                {
                    "id": "cryptodata",
                    "name": "加密货币数据",
                    "type": "加密货币",
                    "url": "https://api.coingecko.com",
                    "rate_limit": "50次/分钟",
                    "enabled": True,
                    "last_test": None,
                    "status": "未测试",
                    "last_collection": None,
                    "collection_status": "未采集",
                    "collection_count": 0,
                    "collection_errors": 0,
                    "total_collections": 0,
                    "total_records": 0
                },
                {
                    "id": "akshare_stock_a",
                    "name": "AKShare A股数据",
                    "type": "股票数据",
                    "url": "https://www.akshare.xyz",
                    "rate_limit": "60次/分钟",
                    "enabled": True,
                    "last_test": None,
                    "status": "未测试",
                    "symbols": ["000001", "600519", "000858"],
                    "last_collection": None,
                    "collection_status": "未采集",
                    "collection_count": 0,
                    "collection_errors": 0,
                    "total_collections": 0,
                    "total_records": 0
                },
                {
                    "id": "baostock_stock_a",
                    "name": "BaoStock A股数据",
                    "type": "股票数据",
                    "url": "http://www.baostock.com",
                    "rate_limit": "30次/分钟",
                    "enabled": True,
                    "last_test": None,
                    "status": "未测试",
                    "symbols": ["000001", "600519", "000858"],
                    "last_collection": None,
                    "collection_status": "未采集",
                    "collection_count": 0,
                    "collection_errors": 0,
                    "total_collections": 0,
                    "total_records": 0
                }
            ],
            self.env: {
                "auto_backup": False,
                "validation_strict": False
            }
        }

    def save_config(self, format_type: str = 'json') -> bool:
        """保存配置（优先PostgreSQL，降级到文件系统）"""
        try:
            # 从配置管理器获取数据
            config_data = self._get_config_from_manager()
            data_sources = config_data.get('data_sources', [])
            
            # 优先尝试保存到PostgreSQL
            pg_success = False
            try:
                pg_success = self._save_to_postgresql(config_data)
                if pg_success:
                    logger.info(f"✅ 配置已保存到PostgreSQL, 数据源数量: {len(data_sources)}")
            except Exception as e:
                logger.warning(f"⚠️ 保存到PostgreSQL失败: {e}")
            
            # 如果PostgreSQL保存失败，降级到文件系统
            if not pg_success:
                logger.info("📝 降级到文件系统保存配置")
                
                config_file = self._get_config_file_path(format_type)
                
                # 确保目录存在
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                
                # 检查现有文件格式，保持一致
                file_format_is_list = False
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                            file_format_is_list = isinstance(existing_data, list)
                    except:
                        pass
                
                # 如果文件是列表格式，直接保存data_sources列表；否则保存完整配置
                if file_format_is_list:
                    with open(config_file, 'w', encoding='utf-8') as f:
                        json.dump(data_sources, f, ensure_ascii=False, indent=2)
                else:
                    with open(config_file, 'w', encoding='utf-8') as f:
                        json.dump(config_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"📝 配置已保存到文件系统: {config_file}, 数据源数量: {len(data_sources)}")
            
            # 记录前3个数据源的enabled状态
            for i, source in enumerate(data_sources[:3]):
                logger.debug(f"  保存的数据源 {i}: id={source.get('id')}, name={source.get('name')}, enabled={source.get('enabled')}")
            
            return True

        except Exception as e:
            logger.error(f"❌ 保存配置失败: {e}")
            return False
    
    def _save_to_postgresql(self, config_data: Dict[str, Any]) -> bool:
        """尝试保存配置到PostgreSQL（可选功能）"""
        try:
            from .postgresql_persistence import get_db_connection, return_db_connection
            
            conn = get_db_connection()
            if not conn:
                return False
            
            try:
                cursor = conn.cursor()
                
                # 创建表（如果不存在）
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_source_configs (
                        id SERIAL PRIMARY KEY,
                        config_key VARCHAR(255) UNIQUE NOT NULL,
                        config_data JSONB NOT NULL,
                        environment VARCHAR(50) NOT NULL,
                        version VARCHAR(50),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 保存配置
                config_key = f"data_sources_{self.env}"
                cursor.execute("""
                    INSERT INTO data_source_configs (config_key, config_data, environment, version, updated_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (config_key) 
                    DO UPDATE SET 
                        config_data = EXCLUDED.config_data,
                        environment = EXCLUDED.environment,
                        version = EXCLUDED.version,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    config_key,
                    json.dumps(config_data, ensure_ascii=False),
                    self.env,
                    config_data.get('metadata', {}).get('version', '1.0.0')
                ))
                
                conn.commit()
                logger.debug(f"数据源配置已保存到PostgreSQL: {config_key}")
                return True
                
            finally:
                return_db_connection(conn)
                
        except Exception as e:
            logger.debug(f"保存到PostgreSQL失败: {e}")
            return False
    
    def _load_from_postgresql(self) -> Optional[Dict[str, Any]]:
        """尝试从PostgreSQL加载配置（可选功能）"""
        # #region agent log
        import json as json_module
        import time
        try:
            import os

            debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
            with open(debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"data_source_config_manager.py:_load_from_postgresql:463","message":"开始从PostgreSQL加载配置","data":{"env":self.env},"timestamp":int(time.time()*1000)})+'\n')
        except: pass
        # #endregion
        
        try:
            from .postgresql_persistence import get_db_connection, return_db_connection
            
            # #region agent log
            try:
                import os
                debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
                with open(debug_log_path, 'a', encoding='utf-8') as f:
                    f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"data_source_config_manager.py:_load_from_postgresql:468","message":"准备获取数据库连接","data":{},"timestamp":int(time.time()*1000)})+'\n')
            except: pass
            # #endregion
            
            conn = get_db_connection()
            
            # #region agent log
            try:
                import os

                debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
                with open(debug_log_path, 'a', encoding='utf-8') as f:
                    f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"data_source_config_manager.py:_load_from_postgresql:470","message":"数据库连接获取结果","data":{"conn_is_none":conn is None},"timestamp":int(time.time()*1000)})+'\n')
            except: pass
            # #endregion
            
            if not conn:
                # #region agent log
                try:
                    import os

                    debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"data_source_config_manager.py:_load_from_postgresql:471","message":"数据库连接为None，返回None","data":{},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
                return None
            
            try:
                cursor = conn.cursor()
                config_key = f"data_sources_{self.env}"
                
                # #region agent log
                try:
                    import os

                    debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"data_source_config_manager.py:_load_from_postgresql:476","message":"准备查询PostgreSQL表","data":{"config_key":config_key},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
                
                cursor.execute("""
                    SELECT config_data, version, updated_at
                    FROM data_source_configs
                    WHERE config_key = %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (config_key,))
                
                row = cursor.fetchone()
                
                # #region agent log
                try:
                    import os

                    debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"data_source_config_manager.py:_load_from_postgresql:485","message":"PostgreSQL查询结果","data":{"row_is_none":row is None},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
                
                if row:
                    config_data = json.loads(row[0])
                    logger.debug(f"从PostgreSQL加载数据源配置: {config_key}")
                    # #region agent log
                    try:
                        import os

                        debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
                        with open(debug_log_path, 'a', encoding='utf-8') as f:
                            f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"data_source_config_manager.py:_load_from_postgresql:489","message":"成功从PostgreSQL加载配置","data":{"config_key":config_key,"has_data_sources":"data_sources" in config_data},"timestamp":int(time.time()*1000)})+'\n')
                    except: pass
                    # #endregion
                    return config_data
                
                # #region agent log
                try:
                    import os

                    debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
                    with open(debug_log_path, 'a', encoding='utf-8') as f:
                        f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"data_source_config_manager.py:_load_from_postgresql:490","message":"PostgreSQL中未找到配置数据","data":{"config_key":config_key},"timestamp":int(time.time()*1000)})+'\n')
                except: pass
                # #endregion
                return None
                
            finally:
                return_db_connection(conn)
                
        except Exception as e:
            # #region agent log
            try:
                import traceback
                import os

                debug_log_path = os.getenv('DEBUG_LOG_PATH', '/app/data/debug.log')
                with open(debug_log_path, 'a', encoding='utf-8') as f:
                    f.write(json_module.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"data_source_config_manager.py:_load_from_postgresql:495","message":"从PostgreSQL加载失败，异常信息","data":{"exception_type":type(e).__name__,"exception_msg":str(e),"traceback":traceback.format_exc()},"timestamp":int(time.time()*1000)})+'\n')
            except: pass
            # #endregion
            logger.debug(f"从PostgreSQL加载失败: {e}")
            return None

    def _get_config_from_manager(self) -> Dict[str, Any]:
        """从配置管理器获取配置数据"""
        config_data = {}

        # 核心配置
        data_sources = self.config_manager.get('data_sources.core.data_sources', default=[])
        config_data['data_sources'] = data_sources

        # 元数据
        config_data['metadata'] = {
            'version': self.config_manager.get('metadata.version', default='1.0.0'),
            'last_updated': datetime.now().isoformat(),
            'environment': self.env
        }

        # 环境特定配置
        env_config = {}
        try:
            # 获取环境相关的所有配置
            env_section = self.config_manager.get_section(self.env)
            if env_section:
                env_config.update(env_section)
        except:
            pass

        if env_config:
            config_data[self.env] = env_config

        return config_data

    def get_data_sources(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """获取所有数据源配置
        
        Args:
            force_reload: 是否强制重新加载配置（默认False，优先使用内存中的数据）
        """
        # 先检查内存中是否有数据
        sources = self.config_manager.get('data_sources.core.data_sources', default=[])
        
        # 只有在强制重新加载或内存中没有数据时才重新加载
        if force_reload or not sources or len(sources) == 0:
            logger.info("重新加载数据源配置（强制重新加载或内存中无数据）")
            self.load_config()
            sources = self.config_manager.get('data_sources.core.data_sources', default=[])
        else:
            logger.debug(f"使用内存中的数据源配置（{len(sources)} 个），跳过重新加载")

        logger.info(f"从配置管理器获取数据源: {len(sources)} 个")
        # 记录前5个数据源的enabled状态，用于调试
        for i, source in enumerate(sources[:5]):
            logger.debug(f"返回数据源 {i}: id={source.get('id')}, name={source.get('name')}, enabled={source.get('enabled')}")
        if len(sources) > 5:
            logger.debug(f"... 还有 {len(sources) - 5} 个数据源")

        # 检查是否包含Baostock数据源
        baostock_sources = [s for s in sources if 'baostock' in s.get('id', '').lower()]
        if not baostock_sources:
            logger.warning("未找到Baostock数据源，检查配置文件")
        else:
            logger.info(f"找到 {len(baostock_sources)} 个Baostock数据源")

        return sources

    def get_data_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        """获取指定数据源配置"""
        # 直接从配置管理器获取，避免触发load_config导致的ID修复
        data_sources = self.config_manager.get('data_sources.core.data_sources', default=[])
        for source in data_sources:
            if source.get('id') == source_id:
                return source
        return None

    def add_data_source(self, source_config: Dict[str, Any]) -> bool:
        """添加数据源配置"""
        try:
            # 验证配置
            if not self._validate_data_source(source_config, -1):
                return False

            # 检查ID是否已存在 - 直接从配置管理器获取，避免触发load_config
            data_sources = self.config_manager.get('data_sources.core.data_sources', default=[])
            for source in data_sources:
                if source.get('id') == source_config['id']:
                    logger.error(f"数据源ID已存在: {source_config['id']}")
                    return False

            # 添加数据源
            logger.info(f"添加前source_config ID: {source_config['id']}")
            data_sources.append(source_config)
            self.config_manager.set('data_sources.core.data_sources', data_sources)
            logger.info(f"设置后source_config ID: {source_config['id']}")

            # 自动保存
            if self.auto_save:
                logger.info(f"保存前source_config ID: {source_config['id']}")
                self.save_config()
                logger.info(f"保存后source_config ID: {source_config['id']}")

            logger.info(f"数据源已添加: {source_config['id']}")
            
            # 记录审计日志
            log_config_change('add', source_config['id'], {
                'name': source_config.get('name'),
                'type': source_config.get('type'),
                'enabled': source_config.get('enabled', True)
            })
            
            # 清空缓存
            clear_config_cache()
            
            return True

        except Exception as e:
            logger.error(f"添加数据源失败: {e}")
            return False

    def update_data_source(self, source_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新数据源配置（使用并发控制保护，符合基础设施层架构设计）
        
        Args:
            source_id: 数据源ID
            updates: 要更新的配置项
            
        Returns:
            bool: 是否更新成功
        """
        # 使用并发控制器保护配置更新操作（符合基础设施层架构设计：并发控制）
        lock_resource = f"config_update:{source_id}"
        concurrency_controller = get_config_update_controller()
        
        # 尝试获取锁（超时5秒，防止死锁）
        lock_acquired = concurrency_controller.acquire_lock(lock_resource, timeout=5.0)
        if not lock_acquired:
            logger.warning(f"获取配置更新锁失败: {source_id}，跳过更新")
            return False
        
        try:
            # 直接从配置管理器获取，避免触发load_config
            data_sources = self.config_manager.get('data_sources.core.data_sources', default=[])

            for i, source in enumerate(data_sources):
                if source.get('id') == source_id:
                    # 合并更新
                    updated_source = source.copy()
                    updated_source.update(updates)

                    # 验证更新后的配置
                    if not self._validate_data_source(updated_source, i):
                        return False

                    data_sources[i] = updated_source
                    self.config_manager.set('data_sources.core.data_sources', data_sources)

                    # 自动保存
                    if self.auto_save:
                        self.save_config()

                    logger.info(f"数据源已更新: {source_id}")
                    
                    # 记录审计日志
                    log_config_change('update', source_id, {
                        'updated_fields': list(updates.keys()),
                        'previous_enabled': source.get('enabled'),
                        'new_enabled': updated_source.get('enabled')
                    })
                    
                    # 清空缓存
                    clear_config_cache()
                    
                    return True

            logger.error(f"数据源不存在: {source_id}")
            return False

        except Exception as e:
            logger.error(f"更新数据源失败: {e}")
            return False
        finally:
            # 释放锁
            concurrency_controller.release_lock(lock_resource)

    def delete_data_source(self, source_id: str) -> bool:
        """删除数据源配置"""
        try:
            data_sources = self.get_data_sources()
            original_length = len(data_sources)

            # 过滤掉要删除的数据源
            data_sources = [s for s in data_sources if s.get('id') != source_id]

            if len(data_sources) == original_length:
                logger.error(f"数据源不存在: {source_id}")
                return False

            self.config_manager.set('data_sources.core.data_sources', data_sources)

            # 自动保存
            if self.auto_save:
                self.save_config()

            logger.info(f"数据源已删除: {source_id}")
            
            # 记录审计日志
            log_config_change('delete', source_id, {
                'deleted_at': datetime.now().isoformat()
            })
            
            # 清空缓存
            clear_config_cache()
            
            return True

        except Exception as e:
            logger.error(f"删除数据源失败: {e}")
            return False

    def reload_config(self) -> bool:
        """重新加载配置"""
        logger.info("重新加载数据源配置")
        return self.load_config()

    def get_config_stats(self) -> Dict[str, Any]:
        """获取配置统计信息"""
        data_sources = self.get_data_sources()

        stats = {
            'total_sources': len(data_sources),
            'enabled_sources': len([s for s in data_sources if s.get('enabled', True)]),
            'disabled_sources': len([s for s in data_sources if not s.get('enabled', True)]),
            'sources_by_type': {},
            'last_modified': self._last_modified.isoformat() if self._last_modified else None,
            'environment': self.env,
            'auto_save': self.auto_save
        }

        # 按类型统计
        for source in data_sources:
            source_type = source.get('type', 'unknown')
            stats['sources_by_type'][source_type] = stats['sources_by_type'].get(source_type, 0) + 1

        return stats

    def validate_all_sources(self) -> Dict[str, Any]:
        """验证所有数据源配置"""
        data_sources = self.get_data_sources()
        validation_results = {
            'valid': True,
            'total': len(data_sources),
            'valid_count': 0,
            'invalid_count': 0,
            'issues': []
        }

        for i, source in enumerate(data_sources):
            is_valid = self._validate_data_source(source, i)
            if is_valid:
                validation_results['valid_count'] += 1
            else:
                validation_results['valid'] = False
                validation_results['invalid_count'] += 1
                validation_results['issues'].append({
                    'index': i,
                    'source_id': source.get('id', 'unknown'),
                    'issues': ['配置验证失败']
                })

        return validation_results

    def backup_config(self, backup_path: Optional[str] = None) -> bool:
        """备份配置"""
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self.config_dir}/backups/data_sources_config_{timestamp}.json"

            os.makedirs(os.path.dirname(backup_path), exist_ok=True)

            config_data = self._get_config_from_manager()

            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)

            logger.info(f"配置已备份到: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"备份配置失败: {e}")
            return False

    def get_config_manager(self) -> UnifiedConfigManager:
        """获取配置管理器实例"""
        return self.config_manager

    def get_active_data_sources(self, force_reload: bool = False) -> List[Dict[str, Any]]:
        """获取所有活跃的数据源配置

        Args:
            force_reload: 是否强制重新加载配置

        Returns:
            List[Dict[str, Any]]: 活跃数据源列表
        """
        all_sources = self.get_data_sources(force_reload)
        active_sources = [source for source in all_sources if source.get('enabled', False)]

        logger.info(f"获取活跃数据源: {len(active_sources)}/{len(all_sources)} 个")
        return active_sources

    def export_config(self) -> Dict[str, Any]:
        """导出当前完整配置

        Returns:
            Dict[str, Any]: 完整的数据源配置
        """
        try:
            # 优先使用load_data_sources()函数获取最新的数据源配置
            try:
                from src.gateway.web.config_manager import load_data_sources
                data_sources = load_data_sources()
                config_data = {
                    "data_sources": data_sources,
                    "metadata": {
                        "last_updated": datetime.now().isoformat(),
                        "version": self.config_manager.get('data_sources.metadata.version', '1.0.0'),
                        "environment": self.env
                    }
                }
                logger.info(f"从load_data_sources()导出配置成功，数据源数量: {len(data_sources)}")
                return config_data
            except Exception as e:
                logger.warning(f"使用load_data_sources()失败: {e}")
                
                # 回退到原来的逻辑
                # 先尝试从PostgreSQL加载最新配置
                pg_config = self._load_from_postgresql()
                if pg_config:
                    # 如果从PostgreSQL加载成功，使用PostgreSQL的配置
                    config_data = pg_config
                    logger.info(f"从PostgreSQL导出配置成功，数据源数量: {len(config_data.get('data_sources', []))}")
                else:
                    # 否则使用当前配置管理器中的配置
                    config_data = {
                        "data_sources": self.config_manager.get('data_sources.core.data_sources', []),
                        "metadata": {
                            "last_updated": self.config_manager.get('data_sources.metadata.last_updated', datetime.now().isoformat()),
                            "version": self.config_manager.get('data_sources.metadata.version', '1.0.0'),
                            "environment": self.config_manager.get('data_sources.metadata.environment', self.env)
                        }
                    }
                    logger.info(f"从配置管理器导出配置成功，数据源数量: {len(config_data.get('data_sources', []))}")
                return config_data
        except Exception as e:
            logger.error(f"导出配置失败: {e}")
            raise

    def import_config(self, config_data: Dict[str, Any]) -> bool:
        """导入配置并更新系统

        Args:
            config_data: 要导入的配置数据

        Returns:
            bool: 导入是否成功
        """
        try:
            # 验证配置格式
            if not isinstance(config_data, dict):
                logger.error("导入配置格式错误，必须是字典")
                return False

            # 验证并修复配置
            if not self._validate_and_fix_config(config_data):
                logger.error("配置验证失败")
                return False

            # 加载配置到管理器
            self._load_config_to_manager(config_data)
            self._cache_config_data(config_data)

            # 保存配置
            self.save_config(config_data)

            logger.info(f"导入配置成功，数据源数量: {len(config_data.get('data_sources', []))}")
            return True
        except Exception as e:
            logger.error(f"导入配置失败: {e}")
            return False

    def get_active_symbols(self, force_reload: bool = False) -> List[str]:
        """获取所有活跃数据源中的股票代码

        Args:
            force_reload: 是否强制重新加载配置

        Returns:
            List[str]: 活跃股票代码列表（去重）
        """
        active_sources = self.get_active_data_sources(force_reload)
        symbols = set()

        for source in active_sources:
            # 提取股票代码 - 支持多种格式
            if 'symbol' in source and source['symbol']:
                symbols.add(source['symbol'])
            elif 'symbols' in source and source['symbols']:
                # 如果是symbols列表，添加所有
                if isinstance(source['symbols'], list):
                    symbols.update(source['symbols'])
                elif isinstance(source['symbols'], str):
                    # 如果是逗号分隔的字符串，分割后添加
                    symbol_list = [s.strip() for s in source['symbols'].split(',') if s.strip()]
                    symbols.update(symbol_list)
            elif 'custom_stocks' in source and source['custom_stocks']:
                # 如果是custom_stocks格式，提取code字段
                if isinstance(source['custom_stocks'], list):
                    for stock in source['custom_stocks']:
                        if isinstance(stock, dict) and 'code' in stock:
                            symbols.add(stock['code'])
                        elif isinstance(stock, str):
                            symbols.add(stock)
            elif 'config' in source and isinstance(source['config'], dict):
                # 检查config.custom_stocks
                config = source['config']
                if 'custom_stocks' in config and config['custom_stocks']:
                    if isinstance(config['custom_stocks'], list):
                        for stock in config['custom_stocks']:
                            if isinstance(stock, dict) and 'code' in stock:
                                symbols.add(stock['code'])
                            elif isinstance(stock, str):
                                symbols.add(stock)

        symbol_list = sorted(list(symbols))
        logger.info(f"从活跃数据源中提取到股票代码: {len(symbol_list)} 个")
        return symbol_list


# 全局实例
_data_source_config_manager = None

# 配置缓存（带TTL）
_config_cache = {}
_config_cache_ttl = 300  # 5分钟缓存
_config_cache_timestamp = 0

# 配置审计日志
_config_audit_log = []
_max_audit_log_size = 1000

def get_data_source_config_manager() -> DataSourceConfigManager:
    """获取数据源配置管理器实例"""
    global _data_source_config_manager
    if _data_source_config_manager is None:
        _data_source_config_manager = DataSourceConfigManager()
    return _data_source_config_manager


def get_cached_config(config_key: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
    """获取缓存的配置（带TTL）
    
    Args:
        config_key: 配置键
        force_refresh: 强制刷新缓存
        
    Returns:
        配置数据或None
    """
    global _config_cache, _config_cache_timestamp
    
    current_time = datetime.now().timestamp()
    
    # 检查缓存是否过期
    if force_refresh or current_time - _config_cache_timestamp > _config_cache_ttl:
        _config_cache.clear()
        _config_cache_timestamp = current_time
        logger.debug("配置缓存已过期，已清空")
        return None
    
    return _config_cache.get(config_key)


def set_cached_config(config_key: str, config_data: Dict[str, Any]):
    """设置缓存的配置
    
    Args:
        config_key: 配置键
        config_data: 配置数据
    """
    global _config_cache, _config_cache_timestamp
    
    _config_cache[config_key] = config_data
    _config_cache_timestamp = datetime.now().timestamp()
    logger.debug(f"配置已缓存: {config_key}")


def log_config_change(action: str, source_id: str, details: Dict[str, Any] = None):
    """记录配置变更审计日志
    
    Args:
        action: 操作类型（add/update/delete/enable/disable）
        source_id: 数据源ID
        details: 详细信息
    """
    global _config_audit_log
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'action': action,
        'source_id': source_id,
        'details': details or {},
        'user': 'system'  # 可以扩展为从请求上下文获取用户信息
    }
    
    _config_audit_log.append(log_entry)
    
    # 限制日志大小
    if len(_config_audit_log) > _max_audit_log_size:
        _config_audit_log = _config_audit_log[-_max_audit_log_size:]
    
    logger.info(f"配置变更记录: {action} - {source_id}")


def get_config_audit_log(limit: int = 100, source_id: str = None) -> List[Dict[str, Any]]:
    """获取配置变更审计日志
    
    Args:
        limit: 返回记录数限制
        source_id: 过滤特定数据源
        
    Returns:
        审计日志列表
    """
    global _config_audit_log
    
    logs = _config_audit_log
    
    if source_id:
        logs = [log for log in logs if log['source_id'] == source_id]
    
    return logs[-limit:]


def clear_config_cache():
    """清空配置缓存"""
    global _config_cache, _config_cache_timestamp
    
    _config_cache.clear()
    _config_cache_timestamp = 0
    logger.info("配置缓存已手动清空")


def get_config_cache_stats() -> Dict[str, Any]:
    """获取配置缓存统计信息
    
    Returns:
        缓存统计信息
    """
    global _config_cache, _config_cache_timestamp, _config_cache_ttl
    
    current_time = datetime.now().timestamp()
    cache_age = current_time - _config_cache_timestamp if _config_cache_timestamp > 0 else 0
    
    return {
        'cache_size': len(_config_cache),
        'cache_ttl': _config_cache_ttl,
        'cache_age_seconds': cache_age,
        'is_valid': cache_age < _config_cache_ttl,
        'audit_log_size': len(_config_audit_log)
    }
