#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略持久化实现
Strategy Persistence Implementation

负责策略数据的存储、加载和管理
实现PostgreSQL优先存储策略：
- 主存储: PostgreSQL 数据库
- 降级存储: 文件系统 (JSON)

作者: RQA2025 Team
日期: 2026-03-22
"""

import json
import os
import time
import hashlib
import logging
import threading
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

from ..interfaces.strategy_interfaces import IStrategyPersistence

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetadata:
    """策略元数据"""
    strategy_id: str
    strategy_name: str
    strategy_type: str
    version: str
    status: str
    created_at: str
    updated_at: str
    description: str = ""
    author: str = ""
    tags: List[str] = None
    config: Dict[str, Any] = None
    storage_type: str = "filesystem"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.config is None:
            self.config = {}


class StrategyPersistence(IStrategyPersistence):
    """
    策略持久化实现
    
    实现PostgreSQL优先存储，数据库连接失败时自动降级到文件系统。
    """
    
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 1.0
    
    def __init__(self, storage_path: str = None, enable_postgresql: bool = True):
        """
        初始化持久化管理器
        
        Args:
            storage_path: 文件系统存储路径
            enable_postgresql: 是否启用PostgreSQL存储
        """
        if storage_path is None:
            self.storage_path = Path.home() / ".rqa2025" / "strategies"
        else:
            self.storage_path = Path(storage_path)
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._strategies_file = self.storage_path / "strategies.json"
        self._configs_file = self.storage_path / "configs.json"
        
        self._strategy_cache: Dict[str, Dict[str, Any]] = {}
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        
        self._lock = threading.Lock()
        
        self._pg_config = None
        self._pg_available = False
        
        if enable_postgresql:
            self._pg_config = self._get_postgresql_config()
            self._pg_available = self._test_postgresql_connection()
            if self._pg_available:
                self._ensure_tables_exist()
                logger.info("✅ StrategyPersistence: PostgreSQL 存储已启用")
            else:
                logger.warning("⚠️ StrategyPersistence: PostgreSQL 不可用，使用文件系统存储")
        
        self._load_data()
        
        self.stats = {
            'total_saves': 0,
            'total_loads': 0,
            'postgresql_saves': 0,
            'filesystem_saves': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

    def _get_postgresql_config(self) -> Optional[Dict[str, str]]:
        """
        获取PostgreSQL配置（使用统一配置模块）
        
        优先使用统一数据库配置模块，确保配置一致性。
        密码必须从环境变量读取，禁止硬编码。
        
        Returns:
            数据库配置字典，获取失败返回None
        """
        try:
            # 优先使用统一数据库配置模块
            from src.infrastructure.persistence.database_config import get_db_config
            config = get_db_config()
            return config.to_dict()
        except ImportError:
            logger.debug("统一数据库配置模块导入失败，尝试从环境变量获取")
            
            # 密码必须从环境变量读取，禁止硬编码
            password = os.getenv("POSTGRES_PASSWORD")
            if not password:
                logger.error(
                    "数据库密码未设置！请设置环境变量 POSTGRES_PASSWORD。\n"
                    "示例：set POSTGRES_PASSWORD=YourSecurePassword\n"
                    "或：export POSTGRES_PASSWORD=YourSecurePassword"
                )
                return None
            
            return {
                "host": os.getenv("POSTGRES_HOST", "postgres"),
                "port": os.getenv("POSTGRES_PORT", "5432"),
                "database": os.getenv("POSTGRES_DB", "rqa2025_prod"),
                "user": os.getenv("POSTGRES_USER", "rqa2025_admin"),
                "password": password
            }
        except Exception as e:
            logger.error(f"获取PostgreSQL配置失败: {e}")
            return None

    def _get_db_connection(self):
        """获取数据库连接（带重试机制）"""
        if not self._pg_config:
            return None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                import psycopg2
                conn = psycopg2.connect(
                    host=self._pg_config["host"],
                    port=self._pg_config["port"],
                    database=self._pg_config["database"],
                    user=self._pg_config["user"],
                    password=self._pg_config["password"],
                    connect_timeout=5
                )
                return conn
            except Exception as e:
                logger.debug(f"PostgreSQL连接失败 (尝试 {attempt + 1}/{self.MAX_RETRIES}): {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY_BASE * (2 ** attempt))
        
        return None

    def _test_postgresql_connection(self) -> bool:
        """测试PostgreSQL连接是否可用"""
        conn = self._get_db_connection()
        if conn:
            conn.close()
            return True
        return False

    def _ensure_tables_exist(self):
        """确保数据库表存在"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return
            
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS strategies (
                        id SERIAL PRIMARY KEY,
                        strategy_id VARCHAR(128) NOT NULL UNIQUE,
                        strategy_name VARCHAR(255) NOT NULL,
                        strategy_type VARCHAR(64) NOT NULL,
                        version VARCHAR(32) DEFAULT '1.0.0',
                        status VARCHAR(32) DEFAULT 'draft',
                        description TEXT,
                        author VARCHAR(128),
                        tags JSONB DEFAULT '[]',
                        config JSONB DEFAULT '{}',
                        strategy_data JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_strategies_id 
                    ON strategies(strategy_id)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_strategies_type 
                    ON strategies(strategy_type)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_strategies_status 
                    ON strategies(status)
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_configs (
                        id SERIAL PRIMARY KEY,
                        strategy_id VARCHAR(128) NOT NULL,
                        config_name VARCHAR(128) NOT NULL,
                        config_data JSONB DEFAULT '{}',
                        version VARCHAR(32) DEFAULT '1.0.0',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE,
                        UNIQUE(strategy_id, config_name)
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_strategy_configs_strategy 
                    ON strategy_configs(strategy_id)
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS strategy_lifecycle (
                        id SERIAL PRIMARY KEY,
                        strategy_id VARCHAR(128) NOT NULL,
                        current_stage VARCHAR(64) NOT NULL,
                        stage_history JSONB DEFAULT '[]',
                        next_allowed_actions JSONB DEFAULT '[]',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB DEFAULT '{}'
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_strategy_lifecycle_strategy 
                    ON strategy_lifecycle(strategy_id)
                """)
            
            conn.commit()
            logger.debug("策略存储表已确保存在")
            
        except Exception as e:
            logger.error(f"创建策略存储表失败: {e}")
        finally:
            if conn:
                conn.close()

    def _load_data(self):
        """加载现有数据"""
        try:
            if self._strategies_file.exists():
                with open(self._strategies_file, 'r', encoding='utf-8') as f:
                    self._strategy_cache = json.load(f)
        except Exception:
            self._strategy_cache = {}

        try:
            if self._configs_file.exists():
                with open(self._configs_file, 'r', encoding='utf-8') as f:
                    self._config_cache = json.load(f)
        except Exception:
            self._config_cache = {}
        
        if self._pg_available:
            self._load_from_postgresql()

    def _load_from_postgresql(self):
        """从PostgreSQL加载策略数据"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT strategy_id, strategy_name, strategy_type, version,
                           status, description, author, tags, config, strategy_data,
                           created_at, updated_at
                    FROM strategies
                    WHERE is_active = TRUE
                """)
                
                for row in cur.fetchall():
                    strategy_id = row[0]
                    self._strategy_cache[strategy_id] = {
                        'strategy_id': strategy_id,
                        'strategy_name': row[1],
                        'strategy_type': row[2],
                        'version': row[3],
                        'status': row[4],
                        'description': row[5] or '',
                        'author': row[6] or '',
                        'tags': row[7] or [],
                        'config': row[8] or {},
                        'strategy_data': row[9] or {},
                        'created_at': str(row[10]) if row[10] else None,
                        'updated_at': str(row[11]) if row[11] else None,
                        'storage_type': 'postgresql'
                    }
            
            logger.info(f"从PostgreSQL加载了 {len(self._strategy_cache)} 个策略")
            
        except Exception as e:
            logger.error(f"从PostgreSQL加载策略失败: {e}")
        finally:
            if conn:
                conn.close()

    def save_strategy(self, strategy_id: str, strategy_data: Dict[str, Any]) -> bool:
        """
        保存策略（PostgreSQL优先，降级到文件系统）
        
        Args:
            strategy_id: 策略ID
            strategy_data: 策略数据
            
        Returns:
            保存是否成功
        """
        with self._lock:
            try:
                strategy_data['_saved_at'] = datetime.now().isoformat()
                strategy_data['strategy_id'] = strategy_id
                
                storage_type = "memory"
                
                if self._pg_available:
                    if self._save_to_postgresql(strategy_id, strategy_data):
                        storage_type = "postgresql"
                        self.stats['postgresql_saves'] += 1
                    else:
                        logger.warning("PostgreSQL保存失败，降级到文件系统")
                
                if storage_type == "memory":
                    self._save_to_filesystem(strategy_id, strategy_data)
                    storage_type = "filesystem"
                    self.stats['filesystem_saves'] += 1
                
                strategy_data['storage_type'] = storage_type
                self._strategy_cache[strategy_id] = strategy_data
                
                self.stats['total_saves'] += 1
                logger.info(f"✅ 策略已保存: {strategy_id} ({storage_type})")
                
                return True
                
            except Exception as e:
                logger.error(f"保存策略失败: {e}")
                return False

    def _save_to_postgresql(self, strategy_id: str, strategy_data: Dict[str, Any]) -> bool:
        """保存策略到PostgreSQL"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return False
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO strategies (
                        strategy_id, strategy_name, strategy_type, version,
                        status, description, author, tags, config, strategy_data,
                        created_at, updated_at, is_active
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (strategy_id) DO UPDATE SET
                        strategy_name = EXCLUDED.strategy_name,
                        strategy_type = EXCLUDED.strategy_type,
                        version = EXCLUDED.version,
                        status = EXCLUDED.status,
                        description = EXCLUDED.description,
                        author = EXCLUDED.author,
                        tags = EXCLUDED.tags,
                        config = EXCLUDED.config,
                        strategy_data = EXCLUDED.strategy_data,
                        updated_at = CURRENT_TIMESTAMP,
                        is_active = TRUE
                """, (
                    strategy_id,
                    strategy_data.get('strategy_name', strategy_id),
                    strategy_data.get('strategy_type', 'unknown'),
                    strategy_data.get('version', '1.0.0'),
                    strategy_data.get('status', 'active'),
                    strategy_data.get('description', ''),
                    strategy_data.get('author', ''),
                    json.dumps(strategy_data.get('tags', [])),
                    json.dumps(strategy_data.get('config', {})),
                    json.dumps(strategy_data),
                    datetime.now(),
                    datetime.now(),
                    True
                ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"保存到PostgreSQL失败: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def _save_to_filesystem(self, strategy_id: str, strategy_data: Dict[str, Any]):
        """保存策略到文件系统"""
        self._strategy_cache[strategy_id] = strategy_data
        self._save_data()

    def _save_data(self):
        """保存数据到文件"""
        try:
            with open(self._strategies_file, 'w', encoding='utf-8') as f:
                json.dump(self._strategy_cache, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"保存策略文件失败: {e}")

        try:
            with open(self._configs_file, 'w', encoding='utf-8') as f:
                json.dump(self._config_cache, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")

    def load_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        加载策略（优先从缓存，然后PostgreSQL，最后文件系统）
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            策略数据，未找到返回None
        """
        self.stats['total_loads'] += 1
        
        if strategy_id in self._strategy_cache:
            self.stats['cache_hits'] += 1
            return self._strategy_cache[strategy_id]
        
        self.stats['cache_misses'] += 1
        
        if self._pg_available:
            result = self._load_from_postgresql_by_id(strategy_id)
            if result:
                self._strategy_cache[strategy_id] = result
                return result
        
        result = self._strategy_cache.get(strategy_id)
        if result:
            self._strategy_cache[strategy_id] = result
        
        return result

    def _load_from_postgresql_by_id(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """从PostgreSQL加载指定策略"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return None
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT strategy_id, strategy_name, strategy_type, version,
                           status, description, author, tags, config, strategy_data,
                           created_at, updated_at
                    FROM strategies
                    WHERE strategy_id = %s AND is_active = TRUE
                """, (strategy_id,))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                return {
                    'strategy_id': row[0],
                    'strategy_name': row[1],
                    'strategy_type': row[2],
                    'version': row[3],
                    'status': row[4],
                    'description': row[5] or '',
                    'author': row[6] or '',
                    'tags': row[7] or [],
                    'config': row[8] or {},
                    'strategy_data': row[9] or {},
                    'created_at': str(row[10]) if row[10] else None,
                    'updated_at': str(row[11]) if row[11] else None,
                    'storage_type': 'postgresql'
                }
            
        except Exception as e:
            logger.error(f"从PostgreSQL加载策略失败: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def delete_strategy(self, strategy_id: str) -> bool:
        """
        删除策略
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            删除是否成功
        """
        with self._lock:
            try:
                deleted = False
                
                if self._pg_available:
                    deleted = self._delete_from_postgresql(strategy_id) or deleted
                
                if strategy_id in self._strategy_cache:
                    del self._strategy_cache[strategy_id]
                    deleted = True
                
                if strategy_id in self._config_cache:
                    del self._config_cache[strategy_id]
                
                self._save_data()
                
                if deleted:
                    logger.info(f"策略已删除: {strategy_id}")
                
                return deleted
                
            except Exception as e:
                logger.error(f"删除策略失败: {e}")
                return False

    def _delete_from_postgresql(self, strategy_id: str) -> bool:
        """从PostgreSQL删除策略"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return False
            
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE strategies SET is_active = FALSE WHERE strategy_id = %s",
                    (strategy_id,)
                )
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"从PostgreSQL删除策略失败: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def list_strategies(self) -> List[str]:
        """
        列出所有策略
        
        Returns:
            策略ID列表
        """
        return list(self._strategy_cache.keys())

    def list_strategies_by_type(self, strategy_type: str) -> List[str]:
        """
        按类型列出策略
        
        Args:
            strategy_type: 策略类型
            
        Returns:
            策略ID列表
        """
        return [
            sid for sid, data in self._strategy_cache.items()
            if data.get('strategy_type') == strategy_type
        ]

    def list_strategies_by_status(self, status: str) -> List[str]:
        """
        按状态列出策略
        
        Args:
            status: 策略状态
            
        Returns:
            策略ID列表
        """
        return [
            sid for sid, data in self._strategy_cache.items()
            if data.get('status') == status
        ]

    def save_strategy_config(self, strategy_id: str, config: Dict[str, Any]) -> bool:
        """
        保存策略配置
        
        Args:
            strategy_id: 策略ID
            config: 配置数据
            
        Returns:
            保存是否成功
        """
        with self._lock:
            try:
                config['_saved_at'] = datetime.now().isoformat()
                
                if self._pg_available:
                    self._save_config_to_postgresql(strategy_id, config)
                
                self._config_cache[strategy_id] = config
                self._save_data()
                
                return True
                
            except Exception as e:
                logger.error(f"保存策略配置失败: {e}")
                return False

    def _save_config_to_postgresql(self, strategy_id: str, config: Dict[str, Any]) -> bool:
        """保存配置到PostgreSQL"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return False
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO strategy_configs (strategy_id, config_name, config_data, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (strategy_id, config_name) DO UPDATE SET
                        config_data = EXCLUDED.config_data,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    strategy_id,
                    'default',
                    json.dumps(config),
                    datetime.now()
                ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"保存配置到PostgreSQL失败: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def load_strategy_config(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        加载策略配置
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            配置数据
        """
        if strategy_id in self._config_cache:
            return self._config_cache[strategy_id]
        
        if self._pg_available:
            config = self._load_config_from_postgresql(strategy_id)
            if config:
                self._config_cache[strategy_id] = config
                return config
        
        return self._config_cache.get(strategy_id)

    def _load_config_from_postgresql(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """从PostgreSQL加载配置"""
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return None
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT config_data FROM strategy_configs
                    WHERE strategy_id = %s AND config_name = 'default' AND is_active = TRUE
                """, (strategy_id,))
                
                row = cur.fetchone()
                if row:
                    return row[0]
            
            return None
            
        except Exception as e:
            logger.error(f"从PostgreSQL加载配置失败: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def save_lifecycle_state(self, strategy_id: str, lifecycle_data: Dict[str, Any]) -> bool:
        """
        保存策略生命周期状态
        
        Args:
            strategy_id: 策略ID
            lifecycle_data: 生命周期数据
            
        Returns:
            保存是否成功
        """
        if not self._pg_available:
            return False
        
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return False
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO strategy_lifecycle (
                        strategy_id, current_stage, stage_history, 
                        next_allowed_actions, updated_at, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (strategy_id) DO UPDATE SET
                        current_stage = EXCLUDED.current_stage,
                        stage_history = EXCLUDED.stage_history,
                        next_allowed_actions = EXCLUDED.next_allowed_actions,
                        updated_at = CURRENT_TIMESTAMP,
                        metadata = EXCLUDED.metadata
                """, (
                    strategy_id,
                    lifecycle_data.get('current_stage', 'created'),
                    json.dumps(lifecycle_data.get('stage_history', [])),
                    json.dumps(lifecycle_data.get('next_allowed_actions', [])),
                    datetime.now(),
                    json.dumps(lifecycle_data.get('metadata', {}))
                ))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"保存生命周期状态失败: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                conn.close()

    def load_lifecycle_state(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """
        加载策略生命周期状态
        
        Args:
            strategy_id: 策略ID
            
        Returns:
            生命周期数据
        """
        if not self._pg_available:
            return None
        
        conn = None
        try:
            conn = self._get_db_connection()
            if not conn:
                return None
            
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT current_stage, stage_history, next_allowed_actions, metadata
                    FROM strategy_lifecycle
                    WHERE strategy_id = %s
                """, (strategy_id,))
                
                row = cur.fetchone()
                if row:
                    return {
                        'current_stage': row[0],
                        'stage_history': row[1] or [],
                        'next_allowed_actions': row[2] or [],
                        'metadata': row[3] or {}
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"加载生命周期状态失败: {e}")
            return None
        finally:
            if conn:
                conn.close()

    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        stats = self.stats.copy()
        stats['postgresql_available'] = self._pg_available
        stats['cache_size'] = len(self._strategy_cache)
        stats['config_cache_size'] = len(self._config_cache)
        
        total_requests = stats['cache_hits'] + stats['cache_misses']
        stats['cache_hit_rate'] = stats['cache_hits'] / total_requests if total_requests > 0 else 0.0
        
        if self._pg_available:
            conn = None
            try:
                conn = self._get_db_connection()
                if conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*) FROM strategies WHERE is_active = TRUE")
                        stats['postgresql_strategies'] = cur.fetchone()[0]
            except Exception:
                pass
            finally:
                if conn:
                    conn.close()
        
        return stats

    def sync_to_postgresql(self) -> int:
        """
        将文件系统中的数据同步到PostgreSQL
        
        Returns:
            同步的策略数量
        """
        if not self._pg_available:
            logger.warning("PostgreSQL不可用，无法同步")
            return 0
        
        synced = 0
        for strategy_id, strategy_data in self._strategy_cache.items():
            try:
                if self._save_to_postgresql(strategy_id, strategy_data):
                    synced += 1
            except Exception as e:
                logger.error(f"同步策略失败 {strategy_id}: {e}")
        
        logger.info(f"已同步 {synced} 个策略到PostgreSQL")
        return synced


_strategy_persistence = None


def get_strategy_persistence(**kwargs) -> StrategyPersistence:
    """
    获取策略持久化实例
    
    Args:
        **kwargs: StrategyPersistence初始化参数
        
    Returns:
        StrategyPersistence实例
    """
    global _strategy_persistence
    if _strategy_persistence is None:
        _strategy_persistence = StrategyPersistence(**kwargs)
    return _strategy_persistence
