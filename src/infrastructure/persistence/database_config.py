#!/usr/bin/env python3
"""
统一数据库配置模块

提供统一的数据库连接配置，避免各个模块各自维护配置。
所有数据库连接都应使用此模块获取配置。
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """数据库配置数据类"""
    host: str
    port: str
    database: str
    user: str
    password: str
    
    @property
    def connection_string(self) -> str:
        """生成PostgreSQL连接字符串"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    def to_dict(self) -> Dict[str, str]:
        """转换为字典格式"""
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password
        }


class DatabaseConfigManager:
    """数据库配置管理器
    
    统一管理所有数据库连接配置，确保配置一致性。
    
    使用方式:
        # 获取默认配置
        config = DatabaseConfigManager.get_config()
        
        # 使用配置连接数据库
        conn = psycopg2.connect(**config.to_dict())
    """
    
    # 默认配置值（从环境变量读取，如果不存在则使用默认值）
    DEFAULT_HOST = "postgres"
    DEFAULT_PORT = "5432"
    DEFAULT_DATABASE = "rqa2025_prod"
    DEFAULT_USER = "rqa2025_admin"
    DEFAULT_PASSWORD = "rqa2025_secure_password"
    
    _instance: Optional['DatabaseConfigManager'] = None
    _config: Optional[DatabaseConfig] = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """从环境变量加载配置"""
        self._config = DatabaseConfig(
            host=os.getenv("POSTGRES_HOST", self.DEFAULT_HOST),
            port=os.getenv("POSTGRES_PORT", self.DEFAULT_PORT),
            database=os.getenv("POSTGRES_DB", self.DEFAULT_DATABASE),
            user=os.getenv("POSTGRES_USER", self.DEFAULT_USER),
            password=os.getenv("POSTGRES_PASSWORD", self.DEFAULT_PASSWORD)
        )
    
    @classmethod
    def get_config(cls) -> DatabaseConfig:
        """获取数据库配置
        
        Returns:
            DatabaseConfig: 数据库配置对象
        """
        instance = cls()
        return instance._config
    
    @classmethod
    def get_connection_string(cls) -> str:
        """获取数据库连接字符串
        
        Returns:
            str: PostgreSQL连接字符串
        """
        return cls.get_config().connection_string
    
    @classmethod
    def reload_config(cls):
        """重新加载配置（用于配置变更后）"""
        instance = cls()
        instance._load_config()
    
    @classmethod
    def test_connection(cls) -> bool:
        """测试数据库连接
        
        Returns:
            bool: 连接是否成功
        """
        try:
            import psycopg2
            config = cls.get_config()
            conn = psycopg2.connect(**config.to_dict())
            conn.close()
            return True
        except Exception:
            return False


# 便捷函数，直接获取配置
def get_db_config() -> DatabaseConfig:
    """获取数据库配置
    
    Returns:
        DatabaseConfig: 数据库配置对象
    """
    return DatabaseConfigManager.get_config()


def get_db_connection_string() -> str:
    """获取数据库连接字符串
    
    Returns:
        str: PostgreSQL连接字符串
    """
    return DatabaseConfigManager.get_connection_string()


def test_db_connection() -> bool:
    """测试数据库连接
    
    Returns:
        bool: 连接是否成功
    """
    return DatabaseConfigManager.test_connection()
