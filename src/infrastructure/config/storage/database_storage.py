import json
from src.infrastructure.database import DatabaseManager
from src.infrastructure.database.connection_pool import ConnectionPool
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseStorage:
    """数据库存储实现"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.connection_pool = ConnectionPool()
    
    def save_config(self, key: str, config: Dict[str, Any]) -> bool:
        """保存配置到数据库"""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                # 检查是否已存在
                cursor.execute(
                    "SELECT id FROM configs WHERE config_key = %s",
                    (key,)
                )
                
                config_data = json.dumps(config)
                
                if cursor.fetchone():
                    # 更新现有配置
                    cursor.execute(
                        "UPDATE configs SET config_data = %s, updated_at = NOW() WHERE config_key = %s",
                        (config_data, key)
                    )
                else:
                    # 插入新配置
                    cursor.execute(
                        "INSERT INTO configs (config_key, config_data, created_at, updated_at) VALUES (%s, %s, NOW(), NOW())",
                        (key, config_data)
                    )
                
                conn.commit()
                logger.info(f"Saved config {key} to database")
                return True
                
        except Exception as e:
            logger.error(f"Error saving config {key} to database: {e}")
            return False
    
    def load_config(self, key: str) -> Optional[Dict[str, Any]]:
        """从数据库加载配置"""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT config_data FROM configs WHERE config_key = %s",
                    (key,)
                )
                
                result = cursor.fetchone()
                if result:
                    config_data = json.loads(result[0])
                    logger.info(f"Loaded config {key} from database")
                    return config_data
                else:
                    logger.warning(f"Config {key} not found in database")
                    return None
                    
        except Exception as e:
            logger.error(f"Error loading config {key} from database: {e}")
            return None
    
    def delete_config(self, key: str) -> bool:
        """从数据库删除配置"""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "DELETE FROM configs WHERE config_key = %s",
                    (key,)
                )
                
                conn.commit()
                logger.info(f"Deleted config {key} from database")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting config {key} from database: {e}")
            return False
    
    def list_configs(self) -> list:
        """列出所有配置键"""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT config_key FROM configs")
                
                results = cursor.fetchall()
                return [row[0] for row in results]
                
        except Exception as e:
            logger.error(f"Error listing configs from database: {e}")
            return []
    
    def exists(self, key: str) -> bool:
        """检查配置是否存在"""
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT 1 FROM configs WHERE config_key = %s",
                    (key,)
                )
                
                return cursor.fetchone() is not None
                
        except Exception as e:
            logger.error(f"Error checking config {key} existence: {e}")
            return False
