"""
特征质量自定义评分配置管理
提供用户自定义评分配置的CRUD操作
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from src.gateway.web.postgresql_persistence import get_db_connection, return_db_connection

logger = logging.getLogger(__name__)


@dataclass
class UserQualityConfig:
    """用户自定义评分配置"""
    config_id: int
    user_id: str
    feature_name: str
    custom_score: float
    reason: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime


class FeatureQualityConfigManager:
    """特征质量配置管理器"""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, UserQualityConfig]] = {}
        self._cache_enabled = True
    
    def get_user_configs(
        self,
        user_id: str,
        include_inactive: bool = False
    ) -> List[UserQualityConfig]:
        """
        获取用户的自定义评分配置
        
        Args:
            user_id: 用户ID
            include_inactive: 是否包含已禁用的配置
        
        Returns:
            配置列表
        """
        # 检查缓存
        cache_key = f"{user_id}_{include_inactive}"
        if self._cache_enabled and cache_key in self._cache:
            return list(self._cache[cache_key].values())
        
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            query = """
                SELECT config_id, user_id, feature_name, custom_score,
                       reason, is_active, created_at, updated_at
                FROM user_feature_quality_config
                WHERE user_id = %s
            """
            
            if not include_inactive:
                query += " AND is_active = TRUE"
            
            query += " ORDER BY updated_at DESC"
            
            cursor.execute(query, (user_id,))
            
            configs = []
            cache_dict = {}
            
            for row in cursor.fetchall():
                config = UserQualityConfig(
                    config_id=row[0],
                    user_id=row[1],
                    feature_name=row[2],
                    custom_score=row[3],
                    reason=row[4],
                    is_active=row[5],
                    created_at=row[6],
                    updated_at=row[7]
                )
                configs.append(config)
                cache_dict[config.feature_name] = config
            
            # 更新缓存
            if self._cache_enabled:
                self._cache[cache_key] = cache_dict
            
            cursor.close()
            
            logger.info(f"获取到用户 {user_id} 的 {len(configs)} 个自定义评分配置")
            return configs
            
        except Exception as e:
            logger.error(f"获取用户配置失败: {e}")
            return []
        finally:
            if conn:
                return_db_connection(conn)
    
    def get_config_by_feature(
        self,
        user_id: str,
        feature_name: str
    ) -> Optional[UserQualityConfig]:
        """
        获取特定特征的自定义评分配置
        
        Args:
            user_id: 用户ID
            feature_name: 特征名称
        
        Returns:
            配置对象，如果不存在则返回None
        """
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT config_id, user_id, feature_name, custom_score,
                       reason, is_active, created_at, updated_at
                FROM user_feature_quality_config
                WHERE user_id = %s AND feature_name = %s AND is_active = TRUE
            """, (user_id, feature_name))
            
            row = cursor.fetchone()
            cursor.close()
            
            if row:
                return UserQualityConfig(
                    config_id=row[0],
                    user_id=row[1],
                    feature_name=row[2],
                    custom_score=row[3],
                    reason=row[4],
                    is_active=row[5],
                    created_at=row[6],
                    updated_at=row[7]
                )
            
            return None
            
        except Exception as e:
            logger.error(f"获取配置失败: {e}")
            return None
        finally:
            if conn:
                return_db_connection(conn)
    
    def create_config(
        self,
        user_id: str,
        feature_name: str,
        custom_score: float,
        reason: Optional[str] = None
    ) -> Optional[UserQualityConfig]:
        """
        创建自定义评分配置
        
        Args:
            user_id: 用户ID
            feature_name: 特征名称
            custom_score: 自定义评分 (0-1)
            reason: 修改原因
        
        Returns:
            创建的配置对象
        """
        # 验证评分范围
        if not (0.0 <= custom_score <= 1.0):
            logger.error(f"评分 {custom_score} 超出范围 [0, 1]")
            return None
        
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # 检查是否已存在
            cursor.execute("""
                SELECT config_id FROM user_feature_quality_config
                WHERE user_id = %s AND feature_name = %s
            """, (user_id, feature_name))
            
            existing = cursor.fetchone()
            
            if existing:
                # 更新现有配置
                cursor.execute("""
                    UPDATE user_feature_quality_config
                    SET custom_score = %s,
                        reason = %s,
                        is_active = TRUE,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE config_id = %s
                    RETURNING config_id, user_id, feature_name, custom_score,
                              reason, is_active, created_at, updated_at
                """, (custom_score, reason, existing[0]))
                
                logger.info(f"更新用户 {user_id} 的特征 {feature_name} 自定义评分: {custom_score}")
            else:
                # 创建新配置
                cursor.execute("""
                    INSERT INTO user_feature_quality_config
                    (user_id, feature_name, custom_score, reason)
                    VALUES (%s, %s, %s, %s)
                    RETURNING config_id, user_id, feature_name, custom_score,
                              reason, is_active, created_at, updated_at
                """, (user_id, feature_name, custom_score, reason))
                
                logger.info(f"创建用户 {user_id} 的特征 {feature_name} 自定义评分: {custom_score}")
            
            row = cursor.fetchone()
            conn.commit()
            
            # 清除缓存
            self._clear_user_cache(user_id)
            
            config = UserQualityConfig(
                config_id=row[0],
                user_id=row[1],
                feature_name=row[2],
                custom_score=row[3],
                reason=row[4],
                is_active=row[5],
                created_at=row[6],
                updated_at=row[7]
            )
            
            cursor.close()
            return config
            
        except Exception as e:
            logger.error(f"创建配置失败: {e}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                return_db_connection(conn)
    
    def update_config(
        self,
        config_id: int,
        custom_score: Optional[float] = None,
        reason: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> Optional[UserQualityConfig]:
        """
        更新自定义评分配置
        
        Args:
            config_id: 配置ID
            custom_score: 新的评分
            reason: 新的原因
            is_active: 是否激活
        
        Returns:
            更新后的配置对象
        """
        if custom_score is not None and not (0.0 <= custom_score <= 1.0):
            logger.error(f"评分 {custom_score} 超出范围 [0, 1]")
            return None
        
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # 构建更新语句
            updates = []
            params = []
            
            if custom_score is not None:
                updates.append("custom_score = %s")
                params.append(custom_score)
            
            if reason is not None:
                updates.append("reason = %s")
                params.append(reason)
            
            if is_active is not None:
                updates.append("is_active = %s")
                params.append(is_active)
            
            if not updates:
                logger.warning("没有需要更新的字段")
                return None
            
            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(config_id)
            
            query = f"""
                UPDATE user_feature_quality_config
                SET {', '.join(updates)}
                WHERE config_id = %s
                RETURNING config_id, user_id, feature_name, custom_score,
                          reason, is_active, created_at, updated_at
            """
            
            cursor.execute(query, tuple(params))
            row = cursor.fetchone()
            
            if not row:
                logger.warning(f"配置 {config_id} 不存在")
                return None
            
            conn.commit()
            
            # 清除缓存
            self._clear_user_cache(row[1])  # user_id
            
            config = UserQualityConfig(
                config_id=row[0],
                user_id=row[1],
                feature_name=row[2],
                custom_score=row[3],
                reason=row[4],
                is_active=row[5],
                created_at=row[6],
                updated_at=row[7]
            )
            
            cursor.close()
            
            logger.info(f"更新配置 {config_id} 成功")
            return config
            
        except Exception as e:
            logger.error(f"更新配置失败: {e}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                return_db_connection(conn)
    
    def delete_config(self, config_id: int) -> bool:
        """
        删除自定义评分配置
        
        Args:
            config_id: 配置ID
        
        Returns:
            是否删除成功
        """
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # 先获取user_id用于清除缓存
            cursor.execute("""
                SELECT user_id FROM user_feature_quality_config WHERE config_id = %s
            """, (config_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.warning(f"配置 {config_id} 不存在")
                return False
            
            user_id = row[0]
            
            cursor.execute("""
                DELETE FROM user_feature_quality_config WHERE config_id = %s
            """, (config_id,))
            
            conn.commit()
            
            # 清除缓存
            self._clear_user_cache(user_id)
            
            cursor.close()
            
            logger.info(f"删除配置 {config_id} 成功")
            return True
            
        except Exception as e:
            logger.error(f"删除配置失败: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                return_db_connection(conn)
    
    def reset_to_default(self, user_id: str, feature_name: str) -> bool:
        """
        重置为默认评分（禁用自定义配置）
        
        Args:
            user_id: 用户ID
            feature_name: 特征名称
        
        Returns:
            是否重置成功
        """
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE user_feature_quality_config
                SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = %s AND feature_name = %s
            """, (user_id, feature_name))
            
            conn.commit()
            
            # 清除缓存
            self._clear_user_cache(user_id)
            
            cursor.close()
            
            logger.info(f"重置用户 {user_id} 的特征 {feature_name} 为默认评分")
            return True
            
        except Exception as e:
            logger.error(f"重置配置失败: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                return_db_connection(conn)
    
    def batch_create_configs(
        self,
        user_id: str,
        configs: List[Dict]
    ) -> Dict:
        """
        批量创建自定义评分配置
        
        Args:
            user_id: 用户ID
            configs: 配置列表 [{"feature_name": str, "custom_score": float, "reason": str}, ...]
        
        Returns:
            批量操作结果
        """
        results = {
            'success': [],
            'failed': [],
            'total': len(configs)
        }
        
        for config in configs:
            feature_name = config.get('feature_name')
            custom_score = config.get('custom_score')
            reason = config.get('reason')
            
            result = self.create_config(user_id, feature_name, custom_score, reason)
            
            if result:
                results['success'].append({
                    'feature_name': feature_name,
                    'config_id': result.config_id
                })
            else:
                results['failed'].append({
                    'feature_name': feature_name,
                    'error': '创建失败'
                })
        
        logger.info(f"批量创建完成: 成功 {len(results['success'])}, 失败 {len(results['failed'])}")
        return results
    
    def _clear_user_cache(self, user_id: str):
        """清除用户缓存"""
        cache_keys = [k for k in self._cache.keys() if k.startswith(f"{user_id}_")]
        for key in cache_keys:
            del self._cache[key]
        
        if cache_keys:
            logger.debug(f"清除用户 {user_id} 的 {len(cache_keys)} 个缓存项")
    
    def clear_all_cache(self):
        """清除所有缓存"""
        self._cache.clear()
        logger.info("所有配置缓存已清除")


# 全局配置管理器实例
_config_manager: Optional[FeatureQualityConfigManager] = None


def get_config_manager() -> FeatureQualityConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = FeatureQualityConfigManager()
    return _config_manager


def get_user_custom_score(user_id: str, feature_name: str) -> Optional[float]:
    """
    获取用户自定义评分（便捷函数）
    
    Args:
        user_id: 用户ID
        feature_name: 特征名称
    
    Returns:
        自定义评分，如果不存在则返回None
    """
    manager = get_config_manager()
    config = manager.get_config_by_feature(user_id, feature_name)
    
    if config and config.is_active:
        return config.custom_score
    
    return None
