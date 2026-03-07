"""
策略版本管理模块
提供策略版本控制、对比和回滚功能
"""

import json
import logging
import os
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class StrategyVersion:
    """策略版本"""
    version_id: str
    strategy_id: str
    version_number: int
    created_at: float
    created_by: str
    comment: str
    strategy_data: Dict[str, Any]
    change_summary: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    is_active: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'version_id': self.version_id,
            'strategy_id': self.strategy_id,
            'version_number': self.version_number,
            'created_at': self.created_at,
            'created_by': self.created_by,
            'comment': self.comment,
            'strategy_data': self.strategy_data,
            'change_summary': self.change_summary,
            'tags': self.tags,
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyVersion':
        return cls(
            version_id=data['version_id'],
            strategy_id=data['strategy_id'],
            version_number=data['version_number'],
            created_at=data['created_at'],
            created_by=data.get('created_by', 'system'),
            comment=data.get('comment', ''),
            strategy_data=data['strategy_data'],
            change_summary=data.get('change_summary', {}),
            tags=data.get('tags', []),
            is_active=data.get('is_active', True)
        )


class StrategyVersionManager:
    """策略版本管理器"""
    
    def __init__(self, versions_dir: str = "data/strategy_versions"):
        self.versions_dir = versions_dir
        self._ensure_directory()
    
    def _ensure_directory(self):
        """确保版本目录存在"""
        if not os.path.exists(self.versions_dir):
            os.makedirs(self.versions_dir)
            logger.info(f"创建策略版本目录: {self.versions_dir}")
    
    def _get_strategy_versions_dir(self, strategy_id: str) -> str:
        """获取策略版本目录"""
        strategy_dir = os.path.join(self.versions_dir, strategy_id)
        if not os.path.exists(strategy_dir):
            os.makedirs(strategy_dir)
        return strategy_dir
    
    def _generate_version_id(self, strategy_id: str, version_number: int) -> str:
        """生成版本ID"""
        timestamp = int(time.time())
        hash_input = f"{strategy_id}_{version_number}_{timestamp}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"v{version_number}_{hash_suffix}"
    
    def create_version(self, strategy_id: str, strategy_data: Dict, 
                      comment: str = "", created_by: str = "system",
                      tags: List[str] = None) -> StrategyVersion:
        """创建新版本"""
        try:
            # 获取当前最新版本号
            versions = self.list_versions(strategy_id)
            latest_version = max([v.version_number for v in versions]) if versions else 0
            new_version_number = latest_version + 1
            
            # 生成版本ID
            version_id = self._generate_version_id(strategy_id, new_version_number)
            
            # 计算变更摘要
            change_summary = {}
            if versions:
                previous_version = versions[-1]
                change_summary = self._calculate_changes(
                    previous_version.strategy_data, 
                    strategy_data
                )
            
            # 创建版本对象
            version = StrategyVersion(
                version_id=version_id,
                strategy_id=strategy_id,
                version_number=new_version_number,
                created_at=time.time(),
                created_by=created_by,
                comment=comment,
                strategy_data=deepcopy(strategy_data),
                change_summary=change_summary,
                tags=tags or []
            )
            
            # 保存版本
            self._save_version(version)
            
            logger.info(f"创建策略版本成功: {strategy_id} v{new_version_number}")
            return version
            
        except Exception as e:
            logger.error(f"创建策略版本失败: {e}")
            raise
    
    def _save_version(self, version: StrategyVersion):
        """保存版本到文件"""
        try:
            strategy_dir = self._get_strategy_versions_dir(version.strategy_id)
            filepath = os.path.join(strategy_dir, f"{version.version_id}.json")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(version.to_dict(), f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"保存策略版本失败: {e}")
            raise
    
    def _calculate_changes(self, old_data: Dict, new_data: Dict) -> Dict[str, Any]:
        """计算两个版本之间的变更"""
        changes = {
            'added': {},
            'modified': {},
            'removed': {}
        }
        
        # 检查新增和修改
        for key in new_data:
            if key not in old_data:
                changes['added'][key] = new_data[key]
            elif old_data[key] != new_data[key]:
                changes['modified'][key] = {
                    'old': old_data[key],
                    'new': new_data[key]
                }
        
        # 检查删除
        for key in old_data:
            if key not in new_data:
                changes['removed'][key] = old_data[key]
        
        return changes
    
    def get_version(self, strategy_id: str, version_id: str) -> Optional[StrategyVersion]:
        """获取特定版本"""
        try:
            strategy_dir = self._get_strategy_versions_dir(strategy_id)
            filepath = os.path.join(strategy_dir, f"{version_id}.json")
            
            if not os.path.exists(filepath):
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return StrategyVersion.from_dict(data)
                
        except Exception as e:
            logger.error(f"获取策略版本失败: {e}")
            return None
    
    def get_version_by_number(self, strategy_id: str, version_number: int) -> Optional[StrategyVersion]:
        """通过版本号获取版本"""
        versions = self.list_versions(strategy_id)
        for version in versions:
            if version.version_number == version_number:
                return version
        return None
    
    def list_versions(self, strategy_id: str, include_inactive: bool = False) -> List[StrategyVersion]:
        """列出策略的所有版本"""
        try:
            strategy_dir = self._get_strategy_versions_dir(strategy_id)
            versions = []
            
            for filename in os.listdir(strategy_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(strategy_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            version = StrategyVersion.from_dict(data)
                            if include_inactive or version.is_active:
                                versions.append(version)
                    except Exception as e:
                        logger.warning(f"加载版本文件失败 {filename}: {e}")
            
            # 按版本号排序
            versions.sort(key=lambda v: v.version_number)
            return versions
            
        except Exception as e:
            logger.error(f"列出策略版本失败: {e}")
            return []
    
    def compare_versions(self, strategy_id: str, version_id1: str, 
                        version_id2: str) -> Dict[str, Any]:
        """对比两个版本"""
        try:
            version1 = self.get_version(strategy_id, version_id1)
            version2 = self.get_version(strategy_id, version_id2)
            
            if not version1 or not version2:
                return {"error": "版本不存在"}
            
            # 计算差异
            differences = self._calculate_changes(
                version1.strategy_data,
                version2.strategy_data
            )
            
            return {
                'version1': {
                    'version_id': version1.version_id,
                    'version_number': version1.version_number,
                    'created_at': version1.created_at,
                    'comment': version1.comment
                },
                'version2': {
                    'version_id': version2.version_id,
                    'version_number': version2.version_number,
                    'created_at': version2.created_at,
                    'comment': version2.comment
                },
                'differences': differences
            }
            
        except Exception as e:
            logger.error(f"对比策略版本失败: {e}")
            return {"error": str(e)}
    
    def rollback_to_version(self, strategy_id: str, version_id: str,
                           comment: str = "") -> Optional[StrategyVersion]:
        """回滚到指定版本"""
        try:
            # 获取目标版本
            target_version = self.get_version(strategy_id, version_id)
            if not target_version:
                logger.error(f"目标版本不存在: {version_id}")
                return None
            
            # 创建新版本（基于回滚目标）
            rollback_comment = comment or f"回滚到版本 {target_version.version_number}"
            new_version = self.create_version(
                strategy_id=strategy_id,
                strategy_data=deepcopy(target_version.strategy_data),
                comment=rollback_comment,
                tags=['rollback', f'from_v{target_version.version_number}']
            )
            
            logger.info(f"策略回滚成功: {strategy_id} -> v{target_version.version_number}")
            return new_version
            
        except Exception as e:
            logger.error(f"策略回滚失败: {e}")
            return None
    
    def delete_version(self, strategy_id: str, version_id: str, 
                      soft_delete: bool = True) -> bool:
        """删除版本"""
        try:
            version = self.get_version(strategy_id, version_id)
            if not version:
                return False
            
            if soft_delete:
                # 软删除：标记为不活跃
                version.is_active = False
                self._save_version(version)
                logger.info(f"版本软删除成功: {version_id}")
            else:
                # 硬删除：删除文件
                strategy_dir = self._get_strategy_versions_dir(strategy_id)
                filepath = os.path.join(strategy_dir, f"{version_id}.json")
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.info(f"版本硬删除成功: {version_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"删除版本失败: {e}")
            return False
    
    def get_version_statistics(self, strategy_id: str) -> Dict[str, Any]:
        """获取版本统计信息"""
        try:
            versions = self.list_versions(strategy_id, include_inactive=True)
            
            if not versions:
                return {
                    'strategy_id': strategy_id,
                    'total_versions': 0,
                    'active_versions': 0,
                    'latest_version': None
                }
            
            active_versions = [v for v in versions if v.is_active]
            latest_version = max(versions, key=lambda v: v.version_number)
            
            # 计算创建者分布
            creators = {}
            for v in versions:
                creators[v.created_by] = creators.get(v.created_by, 0) + 1
            
            return {
                'strategy_id': strategy_id,
                'total_versions': len(versions),
                'active_versions': len(active_versions),
                'inactive_versions': len(versions) - len(active_versions),
                'latest_version': {
                    'version_number': latest_version.version_number,
                    'version_id': latest_version.version_id,
                    'created_at': latest_version.created_at,
                    'comment': latest_version.comment
                },
                'creator_distribution': creators,
                'first_version_at': min(v.created_at for v in versions),
                'last_version_at': max(v.created_at for v in versions)
            }
            
        except Exception as e:
            logger.error(f"获取版本统计失败: {e}")
            return {"error": str(e)}
    
    def tag_version(self, strategy_id: str, version_id: str, 
                   tags: List[str]) -> bool:
        """为版本添加标签"""
        try:
            version = self.get_version(strategy_id, version_id)
            if not version:
                return False
            
            # 合并标签（去重）
            version.tags = list(set(version.tags + tags))
            self._save_version(version)
            
            logger.info(f"版本标签添加成功: {version_id} - {tags}")
            return True
            
        except Exception as e:
            logger.error(f"添加版本标签失败: {e}")
            return False
    
    def search_versions(self, strategy_id: str, 
                       keyword: str = None,
                       tags: List[str] = None,
                       created_by: str = None,
                       start_time: float = None,
                       end_time: float = None) -> List[StrategyVersion]:
        """搜索版本"""
        try:
            versions = self.list_versions(strategy_id, include_inactive=True)
            results = versions
            
            # 关键词搜索
            if keyword:
                keyword_lower = keyword.lower()
                results = [v for v in results if 
                          keyword_lower in v.comment.lower() or
                          keyword_lower in str(v.strategy_data).lower()]
            
            # 标签筛选
            if tags:
                results = [v for v in results if any(tag in v.tags for tag in tags)]
            
            # 创建者筛选
            if created_by:
                results = [v for v in results if v.created_by == created_by]
            
            # 时间范围筛选
            if start_time:
                results = [v for v in results if v.created_at >= start_time]
            if end_time:
                results = [v for v in results if v.created_at <= end_time]
            
            return results
            
        except Exception as e:
            logger.error(f"搜索版本失败: {e}")
            return []


# 全局版本管理器实例
version_manager = StrategyVersionManager()


# 便捷的API函数
def create_strategy_version(strategy_id: str, strategy_data: Dict,
                           comment: str = "", created_by: str = "system") -> StrategyVersion:
    """创建策略版本"""
    return version_manager.create_version(strategy_id, strategy_data, comment, created_by)


def get_strategy_version(strategy_id: str, version_id: str) -> Optional[StrategyVersion]:
    """获取策略版本"""
    return version_manager.get_version(strategy_id, version_id)


def list_strategy_versions(strategy_id: str) -> List[StrategyVersion]:
    """列出策略版本"""
    return version_manager.list_versions(strategy_id)


def compare_strategy_versions(strategy_id: str, version_id1: str, 
                             version_id2: str) -> Dict[str, Any]:
    """对比策略版本"""
    return version_manager.compare_versions(strategy_id, version_id1, version_id2)


def rollback_strategy_version(strategy_id: str, version_id: str, 
                             comment: str = "") -> Optional[StrategyVersion]:
    """回滚策略版本"""
    return version_manager.rollback_to_version(strategy_id, version_id, comment)
