"""
进程实例池管理模块

管理业务流程进程实例的生命周期
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ProcessInstancePool:
    """
    进程实例池
    
    管理业务流程进程实例的创建、执行和销毁
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化进程实例池
        
        Args:
            config: 进程池配置
        """
        self.config = config or {}
        self.processes: Dict[str, Any] = {}
        self.max_instances = self.config.get('max_instances', 10)
        self.timeout_seconds = self.config.get('timeout_seconds', 300)
        
        logger.info(f"进程实例池初始化完成，最大实例数: {self.max_instances}")
    
    def create_instance(self, process_id: str, process_type: str, 
                       initial_data: Dict[str, Any] = None) -> Optional[str]:
        """
        创建进程实例
        
        Args:
            process_id: 进程ID
            process_type: 进程类型
            initial_data: 初始数据
            
        Returns:
            实例ID，如果创建失败返回None
        """
        try:
            # 检查是否达到最大实例数
            if len(self.processes) >= self.max_instances:
                logger.warning(f"进程池已满，无法创建新实例")
                return None
            
            # 生成实例ID
            instance_id = f"{process_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # 创建实例
            instance = {
                'instance_id': instance_id,
                'process_id': process_id,
                'process_type': process_type,
                'status': 'created',
                'data': initial_data or {},
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            
            self.processes[instance_id] = instance
            logger.info(f"进程实例创建成功: {instance_id}")
            
            return instance_id
            
        except Exception as e:
            logger.error(f"创建进程实例失败: {e}")
            return None
    
    def get_instance(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """
        获取进程实例
        
        Args:
            instance_id: 实例ID
            
        Returns:
            实例信息，如果不存在返回None
        """
        return self.processes.get(instance_id)
    
    def update_instance(self, instance_id: str, 
                       updates: Dict[str, Any]) -> bool:
        """
        更新进程实例
        
        Args:
            instance_id: 实例ID
            updates: 更新内容
            
        Returns:
            是否更新成功
        """
        try:
            if instance_id not in self.processes:
                logger.warning(f"进程实例不存在: {instance_id}")
                return False
            
            self.processes[instance_id].update(updates)
            self.processes[instance_id]['updated_at'] = datetime.now()
            
            logger.debug(f"进程实例更新成功: {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"更新进程实例失败: {e}")
            return False
    
    def delete_instance(self, instance_id: str) -> bool:
        """
        删除进程实例
        
        Args:
            instance_id: 实例ID
            
        Returns:
            是否删除成功
        """
        try:
            if instance_id not in self.processes:
                logger.warning(f"进程实例不存在: {instance_id}")
                return False
            
            del self.processes[instance_id]
            logger.info(f"进程实例删除成功: {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除进程实例失败: {e}")
            return False
    
    def list_instances(self, process_type: str = None, 
                      status: str = None) -> Dict[str, Dict[str, Any]]:
        """
        列出进程实例
        
        Args:
            process_type: 进程类型过滤
            status: 状态过滤
            
        Returns:
            实例字典
        """
        result = {}
        
        for instance_id, instance in self.processes.items():
            # 应用过滤器
            if process_type and instance.get('process_type') != process_type:
                continue
            if status and instance.get('status') != status:
                continue
            
            result[instance_id] = instance
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取进程池统计信息
        
        Returns:
            统计信息字典
        """
        total = len(self.processes)
        
        # 按状态统计
        status_counts = {}
        for instance in self.processes.values():
            status = instance.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_instances': total,
            'max_instances': self.max_instances,
            'available_slots': self.max_instances - total,
            'status_counts': status_counts
        }
    
    def cleanup_expired(self) -> int:
        """
        清理过期实例
        
        Returns:
            清理的实例数量
        """
        expired_count = 0
        current_time = datetime.now()
        
        expired_ids = []
        for instance_id, instance in self.processes.items():
            created_at = instance.get('created_at')
            if created_at:
                elapsed = (current_time - created_at).total_seconds()
                if elapsed > self.timeout_seconds:
                    expired_ids.append(instance_id)
        
        for instance_id in expired_ids:
            self.delete_instance(instance_id)
            expired_count += 1
        
        if expired_count > 0:
            logger.info(f"清理了 {expired_count} 个过期进程实例")
        
        return expired_count
