"""
系统操作管理器

负责管理系统的启动、停止、状态检查和配置更新等操作。
"""

import time
from typing import Dict, Any, List, Optional
import logging


class SystemOperationsManager:
    """系统操作管理器"""
    
    def __init__(self, registry, logger: logging.Logger):
        self.registry = registry
        self.logger = logger

    def start_system(self) -> bool:
        """启动系统"""
        self.logger.info("启动监控告警系统...")
        success_count = 0
        total_count = 0
        
        for component_name in self.registry.list_components():
            component = self.registry.create_component(component_name)
            total_count += 1
            
            if component and hasattr(component, 'start'):
                try:
                    component.start()
                    self.logger.info(f"组件 {component_name} 已启动")
                    success_count += 1
                except Exception as e:
                    self.logger.error(f"启动组件 {component_name} 失败: {e}")
        
        self.logger.info(f"系统启动完成: {success_count}/{total_count} 个组件启动成功")
        return success_count > 0

    def stop_system(self) -> bool:
        """停止系统"""
        self.logger.info("停止监控告警系统...")
        success_count = 0
        total_count = 0
        
        for component_name in self.registry.list_components():
            component = self.registry.create_component(component_name)
            total_count += 1
            
            if component and hasattr(component, 'stop'):
                try:
                    component.stop()
                    self.logger.info(f"组件 {component_name} 已停止")
                    success_count += 1
                except Exception as e:
                    self.logger.error(f"停止组件 {component_name} 失败: {e}")
        
        self.logger.info(f"系统停止完成: {success_count}/{total_count} 个组件停止成功")
        return True

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'system': 'monitoring_alert_system',
            'status': 'running',
            'components': {}
        }

        # 检查各组件状态
        for component_name in self.registry.list_components():
            component = self.registry.create_component(component_name)
            if component:
                comp_status = self._get_component_status(component, component_name)
                status['components'][component_name] = comp_status
            else:
                status['components'][component_name] = 'not_loaded'

        return status

    def get_system_health_report(self) -> Dict[str, Any]:
        """获取系统健康报告"""
        report = {
            'timestamp': time.time(),
            'system': 'monitoring_alert_system',
            'overall_health': 'healthy',
            'components': []
        }

        # 收集各组件健康信息
        for component_name in self.registry.list_components():
            component = self.registry.create_component(component_name)
            if component:
                health_info = self._get_component_health_info(component, component_name)
                report['components'].append(health_info)

        return report

    def update_configuration(self, config: Dict[str, Any]) -> bool:
        """更新配置"""
        self.logger.info("配置已更新")
        # 通知相关组件重新加载配置
        self._notify_config_update(config)
        return True

    def _get_component_status(self, component: Any, component_name: str) -> str:
        """获取组件状态"""
        if hasattr(component, 'is_healthy') and component.is_healthy():
            return 'healthy'
        return 'unknown'

    def _get_component_health_info(self, component: Any, component_name: str) -> Dict[str, Any]:
        """获取组件健康信息"""
        health_info = {
            'name': component_name,
            'status': self._get_component_status(component, component_name),
            'last_check': time.time()
        }
        
        if hasattr(component, 'get_health_info'):
            health_info.update(component.get_health_info())
        
        return health_info

    def _notify_config_update(self, config: Dict[str, Any]) -> None:
        """通知配置更新"""
        for component_name in self.registry.list_components():
            component = self.registry.create_component(component_name)
            if component and hasattr(component, 'reload_config'):
                try:
                    component.reload_config(config)
                    self.logger.info(f"组件 {component_name} 配置已重新加载")
                except Exception as e:
                    self.logger.error(f"重新加载组件 {component_name} 配置失败: {e}")
