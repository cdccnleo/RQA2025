#!/usr/bin/env python3
"""
重构后的Middleware组件实现

基于BaseComponent重构，消除代码重复
原有3个文件（bridge_components, connector_components, middleware_components）
存在高度相似的结构（179-184行/文件）

重构说明：
- 使用BaseComponent统一架构
- 消除重复的ComponentFactory定义
- 减少约300-400行重复代码

创建时间: 2025-11-03
版本: 2.0
"""

from typing import Dict, Any, Optional, List, Callable
from src.core.foundation.base_component import BaseComponent, ComponentFactory, component
import logging


@component("bridge")
class BridgeComponent(BaseComponent):
    """
    桥接组件（重构版）
    
    负责不同系统/模块之间的桥接通信
    基于BaseComponent，自动获得：
    - 日志管理
    - 状态跟踪
    - 错误处理
    - 性能监控
    """
    
    def __init__(self, name: str = "bridge", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._bridges: Dict[str, Callable] = {}
        self._connections: Dict[str, Any] = {}
        self._message_count = 0
    
    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        """初始化桥接组件"""
        try:
            # 注册桥接函数
            bridges = config.get('bridges', {})
            for bridge_name, bridge_func in bridges.items():
                self.register_bridge(bridge_name, bridge_func)
            
            # 初始化连接
            connections = config.get('connections', {})
            self._connections.update(connections)
            
            self._logger.info(f"桥接组件初始化: {len(self._bridges)} 个桥接, {len(self._connections)} 个连接")
            return True
            
        except Exception as e:
            self._logger.error(f"桥接组件初始化失败: {e}")
            return False
    
    def _do_execute(self, *args, **kwargs) -> Any:
        """执行桥接操作"""
        bridge_name = kwargs.get('bridge_name')
        message = kwargs.get('message')
        source = kwargs.get('source')
        target = kwargs.get('target')
        
        if bridge_name not in self._bridges:
            raise ValueError(f"未找到桥接: {bridge_name}")
        
        bridge_func = self._bridges[bridge_name]
        
        try:
            result = bridge_func(
                message=message,
                source=source,
                target=target,
                connections=self._connections
            )
            self._message_count += 1
            self._logger.debug(f"桥接消息成功: {bridge_name}, 总计: {self._message_count}")
            return result
            
        except Exception as e:
            self._logger.error(f"桥接失败: {bridge_name}, 错误: {e}")
            raise
    
    def register_bridge(self, name: str, bridge_func: Callable):
        """注册桥接函数"""
        self._bridges[name] = bridge_func
        self._logger.info(f"注册桥接: {name}")
    
    def register_connection(self, name: str, connection: Any):
        """注册连接"""
        self._connections[name] = connection
        self._logger.info(f"注册连接: {name}")
    
    def send_message(self, bridge_name: str, message: Any, source: str, target: str) -> Any:
        """发送消息（便捷方法）"""
        return self.execute(
            bridge_name=bridge_name,
            message=message,
            source=source,
            target=target
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        info = self.get_info()
        info.update({
            'bridges_count': len(self._bridges),
            'connections_count': len(self._connections),
            'messages_processed': self._message_count
        })
        return info


@component("connector")
class ConnectorComponent(BaseComponent):
    """
    连接器组件（重构版）
    
    负责建立和管理系统连接
    """
    
    def __init__(self, name: str = "connector", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._connectors: Dict[str, Callable] = {}
        self._active_connections: Dict[str, Any] = {}
        self._connection_pool: List[Any] = []
    
    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        """初始化连接器组件"""
        try:
            # 注册连接器
            connectors = config.get('connectors', {})
            for connector_name, connector_func in connectors.items():
                self.register_connector(connector_name, connector_func)
            
            # 初始化连接池
            pool_size = config.get('pool_size', 10)
            self._connection_pool = []
            
            self._logger.info(f"连接器组件初始化: {len(self._connectors)} 个连接器, 池大小: {pool_size}")
            return True
            
        except Exception as e:
            self._logger.error(f"连接器组件初始化失败: {e}")
            return False
    
    def _do_execute(self, *args, **kwargs) -> Any:
        """执行连接操作"""
        operation = kwargs.get('operation', 'connect')
        connector_name = kwargs.get('connector_name')
        
        if operation == 'connect':
            return self.connect(connector_name, **kwargs.get('params', {}))
        elif operation == 'disconnect':
            connection_id = kwargs.get('connection_id')
            return self.disconnect(connection_id)
        elif operation == 'get_connection':
            connection_id = kwargs.get('connection_id')
            return self.get_connection(connection_id)
        else:
            raise ValueError(f"不支持的操作: {operation}")
    
    def register_connector(self, name: str, connector_func: Callable):
        """注册连接器函数"""
        self._connectors[name] = connector_func
        self._logger.info(f"注册连接器: {name}")
    
    def connect(self, connector_name: str, **params) -> str:
        """建立连接"""
        if connector_name not in self._connectors:
            raise ValueError(f"未找到连接器: {connector_name}")
        
        connector_func = self._connectors[connector_name]
        
        try:
            connection = connector_func(**params)
            connection_id = f"{connector_name}_{len(self._active_connections)}"
            self._active_connections[connection_id] = connection
            
            self._logger.info(f"连接建立成功: {connection_id}")
            return connection_id
            
        except Exception as e:
            self._logger.error(f"连接失败: {connector_name}, 错误: {e}")
            raise
    
    def disconnect(self, connection_id: str) -> bool:
        """断开连接"""
        if connection_id in self._active_connections:
            connection = self._active_connections[connection_id]
            
            # 尝试关闭连接
            if hasattr(connection, 'close'):
                try:
                    connection.close()
                except Exception as e:
                    self._logger.warning(f"关闭连接异常: {e}")
            
            del self._active_connections[connection_id]
            self._logger.info(f"连接已断开: {connection_id}")
            return True
        
        return False
    
    def get_connection(self, connection_id: str) -> Optional[Any]:
        """获取连接"""
        return self._active_connections.get(connection_id)
    
    def get_all_connections(self) -> Dict[str, Any]:
        """获取所有活动连接"""
        return self._active_connections.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        info = self.get_info()
        info.update({
            'connectors_count': len(self._connectors),
            'active_connections': len(self._active_connections),
            'connection_ids': list(self._active_connections.keys())
        })
        return info


@component("middleware")
class MiddlewareComponent(BaseComponent):
    """
    中间件组件（重构版）
    
    提供请求/响应处理的中间件机制
    """
    
    def __init__(self, name: str = "middleware", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._middlewares: List[Callable] = []
        self._request_count = 0
        self._error_count = 0
    
    def _do_initialize(self, config: Dict[str, Any]) -> bool:
        """初始化中间件组件"""
        try:
            # 注册中间件
            middlewares = config.get('middlewares', [])
            for middleware_func in middlewares:
                self.add_middleware(middleware_func)
            
            self._logger.info(f"中间件组件初始化: {len(self._middlewares)} 个中间件")
            return True
            
        except Exception as e:
            self._logger.error(f"中间件组件初始化失败: {e}")
            return False
    
    def _do_execute(self, *args, **kwargs) -> Any:
        """执行中间件链"""
        request = kwargs.get('request')
        
        if request is None:
            raise ValueError("request不能为None")
        
        # 执行中间件链
        try:
            result = request
            
            for i, middleware in enumerate(self._middlewares):
                self._logger.debug(f"执行中间件 {i+1}/{len(self._middlewares)}")
                result = middleware(result)
                
                # 如果中间件返回None，中断链
                if result is None:
                    self._logger.warning(f"中间件 {i+1} 返回None，中断链")
                    break
            
            self._request_count += 1
            return result
            
        except Exception as e:
            self._error_count += 1
            self._logger.error(f"中间件链执行失败: {e}")
            raise
    
    def add_middleware(self, middleware_func: Callable):
        """添加中间件"""
        self._middlewares.append(middleware_func)
        self._logger.info(f"添加中间件: {middleware_func.__name__ if hasattr(middleware_func, '__name__') else 'anonymous'}")
    
    def insert_middleware(self, index: int, middleware_func: Callable):
        """在指定位置插入中间件"""
        self._middlewares.insert(index, middleware_func)
        self._logger.info(f"在位置 {index} 插入中间件")
    
    def remove_middleware(self, index: int):
        """移除指定位置的中间件"""
        if 0 <= index < len(self._middlewares):
            self._middlewares.pop(index)
            self._logger.info(f"移除位置 {index} 的中间件")
    
    def process_request(self, request: Any) -> Any:
        """处理请求（便捷方法）"""
        return self.execute(request=request)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        info = self.get_info()
        info.update({
            'middlewares_count': len(self._middlewares),
            'requests_processed': self._request_count,
            'errors': self._error_count,
            'success_rate': f"{(self._request_count - self._error_count) / max(self._request_count, 1) * 100:.2f}%"
        })
        return info


def create_middleware_components() -> Dict[str, BaseComponent]:
    """
    创建所有middleware组件的便捷函数
    
    Returns:
        包含所有组件实例的字典
    """
    factory = ComponentFactory()
    
    components = {
        'bridge': factory.create_component(
            'bridge',
            BridgeComponent,
            {}
        ),
        'connector': factory.create_component(
            'connector',
            ConnectorComponent,
            {}
        ),
        'middleware': factory.create_component(
            'middleware',
            MiddlewareComponent,
            {}
        )
    }
    
    return components


# 向后兼容的别名
IBridgeComponent = BridgeComponent
IConnectorComponent = ConnectorComponent
IMiddlewareComponent = MiddlewareComponent


__all__ = [
    'BridgeComponent',
    'ConnectorComponent',
    'MiddlewareComponent',
    'create_middleware_components',
    # 向后兼容
    'IBridgeComponent',
    'IConnectorComponent',
    'IMiddlewareComponent'
]

