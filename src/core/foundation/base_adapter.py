#!/usr/bin/env python3
"""
统一适配器基类框架

提供所有适配器类的基础抽象和公共功能，消除代码重复
创建时间: 2025-11-03
版本: 1.0
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar, Generic
import logging
from datetime import datetime
from enum import Enum


# 定义泛型类型
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class AdapterStatus(Enum):
    """适配器状态"""
    READY = "ready"
    ADAPTING = "adapting"
    ERROR = "error"
    DISABLED = "disabled"


class IAdapter(ABC, Generic[InputType, OutputType]):
    """
    适配器基础接口
    
    所有适配器必须实现的核心接口
    """
    
    @abstractmethod
    def adapt(self, data: InputType) -> OutputType:
        """
        适配数据
        
        Args:
            data: 输入数据
            
        Returns:
            适配后的输出数据
        """
        pass
    
    @abstractmethod
    def validate_input(self, data: InputType) -> bool:
        """
        验证输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            数据是否有效
        """
        pass


class BaseAdapter(IAdapter[InputType, OutputType]):
    """
    适配器基类
    
    提供所有适配器的公共功能：
    - 日志管理
    - 错误处理
    - 数据验证
    - 性能监控
    - 缓存支持
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        enable_cache: bool = False
    ):
        """
        初始化适配器
        
        Args:
            name: 适配器名称
            config: 配置参数
            enable_cache: 是否启用缓存
        """
        self.name = name
        self.config = config or {}
        self.enable_cache = enable_cache
        
        self._status = AdapterStatus.READY
        self._logger = self._setup_logger()
        self._error_count = 0
        self._success_count = 0
        self._last_error: Optional[Exception] = None
        self._cache: Dict[Any, OutputType] = {}
        self._created_at = datetime.now()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        return logger
    
    def adapt(self, data: InputType) -> OutputType:
        """
        适配数据（主要入口）
        
        Args:
            data: 输入数据
            
        Returns:
            适配后的输出数据
            
        Raises:
            ValueError: 输入数据无效
            RuntimeError: 适配过程失败
        """
        # 检查缓存
        if self.enable_cache:
            cache_key = self._get_cache_key(data)
            if cache_key in self._cache:
                self._logger.debug(f"从缓存获取结果: {cache_key}")
                return self._cache[cache_key]
        
        # 验证输入
        if not self.validate_input(data):
            self._error_count += 1
            self._logger.error(f"适配器 {self.name} 输入数据验证失败")
            raise ValueError(f"无效的输入数据: {type(data)}")
        
        try:
            self._status = AdapterStatus.ADAPTING
            
            # 前置处理
            preprocessed_data = self._preprocess(data)
            
            # 执行适配
            result = self._do_adapt(preprocessed_data)
            
            # 后置处理
            postprocessed_result = self._postprocess(result)
            
            # 验证输出
            if not self._validate_output(postprocessed_result):
                raise ValueError(f"适配器 {self.name} 输出数据验证失败")
            
            # 更新缓存
            if self.enable_cache:
                cache_key = self._get_cache_key(data)
                self._cache[cache_key] = postprocessed_result
            
            self._success_count += 1
            self._status = AdapterStatus.READY
            
            return postprocessed_result
            
        except Exception as e:
            self._status = AdapterStatus.ERROR
            self._error_count += 1
            self._last_error = e
            self._logger.error(f"适配器 {self.name} 执行失败: {e}")
            
            # 尝试错误恢复
            return self._handle_error(data, e)
    
    @abstractmethod
    def _do_adapt(self, data: InputType) -> OutputType:
        """
        子类实现具体的适配逻辑
        
        Args:
            data: 预处理后的输入数据
            
        Returns:
            适配后的输出数据
        """
        pass
    
    def validate_input(self, data: InputType) -> bool:
        """
        验证输入数据（默认实现）
        
        Args:
            data: 输入数据
            
        Returns:
            数据是否有效
        """
        # 默认实现：检查数据不为None
        if data is None:
            self._logger.warning("输入数据为None")
            return False
        return True
    
    def _validate_output(self, data: OutputType) -> bool:
        """
        验证输出数据
        
        Args:
            data: 输出数据
            
        Returns:
            数据是否有效
        """
        # 默认实现：检查数据不为None
        return data is not None
    
    def _preprocess(self, data: InputType) -> InputType:
        """
        数据预处理
        
        Args:
            data: 原始输入数据
            
        Returns:
            预处理后的数据
        """
        # 默认实现：不做处理，子类可以覆盖
        return data
    
    def _postprocess(self, data: OutputType) -> OutputType:
        """
        数据后处理
        
        Args:
            data: 适配后的数据
            
        Returns:
            后处理后的数据
        """
        # 默认实现：不做处理，子类可以覆盖
        return data
    
    def _get_cache_key(self, data: InputType) -> Any:
        """
        生成缓存键
        
        Args:
            data: 输入数据
            
        Returns:
            缓存键
        """
        # 默认实现：使用hash，子类可以覆盖
        try:
            return hash(str(data))
        except:
            return id(data)
    
    def _handle_error(self, data: InputType, error: Exception) -> OutputType:
        """
        错误处理和恢复
        
        Args:
            data: 导致错误的输入数据
            error: 发生的异常
            
        Returns:
            恢复后的结果（如果可能）
            
        Raises:
            RuntimeError: 无法恢复时抛出
        """
        # 默认实现：重新抛出异常，子类可以实现恢复逻辑
        raise RuntimeError(f"适配器 {self.name} 执行失败: {error}") from error
    
    def get_stats(self) -> Dict[str, Any]:
        """获取适配器统计信息"""
        total = self._success_count + self._error_count
        success_rate = (self._success_count / total * 100) if total > 0 else 0
        
        return {
            'name': self.name,
            'status': self._status.value,
            'success_count': self._success_count,
            'error_count': self._error_count,
            'total_count': total,
            'success_rate': f"{success_rate:.2f}%",
            'cache_enabled': self.enable_cache,
            'cache_size': len(self._cache) if self.enable_cache else 0,
            'last_error': str(self._last_error) if self._last_error else None,
            'created_at': self._created_at.isoformat()
        }
    
    def clear_cache(self):
        """清空缓存"""
        if self.enable_cache:
            self._cache.clear()
            self._logger.info(f"适配器 {self.name} 缓存已清空")
    
    def reset_stats(self):
        """重置统计信息"""
        self._success_count = 0
        self._error_count = 0
        self._last_error = None
        self._logger.info(f"适配器 {self.name} 统计信息已重置")
    
    def disable(self):
        """禁用适配器"""
        self._status = AdapterStatus.DISABLED
        self._logger.warning(f"适配器 {self.name} 已禁用")
    
    def enable(self):
        """启用适配器"""
        self._status = AdapterStatus.READY
        self._logger.info(f"适配器 {self.name} 已启用")
    
    def is_healthy(self) -> bool:
        """检查适配器健康状态"""
        if self._status == AdapterStatus.DISABLED:
            return False
        
        total = self._success_count + self._error_count
        if total == 0:
            return True
        
        # 成功率低于50%视为不健康
        success_rate = self._success_count / total
        return success_rate >= 0.5


class AdapterChain(Generic[InputType, OutputType]):
    """
    适配器链
    
    支持将多个适配器串联执行
    """
    
    def __init__(self, name: str):
        self.name = name
        self._adapters: list[BaseAdapter] = []
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def add_adapter(self, adapter: BaseAdapter) -> 'AdapterChain':
        """
        添加适配器到链中
        
        Args:
            adapter: 适配器实例
            
        Returns:
            self（支持链式调用）
        """
        self._adapters.append(adapter)
        self._logger.info(f"添加适配器到链: {adapter.name}")
        return self
    
    def execute(self, data: InputType) -> OutputType:
        """
        执行适配器链
        
        Args:
            data: 输入数据
            
        Returns:
            经过所有适配器处理后的输出数据
        """
        result = data
        
        for adapter in self._adapters:
            if not adapter.is_healthy():
                self._logger.warning(f"跳过不健康的适配器: {adapter.name}")
                continue
            
            try:
                result = adapter.adapt(result)
            except Exception as e:
                self._logger.error(f"适配器链执行失败于 {adapter.name}: {e}")
                raise
        
        return result
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """获取适配器链统计信息"""
        return {
            'chain_name': self.name,
            'adapter_count': len(self._adapters),
            'adapters': [adapter.get_stats() for adapter in self._adapters]
        }


# 便捷的适配器装饰器
def adapter(name: Optional[str] = None, enable_cache: bool = False):
    """
    适配器装饰器
    
    使用示例:
        @adapter("my_adapter", enable_cache=True)
        class MyAdapter(BaseAdapter):
            pass
    """
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            kwargs.setdefault('name', name or cls.__name__)
            kwargs.setdefault('enable_cache', enable_cache)
            original_init(self, *args, **kwargs)
        
        cls.__init__ = new_init
        return cls
    
    return decorator


__all__ = [
    'IAdapter',
    'BaseAdapter',
    'AdapterChain',
    'AdapterStatus',
    'adapter',
    'InputType',
    'OutputType'
]

