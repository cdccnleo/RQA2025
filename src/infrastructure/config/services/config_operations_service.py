
from typing import Any, Dict, List, Callable
from ..core.imports import (
    logging, time
)
from .config_storage_service import ConfigStorageService
from .iconfig_service import (
    IConfigService, BaseConfigService
)

"""
配置操作服务

Phase 1重构：使用组合模式替代多重继承
提供配置的增删改查、验证、监听功能
"""

logger = logging.getLogger(__name__)


class ConfigOperationsService(BaseConfigService, IConfigService):
    """配置操作服务

    提供配置的增删改查操作，组合存储服务和验证服务
    """

    def __init__(self, storage_service: ConfigStorageService):
        """初始化操作服务

        Args:
            storage_service: 存储服务实例
        """
        super().__init__("config_operations_service")
        self._storage_service = storage_service

        # 操作配置
        self._validators: List[Callable] = []
        self._listeners: List[Callable] = []
        self._preprocessors: List[Callable] = []
        self._postprocessors: List[Callable] = []

        # 操作历史
        self._operation_history: List[Dict[str, Any]] = []
        self._max_history_size = 1000

        # 统计信息 - 初始化所有键为0
        self._operation_stats = {
            'get': 0, 'set': 0, 'delete': 0, 'exists': 0,
            'keys': 0, 'clear': 0, 'validation_errors': 0
        }

    def _initialize(self):
        """初始化服务"""
        self._start_time = time.time()
        logger.info("配置操作服务已初始化")

    def reset_operation_stats(self):
        """重置操作统计"""
        self._operation_stats = {k: 0 for k in self._operation_stats}
        logger.info("操作统计已重置")

    def add_validator(self, validator: Callable):
        """添加验证器

        Args:
            validator: 验证函数，接受(key, value)参数
        """
        self._validators.append(validator)

    def remove_validator(self, validator: Callable):
        """移除验证器"""
        if validator in self._validators:
            self._validators.remove(validator)

    def add_listener(self, listener: Callable):
        """添加变更监听器

        Args:
            listener: 监听函数，接受(event_type, data)参数
        """
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable):
        """移除变更监听器"""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def add_preprocessor(self, preprocessor: Callable):
        """添加预处理器

        Args:
            preprocessor: 预处理函数，在操作前调用
        """
        self._preprocessors.append(preprocessor)

    def add_postprocessor(self, postprocessor: Callable):
        """添加后处理器

        Args:
            postprocessor: 后处理函数，在操作后调用
        """
        self._postprocessors.append(postprocessor)

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值或默认值
        """
        self._ensure_initialized()

        try:
            start_time = time.time()

            # 预处理
            self._run_preprocessors('get', key)

            # 从存储服务获取值
            value = self._storage_service.get(key, default)

            duration = time.time() - start_time
            self._record_operation('get', key, success=True, duration=duration)

            # 后处理 - 传入success=True
            self._run_postprocessors('get', key, value, True)

            # 通知监听器
            self._notify_listeners('get', {'key': key, 'value': value})

            self._operation_stats['get'] += 1
            return value

        except Exception as e:
            self._record_operation('get', key, success=False, error=str(e))
            logger.error(f"获取配置失败 {key}: {e}")
            raise

    def set(self, key: str, value: Any) -> bool:
        """设置配置值

        Args:
            key: 配置键
            value: 配置值

        Returns:
            设置是否成功
        """
        self._ensure_initialized()

        try:
            start_time = time.time()

            # 验证
            if not self._validate(key, value):
                self._operation_stats['validation_errors'] += 1
                return False

            # 预处理
            self._run_preprocessors('set', key, value)

            # 通过存储服务设置值
            success = self._storage_service.set(key, value)

            duration = time.time() - start_time
            self._record_operation('set', key, success=success, duration=duration, value=value)

            if success:
                # 后处理 - 传入success
                self._run_postprocessors('set', key, value, success)

                # 通知监听器
                self._notify_listeners('set', {'key': key, 'value': value})

                self._operation_stats['set'] += 1

            return success

        except Exception as e:
            self._record_operation('set', key, success=False, error=str(e))
            logger.error(f"设置配置失败 {key}: {e}")
            raise

    def delete(self, key: str) -> bool:
        """删除配置项

        Args:
            key: 配置键

        Returns:
            删除是否成功
        """
        self._ensure_initialized()

        try:
            start_time = time.time()

            # 预处理
            self._run_preprocessors('delete', key)

            # 通过存储服务删除
            success = self._storage_service.delete(key)

            duration = time.time() - start_time
            self._record_operation('delete', key, success=success, duration=duration)

            if success:
                # 后处理 - 传入success
                self._run_postprocessors('delete', key, None, success)

                # 通知监听器
                self._notify_listeners('delete', {'key': key})

                self._operation_stats['delete'] += 1

            return success

        except Exception as e:
            self._record_operation('delete', key, success=False, error=str(e))
            logger.error(f"删除配置失败 {key}: {e}")
            raise

    def exists(self, key: str) -> bool:
        """检查配置项是否存在

        Args:
            key: 配置键

        Returns:
            是否存在
        """
        self._ensure_initialized()

        try:
            # 通过存储服务检查
            exists = self._storage_service.exists(key)

            self._operation_stats['exists'] += 1
            self._record_operation('exists', key, success=True)

            return exists

        except Exception as e:
            self._record_operation('exists', key, success=False, error=str(e))
            logger.error(f"检查配置存在性失败 {key}: {e}")
            raise

    def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配模式的配置键

        Args:
            pattern: 匹配模式

        Returns:
            配置键列表
        """
        self._ensure_initialized()

        try:
            # 通过存储服务获取键
            keys = self._storage_service.keys(pattern)

            self._operation_stats['keys'] += 1
            self._record_operation('keys', pattern, success=True)

            return keys

        except Exception as e:
            self._record_operation('keys', pattern, success=False, error=str(e))
            logger.error(f"获取配置键失败 {pattern}: {e}")
            raise

    def clear(self) -> bool:
        """清空所有配置

        Returns:
            清空是否成功
        """
        self._ensure_initialized()

        try:
            # 通过存储服务清空
            success = self._storage_service.clear()

            self._record_operation('clear', None, success=success)

            if success:
                # 通知监听器
                self._notify_listeners('clear', {})

                self._operation_stats['clear'] += 1

            return success

        except Exception as e:
            self._record_operation('clear', None, success=False, error=str(e))
            logger.error(f"清空配置失败: {e}")
            raise

    def _validate(self, key: str, value: Any) -> bool:
        """验证配置项

        Args:
            key: 配置键
            value: 配置值

        Returns:
            验证是否通过
        """
        for validator in self._validators:
            try:
                if not validator(key, value):
                    return False
            except Exception as e:
                validator_name = getattr(validator, '__name__', repr(validator))
                logger.warning(f"验证器执行失败 {validator_name}: {e}")
                return False
        return True

    def _run_preprocessors(self, operation: str, *args, **kwargs):
        """运行预处理器"""
        for preprocessor in self._preprocessors:
            try:
                preprocessor(operation, *args, **kwargs)
            except Exception as e:
                preprocessor_name = getattr(preprocessor, '__name__', repr(preprocessor))
                logger.warning(f"预处理器执行失败 {preprocessor_name}: {e}")

    def _run_postprocessors(self, operation: str, *args, **kwargs):
        """运行后处理器"""
        for postprocessor in self._postprocessors:
            try:
                postprocessor(operation, *args, **kwargs)
            except Exception as e:
                postprocessor_name = getattr(postprocessor, '__name__', repr(postprocessor))
                logger.warning(f"后处理器执行失败 {postprocessor_name}: {e}")

    def _notify_listeners(self, event_type: str, data: Dict[str, Any]):
        """通知监听器"""
        for listener in self._listeners:
            try:
                listener(event_type, data)
            except Exception as e:
                listener_name = getattr(listener, '__name__', repr(listener))
                logger.warning(f"监听器执行失败 {listener_name}: {e}")

    def _record_operation(self, operation: str, key: str = None,
                          success: bool = True, duration: float = None,
                          value: Any = None, error: str = None):
        """记录操作历史"""
        record = {
            'timestamp': time.time(),
            'operation': operation,
            'key': key,
            'success': success,
            'duration': duration,
            'value': value,
            'error': error
        }

        self._add_to_history(record)

    def _add_to_history(self, record: Dict[str, Any]):
        """添加历史记录"""
        self._operation_history.append(record)

        # 限制历史记录大小
        if len(self._operation_history) > self._max_history_size:
            self._operation_history.pop(0)

    def get_operation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取操作历史

        Args:
            limit: 返回的最大记录数

        Returns:
            操作历史记录
        """
        if limit <= 0:
            return []
        return self._operation_history[-limit:]

    def get_operation_stats(self) -> Dict[str, Any]:
        """获取操作统计信息"""
        return {
            'operations': dict(self._operation_stats),
            'history_size': len(self._operation_history),
            'uptime': time.time() - (self._start_time or time.time())
        }

    def cleanup(self):
        """清理资源"""
        self._validators.clear()
        self._listeners.clear()
        self._preprocessors.clear()
        self._postprocessors.clear()
        self._operation_history.clear()

        logger.info("配置操作服务已清理")




