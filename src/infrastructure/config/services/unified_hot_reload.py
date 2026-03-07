
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from infrastructure.logging.services.hot_reload_service import HotReloadService
from typing import Dict, Any, Optional, Callable, Set
import logging
"""
统一配置管理器热重载功能
提供配置文件热重载功能
"""

logger = logging.getLogger(__name__)

# 全局热重载实例
_hot_reload_instance = None


def start_hot_reload() -> bool:
    """
unified_hot_reload - 配置管理

    职责说明：
    负责系统配置的统一管理、配置文件的读取、配置验证和配置分发

    核心职责：
    - 配置文件的读取和解析
    - 配置参数的验证
    - 配置的热重载
    - 配置的分发和同步
    - 环境变量管理
    - 配置加密和安全

    相关接口：
    - IConfigComponent
    - IConfigManager
    - IConfigValidator

    启动热重载(全局函数)
    """
    global _hot_reload_instance
    if _hot_reload_instance is None:
        _hot_reload_instance = UnifiedHotReload(enable_hot_reload=True)
        return _hot_reload_instance.start_hot_reload()
    return _hot_reload_instance.start_hot_reload()


def stop_hot_reload() -> bool:
    """停止热重载（全局函数）"""
    global _hot_reload_instance
    if _hot_reload_instance is None:
        return True
    return _hot_reload_instance.stop_hot_reload()


class UnifiedHotReload:

    """统一配置管理器热重载功能"""

    def __init__(self, enable_hot_reload: bool = False):
        """
        初始化热重载功能

        Args:
            enable_hot_reload: 是否启用热重载
        """
        self.enable_hot_reload = enable_hot_reload

        if enable_hot_reload:
            self._hot_reload_service = HotReloadService()
            self._watched_files: Set[str] = set()
        else:
            self._hot_reload_service = None
            self._watched_files: Set[str] = set()

    def start_hot_reload(self) -> bool:
        """启动热重载"""
        if not self.enable_hot_reload or not self._hot_reload_service:
            logger.warning("热重载功能未启用")
            return False

        try:
            success = self._hot_reload_service.start()
            if success:
                logger.info("热重载服务启动成功")
                return success
        except Exception as e:
            logger.error(f"启动热重载服务失败: {e}")
            return False

    def stop_hot_reload(self) -> bool:
        """停止热重载"""
        if not self.enable_hot_reload or not self._hot_reload_service:
            return True

        try:
            success = self._hot_reload_service.stop()
            if success:
                logger.info("热重载服务停止成功")
                return success
        except Exception as e:
            logger.error(f"停止热重载服务失败: {e}")
            return False

    def watch_file(self, file_path: str, callback: Optional[Callable] = None) -> bool:
        """监视文件"""
        if not self.enable_hot_reload or not self._hot_reload_service:
            logger.warning("热重载功能未启用")
            return False

        try:
            if callback is None:

                def on_config_change(changed_file: str, new_config: Dict[str, Any]):
                    logger.info(f"配置文件变更: {changed_file}")
                    # 这里可以添加默认的处理逻辑
                callback = on_config_change

            success = self._hot_reload_service.watch_file(file_path, callback)
            if success:
                self._watched_files.add(file_path)
                logger.info(f"开始监视文件: {file_path}")
            else:
                logger.warning(f"监视文件失败: {file_path}")
            return success
        except Exception as e:
            logger.error(f"监视文件失败 {file_path}: {e}")
            return False

    def unwatch_file(self, file_path: str) -> bool:
        """取消监视文件"""
        if not self.enable_hot_reload or not self._hot_reload_service:
            return True

        try:
            success = self._hot_reload_service.unwatch_file(file_path)
            if success:
                self._watched_files.discard(file_path)
                logger.info(f"停止监视文件: {file_path}")
                return success
        except Exception as e:
            logger.error(f"取消监视文件失败 {file_path}: {e}")
            return False

    def watch_directory(self, directory: str, pattern: str = "*.json",
                        callback: Optional[Callable] = None):
        """监视目录"""
        if not self.enable_hot_reload or not self._hot_reload_service:
            logger.warning("热重载功能未启用")
            return False

        try:
            if callback is None:

                def on_config_change(changed_file: str, new_config: Dict[str, Any]):
                    logger.info(f"配置文件变更: {changed_file}")
                    # 这里可以添加默认的处理逻辑
                callback = on_config_change

            success = self._hot_reload_service.watch_directory(directory, pattern, callback)
            if success:
                logger.info(f"开始监视目录: {directory} (模式: {pattern})")
                return success
        except Exception as e:
            logger.error(f"监视目录失败 {directory}: {e}")
            return False

    def get_hot_reload_status(self) -> Dict[str, Any]:
        """获取热重载状态"""
        if not self.enable_hot_reload or not self._hot_reload_service:
            return {
                "enabled": False,
                "running": False,
                "watched_files": [],
                "watched_directories": []
            }

        try:
            status = self._hot_reload_service.get_status()
            status.update({
                "enabled": self.enable_hot_reload,
                "watched_files": list(self._watched_files)
            })
            return status
        except Exception as e:
            logger.error(f"获取热重载状态失败: {e}")
            return {
                "enabled": self.enable_hot_reload,
                "running": False,
                "error": str(e),
                "watched_files": list(self._watched_files)
            }

    def reload_all_watched_files(self) -> Dict[str, bool]:
        """重新加载所有监视的文件"""
        if not self.enable_hot_reload or not self._hot_reload_service:
            return {}

        try:
            results = {}
            for file_path in self._watched_files:
                try:
                    # 这里可以添加重新加载文件的逻辑
                    results[file_path] = True
                    logger.info(f"重新加载文件: {file_path}")
                except Exception as e:
                    results[file_path] = False
                    logger.error(f"重新加载文件失败 {file_path}: {e}")

            return results
        except Exception as e:
            logger.error(f"重新加载监视文件失败: {e}")
            return {}

    def is_hot_reload_enabled(self) -> bool:
        """检查热重载是否启用"""
        return self.enable_hot_reload and self._hot_reload_service is not None

    def is_hot_reload_running(self) -> bool:
        """检查热重载是否运行中"""
        if not self.is_hot_reload_enabled():
            return False

        try:
            return self._hot_reload_service.is_running()
        except Exception as e:
            logger.error(f"检查热重载状态失败: {e}")
            return False

    def stop_watching(self) -> bool:
        """停止所有监控"""
        if not self.enable_hot_reload or not self._hot_reload_service:
            return True

        try:
            success = self._hot_reload_service.stop()
            if success:
                logger.info("停止所有监控成功")
                return success
        except Exception as e:
            logger.error(f"停止所有监控失败: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self, '_watched_files'):
                self._watched_files.clear()
            if self._hot_reload_service:
                self._hot_reload_service.stop()
                logger.info("热重载服务停止成功")
                logger.debug("配置热加载资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")

    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except Exception:
            pass




