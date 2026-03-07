"""
工具层 - logger.py 测试

测试src/utils/logger.py的基本功能
"""

import sys
from pathlib import Path

# 确保Python路径正确配置（必须在所有导入之前）
project_root = Path(__file__).resolve().parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

# 确保路径在sys.path的最前面
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
if src_path_str in sys.path:
    sys.path.remove(src_path_str)

sys.path.insert(0, project_root_str)
sys.path.insert(0, src_path_str)

import pytest
import logging
from unittest.mock import patch, MagicMock


def test_get_logger():
    """测试get_logger函数"""
    import importlib
    import sys
    from pathlib import Path
    
    # 确保路径配置
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))
    
    # 动态导入
    logger_module = importlib.import_module('src.utils.logger')
    get_logger = logger_module.get_logger
    
    # 测试默认参数
    logger1 = get_logger()
    assert isinstance(logger1, logging.Logger)
    
    # 测试指定名称
    logger2 = get_logger("test_logger")
    assert isinstance(logger2, logging.Logger)
    assert logger2.name == "test_logger"


def test_logger_class_init():
    """测试Logger类初始化"""
    from src.utils.logger import Logger
    
    # 测试默认参数
    logger1 = Logger()
    assert hasattr(logger1, 'logger')
    assert isinstance(logger1.logger, logging.Logger)
    
    # 测试指定名称
    logger2 = Logger("test_logger")
    assert logger2.logger.name == "test_logger"


def test_logger_class_methods():
    """测试Logger类的方法"""
    from src.utils.logger import Logger
    
    logger = Logger("test_logger")
    
    # 测试info方法
    with patch.object(logger.logger, 'info') as mock_info:
        logger.info("test message")
        mock_info.assert_called_once_with("test message")
    
    # 测试error方法
    with patch.object(logger.logger, 'error') as mock_error:
        logger.error("error message")
        mock_error.assert_called_once_with("error message")
    
    # 测试warning方法
    with patch.object(logger.logger, 'warning') as mock_warning:
        logger.warning("warning message")
        mock_warning.assert_called_once_with("warning message")
    
    # 测试debug方法
    with patch.object(logger.logger, 'debug') as mock_debug:
        logger.debug("debug message")
        mock_debug.assert_called_once_with("debug message")


def test_logger_with_args():
    """测试Logger类方法带参数"""
    from src.utils.logger import Logger
    
    logger = Logger("test_logger")
    
    # 测试带位置参数
    with patch.object(logger.logger, 'info') as mock_info:
        logger.info("test %s", "message")
        mock_info.assert_called_once_with("test %s", "message")
    
    # 测试带关键字参数
    with patch.object(logger.logger, 'error') as mock_error:
        logger.error("error", extra={"key": "value"})
        mock_error.assert_called_once_with("error", extra={"key": "value"})

