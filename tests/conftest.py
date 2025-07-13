"""
测试配置文件
"""
import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 数据层测试fixture
@pytest.fixture(scope="session")
def temp_test_dir():
    """临时测试目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_stock_data():
    """示例股票数据"""
    dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
    return pd.DataFrame({
        'date': dates,
        'symbol': ['000001.SZ'] * len(dates),
        'close': [100 + i for i in range(len(dates))],
        'volume': [1000 + i * 10 for i in range(len(dates))],
        'open': [99 + i for i in range(len(dates))],
        'high': [102 + i for i in range(len(dates))],
        'low': [98 + i for i in range(len(dates))]
    })

@pytest.fixture
def sample_index_data():
    """示例指数数据"""
    dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
    return pd.DataFrame({
        'date': dates,
        'symbol': ['HS300'] * len(dates),
        'close': [3000 + i * 10 for i in range(len(dates))],
        'volume': [1000000 + i * 10000 for i in range(len(dates))]
    })

@pytest.fixture
def sample_news_data():
    """示例新闻数据"""
    dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
    return pd.DataFrame({
        'date': dates,
        'title': [f'新闻标题{i}' for i in range(len(dates))],
        'content': [f'新闻内容{i}' for i in range(len(dates))],
        'sentiment': [0.1 + i * 0.1 for i in range(len(dates))]
    })

@pytest.fixture
def sample_financial_data():
    """示例财务数据"""
    dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
    return pd.DataFrame({
        'date': dates,
        'symbol': ['000001.SZ'] * len(dates),
        'revenue': [1000000 + i * 10000 for i in range(len(dates))],
        'profit': [100000 + i * 1000 for i in range(len(dates))],
        'assets': [10000000 + i * 100000 for i in range(len(dates))]
    })

@pytest.fixture
def data_config():
    """数据配置"""
    return {
        "General": {
            'max_concurrent_workers': '2',
            'cache_dir': 'test_cache',
            'max_cache_size': str(1024 * 1024 * 100),
            'cache_ttl': '3600',
        },
        "Stock": {
            'save_path': 'data/stock',
            'max_retries': '3',
            'cache_days': '7',
            'frequency': 'daily',
            'adjust_type': 'none'
        },
        "News": {
            'save_path': 'data/news',
            'max_retries': '3',
            'cache_days': '3'
        },
        "Financial": {
            'save_path': 'data/financial',
            'max_retries': '3',
            'cache_days': '30'
        },
        "Index": {
            'save_path': 'data/index',
            'max_retries': '3',
            'cache_days': '30'
        }
    }

@pytest.fixture
def infrastructure_config():
    """基础设施配置"""
    return {
        "Infrastructure": {
            'monitoring_enabled': 'true',
            'error_handling_enabled': 'true',
            'logging_enabled': 'true',
            'cache_enabled': 'true',
            'database_enabled': 'true'
        },
        "Security": {
            'encryption_enabled': 'true',
            'access_control_enabled': 'true',
            'audit_logging_enabled': 'true'
        },
        "Performance": {
            'auto_scaling_enabled': 'true',
            'load_balancing_enabled': 'true',
            'distributed_cache_enabled': 'true'
        }
    }

# 基础设施层测试fixture
@pytest.fixture
def mock_config_manager():
    """模拟配置管理器"""
    config_manager = Mock()
    config_manager.get.return_value = "test_value"
    config_manager.update_config.return_value = True
    return config_manager

@pytest.fixture
def mock_metrics_collector():
    """模拟指标收集器"""
    metrics_collector = Mock()
    metrics_collector.record_metric.return_value = True
    return metrics_collector

@pytest.fixture
def mock_error_handler():
    """模拟错误处理器"""
    error_handler = Mock()
    error_handler.handle_error.return_value = True
    return error_handler

@pytest.fixture
def mock_logger():
    """模拟日志记录器"""
    logger = Mock()
    logger.info.return_value = None
    logger.error.return_value = None
    logger.warning.return_value = None
    logger.debug.return_value = None
    return logger

@pytest.fixture
def mock_cache_manager():
    """模拟缓存管理器"""
    cache_manager = Mock()
    cache_manager.set.return_value = True
    cache_manager.get.return_value = None
    cache_manager.delete.return_value = True
    return cache_manager

@pytest.fixture
def mock_db_manager():
    """模拟数据库管理器"""
    db_manager = Mock()
    db_manager.store_data.return_value = True
    db_manager.get_data.return_value = None
    db_manager.delete_data.return_value = True
    return db_manager

# 测试标记
def pytest_configure(config):
    """配置pytest标记"""
    config.addinivalue_line(
        "markers", "performance: 标记为性能测试"
    )
    config.addinivalue_line(
        "markers", "error_handling: 标记为错误处理测试"
    )
    config.addinivalue_line(
        "markers", "integration: 标记为集成测试"
    )
    config.addinivalue_line(
        "markers", "unit: 标记为单元测试"
    )

# 测试收集钩子
def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    for item in items:
        # 为数据层测试添加标记
        if "data" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # 为基础设施集成测试添加标记
        if "infrastructure" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # 为性能测试添加标记
        if "performance" in item.nodeid or "Performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        
        # 为错误处理测试添加标记
        if "error" in item.nodeid or "Error" in item.nodeid:
            item.add_marker(pytest.mark.error_handling)

# 测试报告钩子
def pytest_html_report_title(report):
    """设置HTML报告标题"""
    report.title = "RQA2025 数据层核心功能测试报告"

def pytest_html_results_table_header(cells):
    """自定义HTML报告表头"""
    # 简化版本，不修改表头
    pass

def pytest_html_results_table_row(report, cells):
    """自定义HTML报告行"""
    # 简化版本，不修改行
    pass

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """添加测试描述到报告"""
    outcome = yield
    report = outcome.get_result()
    
    # 添加测试描述
    if hasattr(item, 'function'):
        report.description = str(item.function.__doc__)
