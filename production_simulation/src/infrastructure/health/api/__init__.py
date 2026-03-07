"""
健康检查API模块
"""

# 延迟导入，因为DataAPIManager在data_api.py文件末尾定义
# 需要在data_api.py完全加载后才能导入
def _lazy_import():
    """延迟导入DataAPIManager和DataAPI"""
    from .data_api import DataAPIManager, DataAPI
    return DataAPIManager, DataAPI

try:
    # 尝试直接导入
    from .data_api import DataAPIManager
    DataAPI = DataAPIManager
except (ImportError, NameError):
    # 如果失败，使用延迟导入
    try:
        DataAPIManager, DataAPI = _lazy_import()
    except:
        # 提供基础实现
        class DataAPIManager:
            pass
        
        DataAPI = DataAPIManager

__all__ = ['DataAPI', 'DataAPIManager']

