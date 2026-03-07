"""
账户管理器别名模块
提供向后兼容的导入路径

实际实现在 account/account_manager.py 中
"""

try:
    from .account.account_manager import AccountManager
except ImportError:
    # 提供基础实现
    class AccountManager:
        pass

__all__ = ['AccountManager']

