# Minimal integration stub for tests: provides get_data_adapter to trigger fallback
# 注意：此文件已废弃，请使用 src.core.integration 包中的实现
# 为了保持向后兼容性，这里代理到包中的实现

def get_data_adapter():
    """延迟导入获取数据适配器（代理到包实现）"""
    try:
        # 从integration包的__init__.py导入（推荐方式）
        from src.core.integration import get_data_adapter as _get_data_adapter
        return _get_data_adapter()
    except (ImportError, AttributeError, RuntimeError) as e:
        # 如果包不存在或不可用，尝试从business_adapters模块导入
        try:
            from src.core.integration.business_adapters import get_data_adapter as _get_data_adapter
            return _get_data_adapter()
        except (ImportError, AttributeError, RuntimeError):
            # 最后的fallback实现
            raise RuntimeError(f"integration adapter not available in test environment: {e}")


