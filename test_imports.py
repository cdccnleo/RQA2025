#!/usr/bin/env python3
"""测试关键导入"""

import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')

def test_import(path, name):
    try:
        module = __import__(path, fromlist=[name])
        obj = getattr(module, name)
        print(f"✅ {path}.{name} 导入成功")
        return True
    except Exception as e:
        print(f"❌ {path}.{name} 导入失败: {e}")
        return False

def main():
    imports = [
        ('src.core.core_services.api', 'APIService'),
        ('src.core.core_services.core.database_service', 'get_database_service'),
        ('src.infrastructure.utils.logger', 'get_logger'),
        ('src.infrastructure.config.core.unified_manager', 'UnifiedConfigManager'),
        ('src.trading.order_manager', 'OrderManager'),
        ('src.trading.execution_engine', 'ExecutionEngine'),
    ]

    success = True
    for path, name in imports:
        success &= test_import(path, name)

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)