#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量检查Python文件语法
"""

import sys
import py_compile


def check_syntax():
    """检查所有集成测试文件的语法"""
    print('🔧 集成测试文件语法检查')
    print('=' * 60)

    test_files = [
        'tests/integration/test_database_integration.py',
        'tests/integration/test_cache_integration.py',
        'tests/integration/test_message_queue_integration.py',
        'tests/integration/test_business_process_integration.py',
        'tests/integration/test_performance_baseline.py',
        'tests/integration/test_concurrent_performance.py',
        'tests/integration/test_stability.py',
        'tests/integration/test_external_service_integration.py',
        'tests/integration/api/test_feature_api.py'
    ]

    syntax_errors = []

    for file_path in test_files:
        try:
            py_compile.compile(file_path, doraise=True)
            print(f'✅ {file_path} - 语法正确')
        except py_compile.PyCompileError as e:
            print(f'❌ {file_path} - 语法错误: {e}')
            syntax_errors.append((file_path, str(e)))
        except FileNotFoundError:
            print(f'⚠️ {file_path} - 文件不存在')
        except Exception as e:
            print(f'❌ {file_path} - 其他错误: {e}')
            syntax_errors.append((file_path, str(e)))

    print('\n📋 语法检查总结:')
    print('-' * 40)
    print(f'总文件数: {len(test_files)}')
    print(f'语法正确: {len(test_files) - len(syntax_errors)}')
    print(f'语法错误: {len(syntax_errors)}')

    if syntax_errors:
        print('\n❌ 需要修复的文件:')
        for file_path, error in syntax_errors:
            print(f'   • {file_path}: {error}')
        return False
    else:
        print('\n🎉 所有文件语法正确！')
        return True


if __name__ == "__main__":
    success = check_syntax()
    sys.exit(0 if success else 1)
