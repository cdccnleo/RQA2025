import os
import re

# 定义替换映射
replacements = {
    'EventType.ACCESS': 'EventType.SYSTEM_ERROR',
    'EventType.COMPLIANCE': 'EventType.DATA_VALIDATED',
    'EventType.CONFIG_CHANGE': 'EventType.CONFIG_UPDATED',
    'EventType.CONFIG_LOADED': 'EventType.SERVICE_STARTED',
    'EventType.CREATED': 'EventType.ORDER_CREATED',
    'EventType.DATA': 'EventType.DATA_COLLECTED',
    'EventType.DATA_ARRIVAL': 'EventType.DATA_COLLECTED',
    'EventType.DATA_OPERATION': 'EventType.DATA_STORED',
    'EventType.DATA_RECEIVED': 'EventType.DATA_COLLECTED',
    'EventType.ERROR': 'EventType.SYSTEM_ERROR',
    'EventType.ERROR_OCCURRED': 'EventType.SYSTEM_ERROR',
    'EventType.INFO': 'EventType.SYSTEM_HEALTH_CHECK',
    'EventType.MARKET_DATA': 'EventType.DATA_COLLECTED',
    'EventType.ORDER_UPDATE': 'EventType.ORDER_MODIFIED',
    'EventType.PROCESSING_COMPLETE': 'EventType.FEATURE_EXTRACTED',
    'EventType.PROCESS_COMPLETED': 'EventType.WORKFLOW_COMPLETED',
    'EventType.PROCESS_STARTED': 'EventType.VALIDATION_COMPLETED',
    'EventType.RISK_SIGNAL': 'EventType.RISK_CHECK_COMPLETED',
    'EventType.SECURITY': 'EventType.SYSTEM_ERROR',
    'EventType.STRATEGY_SIGNAL': 'EventType.SIGNAL_GENERATED',
    'EventType.SYSTEM': 'EventType.SYSTEM_HEALTH_CHECK',
    'EventType.SYSTEM_EVENT': 'EventType.SYSTEM_ERROR',
    'EventType.SYSTEM_METRICS': 'EventType.PERFORMANCE_ALERT',
    'EventType.TRADE_EXECUTION': 'EventType.EXECUTION_COMPLETED',
    'EventType.USER_MANAGEMENT': 'EventType.SYSTEM_HEALTH_CHECK',
    'EventType.WARNING': 'EventType.PERFORMANCE_ALERT'
}

# 查找所有测试文件
test_files = []
for root, dirs, files in os.walk('tests'):
    for file in files:
        if file.endswith('.py'):
            test_files.append(os.path.join(root, file))

# 修复每个文件
for test_file in test_files:
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        for old, new in replacements.items():
            content = content.replace(old, new)

        if content != original_content:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'Fixed {test_file}')

    except Exception as e:
        print(f'Error processing {test_file}: {e}')

print('Enum value replacement completed')

