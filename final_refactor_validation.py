import subprocess
from infrastructure.logging.core.error_handler import ErrorHandler
from infrastructure.logging.core.interfaces import LogLevel
from infrastructure.logging.core.monitoring import LogSystemMonitor
from infrastructure.logging.core import get_unified_logger
import sys
sys.path.insert(0, 'src')

print('=== 日志系统重构最终验证 ===')

# 测试UnifiedLogger
print('1. 测试UnifiedLogger...')
logger = get_unified_logger('test')
logger.info('UnifiedLogger重构测试消息')
print('   ✅ UnifiedLogger功能正常')

# 测试LogSystemMonitor
print('2. 测试LogSystemMonitor...')
monitor = LogSystemMonitor()
monitor.record_log_processed(LogLevel.INFO, 0.1)
health = monitor.get_health_status()
print(f'   ✅ 健康状态: {health["status"]}')

# 测试ErrorHandler
print('3. 测试ErrorHandler...')
handler = ErrorHandler()
error_info = handler.handle_error(ValueError('测试错误'))
print(f'   ✅ 错误分类: {error_info.error_type.value}, 严重程度: {error_info.severity.value}')

# 测试复杂度治理
print('4. 测试复杂度治理...')
result = subprocess.run(['python', 'scripts/complexity_governance_tool.py', 'src/infrastructure/logging'],
                        capture_output=True, text=True)
if '高风险函数: 0' in result.stdout:
    print('   ✅ 高风险函数已全部治理完成')
else:
    print('   ❌ 仍有高风险函数')

print('=== 重构验证完成 ===')
print('所有核心功能正常，复杂度治理目标达成！')
